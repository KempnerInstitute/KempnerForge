"""Tests for the ``FrameSummary`` reducer that keeps DCP's cross-rank error reports picklable."""

from __future__ import annotations

import copyreg
import io
import linecache
import pickle
import traceback

import pytest
import torch
from torch.distributed.checkpoint.api import _wrap_exception

from kempnerforge.checkpoint.async_save import AsyncCheckpointer
from kempnerforge.checkpoint.manager import CheckpointManager
from kempnerforge.checkpoint.traceback_pickling import (
    reduce_frame_summary,
    register_frame_summary_pickling,
)
from kempnerforge.config.schema import CheckpointConfig

HOLDS_CODE = "_code" in traceback.FrameSummary.__slots__


def _no_space() -> None:
    raise OSError(28, "No space left on device")


def _wrapped_write_error() -> tuple[BaseException, traceback.StackSummary]:
    try:
        _no_space()
    except OSError as error:
        return _wrap_exception(error)
    raise AssertionError("unreachable")


def _round_trip_with_reducer(obj: object) -> object:
    """Pickle ``obj`` with ``reduce_frame_summary`` in a private dispatch table only."""
    buffer = io.BytesIO()
    pickler = pickle.Pickler(buffer)
    pickler.dispatch_table = {
        **copyreg.dispatch_table,
        traceback.FrameSummary: reduce_frame_summary,
    }
    pickler.dump(obj)
    return pickle.loads(buffer.getvalue())


def _frame_fields(frame: traceback.FrameSummary) -> tuple[object, ...]:
    return (
        frame.filename,
        frame.lineno,
        frame.end_lineno,
        frame.colno,
        frame.end_colno,
        frame.name,
        frame.line,
        frame.locals,
    )


@pytest.fixture
def unregistered():
    """No ``FrameSummary`` reducer in the process-wide table for the test's duration; the
    table's entry from before the test is restored after it."""
    before = copyreg.dispatch_table.pop(traceback.FrameSummary, None)
    yield
    copyreg.dispatch_table.pop(traceback.FrameSummary, None)
    if before is not None:
        copyreg.dispatch_table[traceback.FrameSummary] = before


class TestReduceFrameSummary:
    def test_a_wrapped_write_error_round_trips_with_every_frame_field(self):
        """The error and each frame's fields survive the reducer; no code object comes back."""
        error, stack = _wrapped_write_error()
        restored_error, restored_stack = _round_trip_with_reducer((error, stack))
        assert isinstance(restored_error, OSError)
        assert restored_error.errno == 28
        assert "No space left on device" in str(restored_error)
        assert isinstance(restored_stack, traceback.StackSummary)
        assert [_frame_fields(f) for f in restored_stack] == [_frame_fields(f) for f in stack]
        assert restored_stack[-1].name == "_no_space"
        assert restored_stack.format() == stack.format()
        if HOLDS_CODE:
            assert all(f._code is not None for f in stack)
            assert all(f._code is None for f in restored_stack)

    def test_a_frame_keeps_its_source_line_after_the_source_is_gone(self, tmp_path):
        """The line is read on the pickling side, so the receiver needs no source file."""
        source = tmp_path / "raising.py"
        source.write_text("x = 1\nraise OSError(28, 'No space left on device')\n")
        frame = traceback.FrameSummary(str(source), 2, "<module>", lookup_line=False)
        payload = io.BytesIO()
        pickler = pickle.Pickler(payload)
        pickler.dispatch_table = {traceback.FrameSummary: reduce_frame_summary}
        pickler.dump(frame)
        source.unlink()
        linecache.clearcache()
        restored = pickle.loads(payload.getvalue())
        assert isinstance(restored, traceback.FrameSummary)
        assert restored.line == "raise OSError(28, 'No space left on device')"

    def test_a_frame_with_locals_keeps_them(self):
        """Captured locals, already strings, round-trip unchanged."""
        frame = traceback.FrameSummary("f.py", 3, "fn", lookup_line=False, locals={"x": 1})
        restored = _round_trip_with_reducer(frame)
        assert isinstance(restored, traceback.FrameSummary)
        assert restored.locals == {"x": "1"}
        assert _frame_fields(restored) == _frame_fields(frame)


class TestRegisterFrameSummaryPickling:
    def test_registers_once_where_frames_hold_code(self, unregistered, monkeypatch):
        """A ``FrameSummary`` that holds a code object gets the reducer; a second call keeps it."""
        slots = tuple(s for s in traceback.FrameSummary.__slots__ if s != "_code")
        monkeypatch.setattr(traceback.FrameSummary, "__slots__", (*slots, "_code"))
        register_frame_summary_pickling()
        register_frame_summary_pickling()
        assert copyreg.dispatch_table[traceback.FrameSummary] is reduce_frame_summary

    def test_is_a_no_op_where_frames_hold_no_code(self, unregistered, monkeypatch):
        """A ``FrameSummary`` without a code object, as on Python 3.12, pickles as before."""
        slots = tuple(s for s in traceback.FrameSummary.__slots__ if s != "_code")
        monkeypatch.setattr(traceback.FrameSummary, "__slots__", slots)
        register_frame_summary_pickling()
        assert traceback.FrameSummary not in copyreg.dispatch_table

    def test_registers_on_this_interpreter_exactly_when_needed(self, unregistered):
        """On the running Python the reducer is registered iff its frames hold code objects."""
        register_frame_summary_pickling()
        registered = copyreg.dispatch_table.get(traceback.FrameSummary)
        assert (registered is reduce_frame_summary) == HOLDS_CODE

    def test_keeps_a_reducer_registered_before_it(self, monkeypatch):
        """Another reducer already registered for ``FrameSummary`` is left in place."""

        def other(frame):
            return traceback.FrameSummary, (frame.filename, frame.lineno, frame.name)

        monkeypatch.setitem(copyreg.dispatch_table, traceback.FrameSummary, other)
        register_frame_summary_pickling()
        assert copyreg.dispatch_table[traceback.FrameSummary] is other

    def test_a_wrapped_write_error_pickles_process_wide_once_registered(self, unregistered):
        """The default pickler, as DCP's ``gather_object`` uses it, carries the rank's error."""
        register_frame_summary_pickling()
        error, stack = pickle.loads(pickle.dumps(_wrapped_write_error()))
        assert isinstance(error, OSError)
        assert error.errno == 28
        assert stack[-1].name == "_no_space"


class TestCheckpointSetupRegisters:
    def test_an_async_checkpointer_registers_the_reducer(self, unregistered):
        """Building the DCP saver makes a rank's wrapped write error picklable."""
        AsyncCheckpointer()
        error, _ = pickle.loads(pickle.dumps(_wrapped_write_error()))
        assert isinstance(error, OSError)
        assert error.errno == 28

    def test_a_checkpoint_manager_registers_the_reducer(self, unregistered, monkeypatch, tmp_path):
        """Building a ``CheckpointManager`` registers it itself, for its saves and loads."""
        monkeypatch.setattr("kempnerforge.checkpoint.manager.AsyncCheckpointer", lambda mode: None)
        model = torch.nn.Linear(2, 2)
        CheckpointManager(
            CheckpointConfig(dir=str(tmp_path)), model, torch.optim.SGD(model.parameters())
        )
        error, _ = pickle.loads(pickle.dumps(_wrapped_write_error()))
        assert isinstance(error, OSError)
        assert error.errno == 28
