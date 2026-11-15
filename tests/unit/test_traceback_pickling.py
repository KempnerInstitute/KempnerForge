"""Tests for the ``FrameSummary`` reducer that keeps DCP's cross-rank error reports picklable."""

from __future__ import annotations

import copyreg
import io
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
def unregistered(monkeypatch):
    """No ``FrameSummary`` reducer in the process-wide table for the test's duration."""
    monkeypatch.delitem(copyreg.dispatch_table, traceback.FrameSummary, raising=False)


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

    def test_a_frame_whose_source_lives_only_on_its_code_object_keeps_its_line(self):
        """The line is resolved before the code object is dropped."""
        namespace: dict[str, object] = {}
        exec(compile("def boom():\n    raise ValueError('from exec')\n", "<kf>", "exec"), namespace)
        boom = namespace["boom"]
        assert callable(boom)
        with pytest.raises(ValueError) as raised:
            boom()
        stack = traceback.extract_tb(raised.value.__traceback__)
        frame = traceback.FrameSummary(
            stack[-1].filename, stack[-1].lineno, stack[-1].name, lookup_line=False
        )
        if HOLDS_CODE:
            object.__setattr__(frame, "_code", stack[-1]._code)
        restored = _round_trip_with_reducer(frame)
        assert isinstance(restored, traceback.FrameSummary)
        assert restored.line == stack[-1].line
        if HOLDS_CODE:
            assert restored.line == "raise ValueError('from exec')"

    def test_a_frame_with_locals_keeps_them(self):
        """Captured locals, already strings, round-trip unchanged."""
        frame = traceback.FrameSummary("f.py", 3, "fn", lookup_line=False, locals={"x": 1})
        restored = _round_trip_with_reducer(frame)
        assert isinstance(restored, traceback.FrameSummary)
        assert restored.locals == {"x": "1"}
        assert _frame_fields(restored) == _frame_fields(frame)


class TestRegisterFrameSummaryPickling:
    def test_registers_once_where_frames_hold_code(self, unregistered):
        """Registered where a frame holds a code object; untouched elsewhere; idempotent."""
        register_frame_summary_pickling()
        register_frame_summary_pickling()
        if HOLDS_CODE:
            assert copyreg.dispatch_table[traceback.FrameSummary] is reduce_frame_summary
        else:
            assert traceback.FrameSummary not in copyreg.dispatch_table

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
