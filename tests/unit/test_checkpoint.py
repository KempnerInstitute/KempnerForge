"""Unit tests for KempnerForge checkpointing (non-distributed)."""

from __future__ import annotations

import random

import numpy as np
import torch

from kempnerforge.checkpoint.async_save import AsyncCheckpointer
from kempnerforge.checkpoint.state import (
    build_train_state,
    get_rng_state,
    restore_train_state,
    set_rng_state,
)
from kempnerforge.config.schema import AsyncCheckpointMode, CheckpointConfig

# ---------------------------------------------------------------------------
# RNG state capture/restore
# ---------------------------------------------------------------------------


class TestRNGState:
    def test_round_trip(self):
        """RNG state save/restore should produce identical sequences."""
        # Set a known state
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)

        state = get_rng_state()

        # Generate some random numbers
        a_torch = torch.randn(5)
        a_python = random.random()
        a_numpy = np.random.randn(5)

        # Restore state
        set_rng_state(state)

        # Should get the exact same numbers
        b_torch = torch.randn(5)
        b_python = random.random()
        b_numpy = np.random.randn(5)

        assert torch.equal(a_torch, b_torch)
        assert a_python == b_python
        np.testing.assert_array_equal(a_numpy, b_numpy)

    def test_contains_expected_keys(self):
        state = get_rng_state()
        assert "python" in state
        assert "numpy" in state
        assert "torch_cpu" in state


# ---------------------------------------------------------------------------
# Training state assembly
# ---------------------------------------------------------------------------


class TestBuildTrainState:
    def test_basic_fields(self):
        state = build_train_state(step=100, tokens_seen=50000)
        assert state["step"] == 100
        assert state["tokens_seen"] == 50000
        assert "rng" in state

    def test_with_scheduler(self):
        opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10)
        for _ in range(5):
            opt.step()
            sched.step()

        state = build_train_state(step=5, tokens_seen=0, scheduler=sched)
        assert "scheduler" in state
        assert state["scheduler"]["_step_count"] == 6  # 5 steps + initial

    def test_with_extra(self):
        state = build_train_state(step=0, tokens_seen=0, extra={"best_loss": 2.5})
        assert state["best_loss"] == 2.5

    def test_without_optional_fields(self):
        state = build_train_state(step=0, tokens_seen=0)
        assert "scheduler" not in state
        assert "dataloader" not in state


class TestRestoreTrainState:
    def test_restores_step_and_tokens(self):
        state = {"step": 42, "tokens_seen": 99999}
        step, tokens, extra = restore_train_state(state)
        assert step == 42
        assert tokens == 99999
        assert extra == {}

    def test_restores_scheduler(self):
        # Build a scheduler, step it, save its state
        opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
        for _ in range(15):
            opt.step()
            sched.step()
        saved_state = sched.state_dict()

        state = {"step": 15, "tokens_seen": 0, "scheduler": saved_state}

        # Create fresh scheduler and restore
        opt2 = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)
        sched2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=10, gamma=0.5)
        restore_train_state(state, scheduler=sched2)

        # Scheduler internal state should match
        assert sched2.state_dict()["last_epoch"] == saved_state["last_epoch"]
        assert sched2.state_dict()["_last_lr"] == saved_state["_last_lr"]

    def test_restores_rng(self):
        torch.manual_seed(0)
        state = build_train_state(step=0, tokens_seen=0)

        # Advance RNG
        _ = torch.randn(100)

        # Restore should put us back
        restore_train_state(state)
        a = torch.randn(5)

        # Restore again and verify same output
        restore_train_state(state)
        b = torch.randn(5)

        assert torch.equal(a, b)

    def test_defaults_for_missing_keys(self):
        step, tokens, extra = restore_train_state({})
        assert step == 0
        assert tokens == 0
        assert extra == {}

    def test_extra_keys_roundtrip(self):
        state = build_train_state(step=5, tokens_seen=100, extra={"wandb_run_id": "abc123"})
        assert state["wandb_run_id"] == "abc123"
        step, tokens, extra = restore_train_state(state)
        assert step == 5
        assert tokens == 100
        assert extra["wandb_run_id"] == "abc123"


# ---------------------------------------------------------------------------
# Checkpoint retention
# ---------------------------------------------------------------------------


class TestCheckpointRetention:
    def test_cleanup_old_checkpoints(self, tmp_path):
        """Retention policy should keep only the last N checkpoints."""
        from kempnerforge.checkpoint.manager import CheckpointManager

        config = CheckpointConfig(dir=str(tmp_path), keep_last_n=2)
        model = torch.nn.Linear(4, 4)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        mgr = CheckpointManager(config, model, opt)

        # Manually create checkpoint dirs
        for i in [10, 20, 30, 40]:
            (tmp_path / f"step_{i}").mkdir()

        mgr._cleanup()

        remaining = sorted(d.name for d in tmp_path.iterdir() if d.is_dir())
        assert remaining == ["step_30", "step_40"]

    def test_cleanup_preserves_when_under_limit(self, tmp_path):
        from kempnerforge.checkpoint.manager import CheckpointManager

        config = CheckpointConfig(dir=str(tmp_path), keep_last_n=5)
        model = torch.nn.Linear(4, 4)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        mgr = CheckpointManager(config, model, opt)

        for i in [10, 20]:
            (tmp_path / f"step_{i}").mkdir()

        mgr._cleanup()

        remaining = sorted(d.name for d in tmp_path.iterdir() if d.is_dir())
        assert remaining == ["step_10", "step_20"]


# ---------------------------------------------------------------------------
# AsyncCheckpointer
# ---------------------------------------------------------------------------


class TestAsyncCheckpointer:
    def test_default_mode_is_disabled(self):
        ckpt = AsyncCheckpointer()
        assert ckpt.mode == AsyncCheckpointMode.disabled

    def test_is_pending_initially_false(self):
        ckpt = AsyncCheckpointer()
        assert not ckpt.is_pending

    def test_wait_with_no_pending_is_noop(self):
        ckpt = AsyncCheckpointer()
        ckpt.wait()  # Should not raise

    def test_disabled_mode_calls_dcp_save(self, monkeypatch, tmp_path):
        from unittest.mock import MagicMock

        mock_save = MagicMock()
        monkeypatch.setattr("kempnerforge.checkpoint.async_save.dcp.save", mock_save)
        ckpt = AsyncCheckpointer(mode=AsyncCheckpointMode.disabled)
        ckpt.save({"model": {}}, checkpoint_id=str(tmp_path / "step_1"))
        mock_save.assert_called_once()
        assert not ckpt.is_pending

    def test_async_mode_calls_async_save(self, monkeypatch, tmp_path):
        from unittest.mock import MagicMock

        mock_future = MagicMock()
        mock_async = MagicMock(return_value=mock_future)
        monkeypatch.setattr("kempnerforge.checkpoint.async_save.dcp.async_save", mock_async)
        ckpt = AsyncCheckpointer(mode=AsyncCheckpointMode.async_)
        ckpt.save({"model": {}}, checkpoint_id=str(tmp_path / "step_1"))
        mock_async.assert_called_once()
        assert ckpt.is_pending

    def test_wait_resolves_pending_future(self, monkeypatch, tmp_path):
        from unittest.mock import MagicMock

        mock_future = MagicMock()
        monkeypatch.setattr(
            "kempnerforge.checkpoint.async_save.dcp.async_save", MagicMock(return_value=mock_future)
        )
        ckpt = AsyncCheckpointer(mode=AsyncCheckpointMode.async_)
        ckpt.save({"model": {}}, checkpoint_id=str(tmp_path / "step_1"))
        assert ckpt.is_pending
        ckpt.wait()
        mock_future.result.assert_called_once()
        assert not ckpt.is_pending

    def test_save_waits_for_previous(self, monkeypatch, tmp_path):
        from unittest.mock import MagicMock

        mock_future1 = MagicMock()
        mock_future2 = MagicMock()
        mock_async = MagicMock(side_effect=[mock_future1, mock_future2])
        monkeypatch.setattr("kempnerforge.checkpoint.async_save.dcp.async_save", mock_async)
        ckpt = AsyncCheckpointer(mode=AsyncCheckpointMode.async_)
        ckpt.save({"model": {}}, checkpoint_id=str(tmp_path / "step_1"))
        ckpt.save({"model": {}}, checkpoint_id=str(tmp_path / "step_2"))
        # First future should have been waited on before second save
        mock_future1.result.assert_called_once()


# ---------------------------------------------------------------------------
# Dataloader state persistence (two-phase apply)
# ---------------------------------------------------------------------------


def _make_mock_mgr(tmp_path, monkeypatch, *, ignore_freeze_mismatch=False):
    """Build a CheckpointManager with DCP calls mocked out (no distributed)."""
    from unittest.mock import MagicMock

    from kempnerforge.checkpoint.manager import CheckpointManager

    model = torch.nn.Linear(4, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    config = CheckpointConfig(
        dir=str(tmp_path), keep_last_n=5, ignore_freeze_mismatch=ignore_freeze_mismatch
    )
    mgr = CheckpointManager(config, model, opt)
    monkeypatch.setattr(mgr._async_ckpt, "save", MagicMock())
    monkeypatch.setattr("kempnerforge.checkpoint.manager.dcp.load", MagicMock())
    return mgr


class TestDataloaderStatePersistence:
    """Round-trip coverage for dataloader state across save -> load -> apply.

    Training loops call load() before constructing the dataloader (the loader
    depends on phase/annealing state that load() restores). Load stashes the
    dataloader state; apply_dataloader_state() restores it into the freshly
    built loader.
    """

    def test_apply_no_op_when_nothing_pending(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        loader = MagicMock(spec=["load_state_dict"])
        mgr.apply_dataloader_state(loader)
        loader.load_state_dict.assert_not_called()

    def test_apply_restores_state_to_stateful_loader(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        stashed = {"epoch": 3, "batches_yielded": 100, "sampler": {"epoch": 3, "skip_samples": 0}}
        mgr._pending_dataloader_state = stashed

        loader = MagicMock(spec=["load_state_dict"])
        mgr.apply_dataloader_state(loader)

        loader.load_state_dict.assert_called_once_with(stashed)
        assert mgr._pending_dataloader_state is None

    def test_apply_clears_state_for_non_stateful_loader(self, tmp_path, monkeypatch):
        """Prevent the stashed state from leaking into a later (stateful) loader."""
        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        mgr._pending_dataloader_state = {"epoch": 1}

        class PlainLoader:  # no load_state_dict method
            pass

        mgr.apply_dataloader_state(PlainLoader())
        assert mgr._pending_dataloader_state is None

    def test_apply_clears_state_for_none_loader(self, tmp_path, monkeypatch):
        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        mgr._pending_dataloader_state = {"epoch": 1}
        mgr.apply_dataloader_state(None)
        assert mgr._pending_dataloader_state is None

    def test_save_persists_dataloader_state(self, tmp_path, monkeypatch):
        """save() must include dataloader state when a stateful loader is passed."""
        mgr = _make_mock_mgr(tmp_path, monkeypatch)

        class Loader:
            def state_dict(self):
                return {"epoch": 4, "batches_yielded": 200}

        mgr.save(step=1, tokens_seen=64, dataloader=Loader())
        saved = torch.load(tmp_path / "step_1" / "train_state.pt", weights_only=False)
        assert saved["dataloader"] == {"epoch": 4, "batches_yielded": 200}

    def test_load_stashes_dataloader_state_when_no_loader_provided(self, tmp_path, monkeypatch):
        """load(dataloader=None) must stash the dataloader state for later apply."""
        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        ckpt_dir = tmp_path / "step_1"
        ckpt_dir.mkdir()
        saved_state = {"epoch": 2, "batches_yielded": 50}
        torch.save(
            {
                "step": 1,
                "tokens_seen": 64,
                "rng": get_rng_state(),
                "dataloader": saved_state,
            },
            ckpt_dir / "train_state.pt",
        )

        step, tokens, _ = mgr.load(path=str(ckpt_dir))

        assert step == 1
        assert tokens == 64
        assert mgr._pending_dataloader_state == saved_state

    def test_load_restores_directly_when_loader_provided(self, tmp_path, monkeypatch):
        """load(dataloader=X) must restore directly and leave pending state empty."""
        from unittest.mock import MagicMock

        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        ckpt_dir = tmp_path / "step_1"
        ckpt_dir.mkdir()
        saved_state = {"epoch": 2, "batches_yielded": 50}
        torch.save(
            {
                "step": 1,
                "tokens_seen": 64,
                "rng": get_rng_state(),
                "dataloader": saved_state,
            },
            ckpt_dir / "train_state.pt",
        )

        loader = MagicMock(spec=["load_state_dict"])
        mgr.load(path=str(ckpt_dir), dataloader=loader)

        loader.load_state_dict.assert_called_once_with(saved_state)
        assert mgr._pending_dataloader_state is None

    def test_load_no_stash_when_no_dataloader_key(self, tmp_path, monkeypatch):
        """Missing dataloader key in train_state leaves pending state empty."""
        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        ckpt_dir = tmp_path / "step_1"
        ckpt_dir.mkdir()
        torch.save(
            {"step": 1, "tokens_seen": 64, "rng": get_rng_state()},
            ckpt_dir / "train_state.pt",
        )

        mgr.load(path=str(ckpt_dir))
        assert mgr._pending_dataloader_state is None

    def test_round_trip_save_load_apply(self, tmp_path, monkeypatch):
        """Save with loader, load without loader, apply to new loader — state flows through."""
        mgr = _make_mock_mgr(tmp_path, monkeypatch)

        captured: dict[str, dict] = {}

        class RecorderLoader:
            def __init__(self, initial: dict) -> None:
                self._state = initial

            def state_dict(self) -> dict:
                return self._state

            def load_state_dict(self, state: dict) -> None:
                captured["restored"] = state

        saver = RecorderLoader({"epoch": 7, "batches_yielded": 333})
        mgr.save(step=5, tokens_seen=128, dataloader=saver)

        # Simulate a fresh process: build a new manager and load without loader.
        mgr2 = _make_mock_mgr(tmp_path, monkeypatch)
        step, tokens, _ = mgr2.load(path=str(tmp_path / "step_5"))
        assert step == 5
        assert tokens == 128
        assert mgr2._pending_dataloader_state == {"epoch": 7, "batches_yielded": 333}

        # Build loader after load() and apply the stashed state.
        restorer = RecorderLoader({"epoch": 0, "batches_yielded": 0})
        mgr2.apply_dataloader_state(restorer)
        assert captured["restored"] == {"epoch": 7, "batches_yielded": 333}
        assert mgr2._pending_dataloader_state is None


# ---------------------------------------------------------------------------
# VLM freeze metadata: save side, load side, and the cross-arch intersection.
# Mirrors tests/integration/test_vlm_checkpoint.py but runs without CUDA so
# the unit-tests-only CI coverage job exercises this code path.
# ---------------------------------------------------------------------------


def _freeze_meta(*pairs):
    from kempnerforge.config.vlm import FreezeSpec
    from kempnerforge.training.freeze import canonical_freeze_meta

    return canonical_freeze_meta([FreezeSpec(m, f) for (m, f) in pairs])


class TestIntersectFreezeMetaByModule:
    def test_disjoint_keys_filter_to_empty(self):
        from kempnerforge.checkpoint.manager import _intersect_freeze_meta_by_module

        saved = [{"module": "a", "frozen": True}]
        expected = [{"module": "b", "frozen": False}]
        s, e = _intersect_freeze_meta_by_module(saved, expected)
        assert s == [] and e == []

    def test_shared_key_passes_through(self):
        from kempnerforge.checkpoint.manager import _intersect_freeze_meta_by_module

        saved = [{"module": "vision_encoder", "frozen": True}]
        expected = [{"module": "vision_encoder", "frozen": True}]
        s, e = _intersect_freeze_meta_by_module(saved, expected)
        assert s == saved and e == expected

    def test_drops_one_sided_keys(self):
        from kempnerforge.checkpoint.manager import _intersect_freeze_meta_by_module

        saved = [
            {"module": "vision_encoder", "frozen": True},
            {"module": "future_arch", "frozen": False},
        ]
        expected = [
            {"module": "vision_encoder", "frozen": True},
            {"module": "another_arch", "frozen": True},
        ]
        s, e = _intersect_freeze_meta_by_module(saved, expected)
        assert s == [{"module": "vision_encoder", "frozen": True}]
        assert e == [{"module": "vision_encoder", "frozen": True}]

    def test_preserves_canonical_order(self):
        from kempnerforge.checkpoint.manager import _intersect_freeze_meta_by_module

        saved = [
            {"module": "a", "frozen": True},
            {"module": "c", "frozen": False},
        ]
        expected = [
            {"module": "a", "frozen": True},
            {"module": "c", "frozen": False},
        ]
        s, e = _intersect_freeze_meta_by_module(saved, expected)
        # Order from input is preserved (it was canonical going in).
        assert [m["module"] for m in s] == ["a", "c"]
        assert [m["module"] for m in e] == ["a", "c"]


class TestPeekSavedStep:
    def test_returns_none_when_no_checkpoint(self, tmp_path, monkeypatch):
        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        assert mgr.peek_saved_step() is None

    def test_returns_step_from_metadata(self, tmp_path, monkeypatch):
        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        mgr.save(step=42, tokens_seen=128)
        assert mgr.peek_saved_step(path=str(tmp_path / "step_42")) == 42

    def test_returns_none_when_metadata_missing(self, tmp_path, monkeypatch):
        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        ckpt_dir = tmp_path / "step_3"
        ckpt_dir.mkdir()
        # No metadata.json in this dir.
        assert mgr.peek_saved_step(path=str(ckpt_dir)) is None

    def test_returns_none_when_metadata_unreadable(self, tmp_path, monkeypatch):
        import json

        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        ckpt_dir = tmp_path / "step_3"
        ckpt_dir.mkdir()
        # Write malformed JSON; peek should swallow the decode error.
        (ckpt_dir / "metadata.json").write_text("{ not valid json")
        assert mgr.peek_saved_step(path=str(ckpt_dir)) is None

        # Sanity: a valid metadata works.
        (ckpt_dir / "metadata.json").write_text(json.dumps({"step": 7}))
        assert mgr.peek_saved_step(path=str(ckpt_dir)) == 7

    def test_returns_none_when_step_field_absent(self, tmp_path, monkeypatch):
        import json

        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        ckpt_dir = tmp_path / "step_3"
        ckpt_dir.mkdir()
        (ckpt_dir / "metadata.json").write_text(json.dumps({"tokens_seen": 1024}))
        assert mgr.peek_saved_step(path=str(ckpt_dir)) is None


class TestFlushPendingSave:
    def test_delegates_to_async_wait(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        wait_mock = MagicMock()
        monkeypatch.setattr(mgr._async_ckpt, "wait", wait_mock)
        mgr.flush_pending_save()
        wait_mock.assert_called_once()


class TestSaveVLMFreezeMetadata:
    def test_save_writes_vlm_freeze_to_metadata(self, tmp_path, monkeypatch):
        import json

        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        freeze = _freeze_meta(("vision_encoder", True), ("adapter", False))
        mgr.save(step=1, extra={"vlm_freeze": freeze})
        meta = json.loads((tmp_path / "step_1" / "metadata.json").read_text())
        assert meta["vlm_freeze"] == freeze
        assert meta["step"] == 1

    def test_save_omits_vlm_freeze_when_extra_absent(self, tmp_path, monkeypatch):
        import json

        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        mgr.save(step=1)  # no extra
        meta = json.loads((tmp_path / "step_1" / "metadata.json").read_text())
        assert "vlm_freeze" not in meta

    def test_save_omits_vlm_freeze_when_extra_lacks_key(self, tmp_path, monkeypatch):
        import json

        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        mgr.save(step=1, extra={"other": "thing"})
        meta = json.loads((tmp_path / "step_1" / "metadata.json").read_text())
        assert "vlm_freeze" not in meta


class TestLoadVLMFreezeCompare:
    def test_match_passes(self, tmp_path, monkeypatch):
        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        freeze = _freeze_meta(("vision_encoder", True))
        mgr.save(step=1, extra={"vlm_freeze": freeze})

        mgr2 = _make_mock_mgr(tmp_path, monkeypatch)
        step, _, _ = mgr2.load(path=str(tmp_path / "step_1"), vlm_freeze_expected=freeze)
        assert step == 1

    def test_semantic_mismatch_raises(self, tmp_path, monkeypatch):
        import pytest

        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        mgr.save(step=1, extra={"vlm_freeze": _freeze_meta(("vision_encoder", True))})

        mgr2 = _make_mock_mgr(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="VLM freeze mismatch"):
            mgr2.load(
                path=str(tmp_path / "step_1"),
                vlm_freeze_expected=_freeze_meta(("vision_encoder", False)),
            )

    def test_ignore_flag_demotes_to_warning(self, tmp_path, monkeypatch):
        """``ignore_freeze_mismatch=True`` swaps the raise for a warning log.
        The logger lives under ``kempnerforge.*`` whose root has
        ``propagate=False``, so we attach a capturing handler directly
        instead of relying on caplog's propagation path."""
        import logging

        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        mgr.save(step=1, extra={"vlm_freeze": _freeze_meta(("vision_encoder", True))})

        records: list[logging.LogRecord] = []

        class _CaptureHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        target = logging.getLogger("kempnerforge.checkpoint.manager")
        capture = _CaptureHandler(level=logging.WARNING)
        target.addHandler(capture)
        try:
            mgr2 = _make_mock_mgr(tmp_path, monkeypatch, ignore_freeze_mismatch=True)
            step, _, _ = mgr2.load(
                path=str(tmp_path / "step_1"),
                vlm_freeze_expected=_freeze_meta(("vision_encoder", False)),
            )
        finally:
            target.removeHandler(capture)
        assert step == 1
        assert any("VLM freeze mismatch" in rec.getMessage() for rec in records)

    def test_no_expected_skips_compare(self, tmp_path, monkeypatch):
        """Text-only runs pass ``vlm_freeze_expected=None``; saved metadata
        stays on disk but no compare runs."""
        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        mgr.save(step=1, extra={"vlm_freeze": _freeze_meta(("vision_encoder", True))})
        mgr2 = _make_mock_mgr(tmp_path, monkeypatch)
        step, _, _ = mgr2.load(path=str(tmp_path / "step_1"))
        assert step == 1

    def test_no_saved_skips_compare(self, tmp_path, monkeypatch):
        """Older / non-VLM checkpoints have no ``vlm_freeze`` in metadata; a
        VLM run loads them without raising."""
        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        mgr.save(step=1)  # no extra
        mgr2 = _make_mock_mgr(tmp_path, monkeypatch)
        step, _, _ = mgr2.load(
            path=str(tmp_path / "step_1"),
            vlm_freeze_expected=_freeze_meta(("vision_encoder", True)),
        )
        assert step == 1

    def test_corrupt_metadata_logs_warning(self, tmp_path, monkeypatch):
        """Bad metadata.json is logged and treated as no-vlm-freeze; the
        load proceeds rather than crashing. Uses a direct handler attach
        because the kempnerforge logger sets ``propagate=False`` once any
        other test imports its log helpers."""
        import logging

        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        ckpt_dir = tmp_path / "step_1"
        ckpt_dir.mkdir()
        (ckpt_dir / "metadata.json").write_text("not-json")
        torch.save(
            {"step": 1, "tokens_seen": 0, "rng": get_rng_state()},
            ckpt_dir / "train_state.pt",
        )

        records: list[logging.LogRecord] = []

        class _CaptureHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        target = logging.getLogger("kempnerforge.checkpoint.manager")
        capture = _CaptureHandler(level=logging.WARNING)
        target.addHandler(capture)
        try:
            step, _, _ = mgr.load(
                path=str(ckpt_dir),
                vlm_freeze_expected=_freeze_meta(("vision_encoder", True)),
            )
        finally:
            target.removeHandler(capture)
        assert step == 1
        assert any("Could not read" in rec.getMessage() for rec in records)

    def test_cross_arch_extra_expected_key_drops_out(self, tmp_path, monkeypatch):
        """Saved {vision_encoder=True}; expected adds a future-arch key.
        The intersection rule drops the unmatched expected entry."""
        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        mgr.save(step=1, extra={"vlm_freeze": _freeze_meta(("vision_encoder", True))})

        mgr2 = _make_mock_mgr(tmp_path, monkeypatch)
        expected = _freeze_meta(("future_arch", False), ("vision_encoder", True))
        step, _, _ = mgr2.load(path=str(tmp_path / "step_1"), vlm_freeze_expected=expected)
        assert step == 1

    def test_cross_arch_extra_saved_key_drops_out(self, tmp_path, monkeypatch):
        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        saved = _freeze_meta(("future_arch", False), ("vision_encoder", True))
        mgr.save(step=1, extra={"vlm_freeze": saved})

        mgr2 = _make_mock_mgr(tmp_path, monkeypatch)
        expected = _freeze_meta(("vision_encoder", True))
        step, _, _ = mgr2.load(path=str(tmp_path / "step_1"), vlm_freeze_expected=expected)
        assert step == 1

    def test_cross_arch_error_message_lists_dropped(self, tmp_path, monkeypatch):
        import pytest

        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        saved = _freeze_meta(("future_arch", False), ("vision_encoder", True))
        mgr.save(step=1, extra={"vlm_freeze": saved})

        mgr2 = _make_mock_mgr(tmp_path, monkeypatch)
        mismatched = _freeze_meta(("vision_encoder", False))
        with pytest.raises(ValueError) as exc_info:
            mgr2.load(path=str(tmp_path / "step_1"), vlm_freeze_expected=mismatched)
        assert "cross-arch" in str(exc_info.value)
        assert "future_arch" in str(exc_info.value)
