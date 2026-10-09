"""Unit tests for KempnerForge checkpointing (non-distributed)."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed.checkpoint as dcp

from kempnerforge.checkpoint.async_save import AsyncCheckpointer
from kempnerforge.checkpoint.state import (
    build_train_state,
    get_rng_state,
    restore_train_state,
    set_rng_state,
)
from kempnerforge.config.schema import (
    AsyncCheckpointMode,
    CheckpointConfig,
    DynamicCheckpointWindow,
    ModelConfig,
    OptimizerConfig,
)
from kempnerforge.model.transformer import Transformer
from kempnerforge.training.optimizer import build_optimizer

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

    def test_cleanup_protects_dynamic_milestones(self, tmp_path):
        """With a dyn_ckpt_window configured, the strategy's milestone steps
        survive keep_last_n; retention applies only to the later interval
        checkpoints."""
        from kempnerforge.checkpoint.manager import CheckpointManager

        config = CheckpointConfig(
            dir=str(tmp_path),
            interval=1000,
            keep_last_n=2,
            dyn_ckpt_window=DynamicCheckpointWindow(start=0, stop=512),
        )
        model = torch.nn.Linear(4, 4)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        mgr = CheckpointManager(config, model, opt)

        milestones = [0, 1, 2, 4, 256, 512]
        intervals = [1000, 2000, 3000]
        for i in milestones + intervals:
            (tmp_path / f"step_{i}").mkdir()

        mgr._cleanup()

        remaining = sorted(int(d.name.split("_")[1]) for d in tmp_path.iterdir() if d.is_dir())
        # All dynamic milestones kept; only the last keep_last_n=2 interval ckpts kept.
        assert remaining == [0, 1, 2, 4, 256, 512, 2000, 3000]


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


# ---------------------------------------------------------------------------
# Async-checkpoint `latest` safety (regression for the symlink-before-flush bug)
# ---------------------------------------------------------------------------


class TestAsyncLatestSymlinkSafety:
    """`latest` must never resolve to a checkpoint whose DCP shards are not
    durable, and auto-resume must fall back off an interrupted async flush.

    Bug: ``save()`` advanced ``latest`` (and ran ``_cleanup``) right after
    *dispatching* the async DCP save, so a crash during the ~minute-long
    background flush left ``latest`` pointing at a checkpoint with no DCP
    ``.metadata`` (resume then hard-failed with "metadata is None") while
    ``_cleanup`` may already have pruned the last good checkpoint.
    """

    @staticmethod
    def _mgr(tmp_path, mode):
        from kempnerforge.checkpoint.manager import CheckpointManager

        model = torch.nn.Linear(4, 4)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        cfg = CheckpointConfig(dir=str(tmp_path), keep_last_n=2, async_mode=mode)
        return CheckpointManager(cfg, model, opt)

    @staticmethod
    def _install_async_mock(mgr, monkeypatch):
        """Faithfully model ``AsyncCheckpointer`` for async modes.

        Real contract: ``save(new)`` first awaits the previous flush (so the
        previous checkpoint's DCP ``.metadata`` becomes durable) then
        dispatches the new one (no ``.metadata`` yet). ``wait()`` drains the
        last dispatched flush to durability.
        """
        from unittest.mock import MagicMock

        from kempnerforge.checkpoint.manager import _DCP_METADATA_FILE

        st = {"prev": None}

        def _save(state_dict, checkpoint_id, process_group=None):
            if st["prev"] is not None:
                (Path(st["prev"]) / _DCP_METADATA_FILE).write_text("ok")
            st["prev"] = checkpoint_id

        def _wait():
            if st["prev"] is not None:
                (Path(st["prev"]) / _DCP_METADATA_FILE).write_text("ok")

        monkeypatch.setattr(mgr._async_ckpt, "save", _save)
        monkeypatch.setattr(mgr._async_ckpt, "wait", _wait)
        monkeypatch.setattr("kempnerforge.checkpoint.manager.dcp.load", MagicMock())
        return st

    def _latest_target(self, tmp_path):
        latest = Path(tmp_path) / "latest"
        return latest.resolve() if latest.exists() else None

    def test_async_save_defers_latest_until_flush_durable(self, tmp_path, monkeypatch):
        mgr = self._mgr(tmp_path, AsyncCheckpointMode.async_pinned)
        self._install_async_mock(mgr, monkeypatch)

        # First async save: flush in flight, latest must NOT point at step_10.
        mgr.save(step=10)
        assert self._latest_target(tmp_path) is None, (
            "latest advanced before the async flush was durable"
        )

        # Second save awaits step_10's flush -> step_10 now durable + committed.
        mgr.save(step=20)
        tgt = self._latest_target(tmp_path)
        assert tgt == (tmp_path / "step_10").resolve()
        assert (tgt / ".metadata").exists()

        # Final drain (training loop calls wait() after the loop) commits step_20.
        mgr.wait()
        tgt = self._latest_target(tmp_path)
        assert tgt == (tmp_path / "step_20").resolve()
        assert (tgt / ".metadata").exists()

    def test_latest_invariant_holds_every_cycle(self, tmp_path, monkeypatch):
        mgr = self._mgr(tmp_path, AsyncCheckpointMode.async_pinned)
        self._install_async_mock(mgr, monkeypatch)
        for s in (5, 10, 15, 20, 25):
            mgr.save(step=s)
            tgt = self._latest_target(tmp_path)
            # Invariant: if latest exists it MUST be a durable checkpoint.
            if tgt is not None:
                assert (tgt / ".metadata").exists(), (
                    f"latest -> {tgt} but DCP .metadata missing (cycle step {s})"
                )
        mgr.wait()
        tgt = self._latest_target(tmp_path)
        assert tgt == (tmp_path / "step_25").resolve()
        assert (tgt / ".metadata").exists()

    def test_sync_mode_commits_latest_immediately(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        mgr = self._mgr(tmp_path, AsyncCheckpointMode.disabled)
        # Sync dcp.save is blocking; emulate it writing .metadata before return.
        from kempnerforge.checkpoint.manager import _DCP_METADATA_FILE

        def _save(state_dict, checkpoint_id, process_group=None):
            (Path(checkpoint_id) / _DCP_METADATA_FILE).write_text("ok")

        monkeypatch.setattr(mgr._async_ckpt, "save", _save)
        monkeypatch.setattr("kempnerforge.checkpoint.manager.dcp.load", MagicMock())

        mgr.save(step=7)
        tgt = self._latest_target(tmp_path)
        assert tgt == (tmp_path / "step_7").resolve()
        assert (tgt / ".metadata").exists()

    def _build_ckpt(self, tmp_path, step, *, complete):
        d = Path(tmp_path) / f"step_{step}"
        d.mkdir(parents=True, exist_ok=True)
        torch.save({"step": step, "tokens_seen": step * 100, "rng": {}}, d / "train_state.pt")
        (d / "metadata.json").write_text(json.dumps({"step": step, "tokens_seen": step * 100}))
        if complete:
            (d / ".metadata").write_text("ok")
        return d

    def test_resume_falls_back_to_newest_complete(self, tmp_path, monkeypatch):
        import logging
        from unittest.mock import MagicMock

        self._build_ckpt(tmp_path, 10, complete=True)
        incomplete = self._build_ckpt(tmp_path, 20, complete=False)
        latest = Path(tmp_path) / "latest"
        latest.symlink_to(incomplete.name)

        mgr = self._mgr(tmp_path, AsyncCheckpointMode.async_pinned)
        monkeypatch.setattr("kempnerforge.checkpoint.manager.dcp.load", MagicMock())

        # Attach a handler directly to the manager logger. caplog captures
        # through a root handler and relies on propagation, which the full
        # suite's global logging setup can disable (flaky in CI, passes in
        # isolation). A direct handler is isolation-proof (same pattern as
        # test_checkpoint_security.py).
        records: list[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        mgr_logger = logging.getLogger("kempnerforge.checkpoint.manager")
        handler = _Capture(level=logging.WARNING)
        prior_level = mgr_logger.level
        mgr_logger.setLevel(logging.WARNING)
        mgr_logger.addHandler(handler)
        try:
            step, tokens, _ = mgr.load()
        finally:
            mgr_logger.removeHandler(handler)
            mgr_logger.setLevel(prior_level)

        assert step == 10, "did not fall back to the newest COMPLETE checkpoint"
        assert tokens == 1000
        assert any("interrupted async flush" in r.getMessage() for r in records)

    def test_auto_resume_falls_back_when_given_the_resolved_path(self, tmp_path, monkeypatch):
        """Auto-resume passes the path it resolved, and must still fall back.

        This is the shape production takes: ``training/entry.py`` calls
        ``resolve_resume_path()`` and hands the result to ``load(path=...)``.
        Treating a passed path as "the user asked for this" made every
        production call explicit and left the fallback unreachable.
        """
        from unittest.mock import MagicMock

        self._build_ckpt(tmp_path, 10, complete=True)
        incomplete = self._build_ckpt(tmp_path, 20, complete=False)

        mgr = self._mgr(tmp_path, AsyncCheckpointMode.async_pinned)
        monkeypatch.setattr("kempnerforge.checkpoint.manager.dcp.load", MagicMock())

        step, tokens, _ = mgr.load(path=str(incomplete))

        assert step == 10, "auto-resume did not fall back to the newest COMPLETE checkpoint"
        assert tokens == 1000

    def test_explicit_load_path_still_fails_loudly(self, tmp_path, monkeypatch):
        """A user-named checkpoint is honored as-is, broken or not.

        The fallback is for auto-resume only. When ``checkpoint.load_path`` is
        set the caller asked for that directory by name, so a missing DCP
        ``.metadata`` must surface rather than being silently substituted.
        """
        from unittest.mock import MagicMock

        self._build_ckpt(tmp_path, 10, complete=True)
        incomplete = self._build_ckpt(tmp_path, 20, complete=False)

        mgr = self._mgr(tmp_path, AsyncCheckpointMode.async_pinned)
        mgr.config.load_path = str(incomplete)
        monkeypatch.setattr("kempnerforge.checkpoint.manager.dcp.load", MagicMock())

        step, tokens, _ = mgr.load(path=str(incomplete))

        assert step == 20, "explicit load_path was not honored as-is"
        assert tokens == 2000

    def test_complete_checkpoint_resumes_unchanged(self, tmp_path, monkeypatch):
        """Regression guard: a durable checkpoint is never redirected."""
        from unittest.mock import MagicMock

        self._build_ckpt(tmp_path, 10, complete=True)
        newest = self._build_ckpt(tmp_path, 20, complete=True)

        mgr = self._mgr(tmp_path, AsyncCheckpointMode.async_pinned)
        monkeypatch.setattr("kempnerforge.checkpoint.manager.dcp.load", MagicMock())

        step, tokens, _ = mgr.load(path=str(newest))

        assert step == 20
        assert tokens == 2000

    def test_resume_no_fallback_when_dcp_excluded(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        self._build_ckpt(tmp_path, 10, complete=True)
        incomplete = self._build_ckpt(tmp_path, 20, complete=False)
        latest = Path(tmp_path) / "latest"
        latest.symlink_to(incomplete.name)

        mgr = self._mgr(tmp_path, AsyncCheckpointMode.async_pinned)
        monkeypatch.setattr("kempnerforge.checkpoint.manager.dcp.load", MagicMock())

        # DCP fully excluded (fine-tune style): durability is irrelevant, the
        # incomplete `latest` target itself must still be honored.
        step, tokens, _ = mgr.load(exclude_keys=["model", "optimizer"])
        assert step == 20
        assert tokens == 2000

    def test_cleanup_never_deletes_latest_or_in_flight(self, tmp_path, monkeypatch):
        mgr = self._mgr(tmp_path, AsyncCheckpointMode.async_pinned)
        mgr.config.keep_last_n = 1
        dirs = {s: self._build_ckpt(tmp_path, s, complete=True) for s in (1, 2, 3, 4, 5)}

        latest = Path(tmp_path) / "latest"
        latest.symlink_to(dirs[3].name)  # latest -> step_3
        mgr._pending_finalize = (5, dirs[5])  # step_5 async flush in flight

        mgr._cleanup()

        assert dirs[3].exists(), "cleanup deleted the live `latest` target"
        assert dirs[5].exists(), "cleanup deleted the in-flight pending checkpoint"
        # keep_last_n=1 still prunes the genuinely-stale ones.
        assert not dirs[1].exists()
        assert not dirs[2].exists()

    # --- coverage for the distributed barrier + resolution edge paths ---

    def test_save_runs_barrier_when_distributed(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        mgr = self._mgr(tmp_path, AsyncCheckpointMode.disabled)
        monkeypatch.setattr(mgr._async_ckpt, "save", MagicMock())
        barriers = []
        monkeypatch.setattr("kempnerforge.checkpoint.manager.dist.is_initialized", lambda: True)
        monkeypatch.setattr(
            "kempnerforge.checkpoint.manager.dist.barrier",
            lambda *a, **k: barriers.append(True),
        )
        mgr.save(step=1)
        assert barriers, "save() did not barrier when distributed is initialized"

    def test_wait_and_flush_drain_and_barrier_when_distributed(self, tmp_path, monkeypatch):
        mgr = self._mgr(tmp_path, AsyncCheckpointMode.async_pinned)
        barriers = []
        monkeypatch.setattr("kempnerforge.checkpoint.manager.dist.is_initialized", lambda: True)
        monkeypatch.setattr(
            "kempnerforge.checkpoint.manager.dist.barrier",
            lambda *a, **k: barriers.append(True),
        )
        # No pending finalize: drain is a no-op, barrier still runs.
        mgr.wait()
        mgr.flush_pending_save()
        assert len(barriers) == 2

    def test_newest_complete_none_when_base_dir_missing(self, tmp_path):
        mgr = self._mgr(tmp_path / "does_not_exist", AsyncCheckpointMode.async_pinned)
        assert mgr._newest_complete_checkpoint() is None

    def test_newest_complete_none_when_no_durable_checkpoint(self, tmp_path):
        mgr = self._mgr(tmp_path, AsyncCheckpointMode.async_pinned)
        self._build_ckpt(tmp_path, 1, complete=False)
        self._build_ckpt(tmp_path, 2, complete=False)
        assert mgr._newest_complete_checkpoint() is None

    def test_resolve_dcp_load_dir_returns_resolved_when_no_fallback(self, tmp_path):
        mgr = self._mgr(tmp_path, AsyncCheckpointMode.async_pinned)
        incomplete = self._build_ckpt(tmp_path, 5, complete=False)
        # No complete checkpoint anywhere -> fallback is None -> resolved
        # is returned unchanged (caller then surfaces the real load error).
        assert mgr._resolve_dcp_load_dir(incomplete, None) == incomplete


# ---------------------------------------------------------------------------
# Dedicated sync-barrier group (async-save / barrier deadlock fix)
# ---------------------------------------------------------------------------


class TestSyncBarrierGroup:
    """save()'s end barrier must run on a dedicated gloo group, not the
    default process group that DCP's async-save background thread uses.

    Sharing that group across the main thread (barrier) and DCP's executor
    thread (all_gather / reduce_scatter) interleaves two collectives on one
    communicator and deadlocks under tight async saves.
    """

    def _mgr(self, tmp_path):
        from kempnerforge.checkpoint.manager import CheckpointManager

        model = torch.nn.Linear(4, 4)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        return CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model, opt)

    def test_sync_group_none_when_not_distributed(self, tmp_path):
        mgr = self._mgr(tmp_path)
        assert mgr._sync_group is None

    def test_sync_barrier_noop_when_not_distributed(self, tmp_path):
        # Must not raise when distributed is not initialized.
        self._mgr(tmp_path)._sync_barrier()

    def test_init_creates_dedicated_gloo_group(self, tmp_path, monkeypatch):
        import kempnerforge.checkpoint.manager as mgr_mod

        calls = {}

        def fake_new_group(backend=None):
            calls["backend"] = backend
            return "SENTINEL_GLOO_GROUP"

        monkeypatch.setattr(mgr_mod.dist, "is_initialized", lambda: True)
        monkeypatch.setattr(mgr_mod.dist, "new_group", fake_new_group)
        monkeypatch.setattr(mgr_mod.dist, "get_rank", lambda: 0)

        mgr = self._mgr(tmp_path)
        assert calls["backend"] == "gloo"
        assert mgr._sync_group == "SENTINEL_GLOO_GROUP"

    def test_sync_barrier_uses_dedicated_group_not_default(self, tmp_path, monkeypatch):
        import kempnerforge.checkpoint.manager as mgr_mod

        barrier_calls = []

        monkeypatch.setattr(mgr_mod.dist, "is_initialized", lambda: True)
        monkeypatch.setattr(mgr_mod.dist, "new_group", lambda backend=None: "GLOO_PG")
        monkeypatch.setattr(mgr_mod.dist, "get_rank", lambda: 0)
        monkeypatch.setattr(mgr_mod.dist, "barrier", lambda group=None: barrier_calls.append(group))

        mgr = self._mgr(tmp_path)
        mgr._sync_barrier()

        # The barrier must target the dedicated gloo group, never the
        # default group (group=None) that DCP's async thread uses.
        assert barrier_calls == ["GLOO_PG"]


# ---------------------------------------------------------------------------
# CheckpointManager.load -- moment restore + exclude_keys branches
# ---------------------------------------------------------------------------


class TestCheckpointManagerLoad:
    """End-to-end ``load()`` tests on the real DCP save/load path
    (single-process mode). Lives in tests/unit/ so it counts toward CI
    coverage of ``manager.py`` -- the same single-GPU regression coverage
    that previously lived in tests/integration/.
    """

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_CONFIG = ModelConfig(dim=64, n_layers=2, n_heads=2, vocab_size=256, max_seq_len=64)

    @staticmethod
    def _moments(optimizer):
        out = []
        for group in optimizer.param_groups:
            for p in group["params"]:
                st = optimizer.state.get(p, {})
                if "exp_avg" in st:
                    out.append((st["exp_avg"].clone(), st["exp_avg_sq"].clone()))
        return out

    def _train_few_steps(self, model, opt):
        for _ in range(3):
            tokens = torch.randint(0, 256, (2, 32), device=self.DEVICE)
            model(tokens).sum().backward()
            opt.step()
            opt.zero_grad()

    def test_restores_optimizer_moments_into_fresh_optimizer(self, tmp_path):
        """Regression: the manager's DCP path used to fill the load template
        from a freshly-built optimizer's *empty* state_dict, silently dropping
        the moments. After the fix, moments must restore bit-exactly into a
        fresh optimizer."""
        from kempnerforge.checkpoint.manager import CheckpointManager

        torch.manual_seed(0)
        model = Transformer(self.MODEL_CONFIG).to(self.DEVICE)
        opt = build_optimizer(model, OptimizerConfig(lr=1e-3, fused=False))
        self._train_few_steps(model, opt)

        ref = self._moments(opt)
        assert ref and any(ea.abs().sum().item() > 0 for ea, _ in ref), (
            "no non-zero moments to test"
        )

        ckpt_dir = str(tmp_path / "ckpt")
        CheckpointManager(CheckpointConfig(dir=ckpt_dir), model, opt).save(step=3)

        model2 = Transformer(self.MODEL_CONFIG).to(self.DEVICE)
        opt2 = build_optimizer(model2, OptimizerConfig(lr=1e-3, fused=False))
        assert not self._moments(opt2), "fresh optimizer should have empty state before load"
        CheckpointManager(CheckpointConfig(dir=ckpt_dir), model2, opt2).load()

        loaded = self._moments(opt2)
        assert loaded, "optimizer moments not restored (momentum reset on resume)"
        for (rea, rev), (lea, lev) in zip(ref, loaded, strict=True):
            assert torch.equal(rea, lea), "exp_avg not restored bit-exactly"
            assert torch.equal(rev, lev), "exp_avg_sq not restored bit-exactly"

    def test_load_excludes_optimizer(self, tmp_path):
        """``exclude_keys=['optimizer']`` (the scripts/eval.py / fine-tune
        flow): load model state but leave the fresh optimizer untouched.
        Covers the inner ``if load_optim:`` False branch in
        ``CheckpointManager.load``."""
        from kempnerforge.checkpoint.manager import CheckpointManager

        torch.manual_seed(0)
        model = Transformer(self.MODEL_CONFIG).to(self.DEVICE)
        opt = build_optimizer(model, OptimizerConfig(lr=1e-3, fused=False))
        self._train_few_steps(model, opt)
        ref_weights = {n: p.clone() for n, p in model.named_parameters()}

        ckpt_dir = str(tmp_path / "ckpt")
        CheckpointManager(CheckpointConfig(dir=ckpt_dir), model, opt).save(step=3)

        torch.manual_seed(99)
        model2 = Transformer(self.MODEL_CONFIG).to(self.DEVICE)
        opt2 = build_optimizer(model2, OptimizerConfig(lr=1e-3, fused=False))
        pre_load = {n: p.clone() for n, p in model2.named_parameters()}
        CheckpointManager(CheckpointConfig(dir=ckpt_dir), model2, opt2).load(
            exclude_keys=["optimizer"]
        )

        for n, p in model2.named_parameters():
            assert not torch.equal(pre_load[n], p), f"{n} was not loaded"
            assert torch.equal(ref_weights[n], p), f"{n} mismatch vs saved"
        assert not self._moments(opt2), (
            "optimizer should remain empty on exclude_keys=['optimizer']"
        )

    def test_load_excludes_model(self, tmp_path):
        """``exclude_keys=['model']`` (symmetric case): load optimizer moments
        but leave the existing model weights untouched. Covers the inner
        ``if load_model:`` False branch in ``CheckpointManager.load``."""
        from kempnerforge.checkpoint.manager import CheckpointManager

        torch.manual_seed(0)
        model = Transformer(self.MODEL_CONFIG).to(self.DEVICE)
        opt = build_optimizer(model, OptimizerConfig(lr=1e-3, fused=False))
        self._train_few_steps(model, opt)
        ref_moments = self._moments(opt)
        assert ref_moments, "fixture: expected non-empty optimizer state"

        ckpt_dir = str(tmp_path / "ckpt")
        CheckpointManager(CheckpointConfig(dir=ckpt_dir), model, opt).save(step=3)

        torch.manual_seed(99)
        model2 = Transformer(self.MODEL_CONFIG).to(self.DEVICE)
        opt2 = build_optimizer(model2, OptimizerConfig(lr=1e-3, fused=False))
        pre_load = {n: p.clone() for n, p in model2.named_parameters()}
        CheckpointManager(CheckpointConfig(dir=ckpt_dir), model2, opt2).load(exclude_keys=["model"])

        for n, p in model2.named_parameters():
            assert torch.equal(pre_load[n], p), f"{n} should not have changed"
        loaded = self._moments(opt2)
        assert loaded, "optimizer moments should load with exclude_keys=['model']"
        for (rea, rev), (lea, lev) in zip(ref_moments, loaded, strict=True):
            assert torch.equal(rea, lea), "exp_avg not restored"
            assert torch.equal(rev, lev), "exp_avg_sq not restored"


class _UsedAndUnused(torch.nn.Module):
    """``unused`` is trainable but reaches the loss only when ``both`` is set."""

    def __init__(self) -> None:
        super().__init__()
        self.used = torch.nn.Linear(4, 4)
        self.unused = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor, both: bool = False) -> torch.Tensor:
        return self.used(x) + self.unused(x) if both else self.used(x)


class _Renamed(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.renamed = torch.nn.Linear(4, 4)
        self.unused = torch.nn.Linear(4, 4)


class _WithExtra(_UsedAndUnused):
    def __init__(self) -> None:
        super().__init__()
        self.extra = torch.nn.Linear(4, 4)


class _CountingOptimizer(torch.optim.Optimizer):
    """Optimizer whose whole per-parameter state is a plain step counter."""

    def __init__(self, params, lr: float = 1e-2) -> None:
        super().__init__(params, {"lr": lr})

    @torch.no_grad()
    def step(self, closure=None) -> None:  # type: ignore[override]
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                state["step"] = state.get("step", 0) + 1
                param.add_(param.grad, alpha=-group["lr"])


class TestResumeWithUnsteppedParameters:
    """A checkpoint holds optimizer state only for parameters that were stepped."""

    @staticmethod
    def _build(seed, ckpt_dir, model_cls=_UsedAndUnused, freeze_unused=False):
        from kempnerforge.checkpoint.manager import CheckpointManager

        torch.manual_seed(seed)
        model = model_cls()
        if freeze_unused:
            model.unused.requires_grad_(False)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=1e-2)
        manager = CheckpointManager(CheckpointConfig(dir=str(ckpt_dir)), model, optimizer)
        return model, optimizer, manager

    @staticmethod
    def _step(model, optimizer, seed, both=False):
        x = torch.randn(3, 4, generator=torch.Generator().manual_seed(seed))
        model(x, both=both).pow(2).sum().backward()
        optimizer.step()
        optimizer.zero_grad()

    @staticmethod
    def _assert_same_run(model, optimizer, model2, optimizer2, stepped=("used.",)):
        for (name, p), p2 in zip(model.named_parameters(), model2.parameters(), strict=True):
            assert torch.equal(p, p2), name
            state, state2 = optimizer.state.get(p, {}), optimizer2.state.get(p2, {})
            assert state.keys() == state2.keys(), name
            assert bool(state2) is name.startswith(stepped), name
            for key in state:
                assert torch.equal(state[key], state2[key]), f"{name}.{key}"

    def test_resume_leaves_never_stepped_parameters_stateless(self, tmp_path):
        import logging

        model, optimizer, manager = self._build(0, tmp_path)
        self._step(model, optimizer, 1)
        manager.save(step=1)

        records: list[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        mgr_logger = logging.getLogger("kempnerforge.checkpoint.manager")
        handler, prior_level = _Capture(level=logging.INFO), mgr_logger.level
        mgr_logger.addHandler(handler)
        mgr_logger.setLevel(logging.INFO)
        try:
            model2, optimizer2, manager2 = self._build(1, tmp_path)
            assert manager2.load() == (1, 0, {})
        finally:
            mgr_logger.removeHandler(handler)
            mgr_logger.setLevel(prior_level)

        self._assert_same_run(model, optimizer, model2, optimizer2)
        assert [r.getMessage() for r in records if "never stepped" in r.getMessage()] == [
            "2 parameters the saved optimizer held were never stepped and resume without "
            "state: ['unused.bias', 'unused.weight']"
        ]

    def test_resumed_steps_match_the_uninterrupted_run(self, tmp_path):
        model, optimizer, manager = self._build(0, tmp_path)
        self._step(model, optimizer, 1)
        manager.save(step=1)
        self._step(model, optimizer, 2)
        self._step(model, optimizer, 3)

        model2, optimizer2, manager2 = self._build(1, tmp_path)
        manager2.load()
        self._step(model2, optimizer2, 2)
        self._step(model2, optimizer2, 3)
        self._assert_same_run(model, optimizer, model2, optimizer2)

    def test_first_gradient_after_resume_matches_the_uninterrupted_run(self, tmp_path):
        model, optimizer, manager = self._build(0, tmp_path)
        self._step(model, optimizer, 1)
        manager.save(step=1)
        self._step(model, optimizer, 2, both=True)
        self._step(model, optimizer, 3, both=True)

        model2, optimizer2, manager2 = self._build(1, tmp_path)
        manager2.load()
        assert not optimizer2.state.get(model2.unused.weight)
        self._step(model2, optimizer2, 2, both=True)
        self._step(model2, optimizer2, 3, both=True)
        self._assert_same_run(model, optimizer, model2, optimizer2, stepped=("used.", "unused."))

    def test_a_stepped_parameter_missing_part_of_its_state_still_fails(self, tmp_path):
        from torch.distributed.checkpoint.api import CheckpointException

        model, optimizer, manager = self._build(0, tmp_path)
        self._step(model, optimizer, 1)
        del optimizer.state[model.used.weight]["exp_avg_sq"]
        manager.save(step=1)

        _, _, manager2 = self._build(1, tmp_path)
        missing = r"Missing key in checkpoint state_dict: optimizer\.state\.used\.weight\."
        with pytest.raises(CheckpointException, match=missing + "exp_avg_sq"):
            manager2.load()

    @pytest.mark.parametrize(
        ("case", "named"),
        [
            ("frozen_at_save", "unused.weight"),
            ("renamed_without_model", "renamed.weight"),
            ("added_without_model", "extra.weight"),
            ("integer_keyed", "used.weight"),
        ],
    )
    def test_state_the_saved_optimizer_did_not_hold_still_fails(self, tmp_path, case, named):
        """Checkpoints that never held a parameter's state still fail, as they always have."""
        from torch.distributed.checkpoint.api import CheckpointException

        model, optimizer, manager = self._build(0, tmp_path, freeze_unused=case == "frozen_at_save")
        self._step(model, optimizer, 1, both=case == "added_without_model")
        if case == "integer_keyed":
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
            dcp.save(state, checkpoint_id=str(tmp_path / "step_1"))
        else:
            manager.save(step=1)

        model_cls = {"renamed_without_model": _Renamed, "added_without_model": _WithExtra}
        _, _, manager2 = self._build(1, tmp_path, model_cls=model_cls.get(case, _UsedAndUnused))
        exclude = ["model"] if case.endswith("without_model") else None
        with pytest.raises((CheckpointException, RuntimeError), match=re.escape(named)):
            manager2.load(path=str(tmp_path / "step_1"), exclude_keys=exclude)

    def test_the_save_records_the_parameters_held_without_state(self):
        from kempnerforge.checkpoint.manager import _never_stepped_record

        optim_state = {
            # "b" carries the empty state a resumed never-stepped parameter has.
            "state": {"a.weight": {"step": 1}, "b": {}},
            "param_groups": [{"params": ["a.weight", "b"]}, {"params": ["c.bias"]}],
        }
        record = _never_stepped_record(optim_state)
        assert sorted(record) == ["b", "c.bias"]
        assert all(isinstance(entry, torch.Tensor) for entry in record.values())

    def test_only_recorded_parameters_count_as_never_stepped(self):
        from kempnerforge.checkpoint.manager import _params_recorded_never_stepped

        # "absent" has neither state nor a record, so it is not taken as never
        # stepped and its load fails on the missing state.
        fqns = {"a.weight", "absent", "proj.weight", "proj.weight_scale"}
        saved_keys = [
            "model.proj.weight",  # a model entry, not optimizer state
            "optimizer.param_groups.0.lr",
            "optimizer.state.a.weight.nested.moment",
            "optimizer.state.proj.weight_scale.step",
            "optimizer_never_stepped.proj.weight",
            "optimizer_never_stepped.not.in.this.model",
        ]
        assert _params_recorded_never_stepped(fqns, saved_keys) == {"proj.weight"}

        contradicted = "recorded as never stepped but have saved optimizer state: ['a.weight']"
        with pytest.raises(ValueError, match=re.escape(contradicted)):
            _params_recorded_never_stepped(fqns, [*saved_keys, "optimizer_never_stepped.a.weight"])

    def test_only_parameters_the_saved_optimizer_held_stay_stateless(self):
        from kempnerforge.checkpoint.manager import _restore_never_stepped

        groups = [{"params": ["a.weight", "b.weight"]}, {"params": ["a.bias"]}]
        held = {"param_groups": groups, "state": {"a.weight": {"step": 1}}}
        _restore_never_stepped(held, {"b.weight", "a.bias"}, Path("ckpt"))
        assert held["state"] == {"a.weight": {"step": 1}, "b.weight": {}, "a.bias": {}}

        unheld = {"param_groups": groups + [{"params": [0, 1]}], "state": {}}
        message = "for 2 trainable parameters the saved optimizer did not hold: ['c', 'd']"
        with pytest.raises(RuntimeError, match=re.escape(message)):
            _restore_never_stepped(unheld, {"b.weight", "c", "d"}, Path("ckpt"))
        assert unheld["state"] == {}

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            pytest.param(
                lambda state: state["optimizer"]["state"].pop("unused.weight"),
                "Missing key in checkpoint state_dict: optimizer.state.unused.weight.",
                id="whole_state_missing",
            ),
            pytest.param(
                lambda state: state["optimizer"].update(state={}),
                "Missing key in checkpoint state_dict: optimizer.state.",
                id="state_omitted",
            ),
            pytest.param(
                lambda state: state["optimizer"]["state"].update(
                    renamed=state["optimizer"]["state"].pop("unused.weight")
                ),
                "Missing key in checkpoint state_dict: optimizer.state.unused.weight.",
                id="state_key_renamed",
            ),
            pytest.param(
                lambda state: state.update(
                    optimizer_never_stepped={"unused.weight": torch.zeros((), dtype=torch.bool)}
                ),
                "recorded as never stepped but have saved optimizer state: ['unused.weight']",
                id="record_contradicts_saved_state",
            ),
        ],
    )
    def test_a_stepped_parameter_without_a_sound_record_still_fails(
        self, tmp_path, mutate, message
    ):
        """State that went missing is not a parameter that never received a gradient."""
        from torch.distributed.checkpoint.api import CheckpointException
        from torch.distributed.checkpoint.state_dict import (
            get_model_state_dict,
            get_optimizer_state_dict,
        )

        model, optimizer, _ = self._build(0, tmp_path)
        self._step(model, optimizer, 1, both=True)
        saved = {
            "model": get_model_state_dict(model),
            "optimizer": get_optimizer_state_dict(model, optimizer),
        }
        mutate(saved)
        dcp.save(saved, checkpoint_id=str(tmp_path / "step_1"))

        _, _, manager2 = self._build(1, tmp_path)
        with pytest.raises((CheckpointException, ValueError), match=re.escape(message)):
            manager2.load(path=str(tmp_path / "step_1"))

    def test_an_opaque_optimizer_entry_keeps_its_loaded_state(self, tmp_path, monkeypatch):
        """A checkpoint that stores the optimizer as one entry loads it whole."""
        from torch.distributed.checkpoint import FileSystemReader, _version
        from torch.distributed.checkpoint.state_dict import (
            get_model_state_dict,
            get_optimizer_state_dict,
        )

        from kempnerforge.checkpoint.manager import CheckpointManager

        def build(seed):
            torch.manual_seed(seed)
            model = _UsedAndUnused()
            return model, _CountingOptimizer(model.parameters())

        model, optimizer = build(0)
        for seed in (1, 2):
            self._step(model, optimizer, seed, both=True)
        saved = {
            "model": get_model_state_dict(model),
            "optimizer": get_optimizer_state_dict(model, optimizer),
        }
        # A state of plain counters held no tensors, so older writers kept the
        # whole optimizer as a single entry instead of one key per state value.
        monkeypatch.setattr(_version, "_derived_version", "2_3")
        dcp.save(saved, checkpoint_id=str(tmp_path / "step_1"))
        monkeypatch.undo()
        index = FileSystemReader(str(tmp_path / "step_1")).read_metadata()
        assert "optimizer" in index.state_dict_metadata

        model2, optimizer2 = build(1)
        manager2 = CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model2, optimizer2)
        manager2.load(path=str(tmp_path / "step_1"))
        assert [optimizer2.state[p] for p in model2.parameters()] == [{"step": 2}] * 4

    def test_a_resumed_run_saves_and_resumes_again(self, tmp_path):
        """The second checkpoint records the never-stepped parameter the first one did."""
        model, optimizer, manager = self._build(0, tmp_path)
        self._step(model, optimizer, 1)
        manager.save(step=1)

        model2, optimizer2, manager2 = self._build(1, tmp_path)
        manager2.load()
        self._step(model2, optimizer2, 2)
        manager2.save(step=2)

        model3, optimizer3, manager3 = self._build(2, tmp_path)
        assert manager3.load() == (2, 0, {})
        self._step(model, optimizer, 2)
        self._assert_same_run(model, optimizer, model3, optimizer3)


# ---------------------------------------------------------------------------
# Resume with a different component build (config `plugins` changed)
# ---------------------------------------------------------------------------


class TestComponentSwapOnResume:
    def test_resume_into_a_different_component_is_rejected(self, tmp_path):
        """The `plugins` list is not recorded in the checkpoint, so a resume
        that builds a different component must fail the load rather than train
        on a silently mismatched model."""
        import pytest
        from torch.distributed.checkpoint.api import CheckpointException

        from kempnerforge.checkpoint.manager import CheckpointManager
        from kempnerforge.config.schema import AdapterConfig
        from kempnerforge.model.adapter import build_adapter

        ckpt_dir = str(tmp_path / "ckpt")
        saved = build_adapter(AdapterConfig(type="linear"), in_dim=8, out_dim=4)
        opt = build_optimizer(saved, OptimizerConfig(lr=1e-3, fused=False))
        CheckpointManager(CheckpointConfig(dir=ckpt_dir), saved, opt).save(step=1)

        resumed = build_adapter(AdapterConfig(type="mlp_2layer"), in_dim=8, out_dim=4)
        opt2 = build_optimizer(resumed, OptimizerConfig(lr=1e-3, fused=False))
        with pytest.raises(CheckpointException, match="Missing key in checkpoint"):
            CheckpointManager(CheckpointConfig(dir=ckpt_dir), resumed, opt2).load()
