"""Unit tests for KempnerForge checkpointing (non-distributed)."""

from __future__ import annotations

import copy
import json
import os
import random
import re
import shutil
import stat
import sys
import threading
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
from kempnerforge.config.registry import registry
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

    def test_cleanup_retries_a_checkpoint_it_could_not_remove(self, tmp_path, monkeypatch):
        """A failed removal is logged instead of raised, and the next cleanup retries it."""
        import errno

        from kempnerforge.checkpoint import manager as mgr_mod
        from kempnerforge.checkpoint.manager import CheckpointManager

        model = torch.nn.Linear(4, 4)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        mgr = CheckpointManager(CheckpointConfig(dir=str(tmp_path), keep_last_n=2), model, opt)
        for i in [10, 20, 30]:
            (tmp_path / f"step_{i}").mkdir()

        def busy(path, *args, **kwargs):
            raise OSError(errno.ENOTEMPTY, "Directory not empty", str(path))

        warnings: list[str] = []
        monkeypatch.setattr(mgr_mod.shutil, "rmtree", busy)
        monkeypatch.setattr(mgr_mod.logger, "warning", warnings.append)
        mgr._cleanup()
        assert (tmp_path / "step_10").is_dir()
        assert len(warnings) == 1 and "step_10" in warnings[0], warnings

        monkeypatch.undo()
        mgr._cleanup()
        assert sorted(d.name for d in tmp_path.iterdir() if d.is_dir()) == ["step_20", "step_30"]


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


def _mock_dcp_load(monkeypatch):
    """Mock out ``dcp.load`` and the metadata read before it, for a checkpoint with an optimizer."""
    from unittest.mock import MagicMock

    reader = MagicMock()
    reader.read_metadata.return_value.state_dict_metadata = {"optimizer": None}
    monkeypatch.setattr(
        "kempnerforge.checkpoint.manager.dcp.FileSystemReader", MagicMock(return_value=reader)
    )
    monkeypatch.setattr("kempnerforge.checkpoint.manager.dcp.load", MagicMock())


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
    _mock_dcp_load(monkeypatch)
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


class TestSaveTrackingSidecar:
    """``metadata.json`` lists the logging runs live at save time under a ``tracking`` key."""

    MLFLOW = {
        "backend": "mlflow",
        "run_id": "run-1",
        "experiment": "exp",
        "tracking_uri": "http://localhost:5000",
    }
    WANDB = {
        "backend": "wandb",
        "run_id": "w-1",
        "experiment": "proj",
        "entity": "team",
        "url": "https://tracking.example/team/proj/runs/w-1",
    }
    TENSORBOARD = {"backend": "tensorboard", "log_dir": "tb_logs"}

    @staticmethod
    def _sidecar(tmp_path, step):
        with open(tmp_path / f"step_{step}" / "metadata.json") as f:
            return json.load(f)

    def test_block_carries_the_four_mlflow_fields(self, tmp_path, monkeypatch):
        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        mgr.set_tracking_source(lambda: [self.MLFLOW])
        mgr.save(step=1, tokens_seen=64)
        meta = self._sidecar(tmp_path, 1)
        assert meta == {"step": 1, "tokens_seen": 64, "tracking": [self.MLFLOW]}
        assert set(meta["tracking"][0]) == {"backend", "run_id", "experiment", "tracking_uri"}

    def test_every_live_backend_is_listed_in_order(self, tmp_path, monkeypatch):
        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        mgr.set_tracking_source(lambda: [self.WANDB, self.TENSORBOARD, self.MLFLOW])
        mgr.save(step=1)
        assert self._sidecar(tmp_path, 1)["tracking"] == [self.WANDB, self.TENSORBOARD, self.MLFLOW]

    def test_key_is_omitted_when_nothing_is_live(self, tmp_path, monkeypatch):
        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        mgr.save(step=1)  # no source attached
        mgr.set_tracking_source(list)  # attached, but no backend has a run
        mgr.save(step=2)
        for step in (1, 2):
            text = (tmp_path / f"step_{step}" / "metadata.json").read_text()
            assert text == json.dumps({"step": step, "tokens_seen": 0}, indent=2)

    def test_detached_source_stops_recording(self, tmp_path, monkeypatch):
        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        mgr.set_tracking_source(lambda: [self.MLFLOW])
        mgr.save(step=1)
        mgr.set_tracking_source(None)
        mgr.save(step=2)
        assert "tracking" in self._sidecar(tmp_path, 1)
        assert "tracking" not in self._sidecar(tmp_path, 2)

    def test_block_sits_beside_the_vlm_freeze_metadata(self, tmp_path, monkeypatch):
        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        mgr.set_tracking_source(lambda: [self.MLFLOW])
        freeze = _freeze_meta(("vision_encoder", True))
        mgr.save(step=1, extra={"vlm_freeze": freeze})
        meta = self._sidecar(tmp_path, 1)
        assert meta["vlm_freeze"] == freeze
        assert meta["tracking"] == [self.MLFLOW]

    def test_each_save_records_the_runs_live_at_that_moment(self, tmp_path, monkeypatch):
        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        live: list[dict[str, str]] = []
        mgr.set_tracking_source(lambda: list(live))
        mgr.save(step=1)  # before the first log of the process: no run yet
        live.append(self.MLFLOW)
        mgr.save(step=2)
        live.append(self.TENSORBOARD)
        mgr.save(step=3)
        assert "tracking" not in self._sidecar(tmp_path, 1)
        assert self._sidecar(tmp_path, 2)["tracking"] == [self.MLFLOW]
        assert self._sidecar(tmp_path, 3)["tracking"] == [self.MLFLOW, self.TENSORBOARD]

    def test_only_rank_zero_asks_the_source(self, tmp_path, monkeypatch):
        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        asked: list[int] = []

        def source():
            asked.append(1)
            return [self.MLFLOW]

        mgr.set_tracking_source(source)
        mgr._rank = 1  # the rank that does not write metadata.json
        mgr.save(step=1)
        assert asked == []
        assert not (tmp_path / "step_1" / "metadata.json").exists()
        mgr._rank = 0
        mgr.save(step=2)
        assert asked == [1]

    @staticmethod
    def _bad_sources():
        """Sources that break in the ways a save on rank 0 alone must survive."""

        class BadBool(list):
            def __bool__(self):
                raise RuntimeError("no truth value")

        class BadItems(dict):
            def items(self):
                raise RuntimeError("no items")

        def deeply_nested():
            nested: list = []
            for _ in range(sys.getrecursionlimit()):
                nested = [nested]
            return [{"backend": "custom", "deep": nested}]

        def raises():
            raise RuntimeError("no run object")

        return {
            "raises": (raises, "no run object"),
            "unwritable": (
                lambda: [{"backend": "custom", "run_id": object()}],
                "JSON serializable",
            ),
            "truth-value": (lambda: BadBool([{"backend": "custom"}]), "no truth value"),
            "items": (lambda: [BadItems(backend="custom")], "no items"),
            "nested": (deeply_nested, "recursion"),
        }

    @pytest.mark.parametrize("failure", ["raises", "unwritable", "truth-value", "items", "nested"])
    def test_failing_source_costs_the_key_not_the_save(self, tmp_path, monkeypatch, failure):
        from kempnerforge.checkpoint import manager as mgr_mod

        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        source, reason = self._bad_sources()[failure]
        warnings: list[str] = []
        monkeypatch.setattr(
            mgr_mod.logger, "warning", lambda msg, *args: warnings.append(msg % args)
        )
        mgr.set_tracking_source(source)
        mgr.save(step=1, tokens_seen=8)
        assert self._sidecar(tmp_path, 1) == {"step": 1, "tokens_seen": 8}
        assert (tmp_path / "step_1" / "train_state.pt").exists()
        assert (tmp_path / "latest").resolve() == (tmp_path / "step_1").resolve()
        assert len(warnings) == 1 and reason in warnings[0]

    @pytest.mark.parametrize(
        "tracking",
        [
            "oops",
            {"backend": "mlflow"},
            ["mlflow"],
            [{"backend": None}],
            [{"run_id": "r"}],
            [{"backend": " "}],
        ],
        ids=["string", "mapping", "string-entry", "no-name", "missing-name", "blank-name"],
    )
    def test_source_of_the_wrong_shape_costs_the_key_not_the_save(
        self, tmp_path, monkeypatch, tracking
    ):
        from kempnerforge.checkpoint import manager as mgr_mod

        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        warnings: list[str] = []
        monkeypatch.setattr(
            mgr_mod.logger, "warning", lambda msg, *args: warnings.append(msg % args)
        )
        mgr.set_tracking_source(lambda: tracking)
        mgr.save(step=1, tokens_seen=8)
        assert self._sidecar(tmp_path, 1) == {"step": 1, "tokens_seen": 8}
        assert len(warnings) == 1 and "list of mappings" in warnings[0]

    def test_exception_that_cannot_be_formatted_costs_only_the_key(self, tmp_path, monkeypatch):
        import io
        import logging

        class Unprintable(Exception):
            def __str__(self):
                raise RuntimeError("no message")

        def source():
            raise Unprintable()

        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        mgr.set_tracking_source(source)
        monkeypatch.setattr(logging, "raiseExceptions", False)  # a logging error is not raised
        target = logging.getLogger("kempnerforge.checkpoint.manager")
        handler = logging.StreamHandler(io.StringIO())
        target.addHandler(handler)
        try:
            mgr.save(step=1, tokens_seen=8)
        finally:
            target.removeHandler(handler)
        assert self._sidecar(tmp_path, 1) == {"step": 1, "tokens_seen": 8}
        assert (tmp_path / "latest").resolve() == (tmp_path / "step_1").resolve()

    def test_source_must_be_callable_or_none(self, tmp_path, monkeypatch):
        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        with pytest.raises(TypeError, match="callable or None, got list"):
            mgr.set_tracking_source([self.MLFLOW])  # type: ignore[arg-type]
        mgr.save(step=1)  # nothing was attached
        assert "tracking" not in self._sidecar(tmp_path, 1)

    def test_train_state_is_untouched(self, tmp_path, monkeypatch):
        extra = {"mlflow_run_id": "run-1", "wandb_run_id": "w-1", "phase_idx": 2}
        plain = _make_mock_mgr(tmp_path / "plain", monkeypatch)
        plain.save(step=1, extra=dict(extra))
        tracked = _make_mock_mgr(tmp_path / "tracked", monkeypatch)
        tracked.set_tracking_source(lambda: [self.MLFLOW, self.WANDB])
        tracked.save(step=1, extra=dict(extra))
        states = [
            torch.load(root / "step_1" / "train_state.pt", weights_only=False)
            for root in (tmp_path / "plain", tmp_path / "tracked")
        ]
        assert set(states[0]) == set(states[1])
        assert "tracking" not in states[1]
        assert {k: states[1][k] for k in extra} == extra

    def test_trackers_live_runs_reach_the_sidecar(self, tmp_path, monkeypatch):
        from kempnerforge.config.schema import JobConfig, MetricsConfig
        from kempnerforge.metrics.tracker import MetricsTracker, _LoggingBackend

        class Live(_LoggingBackend):
            closed = False

            def log(self, metrics, step):
                pass

            def tracking_entry(self):
                return None if self.closed else dict(TestSaveTrackingSidecar.MLFLOW)

            def close(self):
                self.closed = True

        model = ModelConfig(dim=128, n_layers=2, n_heads=2, vocab_size=256)
        tracker = MetricsTracker(JobConfig(model=model, metrics=MetricsConfig()), num_gpus=1)
        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        mgr.set_tracking_source(tracker.tracking_entries)
        mgr.save(step=1)
        tracker._backends = [Live()]
        mgr.save(step=2)
        tracker.close()
        mgr.save(step=3)
        assert "tracking" not in self._sidecar(tmp_path, 1)
        assert self._sidecar(tmp_path, 2)["tracking"] == [self.MLFLOW]
        assert "tracking" not in self._sidecar(tmp_path, 3)

    @pytest.mark.parametrize("recorded", [True, False], ids=["with-block", "without-block"])
    def test_checkpoint_loads_and_resumes_with_or_without_the_block(self, tmp_path, recorded):
        """A checkpoint written before the key existed resumes exactly as one that has it."""
        from kempnerforge.checkpoint.manager import CheckpointManager

        def build():
            model = torch.nn.Linear(4, 4)
            return model, torch.optim.AdamW(model.parameters(), lr=1e-2)

        torch.manual_seed(0)
        model, opt = build()
        model(torch.ones(2, 4)).sum().backward()
        opt.step()
        extra = {"mlflow_run_id": "run-1", "wandb_run_id": "w-1", "phase_idx": 2}
        config = CheckpointConfig(dir=str(tmp_path / "ckpt"))
        saver = CheckpointManager(config, model, opt)
        saver.set_tracking_source(lambda: [self.MLFLOW, self.WANDB])
        saver.save(step=3, tokens_seen=96, extra=extra)

        sidecar = tmp_path / "ckpt" / "step_3" / "metadata.json"
        meta = json.loads(sidecar.read_text())
        assert "tracking" in meta
        if not recorded:
            del meta["tracking"]
            sidecar.write_text(json.dumps(meta, indent=2))

        model2, opt2 = build()
        loader = CheckpointManager(config, model2, opt2)
        loader.set_tracking_source(lambda: [self.MLFLOW])  # a resumed run records again
        assert loader.peek_saved_step() == 3
        step, tokens_seen, restored = loader.load()
        assert (step, tokens_seen) == (3, 96)
        assert restored == extra  # the run ids still resume from the torch state
        for saved, loaded in zip(model.parameters(), model2.parameters(), strict=True):
            assert torch.equal(saved, loaded)
            assert torch.equal(opt.state[saved]["exp_avg"], opt2.state[loaded]["exp_avg"])


class _Registrar:
    """Answers fixed checkpoint fields and keeps each registration, with where ``latest`` was."""

    def __init__(self, fields=None, error=None):
        self._fields = {} if fields is None else fields
        self._error = error
        self.asked: list[int] = []
        self.registered: list[tuple[int, Path, Path | None]] = []
        self.records: list = []

    def checkpoint_fields(self, step):
        self.asked.append(step)
        return self._fields(step) if callable(self._fields) else self._fields

    def register_checkpoint(self, step, directory, record):
        latest = directory.parent / "latest"
        self.registered.append((step, directory, latest.resolve() if latest.exists() else None))
        self.records.append(record)
        if self._error is not None:
            raise self._error


def _fields_that_raise(step):
    raise RuntimeError("no fields")


class _HeldLock:
    """A lock that says whether it is held, to check what runs under it."""

    def __init__(self):
        self._lock = threading.Lock()
        self.held = False

    def __enter__(self):
        self._lock.acquire()
        self.held = True

    def __exit__(self, *exc):
        self.held = False
        self._lock.release()


class TestSaveCheckpointRegistration:
    """A registrar adds fields to a save's ``tracking`` records, and registers it on commit."""

    MLFLOW = TestSaveTrackingSidecar.MLFLOW
    WANDB = TestSaveTrackingSidecar.WANDB

    @staticmethod
    def _sidecar(tmp_path, step):
        return TestSaveTrackingSidecar._sidecar(tmp_path, step)

    @staticmethod
    def _log(monkeypatch, level):
        from kempnerforge.checkpoint import manager as mgr_mod

        seen: list[str] = []
        monkeypatch.setattr(mgr_mod.logger, level, lambda msg, *args: seen.append(msg % args))
        return seen

    def _registered(self, tmp_path, monkeypatch, fields=None, tracking=None):
        """A manager with a registrar and the given live runs, after a save of step 1."""
        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        registrar = _Registrar(fields)
        mgr.set_tracking_source(lambda: list(tracking or [self.WANDB, self.MLFLOW]))
        mgr.set_checkpoint_registrar(registrar)
        mgr.save(step=1, tokens_seen=8)
        return mgr, registrar

    def test_fields_land_in_the_matching_record_of_the_save_they_were_made_for(
        self, tmp_path, monkeypatch
    ):
        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        mgr.set_tracking_source(lambda: [self.WANDB, self.MLFLOW])
        mgr.set_checkpoint_registrar(
            _Registrar(lambda step: {"mlflow": {"model_name": f"m{step}"}})
        )
        mgr.save(step=1, tokens_seen=8)
        mgr.save(step=2, tokens_seen=16)
        for step in (1, 2):
            assert self._sidecar(tmp_path, step)["tracking"] == [
                self.WANDB,
                {**self.MLFLOW, "model_name": f"m{step}"},
            ]
        assert list(self._sidecar(tmp_path, 1)["tracking"][1])[-1] == "model_name"

    def test_fields_for_a_backend_without_a_record_are_dropped(self, tmp_path, monkeypatch):
        self._registered(tmp_path, monkeypatch, {"wandb": {"x": "1"}}, tracking=[self.MLFLOW])
        assert self._sidecar(tmp_path, 1)["tracking"] == [self.MLFLOW]

    @pytest.mark.parametrize(
        "added", [None, {}, {"mlflow": {}}], ids=["no-registrar", "empty", "no-fields"]
    )
    def test_sidecar_is_byte_identical_when_there_is_nothing_to_add(
        self, tmp_path, monkeypatch, added
    ):
        plain = _make_mock_mgr(tmp_path / "plain", monkeypatch)
        plain.set_tracking_source(lambda: [self.MLFLOW])
        plain.save(step=1, tokens_seen=8)
        registered = _make_mock_mgr(tmp_path / "registered", monkeypatch)
        registered.set_tracking_source(lambda: [self.MLFLOW])
        registered.set_checkpoint_registrar(None if added is None else _Registrar(added))
        registered.save(step=1, tokens_seen=8)
        names = ("plain", "registered")
        texts = [(tmp_path / n / "step_1" / "metadata.json").read_text() for n in names]
        assert texts[0] == texts[1]

    @pytest.mark.parametrize("source", [list, None], ids=["no-live-run", "no-source"])
    def test_fields_are_asked_only_while_a_run_is_live(self, tmp_path, monkeypatch, source):
        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        registrar = _Registrar({"mlflow": {"model_name": "m"}})
        mgr.set_tracking_source(source)
        mgr.set_checkpoint_registrar(registrar)
        mgr.save(step=1, tokens_seen=8)
        assert registrar.asked == []
        assert self._sidecar(tmp_path, 1) == {"step": 1, "tokens_seen": 8}

    def test_only_rank_zero_asks_for_fields_or_registers(self, tmp_path, monkeypatch):
        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        registrar = _Registrar()
        mgr.set_tracking_source(lambda: [self.MLFLOW])
        mgr.set_checkpoint_registrar(registrar)
        mgr._rank = 1
        mgr.save(step=1)
        assert registrar.asked == [] and registrar.registered == []
        mgr._rank = 0
        mgr.save(step=2)
        assert registrar.asked == [2] and [r[0] for r in registrar.registered] == [2]

    @pytest.mark.parametrize(
        ("fields", "reason"),
        [
            pytest.param(_fields_that_raise, "no fields", id="raises"),
            pytest.param(["mlflow"], "expected a dict", id="not-a-dict"),
            pytest.param({"mlflow": "m-1"}, "not a mapping", id="no-fields"),
            pytest.param({"mlflow": {"model_name": 7}}, "strings", id="value"),
            pytest.param({"mlflow": {7: "m-1"}}, "strings", id="key"),
        ],
    )
    def test_failing_fields_cost_only_the_fields_not_the_records_or_the_save(
        self, tmp_path, monkeypatch, fields, reason
    ):
        warnings = self._log(monkeypatch, "warning")
        _, registrar = self._registered(tmp_path, monkeypatch, fields, tracking=[self.MLFLOW])
        assert self._sidecar(tmp_path, 1) == {
            "step": 1,
            "tokens_seen": 8,
            "tracking": [self.MLFLOW],
        }
        assert (tmp_path / "step_1" / "train_state.pt").exists()
        assert (tmp_path / "latest").resolve() == (tmp_path / "step_1").resolve()
        assert [r[0] for r in registrar.registered] == [1]
        assert len(warnings) == 1 and "step 1" in warnings[0] and reason in warnings[0]

    def test_registrar_needs_both_methods_or_none_and_can_be_detached(self, tmp_path, monkeypatch):
        class FieldsOnly:
            def checkpoint_fields(self, step):
                return {}

        class NotCallable:
            checkpoint_fields = register_checkpoint = "no"

        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        mgr.set_tracking_source(lambda: [self.MLFLOW])
        for bad, name in (({}, "dict"), (FieldsOnly(), "FieldsOnly"), (NotCallable(), "Not")):
            with pytest.raises(TypeError, match=f"register_checkpoint, or be None; got {name}"):
                mgr.set_checkpoint_registrar(bad)  # type: ignore[arg-type]
        registrar = _Registrar({"mlflow": {"model_name": "m"}})
        mgr.set_checkpoint_registrar(registrar)
        mgr.save(step=1)
        mgr.set_checkpoint_registrar(None)
        mgr.save(step=2)
        assert self._sidecar(tmp_path, 1)["tracking"] == [{**self.MLFLOW, "model_name": "m"}]
        assert self._sidecar(tmp_path, 2)["tracking"] == [self.MLFLOW]
        assert [r[0] for r in registrar.registered] == [1]

    def test_train_state_is_untouched(self, tmp_path, monkeypatch):
        extra = {"mlflow_run_id": "run-1", "phase_idx": 2}
        plain = _make_mock_mgr(tmp_path / "plain", monkeypatch)
        plain.set_tracking_source(lambda: [self.MLFLOW])
        plain.save(step=1, extra=dict(extra))
        registered = _make_mock_mgr(tmp_path / "registered", monkeypatch)
        registered.set_tracking_source(lambda: [self.MLFLOW])
        registered.set_checkpoint_registrar(_Registrar({"mlflow": {"model_name": "m"}}))
        registered.save(step=1, extra=dict(extra))
        states = [
            torch.load(root / "step_1" / "train_state.pt", weights_only=False)
            for root in (tmp_path / "plain", tmp_path / "registered")
        ]
        assert set(states[0]) == set(states[1])
        assert "model_name" not in states[1]
        assert {k: states[1][k] for k in extra} == extra

    def test_a_checkpoint_is_registered_once_latest_points_at_it(self, tmp_path, monkeypatch):
        mgr, registrar = self._registered(tmp_path, monkeypatch)
        mgr.save(step=2)
        assert registrar.registered == [
            (step, tmp_path / f"step_{step}", (tmp_path / f"step_{step}").resolve())
            for step in (1, 2)
        ]

    def test_an_async_checkpoint_is_registered_only_once_its_flush_is_committed(
        self, tmp_path, monkeypatch
    ):
        mgr = TestAsyncLatestSymlinkSafety._mgr(tmp_path, AsyncCheckpointMode.async_pinned)
        TestAsyncLatestSymlinkSafety._install_async_mock(mgr, monkeypatch)
        registrar = _Registrar()
        mgr.set_tracking_source(lambda: [self.MLFLOW])
        mgr.set_checkpoint_registrar(registrar)
        mgr.save(step=10)
        assert registrar.registered == []
        mgr.save(step=20)
        mgr.wait()
        assert registrar.registered == [
            (step, tmp_path / f"step_{step}", (tmp_path / f"step_{step}").resolve())
            for step in (10, 20)
        ]

    def test_an_async_step_saved_again_is_committed_and_registered_once_after_its_last_flush(
        self, tmp_path, monkeypatch
    ):
        mgr = TestAsyncLatestSymlinkSafety._mgr(tmp_path, AsyncCheckpointMode.async_pinned)
        TestAsyncLatestSymlinkSafety._install_async_mock(mgr, monkeypatch)
        registrar = _Registrar()
        mgr.set_tracking_source(lambda: [self.MLFLOW])
        mgr.set_checkpoint_registrar(registrar)
        mgr.save(step=10)
        mgr.save(step=10)
        assert registrar.registered == []
        assert not (tmp_path / "latest").exists()
        mgr.wait()
        step_dir = tmp_path / "step_10"
        assert (tmp_path / "latest").resolve() == step_dir.resolve()
        assert registrar.registered == [(10, step_dir, step_dir.resolve())]

    def test_an_async_save_still_commits_the_pending_checkpoint_of_another_step(
        self, tmp_path, monkeypatch
    ):
        mgr = TestAsyncLatestSymlinkSafety._mgr(tmp_path, AsyncCheckpointMode.async_pinned)
        TestAsyncLatestSymlinkSafety._install_async_mock(mgr, monkeypatch)
        registrar = _Registrar()
        mgr.set_tracking_source(lambda: [self.MLFLOW])
        mgr.set_checkpoint_registrar(registrar)
        mgr.save(step=10)
        mgr.save(step=20)
        step_dir = tmp_path / "step_10"
        assert (tmp_path / "latest").resolve() == step_dir.resolve()
        assert registrar.registered == [(10, step_dir, step_dir.resolve())]

    def test_a_checkpoint_whose_flush_fails_is_never_registered(self, tmp_path, monkeypatch):
        mgr = TestAsyncLatestSymlinkSafety._mgr(tmp_path, AsyncCheckpointMode.async_pinned)
        TestAsyncLatestSymlinkSafety._install_async_mock(mgr, monkeypatch)
        registrar = _Registrar()
        mgr.set_tracking_source(lambda: [self.MLFLOW])
        mgr.set_checkpoint_registrar(registrar)
        mgr.save(step=10)

        def flush_fails():
            raise RuntimeError("flush failed")

        monkeypatch.setattr(mgr._async_ckpt, "wait", flush_fails)
        with pytest.raises(RuntimeError, match="flush failed"):
            mgr.wait()
        assert registrar.registered == []
        assert not (tmp_path / "latest").exists()

    def test_a_registrar_that_raises_costs_only_the_registration(self, tmp_path, monkeypatch):
        warnings = self._log(monkeypatch, "warning")
        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        mgr.config.keep_last_n = 1
        mgr.set_tracking_source(lambda: [self.MLFLOW])
        mgr.set_checkpoint_registrar(_Registrar(error=RuntimeError("no server")))
        mgr.save(step=1, tokens_seen=8)
        mgr.save(step=2, tokens_seen=16)
        assert (tmp_path / "latest").resolve() == (tmp_path / "step_2").resolve()
        assert not (tmp_path / "step_1").exists()
        assert self._sidecar(tmp_path, 2)["tracking"] == [self.MLFLOW]
        assert len(warnings) == 2 and "step_2" in warnings[1] and "no server" in warnings[1]

    @pytest.mark.parametrize("failing", ["fields", "register", "record", "source"])
    def test_logged_failures_carry_no_uri_credentials(self, tmp_path, monkeypatch, failing):
        error = RuntimeError("POST https://user:secret@tracking.example/api failed")

        def raises(*args):
            raise error

        warnings = self._log(monkeypatch, "warning")
        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        registrar = _Registrar(
            raises if failing == "fields" else None, error if failing == "register" else None
        )
        mgr.set_tracking_source(raises if failing == "source" else lambda: [self.MLFLOW])
        mgr.set_checkpoint_registrar(registrar)
        mgr.save(step=1)
        if failing == "record":
            monkeypatch.setattr(os, "replace", raises)
            registrar.records[0]("mlflow", {"model_id": "m-1"})
        assert len(warnings) == 1 and "POST https://tracking.example/api failed" in warnings[0]
        assert "user:secret@" not in warnings[0]

    def test_an_unprintable_registrar_error_costs_only_the_registration(
        self, tmp_path, monkeypatch
    ):
        class Unprintable(Exception):
            def __str__(self):
                raise RuntimeError("no text")

        warnings = self._log(monkeypatch, "warning")
        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        mgr.set_tracking_source(lambda: [self.MLFLOW])
        mgr.set_checkpoint_registrar(_Registrar(error=Unprintable()))
        mgr.save(step=1)
        assert (tmp_path / "latest").resolve() == (tmp_path / "step_1").resolve()
        assert len(warnings) == 1 and "<unprintable Unprintable>" in warnings[0]

    def test_record_amends_one_record_atomically_through_a_temp_file(self, tmp_path, monkeypatch):
        _, registrar = self._registered(tmp_path, monkeypatch, {"mlflow": {"model_name": "m"}})
        sidecar = tmp_path / "step_1" / "metadata.json"
        sidecar.chmod(0o640)
        replaced: list = []
        replace = os.replace

        def spy(src, dst):
            replaced.append((Path(src), Path(dst), Path(src).read_text()))
            replace(src, dst)

        monkeypatch.setattr(os, "replace", spy)
        registrar.records[0]("mlflow", {"model_id": "m-1"})
        ((src, dst, text),) = replaced
        assert dst == sidecar and src.parent == sidecar.parent and not src.exists()
        expected = {
            "step": 1,
            "tokens_seen": 8,
            "tracking": [self.WANDB, {**self.MLFLOW, "model_name": "m", "model_id": "m-1"}],
        }
        assert sidecar.read_text() == text == json.dumps(expected, indent=2)
        assert stat.S_IMODE(sidecar.stat().st_mode) == 0o640
        assert sorted(p.name for p in sidecar.parent.iterdir()) == [
            "metadata.json",
            "train_state.pt",
        ]

    def test_a_replacement_never_opens_wider_than_the_sidecar_it_replaces(
        self, tmp_path, monkeypatch
    ):
        _, registrar = self._registered(tmp_path, monkeypatch, {"mlflow": {"model_name": "m"}})
        sidecar = tmp_path / "step_1" / "metadata.json"
        sidecar.chmod(0o600)
        created: list = []
        real_open = os.open

        def spy(path, flags, mode=0o777, *args, **kwargs):
            if flags & os.O_CREAT:
                created.append(mode)
            return real_open(path, flags, mode, *args, **kwargs)

        monkeypatch.setattr(os, "open", spy)
        registrar.records[0]("mlflow", {"model_id": "m-1"})
        assert created == [0o600]
        assert stat.S_IMODE(sidecar.stat().st_mode) == 0o600
        assert json.loads(sidecar.read_text())["tracking"][1]["model_id"] == "m-1"

    @pytest.mark.parametrize("failing", ["fsync", "replace"])
    def test_a_failed_amend_leaves_the_sidecar_whole_and_never_raises(
        self, tmp_path, monkeypatch, failing
    ):
        _, registrar = self._registered(tmp_path, monkeypatch)
        sidecar = tmp_path / "step_1" / "metadata.json"
        before = sidecar.read_bytes()

        def refuse(*args):
            raise OSError("disk full")

        monkeypatch.setattr(os, failing, refuse)
        warnings = self._log(monkeypatch, "warning")
        registrar.records[0]("mlflow", {"model_id": "m-1"})
        assert sidecar.read_bytes() == before
        assert sorted(p.name for p in sidecar.parent.iterdir()) == [
            "metadata.json",
            "train_state.pt",
        ]
        assert len(warnings) == 1 and "mlflow" in warnings[0] and "disk full" in warnings[0]

    @pytest.mark.parametrize("failing", ["fsync", "replace"])
    def test_a_failed_rewrite_of_a_saved_step_leaves_its_sidecar_whole(
        self, tmp_path, monkeypatch, failing
    ):
        mgr, _ = self._registered(tmp_path, monkeypatch)
        sidecar = tmp_path / "step_1" / "metadata.json"
        before = sidecar.read_bytes()

        def refuse(*args):
            raise OSError("disk full")

        monkeypatch.setattr(os, failing, refuse)
        with pytest.raises(OSError, match="disk full"):
            mgr.save(step=1, tokens_seen=16)
        assert sidecar.read_bytes() == before
        assert sorted(p.name for p in sidecar.parent.iterdir()) == [
            "metadata.json",
            "train_state.pt",
        ]

    def test_a_new_sidecar_gets_the_mode_of_a_plain_write(self, tmp_path, monkeypatch):
        self._registered(tmp_path, monkeypatch)
        step_dir = tmp_path / "step_1"
        modes = {
            stat.S_IMODE((step_dir / name).stat().st_mode)
            for name in ("metadata.json", "train_state.pt")
        }
        assert len(modes) == 1

    def test_record_skips_quietly_when_the_checkpoint_or_its_record_is_gone(
        self, tmp_path, monkeypatch
    ):
        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        live: list = [self.MLFLOW]
        registrar = _Registrar()
        mgr.set_tracking_source(lambda: list(live))
        mgr.set_checkpoint_registrar(registrar)
        mgr.save(step=1)
        live[:] = [self.WANDB]
        mgr.save(step=2)
        live.clear()
        mgr.save(step=3)
        texts = {step: (tmp_path / f"step_{step}" / "metadata.json").read_text() for step in (2, 3)}
        shutil.rmtree(tmp_path / "step_1")
        warnings = self._log(monkeypatch, "warning")
        notes = self._log(monkeypatch, "debug")
        for record in registrar.records:
            record("mlflow", {"model_id": "m-1"})
        assert warnings == [] and len(notes) == 3
        assert "is gone" in notes[0] and all("no mlflow record" in n for n in notes[1:])
        for step, text in texts.items():
            assert (tmp_path / f"step_{step}" / "metadata.json").read_text() == text

    @pytest.mark.parametrize(
        ("backend", "fields", "step_on_file", "reason"),
        [
            ("mlflow", {"model_id": 7}, 1, "fields of strings"),
            (None, {"model_id": "m-1"}, 1, "fields of strings"),
            ("mlflow", {"model_id": "m-1"}, 99, "not the metadata of step 1"),
        ],
        ids=["value", "backend", "other-step"],
    )
    def test_record_refuses_unrecordable_fields_and_another_steps_sidecar(
        self, tmp_path, monkeypatch, backend, fields, step_on_file, reason
    ):
        _, registrar = self._registered(tmp_path, monkeypatch)
        sidecar = tmp_path / "step_1" / "metadata.json"
        sidecar.write_text(sidecar.read_text().replace('"step": 1', f'"step": {step_on_file}'))
        before = sidecar.read_text()
        warnings = self._log(monkeypatch, "warning")
        registrar.records[0](backend, fields)
        assert sidecar.read_text() == before
        assert len(warnings) == 1 and reason in warnings[0]

    def test_sidecar_writes_amends_and_removals_hold_the_sidecar_lock(self, tmp_path, monkeypatch):
        from kempnerforge.checkpoint import manager as mgr_mod

        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        mgr.config.keep_last_n = 1
        lock = _HeldLock()
        mgr._sidecar_lock = lock  # type: ignore[assignment]
        held: list = []
        replace, rmtree = os.replace, mgr_mod.shutil.rmtree

        def replaced(src, dst):
            held.append((Path(dst).parent.name, lock.held))
            return replace(src, dst)

        def removed(path, *args, **kwargs):
            held.append((f"remove {Path(path).name}", lock.held))
            return rmtree(path, *args, **kwargs)

        monkeypatch.setattr(os, "replace", replaced)
        monkeypatch.setattr(mgr_mod.shutil, "rmtree", removed)
        registrar = _Registrar()
        mgr.set_tracking_source(lambda: [self.MLFLOW])
        mgr.set_checkpoint_registrar(registrar)
        mgr.save(step=1)
        registrar.records[0]("mlflow", {"model_id": "m-1"})
        mgr.save(step=2)
        assert held == [
            ("step_1", True),
            ("step_1", True),
            ("step_2", True),
            ("remove step_1", True),
        ]

    def test_trackers_fields_and_registration_reach_the_sidecar_by_backend_name(
        self, tmp_path, monkeypatch
    ):
        from kempnerforge.config.schema import JobConfig, MetricsConfig
        from kempnerforge.metrics.tracker import MetricsTracker, _LoggingBackend

        class Registering(_LoggingBackend):
            def log(self, metrics, step):
                pass

            def tracking_entry(self):
                return dict(TestSaveCheckpointRegistration.MLFLOW)

            def checkpoint_fields(self, step):
                return {"model_name": f"m-step_{step}"}

            def register_checkpoint(self, step, directory, record):
                record({"model_id": f"{directory.name}-registered"})

        model = ModelConfig(dim=128, n_layers=2, n_heads=2, vocab_size=256)
        tracker = MetricsTracker(JobConfig(model=model, metrics=MetricsConfig()), num_gpus=1)
        tracker._backends = [_LoggingBackend(), Registering()]
        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        mgr.set_tracking_source(tracker.tracking_entries)
        mgr.set_checkpoint_registrar(tracker)
        mgr.save(step=3)
        assert self._sidecar(tmp_path, 3)["tracking"] == [
            {**self.MLFLOW, "model_name": "m-step_3", "model_id": "step_3-registered"}
        ]


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
    def _mgr(tmp_path, mode, pp_rank=None):
        from kempnerforge.checkpoint.manager import CheckpointManager

        model = torch.nn.Linear(4, 4)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        cfg = CheckpointConfig(dir=str(tmp_path), keep_last_n=2, async_mode=mode)
        return CheckpointManager(cfg, model, opt, pp_rank=pp_rank)

    @staticmethod
    def _install_async_mock(mgr, monkeypatch):
        """Faithfully model ``AsyncCheckpointer`` for async modes.

        Real contract: ``save(new)`` first awaits the previous flush (so the
        previous checkpoint's DCP ``.metadata`` becomes durable) then
        dispatches the new one (no ``.metadata`` yet). ``wait()`` drains the
        last dispatched flush to durability.
        """

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
        _mock_dcp_load(monkeypatch)
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
        mgr = self._mgr(tmp_path, AsyncCheckpointMode.disabled)
        # Sync dcp.save is blocking; emulate it writing .metadata before return.
        from kempnerforge.checkpoint.manager import _DCP_METADATA_FILE

        def _save(state_dict, checkpoint_id, process_group=None):
            (Path(checkpoint_id) / _DCP_METADATA_FILE).write_text("ok")

        monkeypatch.setattr(mgr._async_ckpt, "save", _save)
        _mock_dcp_load(monkeypatch)

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

        self._build_ckpt(tmp_path, 10, complete=True)
        incomplete = self._build_ckpt(tmp_path, 20, complete=False)
        latest = Path(tmp_path) / "latest"
        latest.symlink_to(incomplete.name)

        mgr = self._mgr(tmp_path, AsyncCheckpointMode.async_pinned)
        _mock_dcp_load(monkeypatch)

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

        self._build_ckpt(tmp_path, 10, complete=True)
        incomplete = self._build_ckpt(tmp_path, 20, complete=False)

        mgr = self._mgr(tmp_path, AsyncCheckpointMode.async_pinned)
        _mock_dcp_load(monkeypatch)

        step, tokens, _ = mgr.load(path=str(incomplete))

        assert step == 10, "auto-resume did not fall back to the newest COMPLETE checkpoint"
        assert tokens == 1000

    def test_explicit_load_path_still_fails_loudly(self, tmp_path, monkeypatch):
        """A user-named checkpoint is honored as-is, broken or not.

        The fallback is for auto-resume only. When ``checkpoint.load_path`` is
        set the caller asked for that directory by name, so a missing DCP
        ``.metadata`` must surface rather than being silently substituted.
        """

        self._build_ckpt(tmp_path, 10, complete=True)
        incomplete = self._build_ckpt(tmp_path, 20, complete=False)

        mgr = self._mgr(tmp_path, AsyncCheckpointMode.async_pinned)
        mgr.config.load_path = str(incomplete)
        _mock_dcp_load(monkeypatch)

        step, tokens, _ = mgr.load(path=str(incomplete))

        assert step == 20, "explicit load_path was not honored as-is"
        assert tokens == 2000

    def test_missing_load_path_raises(self, tmp_path):
        """A typo'd or moved ``load_path`` must not silently train from scratch."""
        mgr = self._mgr(tmp_path, AsyncCheckpointMode.async_pinned)
        mgr.config.load_path = str(tmp_path / "does_not_exist" / "step_100")

        with pytest.raises(FileNotFoundError, match="does_not_exist"):
            mgr.load()

    def test_missing_explicit_path_raises(self, tmp_path):
        mgr = self._mgr(tmp_path, AsyncCheckpointMode.async_pinned)

        with pytest.raises(FileNotFoundError, match="nope"):
            mgr.load(path=str(tmp_path / "nope"))

    def test_empty_dir_without_load_path_starts_from_scratch(self, tmp_path):
        """Auto-resume from an empty checkpoint dir stays silent."""
        mgr = self._mgr(tmp_path, AsyncCheckpointMode.async_pinned)

        assert mgr.load() == (0, 0, {})

    def test_complete_checkpoint_resumes_unchanged(self, tmp_path, monkeypatch):
        """Regression guard: a durable checkpoint is never redirected."""

        self._build_ckpt(tmp_path, 10, complete=True)
        newest = self._build_ckpt(tmp_path, 20, complete=True)

        mgr = self._mgr(tmp_path, AsyncCheckpointMode.async_pinned)
        _mock_dcp_load(monkeypatch)

        step, tokens, _ = mgr.load(path=str(newest))

        assert step == 20
        assert tokens == 2000

    def test_resume_no_fallback_when_dcp_excluded(self, tmp_path, monkeypatch):
        self._build_ckpt(tmp_path, 10, complete=True)
        incomplete = self._build_ckpt(tmp_path, 20, complete=False)
        latest = Path(tmp_path) / "latest"
        latest.symlink_to(incomplete.name)

        mgr = self._mgr(tmp_path, AsyncCheckpointMode.async_pinned)
        _mock_dcp_load(monkeypatch)

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

    @pytest.mark.parametrize("pp_rank", [None, 0], ids=["no_pp", "pp_stage_0"])
    @pytest.mark.parametrize(
        "mode",
        [AsyncCheckpointMode.async_, AsyncCheckpointMode.async_pinned],
        ids=["async", "async_with_pinned_mem"],
    )
    def test_an_async_save_after_a_rewind_keeps_the_directory_it_writes(
        self, tmp_path, monkeypatch, mode, pp_rank
    ):
        """Newer leftovers fill ``keep_last_n``, yet the save in flight keeps its directory.

        Under pipeline parallelism a stage writes into ``step_N/pp<rank>``, and it is
        ``step_N`` itself that retention must spare.
        """
        mgr = self._mgr(tmp_path, mode, pp_rank)
        self._install_async_mock(mgr, monkeypatch)
        for step in (100, 300, 400, 500):
            self._build_ckpt(tmp_path, step, complete=True)
        (Path(tmp_path) / "latest").symlink_to("step_100")

        mgr.save(step=150)
        mgr.save(step=200)
        mgr.wait()

        written = Path(tmp_path) / "step_200"
        shards = written / f"pp{pp_rank}" if pp_rank is not None else written
        assert self._latest_target(tmp_path) == written.resolve()
        assert (shards / ".metadata").exists()
        assert (written / "train_state.pt").exists()
        step, _, _ = mgr.load()
        assert step == 200

    def test_a_saves_directory_is_pruned_once_a_newer_save_is_under_way(
        self, tmp_path, monkeypatch
    ):
        """Only the latest save is spared as in flight; older ones fall to ``keep_last_n``."""
        mgr = self._mgr(tmp_path, AsyncCheckpointMode.async_)
        mgr.config.keep_last_n = 1
        self._install_async_mock(mgr, monkeypatch)

        for step in (1, 2, 3):
            mgr.save(step=step)
        mgr.wait()

        assert sorted(p.name for p in Path(tmp_path).glob("step_*")) == ["step_3"]
        assert self._latest_target(tmp_path) == (Path(tmp_path) / "step_3").resolve()

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
            pytest.param(
                lambda state: (
                    state.update(
                        optimizer_never_stepped={"unused.weight": torch.zeros((), dtype=torch.bool)}
                    ),
                    state["optimizer"]["state"].pop("unused.weight"),
                    state["optimizer"]["param_groups"][0]["params"].remove("unused.weight"),
                ),
                "the saved optimizer did not hold: ['unused.weight']",
                id="record_names_a_parameter_the_groups_omit",
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
        with pytest.raises(
            (CheckpointException, ValueError, RuntimeError), match=re.escape(message)
        ):
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


class TestMuonResume:
    """Muon is checkpointed together with its inner AdamW."""

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_CONFIG = ModelConfig(dim=64, n_layers=2, n_heads=2, vocab_size=4096, max_seq_len=64)

    def _build(self, seed, **overrides):
        torch.manual_seed(seed)
        model = Transformer(self.MODEL_CONFIG).to(self.DEVICE)
        config = OptimizerConfig(name="muon", lr=1e-3, fused=False, **overrides)
        return model, build_optimizer(model, config)

    @staticmethod
    def _save_without(model, opt, keys, ckpt_dir):
        """Save ``opt`` as it was saved before its param groups had ``keys``."""
        from kempnerforge.checkpoint.manager import CheckpointManager

        for group in opt.param_groups:
            for key in keys:
                del group[key]
        CheckpointManager(CheckpointConfig(dir=str(ckpt_dir)), model, opt).save(step=3)

    def _batches(self, n):
        g = torch.Generator().manual_seed(123)
        vocab = self.MODEL_CONFIG.vocab_size
        return [torch.randint(0, vocab, (2, 32), generator=g).to(self.DEVICE) for _ in range(n)]

    @staticmethod
    def _train(model, opt, batches):
        for tokens in batches:
            logits = model(tokens)
            torch.nn.functional.cross_entropy(logits.flatten(0, 1), tokens.flatten()).backward()
            opt.step()
            opt.zero_grad()

    @staticmethod
    def _state(opt):
        """(name, tensor) for every state entry of ``opt``, in param order."""
        return [
            (name, value.clone())
            for group in opt.param_groups
            for p in group["params"]
            for name, value in sorted(opt.state.get(p, {}).items())
        ]

    def _assert_same_state(self, opt, opt2, what):
        saved, loaded = self._state(opt), self._state(opt2)
        assert saved, f"fixture: {what} has no state to compare"
        assert [n for n, _ in loaded] == [n for n, _ in saved], f"{what} state not restored"
        for (name, s), (_, t) in zip(saved, loaded, strict=True):
            assert torch.equal(s, t), f"{what} {name} not restored bit-exactly"

    def test_checkpointed_optimizers_put_muon_first(self, tmp_path):
        """Muon comes before its inner AdamW; an optimizer without inner ones is alone."""
        from kempnerforge.checkpoint.manager import CheckpointManager

        model, opt = self._build(seed=0)
        manager = CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model, opt)
        assert manager.checkpointed_optimizers == [opt, opt._adam]

        adamw = build_optimizer(model, OptimizerConfig(lr=1e-3, fused=False))
        manager = CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model, adamw)
        assert manager.checkpointed_optimizers == [adamw]

    def test_resume_matches_an_uninterrupted_run(self, tmp_path):
        """A resumed Muon run matches an uninterrupted one."""
        from kempnerforge.checkpoint.manager import CheckpointManager

        batches = self._batches(6)
        ref_model, ref_opt = self._build(seed=0)
        self._train(ref_model, ref_opt, batches)

        model, opt = self._build(seed=0)
        inner_shapes = {tuple(p.shape) for g in opt._adam.param_groups for p in g["params"]}
        assert (4096, 64) in inner_shapes, "fixture: the embedding should be on the inner AdamW"
        self._train(model, opt, batches[:3])
        CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model, opt).save(step=3)

        model2, opt2 = self._build(seed=1)
        CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model2, opt2).load()
        self._assert_same_state(opt._adam, opt2._adam, "inner AdamW")

        self._train(model2, opt2, batches[3:])
        for (name, ref), p in zip(ref_model.named_parameters(), model2.parameters(), strict=True):
            assert torch.equal(ref, p), f"{name} diverged from the uninterrupted run"

    def test_checkpoint_without_inner_state_resumes_muon_and_warns(self, tmp_path):
        """A checkpoint without the inner AdamW resumes Muon alone and warns."""
        import logging

        from torch.distributed.checkpoint.state_dict import (
            get_model_state_dict,
            get_optimizer_state_dict,
        )

        from kempnerforge.checkpoint.manager import CheckpointManager

        model, opt = self._build(seed=0)
        self._train(model, opt, self._batches(3))
        step_dir = tmp_path / "step_3"
        state = {
            "model": get_model_state_dict(model),
            "optimizer": get_optimizer_state_dict(model, opt),
        }
        dcp.save(state, checkpoint_id=str(step_dir))

        records: list[logging.LogRecord] = []

        class _CaptureHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        target = logging.getLogger("kempnerforge.checkpoint.manager")
        capture = _CaptureHandler(level=logging.WARNING)
        target.addHandler(capture)
        model2, opt2 = self._build(seed=1)
        try:
            CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model2, opt2).load(
                path=str(step_dir)
            )
        finally:
            target.removeHandler(capture)

        assert any("holds only Muon's own optimizer state" in r.getMessage() for r in records)
        self._assert_same_state(opt, opt2, "Muon")
        assert not opt2._adam.state, "the inner AdamW should start fresh"
        self._train(model2, opt2, self._batches(1))

    def test_checkpoint_still_loads_into_muon_alone(self, tmp_path):
        """A reader that passes Muon alone still loads Muon's state."""
        from torch.distributed.checkpoint.state_dict import (
            get_optimizer_state_dict,
            set_optimizer_state_dict,
        )

        from kempnerforge.checkpoint.manager import CheckpointManager

        model, opt = self._build(seed=0)
        self._train(model, opt, self._batches(3))
        CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model, opt).save(step=3)

        model2, opt2 = self._build(seed=1)
        state = {"optimizer": get_optimizer_state_dict(model2, opt2)}
        dcp.load(state, checkpoint_id=str(tmp_path / "step_3"))
        set_optimizer_state_dict(model2, opt2, optim_state_dict=state["optimizer"])
        self._assert_same_state(opt, opt2, "Muon")

    def test_checkpoint_saved_before_orthogonalize_resumes_as_local(self, tmp_path):
        """A checkpoint saved before ``orthogonalize`` existed resumes with "local" and matches."""
        from kempnerforge.checkpoint.manager import CheckpointManager

        batches = self._batches(6)
        ref_model, ref_opt = self._build(seed=0)
        self._train(ref_model, ref_opt, batches)

        model, opt = self._build(seed=0)
        self._train(model, opt, batches[:3])
        self._save_without(model, opt, ["orthogonalize"], tmp_path)

        model2, opt2 = self._build(seed=1)
        CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model2, opt2).load()
        assert {g["orthogonalize"] for g in opt2.param_groups} == {"local"}

        self._train(model2, opt2, batches[3:])
        for (name, ref), p in zip(ref_model.named_parameters(), model2.parameters(), strict=True):
            assert torch.equal(ref, p), f"{name} diverged from the uninterrupted run"

    def test_checkpoint_saved_before_orthogonalize_warns(self, tmp_path):
        """Filling in a missing key logs one warning naming it; a complete checkpoint logs none."""
        import logging

        from kempnerforge.checkpoint.manager import CheckpointManager

        records: list[logging.LogRecord] = []

        class _CaptureHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        def warnings_on_load(ckpt_dir) -> list[str]:
            records.clear()
            target = logging.getLogger("kempnerforge.checkpoint.manager")
            capture = _CaptureHandler(level=logging.WARNING)
            target.addHandler(capture)
            model2, opt2 = self._build(seed=1)
            try:
                CheckpointManager(CheckpointConfig(dir=str(ckpt_dir)), model2, opt2).load()
            finally:
                target.removeHandler(capture)
            return [r.getMessage() for r in records if "orthogonalize" in r.getMessage()]

        model, opt = self._build(seed=0)
        self._train(model, opt, self._batches(3))
        CheckpointManager(CheckpointConfig(dir=str(tmp_path / "complete")), model, opt).save(step=3)
        self._save_without(model, opt, ["orthogonalize"], tmp_path / "legacy")

        assert warnings_on_load(tmp_path / "complete") == []
        assert len(warnings_on_load(tmp_path / "legacy")) == 1

    def test_checkpoint_saved_before_orthogonalize_ignores_the_configured_mode(self, tmp_path):
        """It resumes with the mode it was trained with, as a newer checkpoint does."""
        from kempnerforge.checkpoint.manager import CheckpointManager

        model, opt = self._build(seed=0)
        self._train(model, opt, self._batches(3))
        self._save_without(model, opt, ["orthogonalize"], tmp_path)

        model2, opt2 = self._build(seed=1, muon_orthogonalize="exact")
        CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model2, opt2).load()
        assert {g["orthogonalize"] for g in opt2.param_groups} == {"local"}

    def test_checkpoint_resumes_with_the_mode_it_was_saved_with(self, tmp_path):
        """A checkpoint that has ``orthogonalize`` keeps it over the configured mode."""
        from kempnerforge.checkpoint.manager import CheckpointManager

        model, opt = self._build(seed=0, muon_orthogonalize="exact")
        self._train(model, opt, self._batches(3))
        CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model, opt).save(step=3)

        model2, opt2 = self._build(seed=1)
        CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model2, opt2).load()
        assert {g["orthogonalize"] for g in opt2.param_groups} == {"exact"}

    @pytest.mark.parametrize(
        "missing", [["nesterov", "adjust_lr"], ["orthogonalize", "nesterov", "adjust_lr"]]
    )
    def test_checkpoint_saved_before_nesterov_and_adjust_lr_resumes_with_the_old_update(
        self, tmp_path, missing
    ):
        """A checkpoint without ``nesterov`` and ``adjust_lr`` resumes with the update it had."""
        from kempnerforge.checkpoint.manager import CheckpointManager

        batches = self._batches(6)
        ref_model, ref_opt = self._build(seed=0)
        self._train(ref_model, ref_opt, batches)

        model, opt = self._build(seed=0)
        self._train(model, opt, batches[:3])
        self._save_without(model, opt, missing, tmp_path)

        model2, opt2 = self._build(seed=1, muon_nesterov=True, muon_adjust_lr="match_rms_adamw")
        CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model2, opt2).load()
        groups = {(g["orthogonalize"], g["nesterov"], g["adjust_lr"]) for g in opt2.param_groups}
        assert groups == {("local", False, "original")}

        self._train(model2, opt2, batches[3:])
        for (name, ref), p in zip(ref_model.named_parameters(), model2.parameters(), strict=True):
            assert torch.equal(ref, p), f"{name} diverged from the uninterrupted run"

    def test_checkpoint_resumes_with_the_nesterov_and_adjust_lr_it_was_saved_with(self, tmp_path):
        """A checkpoint keeps its ``nesterov`` and ``adjust_lr`` over the configured ones."""
        from kempnerforge.checkpoint.manager import CheckpointManager

        options = {"muon_nesterov": True, "muon_adjust_lr": "match_rms_adamw"}
        batches = self._batches(6)
        ref_model, ref_opt = self._build(seed=0, **options)
        self._train(ref_model, ref_opt, batches)

        model, opt = self._build(seed=0, **options)
        self._train(model, opt, batches[:3])
        CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model, opt).save(step=3)

        model2, opt2 = self._build(seed=1)
        CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model2, opt2).load()
        groups = {(g["nesterov"], g["adjust_lr"]) for g in opt2.param_groups}
        assert groups == {(True, "match_rms_adamw")}

        self._train(model2, opt2, batches[3:])
        for (name, ref), p in zip(ref_model.named_parameters(), model2.parameters(), strict=True):
            assert torch.equal(ref, p), f"{name} diverged from the uninterrupted run"

    def test_checkpoint_without_another_group_key_still_fails(self, tmp_path):
        """Only the listed keys may be missing; a checkpoint without any other fails to load."""
        from torch.distributed.checkpoint.api import CheckpointException

        from kempnerforge.checkpoint.manager import CheckpointManager

        model, opt = self._build(seed=0)
        self._train(model, opt, self._batches(3))
        self._save_without(model, opt, ["ns_steps"], tmp_path)

        model2, opt2 = self._build(seed=1)
        missing = r"Missing key in checkpoint state_dict: optimizer\.param_groups\.\d+\.ns_steps"
        with pytest.raises(CheckpointException, match=missing):
            CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model2, opt2).load()

    def test_checkpoint_without_inner_state_or_orthogonalize_resumes_muon(self, tmp_path):
        """The oldest Muon checkpoints lack the inner AdamW and every legacy group key."""
        from torch.distributed.checkpoint.state_dict import (
            get_model_state_dict,
            get_optimizer_state_dict,
        )

        from kempnerforge.checkpoint.manager import CheckpointManager

        model, opt = self._build(seed=0)
        self._train(model, opt, self._batches(3))
        for group in opt.param_groups:
            del group["orthogonalize"], group["nesterov"], group["adjust_lr"]
        step_dir = tmp_path / "step_3"
        state = {
            "model": get_model_state_dict(model),
            "optimizer": get_optimizer_state_dict(model, opt),
        }
        dcp.save(state, checkpoint_id=str(step_dir))

        model2, opt2 = self._build(seed=1)
        CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model2, opt2).load(
            path=str(step_dir)
        )
        groups = {(g["orthogonalize"], g["nesterov"], g["adjust_lr"]) for g in opt2.param_groups}
        assert groups == {("local", False, "original")}
        self._assert_same_state(opt, opt2, "Muon")


REGISTERED_OPTIMIZERS = sorted(registry.list("optimizer"))
RESUMABLE_OPTIMIZERS = [
    pytest.param(
        name,
        marks=pytest.mark.xfail(
            raises=ZeroDivisionError, strict=True, reason="#236: schedule_free_adamw cannot resume"
        ),
    )
    if name == "schedule_free_adamw"
    else name
    for name in REGISTERED_OPTIMIZERS
]


class TestSaveBeforeFirstStep:
    """The step-0 save and a load without an optimizer, for every registered optimizer."""

    MODEL_CONFIG = ModelConfig(dim=64, n_layers=2, n_heads=2, vocab_size=256, max_seq_len=64)

    def _build(self, name):
        torch.manual_seed(0)
        model = Transformer(self.MODEL_CONFIG)
        return model, build_optimizer(model, OptimizerConfig(name=name, lr=1e-3, fused=False))

    @staticmethod
    def _train(model, opt):
        g = torch.Generator().manual_seed(1)
        for _ in range(4):
            tokens = torch.randint(0, 256, (2, 32), generator=g)
            logits = model(tokens)
            torch.nn.functional.cross_entropy(logits.flatten(0, 1), tokens.flatten()).backward()
            opt.step()
            opt.zero_grad()
        return [p.detach().clone() for p in model.parameters()]

    @pytest.mark.parametrize("name", REGISTERED_OPTIMIZERS)
    def test_save_and_resume_leave_the_run_unchanged(self, tmp_path, name):
        """A step-0 save and its resume leave the run as if it never saved."""
        from kempnerforge.checkpoint.manager import CheckpointManager

        reference = self._train(*self._build(name))

        model, opt = self._build(name)
        manager = CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model, opt)
        manager.save(step=0)
        assert not any(o.state for o in manager.checkpointed_optimizers), (
            "the save left the getter's dummy state in an optimizer"
        )
        saved = self._train(model, opt)

        model2, opt2 = self._build(name)
        CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model2, opt2).load()
        resumed = self._train(model2, opt2)

        for ref, a, b in zip(reference, saved, resumed, strict=True):
            assert torch.equal(ref, a), "saving at step 0 changed the run"
            assert torch.equal(ref, b), "resuming from the step-0 checkpoint changed the run"

    @pytest.mark.parametrize("name", REGISTERED_OPTIMIZERS)
    def test_the_checkpoint_holds_no_optimizer_state(self, tmp_path, name):
        """The step-0 checkpoint has no optimizer entry."""
        from kempnerforge.checkpoint.manager import CheckpointManager

        model, opt = self._build(name)
        CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model, opt).save(step=0)
        keys = dcp.FileSystemReader(str(tmp_path / "step_0")).read_metadata().state_dict_metadata
        assert keys, "fixture: the checkpoint should hold the model"
        assert not [k for k in keys if k.startswith("optimizer")], "the save wrote the optimizer"

    def test_inner_optimizer_state_alone_is_saved(self, tmp_path):
        """A save writes the optimizer when only Muon's inner AdamW holds state."""
        from kempnerforge.checkpoint.manager import CheckpointManager

        model, opt = self._build("muon")
        tokens = torch.randint(0, 256, (2, 32), generator=torch.Generator().manual_seed(1))
        logits = model(tokens)
        torch.nn.functional.cross_entropy(logits.flatten(0, 1), tokens.flatten()).backward()
        opt._adam.step()
        assert opt._adam.state and not opt.state, "fixture: only the inner AdamW should hold state"

        CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model, opt).save(step=1)
        keys = dcp.FileSystemReader(str(tmp_path / "step_1")).read_metadata().state_dict_metadata
        assert [k for k in keys if k.startswith("optimizer.state.")], "no optimizer state saved"

    def _load_model_only(self, tmp_path, name, train_step=None):
        """Load a weights-only checkpoint, plus a train state at ``train_step`` if given."""
        import logging

        from torch.distributed.checkpoint.state_dict import get_model_state_dict

        from kempnerforge.checkpoint.manager import CheckpointManager

        source, _ = self._build(name)
        step_dir = tmp_path / "pretrained"
        dcp.save({"model": get_model_state_dict(source)}, checkpoint_id=str(step_dir))
        if train_step is not None:
            torch.save(
                build_train_state(step=train_step, tokens_seen=0), step_dir / "train_state.pt"
            )

        torch.manual_seed(1)
        model = Transformer(self.MODEL_CONFIG)
        opt = build_optimizer(model, OptimizerConfig(name=name, lr=1e-3, fused=False))
        manager = CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model, opt)
        records: list[logging.LogRecord] = []

        class _CaptureHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        target = logging.getLogger("kempnerforge.checkpoint.manager")
        capture = _CaptureHandler(level=logging.WARNING)
        target.addHandler(capture)
        try:
            loaded = manager.load(path=str(step_dir))
        finally:
            target.removeHandler(capture)
        warnings = [r.getMessage() for r in records]
        return source, model, manager.checkpointed_optimizers, loaded, warnings

    @pytest.mark.parametrize("name", REGISTERED_OPTIMIZERS)
    def test_a_checkpoint_without_an_optimizer_loads_like_a_pretrained_model(self, tmp_path, name):
        """A weights-only checkpoint loads with a fresh optimizer and no warning."""
        source, model, optimizers, loaded, warnings = self._load_model_only(tmp_path, name)
        assert loaded == (0, 0, {})
        for want, got in zip(source.parameters(), model.parameters(), strict=True):
            assert torch.equal(want, got), "the load did not restore the weights"
        assert not any(o.state for o in optimizers), "the optimizer should start fresh"
        assert not warnings, f"nothing was lost at step 0, yet: {warnings}"

    @pytest.mark.parametrize("name", REGISTERED_OPTIMIZERS)
    def test_past_step_zero_a_missing_optimizer_is_warned_about(self, tmp_path, name):
        """Past step 0, a checkpoint without an optimizer loads with a warning."""
        _, _, optimizers, loaded, warnings = self._load_model_only(tmp_path, name, train_step=5)
        assert loaded[0] == 5
        assert not any(o.state for o in optimizers), "the optimizer should start fresh"
        assert len(warnings) == 1 and "no optimizer state at step 5" in warnings[0], warnings

    @pytest.mark.parametrize("name", REGISTERED_OPTIMIZERS)
    def test_loading_it_empties_an_optimizer_that_has_stepped(self, tmp_path, name):
        """Loading the step-0 checkpoint returns a stepped optimizer to its freshly built state."""
        from kempnerforge.checkpoint.manager import CheckpointManager

        model, opt = self._build(name)
        manager = CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model, opt)
        manager.save(step=0)
        initial = [p.detach().clone() for p in model.parameters()]
        self._train(model, opt)
        assert all(o.state for o in manager.checkpointed_optimizers), "fixture: no state to empty"

        manager.load(path=str(tmp_path / "step_0"))
        _, fresh = self._build(name)
        assert opt.state_dict() == fresh.state_dict(), "the load left the optimizer partly stepped"
        for before, p in zip(initial, model.parameters(), strict=True):
            assert torch.equal(before, p), "the load did not restore the initial weights"


def _assert_same_state_dict(expected, actual, where="state_dict"):
    """Assert two optimizer ``state_dict()``s hold the same values, tensors bit for bit."""
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor) and torch.equal(expected, actual), where
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys(), f"{where}: keys {set(expected) ^ set(actual)}"
        for key in expected:
            _assert_same_state_dict(expected[key], actual[key], f"{where}[{key!r}]")
    elif isinstance(expected, list | tuple):
        assert len(expected) == len(actual), f"{where}: lengths differ"
        for i, (e, a) in enumerate(zip(expected, actual, strict=True)):
            _assert_same_state_dict(e, a, f"{where}[{i}]")
    else:
        assert expected == actual, f"{where}: {expected!r} != {actual!r}"


class TestCheckpointEveryOptimizer:
    """Saving and resuming keep every registered optimizer's state exactly."""

    MODEL_CONFIG = ModelConfig(dim=64, n_layers=2, n_heads=2, vocab_size=256, max_seq_len=64)

    def _build(self, name, seed):
        torch.manual_seed(seed)
        model = Transformer(self.MODEL_CONFIG)
        return model, build_optimizer(model, OptimizerConfig(name=name, lr=1e-3, fused=False))

    @staticmethod
    def _batches(n):
        g = torch.Generator().manual_seed(1)
        return [torch.randint(0, 256, (2, 32), generator=g) for _ in range(n)]

    @staticmethod
    def _train(model, opt, batches):
        for tokens in batches:
            logits = model(tokens)
            torch.nn.functional.cross_entropy(logits.flatten(0, 1), tokens.flatten()).backward()
            opt.step()
            opt.zero_grad()

    @pytest.mark.parametrize("name", REGISTERED_OPTIMIZERS)
    def test_saving_leaves_the_optimizer_unchanged(self, tmp_path, name):
        """Saves at step 0 and mid-run change neither the optimizer's state nor the run."""
        from kempnerforge.checkpoint.manager import CheckpointManager

        batches = self._batches(4)
        reference, ref_opt = self._build(name, seed=0)
        self._train(reference, ref_opt, batches)

        model, opt = self._build(name, seed=0)
        manager = CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model, opt)
        for step, segment in ((0, batches[:2]), (2, batches[2:])):
            before = copy.deepcopy(opt.state_dict())
            manager.save(step=step)
            _assert_same_state_dict(before, opt.state_dict(), f"after save(step={step})")
            self._train(model, opt, segment)

        _assert_same_state_dict(ref_opt.state_dict(), opt.state_dict())
        for (n, ref), p in zip(reference.named_parameters(), model.parameters(), strict=True):
            assert torch.equal(ref, p), f"{n} differs from the run that never saved"

    @pytest.mark.parametrize("name", RESUMABLE_OPTIMIZERS)
    def test_resume_restores_the_optimizer_exactly(self, tmp_path, name):
        """A resume restores the saved optimizer state bit for bit and continues the run exactly."""
        from kempnerforge.checkpoint.manager import CheckpointManager

        batches = self._batches(4)
        reference, ref_opt = self._build(name, seed=0)
        self._train(reference, ref_opt, batches)

        model, opt = self._build(name, seed=0)
        self._train(model, opt, batches[:2])
        CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model, opt).save(step=2)

        model2, opt2 = self._build(name, seed=1)
        CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model2, opt2).load()
        _assert_same_state_dict(opt.state_dict(), opt2.state_dict(), "after the load")
        self._train(model2, opt2, batches[2:])

        _assert_same_state_dict(ref_opt.state_dict(), opt2.state_dict())
        for (n, ref), p in zip(reference.named_parameters(), model2.parameters(), strict=True):
            assert torch.equal(ref, p), f"{n} differs from the uninterrupted run"


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
