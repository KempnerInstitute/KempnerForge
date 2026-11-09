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
from concurrent.futures import Future
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
from kempnerforge.training.optimizer import Muon, _should_decay, build_optimizer


class TestRNGState:
    def test_round_trip(self):
        """RNG state save/restore should produce identical sequences."""
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)

        state = get_rng_state()

        a_torch = torch.randn(5)
        a_python = random.random()
        a_numpy = np.random.randn(5)

        set_rng_state(state)

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
        assert state["scheduler"]["_step_count"] == 6

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
        opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
        for _ in range(15):
            opt.step()
            sched.step()
        saved_state = sched.state_dict()

        state = {"step": 15, "tokens_seen": 0, "scheduler": saved_state}

        opt2 = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)
        sched2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=10, gamma=0.5)
        restore_train_state(state, scheduler=sched2)

        assert sched2.state_dict()["last_epoch"] == saved_state["last_epoch"]
        assert sched2.state_dict()["_last_lr"] == saved_state["_last_lr"]

    def test_restores_rng(self):
        torch.manual_seed(0)
        state = build_train_state(step=0, tokens_seen=0)

        _ = torch.randn(100)

        restore_train_state(state)
        a = torch.randn(5)

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


class TestCheckpointRetention:
    def test_cleanup_old_checkpoints(self, tmp_path):
        """Retention policy should keep only the last N checkpoints."""
        from kempnerforge.checkpoint.manager import CheckpointManager

        config = CheckpointConfig(dir=str(tmp_path), keep_last_n=2)
        model = torch.nn.Linear(4, 4)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        mgr = CheckpointManager(config, model, opt)

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


class TestAsyncCheckpointer:
    def test_default_mode_is_disabled(self):
        ckpt = AsyncCheckpointer()
        assert ckpt.mode == AsyncCheckpointMode.disabled

    def test_is_pending_initially_false(self):
        ckpt = AsyncCheckpointer()
        assert not ckpt.is_pending

    def test_wait_with_no_pending_is_noop(self):
        ckpt = AsyncCheckpointer()
        ckpt.wait()

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
        mock_future1.result.assert_called_once()


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

        class PlainLoader:
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

        mgr2 = _make_mock_mgr(tmp_path, monkeypatch)
        step, tokens, _ = mgr2.load(path=str(tmp_path / "step_5"))
        assert step == 5
        assert tokens == 128
        assert mgr2._pending_dataloader_state == {"epoch": 7, "batches_yielded": 333}

        restorer = RecorderLoader({"epoch": 0, "batches_yielded": 0})
        mgr2.apply_dataloader_state(restorer)
        assert captured["restored"] == {"epoch": 7, "batches_yielded": 333}
        assert mgr2._pending_dataloader_state is None


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
        assert mgr.peek_saved_step(path=str(ckpt_dir)) is None

    def test_returns_none_when_metadata_unreadable(self, tmp_path, monkeypatch):
        import json

        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        ckpt_dir = tmp_path / "step_3"
        ckpt_dir.mkdir()
        (ckpt_dir / "metadata.json").write_text("{ not valid json")
        assert mgr.peek_saved_step(path=str(ckpt_dir)) is None

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
        mgr.save(step=1)
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
        mgr.save(step=1)
        mgr.set_tracking_source(list)
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
        mgr.save(step=1)
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
        mgr._rank = 1
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
        monkeypatch.setattr(logging, "raiseExceptions", False)
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
        mgr.save(step=1)
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
        loader.set_tracking_source(lambda: [self.MLFLOW])
        assert loader.peek_saved_step() == 3
        step, tokens_seen, restored = loader.load()
        assert (step, tokens_seen) == (3, 96)
        assert restored == extra
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


class _PrunedDirs(list):
    """Directories passed to ``shutil.rmtree``, in order; ``refused`` holds those NFS kept."""

    def __init__(self) -> None:
        super().__init__()
        self.refused: list[Path] = []

    def still_present(self) -> list[Path]:
        """The directories that exist although their removal went through."""
        return [d for d in self if d not in self.refused and d.exists()]


@pytest.fixture
def pruned_dirs(tmp_path, monkeypatch) -> _PrunedDirs:
    """The directories under ``tmp_path`` passed to ``shutil.rmtree``, in order.

    ``tmp_path`` is the manager's base directory in the tests that use this; other paths
    go to ``shutil.rmtree`` unchanged. On NFS, removing a directory a moment after writing
    into it can fail with ``ENOTEMPTY`` or ``EBUSY`` (an ``.nfsXXXX`` entry keeps it
    non-empty), so such a failure is noted in ``refused`` instead of raised, and a test
    asserts on this list rather than on what the filesystem still holds. Any other error
    is raised as usual.
    """
    import errno

    from kempnerforge.checkpoint import manager as mgr_mod

    rmtree = mgr_mod.shutil.rmtree
    pruned = _PrunedDirs()

    def record(path, *args, **kwargs):
        if not Path(path).is_relative_to(tmp_path):
            return rmtree(path, *args, **kwargs)
        pruned.append(Path(path))
        try:
            rmtree(path, *args, **kwargs)
        except OSError as e:
            if e.errno not in (errno.ENOTEMPTY, errno.EBUSY):
                raise
            pruned.refused.append(Path(path))

    monkeypatch.setattr(mgr_mod.shutil, "rmtree", record)
    return pruned


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

    def test_a_registrar_that_raises_costs_only_the_registration(
        self, tmp_path, monkeypatch, pruned_dirs
    ):
        warnings = self._log(monkeypatch, "warning")
        mgr = _make_mock_mgr(tmp_path, monkeypatch)
        mgr.config.keep_last_n = 1
        mgr.set_tracking_source(lambda: [self.MLFLOW])
        mgr.set_checkpoint_registrar(_Registrar(error=RuntimeError("no server")))
        mgr.save(step=1, tokens_seen=8)
        mgr.save(step=2, tokens_seen=16)
        assert (tmp_path / "latest").resolve() == (tmp_path / "step_2").resolve()
        assert pruned_dirs == [tmp_path / "step_1"]
        assert pruned_dirs.still_present() == []
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
        mgr.save(step=1)
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

        mgr.save(step=10)
        assert self._latest_target(tmp_path) is None, (
            "latest advanced before the async flush was durable"
        )

        mgr.save(step=20)
        tgt = self._latest_target(tmp_path)
        assert tgt == (tmp_path / "step_10").resolve()
        assert (tgt / ".metadata").exists()

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

        step, tokens, _ = mgr.load(exclude_keys=["model", "optimizer"])
        assert step == 20
        assert tokens == 2000

    def test_cleanup_never_deletes_latest_or_in_flight(self, tmp_path, pruned_dirs):
        """Step 6 is newest, so only the latest and in-flight protections keep steps 3 and 5."""
        mgr = self._mgr(tmp_path, AsyncCheckpointMode.async_pinned)
        mgr.config.keep_last_n = 1
        dirs = {s: self._build_ckpt(tmp_path, s, complete=True) for s in (1, 2, 3, 4, 5, 6)}

        latest = Path(tmp_path) / "latest"
        latest.symlink_to(dirs[3].name)
        mgr._pending_finalize = (5, dirs[5])

        mgr._cleanup()

        assert dirs[3].exists(), "cleanup deleted the live `latest` target"
        assert dirs[5].exists(), "cleanup deleted the in-flight pending checkpoint"
        assert pruned_dirs == [dirs[1], dirs[2], dirs[4]]
        assert pruned_dirs.still_present() == []

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

    @staticmethod
    def _record_agreements(monkeypatch) -> list[object]:
        """Run the manager's status agreement as a one-rank all-reduce, recording each one."""
        groups: list[object] = []
        monkeypatch.setattr("kempnerforge.checkpoint.manager.dist.is_initialized", lambda: True)
        monkeypatch.setattr("kempnerforge.checkpoint.manager.dist.get_rank", lambda group=None: 0)
        monkeypatch.setattr(
            "kempnerforge.checkpoint.manager.dist.get_world_size", lambda group=None: 1
        )
        monkeypatch.setattr(
            "kempnerforge.checkpoint.manager.dist.all_reduce",
            lambda flags, group=None: groups.append(group),
        )
        return groups

    def test_save_agrees_when_distributed(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        mgr = self._mgr(tmp_path, AsyncCheckpointMode.disabled)
        monkeypatch.setattr(mgr._async_ckpt, "save", MagicMock())
        agreements = self._record_agreements(monkeypatch)
        mgr.save(step=1)
        assert agreements, "save() did not agree with the other ranks when distributed"

    def test_wait_and_flush_drain_and_agree_when_distributed(self, tmp_path, monkeypatch):
        mgr = self._mgr(tmp_path, AsyncCheckpointMode.async_pinned)
        agreements = self._record_agreements(monkeypatch)
        mgr.wait()
        mgr.flush_pending_save()
        assert len(agreements) == 2

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
        assert mgr._resolve_dcp_load_dir(incomplete, None) == incomplete


def _no_space(*args: object, **kwargs: object) -> None:
    raise OSError(28, "No space left on device")


class TestAFailedSaveCommitsNothing:
    """A part of a save that raises reaches the caller, and that save is never committed."""

    @staticmethod
    def _mgr(tmp_path, mode):
        from kempnerforge.checkpoint.manager import CheckpointManager

        model = torch.nn.Linear(4, 4)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        return CheckpointManager(CheckpointConfig(dir=str(tmp_path), async_mode=mode), model, opt)

    @staticmethod
    def _flushes(monkeypatch, failure: BaseException | None = None) -> list[str]:
        """Make each ``dcp.async_save`` a flush that has ended, failed with ``failure`` if given.

        Returns the checkpoint ids the saves were dispatched to.
        """
        dispatched: list[str] = []

        def async_save(state_dict, checkpoint_id, process_group=None):
            dispatched.append(checkpoint_id)
            flush: Future[None] = Future()
            if failure is None:
                flush.set_result(None)
            else:
                flush.set_exception(failure)
            return flush

        monkeypatch.setattr("kempnerforge.checkpoint.async_save.dcp.async_save", async_save)
        return dispatched

    def test_a_failed_shard_write_skips_the_train_state(self, tmp_path, monkeypatch):
        mgr = self._mgr(tmp_path, AsyncCheckpointMode.disabled)
        monkeypatch.setattr(mgr._async_ckpt, "save", _no_space)
        with pytest.raises(OSError, match="No space left"):
            mgr.save(step=1)
        assert not (tmp_path / "step_1" / "train_state.pt").exists()
        assert not (tmp_path / "latest").exists()

    def test_a_dcp_failure_is_reported_to_the_other_ranks(self, tmp_path, monkeypatch):
        """DCP's ``CheckpointException``, a ``BaseException``, still reaches the agreement."""
        failure = dcp.CheckpointException("write failed", {})

        def write(*args, **kwargs):
            raise failure

        mgr = self._mgr(tmp_path, AsyncCheckpointMode.disabled)
        monkeypatch.setattr(mgr._async_ckpt, "save", write)
        reported: list[list[int]] = []
        monkeypatch.setattr("kempnerforge.checkpoint.manager.dist.is_initialized", lambda: True)
        monkeypatch.setattr("kempnerforge.checkpoint.manager.dist.get_rank", lambda group=None: 0)
        monkeypatch.setattr(
            "kempnerforge.checkpoint.manager.dist.get_world_size", lambda group=None: 1
        )
        monkeypatch.setattr(
            "kempnerforge.checkpoint.manager.dist.all_reduce",
            lambda flags, group=None: reported.append(flags.tolist()),
        )
        with pytest.raises(dcp.CheckpointException) as raised:
            mgr.save(step=1)
        assert raised.value is failure
        assert reported == [[1]]

    @pytest.mark.parametrize(
        "mode", [AsyncCheckpointMode.disabled, AsyncCheckpointMode.async_], ids=str
    )
    def test_a_failed_metadata_write_commits_nothing(self, tmp_path, monkeypatch, mode):
        mgr = self._mgr(tmp_path, mode)
        monkeypatch.setattr(mgr._async_ckpt, "save", lambda *a, **k: None)
        monkeypatch.setattr(mgr._async_ckpt, "wait", lambda: None)
        monkeypatch.setattr("kempnerforge.checkpoint.manager._replace_file", _no_space)
        with pytest.raises(OSError, match="No space left"):
            mgr.save(step=1)
        mgr.wait()
        assert not (tmp_path / "latest").exists()

    def test_a_failed_commit_raises(self, tmp_path, monkeypatch):
        mgr = self._mgr(tmp_path, AsyncCheckpointMode.disabled)
        monkeypatch.setattr(mgr._async_ckpt, "save", lambda *a, **k: None)
        monkeypatch.setattr(mgr, "_commit_latest", _no_space)
        with pytest.raises(OSError, match="No space left"):
            mgr.save(step=1)

    @pytest.mark.parametrize("drain", ["wait", "flush_pending_save"])
    def test_a_failed_flush_raises_and_commits_nothing(self, tmp_path, monkeypatch, drain):
        mgr = self._mgr(tmp_path, AsyncCheckpointMode.async_)
        monkeypatch.setattr(mgr._async_ckpt, "save", lambda *a, **k: None)
        mgr.save(step=1)
        monkeypatch.setattr(mgr._async_ckpt, "wait", _no_space)
        with pytest.raises(OSError, match="No space left"):
            getattr(mgr, drain)()
        assert mgr._pending_finalize is None
        assert not (tmp_path / "latest").exists()

    def test_a_flush_that_failed_before_the_next_save_commits_nothing(self, tmp_path, monkeypatch):
        mgr = self._mgr(tmp_path, AsyncCheckpointMode.async_)
        dispatched = self._flushes(monkeypatch, OSError(28, "No space left on device"))
        mgr.save(step=1)
        with pytest.raises(OSError, match="No space left"):
            mgr.save(step=2)
        assert dispatched == [str(tmp_path / "step_1")]
        assert mgr._pending_finalize is None
        assert not (tmp_path / "latest").exists()
        assert not (tmp_path / "step_2" / "train_state.pt").exists()

    def test_a_save_that_fails_to_dispatch_still_commits_the_one_before(
        self, tmp_path, monkeypatch
    ):
        """Staging step 2 fails after step 1's flush ended: only step 2 is dropped."""
        mgr = self._mgr(tmp_path, AsyncCheckpointMode.async_)
        self._flushes(monkeypatch)
        mgr.save(step=1)
        monkeypatch.setattr("kempnerforge.checkpoint.async_save.dcp.async_save", _no_space)
        with pytest.raises(OSError, match="No space left"):
            mgr.save(step=2)
        assert (tmp_path / "latest").resolve() == (tmp_path / "step_1").resolve()
        assert not (tmp_path / "step_2" / "train_state.pt").exists()
        mgr.wait()
        assert (tmp_path / "latest").resolve() == (tmp_path / "step_1").resolve()

    @pytest.mark.parametrize(
        "mode", [AsyncCheckpointMode.disabled, AsyncCheckpointMode.async_], ids=str
    )
    def test_a_pruning_error_once_latest_moved_fails_nothing(self, tmp_path, monkeypatch, mode):
        """A checkpoint is committed once ``latest`` names it; pruning after that only warns."""
        mgr = self._mgr(tmp_path, mode)
        monkeypatch.setattr(mgr._async_ckpt, "save", lambda *a, **k: None)
        monkeypatch.setattr(mgr, "_cleanup", _no_space)
        warned: list[str] = []
        monkeypatch.setattr(
            "kempnerforge.checkpoint.manager.logger.warning",
            lambda message, *args: warned.append(str(message)),
        )
        mgr.save(step=1)
        mgr.wait()
        assert (tmp_path / "latest").resolve() == (tmp_path / "step_1").resolve()
        assert any("No space left" in message for message in warned), warned


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

    def test_agree_without_a_process_group_raises_only_its_own_error(self, tmp_path):
        mgr = self._mgr(tmp_path)
        mgr._agree(None)
        with pytest.raises(OSError, match="No space left"):
            mgr._agree(OSError(28, "No space left on device"))

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

    @staticmethod
    def _four_ranks(monkeypatch, failing_elsewhere: list[int]) -> list[object]:
        """Make this rank 0 of 4, with ``failing_elsewhere`` reporting a failure in the all-reduce.

        Returns the groups the all-reduce was issued on.
        """
        import kempnerforge.checkpoint.manager as mgr_mod

        groups: list[object] = []

        def all_reduce(flags, group=None):
            groups.append(group)
            flags[failing_elsewhere] = 1

        monkeypatch.setattr(mgr_mod.dist, "is_initialized", lambda: True)
        monkeypatch.setattr(mgr_mod.dist, "new_group", lambda backend=None: "GLOO_PG")
        monkeypatch.setattr(mgr_mod.dist, "get_rank", lambda group=None: 0)
        monkeypatch.setattr(mgr_mod.dist, "get_world_size", lambda group=None: 4)
        monkeypatch.setattr(mgr_mod.dist, "all_reduce", all_reduce)
        return groups

    def test_agree_uses_dedicated_group_not_default(self, tmp_path, monkeypatch):
        groups = self._four_ranks(monkeypatch, [])
        self._mgr(tmp_path)._agree(None)
        assert groups == ["GLOO_PG"]

    def test_agree_raises_on_every_rank_when_others_failed(self, tmp_path, monkeypatch):
        self._four_ranks(monkeypatch, [1, 3])
        with pytest.raises(RuntimeError, match=r"Checkpoint save failed on ranks \[1, 3\]"):
            self._mgr(tmp_path)._agree(None)

    def test_agree_reraises_this_ranks_own_error(self, tmp_path, monkeypatch):
        self._four_ranks(monkeypatch, [2])
        with pytest.raises(OSError, match="No space left"):
            self._mgr(tmp_path)._agree(OSError(28, "No space left on device"))


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
            torch.save(
                build_train_state(step=1, tokens_seen=0), tmp_path / "step_1" / "train_state.pt"
            )
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
            "state": {"a.weight": {"step": 1}, "b": {}},
            "param_groups": [{"params": ["a.weight", "b"]}, {"params": ["c.bias"]}],
        }
        record = _never_stepped_record(optim_state)
        assert sorted(record) == ["b", "c.bias"]
        assert all(isinstance(entry, torch.Tensor) for entry in record.values())

    def test_only_recorded_parameters_count_as_never_stepped(self):
        from kempnerforge.checkpoint.manager import _params_recorded_never_stepped

        fqns = {"a.weight", "absent", "proj.weight", "proj.weight_scale"}
        saved_keys = [
            "model.proj.weight",
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
        torch.save(build_train_state(step=1, tokens_seen=0), tmp_path / "step_1" / "train_state.pt")

        _, _, manager2 = self._build(1, tmp_path)
        with pytest.raises(
            (CheckpointException, ValueError, RuntimeError), match=re.escape(message)
        ):
            manager2.load(path=str(tmp_path / "step_1"))

    def test_an_opaque_optimizer_entry_keeps_its_loaded_state(self, tmp_path, monkeypatch):
        """A checkpoint that stores the optimizer as one entry loads it whole, and no named
        parameter counts as left out of it."""
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
        monkeypatch.setattr(_version, "_derived_version", "2_3")
        dcp.save(saved, checkpoint_id=str(tmp_path / "step_1"))
        monkeypatch.undo()
        torch.save(build_train_state(step=2, tokens_seen=0), tmp_path / "step_1" / "train_state.pt")
        index = FileSystemReader(str(tmp_path / "step_1")).read_metadata()
        assert "optimizer" in index.state_dict_metadata

        model2, optimizer2 = build(1)
        manager2 = CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model2, optimizer2)
        manager2.load(path=str(tmp_path / "step_1"), fresh_if_unheld=["unused.weight"])
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


class TestFreshIfUnheld:
    """``load(fresh_if_unheld=...)`` resumes the named parameters a saved optimizer did not
    hold with empty state, and every other parameter with its saved state."""

    _build = staticmethod(TestResumeWithUnsteppedParameters._build)
    _step = staticmethod(TestResumeWithUnsteppedParameters._step)
    _assert_same_run = staticmethod(TestResumeWithUnsteppedParameters._assert_same_run)
    UNUSED = ("unused.weight", "unused.bias")

    def test_a_module_the_saved_optimizer_left_out_resumes_without_state(self, tmp_path):
        model, optimizer, manager = self._build(0, tmp_path, freeze_unused=True)
        self._step(model, optimizer, 1)
        manager.save(step=1)

        model2, optimizer2, manager2 = self._build(1, tmp_path)
        loaded = manager2.load(path=str(tmp_path / "step_1"), fresh_if_unheld=self.UNUSED)
        assert loaded == (1, 0, {})
        self._assert_same_run(model, optimizer, model2, optimizer2)
        assert [len(g["params"]) for g in optimizer2.param_groups] == [4]
        assert optimizer2.param_groups[0]["lr"] == 1e-2

        self._step(model2, optimizer2, 2, both=True)
        steps = {name: optimizer2.state[p]["step"].item() for name, p in model2.named_parameters()}
        assert steps == {"used.weight": 2, "used.bias": 2, "unused.weight": 1, "unused.bias": 1}

    def test_a_left_out_parameter_that_is_not_named_still_fails(self, tmp_path):
        from torch.distributed.checkpoint.api import CheckpointException

        model, optimizer, manager = self._build(0, tmp_path, freeze_unused=True)
        self._step(model, optimizer, 1)
        manager.save(step=1)

        _, _, manager2 = self._build(1, tmp_path)
        missing = r"Missing key in checkpoint state_dict: optimizer\.state\.unused\.bias\."
        with pytest.raises(CheckpointException, match=missing):
            manager2.load(path=str(tmp_path / "step_1"), fresh_if_unheld=["unused.weight"])

    def test_a_never_stepped_record_keeps_its_parameter_out_of_the_left_out(self, tmp_path):
        model, optimizer, manager = self._build(0, tmp_path)
        self._step(model, optimizer, 1)
        manager.save(step=1)

        model2, optimizer2, manager2 = self._build(1, tmp_path)
        manager2.load(path=str(tmp_path / "step_1"), fresh_if_unheld=self.UNUSED)
        self._assert_same_run(model, optimizer, model2, optimizer2)

    def test_names_are_compared_without_the_activation_checkpoint_prefix(self, tmp_path):
        model, optimizer, manager = self._build(0, tmp_path, freeze_unused=True)
        self._step(model, optimizer, 1)
        manager.save(step=1)

        model2, optimizer2, manager2 = self._build(1, tmp_path)
        manager2.load(
            path=str(tmp_path / "step_1"),
            fresh_if_unheld=[
                "unused._checkpoint_wrapped_module.weight",
                "unused._checkpoint_wrapped_module.bias",
            ],
        )
        self._assert_same_run(model, optimizer, model2, optimizer2)

    @staticmethod
    def _grouped(seed, tmp_path, groups_of, lrs):
        """A model with ``unused`` frozen unless a group holds it, and an AdamW whose groups
        ``groups_of`` picks from the model, each with the lr in ``lrs``."""
        from kempnerforge.checkpoint.manager import CheckpointManager

        torch.manual_seed(seed)
        model = _UsedAndUnused()
        groups = [
            {"params": params, "lr": lr} for params, lr in zip(groups_of(model), lrs, strict=True)
        ]
        held = {id(p) for group in groups for p in group["params"]}
        model.unused.requires_grad_(any(id(p) in held for p in model.unused.parameters()))
        optimizer = torch.optim.AdamW(groups)
        manager = CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model, optimizer)
        return model, optimizer, manager

    def test_left_out_parameters_join_the_empty_saved_group_in_their_groups_place(self, tmp_path):
        """With no saved peer, they take the saved settings of the empty group in their place."""
        model, optimizer, manager = self._grouped(
            0, tmp_path, lambda m: [list(m.used.parameters()), []], [5e-3, 2e-3]
        )
        self._step(model, optimizer, 1)
        manager.save(step=1)

        model2, optimizer2, manager2 = self._grouped(
            1,
            tmp_path,
            lambda m: [list(m.used.parameters()), list(m.unused.parameters())],
            [1e-2, 1e-2],
        )
        manager2.load(path=str(tmp_path / "step_1"), fresh_if_unheld=self.UNUSED)
        assert [g["lr"] for g in optimizer2.param_groups] == [5e-3, 2e-3]
        assert [len(g["params"]) for g in optimizer2.param_groups] == [2, 2]
        assert not any(optimizer2.state.get(p) for p in model2.unused.parameters())
        self._step(model2, optimizer2, 2, both=True)
        assert all(optimizer2.state[p]["step"].item() == 1 for p in model2.unused.parameters())

    def test_a_group_no_saved_group_fits_is_refused(self, tmp_path):
        """The saved group in their group's place holds other parameters, so no saved group
        gives the left-out parameters their settings."""
        model, optimizer, manager = self._grouped(
            0, tmp_path, lambda m: [[m.used.weight], [m.used.bias]], [1e-2, 1e-2]
        )
        self._step(model, optimizer, 1)
        manager.save(step=1)

        _, _, manager2 = self._grouped(
            1,
            tmp_path,
            lambda m: [list(m.used.parameters()), list(m.unused.parameters())],
            [1e-2, 1e-2],
        )
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "has no param group for 2 parameters its optimizer did not hold: "
                "['unused.weight', 'unused.bias']"
            ),
        ):
            manager2.load(path=str(tmp_path / "step_1"), fresh_if_unheld=self.UNUSED)

    @pytest.mark.parametrize(
        "mutate",
        [
            pytest.param(
                lambda state: state["optimizer"]["state"].pop("unused.weight"),
                id="whole_state_missing",
            ),
            pytest.param(
                lambda state: state["optimizer"]["state"].update(
                    renamed=state["optimizer"]["state"].pop("unused.weight")
                ),
                id="state_key_renamed",
            ),
        ],
    )
    def test_a_parameter_the_saved_optimizer_held_without_state_still_fails(self, tmp_path, mutate):
        """The saved groups list it, so its state went missing; naming it does not let it
        resume without state."""
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
        torch.save(build_train_state(step=1, tokens_seen=0), tmp_path / "step_1" / "train_state.pt")

        _, _, manager2 = self._build(1, tmp_path)
        with pytest.raises(
            RuntimeError,
            match=re.escape("for 1 parameters the saved optimizer held: ['unused.weight']"),
        ):
            manager2.load(path=str(tmp_path / "step_1"), fresh_if_unheld=["unused.weight"])

    def test_never_stepped_and_left_out_parameters_resume_together(self, tmp_path):
        """``extra`` was held but never stepped, ``unused`` was left out: both resume without
        state, ``used`` with its saved state."""
        model, optimizer, manager = self._build(
            0, tmp_path, model_cls=_WithExtra, freeze_unused=True
        )
        self._step(model, optimizer, 1)
        manager.save(step=1)

        model2, optimizer2, manager2 = self._build(1, tmp_path, model_cls=_WithExtra)
        manager2.load(path=str(tmp_path / "step_1"), fresh_if_unheld=self.UNUSED)
        self._assert_same_run(model, optimizer, model2, optimizer2)
        assert [len(g["params"]) for g in optimizer2.param_groups] == [6]


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
        torch.save(build_train_state(step=3, tokens_seen=0), step_dir / "train_state.pt")

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

    @staticmethod
    def _save_stepped(model, opt, ckpt_dir):
        """Step ``opt`` once on random gradients and save it."""
        from kempnerforge.checkpoint.manager import CheckpointManager

        for p in model.parameters():
            p.grad = torch.randn_like(p)
        opt.step()
        CheckpointManager(CheckpointConfig(dir=str(ckpt_dir)), model, opt).save(step=1)

    def test_checkpoint_from_the_shape_split_fails_naming_what_moved(self, tmp_path):
        """A checkpoint with the embedding and head on Muon fails to resume, naming both."""
        from kempnerforge.checkpoint.manager import CheckpointManager

        config = ModelConfig(dim=64, n_layers=1, n_heads=2, vocab_size=256, max_seq_len=64)
        model = Transformer(config).to(self.DEVICE)
        matrices = [p for p in model.parameters() if p.ndim == 2]
        rest = [p for p in model.parameters() if p.ndim != 2]
        old = Muon([{"params": matrices}], [{"params": rest, "weight_decay": 0.0}], lr=1e-3)
        self._save_stepped(model, old, tmp_path)

        model2 = Transformer(config).to(self.DEVICE)
        opt2 = build_optimizer(model2, OptimizerConfig(name="muon", lr=1e-3, fused=False))
        manager = CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model2, opt2)
        moved = "2 parameters: ['output_head.proj.weight', 'token_embedding.embedding.weight']"
        with pytest.raises(RuntimeError, match=re.escape(moved)) as raised:
            manager.load()
        assert "exclude_from_loading" in str(raised.value)

    def test_resume_with_muon_router_flipped_fails_naming_the_gate(self, tmp_path):
        """Flipping muon_router across a resume fails, naming the router gate."""
        from kempnerforge.checkpoint.manager import CheckpointManager

        config = ModelConfig(
            dim=64, n_layers=1, n_heads=2, vocab_size=256, max_seq_len=64, num_experts=4
        )
        model = Transformer(config).to(self.DEVICE)
        opt = build_optimizer(model, OptimizerConfig(name="muon", lr=1e-3, fused=False))
        self._save_stepped(model, opt, tmp_path)

        model2 = Transformer(config).to(self.DEVICE)
        options = OptimizerConfig(name="muon", lr=1e-3, fused=False, muon_router=True)
        manager = CheckpointManager(
            CheckpointConfig(dir=str(tmp_path)), model2, build_optimizer(model2, options)
        )
        with pytest.raises(RuntimeError, match=re.escape("['layers.0.mlp.router.gate.weight']")):
            manager.load()

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
        torch.save(build_train_state(step=3, tokens_seen=0), step_dir / "train_state.pt")

        model2, opt2 = self._build(seed=1)
        CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model2, opt2).load(
            path=str(step_dir)
        )
        groups = {(g["orthogonalize"], g["nesterov"], g["adjust_lr"]) for g in opt2.param_groups}
        assert groups == {("local", False, "original")}
        self._assert_same_state(opt, opt2, "Muon")

    LEFT_OUT = ("token_embedding.embedding.weight", "output_head.proj.weight")

    def _build_without_the_embeddings(self, seed):
        """A model whose token embedding and output head do not train, and its Muon."""
        torch.manual_seed(seed)
        model = Transformer(self.MODEL_CONFIG).to(self.DEVICE)
        for name, param in model.named_parameters():
            param.requires_grad_(name not in self.LEFT_OUT)
        return model, build_optimizer(model, OptimizerConfig(name="muon", lr=1e-3, fused=False))

    def test_a_checkpoint_without_the_inner_group_of_an_unfrozen_module_resumes(self, tmp_path):
        """A checkpoint whose inner AdamW kept no group for left-out parameters resumes.

        The save left out the embedding and head, so the checkpoint holds one group fewer.
        They resume in their group as built, without state, and every other parameter with
        its saved state.
        """
        from kempnerforge.checkpoint.manager import CheckpointManager

        model, opt = self._build_without_the_embeddings(seed=0)
        self._train(model, opt, self._batches(2))
        CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model, opt).save(step=2)
        assert [len(g["params"]) for g in opt._adam.param_groups] == [5]

        model2, opt2 = self._build(seed=1)
        built = [
            {k: v for k, v in g.items() if k not in ("params", "lr")}
            for g in opt2._adam.param_groups
        ]
        CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model2, opt2).load(
            path=str(tmp_path / "step_2"), fresh_if_unheld=self.LEFT_OUT
        )
        self._assert_same_state(opt, opt2, "Muon")
        self._assert_same_state(opt._adam, opt2._adam, "inner AdamW")
        assert [len(g["params"]) for g in opt2._adam.param_groups] == [2, 5]
        kept = {k: v for k, v in opt2._adam.param_groups[0].items() if k not in ("params", "lr")}
        assert kept == built[0]
        left_out = dict(model2.named_parameters())
        assert not any(opt2._adam.state.get(left_out[name]) for name in self.LEFT_OUT)

        before = {name: left_out[name].detach().clone() for name in self.LEFT_OUT}
        self._train(model2, opt2, self._batches(1))
        assert [g["lr"] for g in opt2._adam.param_groups] == [1e-3, 1e-3]
        for name in self.LEFT_OUT:
            assert opt2._adam.state[left_out[name]]["step"].item() == 1
            assert not torch.equal(before[name], left_out[name]), f"{name} did not train"

    def test_a_checkpoint_with_fewer_groups_for_other_parameters_names_the_way_out(self, tmp_path):
        """A group mismatch the left-out names do not explain names the way out.

        The missing group holds parameters the load was not told to resume fresh, so the
        groups cannot be matched; the error says how to continue from the weights.
        """
        from kempnerforge.checkpoint.manager import CheckpointManager

        model, opt = self._build_without_the_embeddings(seed=0)
        self._train(model, opt, self._batches(2))
        CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model, opt).save(step=2)

        model2, opt2 = self._build(seed=1)
        manager2 = CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model2, opt2)
        way_out = (
            r'holds 2 optimizer param groups and the optimizer being loaded has 3.*\["optimizer"\]'
        )
        with pytest.raises(RuntimeError, match=way_out):
            manager2.load(path=str(tmp_path / "step_2"))

    def test_a_checkpoint_that_lists_left_out_names_without_their_state_still_fails(self, tmp_path):
        """Fewer saved groups do not excuse missing state for names a saved group lists.

        The saved inner groups are merged into one that still lists the embedding and head,
        whose state is gone, so the save held them and their state went missing.
        """
        from torch.distributed.checkpoint.state_dict import (
            get_model_state_dict,
            get_optimizer_state_dict,
        )

        from kempnerforge.checkpoint.manager import CheckpointManager

        model, opt = self._build(seed=0)
        self._train(model, opt, self._batches(2))
        optim_state = get_optimizer_state_dict(model, [opt, opt._adam])
        muon_group, embeddings, norms = optim_state["param_groups"]
        optim_state["param_groups"] = [
            muon_group,
            {**norms, "params": [*embeddings["params"], *norms["params"]]},
        ]
        for name in self.LEFT_OUT:
            del optim_state["state"][name]
        step_dir = tmp_path / "step_2"
        dcp.save(
            {"model": get_model_state_dict(model), "optimizer": optim_state},
            checkpoint_id=str(step_dir),
        )
        torch.save(build_train_state(step=2, tokens_seen=0), step_dir / "train_state.pt")

        model2, opt2 = self._build(seed=1)
        manager2 = CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model2, opt2)
        held = re.escape(f"for 2 parameters the saved optimizer held: {sorted(self.LEFT_OUT)}")
        with pytest.raises(RuntimeError, match=held):
            manager2.load(path=str(step_dir), fresh_if_unheld=self.LEFT_OUT)

    def test_a_group_of_the_scheduled_optimizer_is_never_taken(self, tmp_path):
        """Its LR scheduler restores one rate per saved group, so such a group is refused."""
        from kempnerforge.checkpoint.manager import _take_groups_the_save_lacks

        outer, inner = object(), object()
        state = {"param_groups": [{"params": ["a"]}, {"params": ["b"]}]}
        with pytest.raises(RuntimeError, match="cannot be matched"):
            _take_groups_the_save_lacks(state, [outer, inner], outer, {"a"}, 1, tmp_path)

        owners = [outer, inner]
        taken = _take_groups_the_save_lacks(state, owners, outer, {"b"}, 1, tmp_path)
        assert taken == [(1, {"params": ["b"]})]
        assert state["param_groups"] == [{"params": ["a"]}]
        assert owners == [outer]


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


class TestResumeWithoutTrainState:
    """A checkpoint that holds optimizer state resumes only together with its ``train_state.pt``."""

    @staticmethod
    def _build(seed, ckpt_dir, load_path=None):
        from kempnerforge.checkpoint.manager import CheckpointManager

        torch.manual_seed(seed)
        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        config = CheckpointConfig(dir=str(ckpt_dir), load_path=load_path)
        return model, optimizer, CheckpointManager(config, model, optimizer)

    def _saved_without_train_state(self, tmp_path, lost=("train_state.pt",)):
        """A step-500 checkpoint of a stepped run whose files in ``lost`` are gone."""
        model, optimizer, manager = self._build(0, tmp_path / "run")
        model(torch.ones(2, 4)).sum().backward()
        optimizer.step()
        manager.save(step=500, tokens_seen=1_000_000)
        step_dir = tmp_path / "run" / "step_500"
        for name in lost:
            (step_dir / name).unlink()
        return model, step_dir

    @staticmethod
    def _logged(call):
        """Run ``call`` and return its result and the manager's records as (level, message)."""
        import logging

        records: list[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        target = logging.getLogger("kempnerforge.checkpoint.manager")
        handler, prior_level = _Capture(level=logging.INFO), target.level
        target.addHandler(handler)
        target.setLevel(logging.INFO)
        try:
            result = call()
        finally:
            target.removeHandler(handler)
            target.setLevel(prior_level)
        return result, [(r.levelname, r.getMessage()) for r in records]

    @staticmethod
    def _fresh_start(ckpt_dir):
        """The record of a load that found no ``train_state.pt`` in ``ckpt_dir``."""
        return (
            "INFO",
            f"{ckpt_dir} has no train_state.pt; the step counter, token count, LR schedule, "
            f"data position and RNG state start fresh at step 0",
        )

    @pytest.mark.parametrize(
        "lost",
        [("train_state.pt",), ("train_state.pt", "metadata.json")],
        ids=["train_state_lost", "cut_short_before_both"],
    )
    @pytest.mark.parametrize(
        ("via", "exclude"),
        [("latest", None), ("resolved_path", None), ("load_path", None), ("load_path", ["model"])],
    )
    def test_loading_its_optimizer_state_is_refused_before_anything_loads(
        self, tmp_path, via, exclude, lost
    ):
        """Auto-resume (``latest`` or the path it resolved) and ``load_path`` all refuse it.

        A save cut short after its shards has written neither ``train_state.pt`` nor
        ``metadata.json``, so a missing ``metadata.json`` does not make it an import.
        """
        _, step_dir = self._saved_without_train_state(tmp_path, lost)
        if via == "load_path":
            model, optimizer, manager = self._build(1, tmp_path / "fresh", str(step_dir))
        else:
            model, optimizer, manager = self._build(1, tmp_path / "run")
        before = [p.detach().clone() for p in model.parameters()]

        refused = f"Cannot resume {step_dir}: it holds optimizer state but no train_state.pt"
        path = str(step_dir) if via == "resolved_path" else None
        with pytest.raises(FileNotFoundError, match=re.escape(refused)):
            manager.load(path=path, exclude_keys=exclude)

        for want, got in zip(before, model.parameters(), strict=True):
            assert torch.equal(want, got), "the refused load changed the weights"
        assert not optimizer.state, "the refused load filled the optimizer"

    def test_its_weights_alone_load_at_step_zero_and_say_what_starts_fresh(self, tmp_path):
        """Excluding the optimizer loads the weights, as a warm start from them does."""
        source, step_dir = self._saved_without_train_state(tmp_path)
        model, optimizer, manager = self._build(1, tmp_path / "fresh", str(step_dir))

        loaded, records = self._logged(lambda: manager.load(exclude_keys=["optimizer"]))

        assert loaded == (0, 0, {})
        for want, got in zip(source.parameters(), model.parameters(), strict=True):
            assert torch.equal(want, got), "the weights were not loaded"
        assert not optimizer.state, "the optimizer should start fresh"
        assert self._fresh_start(step_dir) in records, records

    def test_a_checkpoint_without_an_optimizer_says_what_starts_fresh(self, tmp_path):
        """A weights-only checkpoint, like an ``hf-to-dcp`` import, loads at step 0 and says so."""
        from torch.distributed.checkpoint.state_dict import get_model_state_dict

        source, _, _ = self._build(0, tmp_path / "unused")
        imported = tmp_path / "pretrained"
        dcp.save({"model": get_model_state_dict(source)}, checkpoint_id=str(imported))
        model, optimizer, manager = self._build(1, tmp_path / "fresh", str(imported))

        loaded, records = self._logged(manager.load)

        assert loaded == (0, 0, {})
        for want, got in zip(source.parameters(), model.parameters(), strict=True):
            assert torch.equal(want, got), "the weights were not loaded"
        assert not optimizer.state, "the optimizer should start fresh"
        assert [message for level, message in records if level != "INFO"] == []
        assert self._fresh_start(imported) in records, records

    @pytest.mark.parametrize("kind", ["absent", "a_file"])
    def test_a_path_that_is_no_directory_is_not_missing_its_train_state(self, tmp_path, kind):
        """Only a checkpoint directory holds optimizer state, so any other path reports False."""
        from kempnerforge.checkpoint.manager import missing_train_state

        path = tmp_path / kind
        if kind == "a_file":
            path.write_text("")

        assert not missing_train_state(path)


def _free_port() -> int:
    """An unused local TCP port for a gloo rendezvous."""
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _load_with_a_stale_view(rank: int, world: int, port: int, ckpt_dir: str, stale: int) -> None:
    """Save on every rank, then load while rank ``stale`` cannot see ``train_state.pt``."""
    from datetime import timedelta
    from unittest.mock import patch

    import torch.distributed as dist

    from kempnerforge.checkpoint.manager import CheckpointManager

    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world,
        init_method=f"tcp://127.0.0.1:{port}",
        timeout=timedelta(seconds=30),
    )
    try:
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model(torch.ones(2, 4)).sum().backward()
        optimizer.step()
        manager = CheckpointManager(CheckpointConfig(dir=ckpt_dir), model, optimizer)
        manager.save(step=3, tokens_seen=30)
        exists = Path.exists

        def stale_exists(path: Path) -> bool:
            return False if rank == stale and path.name == "train_state.pt" else exists(path)

        with patch.object(Path, "exists", stale_exists):
            if stale == 0:
                refused = "holds optimizer state but no train_state.pt"
                with pytest.raises(FileNotFoundError, match=refused):
                    manager.load()
            else:
                assert manager.load() == (3, 30, {})
    finally:
        dist.destroy_process_group()


class TestResumeWithoutTrainStateAcrossRanks:
    """Every rank takes rank 0's view of ``train_state.pt``, so they refuse or load together."""

    @pytest.mark.parametrize("stale", [0, 1], ids=["rank_0_sees_none", "rank_1_sees_none"])
    def test_every_rank_follows_rank_zero(self, tmp_path, stale):
        """No rank raises alone and leaves the others waiting in a collective."""
        import torch.multiprocessing as mp

        mp.spawn(_load_with_a_stale_view, args=(2, _free_port(), str(tmp_path), stale), nprocs=2)


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


WEIGHT_DECAY = 0.5
DEFAULT_SCOPE, ALL_SCOPE = "exclude_norms_and_biases", "all"


class _ReachedTheModel(Exception):
    """Raised in place of the model load, to observe that a load got past every check."""


class TestWeightDecayScopeOnResume:
    """A resume that restores optimizer state refuses one saved under another scope."""

    MODEL_CONFIG = ModelConfig(dim=64, n_layers=2, n_heads=2, vocab_size=256, max_seq_len=64)

    def _build(self, name, scope, seed):
        torch.manual_seed(seed)
        model = Transformer(self.MODEL_CONFIG)
        config = OptimizerConfig(
            name=name, lr=1e-3, weight_decay=WEIGHT_DECAY, weight_decay_scope=scope, fused=False
        )
        return model, build_optimizer(model, config)

    @staticmethod
    def _train(model, opt, steps):
        g = torch.Generator().manual_seed(1)
        for _ in range(steps):
            tokens = torch.randint(0, 256, (2, 32), generator=g)
            logits = model(tokens)
            torch.nn.functional.cross_entropy(logits.flatten(0, 1), tokens.flatten()).backward()
            opt.step()
            opt.zero_grad()

    @staticmethod
    def _manager(tmp_path, model, opt, scope):
        from kempnerforge.checkpoint.manager import CheckpointManager

        return CheckpointManager(
            CheckpointConfig(dir=str(tmp_path)), model, opt, weight_decay_scope=scope
        )

    def _save(self, tmp_path, name, scope, steps=3, record=True):
        model, opt = self._build(name, scope, seed=0)
        self._train(model, opt, steps)
        self._manager(tmp_path, model, opt, scope if record else None).save(step=steps)
        return tmp_path / f"step_{steps}"

    def _resume(self, tmp_path, name, scope):
        model, opt = self._build(name, scope, seed=1)
        return model, opt, self._manager(tmp_path, model, opt, scope)

    @staticmethod
    def _by_name(model, manager):
        names = {id(p): n for n, p in model.named_parameters()}
        decay, state = {}, {}
        for optimizer in manager.checkpointed_optimizers:
            for group in optimizer.param_groups:
                for p in group["params"]:
                    decay[names[id(p)]] = group["weight_decay"]
            for p, entries in optimizer.state.items():
                for key, value in entries.items():
                    copy_ = value.detach().clone() if torch.is_tensor(value) else value
                    state[(names[id(p)], key)] = copy_
        return decay, state

    def test_the_save_records_the_scope(self, tmp_path):
        """``metadata.json`` carries the scope the manager was built with."""
        step_dir = self._save(tmp_path, "adamw", ALL_SCOPE)
        assert json.loads((step_dir / "metadata.json").read_text())["weight_decay_scope"] == "all"

    @pytest.mark.parametrize(
        ("saved", "resumed"), [(DEFAULT_SCOPE, ALL_SCOPE), (ALL_SCOPE, DEFAULT_SCOPE)]
    )
    @pytest.mark.parametrize("name", REGISTERED_OPTIMIZERS)
    def test_a_different_scope_is_refused_before_anything_loads(
        self, tmp_path, name, saved, resumed
    ):
        """Every optimizer refuses a scope change both ways and leaves the live run untouched."""
        self._save(tmp_path, name, saved)
        model, _, manager = self._resume(tmp_path, name, resumed)
        before = [p.detach().clone() for p in model.parameters()]
        with pytest.raises(
            ValueError, match=f"weight_decay_scope='{saved}', but this run uses '{resumed}'"
        ):
            manager.load()
        for b, p in zip(before, model.parameters(), strict=True):
            assert torch.equal(b, p), "a refused load changed the model"
        assert not any(o.state for o in manager.checkpointed_optimizers)

    @pytest.mark.parametrize("scope", [DEFAULT_SCOPE, ALL_SCOPE])
    @pytest.mark.parametrize("name", RESUMABLE_OPTIMIZERS)
    def test_the_same_scope_resumes_exactly(self, tmp_path, name, scope):
        """A resume under the saved scope restores all optimizer state and continues exactly."""
        model, opt = self._build(name, scope, seed=0)
        self._train(model, opt, 3)
        manager = self._manager(tmp_path, model, opt, scope)
        manager.save(step=3)
        decay, state = self._by_name(model, manager)
        self._train(model, opt, 2)
        reference = [p.detach().clone() for p in model.parameters()]

        model2, opt2, manager2 = self._resume(tmp_path, name, scope)
        manager2.load()
        decay2, state2 = self._by_name(model2, manager2)
        expected = {
            n: WEIGHT_DECAY if _should_decay(n, p, scope) else 0.0
            for n, p in model2.named_parameters()
        }
        assert decay == decay2 == expected
        assert state, "the saved optimizer held no state to compare"
        assert state2.keys() == state.keys()
        for key, value in state.items():
            if torch.is_tensor(value):
                assert torch.equal(state2[key], value), key
            else:
                assert state2[key] == value, key
        self._train(model2, opt2, 2)
        for a, b in zip(reference, model2.parameters(), strict=True):
            assert torch.equal(a, b), "the resumed run diverged from the uninterrupted one"

    def test_a_checkpoint_without_a_recorded_scope_counts_as_the_default(self, tmp_path):
        """A checkpoint written before the scope was recorded resumes only under the default."""
        self._save(tmp_path, "muon", DEFAULT_SCOPE, record=False)
        self._resume(tmp_path, "muon", DEFAULT_SCOPE)[2].load()
        with pytest.raises(ValueError, match=f"weight_decay_scope='{DEFAULT_SCOPE}'"):
            self._resume(tmp_path, "muon", ALL_SCOPE)[2].load()

    def test_a_missing_metadata_file_counts_as_the_default(self, tmp_path):
        """A checkpoint with optimizer state and no ``metadata.json`` resumes under the default."""
        (self._save(tmp_path, "adamw", DEFAULT_SCOPE) / "metadata.json").unlink()
        step, _, _ = self._resume(tmp_path, "adamw", DEFAULT_SCOPE)[2].load()
        assert step == 3

    @pytest.mark.parametrize(
        "content",
        [
            b"\xff\xfe",
            b"{not json",
            b"null",
            b"[]",
            b'{"step": 3, "weight_decay_scope": "x"}',
            b"[" * 100_000 + b"]" * 100_000,
        ],
        ids=["bad-encoding", "bad-json", "null", "list", "unknown-scope", "too-deep"],
    )
    def test_a_metadata_file_without_a_valid_scope_record_is_refused(self, tmp_path, content):
        """Anything but a JSON object with a known scope fails the load with a clear error."""
        (self._save(tmp_path, "adamw", ALL_SCOPE) / "metadata.json").write_bytes(content)
        with pytest.raises(ValueError, match="Cannot check the weight_decay_scope"):
            self._resume(tmp_path, "adamw", ALL_SCOPE)[2].load()

    def test_an_unknown_recorded_scope_names_the_valid_ones(self, tmp_path):
        """The error for an unknown recorded scope lists the scopes that exist."""
        (self._save(tmp_path, "adamw", ALL_SCOPE) / "metadata.json").write_text(
            json.dumps({"step": 3, "weight_decay_scope": "norms"})
        )
        with pytest.raises(ValueError, match="'norms', which is not one of"):
            self._resume(tmp_path, "adamw", ALL_SCOPE)[2].load()

    def test_excluding_the_optimizer_skips_the_check(self, tmp_path):
        """A load that leaves out the optimizer state loads the model under any scope."""
        self._save(tmp_path, "muon", DEFAULT_SCOPE)
        model, _, manager = self._resume(tmp_path, "muon", ALL_SCOPE)
        before = [p.detach().clone() for p in model.parameters()]
        manager.load(exclude_keys=["optimizer"])
        assert any(not torch.equal(b, p) for b, p in zip(before, model.parameters(), strict=True))
        assert not any(o.state for o in manager.checkpointed_optimizers)

    @pytest.mark.parametrize(
        ("saved", "resumed"), [(DEFAULT_SCOPE, ALL_SCOPE), (ALL_SCOPE, DEFAULT_SCOPE)]
    )
    def test_a_checkpoint_without_optimizer_state_loads_under_the_other_scope(
        self, tmp_path, saved, resumed
    ):
        """A step-0 checkpoint holds no optimizer state, so it loads with a fresh optimizer."""
        self._save(tmp_path, "adamw", saved, steps=0)
        model, opt, manager = self._resume(tmp_path, "adamw", resumed)
        step, _, _ = manager.load()
        assert step == 0
        saved_model, _ = self._build("adamw", saved, seed=0)
        for a, b in zip(saved_model.parameters(), model.parameters(), strict=True):
            assert torch.equal(a, b), "the model was not loaded"
        assert not any(o.state for o in manager.checkpointed_optimizers)
        decay, _ = self._by_name(model, manager)
        assert decay == {
            n: WEIGHT_DECAY if _should_decay(n, p, resumed) else 0.0
            for n, p in model.named_parameters()
        }
        self._train(model, opt, 1)
        assert opt.state, "the fresh optimizer did not step"

    @staticmethod
    def _as_rank(monkeypatch, manager, rank, holding=(), failing=(), sent=None):
        """Run ``manager`` as ``rank`` of two, with what the other rank reports scripted.

        ``holding`` and ``failing`` list the other ranks whose DCP index holds optimizer state
        or could not be read; ``sent`` is what rank 0's scope broadcast delivers to any other
        rank. Returns the collectives issued, in order.
        """
        import kempnerforge.checkpoint.manager as mgr_mod

        calls: list[tuple] = []

        def all_reduce(flags, group=None):
            calls.append(("all_reduce", group, flags.tolist()))
            for r in failing:
                flags[0, r] = 1
            for r in holding:
                flags[1, r] = 1

        def broadcast_object_list(objects, src):
            calls.append(("broadcast", src, list(objects)))
            if rank != 0:
                objects[0] = sent

        monkeypatch.setattr(mgr_mod.dist, "is_initialized", lambda: True)
        monkeypatch.setattr(mgr_mod.dist, "get_rank", lambda group=None: rank)
        monkeypatch.setattr(mgr_mod.dist, "get_world_size", lambda group=None: 2)
        monkeypatch.setattr(mgr_mod.dist, "all_reduce", all_reduce)
        monkeypatch.setattr(mgr_mod.dist, "broadcast_object_list", broadcast_object_list)
        manager._rank = rank
        manager._sync_group = "GLOO_GROUP"
        return calls

    @staticmethod
    def _stop_at_the_model_load(monkeypatch):
        """Make the load stop, with ``_ReachedTheModel``, once it starts loading the model."""
        import kempnerforge.checkpoint.manager as mgr_mod

        def reached(_model):
            raise _ReachedTheModel

        monkeypatch.setattr(mgr_mod, "get_model_state_dict", reached)

    @pytest.mark.parametrize("rank", [0, 1])
    @pytest.mark.parametrize(
        ("holds_here", "holds_elsewhere"),
        [(False, False), (True, False), (False, True), (True, True)],
        ids=["none", "here", "elsewhere", "both"],
    )
    def test_ranks_that_disagree_take_the_same_path(
        self, tmp_path, monkeypatch, rank, holds_here, holds_elsewhere
    ):
        """Every rank checks the scope when any rank restores optimizer state, and none otherwise.

        Each rank reports through one all-reduce on the manager's gloo group whether its DCP
        index holds optimizer state, so a rank whose own part holds none still refuses a
        scope mismatch when another rank's part holds some. The scope check broadcasts a
        ``(scope, error)`` tuple, which tells it apart from the train-state check's broadcast.
        """
        self._save(tmp_path, "adamw", DEFAULT_SCOPE, steps=3 if holds_here else 0)
        _, _, manager = self._resume(tmp_path, "adamw", ALL_SCOPE)
        calls = self._as_rank(
            monkeypatch,
            manager,
            rank,
            holding=[1 - rank] if holds_elsewhere else [],
            sent=(DEFAULT_SCOPE, None),
        )
        self._stop_at_the_model_load(monkeypatch)
        if holds_here or holds_elsewhere:
            with pytest.raises(ValueError, match="but this run uses 'all'"):
                manager.load()
        else:
            with pytest.raises(_ReachedTheModel):
                manager.load()
        own = [[0, 0], [0, 0]]
        own[1][rank] = int(holds_here)
        assert calls[0] == ("all_reduce", "GLOO_GROUP", own)
        scope_broadcasts = [c for c in calls if c[0] == "broadcast" and isinstance(c[2][0], tuple)]
        assert len(scope_broadcasts) == int(holds_here or holds_elsewhere)

    def test_a_rank_that_cannot_read_its_index_still_reports_before_raising(
        self, tmp_path, monkeypatch
    ):
        """A failed read still reaches the all-reduce, so the other ranks do not wait on it."""
        step_dir = self._save(tmp_path, "adamw", DEFAULT_SCOPE)
        (step_dir / ".metadata").unlink()
        _, _, manager = self._resume(tmp_path, "adamw", DEFAULT_SCOPE)
        calls = self._as_rank(monkeypatch, manager, 1, sent=(DEFAULT_SCOPE, None))
        with pytest.raises(FileNotFoundError):
            manager.load(path=str(step_dir))
        assert calls == [("all_reduce", "GLOO_GROUP", [[0, 1], [0, 0]])]

    def test_a_single_process_raises_its_own_read_failure(self, tmp_path):
        """Without a process group a failed read of the DCP index raises as it is."""
        step_dir = self._save(tmp_path, "adamw", DEFAULT_SCOPE)
        (step_dir / ".metadata").unlink()
        with pytest.raises(FileNotFoundError):
            self._resume(tmp_path, "adamw", DEFAULT_SCOPE)[2].load(path=str(step_dir))

    def test_every_other_rank_raises_when_one_cannot_read_its_index(self, tmp_path, monkeypatch):
        """A rank that read its index raises too, naming the rank that could not."""
        self._save(tmp_path, "adamw", DEFAULT_SCOPE)
        _, _, manager = self._resume(tmp_path, "adamw", DEFAULT_SCOPE)
        calls = self._as_rank(monkeypatch, manager, 0, failing=[1])
        self._stop_at_the_model_load(monkeypatch)
        with pytest.raises(RuntimeError, match=r"on ranks \[1\]"):
            manager.load()
        assert [c[0] for c in calls] == ["all_reduce"]

    def test_a_manager_without_a_scope_neither_records_nor_checks(self, tmp_path):
        """Callers that do not pass a scope keep the previous behavior."""
        step_dir = self._save(tmp_path, "adamw", ALL_SCOPE, record=False)
        assert "weight_decay_scope" not in json.loads((step_dir / "metadata.json").read_text())
        model, opt = self._build("adamw", ALL_SCOPE, seed=1)
        step, _, _ = self._manager(tmp_path, model, opt, None).load()
        assert step == 3

    def test_rank_zero_broadcasts_what_it_reads(self, tmp_path, monkeypatch):
        """Once the ranks agree that optimizer state is restored, rank 0 sends the saved scope."""
        self._save(tmp_path, "adamw", ALL_SCOPE)
        _, _, manager = self._resume(tmp_path, "adamw", DEFAULT_SCOPE)
        calls = self._as_rank(monkeypatch, manager, 0)
        with pytest.raises(ValueError, match="but this run uses"):
            manager.load()
        assert calls == [
            ("all_reduce", "GLOO_GROUP", [[0, 0], [1, 0]]),
            ("broadcast", 0, [(ALL_SCOPE, None)]),
        ]

    def test_other_ranks_act_on_what_rank_zero_sends(self, tmp_path, monkeypatch):
        """A rank other than 0 never reads ``metadata.json``, so it cannot disagree with rank 0."""
        step_dir = self._save(tmp_path, "adamw", DEFAULT_SCOPE)
        (step_dir / "metadata.json").write_text("{not json")
        _, _, manager = self._resume(tmp_path, "adamw", DEFAULT_SCOPE)
        self._as_rank(monkeypatch, manager, 1, sent=(ALL_SCOPE, None))
        with pytest.raises(ValueError, match="weight_decay_scope='all', but this run uses"):
            manager.load()

    @pytest.mark.parametrize(
        "error", [OSError("disk gone"), RecursionError("too deep")], ids=["oserror", "recursion"]
    )
    def test_any_failure_on_rank_zero_is_broadcast(self, tmp_path, monkeypatch, error):
        """Whatever stops rank 0 reading still reaches the broadcast, so no rank waits forever."""
        self._save(tmp_path, "adamw", DEFAULT_SCOPE)
        _, _, manager = self._resume(tmp_path, "adamw", DEFAULT_SCOPE)

        def fail(_ckpt_dir):
            raise error

        monkeypatch.setattr("kempnerforge.checkpoint.manager._read_weight_decay_scope", fail)
        calls = self._as_rank(monkeypatch, manager, 0)
        with pytest.raises(ValueError, match="Cannot check the weight_decay_scope"):
            manager.load()
        assert calls[-1] == ("broadcast", 0, [(None, f"{type(error).__name__}: {error}")])

    def test_other_ranks_raise_what_rank_zero_could_not_read(self, tmp_path, monkeypatch):
        """A rank other than 0 raises the reason rank 0 sent, without reading anything itself."""
        self._save(tmp_path, "adamw", DEFAULT_SCOPE)
        _, _, manager = self._resume(tmp_path, "adamw", DEFAULT_SCOPE)
        self._as_rank(monkeypatch, manager, 1, sent=(None, "rank 0 could not read it"))
        with pytest.raises(ValueError, match="rank 0 could not read it"):
            manager.load()
