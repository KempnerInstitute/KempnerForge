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
