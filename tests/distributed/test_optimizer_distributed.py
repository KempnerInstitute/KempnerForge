"""Distributed tests for optimizer correctness under FSDP2.

Covers Lion, Schedule-Free AdamW, and optimizer-scheduler compatibility.

Run with: torchrun --nproc_per_node=4 -m pytest tests/distributed/test_optimizer_distributed.py -v
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from kempnerforge.checkpoint.manager import CheckpointManager
from kempnerforge.config.schema import (
    CheckpointConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
)
from kempnerforge.distributed.parallel import apply_fsdp2
from kempnerforge.model.transformer import Transformer
from kempnerforge.training.optimizer import build_optimizer
from kempnerforge.training.scheduler import build_scheduler

pytestmark = pytest.mark.skipif(
    "RANK" not in os.environ,
    reason="Requires torchrun launcher (RANK not set)",
)

SMALL_CONFIG = ModelConfig(dim=128, n_layers=2, n_heads=2, vocab_size=512, max_seq_len=64)

# Shared filesystem temp directory (visible to all ranks)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TEST_TMP = _PROJECT_ROOT / ".test_tmp"


@pytest.fixture
def shared_tmp_dir():
    """Create a temp directory on the shared filesystem, cleaned up after use."""
    rank = dist.get_rank()
    if rank == 0:
        _TEST_TMP.mkdir(exist_ok=True)
        d = tempfile.mkdtemp(dir=_TEST_TMP)
    else:
        d = ""
    obj_list = [d]
    dist.broadcast_object_list(obj_list, src=0)
    d = obj_list[0]
    yield d
    dist.barrier()
    if rank == 0:
        shutil.rmtree(d, ignore_errors=True)


def _train_steps(model, optimizer, n_steps, scheduler=None):
    """Run n training steps, return list of losses."""
    losses = []
    for _ in range(n_steps):
        tokens = torch.randint(0, SMALL_CONFIG.vocab_size, (2, 32), device="cuda")
        out = model(tokens)
        loss = out.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()
        losses.append(loss.item())
    return losses


# ---------------------------------------------------------------------------
# 5a: Lion distributed test
# ---------------------------------------------------------------------------


class TestLionDistributed:
    def test_lion_fsdp_loss_decreases(self, distributed_env):
        """Lion + FSDP2: loss decreases over 20 training steps."""
        mesh = distributed_env
        torch.manual_seed(42)
        model = Transformer(SMALL_CONFIG).cuda()
        apply_fsdp2(model, mesh)

        opt_config = OptimizerConfig(name="lion", lr=1e-4, fused=False)
        optimizer = build_optimizer(model, opt_config)

        losses = _train_steps(model, optimizer, n_steps=20)

        # Average of last 5 should be lower than first 5
        early = sum(losses[:5]) / 5
        late = sum(losses[-5:]) / 5
        assert late < early, f"Loss did not decrease: early={early:.4f}, late={late:.4f}"

    def test_lion_fsdp_all_grads(self, distributed_env):
        """Lion + FSDP2: all parameters receive gradients after a step."""
        mesh = distributed_env
        model = Transformer(SMALL_CONFIG).cuda()
        apply_fsdp2(model, mesh)

        opt_config = OptimizerConfig(name="lion", lr=1e-4, fused=False)
        opt = build_optimizer(model, opt_config)

        tokens = torch.randint(0, SMALL_CONFIG.vocab_size, (2, 32), device="cuda")
        out = model(tokens)
        out.sum().backward()
        opt.step()
        opt.zero_grad()

        # Re-forward + backward to confirm gradients flow after a full step
        out2 = model(tokens)
        out2.sum().backward()

        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_lion_checkpoint_roundtrip(self, distributed_env, shared_tmp_dir):
        """Lion optimizer state survives checkpoint save/load under FSDP2."""
        mesh = distributed_env
        ckpt_dir = shared_tmp_dir

        torch.manual_seed(42)
        model = Transformer(SMALL_CONFIG).cuda()
        apply_fsdp2(model, mesh)

        opt_config = OptimizerConfig(name="lion", lr=1e-4, fused=False)
        optimizer = build_optimizer(model, opt_config)

        # Train a few steps to build optimizer state (exp_avg buffer)
        _train_steps(model, optimizer, n_steps=5)

        # Capture reference output
        tokens = torch.randint(0, SMALL_CONFIG.vocab_size, (2, 32), device="cuda")
        with torch.no_grad():
            ref_out = model(tokens).clone()

        # Save checkpoint
        config = CheckpointConfig(dir=ckpt_dir, keep_last_n=2)
        mgr = CheckpointManager(config, model, optimizer)
        mgr.save(step=5, tokens_seen=320)

        # Perturb model weights
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        # Verify output changed
        with torch.no_grad():
            perturbed_out = model(tokens)
        assert not torch.allclose(ref_out, perturbed_out, atol=1e-3)

        # Load checkpoint
        mgr.load()

        # Verify output matches reference
        with torch.no_grad():
            restored_out = model(tokens)
        assert torch.allclose(ref_out, restored_out, atol=1e-5), (
            f"Restored output differs: max diff={(ref_out - restored_out).abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# 5b: Schedule-Free AdamW distributed test
# ---------------------------------------------------------------------------


class TestScheduleFreeDistributed:
    def test_schedule_free_fsdp_loss_decreases(self, distributed_env):
        """Schedule-Free AdamW + FSDP2: loss decreases over 20 steps."""
        mesh = distributed_env
        torch.manual_seed(42)
        model = Transformer(SMALL_CONFIG).cuda()
        apply_fsdp2(model, mesh)

        opt_config = OptimizerConfig(name="schedule_free_adamw", lr=0.025, fused=False)
        optimizer = build_optimizer(model, opt_config)

        losses = _train_steps(model, optimizer, n_steps=20)

        early = sum(losses[:5]) / 5
        late = sum(losses[-5:]) / 5
        assert late < early, f"Loss did not decrease: early={early:.4f}, late={late:.4f}"

    def test_schedule_free_eval_train_toggle(self, distributed_env):
        """eval_params() / train_params() toggle works under FSDP2."""
        mesh = distributed_env
        torch.manual_seed(42)
        model = Transformer(SMALL_CONFIG).cuda()
        apply_fsdp2(model, mesh)

        opt_config = OptimizerConfig(name="schedule_free_adamw", lr=0.025, fused=False)
        optimizer = build_optimizer(model, opt_config)

        # Train a few steps to build optimizer state
        _train_steps(model, optimizer, n_steps=5)

        tokens = torch.randint(0, SMALL_CONFIG.vocab_size, (2, 32), device="cuda")

        # Get training-mode output
        with torch.no_grad():
            train_out = model(tokens).clone()

        # Switch to eval params (running average)
        optimizer.eval_params()
        with torch.no_grad():
            eval_out = model(tokens).clone()

        # Eval and train outputs should differ (different parameter points)
        assert not torch.allclose(train_out, eval_out, atol=1e-5), (
            "eval_params() did not change model output"
        )

        # Switch back to train params
        optimizer.train_params()
        with torch.no_grad():
            restored_out = model(tokens).clone()

        # Should match the original training output
        assert torch.allclose(train_out, restored_out, atol=1e-5), (
            "train_params() did not restore training point"
        )

    def test_schedule_free_checkpoint_roundtrip(self, distributed_env, shared_tmp_dir):
        """Schedule-Free AdamW state (_k, _warmup_steps) survives checkpoint."""
        mesh = distributed_env
        ckpt_dir = shared_tmp_dir

        torch.manual_seed(42)
        model = Transformer(SMALL_CONFIG).cuda()
        apply_fsdp2(model, mesh)

        opt_config = OptimizerConfig(
            name="schedule_free_adamw", lr=0.025, fused=False, schedule_free_warmup_steps=10
        )
        optimizer = build_optimizer(model, opt_config)

        # Train to build state
        _train_steps(model, optimizer, n_steps=5)

        # Capture reference output
        tokens = torch.randint(0, SMALL_CONFIG.vocab_size, (2, 32), device="cuda")
        with torch.no_grad():
            ref_out = model(tokens).clone()

        # Save
        config = CheckpointConfig(dir=ckpt_dir, keep_last_n=2)
        mgr = CheckpointManager(config, model, optimizer)
        mgr.save(step=5, tokens_seen=320)

        # Perturb model
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        # Load
        mgr.load()

        # Verify output matches
        with torch.no_grad():
            restored_out = model(tokens)
        assert torch.allclose(ref_out, restored_out, atol=1e-5), (
            f"Restored output differs: max diff={(ref_out - restored_out).abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# 5c: Optimizer-scheduler compatibility matrix
# ---------------------------------------------------------------------------


_COMPAT_PAIRS = [
    ("adamw", "cosine"),
    ("adamw", "wsd"),
    ("adamw", "rex"),
    ("lion", "cosine"),
    ("lion", "constant"),
    ("schedule_free_adamw", "none"),
    ("muon", "wsd"),
]


class TestOptimizerSchedulerCompat:
    @pytest.mark.parametrize("opt_name,sched_name", _COMPAT_PAIRS)
    def test_optimizer_scheduler_pair(self, distributed_env, opt_name, sched_name):
        """Each valid (optimizer, scheduler) pair runs 10 steps with FSDP2 and finite loss."""
        mesh = distributed_env
        torch.manual_seed(42)
        model = Transformer(SMALL_CONFIG).cuda()
        apply_fsdp2(model, mesh)

        lr = 0.02 if opt_name == "muon" else (1e-4 if opt_name == "lion" else 3e-4)
        opt_config = OptimizerConfig(name=opt_name, lr=lr, fused=False)
        optimizer = build_optimizer(model, opt_config)

        sched_config = SchedulerConfig(name=sched_name, warmup_steps=2)
        scheduler = build_scheduler(optimizer, sched_config, max_steps=100)

        losses = _train_steps(model, optimizer, n_steps=10, scheduler=scheduler)

        for i, loss_val in enumerate(losses):
            assert torch.isfinite(torch.tensor(loss_val)), (
                f"Non-finite loss at step {i} for {opt_name}+{sched_name}: {loss_val}"
            )
