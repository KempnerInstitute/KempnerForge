"""End-to-end distributed training test.

Run with: torchrun --nproc_per_node=4 -m pytest tests/distributed/test_e2e_training.py -v

Tests:
  1. Train 50 steps under FSDP → loss should decrease
  2. Save checkpoint → load into fresh model → loss matches
  3. Gradient accumulation: N micro-steps matches 1 large batch
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed._composable.fsdp import MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh

from kempnerforge.config.schema import ModelConfig, OptimizerConfig
from kempnerforge.distributed.parallel import apply_fsdp2
from kempnerforge.distributed.setup import get_world_info
from kempnerforge.distributed.utils import clip_grad_norm_
from kempnerforge.model.transformer import Transformer
from kempnerforge.training.grad import maybe_no_sync
from kempnerforge.training.optimizer import build_optimizer

# Skip if not under torchrun
pytestmark = pytest.mark.skipif(
    "RANK" not in os.environ,
    reason="Requires torchrun launcher (RANK not set)",
)

SMALL_CONFIG = ModelConfig(dim=128, n_layers=2, n_heads=2, vocab_size=256, max_seq_len=64)

# Shared filesystem temp directory (visible to all nodes)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TEST_TMP = _PROJECT_ROOT / ".test_tmp"


def _make_fsdp_model(config, device, world_size, mp_policy=None):
    """Build a model with FSDP wrapping."""
    model = Transformer(config).to(device)
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp_shard",))
    apply_fsdp2(model, mesh, mp_policy=mp_policy)
    return model


class TestE2ETraining:
    @pytest.mark.slow
    def test_fsdp_training_loss_decreases(self):
        """Train 50 steps under FSDP on fixed data → loss should decrease."""
        rank, local_rank, world_size = get_world_info()
        device = torch.device(f"cuda:{local_rank}")
        torch.manual_seed(42 + rank)

        model = _make_fsdp_model(SMALL_CONFIG, device, world_size)
        optimizer = build_optimizer(model, OptimizerConfig(lr=1e-3, fused=False))

        # Fixed data across ranks (same seed → same data)
        torch.manual_seed(42)
        tokens = torch.randint(0, 256, (4, 32), device=device)
        labels = tokens.clone()

        losses = []
        for _ in range(50):
            logits = model(tokens)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            clip_grad_norm_(model, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss did not decrease under FSDP: {losses[0]:.4f} → {losses[-1]:.4f}"
        )

    @pytest.mark.slow
    def test_checkpoint_resume_under_fsdp(self):
        """Save checkpoint under FSDP → load into fresh model → loss matches."""
        from torch.distributed.checkpoint.state_dict import (
            get_model_state_dict,
            set_model_state_dict,
        )

        rank, local_rank, world_size = get_world_info()
        device = torch.device(f"cuda:{local_rank}")

        # Use float32 to get exact loss comparison (bf16 has too little precision)
        fp32_policy = MixedPrecisionPolicy()

        torch.manual_seed(42)
        model = _make_fsdp_model(SMALL_CONFIG, device, world_size, mp_policy=fp32_policy)
        optimizer = build_optimizer(model, OptimizerConfig(lr=1e-3, fused=False))

        # Fixed data
        torch.manual_seed(42)
        tokens = torch.randint(0, 256, (4, 32), device=device)
        labels = tokens.clone()

        # Train 20 steps
        for _ in range(20):
            logits = model(tokens)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Compute loss with the FINAL weights (training loss is pre-step)
        with torch.no_grad():
            logits = model(tokens)
            loss_before_save = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            ).item()

        # Save via DCP — all ranks must use the same directory
        import tempfile

        import torch.distributed.checkpoint as dcp

        if rank == 0:
            _TEST_TMP.mkdir(exist_ok=True)
        tmpdir = tempfile.mkdtemp(dir=str(_TEST_TMP)) if rank == 0 else ""
        tmpdir_list = [tmpdir]
        dist.broadcast_object_list(tmpdir_list, src=0)
        tmpdir = tmpdir_list[0]

        try:
            model_state = get_model_state_dict(model)
            dcp.save({"model": model_state}, checkpoint_id=tmpdir)
            dist.barrier()

            # Load into fresh model
            torch.manual_seed(42)
            model2 = _make_fsdp_model(SMALL_CONFIG, device, world_size, mp_policy=fp32_policy)
            model_state2 = get_model_state_dict(model2)
            dcp.load({"model": model_state2}, checkpoint_id=tmpdir)
            set_model_state_dict(model2, model_state2)
            dist.barrier()

            # Compute loss on same data — should match
            with torch.no_grad():
                logits2 = model2(tokens)
                loss2 = F.cross_entropy(logits2.view(-1, logits2.size(-1)), labels.view(-1))

            assert abs(loss_before_save - loss2.item()) < 1e-4, (
                f"Loss mismatch: before_save={loss_before_save:.6f}, after_load={loss2.item():.6f}"
            )
        finally:
            dist.barrier()
            if rank == 0:
                shutil.rmtree(tmpdir, ignore_errors=True)

    def test_gradient_accumulation_correctness(self):
        """Gradient accumulation over N micro-steps should approximate 1 large batch."""
        rank, local_rank, world_size = get_world_info()
        device = torch.device(f"cuda:{local_rank}")

        grad_accum_steps = 4
        micro_batch = 2
        large_batch = micro_batch * grad_accum_steps

        # Same data
        torch.manual_seed(42)
        all_tokens = torch.randint(0, 256, (large_batch, 32), device=device)
        all_labels = all_tokens.clone()

        # --- Method 1: single large batch ---
        torch.manual_seed(99)
        model_large = _make_fsdp_model(SMALL_CONFIG, device, world_size)

        logits = model_large(all_tokens)
        loss_large = F.cross_entropy(logits.view(-1, logits.size(-1)), all_labels.view(-1))
        loss_large.backward()

        # Collect gradients
        large_grad_norm = clip_grad_norm_(model_large, float("inf"))

        # --- Method 2: accumulated micro-batches ---
        torch.manual_seed(99)
        model_accum = _make_fsdp_model(SMALL_CONFIG, device, world_size)

        for i in range(grad_accum_steps):
            start = i * micro_batch
            end = start + micro_batch
            micro_tokens = all_tokens[start:end]
            micro_labels = all_labels[start:end]

            with maybe_no_sync(model_accum, i, grad_accum_steps):
                logits = model_accum(micro_tokens)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), micro_labels.view(-1))
                scaled_loss = loss / grad_accum_steps
                scaled_loss.backward()

        accum_grad_norm = clip_grad_norm_(model_accum, float("inf"))

        # Gradient norms should be close (not exact due to floating point order)
        large_val = (
            large_grad_norm.item() if isinstance(large_grad_norm, torch.Tensor) else large_grad_norm
        )
        accum_val = (
            accum_grad_norm.item() if isinstance(accum_grad_norm, torch.Tensor) else accum_grad_norm
        )

        # Allow 5% relative tolerance
        rel_diff = abs(large_val - accum_val) / max(large_val, 1e-8)
        assert rel_diff < 0.05, (
            f"Gradient norms differ: large={large_val:.6f}, accum={accum_val:.6f}, "
            f"rel_diff={rel_diff:.4f}"
        )
