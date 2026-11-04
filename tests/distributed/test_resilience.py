"""Distributed tests for resilience modules.

Run with: torchrun --nproc_per_node=4 -m pytest tests/distributed/test_resilience.py -v
"""

from __future__ import annotations

import os

import pytest
import torch

from kempnerforge.config.schema import ModelConfig
from kempnerforge.distributed.parallel import apply_fsdp2
from kempnerforge.distributed.setup import get_world_info
from kempnerforge.model.transformer import Transformer
from kempnerforge.resilience.health import NaNDetector, check_gpu_health, check_nccl_health

pytestmark = pytest.mark.skipif(
    "RANK" not in os.environ,
    reason="Requires torchrun launcher (RANK not set)",
)

SMALL_CONFIG = ModelConfig(dim=128, n_layers=2, n_heads=2, vocab_size=256, max_seq_len=64)


class TestNCCLHealth:
    def test_nccl_health_check_passes(self):
        """All ranks should agree that NCCL is healthy."""
        result = check_nccl_health()
        assert result is True

    def test_gpu_health_per_rank(self):
        """Each rank should see a healthy GPU."""
        _, local_rank, _ = get_world_info()
        health = check_gpu_health(device=local_rank)
        assert health["cuda_available"] is True
        assert health["compute_ok"] is True
        assert health["memory_ok"] is True


class TestNaNDetectorDistributed:
    def test_nan_detection_with_fsdp_model(self):
        """NaN gradient check works on FSDP-sharded model."""
        rank, local_rank, world_size = get_world_info()
        device = torch.device(f"cuda:{local_rank}")

        model = Transformer(SMALL_CONFIG).to(device)

        from torch.distributed.device_mesh import init_device_mesh

        device_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp_shard",))
        apply_fsdp2(model, device_mesh)

        tokens = torch.randint(0, 256, (2, 32), device=device)
        logits = model(tokens)
        loss = logits.sum()
        loss.backward()

        det = NaNDetector(action="warn")
        grads_ok = det.check_gradients(model, step=1)
        assert grads_ok is True

        loss_ok = det.check_loss(loss.item(), step=1)
        assert loss_ok is True

    def test_nan_detected_across_ranks(self):
        """Inject NaN on one rank and verify detection."""
        rank, local_rank, _ = get_world_info()
        device = torch.device(f"cuda:{local_rank}")

        model = torch.nn.Linear(16, 16).to(device)
        x = torch.randn(4, 16, device=device)
        loss = model(x).sum()
        loss.backward()

        if rank == 0:
            model.weight.grad[0, 0] = float("nan")

        det = NaNDetector(action="warn")
        result = det.check_gradients(model, step=1)

        if rank == 0:
            assert result is False
        else:
            assert result is True

    def test_bad_grad_norm_on_one_rank_skips_on_every_rank(self):
        """A NaN gradient norm on rank 0 alone, with finite losses, fails the check everywhere."""
        rank, _, _ = get_world_info()
        det = NaNDetector(action="warn")
        grad_norm = float("nan") if rank == 0 else 1.0
        assert det.check_loss(0.0, step=1, grad_norm=grad_norm) is False
        assert det.state.consecutive_nans == 1
        assert det.check_loss(0.0, step=2, grad_norm=1.0) is True
        assert det.state.consecutive_nans == 0

    def test_loss_nan_detection_all_ranks(self):
        """All ranks detect NaN loss independently."""
        det = NaNDetector(action="warn")

        result = det.check_loss(float("nan"), step=1)
        assert result is False
        assert det.state.consecutive_nans == 1

        result = det.check_loss(2.5, step=2)
        assert result is True
        assert det.state.consecutive_nans == 0
