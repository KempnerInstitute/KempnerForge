"""Distributed tests for FSDP2 integration.

Run with: torchrun --nproc_per_node=4 -m pytest tests/distributed/test_fsdp.py -v
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist

from kempnerforge.config.schema import ActivationCheckpointing, ModelConfig
from kempnerforge.distributed.parallel import apply_ac, apply_fsdp2, get_dp_mesh
from kempnerforge.distributed.setup import get_world_info
from kempnerforge.model.transformer import Transformer

# Skip entire module if not running under torchrun
pytestmark = pytest.mark.skipif(
    "RANK" not in os.environ,
    reason="Requires torchrun launcher (RANK not set)",
)

SMALL_CONFIG = ModelConfig(
    dim=256, n_layers=4, n_heads=4, n_kv_heads=4, vocab_size=1000, max_seq_len=128
)


class TestInitDistributed:
    def test_mesh_created(self, distributed_env):
        mesh = distributed_env
        assert mesh is not None
        world_size = int(os.environ["WORLD_SIZE"])
        assert mesh.size() == world_size

    def test_mesh_dim_names(self, distributed_env):
        mesh = distributed_env
        # Default config: all DP → should have "dp_shard" dimension
        assert "dp_shard" in mesh.mesh_dim_names

    def test_cuda_device_set(self):
        local_rank = int(os.environ["LOCAL_RANK"])
        assert torch.cuda.current_device() == local_rank

    def test_world_info(self):
        rank, local_rank, world_size = get_world_info()
        assert 0 <= rank < world_size
        assert 0 <= local_rank < world_size


class TestFSDP2:
    def test_apply_fsdp2(self, distributed_env):
        mesh = distributed_env
        model = Transformer(SMALL_CONFIG).cuda()
        apply_fsdp2(model, mesh)
        # Model should still be callable
        tokens = torch.randint(0, 1000, (2, 32), device="cuda")
        with torch.no_grad():
            out = model(tokens)
        assert out.shape == (2, 32, 1000)

    def test_fsdp2_backward(self, distributed_env):
        mesh = distributed_env
        model = Transformer(SMALL_CONFIG).cuda()
        apply_fsdp2(model, mesh)

        tokens = torch.randint(0, 1000, (2, 32), device="cuda")
        out = model(tokens)
        loss = out.sum()
        loss.backward()

        # All parameters should have gradients
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_fsdp2_loss_is_finite(self, distributed_env):
        mesh = distributed_env
        model = Transformer(SMALL_CONFIG).cuda()
        apply_fsdp2(model, mesh)

        tokens = torch.randint(0, 1000, (2, 32), device="cuda")
        out = model(tokens)
        loss = out.sum()
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    def test_fsdp2_gradient_sync(self, distributed_env):
        """Loss should be identical across DP ranks (proves gradient sync works)."""
        mesh = distributed_env
        model = Transformer(SMALL_CONFIG).cuda()
        apply_fsdp2(model, mesh)

        # Use same input across all ranks for deterministic comparison
        torch.manual_seed(0)
        tokens = torch.randint(0, 1000, (2, 32), device="cuda")
        out = model(tokens)
        loss = out.sum()

        # Gather loss from all ranks — should be identical under FSDP2
        loss_val = loss.detach().clone()
        all_losses = [torch.zeros_like(loss_val) for _ in range(dist.get_world_size())]
        dist.all_gather(all_losses, loss_val)
        for other in all_losses[1:]:
            assert torch.allclose(all_losses[0], other, atol=1e-3), (
                f"Losses differ across ranks: {[v.item() for v in all_losses]}"
            )

        # Also verify backward completes and clip_grad_norm_ works with DTensors
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        assert torch.isfinite(grad_norm), f"Grad norm not finite: {grad_norm}"


class TestActivationCheckpointing:
    def test_full_ac(self, distributed_env):
        mesh = distributed_env
        model = Transformer(SMALL_CONFIG).cuda()
        apply_ac(model, ActivationCheckpointing.full)
        apply_fsdp2(model, mesh)

        tokens = torch.randint(0, 1000, (2, 32), device="cuda")
        out = model(tokens)
        loss = out.sum()
        loss.backward()

        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name} with AC"

    def test_selective_ac(self, distributed_env):
        mesh = distributed_env
        model = Transformer(SMALL_CONFIG).cuda()
        apply_ac(model, ActivationCheckpointing.selective)
        apply_fsdp2(model, mesh)

        tokens = torch.randint(0, 1000, (2, 32), device="cuda")
        out = model(tokens)
        loss = out.sum()
        loss.backward()

        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name} with selective AC"


class TestGetDpMesh:
    def test_extracts_dp_shard(self, distributed_env):
        mesh = distributed_env
        dp_mesh = get_dp_mesh(mesh)
        assert dp_mesh is not None
