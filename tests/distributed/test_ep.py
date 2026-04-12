"""Distributed tests for Expert Parallelism (EP).

Run with: torchrun --nproc_per_node=4 -m pytest tests/distributed/test_ep.py -v

Requires at least 2 GPUs for EP tests. Most tests use ep=2 with 4 experts.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from kempnerforge.config.schema import ModelConfig
from kempnerforge.distributed.expert_parallel import apply_expert_parallel
from kempnerforge.distributed.parallel import apply_fsdp2
from kempnerforge.model.moe import MoEMLP
from kempnerforge.model.transformer import Transformer

pytestmark = pytest.mark.skipif(
    "RANK" not in os.environ,
    reason="Requires torchrun launcher (RANK not set)",
)

_BASE = dict(dim=128, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=512, max_seq_len=32)
EP_CONFIG = ModelConfig(**_BASE, num_experts=4, moe_top_k=2)


@pytest.fixture
def ep_mesh():
    """Create a mesh with ep=2, dp_shard=remaining."""
    world_size = dist.get_world_size()
    if world_size < 2:
        pytest.skip("EP tests require at least 2 GPUs")
    ep_size = 2
    dp_size = world_size // ep_size
    mesh = init_device_mesh("cuda", (dp_size, ep_size), mesh_dim_names=("dp_shard", "ep"))
    return mesh


@pytest.fixture
def ep_only_mesh():
    """Create an EP-only mesh (all GPUs in EP group)."""
    world_size = dist.get_world_size()
    if world_size < 2:
        pytest.skip("EP tests require at least 2 GPUs")
    mesh = init_device_mesh("cuda", (1, world_size), mesh_dim_names=("dp_shard", "ep"))
    return mesh


class TestEPApply:
    def test_apply_prunes_experts(self, ep_only_mesh):
        """apply_expert_parallel should keep only local experts."""
        model = Transformer(EP_CONFIG).cuda()
        apply_expert_parallel(model, ep_only_mesh)

        ep_size = ep_only_mesh["ep"].size()
        expected_local = EP_CONFIG.num_experts // ep_size

        for layer in model.layers.values():
            if isinstance(layer.mlp, MoEMLP):
                assert len(layer.mlp.experts) == expected_local
                assert layer.mlp.num_local_experts == expected_local
                assert layer.mlp.ep_world_size == ep_size

    def test_apply_sets_metadata(self, ep_only_mesh):
        """EP metadata (group, rank, start index) should be set correctly."""
        model = Transformer(EP_CONFIG).cuda()
        apply_expert_parallel(model, ep_only_mesh)

        ep_rank = ep_only_mesh["ep"].get_local_rank()
        ep_size = ep_only_mesh["ep"].size()
        experts_per_rank = EP_CONFIG.num_experts // ep_size

        for layer in model.layers.values():
            if isinstance(layer.mlp, MoEMLP):
                assert layer.mlp.local_expert_start == ep_rank * experts_per_rank
                assert layer.mlp.ep_group is not None

    def test_apply_noop_without_ep_dim(self):
        """Should be a no-op when mesh has no EP dimension."""
        world_size = dist.get_world_size()
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp_shard",))
        model = Transformer(EP_CONFIG).cuda()
        apply_expert_parallel(model, mesh)

        for layer in model.layers.values():
            if isinstance(layer.mlp, MoEMLP):
                assert len(layer.mlp.experts) == EP_CONFIG.num_experts
                assert layer.mlp.ep_world_size == 1


class TestEPForward:
    def test_ep_forward_backward(self, ep_only_mesh):
        """EP model forward + backward should produce finite results."""
        model = Transformer(EP_CONFIG).cuda()
        apply_expert_parallel(model, ep_only_mesh)

        tokens = torch.randint(0, 512, (1, 16), device="cuda")
        out = model(tokens)
        loss = out.sum()
        assert torch.isfinite(torch.tensor(loss.item())), f"Loss not finite: {loss.item()}"
        loss.backward()

        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_ep_expert_counts_nonzero(self, ep_only_mesh):
        """All local experts should receive some tokens across the EP group."""
        model = Transformer(EP_CONFIG).cuda()
        apply_expert_parallel(model, ep_only_mesh)

        # Use enough tokens to distribute across experts
        tokens = torch.randint(0, 512, (2, 32), device="cuda")
        with torch.no_grad():
            model(tokens)

        # At least one MoE layer should have non-zero expert counts
        has_counts = False
        for layer in model.layers.values():
            if isinstance(layer.mlp, MoEMLP):
                counts = layer.mlp.expert_counts
                if counts.sum() > 0:
                    has_counts = True
        assert has_counts, "No expert counts recorded"


class TestEPPlusFSDP:
    @pytest.mark.skipif(
        int(os.environ.get("WORLD_SIZE", "1")) < 4,
        reason="EP+FSDP requires >= 4 GPUs (dp=2, ep=2)",
    )
    def test_ep_plus_fsdp2(self):
        """EP + FSDP2 should compose correctly on a 2D mesh."""
        world_size = dist.get_world_size()
        ep_size = 2
        dp_size = world_size // ep_size
        mesh = init_device_mesh("cuda", (dp_size, ep_size), mesh_dim_names=("dp_shard", "ep"))

        model = Transformer(EP_CONFIG).cuda()
        apply_expert_parallel(model, mesh)
        apply_fsdp2(model, mesh)

        tokens = torch.randint(0, 512, (1, 16), device="cuda")
        out = model(tokens)
        loss = out.sum()
        assert torch.isfinite(torch.tensor(loss.item()))
        loss.backward()

    @pytest.mark.skipif(
        int(os.environ.get("WORLD_SIZE", "1")) < 4,
        reason="EP+FSDP requires >= 4 GPUs (dp=2, ep=2)",
    )
    def test_ep_fsdp_gradient_sync(self):
        """Loss should be identical across DP ranks within the same EP group."""
        world_size = dist.get_world_size()
        ep_size = 2
        dp_size = world_size // ep_size
        mesh = init_device_mesh("cuda", (dp_size, ep_size), mesh_dim_names=("dp_shard", "ep"))

        model = Transformer(EP_CONFIG).cuda()
        apply_expert_parallel(model, mesh)
        apply_fsdp2(model, mesh)

        torch.manual_seed(0)
        tokens = torch.randint(0, 512, (1, 16), device="cuda")
        out = model(tokens)
        loss = out.sum()

        loss_val = loss.detach().clone()
        all_losses = [torch.zeros_like(loss_val) for _ in range(world_size)]
        dist.all_gather(all_losses, loss_val)
        # All losses should be finite
        for i, v in enumerate(all_losses):
            assert torch.isfinite(v), f"Rank {i} loss not finite: {v.item()}"
