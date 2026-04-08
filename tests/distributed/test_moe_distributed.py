"""Distributed tests for MoE under FSDP2 and TP.

Run with: torchrun --nproc_per_node=4 -m pytest tests/distributed/test_moe_distributed.py -v
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.device_mesh import init_device_mesh

from kempnerforge.config.schema import ModelConfig
from kempnerforge.distributed.parallel import apply_fsdp2
from kempnerforge.distributed.tensor_parallel import apply_tensor_parallel
from kempnerforge.model.moe import MoEMLP
from kempnerforge.model.transformer import Transformer

pytestmark = pytest.mark.skipif(
    "RANK" not in os.environ,
    reason="Requires torchrun launcher (RANK not set)",
)

# n_heads=8, n_kv_heads=4 divisible by TP=2 and TP=4
_BASE = dict(dim=256, n_layers=4, n_heads=8, n_kv_heads=4, vocab_size=1000, max_seq_len=64)
MOE_CONFIG = ModelConfig(**_BASE, num_experts=4, moe_top_k=2)
MOE_FREQ_CONFIG = ModelConfig(**_BASE, num_experts=4, moe_top_k=2, moe_frequency=2)
DENSE_CONFIG = ModelConfig(**_BASE)


class TestMoEFSDP:
    def test_moe_fsdp_forward_backward(self, distributed_env):
        """MoE model + FSDP2 on multiple GPUs: forward + backward succeeds."""
        mesh = distributed_env
        model = Transformer(MOE_CONFIG).cuda()
        apply_fsdp2(model, mesh)

        tokens = torch.randint(0, 1000, (2, 32), device="cuda")
        out = model(tokens)
        loss = out.sum()
        assert torch.isfinite(loss), f"Loss not finite: {loss.item()}"
        loss.backward()

        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_moe_fsdp_gradient_sync(self, distributed_env):
        """Loss should be identical across DP ranks (proves gradient sync works for experts)."""
        mesh = distributed_env
        model = Transformer(MOE_CONFIG).cuda()
        apply_fsdp2(model, mesh)

        torch.manual_seed(0)
        tokens = torch.randint(0, 1000, (2, 32), device="cuda")
        out = model(tokens)
        loss = out.sum()

        loss_val = loss.detach().clone()
        all_losses = [torch.zeros_like(loss_val) for _ in range(dist.get_world_size())]
        dist.all_gather(all_losses, loss_val)
        for other in all_losses[1:]:
            assert torch.allclose(all_losses[0], other, atol=1e-3), (
                f"MoE losses differ across ranks: {[v.item() for v in all_losses]}"
            )

    def test_moe_fsdp_aux_loss_finite(self, distributed_env):
        """Aux loss should be finite under FSDP."""
        mesh = distributed_env
        model = Transformer(MOE_CONFIG).cuda()
        apply_fsdp2(model, mesh)

        tokens = torch.randint(0, 1000, (2, 32), device="cuda")
        with torch.no_grad():
            model(tokens)

        aux_loss = model.get_moe_aux_loss()
        assert torch.isfinite(aux_loss), f"Aux loss not finite: {aux_loss.item()}"
        assert aux_loss.item() > 0

    def test_moe_frequency_fsdp(self, distributed_env):
        """MoE with alternating dense/MoE layers works under FSDP."""
        mesh = distributed_env
        model = Transformer(MOE_FREQ_CONFIG).cuda()
        apply_fsdp2(model, mesh)

        tokens = torch.randint(0, 1000, (2, 32), device="cuda")
        out = model(tokens)
        loss = out.sum()
        assert torch.isfinite(loss)
        loss.backward()


class TestMoETP:
    @pytest.fixture
    def tp_mesh(self):
        """Create a TP-only mesh compatible with MOE_CONFIG head counts."""
        world_size = dist.get_world_size()
        valid_tp = [
            s
            for s in [2, 4, 8]
            if MOE_CONFIG.n_heads % s == 0 and MOE_CONFIG.n_kv_heads % s == 0
        ]
        if world_size in valid_tp or (
            MOE_CONFIG.n_heads % world_size == 0 and MOE_CONFIG.n_kv_heads % world_size == 0
        ):
            tp_size = world_size
        else:
            candidates = [s for s in valid_tp if s <= world_size]
            if not candidates:
                pytest.skip(f"No valid TP degree for world_size={world_size}")
            tp_size = max(candidates)
        dp_size = world_size // tp_size
        mesh = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))
        return mesh

    def test_moe_tp_forward_backward(self, tp_mesh):
        """MoE model + TP: forward + backward succeeds with experts replicated."""
        model = Transformer(MOE_CONFIG).cuda()
        apply_tensor_parallel(model, tp_mesh)

        tokens = torch.randint(0, 1000, (1, 32), device="cuda")
        out = model(tokens)
        loss = out.sum()
        assert torch.isfinite(torch.tensor(loss.item()))
        loss.backward()

    def test_moe_tp_output_matches_across_ranks(self, tp_mesh):
        """All TP ranks should produce identical output."""
        torch.manual_seed(42)
        model = Transformer(MOE_CONFIG).cuda()
        apply_tensor_parallel(model, tp_mesh)

        torch.manual_seed(0)
        tokens = torch.randint(0, 1000, (1, 16), device="cuda")
        with torch.no_grad():
            out = model(tokens)

        all_outs = [torch.zeros_like(out) for _ in range(dist.get_world_size())]
        dist.all_gather(all_outs, out)
        for rank_out in all_outs[1:]:
            assert torch.allclose(all_outs[0], rank_out, atol=1e-3), (
                f"MoE TP outputs differ: max diff = {(all_outs[0] - rank_out).abs().max().item()}"
            )

    def test_moe_blocks_have_no_mlp_tp(self, tp_mesh):
        """MoE blocks should have replicated experts (no TP sharding on mlp.*)."""
        model = Transformer(MOE_CONFIG).cuda()
        apply_tensor_parallel(model, tp_mesh)

        for layer in model.layers.values():
            if isinstance(layer.mlp, MoEMLP):
                # Expert params should be plain tensors, not DTensors
                for p in layer.mlp.experts.parameters():
                    from torch.distributed.tensor import DTensor

                    assert not isinstance(p, DTensor), "Expert weights should not be DTensors"

    def test_moe_frequency_tp(self, tp_mesh):
        """Alternating dense/MoE with TP: dense blocks get MLP TP, MoE blocks don't."""
        model = Transformer(MOE_FREQ_CONFIG).cuda()
        apply_tensor_parallel(model, tp_mesh)

        tokens = torch.randint(0, 1000, (1, 32), device="cuda")
        out = model(tokens)
        loss = out.sum()
        assert torch.isfinite(torch.tensor(loss.item()))
        loss.backward()


class TestMoETPPlusFSDP:
    @pytest.mark.skipif(
        int(os.environ.get("WORLD_SIZE", "1")) < 4,
        reason="TP+FSDP requires >= 4 GPUs (dp=2, tp=2)",
    )
    def test_moe_tp_plus_fsdp2(self):
        """MoE with TP + FSDP2 on a 2D mesh (dp=2, tp=2)."""
        world_size = dist.get_world_size()
        tp_size = 2
        dp_size = world_size // tp_size
        mesh_2d = init_device_mesh(
            "cuda", (dp_size, tp_size), mesh_dim_names=("dp_shard", "tp")
        )

        model = Transformer(MOE_CONFIG).cuda()
        apply_tensor_parallel(model, mesh_2d)

        dp_mesh = mesh_2d["dp_shard"]
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for layer in model.layers.values():
            fully_shard(layer, mesh=dp_mesh, mp_policy=mp_policy)
        fully_shard(model, mesh=dp_mesh, mp_policy=mp_policy)

        tokens = torch.randint(0, 1000, (1, 16), device="cuda")
        out = model(tokens)
        loss = out.sum()
        assert torch.isfinite(torch.tensor(loss.item()))
        loss.backward()

    @pytest.mark.skipif(
        int(os.environ.get("WORLD_SIZE", "1")) < 4,
        reason="TP+FSDP requires >= 4 GPUs (dp=2, tp=2)",
    )
    def test_moe_frequency_tp_plus_fsdp2(self):
        """Alternating dense/MoE with TP + FSDP2."""
        world_size = dist.get_world_size()
        tp_size = 2
        dp_size = world_size // tp_size
        mesh_2d = init_device_mesh(
            "cuda", (dp_size, tp_size), mesh_dim_names=("dp_shard", "tp")
        )

        model = Transformer(MOE_FREQ_CONFIG).cuda()
        apply_tensor_parallel(model, mesh_2d)

        dp_mesh = mesh_2d["dp_shard"]
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for layer in model.layers.values():
            fully_shard(layer, mesh=dp_mesh, mp_policy=mp_policy)
        fully_shard(model, mesh=dp_mesh, mp_policy=mp_policy)

        tokens = torch.randint(0, 1000, (1, 16), device="cuda")
        out = model(tokens)
        loss = out.sum()
        assert torch.isfinite(torch.tensor(loss.item()))
        loss.backward()
