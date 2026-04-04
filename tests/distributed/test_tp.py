"""Distributed tests for tensor parallelism.

Run with: torchrun --nproc_per_node=2 -m pytest tests/distributed/test_tp.py -v
(TP degree must divide n_heads and n_kv_heads)
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from kempnerforge.config.schema import ModelConfig
from kempnerforge.distributed.tensor_parallel import apply_tensor_parallel
from kempnerforge.model.transformer import Transformer

pytestmark = pytest.mark.skipif(
    "RANK" not in os.environ,
    reason="Requires torchrun launcher (RANK not set)",
)


@pytest.fixture
def tp_mesh():
    """Create a TP mesh from the existing process group."""
    world_size = dist.get_world_size()
    return init_device_mesh("cuda", (world_size,), mesh_dim_names=("tp",))


# Config where n_heads and n_kv_heads are divisible by TP degree (2 or 4)
TP_CONFIG = ModelConfig(
    dim=256, n_layers=2, n_heads=8, n_kv_heads=4, vocab_size=1000, max_seq_len=64
)


class TestTensorParallel:
    def test_apply_tp(self, tp_mesh):
        mesh = tp_mesh
        model = Transformer(TP_CONFIG).cuda()
        apply_tensor_parallel(model, mesh)

        tokens = torch.randint(0, 1000, (1, 32), device="cuda")
        with torch.no_grad():
            out = model(tokens)
        assert out.shape == (1, 32, 1000)

    def test_tp_backward(self, tp_mesh):
        mesh = tp_mesh
        model = Transformer(TP_CONFIG).cuda()
        apply_tensor_parallel(model, mesh)

        tokens = torch.randint(0, 1000, (1, 32), device="cuda")
        out = model(tokens)
        loss = out.sum()
        loss.backward()

        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_tp_output_matches_across_ranks(self, tp_mesh):
        """With TP, all ranks should produce identical output (after all-reduce)."""
        mesh = tp_mesh

        # Same model init seed across ranks
        torch.manual_seed(42)
        model = Transformer(TP_CONFIG).cuda()
        apply_tensor_parallel(model, mesh)

        # Same input across all ranks
        torch.manual_seed(0)
        tokens = torch.randint(0, 1000, (1, 16), device="cuda")
        with torch.no_grad():
            out = model(tokens)

        # Gather outputs from all ranks and compare
        all_outs = [torch.zeros_like(out) for _ in range(dist.get_world_size())]
        dist.all_gather(all_outs, out)
        for rank_out in all_outs[1:]:
            assert torch.allclose(all_outs[0], rank_out, atol=1e-3), (
                f"TP outputs differ: max diff = {(all_outs[0] - rank_out).abs().max().item()}"
            )


class TestTPWithFSDP:
    @pytest.mark.skipif(
        int(os.environ.get("WORLD_SIZE", "1")) < 4,
        reason="TP+FSDP composition requires >= 4 GPUs (dp=2, tp=2)",
    )
    def test_tp_plus_fsdp2(self):
        """TP + FSDP2 composition on a 2D mesh (dp=2, tp=2)."""
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard

        world_size = dist.get_world_size()
        tp_size = 2
        dp_size = world_size // tp_size

        # Create a 2D mesh with separate DP and TP dimensions
        mesh_2d = init_device_mesh(
            "cuda", (dp_size, tp_size), mesh_dim_names=("dp_shard", "tp")
        )

        model = Transformer(TP_CONFIG).cuda()

        # Apply TP on the tp sub-mesh
        apply_tensor_parallel(model, mesh_2d)

        # Apply FSDP2 on the dp sub-mesh
        dp_mesh = mesh_2d["dp_shard"]
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for layer in model.layers.values():
            fully_shard(layer, mesh=dp_mesh, mp_policy=mp_policy)
        fully_shard(model, mesh=dp_mesh, mp_policy=mp_policy)

        tokens = torch.randint(0, 1000, (1, 16), device="cuda")
        out = model(tokens)
        loss = out.sum()
        loss.backward()
        assert torch.isfinite(torch.tensor(loss.item()))
