"""Distributed tests for tensor parallelism.

Run with: torchrun --nproc_per_node=2 -m pytest tests/distributed/test_tp.py -v
(TP degree must divide n_heads and n_kv_heads)
"""

from __future__ import annotations

import os
from dataclasses import replace

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


# Config where n_heads and n_kv_heads are divisible by TP degree (2 or 4)
TP_CONFIG = ModelConfig(
    dim=256, n_layers=2, n_heads=8, n_kv_heads=4, vocab_size=1000, max_seq_len=64
)

# FlexAttention needs seq_len >= FLEX_BLOCK_SIZE, so this variant carries a
# longer max_seq_len than TP_CONFIG. Heads are unchanged, since the point is
# that the BlockMask is head-broadcast (H=None) and therefore indifferent to
# how TP shards the head dimension.
TP_FLEX_CONFIG = ModelConfig(
    dim=256,
    n_layers=2,
    n_heads=8,
    n_kv_heads=4,
    vocab_size=1000,
    max_seq_len=256,
    attention_backend="flex",
)

# TP degree must divide n_heads and n_kv_heads. When world_size > n_kv_heads
# (e.g. 12 GPUs but only 4 kv_heads), use a sub-mesh of valid size.
_VALID_TP_SIZES = [
    s for s in [2, 4, 8] if TP_CONFIG.n_heads % s == 0 and TP_CONFIG.n_kv_heads % s == 0
]


@pytest.fixture
def tp_mesh():
    """Create a TP mesh that is compatible with TP_CONFIG head counts."""
    world_size = dist.get_world_size()
    # Use full world if it divides heads, otherwise pick largest valid sub-mesh
    if TP_CONFIG.n_heads % world_size == 0 and TP_CONFIG.n_kv_heads % world_size == 0:
        tp_size = world_size
    else:
        candidates = [s for s in _VALID_TP_SIZES if s <= world_size]
        if not candidates:
            pytest.skip(f"No valid TP degree for world_size={world_size}")
        tp_size = max(candidates)
    # Create a 2D mesh: (dp, tp) so only tp_size ranks participate in TP
    dp_size = world_size // tp_size
    mesh = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))
    return mesh["tp"]


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
        mesh_2d = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp_shard", "tp"))

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


class TestTensorParallelFlexAttention:
    """Packed FlexAttention under tensor parallelism.

    TP shards the head dimension, so each rank's attention sees ``n_heads / tp``
    query heads and ``n_kv_heads / tp`` key/value heads. The BlockMask is built
    with ``H=None`` and therefore broadcasts over whatever local head count a
    rank holds; ``n_rep`` is computed from the global counts and is invariant
    under that sharding. These tests are what makes that argument checkable
    rather than merely plausible.
    """

    SEQ = 256

    def _batch(self):
        torch.manual_seed(0)
        tokens = torch.randint(0, 1000, (2, self.SEQ), device="cuda")
        half = self.SEQ // 2
        row = [0] * half + [1] * (self.SEQ - half)
        return tokens, torch.tensor([row] * 2, device="cuda"), half

    def _model(self, backend, mesh):
        torch.manual_seed(42)
        config = replace(TP_FLEX_CONFIG, attention_backend=backend)
        model = Transformer(config).cuda()
        apply_tensor_parallel(model, mesh)
        return model.eval()

    def test_flex_matches_sdpa_under_tp(self, tp_mesh):
        """Same weights, same packed batch: flex reproduces the dense-mask path."""
        tokens, doc_ids, _ = self._batch()
        with torch.no_grad():
            expected = self._model("sdpa", tp_mesh)(tokens, doc_ids=doc_ids)
            actual = self._model("flex", tp_mesh)(tokens, doc_ids=doc_ids)
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)

    def test_flex_isolates_documents_under_tp(self, tp_mesh):
        """Sharding heads must not let one document see another."""
        tokens, doc_ids, boundary = self._batch()
        model = self._model("flex", tp_mesh)
        perturbed = tokens.clone()
        perturbed[:, :boundary] = (perturbed[:, :boundary] + 1) % 1000
        with torch.no_grad():
            base = model(tokens, doc_ids=doc_ids)
            moved = model(perturbed, doc_ids=doc_ids)
        torch.testing.assert_close(base[:, boundary:], moved[:, boundary:], rtol=0, atol=0)
        assert not torch.allclose(base[:, :boundary], moved[:, :boundary])

    def test_flex_output_matches_across_ranks(self, tp_mesh):
        """Every TP rank must agree after the row-parallel all-reduce."""
        tokens, doc_ids, _ = self._batch()
        with torch.no_grad():
            out = self._model("flex", tp_mesh)(tokens, doc_ids=doc_ids)
        gathered = [torch.zeros_like(out) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, out)
        for other in gathered[1:]:
            torch.testing.assert_close(gathered[0], other, rtol=1e-3, atol=1e-3)

    def test_flex_backward_under_tp(self, tp_mesh):
        tokens, doc_ids, _ = self._batch()
        model = self._model("flex", tp_mesh)
        model.train()
        model(tokens, doc_ids=doc_ids).sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"no gradient for {name}"
            assert torch.isfinite(
                param.grad.to_local() if hasattr(param.grad, "to_local") else param.grad
            ).all(), name


class TestTensorParallelMatchesSingleGPU:
    """Sharding must not change the answer, only where it is computed.

    The other TP tests compare backends against each other under TP, or check
    that ranks agree with one another. Neither would catch sharding that is
    wrong in the same way everywhere -- all ranks can agree on the wrong answer.
    This compares against an unsharded model built from identical weights.
    """

    SEQ = 256

    def _inputs(self, device):
        torch.manual_seed(0)
        tokens = torch.randint(0, 1000, (2, self.SEQ), device=device)
        half = self.SEQ // 2
        row = [0] * half + [1] * (self.SEQ - half)
        return tokens, torch.tensor([row] * 2, device=device), half

    @pytest.mark.parametrize("backend", ["sdpa", "flex"])
    @pytest.mark.parametrize("packed", [False, True], ids=["unpacked", "packed"])
    def test_matches_unsharded_reference(self, tp_mesh, backend, packed):
        config = replace(TP_FLEX_CONFIG, attention_backend=backend)
        device = torch.device("cuda")
        tokens, doc_ids, _ = self._inputs(device)
        kwargs = {"doc_ids": doc_ids} if packed else {}

        # Same seed on both, so any difference is sharding rather than init.
        torch.manual_seed(42)
        reference = Transformer(config).cuda().eval()
        torch.manual_seed(42)
        sharded = Transformer(config).cuda()
        apply_tensor_parallel(sharded, tp_mesh)
        sharded.eval()

        with torch.no_grad():
            expected = reference(tokens, **kwargs)
            actual = sharded(tokens, **kwargs)

        # TP changes reduction order in the row-parallel all-reduce, so this is
        # a numerical-agreement bound rather than bit-equality.
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
