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


class TestFSDP2FlexAttention:
    """Packed FlexAttention under FSDP2.

    The interaction worth checking is ``default_mp_policy``'s
    ``cast_forward_inputs=True``: FSDP2 walks the forward inputs at each wrapped
    module boundary and casts them to bf16, and ``doc_ids`` is an int64 index
    tensor that could plausibly be mangled on the way in -- a mangled mask does
    not raise, it silently stops isolating documents. The ``BlockMask`` itself
    is built inside ``Transformer.forward``, below this boundary, so it never
    crosses the cast.
    """

    SEQ = 256
    CONFIG = ModelConfig(
        dim=256,
        n_layers=4,
        n_heads=4,
        n_kv_heads=4,
        vocab_size=1000,
        max_seq_len=256,
        attention_backend="flex",
    )

    def _batch(self):
        torch.manual_seed(0)
        tokens = torch.randint(0, 1000, (2, self.SEQ), device="cuda")
        half = self.SEQ // 2
        row = [0] * half + [1] * (self.SEQ - half)
        return tokens, torch.tensor([row] * 2, device="cuda"), half

    def _model(self, mesh, backend="flex", ac_mode=None):
        from dataclasses import replace

        torch.manual_seed(42)
        model = Transformer(replace(self.CONFIG, attention_backend=backend)).cuda()
        if ac_mode is not None:
            apply_ac(model, ac_mode)
        apply_fsdp2(model, mesh)
        return model

    def test_flex_matches_sdpa_under_fsdp(self, distributed_env):
        """Sharding parameters must not change which tokens attend to which.

        Tolerance is bf16-sized because ``default_mp_policy`` computes in bf16;
        the exactness claim lives in the isolation test below, which is
        dtype-independent.
        """
        tokens, doc_ids, _ = self._batch()
        with torch.no_grad():
            expected = self._model(distributed_env, backend="sdpa")(tokens, doc_ids=doc_ids)
            actual = self._model(distributed_env, backend="flex")(tokens, doc_ids=doc_ids)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    def test_doc_ids_survive_the_forward_input_cast(self, distributed_env):
        """int64 doc_ids must reach attention unmangled by the bf16 input cast.

        Exact equality, not a tolerance: document 1 may not move *at all* when
        document 0 changes. If the cast corrupted doc_ids the mask would widen
        silently and this is the only thing that would notice.
        """
        tokens, doc_ids, boundary = self._batch()
        model = self._model(distributed_env)
        perturbed = tokens.clone()
        perturbed[:, :boundary] = (perturbed[:, :boundary] + 1) % 1000
        with torch.no_grad():
            base = model(tokens, doc_ids=doc_ids)
            moved = model(perturbed, doc_ids=doc_ids)
        torch.testing.assert_close(base[:, boundary:], moved[:, boundary:], rtol=0, atol=0)
        assert not torch.allclose(base[:, :boundary], moved[:, :boundary])

    def test_flex_backward_under_fsdp(self, distributed_env):
        tokens, doc_ids, _ = self._batch()
        model = self._model(distributed_env)
        model(tokens, doc_ids=doc_ids).sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"no gradient for {name}"
            grad = param.grad.to_local() if hasattr(param.grad, "to_local") else param.grad
            assert torch.isfinite(grad).all(), f"non-finite gradient for {name}"

    @pytest.mark.parametrize(
        "ac_mode",
        [ActivationCheckpointing.selective, ActivationCheckpointing.full],
        ids=["ac_selective", "ac_full"],
    )
    def test_flex_with_activation_checkpointing_under_fsdp(self, distributed_env, ac_mode):
        """AC + FSDP2 + flex, the full production stack for a packed run."""
        tokens, doc_ids, boundary = self._batch()
        model = self._model(distributed_env, ac_mode=ac_mode)

        perturbed = tokens.clone()
        perturbed[:, :boundary] = (perturbed[:, :boundary] + 1) % 1000
        with torch.no_grad():
            base = model(tokens, doc_ids=doc_ids)
            moved = model(perturbed, doc_ids=doc_ids)
        torch.testing.assert_close(base[:, boundary:], moved[:, boundary:], rtol=0, atol=0)

        model(tokens, doc_ids=doc_ids).sum().backward()
        for name, param in model.named_parameters():
            grad = param.grad.to_local() if hasattr(param.grad, "to_local") else param.grad
            assert grad is not None and torch.isfinite(grad).all(), name


class TestFSDP2MatchesSingleGPU:
    """Parameter sharding must not change the answer.

    As with TP, the other FSDP tests compare backends against each other under
    FSDP; none check the sharded model against an unsharded one.
    """

    SEQ = 256
    CONFIG = ModelConfig(
        dim=256, n_layers=4, n_heads=4, n_kv_heads=4, vocab_size=1000, max_seq_len=256
    )

    def _inputs(self):
        torch.manual_seed(0)
        tokens = torch.randint(0, 1000, (2, self.SEQ), device="cuda")
        half = self.SEQ // 2
        row = [0] * half + [1] * (self.SEQ - half)
        return tokens, torch.tensor([row] * 2, device="cuda")

    @pytest.mark.parametrize("backend", ["sdpa", "flex"])
    @pytest.mark.parametrize("packed", [False, True], ids=["unpacked", "packed"])
    def test_matches_unsharded_reference(self, distributed_env, backend, packed):
        from dataclasses import replace

        config = replace(self.CONFIG, attention_backend=backend)
        tokens, doc_ids = self._inputs()
        kwargs = {"doc_ids": doc_ids} if packed else {}

        torch.manual_seed(42)
        reference = Transformer(config).cuda().to(torch.bfloat16).eval()
        torch.manual_seed(42)
        sharded = Transformer(config).cuda()
        apply_fsdp2(sharded, distributed_env)
        sharded.eval()

        with torch.no_grad():
            expected = reference(tokens, **kwargs)
            actual = sharded(tokens, **kwargs)

        # default_mp_policy computes in bf16, so this is a bf16-sized bound.
        torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)
