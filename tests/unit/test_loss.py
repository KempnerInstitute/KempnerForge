"""Unit tests for loss functions."""

from __future__ import annotations

import copy
import functools
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from kempnerforge.config.registry import registry
from kempnerforge.config.schema import TrainConfig
from kempnerforge.training.loss import (
    build_loss_fn,
    chunked_cross_entropy_loss,
    cross_entropy_loss,
    z_loss,
)

ALL_MASKED_LOSSES = {
    "cross_entropy": cross_entropy_loss,
    "chunked_one_chunk": functools.partial(chunked_cross_entropy_loss, chunk_size=64),
    "chunked_many_chunks": functools.partial(chunked_cross_entropy_loss, chunk_size=4),
}

FP32_LOSSES = {
    "cross_entropy": functools.partial(cross_entropy_loss, fp32=True),
    **{
        f"chunked_{size}": functools.partial(chunked_cross_entropy_loss, chunk_size=size, fp32=True)
        for size in (1, 3, 7, 64, 4096)
    },
}


def _fp64_reference(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Loss and logit gradient of ``F.cross_entropy`` on an fp64 copy of the logits."""
    flat = logits.detach().double().reshape(-1, logits.shape[-1]).requires_grad_(True)
    loss = F.cross_entropy(flat, labels.reshape(-1), ignore_index=-100)
    loss.backward()
    assert flat.grad is not None
    return loss.detach(), flat.grad.reshape(logits.shape)


class TestCrossEntropy:
    def test_basic_shape(self):
        logits = torch.randn(2, 8, 100)
        labels = torch.randint(0, 100, (2, 8))
        loss = cross_entropy_loss(logits, labels)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_perfect_prediction(self):
        logits = torch.zeros(1, 4, 10)
        labels = torch.zeros(1, 4, dtype=torch.long)
        logits[:, :, 0] = 100.0
        loss = cross_entropy_loss(logits, labels)
        assert loss.item() < 0.01


class TestChunkedCrossEntropy:
    def test_matches_standard_ce(self):
        """Chunked CE must produce the same result as standard CE."""
        torch.manual_seed(42)
        logits = torch.randn(4, 32, 256)
        labels = torch.randint(0, 256, (4, 32))

        standard = cross_entropy_loss(logits, labels)
        chunked = chunked_cross_entropy_loss(logits, labels, chunk_size=32)

        torch.testing.assert_close(chunked, standard, rtol=1e-4, atol=1e-4)

    def test_small_batch_bypasses_chunking(self):
        """When num_tokens <= chunk_size, should use standard CE path."""
        logits = torch.randn(2, 8, 50)
        labels = torch.randint(0, 50, (2, 8))

        standard = cross_entropy_loss(logits, labels)
        chunked = chunked_cross_entropy_loss(logits, labels, chunk_size=4096)

        torch.testing.assert_close(chunked, standard, rtol=1e-5, atol=1e-5)

    def test_chunked_path_activates(self):
        """Chunking path should produce correct results with various chunk sizes."""
        torch.manual_seed(0)
        logits = torch.randn(8, 16, 512)
        labels = torch.randint(0, 512, (8, 16))

        standard = cross_entropy_loss(logits, labels)
        for cs in [16, 32, 64, 100]:
            chunked = chunked_cross_entropy_loss(logits, labels, chunk_size=cs)
            torch.testing.assert_close(chunked, standard, rtol=1e-4, atol=1e-4)

    def test_uneven_chunks(self):
        """Last chunk may be smaller — should still be correct."""
        torch.manual_seed(1)
        logits = torch.randn(3, 10, 200)
        labels = torch.randint(0, 200, (3, 10))

        standard = cross_entropy_loss(logits, labels)
        chunked = chunked_cross_entropy_loss(logits, labels, chunk_size=7)

        torch.testing.assert_close(chunked, standard, rtol=1e-4, atol=1e-4)

    def test_gradient_flows(self):
        logits = torch.randn(4, 16, 512, requires_grad=True)
        labels = torch.randint(0, 512, (4, 16))
        loss = chunked_cross_entropy_loss(logits, labels, chunk_size=16)
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()

    def test_gradient_matches_standard(self):
        """Gradients from chunked path should match standard CE."""
        torch.manual_seed(42)
        logits_a = torch.randn(4, 16, 256, requires_grad=True)
        logits_b = logits_a.detach().clone().requires_grad_(True)
        labels = torch.randint(0, 256, (4, 16))

        loss_std = cross_entropy_loss(logits_a, labels)
        loss_std.backward()

        loss_chunked = chunked_cross_entropy_loss(logits_b, labels, chunk_size=16)
        loss_chunked.backward()

        torch.testing.assert_close(logits_a.grad, logits_b.grad, rtol=1e-4, atol=1e-4)


class TestFp32CrossEntropy:
    """With ``fp32=True``, bf16 logits get an fp32 loss and gradient, whole or chunked."""

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=[
                    pytest.mark.gpu,
                    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
                ],
            ),
        ],
    )
    @pytest.mark.parametrize("loss_name", sorted(FP32_LOSSES))
    def test_bf16_logits_match_fp64(self, loss_name: str, device: str) -> None:
        torch.manual_seed(0)
        logits = (torch.randn(4, 16, 1000) * 4).bfloat16().to(device).requires_grad_(True)
        labels = torch.randint(0, 1000, (4, 16)).to(device)
        labels[0, :5] = -100
        loss = FP32_LOSSES[loss_name](logits, labels)
        loss.backward()
        ref_loss, ref_grad = _fp64_reference(logits, labels)
        assert loss.dtype == torch.float32
        assert logits.grad is not None and logits.grad.dtype == torch.bfloat16
        torch.testing.assert_close(loss.double(), ref_loss, rtol=1e-6, atol=0.0)
        torch.testing.assert_close(logits.grad.double(), ref_grad, rtol=2**-8, atol=1e-12)

    @pytest.mark.parametrize("loss_name", ["cross_entropy", "chunked_3"])
    def test_fp64_logits_stay_fp64_and_pass_gradcheck(self, loss_name: str) -> None:
        torch.manual_seed(0)
        loss_fn = FP32_LOSSES[loss_name]
        logits = torch.randn(10, 7, dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([0, 1, -100, 3, 4, 5, 6, -100, 2, 1])
        assert loss_fn(logits, labels).dtype == torch.float64
        assert torch.autograd.gradcheck(lambda x: loss_fn(x, labels), (logits,))

    @pytest.mark.parametrize("loss_name", ["cross_entropy", "chunked_7"])
    def test_backward_saves_no_fp32_copy_of_the_logits(self, loss_name: str) -> None:
        logits = torch.randn(64, 300).bfloat16().requires_grad_(True)
        labels = torch.randint(0, 300, (64,))
        saved: list[torch.Tensor] = []

        def pack(t: torch.Tensor) -> torch.Tensor:
            saved.append(t)
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            loss = FP32_LOSSES[loss_name](logits, labels)
        loss.backward()
        assert any(t.data_ptr() == logits.data_ptr() for t in saved)
        assert all(t.data_ptr() == logits.data_ptr() or t.numel() <= 64 for t in saved)

    @pytest.mark.parametrize("loss_name", ["cross_entropy", "chunked_3"])
    def test_ignored_rows_get_zero_gradient_even_when_not_finite(self, loss_name: str) -> None:
        logits = torch.randn(6, 50)
        logits[2] = float("nan")
        logits[4, 0] = float("inf")
        logits.requires_grad_(True)
        labels = torch.tensor([1, 2, -100, 3, -100, 4])
        loss = FP32_LOSSES[loss_name](logits, labels)
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(loss)
        assert torch.equal(logits.grad[[2, 4]], torch.zeros(2, 50))
        assert torch.isfinite(logits.grad).all()

    def test_all_masked_returns_fp32_zero(self) -> None:
        logits = torch.randn(2, 8, 16).bfloat16()
        loss = FP32_LOSSES["cross_entropy"](logits, torch.full((2, 8), -100))
        assert loss.dtype == torch.float32 and loss.item() == 0.0

    @pytest.mark.gpu
    @pytest.mark.parametrize("chunk_size", [None, 1024])
    def test_peak_memory_holds_no_full_fp32_copy(self, chunk_size: int | None) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        tokens, vocab = 8192, 32768
        logits = torch.randn(tokens, vocab, device="cuda", dtype=torch.bfloat16)
        logits.requires_grad_(True)
        labels = torch.randint(0, vocab, (tokens,), device="cuda")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        if chunk_size is None:
            cross_entropy_loss(logits, labels, fp32=True).backward()
        else:
            chunked_cross_entropy_loss(logits, labels, chunk_size=chunk_size, fp32=True).backward()
        torch.cuda.synchronize()
        extra = torch.cuda.max_memory_allocated() - before
        grad_bytes = tokens * vocab * 2
        chunk_fp32_bytes = 1024 * vocab * 4
        assert extra <= grad_bytes + 3 * chunk_fp32_bytes
        assert extra < tokens * vocab * 4


class TestZLoss:
    def test_zero_weight_returns_zero(self):
        logits = torch.randn(2, 8, 100)
        result = z_loss(logits, weight=0.0)
        assert result.item() == 0.0

    def test_small_logits_small_zloss(self):
        """Small logits should produce small z-loss."""
        logits = torch.zeros(2, 8, 100)
        result = z_loss(logits, weight=1e-4)
        assert result.item() < 0.01

    def test_large_logits_large_zloss(self):
        """Large logits should produce proportionally larger z-loss."""
        small = torch.randn(2, 8, 100)
        large = small * 100.0

        z_small = z_loss(small, weight=1e-4)
        z_large = z_loss(large, weight=1e-4)

        assert z_large.item() > z_small.item() * 10

    def test_gradient_flows(self):
        logits = torch.randn(2, 8, 100, requires_grad=True)
        result = z_loss(logits, weight=1e-4)
        result.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()

    def test_positive_for_nonzero_weight(self):
        logits = torch.randn(2, 8, 100)
        result = z_loss(logits, weight=1e-4)
        assert result.item() > 0


class TestBuildLossFn:
    def test_standard_ce(self):
        config = TrainConfig(loss_fn="cross_entropy")
        fn = build_loss_fn(config)
        logits = torch.randn(2, 8, 100)
        labels = torch.randint(0, 100, (2, 8))
        loss = fn(logits, labels)
        torch.testing.assert_close(loss, cross_entropy_loss(logits, labels))

    def test_chunked_ce_binds_chunk_size(self):
        config = TrainConfig(loss_fn="chunked_cross_entropy", ce_chunk_size=16)
        fn = build_loss_fn(config)
        logits = torch.randn(4, 32, 256)
        labels = torch.randint(0, 256, (4, 32))
        expected = chunked_cross_entropy_loss(logits, labels, chunk_size=16)
        torch.testing.assert_close(fn(logits, labels), expected)

    def test_fp32_off_returns_the_loss_unchanged(self) -> None:
        assert build_loss_fn(TrainConfig()) is cross_entropy_loss

    def test_fp32_off_keeps_bf16_logits_in_bf16(self) -> None:
        logits = torch.randn(2, 8, 100).bfloat16()
        loss = build_loss_fn(TrainConfig())(logits, torch.randint(0, 100, (2, 8)))
        assert loss.dtype == torch.bfloat16

    @pytest.mark.parametrize(
        ("loss_fn", "ce_chunk_size", "keywords"),
        [
            ("cross_entropy", 16, {"fp32": True}),
            ("chunked_cross_entropy", 0, {"chunk_size": 4096, "fp32": True}),
            ("chunked_cross_entropy", 16, {"chunk_size": 16, "fp32": True}),
        ],
    )
    def test_fp32_on_binds_fp32(self, loss_fn: str, ce_chunk_size: int, keywords: dict) -> None:
        config = TrainConfig(loss_fn=loss_fn, ce_chunk_size=ce_chunk_size, fp32_cross_entropy=True)
        fn = build_loss_fn(config)
        assert isinstance(fn, functools.partial)
        assert fn.keywords == keywords
        logits = torch.randn(2, 8, 100).bfloat16()
        assert fn(logits, torch.randint(0, 100, (2, 8))).dtype == torch.float32

    def test_fp32_on_rejects_another_loss(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(registry._get_store("loss"), "mse_test", lambda x, y: x)
        config = TrainConfig(loss_fn="mse_test", fp32_cross_entropy=True)
        with pytest.raises(ValueError, match="fp32_cross_entropy applies to"):
            build_loss_fn(config)

    def test_chunked_ce_default_chunk_size(self):
        """ce_chunk_size=0 should default to 4096."""
        config = TrainConfig(loss_fn="chunked_cross_entropy", ce_chunk_size=0)
        fn = build_loss_fn(config)
        logits = torch.randn(2, 8, 50)
        labels = torch.randint(0, 50, (2, 8))
        expected = chunked_cross_entropy_loss(logits, labels, chunk_size=4096)
        torch.testing.assert_close(fn(logits, labels), expected)

    def test_composes_z_loss(self):
        config = TrainConfig(loss_fn="cross_entropy", z_loss_weight=1e-4)
        fn = build_loss_fn(config)
        logits = torch.randn(2, 8, 100)
        labels = torch.randint(0, 100, (2, 8))
        result = fn(logits, labels)
        base = cross_entropy_loss(logits, labels)
        z = z_loss(logits, 1e-4)
        torch.testing.assert_close(result, base + z)

    def test_z_loss_not_added_when_zero(self):
        config = TrainConfig(loss_fn="cross_entropy", z_loss_weight=0.0)
        fn = build_loss_fn(config)
        logits = torch.randn(2, 8, 100)
        labels = torch.randint(0, 100, (2, 8))
        result = fn(logits, labels)
        base = cross_entropy_loss(logits, labels)
        torch.testing.assert_close(result, base)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _rank_with_nothing_supervised_joins(rank: int, world: int, port: int, loss_name: str) -> None:
    """Rank 0's micro-batch is all ``-100``; FSDP2 still reduces every gradient with rank 1."""
    loss_fn = ALL_MASKED_LOSSES[loss_name]
    dist.init_process_group(
        "gloo", rank=rank, world_size=world, init_method=f"tcp://127.0.0.1:{port}"
    )
    try:
        mesh = init_device_mesh("cpu", (world,))
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 16))
        reference = copy.deepcopy(model)
        for layer in model:
            fully_shard(layer, mesh=mesh)
        fully_shard(model, mesh=mesh)

        generator = torch.Generator().manual_seed(1)
        inputs = torch.randn(world, 2, 4, 8, generator=generator)
        labels = torch.randint(0, 16, (world, 2, 4), generator=generator)
        labels[0] = -100
        loss_fn(model(inputs[rank]), labels[rank]).backward()

        sum(loss_fn(reference(inputs[r]), labels[r]) for r in range(1, world)).div(world).backward()
        for p, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(p.grad.full_tensor(), expected.grad)
    finally:
        dist.destroy_process_group()


class TestAllMaskedBatch:
    """A batch with every label ``-100`` has nothing to supervise. Both registered losses
    return an exact zero that stays on the autograd graph, so backward runs through the
    model on that rank as on its peers, and logits that are not finite cannot make it NaN."""

    @pytest.mark.parametrize("bad_logit", [None, float("nan"), float("inf")])
    @pytest.mark.parametrize("loss_name", sorted(ALL_MASKED_LOSSES))
    def test_exact_zero_on_the_graph(self, loss_name, bad_logit):
        logits = torch.randn(2, 8, 16)
        if bad_logit is not None:
            logits[0, 0, 0] = bad_logit
        logits.requires_grad_(True)
        loss = ALL_MASKED_LOSSES[loss_name](logits, torch.full((2, 8), -100))
        assert loss.item() == 0.0
        assert loss.dtype == torch.float32
        loss.backward()
        assert torch.equal(logits.grad, torch.zeros_like(logits))

    @pytest.mark.parametrize("loss_name", sorted(ALL_MASKED_LOSSES))
    def test_backward_reaches_the_model_parameters(self, loss_name):
        model = nn.Linear(4, 16)
        loss = ALL_MASKED_LOSSES[loss_name](model(torch.randn(2, 8, 4)), torch.full((2, 8), -100))
        loss.backward()
        assert torch.equal(model.weight.grad, torch.zeros_like(model.weight))
        assert torch.equal(model.bias.grad, torch.zeros_like(model.bias))

    @pytest.mark.parametrize("loss_name", ["cross_entropy", "chunked_many_chunks"])
    def test_fsdp2_rank_with_nothing_supervised_still_reduces(self, loss_name):
        mp.spawn(_rank_with_nothing_supervised_joins, args=(2, _free_port(), loss_name), nprocs=2)
