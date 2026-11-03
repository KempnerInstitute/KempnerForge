"""Unit tests for loss functions."""

from __future__ import annotations

import functools
from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F

from kempnerforge.config.schema import TrainConfig
from kempnerforge.training.loss import (
    build_loss_fn,
    chunked_cross_entropy_loss,
    cross_entropy_loss,
    z_loss,
)


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
        logits[:, :, 0] = 100.0  # High confidence on class 0
        loss = cross_entropy_loss(logits, labels)
        assert loss.item() < 0.01


class TestChunkedCrossEntropy:
    def test_matches_standard_ce(self):
        """Chunked CE must produce the same result as standard CE."""
        torch.manual_seed(42)
        logits = torch.randn(4, 32, 256)  # num_tokens=128
        labels = torch.randint(0, 256, (4, 32))

        standard = cross_entropy_loss(logits, labels)
        chunked = chunked_cross_entropy_loss(logits, labels, chunk_size=32)  # 4 chunks

        torch.testing.assert_close(chunked, standard, rtol=1e-4, atol=1e-4)

    def test_small_batch_bypasses_chunking(self):
        """When num_tokens <= chunk_size, should use standard CE path."""
        logits = torch.randn(2, 8, 50)  # num_tokens=16
        labels = torch.randint(0, 50, (2, 8))

        standard = cross_entropy_loss(logits, labels)
        chunked = chunked_cross_entropy_loss(logits, labels, chunk_size=4096)

        torch.testing.assert_close(chunked, standard, rtol=1e-5, atol=1e-5)

    def test_chunked_path_activates(self):
        """Chunking path should produce correct results with various chunk sizes."""
        torch.manual_seed(0)
        logits = torch.randn(8, 16, 512)  # num_tokens=128
        labels = torch.randint(0, 512, (8, 16))

        standard = cross_entropy_loss(logits, labels)
        # Various chunk sizes — all should match standard CE
        for cs in [16, 32, 64, 100]:
            chunked = chunked_cross_entropy_loss(logits, labels, chunk_size=cs)
            torch.testing.assert_close(chunked, standard, rtol=1e-4, atol=1e-4)

    def test_uneven_chunks(self):
        """Last chunk may be smaller — should still be correct."""
        torch.manual_seed(1)
        logits = torch.randn(3, 10, 200)  # num_tokens=30
        labels = torch.randint(0, 200, (3, 10))

        standard = cross_entropy_loss(logits, labels)
        chunked = chunked_cross_entropy_loss(logits, labels, chunk_size=7)  # 4 full + 1 partial

        torch.testing.assert_close(chunked, standard, rtol=1e-4, atol=1e-4)

    def test_gradient_flows(self):
        logits = torch.randn(4, 16, 512, requires_grad=True)  # num_tokens=64
        labels = torch.randint(0, 512, (4, 16))
        loss = chunked_cross_entropy_loss(logits, labels, chunk_size=16)  # chunking active
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
    """bf16 logits get an fp32 loss and gradient, computed one chunk at a time."""

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
    @pytest.mark.parametrize("chunk_size", [1, 7, 64, 4096])
    @pytest.mark.parametrize("loss_fn", [cross_entropy_loss, chunked_cross_entropy_loss])
    def test_bf16_logits_match_fp64(
        self, loss_fn: Callable[..., torch.Tensor], chunk_size: int, device: str
    ) -> None:
        torch.manual_seed(0)
        logits = (torch.randn(4, 16, 1000) * 4).bfloat16().to(device).requires_grad_(True)
        labels = torch.randint(0, 1000, (4, 16)).to(device)
        labels[0, :5] = -100
        loss = loss_fn(logits, labels, chunk_size=chunk_size)
        loss.backward()
        ref_loss, ref_grad = _fp64_reference(logits, labels)
        assert loss.dtype == torch.float32
        assert logits.grad is not None and logits.grad.dtype == torch.bfloat16
        torch.testing.assert_close(loss.double(), ref_loss, rtol=1e-6, atol=0.0)
        torch.testing.assert_close(logits.grad.double(), ref_grad, rtol=2**-8, atol=1e-12)

    def test_fp64_logits_stay_fp64_and_pass_gradcheck(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(10, 7, dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([0, 1, -100, 3, 4, 5, 6, -100, 2, 1])
        assert cross_entropy_loss(logits, labels, chunk_size=3).dtype == torch.float64
        assert torch.autograd.gradcheck(
            lambda x: cross_entropy_loss(x, labels, chunk_size=3), (logits,)
        )

    def test_backward_saves_no_fp32_copy_of_the_logits(self) -> None:
        logits = torch.randn(64, 300).bfloat16().requires_grad_(True)
        labels = torch.randint(0, 300, (64,))
        saved: list[torch.Tensor] = []

        def pack(t: torch.Tensor) -> torch.Tensor:
            saved.append(t)
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            loss = cross_entropy_loss(logits, labels, chunk_size=8)
        loss.backward()
        assert any(t.data_ptr() == logits.data_ptr() for t in saved)
        assert all(t.data_ptr() == logits.data_ptr() or t.numel() <= 64 for t in saved)

    def test_ignored_rows_get_zero_gradient_even_when_not_finite(self) -> None:
        logits = torch.randn(6, 50)
        logits[2] = float("nan")
        logits[4, 0] = float("inf")
        logits.requires_grad_(True)
        labels = torch.tensor([1, 2, -100, 3, -100, 4])
        loss = cross_entropy_loss(logits, labels, chunk_size=4)
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(loss)
        assert torch.equal(logits.grad[[2, 4]], torch.zeros(2, 50))
        assert torch.isfinite(logits.grad).all()

    def test_all_masked_returns_fp32_zero(self) -> None:
        logits = torch.randn(2, 8, 16).bfloat16()
        loss = cross_entropy_loss(logits, torch.full((2, 8), -100), chunk_size=4)
        assert loss.dtype == torch.float32 and loss.item() == 0.0

    @pytest.mark.gpu
    def test_peak_memory_holds_no_full_fp32_copy(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        tokens, vocab, chunk = 8192, 32768, 1024
        logits = torch.randn(tokens, vocab, device="cuda", dtype=torch.bfloat16)
        logits.requires_grad_(True)
        labels = torch.randint(0, vocab, (tokens,), device="cuda")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        cross_entropy_loss(logits, labels, chunk_size=chunk).backward()
        torch.cuda.synchronize()
        extra = torch.cuda.max_memory_allocated() - before
        grad_bytes = tokens * vocab * 2
        chunk_fp32_bytes = chunk * vocab * 4
        assert extra <= grad_bytes + 3 * chunk_fp32_bytes
        assert extra < tokens * vocab * 4


class TestZLoss:
    def test_zero_weight_returns_zero(self):
        logits = torch.randn(2, 8, 100)
        result = z_loss(logits, weight=0.0)
        assert result.item() == 0.0

    def test_small_logits_small_zloss(self):
        """Small logits should produce small z-loss."""
        logits = torch.zeros(2, 8, 100)  # All zeros
        result = z_loss(logits, weight=1e-4)
        # logsumexp of zeros = log(vocab) ≈ 4.6, squared ≈ 21.2, * 1e-4 ≈ 0.002
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

    @pytest.mark.parametrize("loss_fn", ["cross_entropy", "chunked_cross_entropy"])
    @pytest.mark.parametrize(("ce_chunk_size", "expected"), [(0, 4096), (16, 16)])
    def test_both_losses_get_the_chunk_size(
        self, loss_fn: str, ce_chunk_size: int, expected: int
    ) -> None:
        fn = build_loss_fn(TrainConfig(loss_fn=loss_fn, ce_chunk_size=ce_chunk_size))
        assert isinstance(fn, functools.partial)
        assert fn.keywords == {"chunk_size": expected}

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
