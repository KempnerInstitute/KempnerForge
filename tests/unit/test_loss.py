"""Unit tests for loss functions."""

from __future__ import annotations

import torch

from kempnerforge.training.loss import (
    chunked_cross_entropy_loss,
    cross_entropy_loss,
    z_loss,
)


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
