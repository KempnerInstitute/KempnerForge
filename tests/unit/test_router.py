"""Unit tests for MoE router components."""

from __future__ import annotations

import torch

from kempnerforge.config.registry import registry
from kempnerforge.model.router import SoftmaxTopKRouter


class TestSoftmaxTopKRouter:
    def test_output_shapes(self):
        router = SoftmaxTopKRouter(dim=64, num_experts=8, top_k=2)
        x = torch.randn(100, 64)
        weights, indices = router(x)
        assert weights.shape == (100, 2)
        assert indices.shape == (100, 2)

    def test_top_k_range(self):
        router = SoftmaxTopKRouter(dim=64, num_experts=8, top_k=2)
        x = torch.randn(100, 64)
        _, indices = router(x)
        assert indices.min() >= 0
        assert indices.max() < 8

    def test_weights_sum_to_one(self):
        router = SoftmaxTopKRouter(dim=64, num_experts=8, top_k=2)
        x = torch.randn(100, 64)
        weights, _ = router(x)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_aux_loss_positive(self):
        router = SoftmaxTopKRouter(dim=64, num_experts=8, top_k=2)
        x = torch.randn(100, 64)
        router(x)
        assert router.aux_loss.item() > 0

    def test_aux_loss_minimal_at_uniform(self):
        """With uniform gate weights, aux_loss should be close to 1.0.

        The minimum of the Switch-style aux loss N * sum(f_i * P_i) is 1.0
        when all experts are equally loaded (f_i = 1/N) and equally probable
        (P_i = 1/N), giving N * N * (1/N * 1/N) = 1.0.
        """
        router = SoftmaxTopKRouter(dim=64, num_experts=8, top_k=2)
        # Zero out gate weights → uniform softmax probabilities
        router.gate.weight.data.zero_()
        x = torch.randn(1024, 64)
        router(x)
        # With uniform routing, aux_loss ≈ 1.0
        assert abs(router.aux_loss.item() - 1.0) < 0.05

    def test_expert_counts(self):
        router = SoftmaxTopKRouter(dim=64, num_experts=8, top_k=2)
        x = torch.randn(100, 64)
        router(x)
        # Each token picks top_k experts, so total assignments = num_tokens * top_k
        assert router.expert_counts.sum().item() == 100 * 2
        assert router.expert_counts.shape == (8,)

    def test_backward(self):
        router = SoftmaxTopKRouter(dim=64, num_experts=8, top_k=2)
        x = torch.randn(100, 64)
        weights, _ = router(x)
        loss = weights.sum()
        loss.backward()
        assert router.gate.weight.grad is not None
        assert router.gate.weight.grad.abs().sum() > 0

    def test_registry(self):
        builder = registry.get("router", "softmax_topk")
        router = builder(dim=64, num_experts=8, top_k=2)
        assert isinstance(router, SoftmaxTopKRouter)
        x = torch.randn(10, 64)
        weights, indices = router(x)
        assert weights.shape == (10, 2)
        assert indices.shape == (10, 2)
