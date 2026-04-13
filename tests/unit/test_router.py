"""Unit tests for MoE router components."""

from __future__ import annotations

import math

import torch

from kempnerforge.config.registry import registry
from kempnerforge.model.router import SigmoidTopKRouter, SoftmaxTopKRouter


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


# ---------------------------------------------------------------------------
# Sequence-level auxiliary loss (SigmoidTopKRouter)
# ---------------------------------------------------------------------------


class TestSigmoidSequenceAuxLoss:
    def test_aux_loss_zero_when_disabled(self):
        router = SigmoidTopKRouter(dim=64, num_experts=8, top_k=2, sequence_aux_loss_weight=0.0)
        x = torch.randn(32, 64)
        router(x)
        assert router.aux_loss.item() == 0.0

    def test_aux_loss_positive_when_enabled(self):
        router = SigmoidTopKRouter(dim=64, num_experts=8, top_k=2, sequence_aux_loss_weight=0.01)
        router.train()
        x = torch.randn(128, 64)
        router(x)
        assert router.aux_loss.item() > 0.0

    def test_aux_loss_is_switch_style(self):
        """Aux loss uses Switch-style N * sum(f_i * P_i) formulation."""
        router = SigmoidTopKRouter(dim=64, num_experts=4, top_k=2, sequence_aux_loss_weight=1.0)
        x = torch.randn(128, 64)
        router(x)
        # With weight=1.0, loss = num_experts * sum(f_i * P_i)
        # P_i ≈ 0.5 (sigmoid of random-ish values), sum(f_i) = 1
        # So loss ≈ num_experts * 0.5 = 2.0 (order of magnitude)
        assert router.aux_loss.item() > 0.1  # non-trivial
        assert torch.isfinite(router.aux_loss)

    def test_aux_loss_backward(self):
        """Gradients flow through sequence aux loss via mean routing scores."""
        router = SigmoidTopKRouter(dim=64, num_experts=4, top_k=2, sequence_aux_loss_weight=1.0)
        router.train()
        x = torch.randn(64, 64)
        router(x)
        router.aux_loss.backward()
        # Gradient flows through sigmoid scores → gate weights
        assert router.gate.weight.grad is not None
        assert router.gate.weight.grad.abs().sum() > 0

    def test_registry_with_sequence_aux_loss(self):
        """Builder passes sequence_aux_loss_weight through kwargs."""
        builder = registry.get("router", "sigmoid_topk")
        router = builder(64, 8, 2, sequence_aux_loss_weight=0.01)
        assert isinstance(router, SigmoidTopKRouter)
        assert router.sequence_aux_loss_weight == 0.01


# ---------------------------------------------------------------------------
# Adaptive bias schedule (SigmoidTopKRouter)
# ---------------------------------------------------------------------------


class TestSigmoidBiasSchedule:
    def test_constant_rate_unchanged(self):
        router = SigmoidTopKRouter(dim=64, num_experts=4, top_k=2, bias_schedule="constant")
        router.set_step(500, 1000)
        assert router._effective_bias_rate() == router.bias_update_rate

    def test_cosine_decay_decreases(self):
        router = SigmoidTopKRouter(dim=64, num_experts=4, top_k=2, bias_schedule="cosine_decay")
        router.set_step(0, 1000)
        rate_start = router._effective_bias_rate()
        router.set_step(500, 1000)
        rate_mid = router._effective_bias_rate()
        router.set_step(1000, 1000)
        rate_end = router._effective_bias_rate()
        # Cosine decay: start=full, mid=half, end≈0
        assert rate_start > rate_mid > rate_end
        assert abs(rate_start - router.bias_update_rate) < 1e-7
        assert abs(rate_mid - router.bias_update_rate * 0.5) < 1e-7
        assert rate_end < 1e-7

    def test_cosine_decay_formula(self):
        router = SigmoidTopKRouter(dim=64, num_experts=4, top_k=2, bias_schedule="cosine_decay")
        rate = router.bias_update_rate
        for step, total in [(0, 1000), (250, 1000), (750, 1000)]:
            router.set_step(step, total)
            expected = rate * 0.5 * (1.0 + math.cos(math.pi * step / total))
            assert abs(router._effective_bias_rate() - expected) < 1e-7

    def test_linear_warmup_increases(self):
        router = SigmoidTopKRouter(dim=64, num_experts=4, top_k=2, bias_schedule="linear_warmup")
        router.set_step(0, 1000)
        rate_start = router._effective_bias_rate()
        router.set_step(50, 1000)  # 5% → within warmup
        rate_warmup = router._effective_bias_rate()
        router.set_step(200, 1000)  # 20% → past warmup (10%)
        rate_post = router._effective_bias_rate()
        assert rate_start < rate_warmup < rate_post
        # After warmup, should be at full rate
        assert abs(rate_post - router.bias_update_rate) < 1e-7

    def test_set_step_propagation(self):
        """set_step stores step and max_steps correctly."""
        router = SigmoidTopKRouter(dim=64, num_experts=4, top_k=2)
        router.set_step(42, 100)
        assert router._step == 42
        assert router._max_steps == 100

    def test_bias_schedule_affects_training(self):
        """Cosine decay schedule produces smaller bias changes late in training."""
        torch.manual_seed(42)
        router = SigmoidTopKRouter(
            dim=64,
            num_experts=4,
            top_k=2,
            bias_update_rate=0.01,
            bias_schedule="cosine_decay",
        )
        router.train()

        # Early training: large bias update rate
        router.set_step(0, 1000)
        bias_before_early = router.expert_bias.data.clone()
        router(torch.randn(64, 64))
        delta_early = (router.expert_bias.data - bias_before_early).abs().max().item()

        # Late training: small bias update rate
        router.set_step(990, 1000)
        bias_before_late = router.expert_bias.data.clone()
        router(torch.randn(64, 64))
        delta_late = (router.expert_bias.data - bias_before_late).abs().max().item()

        assert delta_early > delta_late
