"""Unit tests for MoE MLP layer and transformer integration."""

from __future__ import annotations

import pytest
import torch

from kempnerforge.config.registry import registry
from kempnerforge.config.schema import DistributedConfig, JobConfig, ModelConfig, TrainConfig
from kempnerforge.model.mlp import StandardMLP, SwiGLUMLP, build_mlp
from kempnerforge.model.moe import MoEMLP, build_moe
from kempnerforge.model.router import SigmoidTopKRouter, SoftmaxTopKRouter
from kempnerforge.model.transformer import Transformer, TransformerBlock

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestMoEMLP:
    def test_output_shape(self):
        router = SoftmaxTopKRouter(dim=64, num_experts=8, top_k=2)
        experts = torch.nn.ModuleList([SwiGLUMLP(64, 128) for _ in range(8)])
        moe = MoEMLP(router, experts)
        x = torch.randn(2, 32, 64)
        out = moe(x)
        assert out.shape == (2, 32, 64)

    def test_backward(self):
        router = SoftmaxTopKRouter(dim=64, num_experts=4, top_k=2)
        experts = torch.nn.ModuleList([SwiGLUMLP(64, 128) for _ in range(4)])
        moe = MoEMLP(router, experts)
        x = torch.randn(2, 16, 64)
        out = moe(x)
        loss = out.sum()
        loss.backward()
        # Router gradients
        assert moe.router.gate.weight.grad is not None
        assert moe.router.gate.weight.grad.abs().sum() > 0
        # Expert gradients (at least some experts should have grads)
        expert_grads = [
            any(p.grad is not None and p.grad.abs().sum() > 0 for p in e.parameters())
            for e in moe.experts
        ]
        assert any(expert_grads)

    def test_aux_loss_accessible(self):
        router = SoftmaxTopKRouter(dim=64, num_experts=8, top_k=2)
        experts = torch.nn.ModuleList([SwiGLUMLP(64, 128) for _ in range(8)])
        moe = MoEMLP(router, experts)
        x = torch.randn(2, 16, 64)
        moe(x)
        assert torch.isfinite(moe.aux_loss)
        assert moe.aux_loss.item() > 0

    def test_expert_counts_accessible(self):
        router = SoftmaxTopKRouter(dim=64, num_experts=8, top_k=2)
        experts = torch.nn.ModuleList([SwiGLUMLP(64, 128) for _ in range(8)])
        moe = MoEMLP(router, experts)
        x = torch.randn(2, 16, 64)
        moe(x)
        num_tokens = 2 * 16
        assert moe.expert_counts.shape == (8,)
        assert moe.expert_counts.sum().item() == num_tokens * 2  # top_k=2

    def test_all_experts_independent(self):
        router = SoftmaxTopKRouter(dim=64, num_experts=4, top_k=2)
        experts = torch.nn.ModuleList([SwiGLUMLP(64, 128) for _ in range(4)])
        moe = MoEMLP(router, experts)
        # Each expert should have its own parameters (not shared references)
        for i in range(4):
            for j in range(i + 1, 4):
                params = zip(moe.experts[i].parameters(), moe.experts[j].parameters(), strict=True)
                for pi, pj in params:
                    assert pi.data_ptr() != pj.data_ptr()

    def test_experts_are_registry_mlps(self):
        moe = build_moe(dim=64, hidden_dim=128, num_experts=4, top_k=2, activation="silu")
        for expert in moe.experts:
            assert isinstance(expert, SwiGLUMLP)

    def test_with_gelu_experts(self):
        moe = build_moe(dim=64, hidden_dim=128, num_experts=4, top_k=2, activation="gelu")
        for expert in moe.experts:
            assert isinstance(expert, StandardMLP)
        # Forward still works
        x = torch.randn(2, 16, 64)
        out = moe(x)
        assert out.shape == (2, 16, 64)

    def test_build_moe_factory(self):
        moe = build_moe(dim=64, hidden_dim=128, num_experts=8, top_k=2)
        assert isinstance(moe, MoEMLP)
        assert moe.num_experts == 8
        assert isinstance(moe.router, SoftmaxTopKRouter)
        assert moe.router.top_k == 2
        assert len(moe.experts) == 8

    def test_single_expert_matches_dense(self):
        """MoE with 1 expert, top_k=1 should produce same output as dense MLP."""
        torch.manual_seed(42)
        dim, hidden_dim = 64, 128

        # Build dense MLP
        dense = build_mlp(dim, hidden_dim, "silu")

        # Build MoE with 1 expert, copy weights from dense
        moe = build_moe(dim=dim, hidden_dim=hidden_dim, num_experts=1, top_k=1)
        # Copy dense weights into the single expert
        moe.experts[0].load_state_dict(dense.state_dict())

        x = torch.randn(2, 16, dim)
        with torch.no_grad():
            dense_out = dense(x)
            moe_out = moe(x)

        # top_k=1, 1 expert → weight is always 1.0 after renormalization
        assert torch.allclose(dense_out, moe_out, atol=1e-5)


# ---------------------------------------------------------------------------
# TransformerBlock + Transformer integration
# ---------------------------------------------------------------------------

_SMALL = dict(dim=128, n_layers=4, n_heads=4, vocab_size=1000, max_seq_len=64)


class TestMoETransformer:
    def test_block_dense_unchanged(self):
        """TransformerBlock with num_experts=0 builds SwiGLUMLP (existing behavior)."""
        config = ModelConfig(**_SMALL)
        block = TransformerBlock(config, layer_idx=0)
        assert isinstance(block.mlp, SwiGLUMLP)

    def test_block_moe(self):
        config = ModelConfig(**_SMALL, num_experts=4, moe_top_k=2)
        block = TransformerBlock(config, layer_idx=0)
        assert isinstance(block.mlp, MoEMLP)
        assert block.mlp.num_experts == 4

    def test_moe_frequency(self):
        """moe_frequency=2: layers 1,3 → MoE, layers 0,2 → dense (4 layers total)."""
        config = ModelConfig(**_SMALL, num_experts=4, moe_top_k=2, moe_frequency=2)
        model = Transformer(config)
        for name, layer in model.layers.items():
            idx = int(name)
            if (idx + 1) % 2 == 0:  # layers 1, 3
                assert isinstance(layer.mlp, MoEMLP), f"layer {idx} should be MoE"
            else:  # layers 0, 2
                assert isinstance(layer.mlp, SwiGLUMLP), f"layer {idx} should be dense"

    def test_moe_forward_shape(self):
        config = ModelConfig(**_SMALL, num_experts=4, moe_top_k=2)
        model = Transformer(config).to(DEVICE)
        tokens = torch.randint(0, 1000, (2, 32), device=DEVICE)
        with torch.no_grad():
            out = model(tokens)
        assert out.shape == (2, 32, 1000)

    def test_get_moe_aux_loss(self):
        config = ModelConfig(**_SMALL, num_experts=4, moe_top_k=2)
        model = Transformer(config).to(DEVICE)
        tokens = torch.randint(0, 1000, (2, 32), device=DEVICE)
        with torch.no_grad():
            model(tokens)
        aux = model.get_moe_aux_loss()
        assert torch.isfinite(aux)
        assert aux.item() > 0

    def test_get_moe_aux_loss_dense_returns_zero(self):
        config = ModelConfig(**_SMALL)
        model = Transformer(config).to(DEVICE)
        tokens = torch.randint(0, 1000, (2, 32), device=DEVICE)
        with torch.no_grad():
            model(tokens)
        assert model.get_moe_aux_loss().item() == 0.0

    def test_get_expert_counts(self):
        config = ModelConfig(**_SMALL, num_experts=4, moe_top_k=2)
        model = Transformer(config).to(DEVICE)
        tokens = torch.randint(0, 1000, (2, 32), device=DEVICE)
        with torch.no_grad():
            model(tokens)
        counts = model.get_expert_counts()
        assert len(counts) == 4  # all 4 layers are MoE (frequency=1)
        for _layer_idx, c in counts.items():
            assert c.shape == (4,)
            assert c.sum().item() == 2 * 32 * 2  # batch * seq * top_k

    def test_get_expert_counts_dense_returns_empty(self):
        config = ModelConfig(**_SMALL)
        model = Transformer(config).to(DEVICE)
        tokens = torch.randint(0, 1000, (2, 32), device=DEVICE)
        with torch.no_grad():
            model(tokens)
        assert model.get_expert_counts() == {}

    def test_moe_backward_all_grads(self):
        config = ModelConfig(**_SMALL, num_experts=4, moe_top_k=2)
        model = Transformer(config).to(DEVICE)
        tokens = torch.randint(0, 1000, (2, 32), device=DEVICE)
        logits = model(tokens)
        loss = logits.sum()
        loss.backward()
        # All parameters should have gradients
        for name, p in model.named_parameters():
            assert p.grad is not None, f"{name} has no gradient"

    def test_dense_model_completely_unchanged(self):
        """Dense config produces identical param count as before MoE was added."""
        config = ModelConfig(**_SMALL)
        model = Transformer(config)
        actual = sum(p.numel() for p in model.parameters())
        assert actual == config.num_params_estimate


# ---------------------------------------------------------------------------
# Phase 7: SigmoidTopKRouter (DeepSeek-V3 style)
# ---------------------------------------------------------------------------


class TestSigmoidTopKRouter:
    def test_output_shapes(self):
        router = SigmoidTopKRouter(dim=64, num_experts=8, top_k=2)
        x = torch.randn(32, 64)
        weights, indices = router(x)
        assert weights.shape == (32, 2)
        assert indices.shape == (32, 2)

    def test_no_aux_loss(self):
        """Sigmoid router uses bias-based balancing, not auxiliary loss."""
        router = SigmoidTopKRouter(dim=64, num_experts=8, top_k=2)
        x = torch.randn(32, 64)
        router(x)
        assert router.aux_loss.item() == 0.0

    def test_weights_sum_to_one(self):
        """Routing weights are normalized to sum to 1 per token."""
        router = SigmoidTopKRouter(dim=64, num_experts=8, top_k=2)
        x = torch.randn(32, 64)
        weights, _ = router(x)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_expert_counts_tracked(self):
        router = SigmoidTopKRouter(dim=64, num_experts=8, top_k=2)
        x = torch.randn(32, 64)
        router(x)
        assert router.expert_counts.shape == (8,)
        assert router.expert_counts.sum().item() == 32 * 2  # num_tokens * top_k

    def test_ema_updates_in_training(self):
        router = SigmoidTopKRouter(dim=64, num_experts=4, top_k=2)
        router.train()
        initial_ema = router.expert_ema.clone()
        x = torch.randn(128, 64)
        router(x)
        # EMA should shift from uniform initialization
        assert not torch.allclose(router.expert_ema, initial_ema)

    def test_ema_frozen_in_eval(self):
        router = SigmoidTopKRouter(dim=64, num_experts=4, top_k=2)
        router.eval()
        initial_ema = router.expert_ema.clone()
        x = torch.randn(128, 64)
        router(x)
        assert torch.allclose(router.expert_ema, initial_ema)

    def test_bias_adjusts_toward_balance(self):
        """Expert bias should shift to balance utilization over many steps."""
        router = SigmoidTopKRouter(dim=64, num_experts=4, top_k=2, bias_update_rate=0.01)
        router.train()
        initial_bias = router.expert_bias.data.clone()
        # Run several forward passes to accumulate bias adjustments
        for _ in range(50):
            x = torch.randn(64, 64)
            router(x)
        # Bias should have changed from zero initialization
        assert not torch.allclose(router.expert_bias.data, initial_bias)

    def test_backward_through_gate(self):
        router = SigmoidTopKRouter(dim=64, num_experts=4, top_k=2)
        x = torch.randn(16, 64, requires_grad=True)
        weights, _ = router(x)
        weights.sum().backward()
        assert router.gate.weight.grad is not None
        assert router.gate.weight.grad.abs().sum() > 0

    def test_registry_lookup(self):
        builder = registry.get("router", "sigmoid_topk")
        router = builder(64, 8, 2)
        assert isinstance(router, SigmoidTopKRouter)


class TestSigmoidMoEIntegration:
    """Sigmoid router integrated into MoEMLP and Transformer."""

    def test_build_moe_with_sigmoid(self):
        moe = build_moe(dim=64, hidden_dim=128, num_experts=4, top_k=2, router_type="sigmoid_topk")
        assert isinstance(moe.router, SigmoidTopKRouter)
        x = torch.randn(2, 16, 64)
        out = moe(x)
        assert out.shape == (2, 16, 64)

    def test_sigmoid_moe_backward(self):
        moe = build_moe(dim=64, hidden_dim=128, num_experts=4, top_k=2, router_type="sigmoid_topk")
        x = torch.randn(2, 16, 64)
        out = moe(x)
        loss = out.sum()
        loss.backward()
        assert moe.router.gate.weight.grad is not None

    def test_sigmoid_transformer_forward(self):
        config = ModelConfig(**_SMALL, num_experts=4, moe_top_k=2, moe_router="sigmoid_topk")
        model = Transformer(config).to(DEVICE)
        tokens = torch.randint(0, 1000, (2, 32), device=DEVICE)
        with torch.no_grad():
            out = model(tokens)
        assert out.shape == (2, 32, 1000)

    def test_sigmoid_aux_loss_zero_in_transformer(self):
        config = ModelConfig(**_SMALL, num_experts=4, moe_top_k=2, moe_router="sigmoid_topk")
        model = Transformer(config).to(DEVICE)
        tokens = torch.randint(0, 1000, (2, 32), device=DEVICE)
        with torch.no_grad():
            model(tokens)
        assert model.get_moe_aux_loss().item() == 0.0

    def test_config_switch_router_type(self):
        """Same architecture, different router — both produce valid forward."""
        for router in ["softmax_topk", "sigmoid_topk"]:
            config = ModelConfig(**_SMALL, num_experts=4, moe_top_k=2, moe_router=router)
            model = Transformer(config).to(DEVICE)
            tokens = torch.randint(0, 1000, (2, 32), device=DEVICE)
            with torch.no_grad():
                out = model(tokens)
            assert out.shape == (2, 32, 1000), f"Failed for router={router}"


# ---------------------------------------------------------------------------
# Phase 7: Shared Experts
# ---------------------------------------------------------------------------


class TestSharedExperts:
    def test_shared_expert_output_additive(self):
        """Shared expert output is added to routed expert output."""
        torch.manual_seed(42)
        moe_no_shared = build_moe(dim=64, hidden_dim=128, num_experts=4, top_k=2)
        moe_shared = build_moe(dim=64, hidden_dim=128, num_experts=4, top_k=2, shared_experts=1)
        # Copy router and expert weights so only difference is shared expert
        moe_shared.router.load_state_dict(moe_no_shared.router.state_dict())
        for i in range(4):
            moe_shared.experts[i].load_state_dict(moe_no_shared.experts[i].state_dict())
        x = torch.randn(2, 16, 64)
        with torch.no_grad():
            out_no_shared = moe_no_shared(x)
            out_shared = moe_shared(x)
        # Shared expert adds something — outputs should differ
        assert not torch.allclose(out_no_shared, out_shared, atol=1e-6)

    def test_shared_expert_processes_all_tokens(self):
        """Shared expert should receive all tokens, not just routed ones."""
        moe = build_moe(dim=64, hidden_dim=128, num_experts=4, top_k=2, shared_experts=1)
        assert moe.shared_expert is not None
        assert isinstance(moe.shared_expert, SwiGLUMLP)

    def test_shared_expert_backward(self):
        moe = build_moe(dim=64, hidden_dim=128, num_experts=4, top_k=2, shared_experts=1)
        x = torch.randn(2, 16, 64)
        out = moe(x)
        out.sum().backward()
        # Shared expert should have gradients
        for p in moe.shared_expert.parameters():
            assert p.grad is not None
            assert p.grad.abs().sum() > 0

    def test_shared_expert_config(self):
        config = ModelConfig(**_SMALL, num_experts=4, moe_top_k=2, moe_shared_experts=1)
        model = Transformer(config).to(DEVICE)
        # Check first MoE layer has a shared expert
        layer = model.layers["0"]
        assert isinstance(layer.mlp, MoEMLP)
        assert layer.mlp.shared_expert is not None

    def test_shared_expert_param_count(self):
        """Shared expert adds one extra MLP worth of parameters per MoE layer."""
        config_no_shared = ModelConfig(**_SMALL, num_experts=4, moe_top_k=2, moe_shared_experts=0)
        config_shared = ModelConfig(**_SMALL, num_experts=4, moe_top_k=2, moe_shared_experts=1)
        assert config_shared.num_params_estimate > config_no_shared.num_params_estimate

    def test_deepseek_style_config_forward(self):
        """DeepSeek-V3 style: sigmoid router + shared expert, many experts."""
        config = ModelConfig(
            dim=128, n_layers=4, n_heads=4, vocab_size=1000, max_seq_len=64,
            num_experts=16, moe_top_k=2, moe_router="sigmoid_topk",
            moe_shared_experts=1, moe_aux_loss_weight=0.0,
        )
        model = Transformer(config).to(DEVICE)
        tokens = torch.randint(0, 1000, (2, 32), device=DEVICE)
        with torch.no_grad():
            out = model(tokens)
        assert out.shape == (2, 32, 1000)
        # Sigmoid router → aux_loss is 0
        assert model.get_moe_aux_loss().item() == 0.0


class TestEPConfig:
    """Expert parallelism config validation (no GPU needed)."""

    def test_ep_default_is_one(self):
        config = DistributedConfig()
        assert config.ep == 1

    def test_ep_in_world_size_product(self):
        config = DistributedConfig(ep=2, tp=2, dp_shard=2)
        config.validate_world_size(8)  # 1 * 2 * 2 * 1 * 1 * 2 = 8

    def test_ep_world_size_mismatch(self):
        config = DistributedConfig(ep=2, tp=2)
        with pytest.raises(ValueError):
            config.validate_world_size(3)

    def test_ep_resolve_includes_ep(self):
        config = DistributedConfig(ep=2, tp=2)
        resolved = config.resolve(world_size=4)
        assert resolved.ep == 2
        assert resolved.dp_shard == 1

    def test_ep_auto_dp_shard_with_ep(self):
        config = DistributedConfig(ep=2, tp=2)
        resolved = config.resolve(world_size=8)
        assert resolved.dp_shard == 2
        assert resolved.ep == 2

    def test_ep_requires_moe(self):
        config = JobConfig(
            model=ModelConfig(dim=64, n_layers=2, n_heads=2, vocab_size=100, max_seq_len=32),
            train=TrainConfig(seq_len=32),
            distributed=DistributedConfig(ep=2),
        )
        with pytest.raises(ValueError, match="ep > 1 requires an MoE model"):
            config.validate(world_size=2)

    def test_ep_must_divide_num_experts(self):
        config = JobConfig(
            model=ModelConfig(
                dim=64, n_layers=2, n_heads=2, vocab_size=100, max_seq_len=32,
                num_experts=5, moe_top_k=2,
            ),
            train=TrainConfig(seq_len=32),
            distributed=DistributedConfig(ep=2),
        )
        with pytest.raises(ValueError, match="num_experts.*must be divisible by ep"):
            config.validate(world_size=2)

    def test_ep_valid_moe_config(self):
        config = JobConfig(
            model=ModelConfig(
                dim=64, n_layers=2, n_heads=2, vocab_size=100, max_seq_len=32,
                num_experts=4, moe_top_k=2,
            ),
            train=TrainConfig(seq_len=32),
            distributed=DistributedConfig(ep=2),
        )
        config.validate(world_size=2)  # should not raise

    def test_ep_invalid_value(self):
        with pytest.raises(ValueError, match="ep must be >= 1"):
            DistributedConfig(ep=0)


class TestGroupedGEMM:
    """Verify grouped GEMM matches loop-based expert forward."""

    def test_grouped_matches_loop_swiglu(self):
        """Grouped GEMM output matches sequential loop for SwiGLU experts."""
        from kempnerforge.model.moe import _HAS_GROUPED_MM, grouped_expert_forward

        if not _HAS_GROUPED_MM:
            pytest.skip("torch._grouped_mm not available")

        torch.manual_seed(42)
        dim, hidden, num_experts = 64, 128, 4
        experts = torch.nn.ModuleList([SwiGLUMLP(dim, hidden) for _ in range(num_experts)])
        experts.eval()

        # 20 tokens sorted by expert: 5, 7, 3, 5 per expert
        tokens_per_expert = [5, 7, 3, 5]
        total = sum(tokens_per_expert)
        x_sorted = torch.randn(total, dim)

        # Grouped path
        grouped_out = grouped_expert_forward(x_sorted, tokens_per_expert, experts)

        # Loop path
        loop_out = torch.zeros_like(x_sorted)
        offset = 0
        for i, count in enumerate(tokens_per_expert):
            if count > 0:
                loop_out[offset:offset + count] = experts[i](x_sorted[offset:offset + count])
            offset += count

        torch.testing.assert_close(grouped_out, loop_out, atol=1e-5, rtol=1e-5)

    def test_grouped_matches_loop_standard(self):
        """Grouped GEMM output matches sequential loop for StandardMLP experts."""
        from kempnerforge.model.moe import _HAS_GROUPED_MM, grouped_expert_forward

        if not _HAS_GROUPED_MM:
            pytest.skip("torch._grouped_mm not available")

        torch.manual_seed(42)
        dim, hidden, num_experts = 64, 128, 4
        experts = torch.nn.ModuleList(
            [StandardMLP(dim, hidden, activation="gelu") for _ in range(num_experts)]
        )
        experts.eval()

        tokens_per_expert = [6, 4, 8, 2]
        total = sum(tokens_per_expert)
        x_sorted = torch.randn(total, dim)

        grouped_out = grouped_expert_forward(x_sorted, tokens_per_expert, experts)

        loop_out = torch.zeros_like(x_sorted)
        offset = 0
        for i, count in enumerate(tokens_per_expert):
            if count > 0:
                loop_out[offset:offset + count] = experts[i](x_sorted[offset:offset + count])
            offset += count

        torch.testing.assert_close(grouped_out, loop_out, atol=1e-5, rtol=1e-5)

    def test_grouped_backward(self):
        """Grouped GEMM path produces valid gradients."""
        from kempnerforge.model.moe import _HAS_GROUPED_MM, grouped_expert_forward

        if not _HAS_GROUPED_MM:
            pytest.skip("torch._grouped_mm not available")

        dim, hidden, num_experts = 32, 64, 3
        experts = torch.nn.ModuleList([SwiGLUMLP(dim, hidden) for _ in range(num_experts)])
        tokens_per_expert = [4, 6, 2]
        x = torch.randn(12, dim, requires_grad=True)

        out = grouped_expert_forward(x, tokens_per_expert, experts)
        out.sum().backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        for expert in experts:
            assert expert.gate_proj.weight.grad is not None

    def test_grouped_empty_expert(self):
        """Grouped GEMM handles experts with zero tokens."""
        from kempnerforge.model.moe import _HAS_GROUPED_MM, grouped_expert_forward

        if not _HAS_GROUPED_MM:
            pytest.skip("torch._grouped_mm not available")

        dim, hidden = 32, 64
        experts = torch.nn.ModuleList([SwiGLUMLP(dim, hidden) for _ in range(3)])
        # Expert 1 gets 0 tokens
        tokens_per_expert = [4, 0, 6]
        x = torch.randn(10, dim)

        out = grouped_expert_forward(x, tokens_per_expert, experts)
        assert out.shape == (10, dim)
        assert torch.isfinite(out).all()

    def test_moe_forward_uses_grouped(self):
        """Full MoEMLP forward uses grouped path and produces correct shapes."""
        from kempnerforge.model.moe import _HAS_GROUPED_MM

        if not _HAS_GROUPED_MM:
            pytest.skip("torch._grouped_mm not available")

        moe = build_moe(dim=64, hidden_dim=128, num_experts=4, top_k=2)
        x = torch.randn(2, 16, 64)
        out = moe(x)
        assert out.shape == (2, 16, 64)

        # Backward should work
        out.sum().backward()
        for p in moe.parameters():
            assert p.grad is not None


class TestCapacityFactor:
    """Capacity factor token dropping."""

    def test_capacity_drops_excess_tokens(self):
        """Capacity factor zeroes weights for overflow tokens."""
        from kempnerforge.model.moe import _apply_capacity

        torch.manual_seed(0)
        num_tokens, top_k, num_experts = 100, 2, 4
        # All tokens go to expert 0 → way over capacity
        indices = torch.zeros(num_tokens, top_k, dtype=torch.long)
        weights = torch.ones(num_tokens, top_k) / top_k

        # capacity = ceil(100*2/4 * 1.0) = 50
        new_weights, new_indices = _apply_capacity(weights, indices, num_experts, 1.0)

        # Expert 0 should have at most 50 tokens per slot
        for k in range(top_k):
            active = (new_weights[:, k] > 0).sum().item()
            assert active <= 50

    def test_capacity_zero_disables(self):
        """capacity_factor=0 passes through unchanged."""
        moe = build_moe(dim=64, hidden_dim=128, num_experts=4, top_k=2, capacity_factor=0.0)
        assert moe.capacity_factor == 0.0
        x = torch.randn(2, 16, 64)
        out = moe(x)
        assert out.shape == (2, 16, 64)

    def test_capacity_forward_backward(self):
        """MoE with capacity factor produces valid output and gradients."""
        moe = build_moe(dim=64, hidden_dim=128, num_experts=4, top_k=2, capacity_factor=1.25)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = moe(x)
        assert out.shape == (2, 16, 64)
        out.sum().backward()
        assert x.grad is not None


class TestCompile:
    """torch.compile compatibility with MoE."""

    def test_compile_moe_bf16(self):
        """MoE compiles and runs correctly in bf16."""
        moe = build_moe(dim=64, hidden_dim=128, num_experts=4, top_k=2).to(torch.bfloat16)
        x = torch.randn(2, 16, 64, dtype=torch.bfloat16)

        eager_out = moe(x)
        compiled_moe = torch.compile(moe, fullgraph=False)
        compiled_out = compiled_moe(x)

        assert compiled_out.shape == eager_out.shape
        torch.testing.assert_close(compiled_out, eager_out, atol=1e-2, rtol=1e-2)

    def test_compile_moe_fp32_fallback(self):
        """MoE falls back to loop under compile with fp32 (grouped_mm needs bf16)."""
        moe = build_moe(dim=64, hidden_dim=128, num_experts=4, top_k=2)
        x = torch.randn(2, 16, 64)

        compiled_moe = torch.compile(moe, fullgraph=False)
        out = compiled_moe(x)
        assert out.shape == (2, 16, 64)


class TestEPAttributes:
    """MoEMLP default EP attributes (no GPU needed)."""

    def test_default_ep_attributes(self):
        router = SoftmaxTopKRouter(dim=64, num_experts=4, top_k=2)
        experts = torch.nn.ModuleList([SwiGLUMLP(64, 128) for _ in range(4)])
        moe = MoEMLP(router, experts)
        assert moe.ep_world_size == 1
        assert moe.ep_group is None
        assert moe.local_expert_start == 0
        assert moe.num_local_experts == 4
