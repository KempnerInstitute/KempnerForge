"""Unit tests for MoE MLP layer and transformer integration."""

from __future__ import annotations

import torch

from kempnerforge.config.schema import ModelConfig
from kempnerforge.model.mlp import StandardMLP, SwiGLUMLP, build_mlp
from kempnerforge.model.moe import MoEMLP, build_moe
from kempnerforge.model.router import SoftmaxTopKRouter
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
                for pi, pj in zip(moe.experts[i].parameters(), moe.experts[j].parameters()):
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
        for layer_idx, c in counts.items():
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
