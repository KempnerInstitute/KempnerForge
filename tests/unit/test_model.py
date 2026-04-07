"""Unit tests for KempnerForge model components."""

from __future__ import annotations

import math

import pytest
import torch

from kempnerforge.config.schema import ModelConfig
from kempnerforge.model.attention import Attention
from kempnerforge.model.embedding import OutputHead, TokenEmbedding
from kempnerforge.model.init import init_weights
from kempnerforge.model.mlp import StandardMLP, SwiGLUMLP, build_mlp
from kempnerforge.model.norm import RMSNorm, build_norm
from kempnerforge.model.position import apply_rope, precompute_rope_frequencies
from kempnerforge.model.transformer import Transformer, TransformerBlock

# Use GPU if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(64).to(DEVICE)
        x = torch.randn(2, 16, 64, device=DEVICE)
        assert norm(x).shape == (2, 16, 64)

    def test_preserves_dtype(self):
        norm = RMSNorm(64).to(DEVICE)
        x = torch.randn(2, 16, 64, device=DEVICE, dtype=torch.bfloat16)
        assert norm(x).dtype == torch.bfloat16

    def test_unit_rms(self):
        """After RMSNorm, the RMS of each vector should be approximately 1."""
        norm = RMSNorm(256, eps=1e-5).to(DEVICE)
        x = torch.randn(4, 32, 256, device=DEVICE)
        out = norm(x).float()
        rms = out.pow(2).mean(-1).sqrt()
        # Not exactly 1.0 because of the learned weight, but close with default init (ones)
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.05)


class TestBuildNorm:
    def test_rmsnorm(self):
        assert isinstance(build_norm("rmsnorm", 64), RMSNorm)

    def test_layernorm(self):
        assert isinstance(build_norm("layernorm", 64), torch.nn.LayerNorm)

    def test_unknown(self):
        with pytest.raises(KeyError, match="Unknown norm"):
            build_norm("batchnorm", 64)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


class TestRoPE:
    def test_frequency_shape(self):
        cos, sin = precompute_rope_frequencies(head_dim=128, max_seq_len=256, theta=10000.0)
        assert cos.shape == (256, 64)  # head_dim // 2
        assert sin.shape == (256, 64)

    def test_real_dtype(self):
        cos, sin = precompute_rope_frequencies(head_dim=64, max_seq_len=16)
        assert cos.is_floating_point()
        assert sin.is_floating_point()

    def test_unit_magnitude(self):
        """cos² + sin² should equal 1 for all entries."""
        cos, sin = precompute_rope_frequencies(head_dim=64, max_seq_len=128)
        assert torch.allclose(cos.pow(2) + sin.pow(2), torch.ones_like(cos), atol=1e-5)

    def test_position_zero_is_identity(self):
        """At position 0, rotation should be by angle 0 → cos=1, sin=0."""
        cos, sin = precompute_rope_frequencies(head_dim=64, max_seq_len=16)
        assert torch.allclose(cos[0], torch.ones(32), atol=1e-5)
        assert torch.allclose(sin[0], torch.zeros(32), atol=1e-5)

    def test_apply_rope_shape(self):
        cos, sin = precompute_rope_frequencies(head_dim=64, max_seq_len=128)
        x = torch.randn(2, 8, 128, 64)  # (batch, heads, seq, head_dim)
        out = apply_rope(x, cos, sin)
        assert out.shape == x.shape

    def test_apply_rope_preserves_dtype(self):
        cos, sin = precompute_rope_frequencies(head_dim=64, max_seq_len=32)
        x = torch.randn(1, 4, 32, 64, dtype=torch.bfloat16)
        out = apply_rope(x, cos, sin)
        assert out.dtype == torch.bfloat16

    def test_position_zero_is_identity_in_practice(self):
        """At position 0, apply_rope should not change the input."""
        cos, sin = precompute_rope_frequencies(head_dim=64, max_seq_len=4)
        x = torch.randn(1, 1, 1, 64)  # single position
        out = apply_rope(x, cos[:1], sin[:1])
        assert torch.allclose(out, x, atol=1e-5)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class TestAttention:
    def test_mha_output_shape(self):
        attn = Attention(dim=256, n_heads=8, n_kv_heads=8).to(DEVICE)
        cos, sin = precompute_rope_frequencies(32, 64)
        cos, sin = cos.to(DEVICE), sin.to(DEVICE)
        x = torch.randn(2, 64, 256, device=DEVICE)
        out = attn(x, cos, sin)
        assert out.shape == (2, 64, 256)

    def test_gqa_output_shape(self):
        attn = Attention(dim=256, n_heads=8, n_kv_heads=2).to(DEVICE)
        cos, sin = precompute_rope_frequencies(32, 32)
        cos, sin = cos.to(DEVICE), sin.to(DEVICE)
        x = torch.randn(1, 32, 256, device=DEVICE)
        out = attn(x, cos, sin)
        assert out.shape == (1, 32, 256)

    def test_mqa_output_shape(self):
        attn = Attention(dim=256, n_heads=8, n_kv_heads=1).to(DEVICE)
        cos, sin = precompute_rope_frequencies(32, 16)
        cos, sin = cos.to(DEVICE), sin.to(DEVICE)
        x = torch.randn(1, 16, 256, device=DEVICE)
        out = attn(x, cos, sin)
        assert out.shape == (1, 16, 256)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class TestMLP:
    def test_swiglu_shape(self):
        mlp = SwiGLUMLP(256, 512).to(DEVICE)
        x = torch.randn(2, 32, 256, device=DEVICE)
        assert mlp(x).shape == (2, 32, 256)

    def test_standard_shape(self):
        mlp = StandardMLP(256, 512, activation="gelu").to(DEVICE)
        x = torch.randn(2, 32, 256, device=DEVICE)
        assert mlp(x).shape == (2, 32, 256)

    def test_build_mlp_silu_is_swiglu(self):
        assert isinstance(build_mlp(256, 512, "silu"), SwiGLUMLP)

    def test_build_mlp_gelu_is_standard(self):
        assert isinstance(build_mlp(256, 512, "gelu"), StandardMLP)

    def test_swiglu_has_three_matrices(self):
        mlp = SwiGLUMLP(256, 512)
        params = list(mlp.parameters())
        assert len(params) == 3  # gate, up, down


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


class TestEmbedding:
    def test_token_embedding_shape(self):
        emb = TokenEmbedding(1000, 256).to(DEVICE)
        tokens = torch.randint(0, 1000, (2, 32), device=DEVICE)
        assert emb(tokens).shape == (2, 32, 256)

    def test_output_head_shape(self):
        head = OutputHead(256, 1000).to(DEVICE)
        x = torch.randn(2, 32, 256, device=DEVICE)
        assert head(x).shape == (2, 32, 1000)

    def test_weight_tying(self):
        emb = TokenEmbedding(1000, 256)
        head = OutputHead(256, 1000)
        head.tie_weights(emb)
        assert head.proj.weight is emb.embedding.weight


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------


class TestTransformerBlock:
    def test_output_shape(self):
        config = ModelConfig(dim=256, n_layers=1, n_heads=4, vocab_size=1000, max_seq_len=64)
        block = TransformerBlock(config, layer_idx=0).to(DEVICE)
        cos, sin = precompute_rope_frequencies(64, 32)
        cos, sin = cos.to(DEVICE), sin.to(DEVICE)
        x = torch.randn(2, 32, 256, device=DEVICE)
        assert block(x, cos, sin).shape == (2, 32, 256)

    def test_residual_connection(self):
        """Output should differ from input (attention + MLP do something)."""
        config = ModelConfig(dim=256, n_layers=1, n_heads=4, vocab_size=1000, max_seq_len=64)
        block = TransformerBlock(config, layer_idx=0).to(DEVICE)
        cos, sin = precompute_rope_frequencies(64, 16)
        cos, sin = cos.to(DEVICE), sin.to(DEVICE)
        x = torch.randn(1, 16, 256, device=DEVICE)
        out = block(x, cos, sin)
        assert not torch.allclose(out, x)


# ---------------------------------------------------------------------------
# Full Transformer
# ---------------------------------------------------------------------------


class TestTransformer:
    @pytest.fixture
    def small_config(self):
        return ModelConfig(
            dim=256, n_layers=4, n_heads=4, n_kv_heads=4, vocab_size=1000, max_seq_len=128
        )

    def test_forward_shape(self, small_config):
        model = Transformer(small_config).to(DEVICE)
        tokens = torch.randint(0, 1000, (2, 64), device=DEVICE)
        with torch.no_grad():
            out = model(tokens)
        assert out.shape == (2, 64, 1000)

    def test_param_count_matches_estimate(self, small_config):
        model = Transformer(small_config)
        actual = sum(p.numel() for p in model.parameters())
        assert actual == small_config.num_params_estimate

    def test_bf16_no_warnings(self, small_config):
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            model = Transformer(small_config).to(DEVICE).to(torch.bfloat16)
            tokens = torch.randint(0, 1000, (1, 32), device=DEVICE)
            with torch.no_grad():
                out = model(tokens)
            assert out.dtype == torch.bfloat16

    def test_meta_device(self, small_config):
        with torch.device("meta"):
            model = Transformer(small_config)
        assert all(p.device.type == "meta" for p in model.parameters())

    def test_llama_7b_param_count(self):
        config = ModelConfig(
            dim=4096,
            n_layers=32,
            n_heads=32,
            n_kv_heads=32,
            vocab_size=32000,
            ffn_hidden_dim=11008,
        )
        with torch.device("meta"):
            model = Transformer(config)
        total = sum(p.numel() for p in model.parameters())
        assert 6.5e9 < total < 7.0e9

    def test_gqa_forward(self):
        config = ModelConfig(
            dim=256, n_layers=2, n_heads=8, n_kv_heads=2, vocab_size=1000, max_seq_len=64
        )
        model = Transformer(config).to(DEVICE)
        tokens = torch.randint(0, 1000, (1, 32), device=DEVICE)
        with torch.no_grad():
            out = model(tokens)
        assert out.shape == (1, 32, 1000)

    def test_weight_tying(self):
        config = ModelConfig(
            dim=256,
            n_layers=2,
            n_heads=4,
            vocab_size=1000,
            max_seq_len=64,
            tie_embeddings=True,
        )
        model = Transformer(config)
        assert model.output_head.proj.weight is model.token_embedding.embedding.weight

    def test_layer_keys_are_string_indices(self, small_config):
        model = Transformer(small_config)
        assert list(model.layers.keys()) == ["0", "1", "2", "3"]


# ---------------------------------------------------------------------------
# Weight initialization
# ---------------------------------------------------------------------------

_INIT_CONFIG = ModelConfig(dim=128, n_layers=4, n_heads=4, vocab_size=256, max_seq_len=32)


class TestInitWeights:
    def test_linear_weights_are_normal(self):
        """init_weights should set linear weights to ~N(0, init_std)."""
        model = Transformer(_INIT_CONFIG)
        init_weights(model, _INIT_CONFIG)
        for name, p in model.named_parameters():
            if (
                p.dim() >= 2
                and "norm" not in name
                and not name.endswith(("o_proj.weight", "down_proj.weight"))
            ):
                assert abs(p.std().item() - _INIT_CONFIG.init_std) < 0.015, (
                    f"{name}: std={p.std().item():.4f}, expected ~{_INIT_CONFIG.init_std}"
                )

    def test_residual_projections_are_scaled(self):
        """o_proj and down_proj should have std = init_std / sqrt(2 * n_layers)."""
        model = Transformer(_INIT_CONFIG)
        init_weights(model, _INIT_CONFIG)
        expected_std = _INIT_CONFIG.init_std / math.sqrt(2.0 * _INIT_CONFIG.n_layers)
        for name, p in model.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                assert abs(p.std().item() - expected_std) < 0.01, (
                    f"{name}: std={p.std().item():.4f}, expected ~{expected_std:.4f}"
                )

    def test_norm_weights_unchanged(self):
        """Norm weights should remain at 1.0 (default)."""
        model = Transformer(_INIT_CONFIG)
        init_weights(model, _INIT_CONFIG)
        for name, p in model.named_parameters():
            if "norm" in name and "weight" in name:
                assert torch.allclose(p, torch.ones_like(p)), f"{name} should be all ones"

    def test_custom_init_std(self):
        """Custom init_std should be respected."""
        config = ModelConfig(
            dim=128, n_layers=4, n_heads=4, vocab_size=256, max_seq_len=32, init_std=0.01
        )
        model = Transformer(config)
        init_weights(model, config)
        stds = [
            p.std().item()
            for n, p in model.named_parameters()
            if p.dim() >= 2
            and "norm" not in n
            and not n.endswith(("o_proj.weight", "down_proj.weight"))
        ]
        assert all(abs(s - 0.01) < 0.01 for s in stds)

    def test_skips_meta_parameters(self):
        """init_weights should skip parameters on meta device without error."""
        with torch.device("meta"):
            model = Transformer(_INIT_CONFIG)
        # Should not raise even though all params are on meta device
        init_weights(model, _INIT_CONFIG)
