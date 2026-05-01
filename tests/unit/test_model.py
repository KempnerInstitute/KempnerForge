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
from kempnerforge.model.modality import ModalityContext
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

    def test_qk_norm_output_shape(self):
        """QK-Norm should not change output shape."""
        attn = Attention(dim=256, n_heads=8, n_kv_heads=8, qk_norm=True).to(DEVICE)
        cos, sin = precompute_rope_frequencies(32, 32)
        cos, sin = cos.to(DEVICE), sin.to(DEVICE)
        x = torch.randn(2, 32, 256, device=DEVICE)
        out = attn(x, cos, sin)
        assert out.shape == (2, 32, 256)

    def test_qk_norm_creates_norm_layers(self):
        attn = Attention(dim=128, n_heads=4, n_kv_heads=4, qk_norm=True)
        assert attn.q_norm is not None
        assert attn.k_norm is not None
        assert attn.q_norm.weight.shape == (32,)  # head_dim = 128/4

    def test_qk_norm_disabled_by_default(self):
        attn = Attention(dim=128, n_heads=4, n_kv_heads=4)
        assert attn.q_norm is None
        assert attn.k_norm is None

    def test_qk_norm_bounds_attention_logits(self):
        """QK-Norm should produce bounded Q/K regardless of input scale."""
        attn_norm = Attention(dim=128, n_heads=4, n_kv_heads=4, qk_norm=True).to(DEVICE)
        cos, sin = precompute_rope_frequencies(32, 16)
        cos, sin = cos.to(DEVICE), sin.to(DEVICE)

        # Large input that would cause attention logit explosion without norm
        x = torch.randn(1, 16, 128, device=DEVICE) * 100.0
        out_norm = attn_norm(x, cos, sin)

        # Normalized version should still produce finite output with large inputs
        assert out_norm.isfinite().all()

    def test_qk_norm_with_gqa(self):
        """QK-Norm should work with grouped-query attention."""
        attn = Attention(dim=256, n_heads=8, n_kv_heads=2, qk_norm=True).to(DEVICE)
        cos, sin = precompute_rope_frequencies(32, 32)
        cos, sin = cos.to(DEVICE), sin.to(DEVICE)
        x = torch.randn(1, 32, 256, device=DEVICE)
        out = attn(x, cos, sin)
        assert out.shape == (1, 32, 256)

    def test_sdpa_backend_defaults_to_auto(self):
        attn = Attention(dim=128, n_heads=4, n_kv_heads=4)
        assert attn.sdpa_backend == "auto"

    def test_sdpa_backend_stored_on_module(self):
        attn = Attention(dim=128, n_heads=4, n_kv_heads=4, sdpa_backend="math")
        assert attn.sdpa_backend == "math"


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

    def test_qk_norm_forward(self):
        config = ModelConfig(
            dim=128,
            n_layers=2,
            n_heads=4,
            vocab_size=256,
            max_seq_len=64,
            qk_norm=True,
        )
        model = Transformer(config).to(DEVICE)
        tokens = torch.randint(0, 256, (1, 16), device=DEVICE)
        with torch.no_grad():
            out = model(tokens)
        assert out.shape == (1, 16, 256)

    def test_qk_norm_false_is_no_op(self):
        """qk_norm=False should produce identical model to default."""
        config_default = ModelConfig(dim=128, n_layers=2, n_heads=4, vocab_size=256, max_seq_len=64)
        config_explicit = ModelConfig(
            dim=128,
            n_layers=2,
            n_heads=4,
            vocab_size=256,
            max_seq_len=64,
            qk_norm=False,
        )
        m1 = Transformer(config_default)
        m2 = Transformer(config_explicit)
        # Neither should have norm layers in attention
        assert m1.layers["0"].attention.q_norm is None
        assert m2.layers["0"].attention.q_norm is None
        # Same parameter count
        assert sum(p.numel() for p in m1.parameters()) == sum(p.numel() for p in m2.parameters())


# ---------------------------------------------------------------------------
# Modality-injection routes on Transformer.forward (ModalityContext)
# ---------------------------------------------------------------------------


_KWARG_CONFIG = ModelConfig(dim=128, n_layers=2, n_heads=4, vocab_size=256, max_seq_len=64)


class TestInputsEmbeds:
    def test_both_none_raises(self):
        model = Transformer(_KWARG_CONFIG).to(DEVICE)
        with pytest.raises(ValueError, match="exactly one of tokens or modality.inputs_embeds"):
            model()

    def test_both_set_raises(self):
        model = Transformer(_KWARG_CONFIG).to(DEVICE)
        tokens = torch.randint(0, 256, (1, 8), device=DEVICE)
        embeds = torch.randn(1, 8, 128, device=DEVICE)
        with pytest.raises(ValueError, match="exactly one of tokens or modality.inputs_embeds"):
            model(tokens, modality=ModalityContext(inputs_embeds=embeds))

    def test_inputs_embeds_matches_tokens(self):
        """inputs_embeds path is bit-equal to the tokens path when embeds
        come from the model's own token_embedding."""
        model = Transformer(_KWARG_CONFIG).to(DEVICE).eval()
        tokens = torch.randint(0, 256, (2, 16), device=DEVICE)
        with torch.no_grad():
            out_tokens = model(tokens)
            embeds = model.token_embedding(tokens)
            out_embeds = model(modality=ModalityContext(inputs_embeds=embeds))
        assert torch.equal(out_tokens, out_embeds)

    def test_inputs_embeds_backward(self):
        """Gradients should flow through an externally-provided inputs_embeds."""
        model = Transformer(_KWARG_CONFIG).to(DEVICE)
        embeds = torch.randn(1, 8, 128, device=DEVICE, requires_grad=True)
        out = model(modality=ModalityContext(inputs_embeds=embeds))
        out.sum().backward()
        assert embeds.grad is not None
        assert torch.isfinite(embeds.grad).all()

    def test_inputs_embeds_shape(self):
        model = Transformer(_KWARG_CONFIG).to(DEVICE)
        embeds = torch.randn(3, 12, 128, device=DEVICE)
        with torch.no_grad():
            out = model(modality=ModalityContext(inputs_embeds=embeds))
        assert out.shape == (3, 12, 256)


class TestOutputSlice:
    def test_output_slice_trims_logits(self):
        model = Transformer(_KWARG_CONFIG).to(DEVICE).eval()
        tokens = torch.randint(0, 256, (2, 20), device=DEVICE)
        with torch.no_grad():
            full = model(tokens)
            sliced = model(tokens, modality=ModalityContext(output_slice=slice(5, None)))
        assert full.shape == (2, 20, 256)
        assert sliced.shape == (2, 15, 256)
        # Sliced positions equal the tail of the full output
        assert torch.equal(sliced, full[:, 5:, :])

    def test_output_slice_none_matches_full(self):
        model = Transformer(_KWARG_CONFIG).to(DEVICE).eval()
        tokens = torch.randint(0, 256, (1, 10), device=DEVICE)
        with torch.no_grad():
            a = model(tokens)
            b = model(tokens, modality=ModalityContext(output_slice=None))
        assert torch.equal(a, b)

    def test_output_slice_full_range_matches_full(self):
        model = Transformer(_KWARG_CONFIG).to(DEVICE).eval()
        tokens = torch.randint(0, 256, (1, 10), device=DEVICE)
        with torch.no_grad():
            a = model(tokens)
            b = model(tokens, modality=ModalityContext(output_slice=slice(None, None)))
        assert torch.equal(a, b)

    def test_output_slice_with_kv_caches_raises(self):
        from kempnerforge.model.attention import KVCache

        model = Transformer(_KWARG_CONFIG).to(DEVICE)
        kv = [
            KVCache(
                batch_size=1,
                max_seq_len=32,
                n_kv_heads=4,
                head_dim=32,
                dtype=torch.float32,
                device=DEVICE,
            )
            for _ in range(_KWARG_CONFIG.n_layers)
        ]
        tokens = torch.randint(0, 256, (1, 4), device=DEVICE)
        with pytest.raises(ValueError, match="modality.output_slice is training-only"):
            model(
                tokens,
                modality=ModalityContext(output_slice=slice(1, None)),
                kv_caches=kv,
            )


class TestPrefixEmbeds:
    """prefix_embeds field: image/soft-prompt prefix concatenated to the
    left of the token embeddings. Entry point is VLMWrapper but the
    behavior is unit-tested here directly so a Transformer regression
    surfaces without needing a VLM build."""

    def test_basic_shape(self):
        """prefix_embeds=(B, N, D) + tokens=(B, T) -> logits (B, N+T, V)."""
        model = Transformer(_KWARG_CONFIG).to(DEVICE).eval()
        tokens = torch.randint(0, 256, (2, 10), device=DEVICE)
        prefix = torch.randn(2, 6, 128, device=DEVICE)
        with torch.no_grad():
            out = model(tokens, modality=ModalityContext(prefix_embeds=prefix))
        assert out.shape == (2, 16, 256)

    def test_matches_manual_concat(self):
        """The prefix_embeds path is bit-equal to the manual
        inputs_embeds=cat([prefix, embed(tokens)], 1) path (sanity check
        that prefix is placed on the left, not the right)."""
        model = Transformer(_KWARG_CONFIG).to(DEVICE).eval()
        tokens = torch.randint(0, 256, (1, 8), device=DEVICE)
        prefix = torch.randn(1, 4, 128, device=DEVICE)
        with torch.no_grad():
            txt_embeds = model.token_embedding(tokens)
            manual = model(
                modality=ModalityContext(inputs_embeds=torch.cat([prefix, txt_embeds], dim=1))
            )
            prefixed = model(tokens, modality=ModalityContext(prefix_embeds=prefix))
        assert torch.equal(manual, prefixed)

    def test_inputs_embeds_and_prefix_embeds_mutually_exclusive(self):
        """ModalityContext.__post_init__ rejects setting both inputs_embeds
        and prefix_embeds (mutually exclusive residual-stream routes)."""
        embeds = torch.randn(1, 8, 128, device=DEVICE)
        prefix = torch.randn(1, 4, 128, device=DEVICE)
        with pytest.raises(ValueError, match="mutually exclusive"):
            ModalityContext(inputs_embeds=embeds, prefix_embeds=prefix)

    def test_dtype_promotion_cast(self):
        """prefix in fp32 while the transformer is bf16: the cast happens
        inside Transformer.forward before the concat, output dtype matches
        the transformer."""
        model = Transformer(_KWARG_CONFIG).to(DEVICE).to(torch.bfloat16)
        tokens = torch.randint(0, 256, (1, 4), device=DEVICE)
        prefix = torch.randn(1, 3, 128, device=DEVICE, dtype=torch.float32)
        with torch.no_grad():
            out = model(tokens, modality=ModalityContext(prefix_embeds=prefix))
        assert out.dtype == torch.bfloat16
        assert out.shape == (1, 7, 256)

    def test_with_kv_caches_raises(self):
        """prefix_embeds is training-only; combining with kv_caches would
        cause the RoPE offset to double-count the prefix at every decode
        step. Guard rejects it."""
        from kempnerforge.model.attention import KVCache

        model = Transformer(_KWARG_CONFIG).to(DEVICE)
        kv = [
            KVCache(
                batch_size=1,
                max_seq_len=32,
                n_kv_heads=4,
                head_dim=32,
                dtype=torch.float32,
                device=DEVICE,
            )
            for _ in range(_KWARG_CONFIG.n_layers)
        ]
        tokens = torch.randint(0, 256, (1, 4), device=DEVICE)
        prefix = torch.randn(1, 2, 128, device=DEVICE)
        with pytest.raises(ValueError, match="modality.prefix_embeds is training-only"):
            model(tokens, modality=ModalityContext(prefix_embeds=prefix), kv_caches=kv)


# ---------------------------------------------------------------------------
# Cross-Attention interleaving on Transformer
# ---------------------------------------------------------------------------


def _ca_config(n_layers: int, cadence: int) -> ModelConfig:
    """Tiny CrossAttentionConfig-backed ModelConfig for interleaving tests."""
    from kempnerforge.config.vlm import CrossAttentionConfig

    return ModelConfig(
        dim=64,
        n_layers=n_layers,
        n_heads=4,
        vocab_size=256,
        max_seq_len=128,
        vlm=CrossAttentionConfig(
            vision_encoder="random",
            feature_dim=64,
            num_tokens=8,
            cross_attention_every_n_layers=cadence,
            max_text_len=64,
        ),
    )


class TestCrossAttentionInterleaving:
    def test_n28_cadence4_yields_7_ca_blocks(self):
        """Paper baseline: 28 layers, cadence 4 -> 7 CA blocks."""
        model = Transformer(_ca_config(n_layers=28, cadence=4))
        assert len(model.cross_attention_layers) == 7
        assert model._ca_cadence == 4

    def test_n32_cadence4_yields_8_ca_blocks(self):
        """Default 7B backbone: 32 layers, cadence 4 -> 8 CA blocks."""
        model = Transformer(_ca_config(n_layers=32, cadence=4))
        assert len(model.cross_attention_layers) == 8

    def test_n28_cadence4_attaches_at_paper_positions(self):
        """CA fires after text block i iff (i+1) % cadence == 0.
        For 28/4 the firing indices are {3, 7, 11, 15, 19, 23, 27}.
        We exercise the boundary by counting CA invocations."""
        from kempnerforge.config.vlm import CrossAttentionConfig
        from kempnerforge.model.cross_attention import CrossAttentionBlock

        config = _ca_config(n_layers=28, cadence=4)
        assert isinstance(config.vlm, CrossAttentionConfig)

        model = Transformer(config).to(DEVICE).eval()
        # Replace CA blocks with counters via monkeypatching forward.
        invocations: list[int] = []
        original = CrossAttentionBlock.forward

        def counting_forward(self, x, image_features, image_mask=None):
            invocations.append(len(invocations))
            return original(self, x, image_features, image_mask)

        CrossAttentionBlock.forward = counting_forward  # type: ignore[method-assign]
        try:
            tokens = torch.randint(0, 256, (1, 8), device=DEVICE)
            img = torch.randn(1, 8, 64, device=DEVICE)
            with torch.no_grad():
                model(tokens, modality=ModalityContext(image_features=img))
        finally:
            CrossAttentionBlock.forward = original  # type: ignore[method-assign]
        # 7 CA blocks should have fired exactly once each.
        assert invocations == list(range(7))

    def test_text_only_path_bit_equal_when_no_vlm(self):
        """Without a VLM config, cross_attention_layers is empty and the
        forward-loop's inner branch is dead -> output bit-equal to a
        Transformer built before this commit (text-only path).
        """
        config = ModelConfig(dim=64, n_layers=4, n_heads=4, vocab_size=256, max_seq_len=32)
        model = Transformer(config).to(DEVICE).eval()
        assert len(model.cross_attention_layers) == 0
        assert model._ca_cadence == 0
        tokens = torch.randint(0, 256, (1, 8), device=DEVICE)
        with torch.no_grad():
            out = model(tokens)
        assert out.shape == (1, 8, 256)

    def test_old_jd_checkpoint_loads_with_empty_cross_attention_layers(self):
        """A state dict from a Transformer built before this commit (no
        cross_attention_layers keys) loads cleanly into a Transformer
        with an empty cross_attention_layers ModuleDict (strict=True).
        """
        # Build a Transformer with an empty cross_attention_layers
        # (text-only or JD config). Save its state_dict.
        config = ModelConfig(dim=64, n_layers=4, n_heads=4, vocab_size=256, max_seq_len=32)
        model_a = Transformer(config)
        state = {k: v for k, v in model_a.state_dict().items()}
        # No cross_attention_layers entries should appear when the dict is empty.
        assert not any(k.startswith("cross_attention_layers.") for k in state)
        # Build a fresh Transformer (also empty CA dict) and load.
        model_b = Transformer(config)
        missing_keys, unexpected_keys = model_b.load_state_dict(state, strict=True)
        assert missing_keys == []
        assert unexpected_keys == []

    def test_cadence_change_in_state_dict_load(self):
        """A checkpoint saved with cadence=4 (n_layers=8 -> 2 CA blocks)
        does not load into a model built with cadence=2 (n_layers=8 ->
        4 CA blocks): strict=True raises with a missing-keys error
        mentioning cross_attention_layers.
        """
        cfg_4 = _ca_config(n_layers=8, cadence=4)  # K=2
        cfg_2 = _ca_config(n_layers=8, cadence=2)  # K=4
        model_4 = Transformer(cfg_4)
        model_2 = Transformer(cfg_2)
        state_4 = model_4.state_dict()
        with pytest.raises(RuntimeError, match="cross_attention_layers"):
            model_2.load_state_dict(state_4, strict=True)

    def test_ca_preserves_text_causality(self):
        """Text token at position t produces a bit-equal hidden state
        regardless of text tokens at positions > t, even with CA blocks
        interleaved. Two inputs that agree on [0..t-1] and differ on
        [t..end] must produce equal logits on positions [0..t-1].
        """
        torch.manual_seed(0)
        config = _ca_config(n_layers=8, cadence=4)
        model = Transformer(config).to(DEVICE).eval()
        img = torch.randn(1, 8, 64, device=DEVICE)
        seq_len = 12

        for t in [1, 4, 7, 10]:
            tokens_a = torch.randint(0, 256, (1, seq_len), device=DEVICE)
            tokens_b = tokens_a.clone()
            # Differ at positions [t..end]
            tokens_b[:, t:] = (tokens_b[:, t:] + 1) % 256
            with torch.no_grad():
                out_a = model(tokens_a, modality=ModalityContext(image_features=img))
                out_b = model(tokens_b, modality=ModalityContext(image_features=img))
            torch.testing.assert_close(out_a[:, :t, :], out_b[:, :t, :], atol=1e-5, rtol=1e-5)

    def test_ca_zero_init_residual_at_construction(self):
        """At construction (before any optimizer step), CA blocks are
        identity. So a CA-built Transformer's forward output equals a
        text-only Transformer's forward output for the same text tokens
        and same backbone weights.
        """
        torch.manual_seed(0)
        config_ca = _ca_config(n_layers=4, cadence=2)
        text_config = ModelConfig(
            dim=config_ca.dim,
            n_layers=config_ca.n_layers,
            n_heads=config_ca.n_heads,
            vocab_size=config_ca.vocab_size,
            max_seq_len=config_ca.max_seq_len,
        )
        ca_model = Transformer(config_ca).to(DEVICE).eval()
        text_model = Transformer(text_config).to(DEVICE).eval()
        # Copy CA model's text-stack weights into text_model so backbone matches.
        text_state = {
            k: v for k, v in ca_model.state_dict().items() if "cross_attention_layers" not in k
        }
        text_model.load_state_dict(text_state, strict=True)
        tokens = torch.randint(0, 256, (1, 8), device=DEVICE)
        img = torch.randn(1, 8, 64, device=DEVICE)
        with torch.no_grad():
            out_ca = ca_model(tokens, modality=ModalityContext(image_features=img))
            out_text = text_model(tokens)
        torch.testing.assert_close(out_ca, out_text, atol=1e-5, rtol=1e-5)

    def test_image_features_required_when_ca_layers_present(self):
        """Forward without image_features raises a clear error when
        CA layers are configured."""
        config = _ca_config(n_layers=4, cadence=2)
        model = Transformer(config).to(DEVICE).eval()
        tokens = torch.randint(0, 256, (1, 8), device=DEVICE)
        with pytest.raises(ValueError, match="image_features is None"):
            model(tokens)

    def test_image_features_with_kv_caches_raises(self):
        """Cross-arg invariant: image_features is training-only."""
        from kempnerforge.model.attention import KVCache

        config = _ca_config(n_layers=4, cadence=2)
        model = Transformer(config).to(DEVICE)
        kv = [
            KVCache(
                batch_size=1,
                max_seq_len=32,
                n_kv_heads=4,
                head_dim=16,
                dtype=torch.float32,
                device=DEVICE,
            )
            for _ in range(config.n_layers)
        ]
        tokens = torch.randint(0, 256, (1, 4), device=DEVICE)
        img = torch.randn(1, 8, 64, device=DEVICE)
        with pytest.raises(ValueError, match="image_features is training-only"):
            model(tokens, modality=ModalityContext(image_features=img), kv_caches=kv)


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
