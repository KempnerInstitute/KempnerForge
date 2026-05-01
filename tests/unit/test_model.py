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


class TestModalityIdsCrossArgs:
    """modality_ids cross-arg invariants on Transformer.forward.

    Intra-context invariants live in test_modality_context.py; here we
    test the forward-arg interactions: dtype check, kv_caches forbids
    modality_ids (training-only).
    """

    def test_modality_ids_wrong_dtype_raises(self):
        model = Transformer(_KWARG_CONFIG).to(DEVICE)
        tokens = torch.randint(0, 256, (1, 8), device=DEVICE)
        prefix = torch.randn(1, 4, 128, device=DEVICE)
        # int32 instead of long — should raise.
        bad_ids = torch.zeros(1, 12, dtype=torch.int32, device=DEVICE)
        with pytest.raises(ValueError, match="modality_ids.dtype must be torch.long"):
            model(
                tokens,
                modality=ModalityContext(prefix_embeds=prefix, modality_ids=bad_ids),
            )

    def test_modality_ids_with_kv_caches_raises(self):
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
        # Use inputs_embeds (allowed with kv_caches; pipeline-PP path) so the
        # modality_ids check is what fires, not prefix_embeds.
        embeds = torch.randn(1, 4, 128, device=DEVICE)
        ids = torch.zeros(1, 4, dtype=torch.long, device=DEVICE)
        with pytest.raises(ValueError, match="modality_ids is training-only"):
            model(
                None,
                modality=ModalityContext(inputs_embeds=embeds, modality_ids=ids),
                kv_caches=kv,
            )

    def test_modality_ids_correct_dtype_no_raise_yet(self):
        """Long-dtype modality_ids passes the dtype check at the top of
        forward. Non-MoT Transformer with modality_ids set in the context
        just no-ops it for the residual stream — pin that behavior."""
        model = Transformer(_KWARG_CONFIG).to(DEVICE)
        tokens = torch.randint(0, 256, (1, 8), device=DEVICE)
        prefix = torch.randn(1, 4, 128, device=DEVICE)
        ids = torch.zeros(1, 12, dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            out = model(
                tokens,
                modality=ModalityContext(prefix_embeds=prefix, modality_ids=ids),
            )
        assert out.shape == (1, 12, 256)


# ---------------------------------------------------------------------------
# MoT (Mixture-of-Transformers) integration on Transformer
# ---------------------------------------------------------------------------


def _mot_config(
    dim: int = 128,
    n_layers: int = 2,
    n_heads: int = 4,
    n_kv_heads: int | None = None,
    num_tokens: int = 8,
    max_text_len: int = 16,
    mot_image_n_heads: int = 0,
    mot_image_n_kv_heads: int = 0,
) -> ModelConfig:
    """Tiny MoT-backed ModelConfig for integration tests."""
    from kempnerforge.config.vlm import MoTConfig

    return ModelConfig(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads or n_heads,
        vocab_size=256,
        max_seq_len=num_tokens + max_text_len,
        ffn_hidden_dim=128,
        vlm=MoTConfig(
            vision_encoder="random",
            num_tokens=num_tokens,
            max_text_len=max_text_len,
            mot_image_n_heads=mot_image_n_heads,
            mot_image_n_kv_heads=mot_image_n_kv_heads,
        ),
    )


class TestMoT:
    """MoTConfig-backed Transformer: per-modality blocks + global SDPA."""

    def test_layers_are_mot_blocks_when_mot(self):
        from kempnerforge.model.mot import MoTBlock

        cfg = _mot_config()
        model = Transformer(cfg).to(DEVICE)
        assert all(isinstance(layer, MoTBlock) for layer in model.layers.values())
        assert model._mot_modalities == ("image", "text")
        assert model._mot_n_image == 8
        assert set(model.mot_norms.keys()) == {"image", "text"}

    def test_layers_are_transformer_blocks_when_not_mot(self):
        """Regression: text-only path keeps TransformerBlock structure."""
        cfg = ModelConfig(dim=128, n_layers=2, n_heads=4, vocab_size=256, max_seq_len=32)
        model = Transformer(cfg).to(DEVICE)
        assert all(isinstance(layer, TransformerBlock) for layer in model.layers.values())
        assert model._mot_modalities == ()
        assert model._mot_n_image == 0
        assert len(model.mot_norms) == 0

    def test_unequal_image_n_heads_raises(self):
        """v1 enforces equal head counts across modalities."""
        with pytest.raises(ValueError, match="equal head counts"):
            cfg = _mot_config(mot_image_n_heads=2)
            Transformer(cfg)

    def test_unequal_image_n_kv_heads_raises(self):
        with pytest.raises(ValueError, match="equal head counts"):
            cfg = _mot_config(mot_image_n_kv_heads=2)
            Transformer(cfg)

    def test_modality_ids_required_when_mot_active(self):
        cfg = _mot_config()
        model = Transformer(cfg).to(DEVICE).eval()
        tokens = torch.randint(0, 256, (1, cfg.vlm.max_text_len), device=DEVICE)  # type: ignore[union-attr]
        prefix = torch.randn(1, cfg.vlm.num_tokens, cfg.dim, device=DEVICE)  # type: ignore[union-attr]
        with pytest.raises(ValueError, match="MoT model requires modality.modality_ids"):
            model(tokens, modality=ModalityContext(prefix_embeds=prefix))

    def test_modality_ids_shape_mismatch_raises(self):
        cfg = _mot_config()
        model = Transformer(cfg).to(DEVICE).eval()
        tokens = torch.randint(0, 256, (1, cfg.vlm.max_text_len), device=DEVICE)  # type: ignore[union-attr]
        prefix = torch.randn(1, cfg.vlm.num_tokens, cfg.dim, device=DEVICE)  # type: ignore[union-attr]
        bad_ids = torch.zeros(1, 5, dtype=torch.long, device=DEVICE)
        with pytest.raises(ValueError, match="modality.modality_ids shape"):
            model(
                tokens,
                modality=ModalityContext(prefix_embeds=prefix, modality_ids=bad_ids),
            )

    def test_forward_output_shape(self):
        cfg = _mot_config()
        model = Transformer(cfg).to(DEVICE).eval()
        n_image = cfg.vlm.num_tokens  # type: ignore[union-attr]
        n_text = cfg.vlm.max_text_len  # type: ignore[union-attr]
        total = n_image + n_text
        tokens = torch.randint(0, 256, (1, n_text), device=DEVICE)
        prefix = torch.randn(1, n_image, cfg.dim, device=DEVICE)
        ids = torch.zeros(1, total, dtype=torch.long, device=DEVICE)
        ids[:, n_image:] = 1
        with torch.no_grad():
            out = model(
                tokens,
                modality=ModalityContext(prefix_embeds=prefix, modality_ids=ids),
            )
        assert out.shape == (1, total, cfg.vocab_size)
        assert torch.isfinite(out).all()

    def test_forward_output_slice_works(self):
        """output_slice composes with the MoT path: trim image tokens off the head input."""
        cfg = _mot_config()
        model = Transformer(cfg).to(DEVICE).eval()
        n_image = cfg.vlm.num_tokens  # type: ignore[union-attr]
        n_text = cfg.vlm.max_text_len  # type: ignore[union-attr]
        tokens = torch.randint(0, 256, (1, n_text), device=DEVICE)
        prefix = torch.randn(1, n_image, cfg.dim, device=DEVICE)
        ids = torch.zeros(1, n_image + n_text, dtype=torch.long, device=DEVICE)
        ids[:, n_image:] = 1
        with torch.no_grad():
            out = model(
                tokens,
                modality=ModalityContext(
                    prefix_embeds=prefix,
                    modality_ids=ids,
                    output_slice=slice(n_image, None),
                ),
            )
        assert out.shape == (1, n_text, cfg.vocab_size)

    def test_state_dict_contains_per_modality_keys(self):
        cfg = _mot_config()
        model = Transformer(cfg).to(DEVICE)
        keys = set(model.state_dict().keys())
        assert "layers.0.attn.q_proj.image.weight" in keys
        assert "layers.0.attn.q_proj.text.weight" in keys
        assert "layers.0.attn.o_proj.image.weight" in keys
        assert "layers.0.attn.o_proj.text.weight" in keys
        assert "layers.0.attn_norm.image.weight" in keys
        assert "layers.0.mlp_norm.text.weight" in keys
        assert "layers.0.mlp.image.gate_proj.weight" in keys
        assert "layers.0.mlp.text.down_proj.weight" in keys
        assert "mot_norms.image.weight" in keys
        assert "mot_norms.text.weight" in keys
        assert not any(".attention.q_proj.weight" in k for k in keys)

    def test_backward_through_mot_path(self):
        """Backward flows through per-modality projections after a non-zero
        o_proj re-init (zero-init residual blocks gradient to upstream
        Q/K/V via the residual chain rule)."""
        cfg = _mot_config()
        model = Transformer(cfg).to(DEVICE)
        with torch.no_grad():
            for layer in model.layers.values():
                for m in layer.modalities:
                    torch.nn.init.normal_(layer.attn.o_proj[m].weight, std=0.01)
                    torch.nn.init.normal_(layer.mlp[m].down_proj.weight, std=0.01)
        n_image = cfg.vlm.num_tokens  # type: ignore[union-attr]
        n_text = cfg.vlm.max_text_len  # type: ignore[union-attr]
        tokens = torch.randint(0, 256, (1, n_text), device=DEVICE)
        prefix = torch.randn(1, n_image, cfg.dim, device=DEVICE)
        ids = torch.zeros(1, n_image + n_text, dtype=torch.long, device=DEVICE)
        ids[:, n_image:] = 1
        out = model(
            tokens,
            modality=ModalityContext(prefix_embeds=prefix, modality_ids=ids),
        )
        out.sum().backward()
        for m in ("image", "text"):
            assert model.layers["0"].attn.q_proj[m].weight.grad is not None  # type: ignore[union-attr]
            assert model.layers["0"].attn.q_proj[m].weight.grad.abs().sum() > 0  # type: ignore[union-attr]


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
