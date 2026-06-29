"""Unit tests for VLMWrapper, inner_transformer, _is_encoder_frozen.

Adapter unit tests moved to ``tests/unit/test_adapter.py`` when ``Adapter``
was renamed to ``MLP2LayerAdapter`` and promoted to a registered component.
"""

from __future__ import annotations

import pytest
import torch

from kempnerforge.config.adapter import AdapterConfig
from kempnerforge.config.model import ModelConfig
from kempnerforge.config.vision import VisionEncoderConfig
from kempnerforge.config.vlm import (
    CrossAttentionConfig,
    FreezeSpec,
    JointDecoderConfig,
    MoMaConfig,
    MoTConfig,
    VLMConfig,
)
from kempnerforge.model.modality import ModalityContext
from kempnerforge.model.transformer import Transformer
from kempnerforge.model.vlm import (
    CrossAttentionStrategy,
    JointDecoderStrategy,
    MoTStrategy,
    VLMWrapper,
    _is_encoder_frozen,
    build_modality_strategy,
    build_vlm_wrapper,
    inner_transformer,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# VLMWrapper
# ---------------------------------------------------------------------------


def _tiny_configs(
    num_image_tokens: int = 8, feature_dim: int = 96
) -> tuple[ModelConfig, VisionEncoderConfig, AdapterConfig, VLMConfig]:
    return (
        ModelConfig(dim=64, n_layers=2, n_heads=4, vocab_size=256, max_seq_len=64),
        VisionEncoderConfig(type="random", feature_dim=feature_dim, num_tokens=num_image_tokens),
        AdapterConfig(),
        VLMConfig(max_text_len=32),
    )


def _build_tiny_wrapper(num_image_tokens: int = 8, feature_dim: int = 96) -> VLMWrapper:
    mc, vc, ac, lc = _tiny_configs(num_image_tokens, feature_dim)
    return build_vlm_wrapper(mc, vc, ac, lc)


class TestVLMWrapper:
    def test_num_image_tokens_from_encoder(self):
        wrapper = _build_tiny_wrapper(num_image_tokens=12)
        assert wrapper.num_image_tokens == 12

    def test_forward_shapes(self):
        wrapper = _build_tiny_wrapper(num_image_tokens=8).to(DEVICE)
        pixels = torch.randn(2, 3, 16, 16, device=DEVICE)
        input_ids = torch.randint(0, 256, (2, 20), device=DEVICE)
        labels = torch.full((2, 20), -100, dtype=torch.long, device=DEVICE)
        logits, labels_out = wrapper(pixels, input_ids, labels)
        # output_slice drops the 8 image positions, so logits cover 20 text positions.
        assert logits.shape == (2, 20, 256)
        assert labels_out is labels

    def test_labels_none_passthrough(self):
        wrapper = _build_tiny_wrapper().to(DEVICE)
        pixels = torch.randn(1, 3, 16, 16, device=DEVICE)
        input_ids = torch.randint(0, 256, (1, 10), device=DEVICE)
        logits, labels_out = wrapper(pixels, input_ids, None)
        assert logits.shape == (1, 10, 256)
        assert labels_out is None

    def test_dtype_mismatch_cast(self):
        """Vision encoder output in fp32, transformer in bf16 -> forward
        still works; the cast happens inside VLMWrapper before concat."""
        wrapper = _build_tiny_wrapper().to(DEVICE)
        wrapper.adapter.to(torch.bfloat16)
        wrapper.transformer.to(torch.bfloat16)
        # Leave vision_encoder in fp32 (default)
        pixels = torch.randn(1, 3, 16, 16, device=DEVICE)
        input_ids = torch.randint(0, 256, (1, 8), device=DEVICE)
        logits, _ = wrapper(pixels, input_ids)
        assert logits.dtype == torch.bfloat16

    def test_backward_gradients_flow(self):
        wrapper = _build_tiny_wrapper().to(DEVICE)
        pixels = torch.randn(1, 3, 16, 16, device=DEVICE)
        input_ids = torch.randint(0, 256, (1, 8), device=DEVICE)
        logits, _ = wrapper(pixels, input_ids)
        logits.sum().backward()
        # Adapter always has grads; transformer params should have grads.
        for p in wrapper.adapter.parameters():
            assert p.grad is not None
        trainable = [p for p in wrapper.transformer.parameters() if p.requires_grad]
        assert any(p.grad is not None for p in trainable)

    def test_dispatch_on_vlm_only(self):
        """build_vlm_wrapper should not require model_type changes; the
        inner transformer is registered under the standard 'transformer'
        model_type, and VLM dispatch happens through the presence of a
        VLMConfig.
        """
        mc, vc, ac, lc = _tiny_configs()
        assert mc.model_type == "transformer"
        wrapper = build_vlm_wrapper(mc, vc, ac, lc)
        assert isinstance(wrapper.transformer, Transformer)

    def test_transformer_reachable(self):
        wrapper = _build_tiny_wrapper()
        assert isinstance(wrapper.transformer, Transformer)


# ---------------------------------------------------------------------------
# inner_transformer helper
# ---------------------------------------------------------------------------


class TestInnerTransformer:
    def test_unwraps_vlm_wrapper(self):
        wrapper = _build_tiny_wrapper()
        inner = inner_transformer(wrapper)
        assert inner is wrapper.transformer
        assert isinstance(inner, Transformer)

    def test_passthrough_for_plain_transformer(self):
        model = Transformer(ModelConfig(dim=64, n_layers=2, n_heads=4, vocab_size=256))
        assert inner_transformer(model) is model

    def test_moe_call_through_helper(self):
        """`inner_transformer(wrapper).set_moe_step(...)` reaches the real
        method even though VLMWrapper itself does not define it."""
        wrapper = _build_tiny_wrapper()
        inner = inner_transformer(wrapper)
        # Not an MoE model, but set_moe_step is defined on Transformer; it
        # should be reachable via the unwrap helper without raising.
        inner.set_moe_step(0, 100)  # type: ignore[attr-defined]

    def test_unwraps_torch_compile_around_vlm_wrapper(self):
        """torch.compile wraps modules in OptimizedModule with ``_orig_mod``;
        ``inner_transformer`` peels both layers to reach the real Transformer."""

        class _FakeCompiled:
            def __init__(self, mod):
                self._orig_mod = mod

        wrapper = _build_tiny_wrapper()
        compiled = _FakeCompiled(wrapper)
        inner = inner_transformer(compiled)  # type: ignore[arg-type]
        assert inner is wrapper.transformer

    def test_unwraps_torch_compile_around_plain_transformer(self):
        model = Transformer(ModelConfig(dim=64, n_layers=2, n_heads=4, vocab_size=256))

        class _FakeCompiled:
            def __init__(self, mod):
                self._orig_mod = mod

        compiled = _FakeCompiled(model)
        assert inner_transformer(compiled) is model  # type: ignore[arg-type]


class TestBuildVLMWrapperErrors:
    def test_max_seq_len_insufficient_for_inferred_encoder_num_tokens(self):
        """When ``vision_encoder.num_tokens=0`` (the "infer from encoder at
        build time" sentinel), the config-time cross-check in
        ``JobConfig.__post_init__`` is skipped. The build-time check in
        ``build_vlm_wrapper`` must catch the case where the encoder's
        resolved ``num_tokens`` + ``max_text_len`` exceeds ``max_seq_len``.
        Regression guard for the validation gap.
        """
        # RandomVisionEncoder's default num_tokens is 16 (see vision.py),
        # so num_tokens=0 -> encoder resolves to 16 at build time.
        # max_text_len=32 -> JD residual = 16 + 32 = 48. max_seq_len=32 < 48.
        mc = ModelConfig(dim=64, n_layers=2, n_heads=4, vocab_size=256, max_seq_len=32)
        vc = VisionEncoderConfig(type="random", num_tokens=0)  # sentinel
        ac = AdapterConfig()
        lc = VLMConfig(max_text_len=32)
        with pytest.raises(ValueError) as exc_info:
            build_vlm_wrapper(mc, vc, ac, lc)
        # Error message must name max_seq_len, encoder.num_tokens, and
        # max_text_len so the user can find the offending knobs.
        msg = str(exc_info.value)
        assert "max_seq_len" in msg
        assert "encoder.num_tokens" in msg
        assert "max_text_len" in msg
        assert "32" in msg  # max_seq_len value
        assert "16" in msg  # encoder.num_tokens value

    def test_build_time_check_skipped_for_cross_attention(self):
        """CA does not extend the residual stream — the build-time check
        must allow max_seq_len < encoder.num_tokens + max_text_len so long
        as max_seq_len >= max_text_len. Without this carve-out, CA configs
        with num_tokens=0 (sentinel) would incorrectly raise.
        """
        # max_seq_len=32, max_text_len=32 -> JD would fail (residual would
        # be 16+32=48), but CA's residual is text-only (0+32=32).
        mc = ModelConfig(dim=64, n_layers=2, n_heads=4, vocab_size=256, max_seq_len=32)
        vc = VisionEncoderConfig(type="random", num_tokens=0)
        ac = AdapterConfig()
        lc = CrossAttentionConfig(max_text_len=32)
        wrapper = build_vlm_wrapper(mc, vc, ac, lc)
        assert isinstance(wrapper, VLMWrapper)
        # CA's residual is text-only, so wrapper.num_image_tokens is 0.
        assert wrapper.num_image_tokens == 0


# ---------------------------------------------------------------------------
# _is_encoder_frozen
# ---------------------------------------------------------------------------


class TestIsEncoderFrozen:
    def test_all_frozen(self):
        specs = [FreezeSpec("vision_encoder", True)]
        assert _is_encoder_frozen(specs) is True

    def test_partial_unfrozen(self):
        specs = [
            FreezeSpec("vision_encoder", True),
            FreezeSpec("vision_encoder.layers.11", False),
        ]
        assert _is_encoder_frozen(specs) is False

    def test_no_relevant_specs(self):
        specs = [FreezeSpec("adapter", True)]
        assert _is_encoder_frozen(specs) is False

    def test_empty_specs(self):
        assert _is_encoder_frozen([]) is False

    def test_nested_key_counts(self):
        specs = [FreezeSpec("vision_encoder.layers.0", True)]
        assert _is_encoder_frozen(specs) is True


# ---------------------------------------------------------------------------
# ModalityStrategy: registry-based dispatch
# ---------------------------------------------------------------------------


def _ca_configs(
    num_image_tokens: int = 8, feature_dim: int = 96, cadence: int = 2
) -> tuple[ModelConfig, VisionEncoderConfig, AdapterConfig, CrossAttentionConfig]:
    return (
        ModelConfig(dim=64, n_layers=4, n_heads=4, vocab_size=256, max_seq_len=64),
        VisionEncoderConfig(type="random", feature_dim=feature_dim, num_tokens=num_image_tokens),
        AdapterConfig(),
        CrossAttentionConfig(max_text_len=32, cross_attention_every_n_layers=cadence),
    )


def _build_ca_tiny_wrapper(*args, **kwargs) -> VLMWrapper:
    mc, vc, ac, lc = _ca_configs(*args, **kwargs)
    return build_vlm_wrapper(mc, vc, ac, lc)


def _mot_configs(
    num_image_tokens: int = 8, feature_dim: int = 96
) -> tuple[ModelConfig, VisionEncoderConfig, AdapterConfig, MoTConfig]:
    return (
        ModelConfig(
            dim=64,
            n_layers=2,
            n_heads=4,
            vocab_size=256,
            max_seq_len=64,
            ffn_hidden_dim=128,
        ),
        VisionEncoderConfig(type="random", feature_dim=feature_dim, num_tokens=num_image_tokens),
        AdapterConfig(),
        MoTConfig(max_text_len=32),
    )


def _build_mot_tiny_wrapper(num_image_tokens: int = 8, feature_dim: int = 96) -> VLMWrapper:
    mc, vc, ac, lc = _mot_configs(num_image_tokens, feature_dim)
    return build_vlm_wrapper(mc, vc, ac, lc)


class TestModalityStrategies:
    def test_joint_decoder_strategy_fills_prefix_and_slice(self):
        wrapper = _build_tiny_wrapper(num_image_tokens=8)
        strategy = JointDecoderStrategy()
        pixel_values = torch.randn(1, 3, 64, 64)
        input_ids = torch.randint(0, 256, (1, 16))
        ctx = strategy.prepare(wrapper, pixel_values, input_ids)
        assert ctx.prefix_embeds is not None
        assert ctx.prefix_embeds.shape == (1, 8, 64)  # (B, N, dim)
        assert ctx.output_slice == slice(8, None)
        # CA-specific fields not set
        assert ctx.image_features is None
        assert ctx.image_mask is None
        assert ctx.inputs_embeds is None
        # MoT-specific field not set
        assert ctx.modality_ids is None

    def test_mot_strategy_fills_prefix_slice_and_modality_ids(self):
        """MoT strategy mirrors JD's prefix+slice setup AND adds modality_ids."""
        wrapper = _build_mot_tiny_wrapper(num_image_tokens=8)
        strategy = MoTStrategy()
        pixel_values = torch.randn(1, 3, 64, 64)
        input_ids = torch.randint(0, 256, (1, 16))
        ctx = strategy.prepare(wrapper, pixel_values, input_ids)
        assert ctx.prefix_embeds is not None
        assert ctx.prefix_embeds.shape == (1, 8, 64)
        assert ctx.output_slice == slice(8, None)
        assert ctx.modality_ids is not None
        assert ctx.modality_ids.shape == (1, 8 + 16)
        assert ctx.modality_ids.dtype == torch.long
        # Position-based: 0 for image positions, 1 for text positions.
        assert (ctx.modality_ids[:, :8] == 0).all()
        assert (ctx.modality_ids[:, 8:] == 1).all()
        assert ctx.inputs_embeds is None

    @pytest.mark.parametrize("t_text", [4, 16, 128])
    def test_mot_modality_ids_shape_dtype_device(self, t_text: int):
        wrapper = _build_mot_tiny_wrapper(num_image_tokens=8).to(DEVICE)
        strategy = MoTStrategy()
        pixel_values = torch.randn(2, 3, 64, 64, device=DEVICE)
        input_ids = torch.randint(0, 256, (2, t_text), device=DEVICE)
        ctx = strategy.prepare(wrapper, pixel_values, input_ids)
        assert ctx.modality_ids is not None
        assert ctx.modality_ids.shape == (2, 8 + t_text)
        assert ctx.modality_ids.dtype == torch.long
        assert ctx.modality_ids.device == input_ids.device

    def test_cross_attention_strategy_fills_image_features(self):
        wrapper = _build_ca_tiny_wrapper(num_image_tokens=8)
        strategy = CrossAttentionStrategy()
        pixel_values = torch.randn(1, 3, 64, 64)
        input_ids = torch.randint(0, 256, (1, 16))
        ctx = strategy.prepare(wrapper, pixel_values, input_ids)
        assert ctx.image_features is not None
        assert ctx.image_features.shape == (1, 8, 64)
        assert ctx.image_mask is None
        # JD-specific fields not set
        assert ctx.prefix_embeds is None
        assert ctx.output_slice is None
        assert ctx.inputs_embeds is None

    def test_strategy_num_image_tokens_arch_specific(self):
        jd_wrapper = _build_tiny_wrapper(num_image_tokens=12)
        ca_wrapper = _build_ca_tiny_wrapper(num_image_tokens=12)
        # JD: extends the residual stream by num_image_tokens.
        assert jd_wrapper.num_image_tokens == 12
        # CA: residual stream is text-only, so no extension.
        assert ca_wrapper.num_image_tokens == 0


class TestVLMWrapperDispatch:
    def test_build_modality_strategy_joint_decoder(self):
        cfg = JointDecoderConfig()
        strategy = build_modality_strategy(cfg)
        assert isinstance(strategy, JointDecoderStrategy)

    def test_build_modality_strategy_cross_attention(self):
        cfg = CrossAttentionConfig()
        strategy = build_modality_strategy(cfg)
        assert isinstance(strategy, CrossAttentionStrategy)

    def test_build_modality_strategy_mot(self):
        cfg = MoTConfig()
        strategy = build_modality_strategy(cfg)
        assert isinstance(strategy, MoTStrategy)

    def test_dispatch_no_isinstance_ladder(self):
        """Sanity check: build_modality_strategy is a pure registry
        lookup. Adding a fake arch via decorator works without editing
        the build function."""
        from kempnerforge.config.registry import registry

        @registry.register_modality_strategy("dispatch_smoke_test_arch")
        class _Smoke:
            def prepare(self, wrapper, pixel_values, input_ids):  # noqa: ARG002
                return ModalityContext()

            def num_image_tokens(self, wrapper):  # noqa: ARG002
                return 0

        # Make a mock vlm config object with the right arch attribute
        class _Cfg:
            arch = "dispatch_smoke_test_arch"

        strategy = build_modality_strategy(_Cfg())  # type: ignore[arg-type]
        assert isinstance(strategy, _Smoke)

    def test_jd_wrapper_uses_jd_strategy(self):
        wrapper = _build_tiny_wrapper()
        assert isinstance(wrapper.strategy, JointDecoderStrategy)

    def test_ca_wrapper_uses_ca_strategy(self):
        wrapper = _build_ca_tiny_wrapper()
        assert isinstance(wrapper.strategy, CrossAttentionStrategy)

    def test_mot_wrapper_uses_mot_strategy(self):
        wrapper = _build_mot_tiny_wrapper()
        assert isinstance(wrapper.strategy, MoTStrategy)

    def test_mot_forward_logits_text_only_shape(self):
        """MoT VLMWrapper forward returns text-only logits (output_slice
        trims the image prefix off the head)."""
        wrapper = _build_mot_tiny_wrapper(num_image_tokens=8).to(DEVICE)
        pixels = torch.randn(2, 3, 16, 16, device=DEVICE)
        input_ids = torch.randint(0, 256, (2, 20), device=DEVICE)
        labels = torch.full((2, 20), -100, dtype=torch.long, device=DEVICE)
        logits, labels_out = wrapper(pixels, input_ids, labels)
        assert logits.shape == (2, 20, 256)
        assert labels_out is labels

    def test_strategy_not_in_module_tree(self):
        """Strategy is a plain Python object, not an nn.Module. Verify
        it is not registered in _modules and does not appear in
        children() / state_dict() / named_parameters().
        """
        wrapper = _build_tiny_wrapper()
        # Strategy is reachable as a plain attribute.
        assert wrapper.strategy is not None
        # But not as a submodule.
        assert wrapper.strategy not in list(wrapper.children())
        assert "strategy" not in dict(wrapper.named_modules())
        # State dict carries no strategy entries.
        for k in wrapper.state_dict():
            assert "strategy" not in k

    def test_jd_forward_logits_text_only_shape(self):
        """JD forward returns logits with shape (B, T, V) — no image positions."""
        wrapper = _build_tiny_wrapper(num_image_tokens=8).to(DEVICE).eval()
        pixel_values = torch.randn(1, 3, 64, 64, device=DEVICE)
        input_ids = torch.randint(0, 256, (1, 16), device=DEVICE)
        with torch.no_grad():
            logits, _ = wrapper(pixel_values, input_ids)
        assert logits.shape == (1, 16, 256)

    def test_ca_forward_logits_text_only_shape(self):
        """CA forward also returns logits with shape (B, T, V): the
        residual stream is text-only, so no extra image positions."""
        wrapper = _build_ca_tiny_wrapper(num_image_tokens=8).to(DEVICE).eval()
        pixel_values = torch.randn(1, 3, 64, 64, device=DEVICE)
        input_ids = torch.randint(0, 256, (1, 16), device=DEVICE)
        with torch.no_grad():
            logits, _ = wrapper(pixel_values, input_ids)
        assert logits.shape == (1, 16, 256)


# ---------------------------------------------------------------------------
# Pooling connector (Phase 1): a pooling adapter reduces the visual-token
# count between the encoder and the LLM. The whole VLM path must use the
# adapter's output count (not encoder.num_tokens) for the prefix length,
# output_slice, modality_ids, and MoT's positional split.
# ---------------------------------------------------------------------------


def _build_pooled_jd_wrapper(adapter_type: str = "avgpool", pool_window: int = 2) -> VLMWrapper:
    # Encoder emits a 4x4 grid (16 tokens); a 2x2 pool -> 4 visual tokens.
    mc = ModelConfig(dim=64, n_layers=2, n_heads=4, vocab_size=256, max_seq_len=64)
    vc = VisionEncoderConfig(type="random", feature_dim=96, num_tokens=16)
    ac = AdapterConfig(type=adapter_type, pool_window=pool_window)
    lc = JointDecoderConfig(max_text_len=32)
    return build_vlm_wrapper(mc, vc, ac, lc)


def _build_pooled_mot_wrapper(pool_window: int = 2) -> VLMWrapper:
    mc = ModelConfig(
        dim=64, n_layers=2, n_heads=4, vocab_size=256, max_seq_len=64, ffn_hidden_dim=128
    )
    vc = VisionEncoderConfig(type="random", feature_dim=96, num_tokens=16)
    ac = AdapterConfig(type="avgpool", pool_window=pool_window)
    lc = MoTConfig(max_text_len=32)
    return build_vlm_wrapper(mc, vc, ac, lc)


class TestPoolingConnector:
    @pytest.mark.parametrize("adapter_type", ["avgpool", "attentional_pool"])
    def test_num_image_tokens_is_pooled_count(self, adapter_type):
        # 16 patch tokens, 2x2 pool -> 4 visual tokens.
        wrapper = _build_pooled_jd_wrapper(adapter_type, pool_window=2)
        assert wrapper.num_image_tokens == 4

    @pytest.mark.parametrize("adapter_type", ["avgpool", "attentional_pool"])
    def test_jd_prefix_length_is_pooled(self, adapter_type):
        wrapper = _build_pooled_jd_wrapper(adapter_type, pool_window=2)
        strategy = JointDecoderStrategy()
        pixels = torch.randn(2, 3, 16, 16)
        input_ids = torch.randint(0, 256, (2, 12))
        ctx = strategy.prepare(wrapper, pixels, input_ids)
        assert ctx.prefix_embeds is not None
        assert ctx.prefix_embeds.shape == (2, 4, 64)  # pooled prefix, model dim
        assert ctx.output_slice == slice(4, None)

    @pytest.mark.parametrize("adapter_type", ["avgpool", "attentional_pool"])
    def test_jd_forward_trims_pooled_prefix(self, adapter_type):
        wrapper = _build_pooled_jd_wrapper(adapter_type, pool_window=2).to(DEVICE).eval()
        pixels = torch.randn(2, 3, 16, 16, device=DEVICE)
        input_ids = torch.randint(0, 256, (2, 20), device=DEVICE)
        with torch.no_grad():
            logits, _ = wrapper(pixels, input_ids)
        # output_slice trims the 4 pooled positions -> logits cover text only.
        assert logits.shape == (2, 20, 256)

    def test_mot_split_uses_pooled_count(self):
        """MoT modality_ids length and the Transformer's positional split
        both key off the pooled count; a mismatch would crash the forward.
        """
        wrapper = _build_pooled_mot_wrapper(pool_window=2).to(DEVICE)
        strategy = MoTStrategy()
        pixels = torch.randn(2, 3, 16, 16, device=DEVICE)
        input_ids = torch.randint(0, 256, (2, 16), device=DEVICE)
        ctx = strategy.prepare(wrapper, pixels, input_ids)
        assert ctx.modality_ids is not None
        assert ctx.modality_ids.shape == (2, 4 + 16)  # pooled prefix + text
        assert (ctx.modality_ids[:, :4] == 0).all()
        assert (ctx.modality_ids[:, 4:] == 1).all()
        # End-to-end forward exercises the build-time _mot_n_image split,
        # which must equal the runtime pooled prefix length (4).
        labels = torch.full((2, 16), -100, dtype=torch.long, device=DEVICE)
        logits, _ = wrapper(pixels, input_ids, labels)
        assert logits.shape == (2, 16, 256)

    def test_projection_adapter_keeps_token_count(self):
        """Regression: a non-pooling adapter must leave num_image_tokens ==
        encoder.num_tokens (the image path stays bit-for-bit unchanged)."""
        wrapper = _build_pooled_jd_wrapper("mlp_2layer", pool_window=2)
        assert wrapper.num_image_tokens == 16  # no pooling -> unchanged


# ---------------------------------------------------------------------------
# Video forward (Phase 3): the wrapper consumes a (B, F, 3, H, W) clip batch.
# _project_visual_features folds the frame axis through the encoder+pooler to
# (B, F*P', dim); the static visual-token count (frames_per_clip * per-frame)
# drives the residual budget + MoT's split and must equal the runtime prefix.
# ---------------------------------------------------------------------------


def _video_wrapper(vlm_cfg, frames: int = 4, *, ffn_hidden_dim: int | None = None) -> VLMWrapper:
    # Encoder: 4x4 patch grid (16 tokens); avgpool 2x2 -> 4 tokens/frame.
    # frames=4 -> 4*4 = 16 visual tokens in the residual prefix.
    mc_kwargs: dict[str, int] = {
        "dim": 64,
        "n_layers": 2,
        "n_heads": 4,
        "vocab_size": 256,
        "max_seq_len": 64,
    }
    if ffn_hidden_dim is not None:
        mc_kwargs["ffn_hidden_dim"] = ffn_hidden_dim
    mc = ModelConfig(**mc_kwargs)
    vc = VisionEncoderConfig(type="random", feature_dim=96, num_tokens=16)
    ac = AdapterConfig(type="avgpool", pool_window=2)
    return build_vlm_wrapper(mc, vc, ac, vlm_cfg, frames_per_clip=frames)


class TestVideoForward:
    def test_num_image_tokens_is_frames_times_per_frame(self):
        # 4 frames * (16 patches -> 2x2 pool -> 4) = 16 visual tokens.
        wrapper = _video_wrapper(JointDecoderConfig(max_text_len=8), frames=4)
        assert wrapper.num_image_tokens == 16

    def test_projector_folds_frame_axis(self):
        wrapper = _video_wrapper(JointDecoderConfig(max_text_len=8), frames=4)
        ctx = JointDecoderStrategy().prepare(
            wrapper, torch.randn(2, 4, 3, 16, 16), torch.randint(0, 256, (2, 6))
        )
        assert ctx.prefix_embeds is not None
        assert ctx.prefix_embeds.shape == (2, 16, 64)  # (B, F*P', dim)
        assert ctx.output_slice == slice(16, None)

    def test_static_count_matches_runtime_prefix(self):
        """MoT's positional split uses the build-time count; it must equal the
        runtime prefix length (frames * per-frame)."""
        wrapper = _video_wrapper(JointDecoderConfig(max_text_len=8), frames=4)
        ctx = JointDecoderStrategy().prepare(
            wrapper, torch.randn(1, 4, 3, 16, 16), torch.randint(0, 256, (1, 6))
        )
        assert ctx.prefix_embeds is not None
        assert ctx.prefix_embeds.shape[1] == wrapper.num_image_tokens == 16

    @pytest.mark.parametrize("arch", ["joint_decoder", "cross_attention", "mot", "moma"])
    def test_video_forward_all_archs(self, arch):
        ffn = 128 if arch in ("mot", "moma") else None
        if arch == "joint_decoder":
            vlm_cfg: VLMConfig = JointDecoderConfig(max_text_len=8)
        elif arch == "cross_attention":
            vlm_cfg = CrossAttentionConfig(max_text_len=8, cross_attention_every_n_layers=2)
        elif arch == "mot":
            vlm_cfg = MoTConfig(max_text_len=8)
        else:
            vlm_cfg = MoMaConfig(max_text_len=8)
        wrapper = _video_wrapper(vlm_cfg, frames=4, ffn_hidden_dim=ffn).to(DEVICE)
        pixels = torch.randn(2, 4, 3, 16, 16, device=DEVICE)
        input_ids = torch.randint(0, 256, (2, 6), device=DEVICE)
        labels = torch.full((2, 6), -100, dtype=torch.long, device=DEVICE)
        logits, _ = wrapper(pixels, input_ids, labels)
        # output_slice trims the F*P' visual prefix -> logits cover text only.
        assert logits.shape == (2, 6, 256)

    def test_video_forward_backward_grads(self):
        wrapper = _video_wrapper(JointDecoderConfig(max_text_len=8), frames=4).to(DEVICE)
        pixels = torch.randn(1, 4, 3, 16, 16, device=DEVICE)
        input_ids = torch.randint(0, 256, (1, 6), device=DEVICE)
        logits, _ = wrapper(pixels, input_ids)
        logits.sum().backward()
        for p in wrapper.adapter.parameters():
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()

    def test_image_path_unchanged_with_4d(self):
        """frames_per_clip=1 + a 4D image batch still works (image path intact)."""
        wrapper = _video_wrapper(JointDecoderConfig(max_text_len=8), frames=1).to(DEVICE).eval()
        pixels = torch.randn(2, 3, 16, 16, device=DEVICE)  # 4D single-image batch
        input_ids = torch.randint(0, 256, (2, 6), device=DEVICE)
        with torch.no_grad():
            logits, _ = wrapper(pixels, input_ids)
        assert logits.shape == (2, 6, 256)
        assert wrapper.num_image_tokens == 4  # per-frame pooled count, frames=1

    def test_video_forward_dtype_mismatch_cast(self):
        """Encoder fp32 + bf16 adapter/transformer: the cast inside the visual
        projector lets the 5D video path run (covers the dtype-cast branch)."""
        wrapper = _video_wrapper(JointDecoderConfig(max_text_len=8), frames=4).to(DEVICE)
        wrapper.adapter.to(torch.bfloat16)
        wrapper.transformer.to(torch.bfloat16)
        # vision_encoder (RandomVisionEncoder) stays fp32
        pixels = torch.randn(2, 4, 3, 16, 16, device=DEVICE)
        input_ids = torch.randint(0, 256, (2, 6), device=DEVICE)
        logits, _ = wrapper(pixels, input_ids)
        assert logits.dtype == torch.bfloat16
        assert logits.shape == (2, 6, 256)

    def test_frame_count_mismatch_raises(self):
        """A clip whose frame count != frames_per_clip is rejected at the
        projection boundary: the static MoT split and seq-len budget assume
        frames_per_clip, so a mismatch is a clear error, not a confusing one."""
        wrapper = _video_wrapper(JointDecoderConfig(max_text_len=8), frames=4).to(DEVICE)
        input_ids = torch.randint(0, 256, (2, 6), device=DEVICE)
        # 2-frame clip into a 4-frame wrapper.
        with pytest.raises(ValueError, match="frames-per-clip mismatch"):
            wrapper(torch.randn(2, 2, 3, 16, 16, device=DEVICE), input_ids)
        # 4D single-image batch into a video (frames_per_clip>1) wrapper.
        with pytest.raises(ValueError, match="frames-per-clip mismatch"):
            wrapper(torch.randn(2, 3, 16, 16, device=DEVICE), input_ids)


class TestFramePaddingMask:
    """frame_mask hides padded video frames from attention (and, for MoMa, from
    expert-choice routing), so real-token outputs are invariant to padded-frame
    content. The image (F=1) path is a no-op, and an all-padded (undecodable)
    clip stays finite via the NaN guard."""

    @staticmethod
    def _arch_wrapper(arch):
        ffn = 128 if arch in ("mot", "moma") else None
        cfgs = {
            "joint_decoder": JointDecoderConfig(max_text_len=8),
            "cross_attention": CrossAttentionConfig(
                max_text_len=8, cross_attention_every_n_layers=2
            ),
            "mot": MoTConfig(max_text_len=8),
            "moma": MoMaConfig(max_text_len=8),
        }
        w = _video_wrapper(cfgs[arch], frames=4, ffn_hidden_dim=ffn).to(DEVICE).eval()
        # Move off zero-init warm-start gating (e.g. MoT/CA zero-init o_proj) so
        # the image path actually contributes — otherwise the no-mask control is
        # vacuous (image gated off at init).
        with torch.no_grad():
            for p in w.parameters():
                p.add_(0.02 * torch.randn_like(p))
        return w

    def test_visual_token_mask_expands_per_frame(self):
        from kempnerforge.model.vlm import _visual_token_mask

        fm = torch.tensor([[True, False, True]])  # 3 frames
        out = _visual_token_mask(fm, num_visual_tokens=6)  # 2 tokens/frame
        assert out.tolist() == [[True, True, False, False, True, True]]
        assert _visual_token_mask(None, 6) is None

    @pytest.mark.parametrize("arch", ["joint_decoder", "cross_attention", "mot", "moma"])
    def test_masked_frames_do_not_affect_real_tokens(self, arch):
        torch.manual_seed(0)
        w = self._arch_wrapper(arch)
        ids = torch.randint(0, 256, (1, 6), device=DEVICE)
        pix = torch.randn(1, 4, 3, 16, 16, device=DEVICE)
        fm = torch.tensor([[True, True, False, False]], device=DEVICE)  # frames 2,3 padded
        pix2 = pix.clone()
        pix2[:, 2:] = torch.randn(1, 2, 3, 16, 16, device=DEVICE)  # corrupt the padded frames
        with torch.no_grad():
            masked_a, _ = w(pix, ids, frame_mask=fm)
            masked_b, _ = w(pix2, ids, frame_mask=fm)
            nomask_a, _ = w(pix, ids)
            nomask_b, _ = w(pix2, ids)
        assert torch.equal(masked_a, masked_b), f"{arch}: masked output depends on padded frames"
        assert not torch.equal(nomask_a, nomask_b), f"{arch}: control — pads should leak unmasked"

    def test_image_f1_mask_is_noop(self):
        torch.manual_seed(0)
        w = _video_wrapper(JointDecoderConfig(max_text_len=8), frames=1).to(DEVICE).eval()
        img = torch.randn(1, 3, 16, 16, device=DEVICE)
        ids = torch.randint(0, 256, (1, 6), device=DEVICE)
        with torch.no_grad():
            no_mask, _ = w(img, ids)
            all_true, _ = w(img, ids, frame_mask=torch.tensor([[True]], device=DEVICE))
        assert torch.equal(no_mask, all_true)

    @pytest.mark.parametrize("arch", ["joint_decoder", "cross_attention", "mot", "moma"])
    def test_undecodable_clip_stays_finite(self, arch):
        # An undecodable clip has frame_mask all-False; the NaN guard must keep
        # softmax finite (no NaN poisoning the batch loss).
        torch.manual_seed(0)
        w = self._arch_wrapper(arch)
        ids = torch.randint(0, 256, (2, 6), device=DEVICE)
        pix = torch.randn(2, 4, 3, 16, 16, device=DEVICE)
        fm = torch.tensor([[False, False, False, False], [True, True, True, True]], device=DEVICE)
        with torch.no_grad():
            logits, _ = w(pix, ids, frame_mask=fm)
        assert torch.isfinite(logits).all(), f"{arch}: NaN/inf with an all-padded clip"
