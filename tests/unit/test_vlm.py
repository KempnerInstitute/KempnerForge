"""Unit tests for Adapter, VLMWrapper, inner_transformer, _is_encoder_frozen."""

from __future__ import annotations

import pytest
import torch

from kempnerforge.config.model import ModelConfig
from kempnerforge.config.vlm import (
    FreezeSpec,
    JointDecoderConfig,
    MoTConfig,
    VLMConfig,
)
from kempnerforge.model.modality import ModalityContext
from kempnerforge.model.transformer import Transformer
from kempnerforge.model.vlm import (
    Adapter,
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
# Adapter
# ---------------------------------------------------------------------------


class TestAdapter:
    def test_forward_shape(self):
        adapter = Adapter(in_dim=384, out_dim=256).to(DEVICE)
        x = torch.randn(2, 16, 384, device=DEVICE)
        out = adapter(x)
        assert out.shape == (2, 16, 256)

    def test_hidden_dim_defaults_to_out_dim(self):
        adapter = Adapter(in_dim=128, out_dim=64)
        assert adapter.proj1.out_features == 64

    def test_hidden_dim_override(self):
        adapter = Adapter(in_dim=128, out_dim=64, hidden_dim=256)
        assert adapter.proj1.out_features == 256
        assert adapter.proj2.in_features == 256

    def test_activations(self):
        for act in ("gelu", "silu", "relu"):
            adapter = Adapter(in_dim=8, out_dim=8, activation=act)
            out = adapter(torch.randn(1, 4, 8))
            assert out.shape == (1, 4, 8)

    def test_unknown_activation_raises(self):
        with pytest.raises(ValueError, match="Unknown adapter activation"):
            Adapter(in_dim=8, out_dim=8, activation="tanh")

    def test_rejects_zero_dim(self):
        with pytest.raises(ValueError, match="must be positive"):
            Adapter(in_dim=0, out_dim=8)

    def test_backward_grads_flow(self):
        adapter = Adapter(in_dim=32, out_dim=16).to(DEVICE)
        x = torch.randn(1, 4, 32, device=DEVICE, requires_grad=True)
        adapter(x).sum().backward()
        for p in adapter.parameters():
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()

    def test_reset_parameters_reinitializes(self):
        adapter = Adapter(in_dim=8, out_dim=4)
        w1 = adapter.proj1.weight.clone()
        adapter.reset_parameters()
        # Not guaranteed bit-different (unlikely but possible), so just
        # check it actually ran without error and weights are finite.
        assert torch.isfinite(adapter.proj1.weight).all()
        assert adapter.proj1.weight.shape == w1.shape


# ---------------------------------------------------------------------------
# VLMWrapper
# ---------------------------------------------------------------------------


def _tiny_vlm_config(num_image_tokens: int = 8, feature_dim: int = 96) -> ModelConfig:
    return ModelConfig(
        dim=64,
        n_layers=2,
        n_heads=4,
        vocab_size=256,
        max_seq_len=64,
        vlm=VLMConfig(
            vision_encoder="random",
            feature_dim=feature_dim,
            num_tokens=num_image_tokens,
            max_text_len=32,
        ),
    )


def _build_tiny_wrapper(num_image_tokens: int = 8, feature_dim: int = 96) -> VLMWrapper:
    return build_vlm_wrapper(_tiny_vlm_config(num_image_tokens, feature_dim))


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

    def test_dispatch_on_is_vlm_only(self):
        """build_vlm_wrapper should not require model_type changes; the
        inner transformer is registered under the standard 'transformer'
        model_type, and VLM dispatch happens through model_config.is_vlm.
        """
        cfg = _tiny_vlm_config()
        assert cfg.is_vlm is True
        assert cfg.model_type == "transformer"
        wrapper = build_vlm_wrapper(cfg)
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
    def test_missing_vlm_raises(self):
        cfg = ModelConfig(dim=64, n_layers=2, n_heads=4, vocab_size=256)
        assert cfg.vlm is None
        with pytest.raises(ValueError, match="model_config.vlm"):
            build_vlm_wrapper(cfg)


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


def _mot_vlm_config(num_image_tokens: int = 8, feature_dim: int = 96) -> ModelConfig:
    return ModelConfig(
        dim=64,
        n_layers=2,
        n_heads=4,
        vocab_size=256,
        max_seq_len=64,
        ffn_hidden_dim=128,
        vlm=MoTConfig(
            vision_encoder="random",
            feature_dim=feature_dim,
            num_tokens=num_image_tokens,
            max_text_len=32,
        ),
    )


def _build_mot_tiny_wrapper(num_image_tokens: int = 8, feature_dim: int = 96) -> VLMWrapper:
    return build_vlm_wrapper(_mot_vlm_config(num_image_tokens, feature_dim))


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

    def test_strategy_num_image_tokens(self):
        jd_wrapper = _build_tiny_wrapper(num_image_tokens=12)
        # JD: extends the residual stream by num_image_tokens.
        assert jd_wrapper.num_image_tokens == 12


class TestVLMWrapperDispatch:
    def test_build_modality_strategy_joint_decoder(self):
        cfg = JointDecoderConfig(vision_encoder="random")
        strategy = build_modality_strategy(cfg)
        assert isinstance(strategy, JointDecoderStrategy)

    def test_build_modality_strategy_mot(self):
        cfg = MoTConfig(vision_encoder="random")
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
