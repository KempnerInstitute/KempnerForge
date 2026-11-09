"""Unit tests for VLMWrapper, inner_transformer, _is_encoder_frozen.

Adapter unit tests moved to ``tests/unit/test_adapter.py`` when ``Adapter``
was renamed to ``MLP2LayerAdapter`` and promoted to a registered component.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict

from kempnerforge.config.adapter import AdapterConfig
from kempnerforge.config.model import ModelConfig
from kempnerforge.config.registry import registry
from kempnerforge.config.vision import VisionEncoderConfig
from kempnerforge.config.vlm import (
    CrossAttentionConfig,
    FreezeSpec,
    JointDecoderConfig,
    MoMaConfig,
    MoTConfig,
    VLMConfig,
)
from kempnerforge.data.vlm_dataset import VLMCollator, pack_vlm_samples
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
    packed_tokens_per_image,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        mc = ModelConfig(dim=64, n_layers=2, n_heads=4, vocab_size=256, max_seq_len=32)
        vc = VisionEncoderConfig(type="random", num_tokens=0)
        ac = AdapterConfig()
        lc = VLMConfig(max_text_len=32)
        with pytest.raises(ValueError) as exc_info:
            build_vlm_wrapper(mc, vc, ac, lc)
        msg = str(exc_info.value)
        assert "max_seq_len" in msg
        assert "encoder.num_tokens" in msg
        assert "max_text_len" in msg
        assert "32" in msg
        assert "16" in msg

    def test_build_time_check_skipped_for_cross_attention(self):
        """CA does not extend the residual stream — the build-time check
        must allow max_seq_len < encoder.num_tokens + max_text_len so long
        as max_seq_len >= max_text_len. Without this carve-out, CA configs
        with num_tokens=0 (sentinel) would incorrectly raise.
        """
        mc = ModelConfig(dim=64, n_layers=2, n_heads=4, vocab_size=256, max_seq_len=32)
        vc = VisionEncoderConfig(type="random", num_tokens=0)
        ac = AdapterConfig()
        lc = CrossAttentionConfig(max_text_len=32)
        wrapper = build_vlm_wrapper(mc, vc, ac, lc)
        assert isinstance(wrapper, VLMWrapper)
        assert wrapper.num_image_tokens == 0


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
        assert ctx.prefix_embeds.shape == (1, 8, 64)
        assert ctx.output_slice == slice(8, None)
        assert ctx.image_features is None
        assert ctx.image_mask is None
        assert ctx.inputs_embeds is None
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
        assert ctx.prefix_embeds is None
        assert ctx.output_slice is None
        assert ctx.inputs_embeds is None

    def test_strategy_num_image_tokens_arch_specific(self):
        jd_wrapper = _build_tiny_wrapper(num_image_tokens=12)
        ca_wrapper = _build_ca_tiny_wrapper(num_image_tokens=12)
        assert jd_wrapper.num_image_tokens == 12
        assert ca_wrapper.num_image_tokens == 0


def _packed_doc(text_len: int, seed: int, images: int = 1) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 256, (text_len,), generator=gen)
    labels = torch.full((text_len,), -100, dtype=torch.long)
    labels[:-1] = ids[1:]
    pixels = (
        torch.randn(3, 16, 16, generator=gen)
        if images == 1
        else torch.randn(images, 3, 16, 16, generator=gen)
    )
    return {
        "pixel_values": pixels,
        "input_ids": ids,
        "labels": labels,
        "text_len": torch.tensor(text_len),
    }


class TestPackedForward:
    def _forward(self, wrapper, batch):
        return wrapper(
            batch["pixel_values"],
            batch["input_ids"],
            batch["labels"],
            doc_ids=batch["doc_ids"],
            position_ids=batch["position_ids"],
            image_rows=batch["image_rows"],
            image_starts=batch["image_starts"],
            image_tokens=int(batch["image_tokens"]),
        )

    def test_logits_cover_the_packed_rows(self):
        wrapper = _build_tiny_wrapper(num_image_tokens=8)
        docs = [_packed_doc(10, 1), _packed_doc(12, 2), _packed_doc(20, 3)]
        batch = pack_vlm_samples(docs, seq_len=40, tokens_per_image=8, pad_id=0)
        assert batch["input_ids"].shape == (2, 40)
        logits, labels = self._forward(wrapper, batch)
        assert logits.shape == (2, 40, 256)
        assert torch.equal(labels, batch["labels"])
        assert torch.isfinite(logits).all()

    def test_a_packed_document_matches_the_prefix_path(self):
        """A document's text gets the prefix route's logits wherever first-fit placed it."""
        wrapper = _build_tiny_wrapper(num_image_tokens=8).eval()
        doc = _packed_doc(12, 4)
        with torch.no_grad():
            prefix, _ = wrapper(doc["pixel_values"].unsqueeze(0), doc["input_ids"].unsqueeze(0))
            alone, _ = self._forward(
                wrapper, pack_vlm_samples([doc], seq_len=32, tokens_per_image=8, pad_id=0)
            )
            after_another, _ = self._forward(
                wrapper,
                pack_vlm_samples(
                    [_packed_doc(6, 5), doc], seq_len=48, tokens_per_image=8, pad_id=0
                ),
            )
        assert torch.allclose(alone[0, 8:20], prefix[0], atol=1e-5)
        assert torch.allclose(after_another[0, 22:34], prefix[0], atol=1e-5)

    def test_documents_do_not_see_each_other(self):
        """Changing the first document leaves the second document's logits unchanged."""
        wrapper = _build_tiny_wrapper(num_image_tokens=8).eval()
        second = _packed_doc(10, 6)
        with torch.no_grad():
            a, _ = self._forward(
                wrapper,
                pack_vlm_samples(
                    [_packed_doc(6, 5), second], seq_len=40, tokens_per_image=8, pad_id=0
                ),
            )
            b, _ = self._forward(
                wrapper,
                pack_vlm_samples(
                    [_packed_doc(6, 7), second], seq_len=40, tokens_per_image=8, pad_id=0
                ),
            )
        assert torch.allclose(a[0, 14:32], b[0, 14:32], atol=1e-5)
        assert not torch.allclose(a[0, :14], b[0, :14])

    def test_text_only_documents_match_the_plain_transformer(self):
        """With no images the rows are ordinary packed text, so the backbone alone decides."""
        wrapper = _build_tiny_wrapper(num_image_tokens=8).eval()
        docs = [_packed_doc(10, 1, images=0), _packed_doc(12, 2, images=0)]
        batch = pack_vlm_samples(
            docs, seq_len=32, tokens_per_image=8, pad_id=0, allow_text_only=True
        )
        assert batch["pixel_values"].shape == (0, 3, 16, 16)
        with torch.no_grad():
            packed, _ = self._forward(wrapper, batch)
            plain = wrapper.transformer(tokens=batch["input_ids"], doc_ids=batch["doc_ids"])
        assert torch.allclose(packed, plain, atol=1e-5)

    @staticmethod
    def _text_only_batch(tokens_per_image: int = 8):
        return pack_vlm_samples(
            [_packed_doc(10, 3, images=0)],
            seq_len=32,
            tokens_per_image=tokens_per_image,
            pad_id=0,
            allow_text_only=True,
        )

    def test_a_text_only_batch_runs_a_blank_image_through_the_encoder_and_adapter(self):
        """Their parameters get zero gradients, so FSDP2's collectives run on every rank."""
        wrapper = _build_tiny_wrapper(num_image_tokens=8)
        logits, _ = self._forward(wrapper, self._text_only_batch())
        logits.sum().backward()
        visual = [
            p
            for module in (wrapper.vision_encoder, wrapper.adapter)
            for p in module.parameters()
            if p.requires_grad
        ]
        assert visual
        assert all(p.grad is not None and torch.count_nonzero(p.grad) == 0 for p in visual)
        assert all(p.grad is not None for p in wrapper.transformer.token_embedding.parameters())

    def test_a_text_only_batch_still_checks_the_image_token_count(self):
        wrapper = _build_tiny_wrapper(num_image_tokens=8)
        with pytest.raises(ValueError, match="emits 8 tokens per image but .* image_tokens=4"):
            self._forward(wrapper, self._text_only_batch(tokens_per_image=4))

    def test_text_and_image_documents_share_a_batch(self):
        wrapper = _build_tiny_wrapper(num_image_tokens=8).eval()
        text_doc = _packed_doc(10, 4, images=0)
        image_doc = _packed_doc(12, 5)
        batch = pack_vlm_samples(
            [image_doc, text_doc], seq_len=40, tokens_per_image=8, pad_id=0, allow_text_only=True
        )
        assert batch["pixel_values"].shape == (1, 3, 16, 16)
        with torch.no_grad():
            packed, _ = self._forward(wrapper, batch)
            alone, _ = self._forward(
                wrapper,
                pack_vlm_samples(
                    [text_doc], seq_len=40, tokens_per_image=8, pad_id=0, allow_text_only=True
                ),
            )
        assert torch.allclose(packed[0, 20:30], alone[0, :10], atol=1e-5)

    def test_backward_reaches_encoder_adapter_and_backbone(self):
        wrapper = _build_tiny_wrapper(num_image_tokens=8)
        batch = pack_vlm_samples([_packed_doc(10, 8)], seq_len=32, tokens_per_image=8, pad_id=0)
        logits, _ = self._forward(wrapper, batch)
        logits.sum().backward()
        assert next(wrapper.adapter.parameters()).grad is not None
        assert all(p.grad is not None for p in wrapper.transformer.token_embedding.parameters())

    def test_wrong_token_count_raises(self):
        wrapper = _build_tiny_wrapper(num_image_tokens=8)
        batch = pack_vlm_samples([_packed_doc(10, 9)], seq_len=32, tokens_per_image=4, pad_id=0)
        with pytest.raises(ValueError, match="emits 8 tokens per image but .* image_tokens=4"):
            self._forward(wrapper, batch)

    def test_placement_needs_both_tensors_and_no_frame_mask(self):
        wrapper = _build_tiny_wrapper(num_image_tokens=8)
        batch = pack_vlm_samples([_packed_doc(10, 9)], seq_len=32, tokens_per_image=8, pad_id=0)
        with pytest.raises(ValueError, match="given together"):
            wrapper(batch["pixel_values"], batch["input_ids"], image_rows=batch["image_rows"])
        with pytest.raises(ValueError, match="no frame_mask"):
            wrapper(
                batch["pixel_values"],
                batch["input_ids"],
                frame_mask=torch.ones(1, 1, dtype=torch.bool),
                image_rows=batch["image_rows"],
                image_starts=batch["image_starts"],
            )

    def test_packed_pixels_require_a_flat_image_batch(self) -> None:
        wrapper = _build_tiny_wrapper(num_image_tokens=8)
        batch = pack_vlm_samples([_packed_doc(10, 9)], seq_len=32, tokens_per_image=8, pad_id=0)
        batch["pixel_values"] = batch["pixel_values"].unsqueeze(0)
        with pytest.raises(ValueError, match=r"pixel_values must be \(N, 3, H, W\)"):
            self._forward(wrapper, batch)

    @pytest.mark.parametrize("mismatch", ["starts_length", "placement_rank"])
    def test_packed_placement_matches_the_image_batch(self, mismatch: str) -> None:
        wrapper = _build_tiny_wrapper(num_image_tokens=8)
        batch = pack_vlm_samples([_packed_doc(10, 9)], seq_len=32, tokens_per_image=8, pad_id=0)
        if mismatch == "starts_length":
            batch["image_starts"] = torch.zeros(2, dtype=torch.long)
        else:
            batch["image_rows"] = batch["image_rows"].unsqueeze(0)
            batch["image_starts"] = batch["image_starts"].unsqueeze(0)
        with pytest.raises(ValueError, match=r"image_rows and image_starts must be \(1,\)"):
            self._forward(wrapper, batch)

    @pytest.mark.parametrize("arch", ["cross_attention", "mot", "moma"])
    def test_other_arches_reject_packed_batches(self, arch):
        strategy = build_modality_strategy(VLMConfig.for_arch(arch))
        batch = pack_vlm_samples([_packed_doc(10, 9)], seq_len=32, tokens_per_image=8, pad_id=0)
        with pytest.raises(NotImplementedError, match="joint_decoder"):
            strategy.prepare(
                _build_tiny_wrapper(),
                batch["pixel_values"],
                batch["input_ids"],
                image_rows=batch["image_rows"],
                image_starts=batch["image_starts"],
            )


class TestTextOnlyPrefixRows:
    """On the prefix route a text-only sample's blank image is hidden by ``frame_mask``."""

    def test_the_text_logits_ignore_the_blank_image(self):
        torch.manual_seed(0)
        wrapper = _build_tiny_wrapper(num_image_tokens=8).eval()
        collator = VLMCollator(pad_id=0, max_text_len=12, allow_text_only=True)
        text_only = {
            "pixel_values": torch.zeros(0, 3, 16, 16),
            "input_ids": torch.arange(1, 13),
            "labels": torch.arange(2, 14),
        }
        batch = collator([text_only])
        noise = batch["pixel_values"] + torch.randn(1, 3, 16, 16)
        with torch.no_grad():
            blank, _ = wrapper(
                batch["pixel_values"], batch["input_ids"], frame_mask=batch["frame_mask"]
            )
            noisy, _ = wrapper(noise, batch["input_ids"], frame_mask=batch["frame_mask"])
            seen, _ = wrapper(noise, batch["input_ids"])
        assert torch.equal(blank, noisy)
        assert not torch.equal(blank, seen)
        assert torch.isfinite(blank).all()


class TestPackedTokensPerImage:
    def test_reads_the_adapter_output_off_the_wrapper(self):
        wrapper = _build_tiny_wrapper(num_image_tokens=8)
        assert packed_tokens_per_image(wrapper) == 8
        assert packed_tokens_per_image(torch.compile(wrapper)) == 8

    def test_rejects_a_bare_module(self):
        with pytest.raises(TypeError, match="needs a VLMWrapper"):
            packed_tokens_per_image(torch.nn.Linear(2, 2))


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
        assert wrapper.strategy is not None
        assert wrapper.strategy not in list(wrapper.children())
        assert "strategy" not in dict(wrapper.named_modules())
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


def _build_pooled_jd_wrapper(adapter_type: str = "avgpool", pool_window: int = 2) -> VLMWrapper:
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
        assert ctx.prefix_embeds.shape == (2, 4, 64)
        assert ctx.output_slice == slice(4, None)

    @pytest.mark.parametrize("adapter_type", ["avgpool", "attentional_pool"])
    def test_jd_forward_trims_pooled_prefix(self, adapter_type):
        wrapper = _build_pooled_jd_wrapper(adapter_type, pool_window=2).to(DEVICE).eval()
        pixels = torch.randn(2, 3, 16, 16, device=DEVICE)
        input_ids = torch.randint(0, 256, (2, 20), device=DEVICE)
        with torch.no_grad():
            logits, _ = wrapper(pixels, input_ids)
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
        assert ctx.modality_ids.shape == (2, 4 + 16)
        assert (ctx.modality_ids[:, :4] == 0).all()
        assert (ctx.modality_ids[:, 4:] == 1).all()
        labels = torch.full((2, 16), -100, dtype=torch.long, device=DEVICE)
        logits, _ = wrapper(pixels, input_ids, labels)
        assert logits.shape == (2, 16, 256)

    def test_projection_adapter_keeps_token_count(self):
        """Regression: a non-pooling adapter must leave num_image_tokens ==
        encoder.num_tokens (the image path stays bit-for-bit unchanged)."""
        wrapper = _build_pooled_jd_wrapper("mlp_2layer", pool_window=2)
        assert wrapper.num_image_tokens == 16


def _video_wrapper(vlm_cfg, frames: int = 4, *, ffn_hidden_dim: int | None = None) -> VLMWrapper:
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
        wrapper = _video_wrapper(JointDecoderConfig(max_text_len=8), frames=4)
        assert wrapper.num_image_tokens == 16

    def test_projector_folds_frame_axis(self):
        wrapper = _video_wrapper(JointDecoderConfig(max_text_len=8), frames=4)
        ctx = JointDecoderStrategy().prepare(
            wrapper, torch.randn(2, 4, 3, 16, 16), torch.randint(0, 256, (2, 6))
        )
        assert ctx.prefix_embeds is not None
        assert ctx.prefix_embeds.shape == (2, 16, 64)
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
        pixels = torch.randn(2, 3, 16, 16, device=DEVICE)
        input_ids = torch.randint(0, 256, (2, 6), device=DEVICE)
        with torch.no_grad():
            logits, _ = wrapper(pixels, input_ids)
        assert logits.shape == (2, 6, 256)
        assert wrapper.num_image_tokens == 4

    def test_video_forward_dtype_mismatch_cast(self):
        """Encoder fp32 + bf16 adapter/transformer: the cast inside the visual
        projector lets the 5D video path run (covers the dtype-cast branch)."""
        wrapper = _video_wrapper(JointDecoderConfig(max_text_len=8), frames=4).to(DEVICE)
        wrapper.adapter.to(torch.bfloat16)
        wrapper.transformer.to(torch.bfloat16)
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
        with pytest.raises(ValueError, match="frames-per-clip mismatch"):
            wrapper(torch.randn(2, 2, 3, 16, 16, device=DEVICE), input_ids)
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
        with torch.no_grad():
            for p in w.parameters():
                p.add_(0.02 * torch.randn_like(p))
        return w

    def test_visual_token_mask_expands_per_frame(self):
        from kempnerforge.model.vlm import _visual_token_mask

        fm = torch.tensor([[True, False, True]])
        out = _visual_token_mask(fm, num_visual_tokens=6)
        assert out.tolist() == [[True, True, False, False, True, True]]
        assert _visual_token_mask(None, 6) is None
        with pytest.raises(ValueError, match="multiple of num_frames"):
            _visual_token_mask(fm, num_visual_tokens=7)

    @pytest.mark.parametrize("arch", ["joint_decoder", "cross_attention", "mot", "moma"])
    def test_masked_frames_do_not_affect_real_tokens(self, arch):
        torch.manual_seed(0)
        w = self._arch_wrapper(arch)
        ids = torch.randint(0, 256, (1, 6), device=DEVICE)
        pix = torch.randn(1, 4, 3, 16, 16, device=DEVICE)
        fm = torch.tensor([[True, True, False, False]], device=DEVICE)
        pix2 = pix.clone()
        pix2[:, 2:] = torch.randn(1, 2, 3, 16, 16, device=DEVICE)
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
        torch.manual_seed(0)
        w = self._arch_wrapper(arch)
        ids = torch.randint(0, 256, (2, 6), device=DEVICE)
        pix = torch.randn(2, 4, 3, 16, 16, device=DEVICE)
        fm = torch.tensor([[False, False, False, False], [True, True, True, True]], device=DEVICE)
        with torch.no_grad():
            logits, _ = w(pix, ids, frame_mask=fm)
        assert torch.isfinite(logits).all(), f"{arch}: NaN/inf with an all-padded clip"


_REGISTERED_NORMS = sorted(registry.list("norm"))


def _pre_norm_wrapper(pre_norm: str = "") -> VLMWrapper:
    mc, vc, _, lc = _tiny_configs()
    return build_vlm_wrapper(mc, vc, AdapterConfig(pre_norm=pre_norm), lc)


class TestVLMAdapterPreNorm:
    def test_off_adds_no_adapter_state(self):
        assert [k for k in _pre_norm_wrapper("").state_dict() if k.startswith("adapter.")] == [
            "adapter.proj1.weight",
            "adapter.proj1.bias",
            "adapter.proj2.weight",
            "adapter.proj2.bias",
        ]

    @pytest.mark.parametrize("norm", _REGISTERED_NORMS)
    def test_on_adds_only_the_norm_keys(self, norm):
        """Everything else -- the transformer built after the adapter
        included -- keeps the same init, so the norm costs no RNG."""
        torch.manual_seed(0)
        off = _pre_norm_wrapper("").state_dict()
        torch.manual_seed(0)
        w = _pre_norm_wrapper(norm)
        on = w.state_dict()
        assert [k for k in on if k not in off] == [
            f"adapter.ln_q.{k}" for k in w.adapter.ln_q.state_dict()
        ]
        assert [k for k in off if k not in on] == []
        assert all(torch.equal(on[k], off[k]) for k in off)

    def test_trains_end_to_end(self):
        torch.manual_seed(0)
        w = _pre_norm_wrapper("rmsnorm").to(DEVICE)
        pix = torch.randn(2, 3, 16, 16, device=DEVICE)
        ids = torch.randint(0, 256, (2, 16), device=DEVICE)
        logits, labels = w(pix, ids, ids.clone())
        assert logits.shape == (2, 16, 256)
        assert labels.shape == (2, 16)
        logits.float().logsumexp(-1).mean().backward()
        grad = w.adapter.ln_q.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and (grad != 0).any()


class TestVLMAdapterPreNormCheckpoint:
    """The same ``get_model_state_dict`` template and ``dcp.load`` call that
    ``CheckpointManager.load`` makes, in a single process."""

    @staticmethod
    def _save(wrapper: VLMWrapper, path) -> None:
        dcp.save({"model": get_model_state_dict(wrapper)}, checkpoint_id=str(path))

    def test_round_trip_restores_the_norm(self, tmp_path):
        torch.manual_seed(0)
        source = _pre_norm_wrapper("layernorm")
        with torch.no_grad():
            source.adapter.ln_q.weight.mul_(3.0)
            source.adapter.ln_q.bias.add_(0.25)
        self._save(source, tmp_path)
        torch.manual_seed(1)
        target = _pre_norm_wrapper("layernorm")
        template = {"model": get_model_state_dict(target)}
        dcp.load(template, checkpoint_id=str(tmp_path))
        set_model_state_dict(target, template["model"])
        for k, v in source.state_dict().items():
            assert torch.equal(target.state_dict()[k], v), k

    def test_checkpoint_without_the_norm_refuses_a_model_with_it(self, tmp_path):
        """DCP rejects a template key the checkpoint lacks, so a checkpoint
        saved with pre_norm off cannot be loaded into a pre_norm model."""
        self._save(_pre_norm_wrapper(""), tmp_path)
        target = _pre_norm_wrapper("rmsnorm")
        with pytest.raises(CheckpointException, match=r"adapter\.ln_q\.weight"):
            dcp.load({"model": get_model_state_dict(target)}, checkpoint_id=str(tmp_path))


class TestVLMDecoupledHeadDim:
    """``model.head_dim_override`` is only useful if it survives the build path
    the trainer actually uses."""

    @staticmethod
    def _wrapper(*, head_dim_override: int = 0) -> VLMWrapper:
        return build_vlm_wrapper(
            ModelConfig(
                dim=64,
                n_layers=2,
                n_heads=4,
                n_kv_heads=4,
                head_dim_override=head_dim_override,
                vocab_size=256,
                max_seq_len=64,
            ),
            VisionEncoderConfig(type="random", feature_dim=96, num_tokens=8),
            AdapterConfig(),
            VLMConfig(max_text_len=32),
        )

    def test_decoupled_head_dim_reaches_the_inner_transformer(self):
        w = self._wrapper(head_dim_override=32)
        sd = w.state_dict()
        assert sd["transformer.layers.0.attention.q_proj.weight"].shape == (128, 64)
        assert sd["transformer.layers.0.attention.o_proj.weight"].shape == (64, 128)

    def test_decoupled_head_dim_trains_end_to_end(self):
        torch.manual_seed(0)
        w = self._wrapper(head_dim_override=32).to(DEVICE)
        pix = torch.randn(2, 3, 16, 16, device=DEVICE)
        ids = torch.randint(0, 256, (2, 16), device=DEVICE)
        logits, labels = w(pix, ids, ids.clone())
        assert logits.shape == (2, 16, 256)
        assert labels.shape == (2, 16)
        logits.sum().backward()
        grad = inner_transformer(w).layers["0"].attention.q_proj.weight.grad
        assert grad is not None and grad.shape == (128, 64)
        assert torch.isfinite(grad).all() and (grad != 0).any()

    @pytest.mark.parametrize("arch", ["joint_decoder", "cross_attention", "mot", "moma"])
    def test_every_attention_module_honors_the_decoupled_head_dim(self, arch):
        """The sweep that matters: an attention that still computed
        ``dim // n_heads`` internally would be silently wrong, not rejected.
        Cross-attention keeps its own head count's width unless the override is
        set, which it is here.
        """
        from kempnerforge.model.attention import Attention
        from kempnerforge.model.cross_attention import CrossAttention
        from kempnerforge.model.mot import MoTAttention

        vlm_cfgs = {
            "joint_decoder": JointDecoderConfig(max_text_len=32),
            "cross_attention": CrossAttentionConfig(
                max_text_len=32, cross_attention_every_n_layers=2
            ),
            "mot": MoTConfig(max_text_len=32),
            "moma": MoMaConfig(max_text_len=32),
        }
        model_cfg = ModelConfig(
            dim=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=4,
            head_dim_override=32,
            ffn_hidden_dim=128,
            vocab_size=256,
            max_seq_len=64,
        )
        w = build_vlm_wrapper(
            model_cfg,
            VisionEncoderConfig(type="random", feature_dim=96, num_tokens=8),
            AdapterConfig(),
            vlm_cfgs[arch],
        )
        found = [
            (n, m.head_dim)
            for n, m in w.named_modules()
            if isinstance(m, Attention | MoTAttention | CrossAttention)
        ]
        assert found, f"{arch}: no attention modules found"
        assert all(hd == 32 for _, hd in found), found

    def test_rope_and_flops_read_the_same_head_dim(self):
        """RoPE table width and the FLOP model both derive from
        config.head_dim, so both must track the override."""
        from kempnerforge.metrics.mfu import estimate_model_flops_per_token
        from kempnerforge.model.position import precompute_rope_frequencies

        cfg = ModelConfig(
            dim=64, n_layers=2, n_heads=4, n_kv_heads=4, head_dim_override=32, vocab_size=256
        )
        cos, _ = precompute_rope_frequencies(cfg.head_dim, cfg.max_seq_len)
        assert cos.shape[-1] == 32 // 2
        wider = ModelConfig(
            dim=64, n_layers=2, n_heads=4, n_kv_heads=4, head_dim_override=64, vocab_size=256
        )
        assert estimate_model_flops_per_token(wider, 64) > estimate_model_flops_per_token(cfg, 64)
