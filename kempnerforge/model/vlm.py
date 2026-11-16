"""Vision-language model wrapper.

The wrapper composes a ``VisionEncoder`` (HF or test stub), a registered
adapter (``MLP2LayerAdapter`` by default; ``LinearAdapter`` available
via the ``adapter`` registry) projecting image features into the LLM
embedding space, and the existing ``Transformer``. The arch-specific
work (composing ``pixel_values`` + ``input_ids`` into a
``ModalityContext``) lives on a ``ModalityStrategy`` that the wrapper
holds, so adding a new arch is one new strategy decorator on
``@registry.register_modality_strategy`` plus one new ``VLMConfig``
subclass, and adding a new adapter is one new builder under
``@registry.register_adapter``. No edits to ``VLMWrapper.forward``,
no ``isinstance`` ladder.

Strategies registered today:

- ``"joint_decoder"`` — image embeds prepended to the text sequence
  via ``ModalityContext.prefix_embeds`` + ``output_slice``.
- ``"cross_attention"`` — image embeds passed via
  ``ModalityContext.image_features`` to the ``CrossAttentionBlock``s
  inside ``Transformer``.
- ``"mot"`` — Mixture-of-Transformers. Same residual-stream layout as
  Joint-Decoder (image-then-text concat, ``output_slice`` trims image
  positions before the head), plus a per-position ``modality_ids``
  tag that the ``MoTBlock`` stack consumes for routing.
- ``"moma"`` — Mixture of Modality-Aware Experts. Same residual layout
  and ``modality_ids`` tagging as MoT; per-layer block has shared
  Q/K/V/O attention but per-modality MoE FFN groups.

``inner_transformer(model)`` is the explicit unwrap helper used by the
training loop when it needs to reach Transformer-internal state
(``set_moe_step``, ``get_moe_aux_loss``, ...). Callers that expect the
raw ``Transformer`` interface pipe through this helper rather than
relying on attribute fallthrough on ``VLMWrapper``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import torch
import torch.nn as nn

from kempnerforge.config.adapter import AdapterConfig
from kempnerforge.config.registry import registry
from kempnerforge.config.schema import ModelConfig
from kempnerforge.config.vision import VisionEncoderConfig
from kempnerforge.config.vlm import FreezeSpec, VLMConfig
from kempnerforge.model.adapter import VisionAdapter, build_adapter
from kempnerforge.model.modality import ModalityContext
from kempnerforge.model.transformer import Transformer
from kempnerforge.model.vision import VisionEncoder


class ModalityStrategy(Protocol):
    """Composes raw VLM inputs into a ``ModalityContext``. One strategy
    per arch, registered via ``@registry.register_modality_strategy``.

    Strategies are stateless (hold no parameters) and read submodules
    off the ``VLMWrapper`` they receive. They are NOT registered as
    submodules of the wrapper, so FSDP2 does not wrap them and DCP
    does not serialize them.
    """

    def prepare(
        self,
        wrapper: VLMWrapper,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
        *,
        image_rows: torch.Tensor | None = None,
        image_starts: torch.Tensor | None = None,
        image_tokens: int | None = None,
        text_rows: torch.Tensor | None = None,
        text_positions: torch.Tensor | None = None,
    ) -> ModalityContext: ...

    def num_image_tokens(self, wrapper: VLMWrapper) -> int: ...


def _encode_and_adapt(wrapper: VLMWrapper, images: torch.Tensor) -> torch.Tensor:
    """Run the vision encoder and adapter over ``(N, 3, H, W)``, casting at their boundary."""
    feats = wrapper.vision_encoder(images)
    adapter_dtype = next(wrapper.adapter.parameters()).dtype
    if feats.dtype != adapter_dtype:
        feats = feats.to(adapter_dtype)
    return wrapper.adapter(feats)


def _check_image_tokens(k: int, image_tokens: int | None) -> None:
    if image_tokens is not None and k != image_tokens:
        raise ValueError(
            f"the packer laid out {image_tokens} tokens per image but the adapter produces "
            f"{k}; set the packer's tokens_per_image to the adapter's output count"
        )


def _text_only_context(
    wrapper: VLMWrapper,
    pixel_values: torch.Tensor,
    image_tokens: int | None,
    output_rows: torch.Tensor | None,
    output_positions: torch.Tensor | None,
) -> ModalityContext:
    """The inline context of a batch with no images, keeping every rank's collectives in step.

    FSDP2's all-gathers and reduce-scatters run on every rank, so the encoder
    and adapter see one blank image here whatever their wrapping, and write
    nothing: the result enters the graph as an empty ``(0, dim)`` inline
    embedding with zero weight, so their parameters get zero gradients rather
    than none.
    """
    blank = pixel_values.new_zeros((1, *pixel_values.shape[1:]))
    embeds = _encode_and_adapt(wrapper, blank)
    _check_image_tokens(embeds.shape[1], image_tokens)
    anchor = (embeds.sum() * 0).expand(0, embeds.shape[-1])
    empty = torch.zeros(0, dtype=torch.long, device=anchor.device)
    return ModalityContext(
        inline_embeds=anchor,
        inline_rows=empty,
        inline_positions=empty,
        output_rows=output_rows,
        output_positions=output_positions,
    )


def _inline_image_context(
    wrapper: VLMWrapper,
    pixel_values: torch.Tensor,
    input_ids: torch.Tensor,
    image_rows: torch.Tensor,
    image_starts: torch.Tensor,
    image_tokens: int | None,
    text_rows: torch.Tensor | None = None,
    text_positions: torch.Tensor | None = None,
) -> ModalityContext:
    """Encode a packed batch's flat images and place each one's tokens inside its row.

    ``pixel_values`` is ``(N, 3, H, W)``, one entry per image (or view);
    ``image_rows[n]`` and ``image_starts[n]`` give the row and first position
    of image ``n``'s ``tokens_per_image`` placeholder slots in ``input_ids``.
    ``image_tokens`` is the slot count the packer laid out; the adapter's
    output must match it or the slots would be filled wrong. The placement
    itself is the packer's and is not re-checked on the device. ``text_rows``
    and ``text_positions`` name the real text tokens, so the head runs on
    them alone instead of on every slot of every row. ``N = 0`` is a batch of
    text-only documents (``_text_only_context``).
    """
    if pixel_values.dim() != 4:
        raise ValueError(
            f"a packed batch's pixel_values must be (N, 3, H, W), got {tuple(pixel_values.shape)}"
        )
    if image_rows.shape != image_starts.shape or image_rows.shape != (pixel_values.shape[0],):
        raise ValueError(
            f"image_rows and image_starts must be ({pixel_values.shape[0]},), got "
            f"{tuple(image_rows.shape)} and {tuple(image_starts.shape)}"
        )
    device = input_ids.device
    output_rows = None if text_rows is None else text_rows.to(device)
    output_positions = None if text_positions is None else text_positions.to(device)
    if pixel_values.shape[0] == 0:
        return _text_only_context(
            wrapper, pixel_values, image_tokens, output_rows, output_positions
        )
    embeds = _encode_and_adapt(wrapper, pixel_values)
    n, k, dim = embeds.shape
    _check_image_tokens(k, image_tokens)
    offsets = torch.arange(k, device=device)
    positions = (image_starts.to(device).unsqueeze(1) + offsets).reshape(-1)
    rows = image_rows.to(device).repeat_interleave(k)
    return ModalityContext(
        inline_embeds=embeds.reshape(n * k, dim),
        inline_rows=rows,
        inline_positions=positions,
        output_rows=output_rows,
        output_positions=output_positions,
    )


def _reject_packed(arch: str, image_rows: torch.Tensor | None) -> None:
    if image_rows is not None:
        raise NotImplementedError(
            f"packed batches (image_rows) are supported by arch='joint_decoder' only, not {arch!r}"
        )


def _project_visual_features(wrapper: VLMWrapper, pixel_values: torch.Tensor) -> torch.Tensor:
    """Encode + adapt visual features into LLM-dim tokens.

    Accepts a single-image batch ``(B, 3, H, W)`` or a video-clip batch
    ``(B, F, 3, H, W)``. For video the frame axis is folded into the batch so
    the per-frame vision encoder + adapter run once over ``B*F`` frames, then
    the per-frame tokens are concatenated back per clip in frame order to
    ``(B, F * tokens_per_frame, dim)``. A single image is just the ``F == 1``
    case with the frame axis absent.

    Casts at the encoder/adapter boundary so the encoder can stay in its HF
    dtype (often fp32) while the adapter and transformer run in bf16.
    """
    is_video = pixel_values.dim() == 5
    effective_frames = pixel_values.shape[1] if is_video else 1
    if effective_frames != wrapper.frames_per_clip:
        raise ValueError(
            f"frames-per-clip mismatch: received {effective_frames} frame(s) or view(s) "
            f"(pixel_values.dim()={pixel_values.dim()}) but the wrapper was built with "
            f"frames_per_clip={wrapper.frames_per_clip}, a clip's frames or a tiled image's "
            "views. Pass exactly that many per sample, or rebuild the wrapper for this count."
        )
    if is_video:
        b, f = pixel_values.shape[0], pixel_values.shape[1]
        encoder_input = pixel_values.reshape(b * f, *pixel_values.shape[2:])
    else:
        encoder_input = pixel_values
    embeds = _encode_and_adapt(wrapper, encoder_input)
    if is_video:
        embeds = embeds.reshape(b, f * embeds.shape[1], embeds.shape[2])
    return embeds


def _visual_token_mask(
    frame_mask: torch.Tensor | None, num_visual_tokens: int
) -> torch.Tensor | None:
    """Expand a per-frame validity mask to per-visual-token.

    ``frame_mask`` is ``(B, F)`` bool (``True`` = real frame). Each frame maps to
    ``num_visual_tokens // F`` visual tokens (frame-contiguous, see
    ``_project_visual_features``), so each frame's bit is repeated over its
    tokens -> ``(B, num_visual_tokens)``. Returns ``None`` when no mask is given
    (the image path, or a caller that passes nothing), read downstream as "all
    tokens valid".
    """
    if frame_mask is None:
        return None
    num_frames = frame_mask.shape[1]
    if num_visual_tokens % num_frames != 0:
        raise ValueError(
            f"_visual_token_mask: num_visual_tokens ({num_visual_tokens}) is not a "
            f"multiple of num_frames ({num_frames}); the per-frame expansion assumes "
            "frame-contiguous visual tokens."
        )
    tokens_per_frame = num_visual_tokens // num_frames
    return frame_mask.repeat_interleave(tokens_per_frame, dim=1)


def _prefix_key_padding_mask(
    frame_mask: torch.Tensor | None, num_visual_tokens: int, input_ids: torch.Tensor
) -> torch.Tensor | None:
    """Residual key-validity mask ``(B, S)`` for the image-prefix arches.

    ``S = num_visual_tokens + T_text``. Visual positions follow the expanded
    per-frame mask; text positions are always valid (trailing text padding is
    causal-safe and is not masked here). Returns ``None`` when no frame_mask is
    given.
    """
    vmask = _visual_token_mask(frame_mask, num_visual_tokens)
    if vmask is None:
        return None
    b, t_text = input_ids.shape
    text_valid = torch.ones(b, t_text, dtype=torch.bool, device=vmask.device)
    return torch.cat([vmask, text_valid], dim=1)


@registry.register_modality_strategy("joint_decoder")
class JointDecoderStrategy:
    """Joint-Decoder: image embeds prepended to the text sequence.

    Forward path: ``feats = vision_encoder(pixel_values)``;
    ``img_embeds = adapter(feats)``; ``ModalityContext(prefix_embeds,
    output_slice)``. The transformer runs over the concatenated
    ``(image, text)`` sequence and ``output_slice`` trims the image
    positions before the LM head.

    A packed batch (``image_rows`` given) takes the inline route instead:
    the flat images are encoded and each one's tokens are written over its
    placeholder slots inside its row, ``doc_ids`` isolates the documents, and
    the head runs on the text positions the packer names. A text-only sample
    on the prefix route arrives with a blank image and a ``frame_mask`` of
    ``False`` that hides its prefix; a packed batch of text-only documents
    arrives with no images at all, and the encoder runs on a blank one so its
    collectives stay in step across ranks.
    """

    def prepare(
        self,
        wrapper: VLMWrapper,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
        *,
        image_rows: torch.Tensor | None = None,
        image_starts: torch.Tensor | None = None,
        image_tokens: int | None = None,
        text_rows: torch.Tensor | None = None,
        text_positions: torch.Tensor | None = None,
    ) -> ModalityContext:
        if image_rows is not None:
            assert image_starts is not None
            return _inline_image_context(
                wrapper,
                pixel_values,
                input_ids,
                image_rows,
                image_starts,
                image_tokens,
                text_rows,
                text_positions,
            )
        img_embeds = _project_visual_features(wrapper, pixel_values)
        n = img_embeds.shape[1]
        return ModalityContext(
            prefix_embeds=img_embeds,
            output_slice=slice(n, None),
            key_padding_mask=_prefix_key_padding_mask(frame_mask, n, input_ids),
        )

    def num_image_tokens(self, wrapper: VLMWrapper) -> int:
        return wrapper.frames_per_clip * wrapper.adapter.output_num_tokens(
            wrapper.vision_encoder.num_tokens
        )


@registry.register_modality_strategy("cross_attention")
class CrossAttentionStrategy:
    """Cross-Attention: image embeds flow as K/V into separate
    cross-attention blocks inside the transformer; the residual stream
    itself carries text only.

    Forward path: ``feats = vision_encoder(pixel_values)``;
    ``img_embeds = adapter(feats)``; ``ModalityContext(image_features,
    image_mask)``. ``image_mask`` carries per-visual-token validity (padded
    video frames are masked out of the image K/V); ``None`` means all image
    tokens are valid (e.g. a single image or a full clip).
    """

    def prepare(
        self,
        wrapper: VLMWrapper,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,  # noqa: ARG002
        frame_mask: torch.Tensor | None = None,
        *,
        image_rows: torch.Tensor | None = None,
        image_starts: torch.Tensor | None = None,  # noqa: ARG002
        image_tokens: int | None = None,  # noqa: ARG002
        text_rows: torch.Tensor | None = None,  # noqa: ARG002
        text_positions: torch.Tensor | None = None,  # noqa: ARG002
    ) -> ModalityContext:
        _reject_packed("cross_attention", image_rows)
        img_embeds = _project_visual_features(wrapper, pixel_values)
        return ModalityContext(
            image_features=img_embeds,
            image_mask=_visual_token_mask(frame_mask, img_embeds.shape[1]),
        )

    def num_image_tokens(self, wrapper: VLMWrapper) -> int:  # noqa: ARG002
        return 0


@registry.register_modality_strategy("mot")
class MoTStrategy:
    """Mixture-of-Transformers: image-then-text residual layout (same as
    Joint-Decoder) plus a per-position ``modality_ids`` tag.

    Forward path: ``feats = vision_encoder(pixel_values)``;
    ``img_embeds = adapter(feats)``;
    ``ModalityContext(prefix_embeds, output_slice, modality_ids)``.

    ``modality_ids`` is built position-based: ``0`` for the first
    ``num_image_tokens`` positions and ``1`` for the rest. The MoT
    forward path uses position-based slicing for v1 routing (the tags
    are validated for shape but not value-matched against positions),
    so a future per-token scatter/gather can land without changing the
    public interface.

    ``output_slice`` trims the image prefix off the residual before
    the LM head, matching ``JointDecoderStrategy``.
    """

    def prepare(
        self,
        wrapper: VLMWrapper,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
        *,
        image_rows: torch.Tensor | None = None,
        image_starts: torch.Tensor | None = None,  # noqa: ARG002
        image_tokens: int | None = None,  # noqa: ARG002
        text_rows: torch.Tensor | None = None,  # noqa: ARG002
        text_positions: torch.Tensor | None = None,  # noqa: ARG002
    ) -> ModalityContext:
        _reject_packed("mot", image_rows)
        img_embeds = _project_visual_features(wrapper, pixel_values)
        n = img_embeds.shape[1]
        b, t_text = input_ids.shape
        modality_ids = torch.zeros(b, n + t_text, dtype=torch.long, device=input_ids.device)
        modality_ids[:, n:] = 1
        return ModalityContext(
            prefix_embeds=img_embeds,
            output_slice=slice(n, None),
            modality_ids=modality_ids,
            key_padding_mask=_prefix_key_padding_mask(frame_mask, n, input_ids),
        )

    def num_image_tokens(self, wrapper: VLMWrapper) -> int:
        return wrapper.frames_per_clip * wrapper.adapter.output_num_tokens(
            wrapper.vision_encoder.num_tokens
        )


@registry.register_modality_strategy("moma")
class MoMaStrategy:
    """Mixture of Modality-Aware Experts: same residual-stream layout as
    Joint-Decoder/MoT (image embeds prepended, ``output_slice`` trims them
    before the LM head), plus a per-position ``modality_ids`` tag the
    MoMa FFN stack consumes for true scatter/gather dispatch (level-1
    deterministic routing by modality).

    Forward path: ``feats = vision_encoder(pixel_values)``;
    ``img_embeds = adapter(feats)``;
    ``ModalityContext(prefix_embeds, output_slice, modality_ids)``.

    Convention: ``modality_ids == 0`` for image positions and
    ``modality_ids == 1`` for text positions, matching the index order
    of ``MoMaConfig.moma_modalities = ("image", "text")``. The MoMa
    FFN uses these tags to dispatch tokens to per-modality expert
    groups; positions are *not* assumed to be in any particular order,
    so interleaved layouts work too (image-prefix is just one
    instantiation).
    """

    def prepare(
        self,
        wrapper: VLMWrapper,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
        *,
        image_rows: torch.Tensor | None = None,
        image_starts: torch.Tensor | None = None,  # noqa: ARG002
        image_tokens: int | None = None,  # noqa: ARG002
        text_rows: torch.Tensor | None = None,  # noqa: ARG002
        text_positions: torch.Tensor | None = None,  # noqa: ARG002
    ) -> ModalityContext:
        _reject_packed("moma", image_rows)
        img_embeds = _project_visual_features(wrapper, pixel_values)
        n = img_embeds.shape[1]
        b, t_text = input_ids.shape
        modality_ids = torch.zeros(b, n + t_text, dtype=torch.long, device=input_ids.device)
        modality_ids[:, n:] = 1
        return ModalityContext(
            prefix_embeds=img_embeds,
            output_slice=slice(n, None),
            modality_ids=modality_ids,
            key_padding_mask=_prefix_key_padding_mask(frame_mask, n, input_ids),
        )

    def num_image_tokens(self, wrapper: VLMWrapper) -> int:
        return wrapper.frames_per_clip * wrapper.adapter.output_num_tokens(
            wrapper.vision_encoder.num_tokens
        )


def build_modality_strategy(vlm: VLMConfig) -> ModalityStrategy:
    """Resolve ``vlm.arch`` to its registered ``ModalityStrategy``.

    Pure registry lookup; no ``isinstance`` ladder, no special-cases.
    Adding a new arch is a single ``@registry.register_modality_strategy``
    decorator on a new strategy class.
    """
    return registry.get_modality_strategy(vlm.arch)()


class VLMWrapper(nn.Module):
    """VLM wrapper, arch-driven by a ``ModalityStrategy``.

    Forward: ``(pixel_values, input_ids, labels) -> (logits, labels)``.
    The strategy composes a ``ModalityContext`` from the raw inputs
    and the wrapper's submodules; ``Transformer.forward`` consumes the
    context. ``num_image_tokens`` is arch-aware and delegates to the
    strategy.

    A packed batch (``pack_vlm_samples``) passes ``doc_ids``, per-document
    ``position_ids``, the image placement ``image_rows``, ``image_starts`` and
    ``image_tokens``, and the text positions ``text_rows`` and
    ``text_positions``; the strategy places each image inside its row and the
    head runs on the text positions, so the logits and the returned labels
    are ``(1, M, ...)`` over those tokens. ``attention_group_ids`` travel to
    ``Transformer.forward`` like ``doc_ids``.
    """

    def __init__(
        self,
        vision_encoder: VisionEncoder,
        adapter: VisionAdapter,
        transformer: Transformer,
        strategy: ModalityStrategy,
        frames_per_clip: int = 1,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.adapter = adapter
        self.transformer = transformer
        self.frames_per_clip = frames_per_clip
        self.strategy = strategy

    @property
    def num_image_tokens(self) -> int:
        return self.strategy.num_image_tokens(self)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
        *,
        doc_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_group_ids: torch.Tensor | None = None,
        image_rows: torch.Tensor | None = None,
        image_starts: torch.Tensor | None = None,
        image_tokens: int | None = None,
        text_rows: torch.Tensor | None = None,
        text_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        placement = (image_rows is None, image_starts is None, image_tokens is None)
        if any(placement) and not all(placement):
            raise ValueError("image_rows, image_starts and image_tokens must be given together")
        if (text_rows is None) != (text_positions is None):
            raise ValueError("text_rows and text_positions must be given together")
        if image_rows is None and text_rows is not None:
            raise ValueError("text_rows and text_positions belong to a packed batch")
        if image_rows is not None and frame_mask is not None:
            raise ValueError("a packed batch carries no frame_mask")
        if image_rows is not None and (doc_ids is None or position_ids is None):
            raise ValueError("a packed batch needs doc_ids and position_ids")
        if image_rows is None:
            modality = self.strategy.prepare(self, pixel_values, input_ids, frame_mask=frame_mask)
        else:
            modality = self.strategy.prepare(
                self,
                pixel_values,
                input_ids,
                frame_mask=frame_mask,
                image_rows=image_rows,
                image_starts=image_starts,
                image_tokens=image_tokens,
                text_rows=text_rows,
                text_positions=text_positions,
            )
            if labels is not None and text_rows is not None:
                assert text_positions is not None
                labels = labels[text_rows, text_positions].unsqueeze(0)
        logits = self.transformer(
            tokens=input_ids,
            modality=modality,
            doc_ids=doc_ids,
            position_ids=position_ids,
            attention_group_ids=attention_group_ids,
        )
        return logits, labels


def inner_transformer(model: nn.Module) -> nn.Module:
    """Return the underlying ``Transformer``, unwrapping ``VLMWrapper``
    and ``torch.compile``.

    Training-loop call sites that need to reach Transformer internals
    (``set_moe_step``, ``get_moe_aux_loss``, ``get_expert_counts``, future
    methods) route through this helper. Explicit unwrap is predictable
    under ``torch.compile`` and FSDP2 wrapping; it also makes the VLM
    branch visible at the call site rather than buried in ``__getattr__``.
    """
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod  # type: ignore[attr-defined]
    return model.transformer if isinstance(model, VLMWrapper) else model


def packed_tokens_per_image(model: nn.Module) -> int:
    """Tokens one image (or view) occupies in a packed row, read off the built ``VLMWrapper``."""
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod  # type: ignore[attr-defined]
    if not isinstance(model, VLMWrapper):
        raise TypeError(f"packed_tokens_per_image needs a VLMWrapper, got {type(model).__name__}")
    return model.num_image_tokens // model.frames_per_clip


def _is_encoder_frozen(specs: Iterable[FreezeSpec]) -> bool:
    """True iff every freeze spec targeting the vision encoder is ``frozen=True``.

    Partial unfreezes (e.g. ``FreezeSpec("vision_encoder.layers.11", False)``)
    leave the encoder partly trainable and return False, so ``build_vlm``
    keeps the encoder in ``.train()`` mode and does not replicate it as a
    cheap full copy under FSDP2.
    """
    relevant = [
        s for s in specs if s.module == "vision_encoder" or s.module.startswith("vision_encoder.")
    ]
    if not relevant:
        return False
    return all(s.frozen for s in relevant)


def build_vlm_wrapper(
    model_config: ModelConfig,
    vision_config: VisionEncoderConfig,
    adapter_config: AdapterConfig,
    vlm_config: VLMConfig,
    frames_per_clip: int = 1,
) -> VLMWrapper:
    """Build a ``VLMWrapper`` from the four top-level configs.

    Used by tests and by ``build_parallel_model``. Constructs the
    vision encoder via the registry (HF weights loaded on CPU), builds
    an adapter via the ``adapter`` registry at the LLM ``dim``, looks
    up the right ``ModalityStrategy`` by arch, and composes them with
    a raw ``Transformer``. Callers that need meta-device / FSDP /
    freeze handling go through ``build_parallel_model`` instead.

    All four configs are required: the schema flip lifted the vision /
    adapter / VLM sections out of ``ModelConfig`` and made them parallel
    siblings.
    """
    encoder_builder = registry.get_vision_encoder(vision_config.type)
    encoder = encoder_builder(
        vision_config.path,
        num_tokens=vision_config.num_tokens if vision_config.num_tokens > 0 else None,
        feature_dim=vision_config.feature_dim if vision_config.feature_dim > 0 else None,
    )
    in_dim = vision_config.feature_dim or encoder.feature_dim
    adapter = build_adapter(
        adapter_config, in_dim=in_dim, out_dim=model_config.dim, norm_eps=model_config.norm_eps
    )
    visual_tokens = frames_per_clip * adapter.output_num_tokens(encoder.num_tokens)
    residual_image_tokens = vlm_config.residual_stream_image_tokens(visual_tokens)
    required = residual_image_tokens + vlm_config.max_text_len
    if model_config.max_seq_len < required:
        raise ValueError(
            f"max_seq_len ({model_config.max_seq_len}) insufficient for VLM at build time: "
            f"encoder.num_tokens ({encoder.num_tokens}) -> adapter visual_tokens "
            f"({visual_tokens}) -> residual_image_tokens ({residual_image_tokens}) + "
            f"vlm.max_text_len ({vlm_config.max_text_len}) = {required}"
        )
    transformer = Transformer(model_config, vlm_config=vlm_config, num_image_tokens=visual_tokens)
    strategy = build_modality_strategy(vlm_config)
    return VLMWrapper(encoder, adapter, transformer, strategy, frames_per_clip=frames_per_clip)
