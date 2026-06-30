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

    ``prepare`` takes an optional ``precomputed_embeds``: when provided (the
    cached-decode path) it is used as the projected visual embeds in place of
    re-running the vision encoder + adapter; when ``None`` (the default) the
    strategy encodes from ``pixel_values`` as usual.
    """

    def prepare(
        self,
        wrapper: VLMWrapper,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        precomputed_embeds: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
    ) -> ModalityContext: ...

    def num_image_tokens(self, wrapper: VLMWrapper) -> int: ...


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
    # The static visual-token count (residual budget, MoT's positional split) is
    # sized for ``frames_per_clip``, so each clip must carry exactly that many
    # frames. Validate here to turn a downstream shape/split error into a clear one.
    effective_frames = pixel_values.shape[1] if is_video else 1
    if effective_frames != wrapper.frames_per_clip:
        raise ValueError(
            f"frames-per-clip mismatch: received {effective_frames} frame(s) "
            f"(pixel_values.dim()={pixel_values.dim()}) but the wrapper was built with "
            f"frames_per_clip={wrapper.frames_per_clip}. Pass a clip with exactly "
            "frames_per_clip frames (or rebuild the wrapper for this frame count)."
        )
    if is_video:
        b, f = pixel_values.shape[0], pixel_values.shape[1]
        encoder_input = pixel_values.reshape(b * f, *pixel_values.shape[2:])
    else:
        encoder_input = pixel_values
    feats = wrapper.vision_encoder(encoder_input)
    # Adapter-agnostic dtype lookup: use the first adapter parameter's dtype
    # so any registered adapter (mlp_2layer, linear, avgpool, ...) works
    # without coupling to a specific submodule attribute.
    adapter_dtype = next(wrapper.adapter.parameters()).dtype
    if feats.dtype != adapter_dtype:
        feats = feats.to(adapter_dtype)
    embeds = wrapper.adapter(feats)
    if is_video:
        # (B*F, P', dim) -> (B, F*P', dim): frame-contiguous, temporal order kept.
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
        # Visual tokens are frame-contiguous (F * tokens_per_frame), so the count
        # must be divisible by the frame count. A future adapter that adds a
        # non-per-frame token (e.g. a global/CLS token) would break this and
        # silently misalign the mask -- fail loudly here instead.
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
    """

    def prepare(
        self,
        wrapper: VLMWrapper,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,  # noqa: ARG002
        precomputed_embeds: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
    ) -> ModalityContext:
        img_embeds = (
            precomputed_embeds
            if precomputed_embeds is not None
            else _project_visual_features(wrapper, pixel_values)
        )
        n = img_embeds.shape[1]  # pooling-aware: the adapter's actual visual-token count
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
        precomputed_embeds: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
    ) -> ModalityContext:
        img_embeds = (
            precomputed_embeds
            if precomputed_embeds is not None
            else _project_visual_features(wrapper, pixel_values)
        )
        return ModalityContext(
            image_features=img_embeds,
            image_mask=_visual_token_mask(frame_mask, img_embeds.shape[1]),
        )

    def num_image_tokens(self, wrapper: VLMWrapper) -> int:  # noqa: ARG002
        # Cross-Attention does not extend the residual stream.
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
        precomputed_embeds: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
    ) -> ModalityContext:
        img_embeds = (
            precomputed_embeds
            if precomputed_embeds is not None
            else _project_visual_features(wrapper, pixel_values)
        )
        n = img_embeds.shape[1]  # pooling-aware: the adapter's actual visual-token count
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
        precomputed_embeds: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
    ) -> ModalityContext:
        img_embeds = (
            precomputed_embeds
            if precomputed_embeds is not None
            else _project_visual_features(wrapper, pixel_values)
        )
        n = img_embeds.shape[1]  # pooling-aware: the adapter's actual visual-token count
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
        # Frames per video clip (1 for a single image). The static visual-token
        # count is ``frames_per_clip * adapter.output_num_tokens(...)``; the
        # strategies use it for ``num_image_tokens`` (residual budget, MoT split).
        self.frames_per_clip = frames_per_clip
        # Strategy is a plain Python object (not nn.Module). nn.Module's
        # __setattr__ only routes Module/Parameter/Tensor attributes into
        # _modules/_parameters/_buffers, so plain objects are stored as
        # ordinary attributes and stay outside the module tree by
        # default. FSDP2 wraps modules; DCP serializes module params;
        # neither sees the strategy. test_strategy_not_in_module_tree
        # pins this contract.
        self.strategy = strategy

    @property
    def num_image_tokens(self) -> int:
        return self.strategy.num_image_tokens(self)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        precomputed_embeds: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Route the text embedding through Transformer.forward so FSDP2's
        # per-module hook intercepts the token_embedding call and
        # materializes the DTensor weight before F.embedding runs. Doing
        # the embedding externally (transformer.token_embedding(input_ids))
        # bypasses FSDP and fails with "mixed torch.Tensor and DTensor".
        # ``precomputed_embeds`` (when set) is forwarded to the strategy as the
        # projected visual embeds, skipping the per-call vision encode (the
        # cached-decode path); ``None`` keeps the default encode-from-pixels path.
        modality = self.strategy.prepare(
            self, pixel_values, input_ids, precomputed_embeds, frame_mask=frame_mask
        )
        logits = self.transformer(tokens=input_ids, modality=modality)
        return logits, labels

    def encode_visual(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Project visual input to LLM-dim embeds once, for cached decode.

        Returns the post-adapter visual embeds that ``ModalityStrategy.prepare``
        computes internally. Callers (e.g. the eval decode loop) encode once and
        pass the result back via the ``precomputed_embeds`` argument of
        ``forward`` so the vision encoder + adapter do not re-run each decode
        step. This is not a KV cache: the transformer still re-runs over the full
        sequence each step.

        Args:
            pixel_values: A single-image batch ``(B, 3, H, W)`` or a video-clip
                batch ``(B, F, 3, H, W)``; frames-per-clip validation is delegated
                to the projection helper.

        Returns:
            The post-adapter visual embeds of shape
            ``(B, num_visual_tokens, dim)``.
        """
        return _project_visual_features(self, pixel_values)


def inner_transformer(model: nn.Module) -> nn.Module:
    """Return the underlying ``Transformer``, unwrapping ``VLMWrapper``
    and ``torch.compile``.

    Training-loop call sites that need to reach Transformer internals
    (``set_moe_step``, ``get_moe_aux_loss``, ``get_expert_counts``, future
    methods) route through this helper. Explicit unwrap is predictable
    under ``torch.compile`` and FSDP2 wrapping; it also makes the VLM
    branch visible at the call site rather than buried in ``__getattr__``.
    """
    # torch.compile wraps the module in OptimizedModule; the original is
    # exposed via ._orig_mod (documented in torch/_dynamo/eval_frame.py).
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod  # type: ignore[attr-defined]
    return model.transformer if isinstance(model, VLMWrapper) else model


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
    # Build-time max_seq_len cross-check using the encoder's resolved
    # num_tokens. ``JobConfig.__post_init__`` runs the same check at config
    # time only when ``vision_encoder.num_tokens > 0``; when the user leaves
    # num_tokens=0 (the "infer from encoder at build time" sentinel) the
    # config-time check is skipped and the residual-stream allocation goes
    # unchecked until the model actually runs. This guard fills that gap.
    in_dim = vision_config.feature_dim or encoder.feature_dim
    adapter = build_adapter(adapter_config, in_dim=in_dim, out_dim=model_config.dim)
    # Visual tokens entering the LLM = the adapter's output count (pooling
    # adapters reduce it; projection adapters are the identity), not the raw
    # encoder patch count. This drives the residual budget and MoT's split.
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
