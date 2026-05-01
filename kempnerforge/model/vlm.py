"""Vision-language model wrapper.

The wrapper composes a ``VisionEncoder`` (HF or test stub), a 2-layer
MLP ``Adapter`` projecting image features into the LLM embedding
space, and the existing ``Transformer``. The arch-specific work
(composing ``pixel_values`` + ``input_ids`` into a
``ModalityContext``) lives on a ``ModalityStrategy`` that the wrapper
holds, so adding a new arch is one new strategy decorator on
``@registry.register_modality_strategy`` plus one new ``VLMConfig``
subclass — no edits to ``VLMWrapper.forward``, no ``isinstance``
ladder.

Strategy registered today:

- ``"joint_decoder"`` — image embeds prepended to the text sequence
  via ``ModalityContext.prefix_embeds`` + ``output_slice``.

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

from kempnerforge.config.registry import registry
from kempnerforge.config.schema import ModelConfig
from kempnerforge.config.vlm import FreezeSpec, VLMConfig
from kempnerforge.model.modality import ModalityContext
from kempnerforge.model.transformer import Transformer
from kempnerforge.model.vision import VisionEncoder

_ADAPTER_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "relu": nn.ReLU,
}


class Adapter(nn.Module):
    """2-layer MLP from image-feature dim to LLM embedding dim.

    Architecture: ``Linear(in_dim, hidden) -> activation -> Linear(hidden, out_dim)``.
    ``hidden_dim=None`` defaults to ``out_dim``.

    ``reset_parameters`` is provided so callers that materialize adapters
    from meta can re-initialize weights with the standard Linear defaults.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int | None = None,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError("Adapter in_dim and out_dim must be positive")
        if activation not in _ADAPTER_ACTIVATIONS:
            raise ValueError(
                f"Unknown adapter activation: {activation!r}. Options: {list(_ADAPTER_ACTIVATIONS)}"
            )
        hidden = hidden_dim if hidden_dim and hidden_dim > 0 else out_dim
        self.proj1 = nn.Linear(in_dim, hidden, bias=True)
        self.act = _ADAPTER_ACTIVATIONS[activation]()
        self.proj2 = nn.Linear(hidden, out_dim, bias=True)

    def reset_parameters(self) -> None:
        """Re-run ``nn.Linear`` default init on both projections.

        Used after ``to_empty(device=...)`` on a meta-device build.
        """
        self.proj1.reset_parameters()
        self.proj2.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj2(self.act(self.proj1(x)))


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
    ) -> ModalityContext: ...

    def num_image_tokens(self, wrapper: VLMWrapper) -> int: ...


def _project_image_features(wrapper: VLMWrapper, pixel_values: torch.Tensor) -> torch.Tensor:
    """Encode + adapt image features. Cast at the encoder/adapter
    boundary so the encoder can stay in its HF dtype (often fp32) while
    the adapter and transformer run in bf16 without an inner dtype
    clash.
    """
    feats = wrapper.vision_encoder(pixel_values)
    adapter_dtype = wrapper.adapter.proj1.weight.dtype
    if feats.dtype != adapter_dtype:
        feats = feats.to(adapter_dtype)
    return wrapper.adapter(feats)


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
    ) -> ModalityContext:
        img_embeds = _project_image_features(wrapper, pixel_values)
        n = wrapper.vision_encoder.num_tokens
        return ModalityContext(prefix_embeds=img_embeds, output_slice=slice(n, None))

    def num_image_tokens(self, wrapper: VLMWrapper) -> int:
        return wrapper.vision_encoder.num_tokens


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
        adapter: Adapter,
        transformer: Transformer,
        strategy: ModalityStrategy,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.adapter = adapter
        self.transformer = transformer
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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Route the text embedding through Transformer.forward so FSDP2's
        # per-module hook intercepts the token_embedding call and
        # materializes the DTensor weight before F.embedding runs. Doing
        # the embedding externally (transformer.token_embedding(input_ids))
        # bypasses FSDP and fails with "mixed torch.Tensor and DTensor".
        modality = self.strategy.prepare(self, pixel_values, input_ids)
        logits = self.transformer(tokens=input_ids, modality=modality)
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


def build_vlm_wrapper(model_config: ModelConfig) -> VLMWrapper:
    """Build a ``VLMWrapper`` from a ``ModelConfig`` with ``vlm`` set.

    Used by tests and by ``build_parallel_model``. Constructs the
    vision encoder via the registry (HF weights loaded on CPU), builds
    a fresh ``Adapter`` at the LLM ``dim``, looks up the right
    ``ModalityStrategy`` by arch, and composes them with a raw
    ``Transformer``. Callers that need meta-device / FSDP / freeze
    handling go through ``build_parallel_model`` instead.
    """
    vlm: VLMConfig | None = model_config.vlm
    if vlm is None:
        raise ValueError("build_vlm_wrapper requires model_config.vlm to be set")
    encoder_builder = registry.get_vision_encoder(vlm.vision_encoder)
    encoder = encoder_builder(
        vlm.vision_encoder_path,
        num_tokens=vlm.num_tokens if vlm.num_tokens > 0 else None,
        feature_dim=vlm.feature_dim if vlm.feature_dim > 0 else None,
    )
    in_dim = vlm.feature_dim or encoder.feature_dim
    adapter = Adapter(
        in_dim=in_dim,
        out_dim=model_config.dim,
        hidden_dim=vlm.adapter_hidden_dim or None,
        activation=vlm.adapter_activation,
    )
    transformer = Transformer(model_config)
    strategy = build_modality_strategy(vlm)
    return VLMWrapper(encoder, adapter, transformer, strategy)
