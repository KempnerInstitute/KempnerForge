"""Modality-injection container for ``Transformer.forward``.

``ModalityContext`` groups all "inputs that flow into the existing
residual stream" so ``Transformer.forward`` stays narrow regardless of
how many architectures are active. Each VLM arch fills the fields it
needs:

- Joint-Decoder fills ``prefix_embeds + output_slice`` (image tokens
  prepended to the text sequence; ``output_slice`` trims them off the
  hidden state before the LM head).
- Cross-Attention fills ``image_features + image_mask`` (image K/V
  flowing into separate cross-attention blocks; the residual stream
  itself carries text only).
- Pipeline-parallel middle stages fill ``inputs_embeds`` (pre-embedded
  hidden state passed across stage boundaries).

Cross-arg invariants involving ``kv_caches`` (a ``Transformer.forward``
argument, not a ``ModalityContext`` field) are enforced at the top of
``Transformer.forward``, not in ``__post_init__``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModalityContext:
    """Modality-injection container.

    Invariants enforced in ``__post_init__``:

    - At most one of ``inputs_embeds``, ``prefix_embeds``,
      ``image_features`` may be set; they are mutually exclusive
      composition routes into the residual stream.
    - ``image_mask`` requires ``image_features`` to be set (a
      free-standing ``image_mask`` with no features is a programming
      error).

    ``output_slice`` composes with the ``tokens`` path AND with the
    ``inputs_embeds`` path; it is not constrained intra-context. The
    cross-arg constraint (``output_slice`` vs ``kv_caches``) lives on
    ``Transformer.forward`` instead.
    """

    inputs_embeds: torch.Tensor | None = None
    prefix_embeds: torch.Tensor | None = None
    output_slice: slice | None = None
    image_features: torch.Tensor | None = None
    image_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        residual_routes = sum(
            x is not None for x in (self.inputs_embeds, self.prefix_embeds, self.image_features)
        )
        if residual_routes > 1:
            raise ValueError(
                "ModalityContext: at most one of inputs_embeds, prefix_embeds, "
                "image_features may be set (mutually exclusive residual-stream routes)"
            )
        if self.image_mask is not None and self.image_features is None:
            raise ValueError("ModalityContext: image_mask requires image_features to be set")
