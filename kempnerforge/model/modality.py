"""Modality-injection container for ``Transformer.forward``.

``ModalityContext`` groups all "inputs that flow into the existing
residual stream" so ``Transformer.forward`` stays narrow regardless of
how many architectures are active. Each VLM arch fills the fields it
needs:

- Joint-Decoder fills ``prefix_embeds + output_slice`` (image tokens
  prepended to the text sequence; ``output_slice`` trims them off the
  hidden state before the LM head).
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

    - At most one of ``inputs_embeds``, ``prefix_embeds`` may be set;
      they are mutually exclusive composition routes into the residual
      stream.

    ``output_slice`` composes with the ``tokens`` path AND with the
    ``inputs_embeds`` path; it is not constrained intra-context. The
    cross-arg constraint (``output_slice`` vs ``kv_caches``) lives on
    ``Transformer.forward`` instead.
    """

    inputs_embeds: torch.Tensor | None = None
    prefix_embeds: torch.Tensor | None = None
    output_slice: slice | None = None

    def __post_init__(self) -> None:
        if self.inputs_embeds is not None and self.prefix_embeds is not None:
            raise ValueError(
                "ModalityContext: at most one of inputs_embeds, prefix_embeds "
                "may be set (mutually exclusive residual-stream routes)"
            )
