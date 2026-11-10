"""Modality-injection container for ``Transformer.forward``.

``ModalityContext`` groups all "inputs that flow into the existing
residual stream" plus per-token routing tags consumed by the residual
stream, so ``Transformer.forward`` stays narrow regardless of how
many architectures are active. Each VLM arch fills the fields it
needs:

- Joint-Decoder fills ``prefix_embeds + output_slice`` (image tokens
  prepended to the text sequence; ``output_slice`` trims them off the
  hidden state before the LM head).
- Cross-Attention fills ``image_features + image_mask`` (image K/V
  flowing into separate cross-attention blocks; the residual stream
  itself carries text only).
- Mixture-of-Transformers fills ``prefix_embeds + output_slice +
  modality_ids``. The residual stream carries (image, text)
  concatenated; ``modality_ids`` tags every position with its
  modality so each layer's MoTBlock can route per-modality
  projections + global self-attention.
- Pipeline-parallel middle stages fill ``inputs_embeds`` (pre-embedded
  hidden state passed across stage boundaries).
- Packed Joint-Decoder batches fill ``inline_embeds + inline_rows +
  inline_positions``: image tokens written over placeholder positions
  inside the token-embedded rows, so each document's images sit before
  its own text and ``doc_ids`` keeps attention inside the document.
- A strategy that chooses the RoPE positions or the attention visibility
  of the residual it lays out (an image's tokens seeing each other, crops
  of one image sharing positions) fills ``position_ids`` and
  ``attention_group_ids``. ``Transformer.forward`` reads them as its
  arguments of the same names, which may not be passed alongside.
- A strategy whose positions are not one integer per token (a 2-D or
  multi-axis RoPE over image crops) fills ``rope_cos`` and ``rope_sin``
  instead: the rotation itself, ``(batch, seq_len, head_dim // 2)`` or
  ``(1, seq_len, head_dim // 2)`` shared by every row, which the
  transformer applies in place of its own table. The strategy builds
  them from ``precompute_rope_frequencies`` with the model's ``head_dim``
  and ``rope_theta``, so the same code positions every forward, training
  and inference alike. ``forward`` moves the rotation tensors to the
  residual's device for each call; build them on that device to avoid
  repeated host-to-device transfers.

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
      ``image_features``, ``inline_embeds`` may be set; they are mutually
      exclusive composition routes into the residual stream.
    - ``inline_embeds`` ``(M, dim)`` requires ``inline_rows`` and
      ``inline_positions``, both ``(M,)`` int64, and they require it:
      token ``m`` is written at ``[inline_rows[m], inline_positions[m]]``
      of the embedded rows.
    - ``image_mask`` requires ``image_features`` to be set (a
      free-standing ``image_mask`` with no features is a programming
      error).
    - ``modality_ids`` requires ``prefix_embeds``, ``inputs_embeds`` or
      ``inline_embeds`` to be set (routing without a residual extension is
      meaningless).
    - ``key_padding_mask`` requires ``prefix_embeds``, ``inputs_embeds`` or
      ``inline_embeds`` to be set (it is a key-validity mask over the
      residual sequence; the Cross-Attention arch masks image K/V via
      ``image_mask`` instead).
    - ``position_ids`` and ``attention_group_ids`` are ``(batch, seq_len)``
      int64 over the residual stream, with the meaning of the
      ``Transformer.forward`` arguments of those names; the dtype and rank
      are checked here, the shape against the residual in ``forward``.
    - ``rope_cos`` and ``rope_sin`` come together, as floating
      ``(batch or 1, seq_len, head_dim // 2)`` tensors of one shape, dtype
      and device, and exclude ``position_ids``; ``forward`` checks the shape
      against the residual.

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
    modality_ids: torch.Tensor | None = None
    key_padding_mask: torch.Tensor | None = None
    inline_embeds: torch.Tensor | None = None
    inline_rows: torch.Tensor | None = None
    inline_positions: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None
    attention_group_ids: torch.Tensor | None = None
    rope_cos: torch.Tensor | None = None
    rope_sin: torch.Tensor | None = None

    def __post_init__(self) -> None:
        residual_routes = sum(
            x is not None
            for x in (
                self.inputs_embeds,
                self.prefix_embeds,
                self.image_features,
                self.inline_embeds,
            )
        )
        if residual_routes > 1:
            raise ValueError(
                "ModalityContext: at most one of inputs_embeds, prefix_embeds, "
                "image_features, inline_embeds may be set (mutually exclusive "
                "residual-stream routes)"
            )
        self._check_inline()
        if self.image_mask is not None and self.image_features is None:
            raise ValueError("ModalityContext: image_mask requires image_features to be set")
        extends_residual = (
            self.prefix_embeds is not None
            or self.inputs_embeds is not None
            or self.inline_embeds is not None
        )
        if self.modality_ids is not None and not extends_residual:
            raise ValueError(
                "ModalityContext: modality_ids requires prefix_embeds, inputs_embeds OR "
                "inline_embeds to be set (routing without a residual extension is meaningless)"
            )
        if self.key_padding_mask is not None and not extends_residual:
            raise ValueError(
                "ModalityContext: key_padding_mask requires prefix_embeds, inputs_embeds OR "
                "inline_embeds to be set (it masks the residual sequence; Cross-Attention "
                "uses image_mask)"
            )
        for name, ids in (
            ("position_ids", self.position_ids),
            ("attention_group_ids", self.attention_group_ids),
        ):
            if ids is None:
                continue
            if ids.dtype != torch.long:
                raise ValueError(
                    f"ModalityContext: {name}.dtype must be torch.int64, got {ids.dtype}"
                )
            if ids.dim() != 2:
                raise ValueError(
                    f"ModalityContext: {name} must be (batch, seq_len), got {tuple(ids.shape)}"
                )
        self._check_rope()

    def _check_inline(self) -> None:
        indices = (self.inline_rows, self.inline_positions)
        if self.inline_embeds is None:
            if any(x is not None for x in indices):
                raise ValueError(
                    "ModalityContext: inline_rows and inline_positions require inline_embeds"
                )
            return
        if any(x is None for x in indices):
            raise ValueError(
                "ModalityContext: inline_embeds requires both inline_rows and inline_positions"
            )
        assert self.inline_rows is not None and self.inline_positions is not None
        if self.inline_embeds.dim() != 2:
            raise ValueError(
                f"ModalityContext: inline_embeds must be (M, dim), got "
                f"{tuple(self.inline_embeds.shape)}"
            )
        m = self.inline_embeds.shape[0]
        for name, index in (
            ("inline_rows", self.inline_rows),
            ("inline_positions", self.inline_positions),
        ):
            if index.dtype != torch.long:
                raise ValueError(
                    f"ModalityContext: {name}.dtype must be torch.long, got {index.dtype}"
                )
            if index.shape != (m,):
                raise ValueError(
                    f"ModalityContext: {name} must be ({m},) to match inline_embeds, got "
                    f"{tuple(index.shape)}"
                )

    def _check_rope(self) -> None:
        if (self.rope_cos is None) != (self.rope_sin is None):
            raise ValueError("ModalityContext: rope_cos and rope_sin must be given together")
        if self.rope_cos is None or self.rope_sin is None:
            return
        if self.position_ids is not None:
            raise ValueError(
                "ModalityContext: rope_cos/rope_sin and position_ids both choose the "
                "rotation; set one"
            )
        for name, table in (("rope_cos", self.rope_cos), ("rope_sin", self.rope_sin)):
            if not table.is_floating_point():
                raise ValueError(
                    f"ModalityContext: {name} must be floating point, got {table.dtype}"
                )
            if table.dim() != 3:
                raise ValueError(
                    f"ModalityContext: {name} must be (batch, seq_len, head_dim // 2), got "
                    f"{tuple(table.shape)}"
                )
        if self.rope_cos.shape != self.rope_sin.shape:
            raise ValueError(
                f"ModalityContext: rope_cos {tuple(self.rope_cos.shape)} and rope_sin "
                f"{tuple(self.rope_sin.shape)} differ in shape"
            )
        if self.rope_cos.dtype != self.rope_sin.dtype:
            raise ValueError(
                f"ModalityContext: rope_cos {self.rope_cos.dtype} and rope_sin "
                f"{self.rope_sin.dtype} differ in dtype"
            )
        if self.rope_cos.device != self.rope_sin.device:
            raise ValueError(
                f"ModalityContext: rope_cos {self.rope_cos.device} and rope_sin "
                f"{self.rope_sin.device} differ in device"
            )
