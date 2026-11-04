"""BlockMask builders for FlexAttention-based self-attention.

FlexAttention replaces the dense ``(B, 1, S, S)`` boolean mask that the packed
path otherwise hands to ``F.scaled_dot_product_attention``: the mask predicate
is compiled into the attention kernel and fully-masked blocks are skipped
rather than computed. For document packing the mask is block-diagonal and
mostly zeros, so that is the difference between paying for the full S x S
attention and paying only for the blocks inside a document.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

_FLEX_COMPILED = torch.compile(flex_attention, dynamic=False)
_CREATE_BLOCK_MASK_COMPILED = torch.compile(create_block_mask, dynamic=False)


def flex_attention_fn(compiled: bool) -> Callable[..., Any]:
    """Return the ``flex_attention`` callable to use.

    Args:
        compiled: Whether to return the ``torch.compile``-wrapped kernel. True on
            CUDA; False on CPU, where the eager decomposition keeps unit tests off
            Inductor's C++ codegen path. Note torch 2.11 has no CPU backward for
            FlexAttention, so the CPU path is forward-only.
    """
    return _FLEX_COMPILED if compiled else flex_attention


def build_doc_causal_block_mask(doc_ids: torch.Tensor, device: torch.device) -> BlockMask:
    """Block-diagonal causal mask: q attends to k iff same document and k <= q.

    Built once per forward and shared by every layer. ``H=None`` broadcasts the
    mask over heads, which is what lets grouped-query attention pass
    ``enable_gqa=True`` instead of materializing repeated K/V heads, and what
    keeps the mask correct under tensor parallelism, where each rank holds only
    a shard of the heads.

    ``seq_len`` must be at least ``FLEX_BLOCK_SIZE``: below it an Inductor-compiled
    model silently leaks attention across document boundaries, so
    ``JobConfig.validate`` rejects that configuration outright. This kernel is not
    the culprit -- it is exact at those lengths, as is the same model under
    ``backend="eager"`` or ``"aot_eager"``; only Inductor codegen diverges.

    Args:
        doc_ids: Per-token document ids, shape ``(batch, seq_len)``.
        device: Device on which to materialize the ``BlockMask``.

    Returns:
        A ``BlockMask`` over ``(batch, seq_len, seq_len)``, head-broadcast.
    """
    return _build_doc_causal_block_mask(doc_ids, device)  # type: ignore[reportCallIssue]


@torch._dynamo.disable
def _build_doc_causal_block_mask(doc_ids: torch.Tensor, device: torch.device) -> BlockMask:
    """``build_doc_causal_block_mask`` body, hidden from Dynamo.

    ``create_block_mask`` is not meant to be traced by an enclosing
    ``torch.compile``. Disabling here costs one graph break at the top of the
    model forward -- not one per layer, and not one per attention call. The
    public wrapper above exists so callers (and pyright) see a real signature;
    ``torch._dynamo.disable`` erases the one it wraps.
    """
    batch, seq_len = doc_ids.shape
    doc_ids = doc_ids.to(device=device, dtype=torch.int32)

    def mask_mod(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        return (kv_idx <= q_idx) & (doc_ids[b, q_idx] == doc_ids[b, kv_idx])

    builder = _CREATE_BLOCK_MASK_COMPILED if device.type == "cuda" else create_block_mask
    return builder(mask_mod, B=batch, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device)
