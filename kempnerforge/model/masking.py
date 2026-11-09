"""Attention mask builders.

One predicate covers every visibility rule self-attention implements:

``visible(q, k) = same_document(q, k)
AND (k <= q OR same_nonnegative_group(q, k)) AND valid_key(k)``

``build_attention_mask`` materializes it once per forward, in whichever form
the configured backend consumes: a FlexAttention ``BlockMask`` under
``attention_backend="flex"``, a dense ``(batch, 1, seq_len, seq_len)`` bool
tensor for ``F.scaled_dot_product_attention`` otherwise, or ``None`` when no
term applies and plain causal attention is already exact. ``Attention`` and
``MoTAttention`` take that one object and derive no mask of their own, so a
further visibility rule is a new argument here rather than a new parameter on
every attention signature.

FlexAttention replaces the dense mask rather than computing it: the predicate
is compiled into the attention kernel and fully-masked blocks are skipped
rather than computed. For document packing the mask is block-diagonal and
mostly zeros, so that is the difference between paying for the full S x S
attention and paying only for the blocks inside a document. Key validity is
deliberately not folded into the ``BlockMask``; those batches take the dense
form instead.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

_FLEX_STATIC = torch.compile(flex_attention, dynamic=False)
_CREATE_BLOCK_MASK_STATIC = torch.compile(create_block_mask, dynamic=False)
_FLEX_COMPILED = _FLEX_STATIC
_CREATE_BLOCK_MASK_COMPILED = _CREATE_BLOCK_MASK_STATIC


def use_dynamic_flex_shapes(enabled: bool) -> None:
    """Compile the flex helpers for a varying batch size (packed VLM rows), or pin shapes again."""
    global _FLEX_COMPILED, _CREATE_BLOCK_MASK_COMPILED
    if enabled:
        _FLEX_COMPILED = torch.compile(flex_attention)
        _CREATE_BLOCK_MASK_COMPILED = torch.compile(create_block_mask)
    else:
        _FLEX_COMPILED = _FLEX_STATIC
        _CREATE_BLOCK_MASK_COMPILED = _CREATE_BLOCK_MASK_STATIC


def flex_attention_fn(compiled: bool) -> Callable[..., Any]:
    """Return the ``flex_attention`` callable to use.

    Args:
        compiled: Whether to return the ``torch.compile``-wrapped kernel. True on
            CUDA; False on CPU, where the eager decomposition keeps unit tests off
            Inductor's C++ codegen path. Note torch 2.11 has no CPU backward for
            FlexAttention, so the CPU path is forward-only.
    """
    return _FLEX_COMPILED if compiled else flex_attention


def _validate_mask_inputs(
    *,
    batch: int,
    seq_len: int,
    doc_ids: torch.Tensor | None = None,
    group_ids: torch.Tensor | None = None,
    key_padding_mask: torch.Tensor | None = None,
) -> None:
    """Reject mask inputs that do not describe this forward's residual stream.

    Every input is indexed per (example, position), so one that merely
    broadcasts would silently apply a single example's documents, groups or key
    validity to the whole batch. Shapes must match exactly.

    Args:
        batch: Batch size of the residual stream.
        seq_len: Length of the residual stream.
        doc_ids: Optional per-token document ids.
        group_ids: Optional per-token visibility group ids.
        key_padding_mask: Optional per-key validity mask.

    Raises:
        ValueError: If any input's shape is not exactly ``(batch, seq_len)``.
    """
    expected = (batch, seq_len)
    for name, tensor in (
        ("doc_ids", doc_ids),
        ("group_ids", group_ids),
        ("key_padding_mask", key_padding_mask),
    ):
        if tensor is not None and tuple(tensor.shape) != expected:
            raise ValueError(
                f"{name} shape {tuple(tensor.shape)} does not match the residual "
                f"shape {expected}. Mask inputs are indexed per example and "
                "position, so a broadcastable shape would apply one example's "
                "mask to the whole batch."
            )


def build_attention_mask(
    *,
    backend: str,
    batch: int,
    seq_len: int,
    device: torch.device,
    doc_ids: torch.Tensor | None = None,
    group_ids: torch.Tensor | None = None,
    key_padding_mask: torch.Tensor | None = None,
) -> torch.Tensor | BlockMask | None:
    """Build the one attention mask for a forward pass.

    The single entry point every caller uses, and the one place the ids are
    validated: it owns which backend form the semantics are expressed in, so
    callers pass ids and attention consumes one mask. ``backend="flex"`` returns
    a ``BlockMask`` whenever the predicate is expressible in it, which is every
    case except key validity; everything else returns the dense form.

    The returned mask is square over the residual stream, so it is valid only
    for a forward whose keys are exactly its queries. Incremental decoding with
    a ``KVCache`` grows the key axis every step, which is why
    ``Transformer.forward`` refuses every mask input alongside ``kv_caches``.

    Args:
        backend: ``ModelConfig.attention_backend``, ``"sdpa"`` or ``"flex"``.
        batch: Batch size of the residual stream.
        seq_len: Query and key length, the residual stream's length.
        device: Device to materialize the mask on. Any ids passed must already
            be on it.
        doc_ids: Optional per-token document ids, shape ``(batch, seq_len)``.
            Tokens attend only within their own document.
        group_ids: Optional per-token bidirectional-visibility group ids,
            shape ``(batch, seq_len)``; negative means no group. Tokens sharing
            a nonnegative id see each other regardless of order. All-negative
            ids give plain causal visibility but still build a mask; ``None``
            keeps the unmasked path.
        key_padding_mask: Optional per-key validity mask, shape
            ``(batch, seq_len)``; ``True`` = attend, ``False`` = drop (e.g. the
            visual tokens of padded video frames).

    Returns:
        A ``BlockMask``, a dense ``(batch, 1, seq_len, seq_len)`` bool tensor,
        or ``None`` when no term applies and attention is plain causal.

    Raises:
        ValueError: If any input's shape is not exactly ``(batch, seq_len)``.
    """
    _validate_mask_inputs(
        batch=batch,
        seq_len=seq_len,
        doc_ids=doc_ids,
        group_ids=group_ids,
        key_padding_mask=key_padding_mask,
    )
    has_ids = doc_ids is not None or group_ids is not None
    if backend == "flex" and key_padding_mask is None and has_ids:
        return build_doc_causal_block_mask(doc_ids, device, group_ids)
    return build_dense_attention_mask(
        batch=batch,
        seq_len=seq_len,
        device=device,
        doc_ids=doc_ids,
        group_ids=group_ids,
        key_padding_mask=key_padding_mask,
    )


def same_group_mask(group_ids: torch.Tensor) -> torch.Tensor:
    """Dense ``same_nonnegative_group`` term, for the dense mask path.

    The counterpart of the group term inside ``build_doc_causal_block_mask``.

    Args:
        group_ids: Per-token group ids, shape ``(batch, seq_len)``; negative
            means no group.

    Returns:
        Bool tensor of shape ``(batch, 1, seq_len, seq_len)``, ``True`` where
        query and key share a group id and that id is nonnegative.
    """
    same = group_ids.unsqueeze(2) == group_ids.unsqueeze(1)
    return (same & (group_ids >= 0).unsqueeze(2)).unsqueeze(1)


def build_dense_attention_mask(
    *,
    batch: int,
    seq_len: int,
    device: torch.device,
    doc_ids: torch.Tensor | None = None,
    group_ids: torch.Tensor | None = None,
    key_padding_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Dense boolean mask for ``F.scaled_dot_product_attention``.

    ``True`` means the query may attend to the key. Starts causal, widens by
    the group term, then narrows by document isolation and key validity, so a
    group never reaches across a document and never revives an invalid key.

    A query row left with no reachable key -- the leading positions of an
    all-padded clip, say -- would softmax over all ``-inf`` and yield NaN, so
    such rows are unmasked instead. Their outputs are discarded downstream, by
    ``output_slice`` or by the clip's labels being all ``-100``.

    An explicit mask is not a FlashAttention-2 shape, so SDPA falls back to the
    mem-efficient or math kernel whenever one is returned. ``None`` keeps the
    ``is_causal`` fast path, which is why it is returned rather than an
    all-causal tensor.

    Shapes are not checked here: ``build_attention_mask`` validates the ids
    before it hands them on.

    Args:
        batch: Batch size of the residual stream.
        seq_len: Query and key length.
        device: Device to materialize the causal term on. Any ids passed must
            already be on it.
        doc_ids: Optional per-token document ids, shape ``(batch, seq_len)``.
        group_ids: Optional per-token group ids, shape ``(batch, seq_len)``;
            negative means no group.
        key_padding_mask: Optional per-key validity mask, shape
            ``(batch, seq_len)``; ``True`` = attend.

    Returns:
        Bool tensor of shape ``(batch, 1, seq_len, seq_len)``, or ``None`` when
        every optional term is ``None`` and the mask would be plain causal.
    """
    if doc_ids is None and group_ids is None and key_padding_mask is None:
        return None
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()
    mask = causal.unsqueeze(0).unsqueeze(0)
    if group_ids is not None:
        mask = mask | same_group_mask(group_ids)
    if doc_ids is not None:
        mask = mask & (doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)).unsqueeze(1)
    if key_padding_mask is not None:
        mask = mask & key_padding_mask.unsqueeze(1).unsqueeze(1)
    return mask | ~mask.any(dim=-1, keepdim=True)


def build_doc_causal_block_mask(
    doc_ids: torch.Tensor | None,
    device: torch.device,
    group_ids: torch.Tensor | None = None,
) -> BlockMask:
    """Block-diagonal causal mask: q attends to k iff same document and k <= q.

    ``group_ids`` widens causality within caller-defined spans:
    ``visible(q, k) = same_document AND (k <= q OR same_nonnegative_group)``.
    Tokens with a negative group id stay purely causal, and document isolation
    always dominates, so a group id reused across documents never joins them.
    Without ``doc_ids`` the document term is dropped and each row is one
    document.

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
        doc_ids: Optional per-token document ids, shape ``(batch, seq_len)``;
            ``None`` treats each row as one document.
        device: Device on which to materialize the ``BlockMask``.
        group_ids: Optional per-token bidirectional-visibility group ids,
            shape ``(batch, seq_len)``; negative means no group. ``None``
            keeps the mask exactly causal-within-document. Ids are compared
            at full int64 width, unlike ``doc_ids``, so any caller-defined id
            is usable.

    Returns:
        A ``BlockMask`` over ``(batch, seq_len, seq_len)``, head-broadcast.

    Raises:
        ValueError: If neither ``doc_ids`` nor ``group_ids`` is given.
    """
    return _build_doc_causal_block_mask(doc_ids, device, group_ids)  # type: ignore[reportCallIssue]


@torch._dynamo.disable
def _build_doc_causal_block_mask(
    doc_ids: torch.Tensor | None,
    device: torch.device,
    group_ids: torch.Tensor | None = None,
) -> BlockMask:
    """``build_doc_causal_block_mask`` body, hidden from Dynamo.

    ``create_block_mask`` is not meant to be traced by an enclosing
    ``torch.compile``. Disabling here costs one graph break at the top of the
    model forward -- not one per layer, and not one per attention call. The
    public wrapper above exists so callers (and pyright) see a real signature;
    ``torch._dynamo.disable`` erases the one it wraps.
    """
    ids = doc_ids if doc_ids is not None else group_ids
    if ids is None:
        raise ValueError("build_doc_causal_block_mask needs doc_ids or group_ids")
    batch, seq_len = ids.shape
    mask_mod = _visibility_mask_mod(
        None if doc_ids is None else doc_ids.to(device=device, dtype=torch.int32),
        None if group_ids is None else group_ids.to(device=device),
    )
    builder = _CREATE_BLOCK_MASK_COMPILED if device.type == "cuda" else create_block_mask
    return builder(mask_mod, B=batch, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device)


def _visibility_mask_mod(
    docs: torch.Tensor | None, groups: torch.Tensor | None
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """The visibility predicate as a FlexAttention ``mask_mod``, with only the terms given.

    Args:
        docs: Per-token document ids on the mask's device, or ``None`` for one
            document per row.
        groups: Per-token group ids on the mask's device, or ``None`` for no
            groups. At least one of the two is given.

    Returns:
        ``mask_mod(b, h, q_idx, kv_idx)``, ``True`` where the query may attend
        to the key.
    """
    if groups is None:
        assert docs is not None

        def doc_causal(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ) -> torch.Tensor:
            return (kv_idx <= q_idx) & (docs[b, q_idx] == docs[b, kv_idx])

        return doc_causal

    def grouped_causal(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        g_q = groups[b, q_idx]
        return (kv_idx <= q_idx) | ((g_q >= 0) & (g_q == groups[b, kv_idx]))

    if docs is None:
        return grouped_causal

    def doc_grouped_causal(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        return grouped_causal(b, h, q_idx, kv_idx) & (docs[b, q_idx] == docs[b, kv_idx])

    return doc_grouped_causal
