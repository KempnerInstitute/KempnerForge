"""Multi-head attention with Grouped-Query Attention (GQA) support.

GQA is the general case:
  - n_kv_heads == n_heads → standard Multi-Head Attention (MHA)
  - n_kv_heads == 1       → Multi-Query Attention (MQA)
  - 1 < n_kv_heads < n_heads → Grouped-Query Attention (GQA)
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import BlockMask

from kempnerforge.model.masking import flex_attention_fn
from kempnerforge.model.norm import RMSNorm
from kempnerforge.model.position import apply_rope

_SDPA_BACKENDS = {
    "flash": SDPBackend.FLASH_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
    "math": SDPBackend.MATH,
}
_MASK_FALLBACK_BACKENDS = frozenset({"flash", "cudnn"})


class KVCache:
    """Pre-allocated KV cache for autoregressive generation.

    Stores key and value tensors for all previous positions, enabling
    incremental decoding without recomputing attention over the full sequence.
    Keys are stored after RoPE application but before GQA expansion.
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.k = torch.zeros(
            batch_size, n_kv_heads, max_seq_len, head_dim, dtype=dtype, device=device
        )
        self.v = torch.zeros(
            batch_size, n_kv_heads, max_seq_len, head_dim, dtype=dtype, device=device
        )
        self.seq_len = 0

    def update(self, k_new: torch.Tensor, v_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new key/value entries and return full cached tensors.

        Args:
            k_new: New keys, shape (batch, n_kv_heads, new_seq_len, head_dim).
            v_new: New values, shape (batch, n_kv_heads, new_seq_len, head_dim).

        Returns:
            Tuple of (all_keys, all_values), each
            (batch, n_kv_heads, total_seq_len, head_dim).
        """
        new_len = k_new.shape[2]
        end = self.seq_len + new_len
        self.k[:, :, self.seq_len : end] = k_new
        self.v[:, :, self.seq_len : end] = v_new
        self.seq_len = end
        return self.k[:, :, :end], self.v[:, :, :end]


def appended_causal_mask(seq_len: int, past: int, device: torch.device) -> torch.Tensor:
    """Visibility of ``seq_len`` queries appended after ``past`` cached keys; ``True`` attends."""
    return torch.ones(seq_len, past + seq_len, dtype=torch.bool, device=device).tril(diagonal=past)


class Attention(nn.Module):
    """Grouped-Query Attention with RoPE and SDPA."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int | None = None,
        qk_norm: bool = False,
        sdpa_backend: str = "auto",
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim or (dim // n_heads)
        self.n_rep = n_heads // n_kv_heads
        self.sdpa_backend = sdpa_backend

        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=norm_eps) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, eps=norm_eps) if qk_norm else None

        self.capture_attention_weights = False
        self.last_attention_weights: torch.Tensor | None = None

    def _sdpa_backends(self, *, appended: bool = False) -> list[SDPBackend]:
        """Return the SDPA backends enabled for one call under a pinned ``sdpa_backend``.

        The flash and cudnn kernels reject a dense mask. ``appended=True`` is
        the decode-time call that attends several appended tokens to a
        non-empty cache through a lower-right mask; it keeps the pinned kernel
        first and also enables the efficient and math kernels, in that priority
        order, which the dispatcher falls back to for that call instead of
        raising. A packed training batch keeps the pinned kernel alone and
        fails loudly as before, so a job never runs on a slower kernel
        unannounced.
        """
        backends = [_SDPA_BACKENDS[self.sdpa_backend]]
        if appended and self.sdpa_backend in _MASK_FALLBACK_BACKENDS:
            backends += [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
        return backends

    def _sdpa_context(self, *, appended: bool = False):
        """Return a context manager that forces a specific SDPA backend.

        When sdpa_backend="auto", returns a no-op and lets PyTorch's
        heuristics select the backend. See ``_sdpa_backends`` for ``appended``.
        """
        if self.sdpa_backend == "auto":
            return contextlib.nullcontext()
        return sdpa_kernel(self._sdpa_backends(appended=appended), set_priority=True)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        *,
        kv_cache: KVCache | None = None,
        attn_mask: torch.Tensor | BlockMask | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Attention consumes a mask; it never derives a visibility rule.
        Document isolation, bidirectional groups, key validity and their
        combination are expressed by ``kempnerforge.model.masking``, which
        builds this one object once per forward and shares it across layers.
        The causal structure itself is attention's own: ``is_causal`` for a
        plain forward, and its lower-right form when several tokens are
        appended to a non-empty cache, which the transformer builds once per
        forward and every layer receives as ``attn_mask``.

        Args:
            x: Input tensor of shape (batch, seq_len, dim).
            rope_cos: RoPE cosine frequencies, shape (seq_len, head_dim // 2) shared
                across the batch or (batch, seq_len, head_dim // 2) per row.
            rope_sin: RoPE sine frequencies, same shape as ``rope_cos``.
            kv_cache: Optional KV cache for incremental generation. The
                tokens of this forward are appended after the cached ones and
                see every cached key plus themselves causally. A visibility
                mask from ``build_attention_mask`` does not compose with a
                cache: it is square over the residual, while the key axis
                grows every step.
            attn_mask: The mask from
                ``kempnerforge.model.masking.build_attention_mask``: a
                FlexAttention ``BlockMask``, a dense bool tensor broadcastable
                to (batch, n_heads, seq_len, seq_len) where ``True`` = attend,
                or ``None`` for plain causal attention. With a cache it is the
                lower-right causal mask ``Transformer.forward`` builds once
                from ``appended_causal_mask`` when several tokens are appended
                to a non-empty cache, (seq_len, past + seq_len); a layer
                called on its own without it derives the same mask.

        Returns:
            Output tensor of shape (batch, seq_len, dim).

        Raises:
            NotImplementedError: If ``capture_attention_weights`` is set and
                ``attn_mask`` is a ``BlockMask``.
        """
        if isinstance(attn_mask, BlockMask) and self.capture_attention_weights:
            raise NotImplementedError(
                "capture_attention_weights requires attention_backend='sdpa'. "
                "FlexAttention fuses the mask into the kernel and never materializes "
                "the attention weight matrix."
            )

        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, -1, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, -1, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, -1, self.head_dim)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)  # type: ignore[reportOptionalCall]

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        past = kv_cache.seq_len if kv_cache is not None else 0
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        if self.n_rep > 1 and not isinstance(attn_mask, BlockMask):
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        if attn_mask is not None and past > 0 and attn_mask.shape[-1] != k.shape[2]:
            raise ValueError(
                f"attn_mask spans {attn_mask.shape[-1]} keys but the cache holds "
                f"{k.shape[2]}; a visibility mask is square over the residual and does "
                "not compose with a non-empty kv_cache"
            )

        if isinstance(attn_mask, BlockMask):
            assert kv_cache is None, (
                "a BlockMask does not compose with kv_cache decode; a decode step "
                "re-derives its own mask as the key axis grows."
            )
            out = flex_attention_fn(q.is_cuda)(
                q,
                k,
                v,
                block_mask=attn_mask,
                scale=self.head_dim**-0.5,
                enable_gqa=self.n_rep > 1,
            )
        elif self.capture_attention_weights:
            out, attn_weights = self._attention_with_weights(q, k, v, past, attn_mask)
            self.last_attention_weights = attn_weights.detach().cpu()
        elif attn_mask is not None:
            with self._sdpa_context(appended=past > 0):
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            if past == 0:
                with self._sdpa_context():
                    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            elif seq_len == 1:
                with self._sdpa_context():
                    out = F.scaled_dot_product_attention(q, k, v)
            else:
                visible = appended_causal_mask(seq_len, past, q.device)
                with self._sdpa_context(appended=True):
                    out = F.scaled_dot_product_attention(q, k, v, attn_mask=visible)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)

        return self.o_proj(out)

    def _attention_with_weights(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        past: int,
        attn_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attention output and weights manually (for analysis).

        SDPA does not return attention weights, so this path computes
        softmax(QK^T / sqrt(d)) explicitly. Only use for interpretability —
        it is slower and uses more memory than the fused SDPA path.

        Args:
            q: Queries, shape (batch, n_heads, seq_len, head_dim).
            k: Keys, shape (batch, n_heads, seq_k, head_dim).
            v: Values, shape (batch, n_heads, seq_k, head_dim).
            past: Number of cached keys the queries are appended after.
            attn_mask: Dense bool mask where ``True`` = attend, or ``None`` for
                causal attention over the keys, cached ones included.

        Returns:
            Tuple of (output, attention_weights).
            output: (batch, n_heads, seq_len, head_dim)
            attention_weights: (batch, n_heads, seq_q, seq_k)
        """
        scale = self.head_dim**-0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attn_mask is not None:
            attn = attn.masked_fill(~attn_mask, float("-inf"))
        elif q.shape[2] > 1:
            visible = appended_causal_mask(q.shape[2], past, q.device)
            attn = attn.masked_fill(~visible, float("-inf"))

        attn_weights = F.softmax(attn, dim=-1)
        out = torch.matmul(attn_weights, v)
        return out, attn_weights
