"""Multi-head attention with Grouped-Query Attention (GQA) support.

GQA is the general case:
  - n_kv_heads == n_heads → standard Multi-Head Attention (MHA)
  - n_kv_heads == 1       → Multi-Query Attention (MQA)
  - 1 < n_kv_heads < n_heads → Grouped-Query Attention (GQA)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from kempnerforge.model.position import apply_rope


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

    def update(
        self, k_new: torch.Tensor, v_new: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
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


class Attention(nn.Module):
    """Grouped-Query Attention with RoPE and SDPA."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim or (dim // n_heads)
        self.n_rep = n_heads // n_kv_heads  # GQA repetition factor

        # Projections
        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        *,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, dim).
            rope_cos: RoPE cosine frequencies, shape (seq_len, head_dim // 2).
            rope_sin: RoPE sine frequencies, shape (seq_len, head_dim // 2).
            kv_cache: Optional KV cache for incremental generation.

        Returns:
            Output tensor of shape (batch, seq_len, dim).
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        # Use -1 for head count so the view works under tensor parallelism
        # (ColwiseParallel shards the output features, reducing the local head count)
        q = self.q_proj(x).view(batch, seq_len, -1, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, -1, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, -1, self.head_dim)

        # Transpose to (batch, heads, seq_len, head_dim) for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE to Q and K
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        # Update KV cache (after RoPE, before GQA expansion)
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        # Expand KV heads for GQA: (batch, n_kv_heads, seq, dim) → (batch, n_heads, seq, dim)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Causal masking: needed for training and prefill (Q seq == K seq).
        # During single-token decode (seq_len=1), the query attends to all
        # cached positions — no causal mask needed (and is_causal=True would
        # incorrectly restrict attention to only the first key position).
        is_causal = kv_cache is None or seq_len > 1

        # Scaled dot-product attention (auto-dispatches to FlashAttention/MemEfficient)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        # Reshape back: (batch, n_heads, seq_len, head_dim) → (batch, seq_len, dim)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)

        return self.o_proj(out)
