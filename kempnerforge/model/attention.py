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
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, dim).
            rope_cos: RoPE cosine frequencies, shape (seq_len, head_dim // 2).
            rope_sin: RoPE sine frequencies, shape (seq_len, head_dim // 2).

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

        # Expand KV heads for GQA: (batch, n_kv_heads, seq, dim) → (batch, n_heads, seq, dim)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Scaled dot-product attention (auto-dispatches to FlashAttention/MemEfficient)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Reshape back: (batch, n_heads, seq_len, head_dim) → (batch, seq_len, dim)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)

        return self.o_proj(out)
