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

from kempnerforge.model.norm import RMSNorm
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


class Attention(nn.Module):
    """Grouped-Query Attention with RoPE and SDPA."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int | None = None,
        qk_norm: bool = False,
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

        # Per-head QK normalization (Gemma, DeepSeek-V3)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else None

        # Attention weight capture (analysis only — not for training)
        self.capture_attention_weights = False
        self._last_attention_weights: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        *,
        kv_cache: KVCache | None = None,
        doc_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, dim).
            rope_cos: RoPE cosine frequencies, shape (seq_len, head_dim // 2).
            rope_sin: RoPE sine frequencies, shape (seq_len, head_dim // 2).
            kv_cache: Optional KV cache for incremental generation.
            doc_ids: Optional per-token document IDs for packed sequences,
                shape (batch, seq_len). When provided, constructs a block-diagonal
                causal mask so tokens only attend within their document.

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

        # QK-Norm: normalize Q and K per-head before RoPE (stabilizes attention logits)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

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

        # Attention masking strategy:
        # 1. Packed sequences (doc_ids provided): block-diagonal causal mask that
        #    isolates documents from each other within a packed sequence.
        # 2. Standard causal: needed for training and prefill (Q seq == K seq).
        # 3. Single-token decode (seq_len=1 with KV cache): no mask needed — the
        #    query attends to all cached positions (is_causal=True would incorrectly
        #    restrict attention to only the first key position).
        if self.capture_attention_weights:
            # Manual attention for weight extraction (analysis only, not for training)
            out, attn_weights = self._attention_with_weights(q, k, v, seq_len, doc_ids, kv_cache)
            self._last_attention_weights = attn_weights.detach().cpu()
        elif doc_ids is not None:
            seq_len_kv = k.shape[2]
            # Block-diagonal mask: same-document AND causal
            doc_mask = doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)  # (B, S, S)
            causal = torch.ones(seq_len, seq_len_kv, dtype=torch.bool, device=q.device).tril()
            attn_mask = (doc_mask & causal).unsqueeze(1)  # (B, 1, S, S)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            is_causal = kv_cache is None or seq_len > 1
            out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        # Reshape back: (batch, n_heads, seq_len, head_dim) → (batch, seq_len, dim)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)

        return self.o_proj(out)

    def _attention_with_weights(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_len: int,
        doc_ids: torch.Tensor | None,
        kv_cache: KVCache | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attention output and weights manually (for analysis).

        SDPA does not return attention weights, so this path computes
        softmax(QK^T / sqrt(d)) explicitly. Only use for interpretability —
        it is slower and uses more memory than the fused SDPA path.

        Returns:
            Tuple of (output, attention_weights).
            output: (batch, n_heads, seq_len, head_dim)
            attention_weights: (batch, n_heads, seq_q, seq_k)
        """
        scale = self.head_dim**-0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        seq_len_kv = k.shape[2]
        if doc_ids is not None:
            doc_mask = doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)
            causal = torch.ones(seq_len, seq_len_kv, dtype=torch.bool, device=q.device).tril()
            mask = ~(doc_mask & causal).unsqueeze(1)
            attn = attn.masked_fill(mask, float("-inf"))
        elif kv_cache is None or seq_len > 1:
            causal = torch.ones(seq_len, seq_len_kv, dtype=torch.bool, device=q.device).triu(
                diagonal=1
            )
            attn = attn.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(attn, dim=-1)
        out = torch.matmul(attn_weights, v)
        return out, attn_weights
