"""Cross-attention block for VLM Cross-Attention architecture.

Text queries cross-attend to image keys/values. Inserted between
regular ``TransformerBlock``s at a configurable cadence; the residual
stream itself carries text only, with image K/V flowing in as a side
channel via ``ModalityContext.image_features``.

Differences from ``Attention``:

- No causal mask on the image axis. Image tokens have no temporal
  order that aligns with text positions, so the full image K/V set is
  visible from every text position.
- No RoPE. RoPE encodes relative position along a single axis;
  cross-attention spans two axes (text Q positions vs image K/V
  positions) with no shared coordinate, so RoPE is dropped on both
  sides. The text axis already has RoPE applied inside each
  preceding ``TransformerBlock``; cross-attention just queries off
  the resulting hidden state.
- Output projection ``o_proj`` and the block's MLP output projection
  are zero-initialized so the block starts as identity. This matches
  Llama-3-V's warm-start: a CA arch added to an existing text-only
  checkpoint contributes zero gradient at step 0 and learns a non-zero
  contribution from there.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from kempnerforge.model.mlp import build_mlp
from kempnerforge.model.norm import build_norm


class CrossAttention(nn.Module):
    """Text Q × image K/V cross-attention with optional GQA."""

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

        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        # Zero-init output projection so the block is identity at construction.
        nn.init.zeros_(self.o_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        image_features: torch.Tensor,
        image_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Text hidden state, shape ``(batch, seq_len, dim)``.
            image_features: Image K/V source, shape
                ``(batch, num_image_tokens, dim)``.
            image_mask: Optional bool mask, shape
                ``(batch, num_image_tokens)``; ``True`` = attend, ``False`` =
                mask out. ``None`` = all image tokens attended to.

        Returns:
            Output tensor of shape ``(batch, seq_len, dim)``.
        """
        batch, seq_len, _ = x.shape
        num_image_tokens = image_features.shape[1]

        # Q from text, K/V from image features. Use -1 for head count so the
        # view works under tensor parallelism (ColwiseParallel shards out_features).
        q = self.q_proj(x).view(batch, seq_len, -1, self.head_dim)
        k = self.k_proj(image_features).view(batch, num_image_tokens, -1, self.head_dim)
        v = self.v_proj(image_features).view(batch, num_image_tokens, -1, self.head_dim)

        # Transpose to (batch, heads, seq, head_dim) for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Expand K/V heads for GQA
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Build SDPA attn_mask from image_mask if present.
        # image_mask: (B, N) bool, True = attend. SDPA accepts a bool mask
        # broadcastable to (B, n_heads, S_q, S_kv); shape (B, 1, 1, N) does
        # the right thing across heads and text Q positions.
        attn_mask = None
        if image_mask is not None:
            attn_mask = image_mask.view(batch, 1, 1, num_image_tokens)

        # Cross-attention: no causal mask on the image axis.
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(out)


class CrossAttentionBlock(nn.Module):
    """Pre-norm wrapper: ``CrossAttention`` + residual + MLP + residual.

    Mirrors ``TransformerBlock``'s outer shape so the freeze + FSDP +
    DCP plumbing applies uniformly. The MLP's output projection is also
    zero-initialized so the whole block is identity at construction.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        ffn_hidden_dim: int,
        norm_type: str = "rmsnorm",
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.attn_norm = build_norm(norm_type, dim)
        self.attn = CrossAttention(dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads)
        self.mlp_norm = build_norm(norm_type, dim)
        self.mlp = build_mlp(dim=dim, hidden_dim=ffn_hidden_dim, activation=activation)

        # Zero-init MLP output projection. SwiGLU uses down_proj; StandardMLP
        # also uses down_proj. Both are nn.Linear, so set the weight to zero.
        # type ignore: build_mlp returns nn.Module statically; both concrete
        # subclasses (SwiGLUMLP, StandardMLP) expose .down_proj.weight, but
        # pyright cannot narrow through build_mlp's return type.
        nn.init.zeros_(self.mlp.down_proj.weight)  # type: ignore[union-attr]

    def forward(
        self,
        x: torch.Tensor,
        image_features: torch.Tensor,
        image_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), image_features, image_mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x
