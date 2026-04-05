"""Rotary Position Embedding (RoPE) for KempnerForge models.

Uses real-valued sin/cos rotation (not complex arithmetic) for
compatibility with DTensor and SequenceParallel.
"""

from __future__ import annotations

import torch


def precompute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin RoPE frequency tables.

    Args:
        head_dim: Dimension per attention head (must be even).
        max_seq_len: Maximum sequence length to precompute.
        theta: Base frequency (10000.0 for standard RoPE).
        device: Device to place the tensor on.

    Returns:
        Tuple of (cos, sin) tensors, each shape (max_seq_len, head_dim // 2).
    """
    assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"

    # Frequency for each dimension pair: theta^{-2i/d} for i in [0, d/2)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))

    # Position indices
    positions = torch.arange(max_seq_len, device=device)

    # Outer product: (max_seq_len, head_dim // 2)
    freqs_table = torch.outer(positions, freqs)

    return freqs_table.cos(), freqs_table.sin()


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embeddings using real-valued rotation.

    Args:
        x: Input tensor of shape (..., seq_len, head_dim).
        cos: Cosine frequencies, shape (seq_len, head_dim // 2).
        sin: Sine frequencies, shape (seq_len, head_dim // 2).

    Returns:
        Tensor with RoPE applied, same shape and dtype as input.
    """
    # Split head dim into two halves for paired rotation
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]

    # Broadcast cos/sin to match x shape: (seq_len, d) → (..., seq_len, d)
    # Cast cos/sin to x's dtype (bf16) instead of casting x to float32,
    # because .float() strips DTensor metadata needed for SequenceParallel.
    ndim = x.ndim
    shape = [1] * (ndim - 2) + list(cos.shape)
    cos = cos.view(*shape).to(x.dtype)
    sin = sin.view(*shape).to(x.dtype)

    # Rotation: [x1, x2] → [x1*cos - x2*sin, x2*cos + x1*sin]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
