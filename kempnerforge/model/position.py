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

    Raises:
        ValueError: ``head_dim`` is odd.
    """
    if head_dim % 2:
        raise ValueError(f"head_dim must be even for rotary embeddings, got {head_dim}")

    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))

    positions = torch.arange(max_seq_len, device=device)

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
        cos: Cosine frequencies, shape (seq_len, head_dim // 2) shared across
            batch rows, or (batch, seq_len, head_dim // 2) per-row (gathered
            from explicit position ids).
        sin: Sine frequencies, same shape as ``cos``.

    Returns:
        Tensor with RoPE applied, same shape and dtype as input.
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]

    ndim = x.ndim
    if cos.ndim == 2:
        shape = [1] * (ndim - 2) + list(cos.shape)
    elif cos.ndim == 3:
        shape = [cos.shape[0]] + [1] * (ndim - 3) + list(cos.shape[1:])
    else:
        raise ValueError(
            "cos/sin must have shape (seq_len, head_dim // 2) or "
            f"(batch, seq_len, head_dim // 2), got {cos.ndim} dims"
        )
    cos = cos.view(*shape).to(x.dtype)
    sin = sin.view(*shape).to(x.dtype)

    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
