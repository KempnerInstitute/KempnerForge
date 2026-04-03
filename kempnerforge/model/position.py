"""Rotary Position Embedding (RoPE) for KempnerForge models."""

from __future__ import annotations

import torch


def precompute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Precompute the complex-valued RoPE frequency table.

    Returns a tensor of shape (max_seq_len, head_dim // 2) containing complex
    exponentials e^{i * pos * freq} for each position and frequency pair.

    Args:
        head_dim: Dimension per attention head (must be even).
        max_seq_len: Maximum sequence length to precompute.
        theta: Base frequency (10000.0 for standard RoPE).
        device: Device to place the tensor on.

    Returns:
        Complex tensor of shape (max_seq_len, head_dim // 2).
    """
    assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"

    # Frequency for each dimension pair: theta^{-2i/d} for i in [0, d/2)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))

    # Position indices
    positions = torch.arange(max_seq_len, device=device)

    # Outer product: (max_seq_len, head_dim // 2)
    freqs_table = torch.outer(positions, freqs)

    # Complex exponentials: e^{i * theta}
    return torch.polar(torch.ones_like(freqs_table), freqs_table)


def apply_rope(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embeddings to a tensor.

    Args:
        x: Input tensor of shape (..., seq_len, head_dim).
        freqs_cis: Complex frequency tensor of shape (seq_len, head_dim // 2)
            or broadcastable prefix.

    Returns:
        Tensor with RoPE applied, same shape and dtype as input.
    """
    # Reshape x into pairs: (..., seq_len, head_dim//2, 2) and view as complex
    dtype = x.dtype
    x_pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_pairs)

    # Broadcast freqs_cis to match x_complex shape
    # x_complex: (batch, n_heads, seq_len, head_dim//2)
    # freqs_cis: (seq_len, head_dim//2) → need to reshape for broadcasting
    ndim = x_complex.ndim
    shape = [1] * (ndim - 2) + list(freqs_cis.shape)
    freqs_cis = freqs_cis.view(*shape)

    # Apply rotation in complex space and convert back to real pairs
    x_rotated = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    return x_rotated.to(dtype)
