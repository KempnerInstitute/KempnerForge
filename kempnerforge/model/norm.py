"""Normalization layers for KempnerForge models."""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Llama-style).

    Simpler and faster than LayerNorm — no mean subtraction, no bias.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # float32 for numerical stability, then cast back
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight).to(dtype)


def build_norm(norm_type: str, dim: int, eps: float = 1e-5) -> nn.Module:
    """Build a normalization layer by name."""
    if norm_type == "rmsnorm":
        return RMSNorm(dim, eps=eps)
    if norm_type == "layernorm":
        return nn.LayerNorm(dim, eps=eps)
    raise ValueError(f"Unknown norm_type: {norm_type!r}. Expected 'rmsnorm' or 'layernorm'.")
