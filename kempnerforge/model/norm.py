"""Normalization layers for KempnerForge models."""

from __future__ import annotations

import torch
import torch.nn as nn

from kempnerforge.config.registry import registry


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


def _build_rmsnorm(dim: int, eps: float = 1e-5) -> RMSNorm:
    return RMSNorm(dim, eps=eps)


def _build_layernorm(dim: int, eps: float = 1e-5) -> nn.LayerNorm:
    return nn.LayerNorm(dim, eps=eps)


registry.register("norm", "rmsnorm", _build_rmsnorm)
registry.register("norm", "layernorm", _build_layernorm)


def build_norm(norm_type: str, dim: int, eps: float = 1e-5) -> nn.Module:
    """Build a normalization layer by name."""
    builder = registry.get("norm", norm_type)
    return builder(dim, eps=eps)
