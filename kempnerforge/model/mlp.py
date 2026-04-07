"""Feed-forward network implementations for KempnerForge models."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from kempnerforge.config.registry import registry


class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward network (Llama-style).

    Architecture: gate_proj + up_proj → SiLU(gate) * up → down_proj
    Uses 3 weight matrices instead of 2, with SiLU gating.
    """

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class StandardMLP(nn.Module):
    """Standard two-layer MLP with configurable activation.

    Architecture: linear → activation → linear
    """

    def __init__(self, dim: int, hidden_dim: int, activation: str = "gelu") -> None:
        super().__init__()
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

        activations = {"gelu": F.gelu, "relu": F.relu, "silu": F.silu}
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation!r}")
        self._activation = activations[activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self._activation(self.up_proj(x)))


def _build_swiglu(dim: int, hidden_dim: int) -> SwiGLUMLP:
    return SwiGLUMLP(dim, hidden_dim)


def _build_standard_gelu(dim: int, hidden_dim: int) -> StandardMLP:
    return StandardMLP(dim, hidden_dim, activation="gelu")


def _build_standard_relu(dim: int, hidden_dim: int) -> StandardMLP:
    return StandardMLP(dim, hidden_dim, activation="relu")


registry.register("mlp", "swiglu", _build_swiglu)
registry.register("mlp", "standard_gelu", _build_standard_gelu)
registry.register("mlp", "standard_relu", _build_standard_relu)


# Map activation config names to registry keys
_ACTIVATION_TO_MLP = {"silu": "swiglu", "gelu": "standard_gelu", "relu": "standard_relu"}


def build_mlp(dim: int, hidden_dim: int, activation: str = "silu") -> nn.Module:
    """Build an MLP by activation name.

    SiLU activation uses SwiGLU (3 matrices); others use standard MLP (2 matrices).
    """
    key = _ACTIVATION_TO_MLP.get(activation, activation)
    builder = registry.get("mlp", key)
    return builder(dim, hidden_dim)
