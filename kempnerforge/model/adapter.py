"""Vision-to-LLM adapter modules.

The adapter projects image features (shape ``(B, num_tokens, feature_dim)``)
into the LLM embedding space (shape ``(B, num_tokens, model.dim)``). It sits
between the vision encoder and the transformer in ``VLMWrapper``.

Adapters register themselves under the ``adapter`` registry category. The
default is ``mlp_2layer`` (a 2-layer MLP, the canonical adapter shape across
LLaVA-family papers). ``linear`` is a single ``nn.Linear`` with no
activation, useful for ablations.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from kempnerforge.config.registry import registry

_ADAPTER_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "relu": nn.ReLU,
}


class MLP2LayerAdapter(nn.Module):
    """2-layer MLP from image-feature dim to LLM embedding dim.

    Architecture: ``Linear(in_dim, hidden) -> activation -> Linear(hidden, out_dim)``.
    ``hidden_dim=None`` defaults to ``out_dim``.

    ``reset_parameters`` is provided so callers that materialize adapters
    from meta can re-initialize weights with the standard Linear defaults.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int | None = None,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError("MLP2LayerAdapter in_dim and out_dim must be positive")
        if activation not in _ADAPTER_ACTIVATIONS:
            raise ValueError(
                f"Unknown adapter activation: {activation!r}. Options: {list(_ADAPTER_ACTIVATIONS)}"
            )
        hidden = hidden_dim if hidden_dim and hidden_dim > 0 else out_dim
        self.proj1 = nn.Linear(in_dim, hidden, bias=True)
        self.act = _ADAPTER_ACTIVATIONS[activation]()
        self.proj2 = nn.Linear(hidden, out_dim, bias=True)

    def reset_parameters(self) -> None:
        """Re-run ``nn.Linear`` default init on both projections.

        Used after ``to_empty(device=...)`` on a meta-device build.
        """
        self.proj1.reset_parameters()
        self.proj2.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj2(self.act(self.proj1(x)))


class LinearAdapter(nn.Module):
    """Single ``nn.Linear`` from image-feature dim to LLM embedding dim.

    No activation, no hidden layer. Useful as an ablation baseline against
    ``MLP2LayerAdapter``.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError("LinearAdapter in_dim and out_dim must be positive")
        self.proj = nn.Linear(in_dim, out_dim, bias=True)

    def reset_parameters(self) -> None:
        self.proj.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


@registry.register_adapter("mlp_2layer")
def _build_mlp_2layer(
    in_dim: int,
    out_dim: int,
    hidden_dim: int | None = None,
    activation: str = "gelu",
    **_: Any,
) -> MLP2LayerAdapter:
    return MLP2LayerAdapter(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_dim=hidden_dim,
        activation=activation,
    )


@registry.register_adapter("linear")
def _build_linear(
    in_dim: int,
    out_dim: int,
    **_: Any,
) -> LinearAdapter:
    return LinearAdapter(in_dim=in_dim, out_dim=out_dim)


def build_adapter(adapter_config, in_dim: int, out_dim: int) -> nn.Module:
    """Dispatch to the registered adapter builder.

    Args:
        adapter_config: ``AdapterConfig`` (or compatible object exposing
            ``type`` and ``extra_kwargs()``).
        in_dim: Source feature dim (the vision encoder's ``feature_dim``).
        out_dim: Target embedding dim (the transformer's ``dim``).

    Returns:
        An ``nn.Module`` with signature ``(B, N, in_dim) -> (B, N, out_dim)``.
    """
    builder = registry.get_adapter(adapter_config.type)
    return builder(in_dim=in_dim, out_dim=out_dim, **adapter_config.extra_kwargs())
