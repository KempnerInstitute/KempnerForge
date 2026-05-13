"""Adapter configuration.

``AdapterConfig`` selects which adapter the VLM wrapper instantiates and
parameterizes the chosen adapter. Dispatched via the ``adapter`` registry
at build time (see ``kempnerforge/model/adapter.py``).

This module is registered-component-shaped (parallel to ``VisionEncoderConfig``,
``VLMConfig``). A follow-up PR will flatten the VLM TOML schema to expose
``[adapter]`` as a top-level section; until then ``build_vlm_wrapper``
constructs an ``AdapterConfig`` internally from the existing ``VLMConfig``
fields (``adapter_hidden_dim``, ``adapter_activation``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kempnerforge.config.registry import registry


@dataclass
class AdapterConfig:
    """Selects the adapter type and parameterizes it.

    Fields:
        type: Registry key for the adapter builder. ``"mlp_2layer"`` (default)
            or ``"linear"``. Custom adapters register additional names.
        hidden_dim: Hidden width for ``mlp_2layer``. ``0`` means "match
            ``out_dim``"; ignored by ``linear``.
        activation: Activation between the two MLP projections. One of
            ``"gelu"`` (default), ``"silu"``, ``"relu"``. Ignored by
            ``linear``.
    """

    type: str = "mlp_2layer"
    hidden_dim: int = 0
    activation: str = "gelu"

    def __post_init__(self) -> None:
        # Late import: importing the adapter module triggers the
        # ``@registry.register_adapter`` decorators that populate the registry.
        # Doing this at module scope creates a circular import via
        # ``kempnerforge.model.__init__`` -> ``transformer.py`` ->
        # ``kempnerforge.config.schema`` -> ``adapter.py``.
        import kempnerforge.model.adapter  # noqa: F401, PLC0415

        registered = tuple(registry.list_adapters())
        if self.type not in registered:
            raise ValueError(
                f"Unknown adapter.type: {self.type!r}. Registered: {sorted(registered)}."
            )
        if self.hidden_dim < 0:
            raise ValueError(f"adapter.hidden_dim must be non-negative (got {self.hidden_dim})")
        if self.activation not in ("gelu", "silu", "relu"):
            raise ValueError(
                f"Unknown adapter.activation: {self.activation!r}. Options: 'gelu', 'silu', 'relu'."
            )

    def extra_kwargs(self) -> dict[str, Any]:
        """Builder kwargs beyond ``in_dim`` / ``out_dim``.

        ``hidden_dim=0`` is mapped to ``None`` so the adapter falls back to
        its own default (e.g., ``out_dim`` for ``MLP2LayerAdapter``).
        """
        return {
            "hidden_dim": self.hidden_dim or None,
            "activation": self.activation,
        }
