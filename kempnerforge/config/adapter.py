"""Adapter (connector) configuration.

``AdapterConfig`` selects which adapter the VLM wrapper instantiates and
parameterizes the chosen adapter. Dispatched via the ``adapter`` registry
at build time (see ``kempnerforge/model/adapter.py``).

In TOML, ``[adapter]`` is a top-level section parallel to ``[model]``,
``[vision_encoder]``, and ``[vlm]``. When ``[vlm]`` is set without an
``[adapter]`` section, ``JobConfig`` materializes the default ``AdapterConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kempnerforge.config.registry import registry


@dataclass
class AdapterConfig:
    """Selects the adapter type and parameterizes it.

    Fields:
        type: Registry key for the adapter builder. Projection adapters
            ``"mlp_2layer"`` (default) / ``"linear"`` keep the token count;
            pooling adapters ``"avgpool"`` / ``"attentional_pool"`` reduce it.
        hidden_dim: Hidden width for ``mlp_2layer``. ``0`` means "match
            ``out_dim``"; ignored by the other types.
        activation: Activation between the two MLP projections. One of
            ``"gelu"`` (default), ``"silu"``, ``"relu"``. ``mlp_2layer`` only.
        pool_window: Pooling kernel side for the pooling adapters (e.g. ``2``
            for image 2×2, ``3`` for video 3×3); ignored by projection adapters.
        pool_heads: Number of attention heads for ``attentional_pool``; must
            divide the vision feature dim. Ignored by the other types.
    """

    type: str = "mlp_2layer"
    hidden_dim: int = 0
    activation: str = "gelu"
    pool_window: int = 2
    pool_heads: int = 16

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
        if self.pool_window <= 0:
            raise ValueError(f"adapter.pool_window must be positive (got {self.pool_window})")
        if self.pool_heads <= 0:
            raise ValueError(f"adapter.pool_heads must be positive (got {self.pool_heads})")

    def extra_kwargs(self) -> dict[str, Any]:
        """Builder kwargs beyond ``in_dim`` / ``out_dim``.

        ``hidden_dim=0`` is mapped to ``None`` so the adapter falls back to
        its own default (e.g., ``out_dim`` for ``MLP2LayerAdapter``). Pooling
        kwargs are always passed; projection builders swallow them via ``**_``.
        """
        return {
            "hidden_dim": self.hidden_dim or None,
            "activation": self.activation,
            "pool_window": self.pool_window,
            "pool_heads": self.pool_heads,
        }

    def output_num_tokens(self, num_input_tokens: int) -> int:
        """Predict the post-adapter token count for ``num_input_tokens`` in.

        Mirrors the built module's ``output_num_tokens`` so config-time
        sequence-length checks match the build-time/runtime token budget.
        Projection adapters are the identity; pooling adapters apply the
        shared ``pooled_token_count`` math. Non-positive inputs (e.g. the
        ``num_tokens=0`` "infer at build time" sentinel) pass through.
        """
        if num_input_tokens <= 0 or self.type not in self._pooling_types():
            return num_input_tokens
        from kempnerforge.model.adapter import (  # noqa: PLC0415
            DIVISIBLE_ONLY_POOL_TYPES,
            pooled_token_count,
        )

        return pooled_token_count(
            num_input_tokens,
            self.pool_window,
            require_divisible=self.type in DIVISIBLE_ONLY_POOL_TYPES,
        )

    @staticmethod
    def _pooling_types() -> tuple[str, ...]:
        from kempnerforge.model.adapter import POOLING_ADAPTER_TYPES  # noqa: PLC0415

        return POOLING_ADAPTER_TYPES
