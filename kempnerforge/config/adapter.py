"""Adapter (connector) configuration.

``AdapterConfig`` selects which adapter the VLM wrapper instantiates and
parameterizes the chosen adapter. Dispatched via the ``adapter`` registry
at build time (see ``kempnerforge/model/adapter.py``).

In TOML, ``[adapter]`` is a top-level section parallel to ``[model]``,
``[vision_encoder]``, and ``[vlm]``. When ``[vlm]`` is set without an
``[adapter]`` section, ``JobConfig`` materializes the default ``AdapterConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from kempnerforge.config.registry import registry


@dataclass
class AdapterConfig:
    """Selects the adapter type and parameterizes it.

    Fields:
        type: Registry key for the adapter builder. Projection adapters
            ``"mlp_2layer"`` (default) / ``"linear"`` keep the token count;
            pooling adapters ``"avgpool"`` / ``"attentional_pool"`` reduce it.
        hidden_dim: Hidden width of a 2-layer MLP. For ``mlp_2layer``, ``0``
            means "match ``out_dim``". For ``avgpool``, a positive value
            replaces the single ``Linear`` after the pool with a 2-layer MLP
            head (``[pre_norm ->] Linear(in_dim, hidden_dim) -> activation ->
            Linear(hidden_dim, out_dim)``), and ``0`` keeps the ``Linear``.
        activation: Activation between the two MLP projections. One of
            ``"gelu"`` (default), ``"silu"``, ``"relu"``. Used by
            ``mlp_2layer``, and by ``avgpool`` when ``hidden_dim > 0``.
        pre_norm: Norm-registry key (``"rmsnorm"`` / ``"layernorm"``) for a norm
            ahead of the MLP's first projection (exposed as ``ln_q``); ``""``
            (default) builds none. Used by ``mlp_2layer`` over the vision
            features, and by ``avgpool`` over the pooled features when
            ``hidden_dim > 0``.
        pool_window: Pooling kernel side for the pooling adapters (e.g. ``2``
            for image 2×2, ``3`` for video 3×3).
        pool_heads: Number of attention heads for ``attentional_pool``; must
            divide the vision feature dim.

    A field the selected adapter does not use must keep its default, or
    construction raises ``ValueError`` naming the field and the adapter type.
    Each adapter declares the fields it uses when it registers (``accepts`` in
    ``registry.register_adapter``); one that declares nothing accepts every
    field.
    """

    type: str = "mlp_2layer"
    hidden_dim: int = 0
    activation: str = "gelu"
    pre_norm: str = ""
    pool_window: int = 2
    pool_heads: int = 16

    def __post_init__(self) -> None:
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
        if self.pre_norm:
            import kempnerforge.model.norm  # noqa: F401, PLC0415

            norms = tuple(registry.list("norm"))
            if self.pre_norm not in norms:
                raise ValueError(
                    f"Unknown adapter.pre_norm: {self.pre_norm!r}. Registered norms: "
                    f"{sorted(norms)}; '' builds none."
                )
        if self.pool_window <= 0:
            raise ValueError(f"adapter.pool_window must be positive (got {self.pool_window})")
        if self.pool_heads <= 0:
            raise ValueError(f"adapter.pool_heads must be positive (got {self.pool_heads})")
        self._reject_unused_fields()

    def _reject_unused_fields(self) -> None:
        declared = registry.get_adapter_accepts(self.type)
        if declared is None:
            return
        named = declared(self) if callable(declared) else declared
        if isinstance(named, str):
            raise TypeError(
                f"accepts for adapter {self.type!r} must return a collection of field names, "
                f"not the string {named!r}"
            )
        used = frozenset(named)
        known = {f.name for f in fields(self)}
        unknown = sorted(used - known)
        if unknown:
            raise ValueError(
                f"adapter.type = {self.type!r} declares fields AdapterConfig does not have: "
                f"{unknown}. AdapterConfig fields: {sorted(known)}."
            )
        unused = [
            f
            for f in fields(self)
            if f.name != "type" and f.name not in used and getattr(self, f.name) != f.default
        ]
        if unused:
            settings = ", ".join(f"adapter.{f.name} = {getattr(self, f.name)!r}" for f in unused)
            defaults = ", ".join(f"{f.name} = {f.default!r}" for f in unused)
            uses = ", ".join(sorted(used)) or "no adapter fields"
            raise ValueError(
                f"adapter.type = {self.type!r} does not use {settings}; with these settings "
                f"it uses {uses}. Leave unused fields at their defaults ({defaults})."
            )

    def extra_kwargs(self) -> dict[str, Any]:
        """Builder kwargs beyond ``in_dim`` / ``out_dim``.

        ``hidden_dim=0`` maps to ``None``, which each adapter reads as its own
        default (``out_dim`` wide for ``MLP2LayerAdapter``, no MLP head for
        ``AvgPoolAdapter``). Every field is passed; a builder takes the ones
        it uses and swallows the rest via ``**_``, which construction has
        checked are at their defaults when the adapter declares its fields.
        """
        return {
            "hidden_dim": self.hidden_dim or None,
            "activation": self.activation,
            "pre_norm": self.pre_norm or None,
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
        from kempnerforge.model.adapter import pooled_token_count  # noqa: PLC0415

        return pooled_token_count(num_input_tokens, self.pool_window)

    @staticmethod
    def _pooling_types() -> tuple[str, ...]:
        from kempnerforge.model.adapter import POOLING_ADAPTER_TYPES  # noqa: PLC0415

        return POOLING_ADAPTER_TYPES
