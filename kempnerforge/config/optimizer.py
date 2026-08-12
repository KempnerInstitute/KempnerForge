"""Optimizer configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OptimizerConfig:
    """Optimizer settings."""

    name: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    fused: bool = True  # Use fused AdamW when available
    # Muon-specific
    muon_momentum: float = 0.95  # Momentum coefficient for Muon
    muon_ns_steps: int = 5  # Newton-Schulz iteration steps for Muon
    muon_adam_lr: float | None = None  # LR for 1D params in Muon's AdamW fallback; None=same as lr
    # Schedule-Free specific
    schedule_free_warmup_steps: int = 0  # Internal warmup for schedule-free optimizer

    # Per-module LR, keyed by ``VLMConfig.module_patterns`` names; a module's
    # rate is ``lr * multiplier`` and anything unlisted stays at ``lr``.
    # Multipliers rather than absolute rates so the scheduler keeps scaling
    # every group by one factor. Lets a pretrained encoder train slower than a
    # freshly initialised adapter.
    lr_multipliers: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        for module, mult in self.lr_multipliers.items():
            if mult <= 0:
                raise ValueError(
                    f"optimizer.lr_multipliers[{module!r}] must be positive (got {mult})"
                )
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not (0 <= self.betas[0] < 1 and 0 <= self.betas[1] < 1):
            raise ValueError("betas must be in [0, 1)")
        if not (0 < self.muon_momentum < 1):
            raise ValueError("muon_momentum must be in (0, 1)")
        if self.muon_ns_steps <= 0:
            raise ValueError("muon_ns_steps must be positive")
        if self.schedule_free_warmup_steps < 0:
            raise ValueError("schedule_free_warmup_steps must be non-negative")
