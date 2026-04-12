"""Optimizer configuration."""

from __future__ import annotations

from dataclasses import dataclass


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

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not (0 <= self.betas[0] < 1 and 0 <= self.betas[1] < 1):
            raise ValueError("betas must be in [0, 1)")
