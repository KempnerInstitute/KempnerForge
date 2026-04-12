"""LR scheduler configuration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SchedulerType(StrEnum):
    cosine = "cosine"
    linear = "linear"
    wsd = "wsd"  # warmup-stable-decay
    constant = "constant"  # warmup then flat LR
    rex = "rex"  # polynomial decay: (1 - t/T)^alpha
    none = "none"  # constant LR (for schedule-free optimizers)


@dataclass
class SchedulerConfig:
    """Learning rate schedule settings."""

    name: SchedulerType = SchedulerType.cosine
    warmup_steps: int = 2000
    decay_steps: int | None = None  # None -> decay over remaining steps
    min_lr_ratio: float = 0.1  # min_lr = lr * min_lr_ratio
    # WSD-specific
    stable_steps: int | None = None  # For WSD: steps at constant LR
    wsd_decay_type: str = "cosine"  # WSD cooldown shape: "cosine", "linear", "sqrt"
    # REX-specific
    rex_alpha: float = 1.0  # Exponent for REX: (1 - t/T)^alpha

    def __post_init__(self) -> None:
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if not (0 <= self.min_lr_ratio <= 1):
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if self.wsd_decay_type not in ("cosine", "linear", "sqrt"):
            raise ValueError(
                f"wsd_decay_type must be 'cosine', 'linear', or 'sqrt', got '{self.wsd_decay_type}'"
            )
        if self.rex_alpha <= 0:
            raise ValueError("rex_alpha must be positive")
