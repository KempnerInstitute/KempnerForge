"""LR scheduler configuration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SchedulerType(StrEnum):
    cosine = "cosine"
    linear = "linear"
    wsd = "wsd"
    constant = "constant"
    rex = "rex"
    none = "none"


@dataclass
class SchedulerConfig:
    """Learning rate schedule settings."""

    name: SchedulerType = SchedulerType.cosine
    warmup_steps: int = 2000
    warmup_start_factor: float = 0.0
    decay_steps: int | None = None
    min_lr_ratio: float = 0.1
    stable_steps: int | None = None
    wsd_decay_type: str = "cosine"
    rex_alpha: float = 1.0

    def __post_init__(self) -> None:
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if not (0 <= self.warmup_start_factor <= 1):
            raise ValueError("warmup_start_factor must be in [0, 1]")
        if not (0 <= self.min_lr_ratio <= 1):
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if self.wsd_decay_type not in ("cosine", "linear", "sqrt"):
            raise ValueError(
                f"wsd_decay_type must be 'cosine', 'linear', or 'sqrt', got '{self.wsd_decay_type}'"
            )
        if self.rex_alpha <= 0:
            raise ValueError("rex_alpha must be positive")
