"""Optimizer configuration."""

from __future__ import annotations

import fnmatch
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

WEIGHT_DECAY_SCOPES = ("exclude_norms_and_biases", "all")
MUON_ADJUST_LR_MODES = ("original", "match_rms_adamw", "spectral_unclamped")
MUON_ORTHOGONALIZE_MODES = ("local", "exact")


def checked_lr_multipliers(multipliers: object) -> dict[str, float]:
    """Return ``multipliers`` as a ``{module: multiplier}`` dict of floats.

    Raises:
        ValueError: ``multipliers`` is not a dict, a key is not a non-empty string, or a
            value is not a positive finite number.
    """
    if not isinstance(multipliers, dict):
        raise ValueError(f"lr_multipliers must map module names to numbers, got {multipliers!r}")
    checked: dict[str, float] = {}
    for module, multiplier in multipliers.items():
        if not isinstance(module, str) or not module:
            raise ValueError(f"lr_multipliers keys must be module names, got {module!r}")
        if (
            isinstance(multiplier, bool)
            or not isinstance(multiplier, int | float)
            or not 0 < multiplier < math.inf
        ):
            raise ValueError(
                f"lr_multipliers[{module!r}] must be a positive finite number, got {multiplier!r}"
            )
        checked[module] = float(multiplier)
    return checked


def overlapping_modules(
    modules: Iterable[str], module_patterns: Mapping[str, Sequence[str]]
) -> tuple[str, str, str, str] | None:
    """Return two of ``modules`` whose patterns overlap, and a pattern of each, or None.

    Two patterns overlap when one, read as a parameter name, matches the other: that is how
    a module inside another is written (``transformer.layers.*`` in ``transformer.*``). Every
    module must be a key of ``module_patterns``.
    """
    listed = sorted(modules)
    for i, first in enumerate(listed):
        for second in listed[i + 1 :]:
            for a in module_patterns[first]:
                for b in module_patterns[second]:
                    if fnmatch.fnmatch(a, b) or fnmatch.fnmatch(b, a):
                        return first, second, a, b
    return None


@dataclass
class OptimizerConfig:
    """Optimizer settings."""

    name: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    fused: bool = True
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    muon_adam_lr: float | None = None
    muon_nesterov: bool = False
    muon_adjust_lr: str = "original"
    muon_orthogonalize: str = "local"
    schedule_free_warmup_steps: int = 0
    weight_decay_scope: str = "exclude_norms_and_biases"
    lr_multipliers: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.weight_decay_scope not in WEIGHT_DECAY_SCOPES:
            raise ValueError(
                f"weight_decay_scope must be one of {WEIGHT_DECAY_SCOPES}, "
                f"got {self.weight_decay_scope!r}"
            )
        if not (0 <= self.betas[0] < 1 and 0 <= self.betas[1] < 1):
            raise ValueError("betas must be in [0, 1)")
        if not (0 < self.muon_momentum < 1):
            raise ValueError("muon_momentum must be in (0, 1)")
        if self.muon_ns_steps <= 0:
            raise ValueError("muon_ns_steps must be positive")
        if self.muon_adjust_lr not in MUON_ADJUST_LR_MODES:
            raise ValueError(
                f"muon_adjust_lr must be one of {MUON_ADJUST_LR_MODES}, got {self.muon_adjust_lr!r}"
            )
        if self.muon_orthogonalize not in MUON_ORTHOGONALIZE_MODES:
            raise ValueError(
                f"muon_orthogonalize must be one of {MUON_ORTHOGONALIZE_MODES}, "
                f"got {self.muon_orthogonalize!r}"
            )
        if self.schedule_free_warmup_steps < 0:
            raise ValueError("schedule_free_warmup_steps must be non-negative")
        self.lr_multipliers = checked_lr_multipliers(self.lr_multipliers)
