"""Optimizer configuration."""

from __future__ import annotations

from dataclasses import dataclass

WEIGHT_DECAY_SCOPES = ("exclude_norms_and_biases", "all")
MUON_ADJUST_LR_MODES = ("original", "match_rms_adamw", "spectral_unclamped")
MUON_ORTHOGONALIZE_MODES = ("local", "exact")


@dataclass
class OptimizerConfig:
    """Optimizer settings."""

    name: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.1
    weight_decay_scope: str = "exclude_norms_and_biases"
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
