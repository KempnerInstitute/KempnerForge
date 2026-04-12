"""Metrics configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MetricsConfig:
    """Logging and metrics settings."""

    log_interval: int = 10  # Log every N steps
    enable_wandb: bool = False
    enable_tensorboard: bool = False
    wandb_project: str = "kempnerforge"
    wandb_run_name: str | None = None  # None -> auto-generated
    tensorboard_dir: str = "tb_logs"

    def __post_init__(self) -> None:
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")
