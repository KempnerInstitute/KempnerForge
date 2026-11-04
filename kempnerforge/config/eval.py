"""Evaluation configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalConfig:
    """Evaluation pipeline settings (disabled by default)."""

    enabled: bool = False
    interval: int = 1000
    steps: int = 50
    dataset_path: str = ""
    file_pattern: str = "*.npy"
    hf_dataset_name: str | None = None
    hf_dataset_config: str | None = None
    hf_dataset_split: str = "validation"

    def __post_init__(self) -> None:
        if self.interval <= 0:
            raise ValueError("eval interval must be positive")
        if self.steps <= 0:
            raise ValueError("eval steps must be positive")
