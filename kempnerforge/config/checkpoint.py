"""Checkpoint configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal


class AsyncCheckpointMode(StrEnum):
    disabled = "disabled"
    async_ = "async"
    async_pinned = "async_with_pinned_mem"


@dataclass
class CheckpointConfig:
    """Checkpointing settings."""

    dir: str = "checkpoints"
    interval: int = 1000  # Save every N steps
    async_mode: AsyncCheckpointMode = AsyncCheckpointMode.disabled
    keep_last_n: int = 3  # Number of checkpoints to retain
    load_path: str | None = None  # Path to load from (for resumption)
    export_dtype: Literal["float32", "bfloat16"] = "bfloat16"
    exclude_from_loading: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.interval <= 0:
            raise ValueError("interval must be positive")
        if self.keep_last_n < 1:
            raise ValueError("keep_last_n must be >= 1")
