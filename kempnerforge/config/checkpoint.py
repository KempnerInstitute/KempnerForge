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
    interval: int = 1000  # Save every N steps (after log_until in "log" mode)
    schedule: Literal["interval", "log"] = "interval"  # "log": dense early saves for dynamics
    log_until: int = 512  # In "log" mode: save at step 0 and powers of two up to this step
    async_mode: AsyncCheckpointMode = AsyncCheckpointMode.disabled
    keep_last_n: int = 3  # Checkpoints to retain; <= 0 keeps all (e.g. for dynamics studies)
    load_path: str | None = None  # Path to load from (for resumption)
    export_dtype: Literal["float32", "bfloat16"] = "bfloat16"
    exclude_from_loading: list[str] = field(default_factory=list)
    # If the saved checkpoint's VLM freeze metadata differs from the current
    # config's freeze specs, the load path raises by default. Setting this
    # to True downgrades the mismatch to a warning. Useful when intentionally
    # switching from frozen to trainable mid-training.
    ignore_freeze_mismatch: bool = False

    def __post_init__(self) -> None:
        if self.interval <= 0:
            raise ValueError("interval must be positive")
        if self.schedule == "log" and self.log_until < 1:
            raise ValueError("log_until must be positive when schedule='log'")

    def should_save(self, step: int) -> bool:
        """Whether to write a checkpoint at ``step``.

        ``interval`` mode saves every ``interval`` steps. ``log`` mode saves at
        step 0 and each power of two up to ``log_until`` -- dense coverage of the
        early-training dynamics -- then falls back to every ``interval`` steps.
        Pair ``log`` mode with ``keep_last_n <= 0`` so the early checkpoints
        survive retention.
        """
        if self.schedule == "log" and step <= self.log_until:
            return step == 0 or (step & (step - 1)) == 0
        return step % self.interval == 0
