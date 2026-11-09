"""Checkpoint configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from kempnerforge.config.registry import registry


class AsyncCheckpointMode(StrEnum):
    disabled = "disabled"
    async_ = "async"
    async_pinned = "async_with_pinned_mem"


@dataclass
class DynamicCheckpointWindow:
    """A bounded step range with a registered checkpoint strategy.

    Inside ``[start, stop]`` the strategy decides which steps to save, and every
    such step is exempt from ``CheckpointConfig.keep_last_n`` retention. Outside
    the window the regular ``CheckpointConfig.interval`` cadence applies.

    ``"power2"`` (default) saves at ``start`` and at every ``start + 2^k`` while
    ``<= stop`` -- tight at the start of the window, doubling thereafter. New
    strategies register via ``@registry.register_dyn_ckpt_strategy(name)`` and
    become selectable by setting ``strategy``.
    """

    start: int = 0
    stop: int = 512
    strategy: str = "power2"

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("dyn_ckpt_window.start must be >= 0")
        if self.stop < self.start:
            raise ValueError("dyn_ckpt_window.stop must be >= start")
        known = registry.list_dyn_ckpt_strategies()
        if self.strategy not in known:
            raise ValueError(
                f"unknown dyn_ckpt_window.strategy {self.strategy!r}; registered: {known}"
            )

    def is_milestone(self, step: int) -> bool:
        """True iff the configured strategy fires at ``step``."""
        return registry.get_dyn_ckpt_strategy(self.strategy)(self, step)


@registry.register_dyn_ckpt_strategy("power2")
def _power2_strategy(window: DynamicCheckpointWindow, step: int) -> bool:
    """Save at ``start`` and at every ``start + 2^k`` while ``<= stop``."""
    if step < window.start or step > window.stop:
        return False
    offset = step - window.start
    return offset == 0 or (offset & (offset - 1)) == 0


WEIGHTS_ONLY_EXCLUDE = frozenset({"optimizer", "train_state"})


def validate_exclude_keys(keys: list[str], name: str) -> None:
    """Check an exclude list: DCP keys only, and ``"train_state"`` only in the weights-only pair."""
    if any(k not in ("model", "optimizer", "train_state") for k in keys):
        raise ValueError(
            f"{name} accepts only 'model', 'optimizer' and/or 'train_state' (got {keys!r})"
        )
    if "train_state" in keys and set(keys) != WEIGHTS_ONLY_EXCLUDE:
        raise ValueError(
            f"{name} with 'train_state' must be ['optimizer', 'train_state']: a weights-only "
            f"warm start loads the model alone (got {keys!r})"
        )


@dataclass
class CheckpointConfig:
    """Checkpointing settings."""

    dir: str = "checkpoints"
    interval: int = 1000
    dyn_ckpt_window: DynamicCheckpointWindow | None = None
    async_mode: AsyncCheckpointMode = AsyncCheckpointMode.disabled
    keep_last_n: int = 3
    load_path: str | None = None
    export_dtype: Literal["float32", "bfloat16"] = "bfloat16"
    exclude_from_loading: list[str] = field(default_factory=list)
    ignore_freeze_mismatch: bool = False

    def __post_init__(self) -> None:
        if self.interval <= 0:
            raise ValueError("interval must be positive")
        if not isinstance(self.exclude_from_loading, list):
            raise ValueError(
                f"exclude_from_loading must be a list (got {self.exclude_from_loading!r})"
            )
        validate_exclude_keys(self.exclude_from_loading, "exclude_from_loading")

    def should_save(self, step: int) -> bool:
        """Whether to write a checkpoint at ``step``.

        Inside ``dyn_ckpt_window``: the registered strategy decides (default
        ``"power2"`` saves at ``start`` and each ``start + 2^k`` while
        ``<= stop``). Outside the window: every ``interval`` steps. Dynamic
        milestones are exempt from ``keep_last_n`` (see
        ``CheckpointManager._cleanup``).
        """
        w = self.dyn_ckpt_window
        if w is not None and w.start <= step <= w.stop:
            return w.is_milestone(step)
        return step % self.interval == 0

    def is_dynamic_milestone(self, step: int) -> bool:
        """True if ``step`` is a milestone of the configured ``dyn_ckpt_window``.

        ``CheckpointManager._cleanup`` excludes these from ``keep_last_n`` so
        the dense early-window checkpoints survive a finite retention.
        """
        return self.dyn_ckpt_window is not None and self.dyn_ckpt_window.is_milestone(step)
