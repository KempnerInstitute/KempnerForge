"""Distributed checkpointing for KempnerForge.

Public API:
  - CheckpointManager: Save/load/cleanup distributed checkpoints
  - AsyncCheckpointer: Non-blocking checkpoint saves
  - build_train_state / restore_train_state: State assembly
  - load_train_state_extras: Read-only accessor for checkpoint ``extra`` metadata
"""

from kempnerforge.checkpoint.async_save import AsyncCheckpointer
from kempnerforge.checkpoint.manager import CheckpointManager, load_train_state_extras
from kempnerforge.checkpoint.state import (
    build_train_state,
    get_rng_state,
    restore_train_state,
    set_rng_state,
)

__all__ = [
    "AsyncCheckpointer",
    "CheckpointManager",
    "build_train_state",
    "get_rng_state",
    "load_train_state_extras",
    "restore_train_state",
    "set_rng_state",
]
