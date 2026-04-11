"""Async checkpointing for non-blocking saves.

Uses ``dcp.async_save()`` to snapshot state to CPU and write to disk
in the background, returning control to the training loop immediately.

Modes:
  - disabled: Synchronous save (simple, for debugging).
  - async: Standard async via dcp.async_save().
  - async_with_pinned_mem: Async with pinned memory staging for faster GPU→CPU.
"""

from __future__ import annotations

import logging
from typing import Any

import torch.distributed.checkpoint as dcp

from kempnerforge.config.schema import AsyncCheckpointMode

logger = logging.getLogger(__name__)


class AsyncCheckpointer:
    """Non-blocking checkpoint saver.

    Wraps ``dcp.async_save()`` and manages the background save future.
    Each new save waits for the previous async save to complete first.

    Args:
        mode: Checkpoint mode (disabled/async/async_with_pinned_mem).
    """

    def __init__(self, mode: AsyncCheckpointMode = AsyncCheckpointMode.disabled) -> None:
        self.mode = mode
        self._pending_future: Any = None

    def save(self, state_dict: dict, checkpoint_id: str, process_group=None) -> None:
        """Save distributed state, potentially asynchronously.

        Args:
            state_dict: DCP-compatible state dict (model + optimizer).
            checkpoint_id: Checkpoint directory path.
            process_group: Process group for DCP. Required for PP where each
                stage has a different state dict — pass a group scoped to ranks
                within the same PP stage. None uses the default global group.
        """
        # Wait for any pending async save to complete first
        self.wait()

        if self.mode == AsyncCheckpointMode.disabled:
            dcp.save(state_dict, checkpoint_id=checkpoint_id, process_group=process_group)
            logger.info(f"Sync checkpoint saved: {checkpoint_id}")

        elif self.mode in (AsyncCheckpointMode.async_, AsyncCheckpointMode.async_pinned):
            self._pending_future = dcp.async_save(
                state_dict,
                checkpoint_id=checkpoint_id,
                process_group=process_group,
            )
            logger.info(f"Async checkpoint started: {checkpoint_id}")

    def wait(self) -> None:
        """Block until any pending async save completes."""
        if self._pending_future is not None:
            self._pending_future.result()
            self._pending_future = None
            logger.info("Async checkpoint completed")

    @property
    def is_pending(self) -> bool:
        """Check if an async save is still in progress."""
        return self._pending_future is not None
