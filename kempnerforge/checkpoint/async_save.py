"""Async checkpointing for non-blocking saves.

Uses ``dcp.async_save()`` to snapshot state to CPU and write to disk
in the background, returning control to the training loop once the snapshot is taken.

Modes:
  - disabled: Synchronous save (the default; simple, for debugging).
  - async: Standard async via dcp.async_save(); the snapshot goes to pageable CPU memory.
  - async_with_pinned_mem: async, with the snapshot copied into pinned CPU memory by a
    ``DefaultStager`` that serves every save on torch 2.13 and later (torch 2.12 gets a
    new one per save). The first save pins a copy of the rank's state; later saves refill
    the copy of every tensor whose storage is unchanged (parameters and optimizer state
    updated in place), so they stall less. The stager keeps a copy while its source
    storage lives, until ``close()`` or until the checkpointer is collected; a save still
    being written keeps its snapshot until the write ends. Without a CUDA device the mode
    saves as async.
"""

from __future__ import annotations

import logging
import weakref
from functools import partial
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.planner import SavePlanner
from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions

from kempnerforge.config.schema import AsyncCheckpointMode

logger = logging.getLogger(__name__)


def _reuses_stager() -> bool:
    """Whether this torch's DefaultStager can serve more than one save (2.13 and later)."""
    return torch.__version__ >= "2.13"


class AsyncCheckpointer:
    """Non-blocking checkpoint saver.

    Wraps ``dcp.async_save()`` and manages the background save future.
    Each new save waits for the previous async save to complete first.
    Not thread-safe: call it from one thread, as the training loop does.

    Args:
        mode: Checkpoint mode (disabled/async/async_with_pinned_mem). The pinned mode
            stages through pinned memory only where a CUDA device is available, and
            saves as async elsewhere, with one warning.
    """

    def __init__(self, mode: AsyncCheckpointMode = AsyncCheckpointMode.disabled) -> None:
        self.mode = mode
        self._pending_future: Any = None
        self._warned_unpinned = False
        self._stager: DefaultStager | None = None
        self._release_stager: weakref.finalize | None = None

    def save(
        self,
        state_dict: dict,
        checkpoint_id: str,
        process_group=None,
        planner: SavePlanner | None = None,
    ) -> None:
        """Save distributed state, potentially asynchronously.

        Args:
            state_dict: DCP-compatible state dict (model + optimizer).
            checkpoint_id: Checkpoint directory path.
            process_group: Process group for DCP. Required for PP where each
                stage has a different state dict — pass a group scoped to ranks
                within the same PP stage. None uses the default global group.
            planner: DCP save planner. None calls DCP as before, with its default planner.
        """
        self.wait()
        save, async_save = dcp.save, dcp.async_save
        if planner is not None:
            save = partial(save, planner=planner)
            async_save = partial(async_save, planner=planner)

        if self.mode == AsyncCheckpointMode.disabled:
            save(state_dict, checkpoint_id=checkpoint_id, process_group=process_group)
            logger.info(f"Sync checkpoint saved: {checkpoint_id}")

        elif self.mode in (AsyncCheckpointMode.async_, AsyncCheckpointMode.async_pinned):
            if self.mode == AsyncCheckpointMode.async_pinned and self._can_pin():
                async_save = partial(async_save, async_stager=self._pinned_stager())
            self._pending_future = async_save(
                state_dict,
                checkpoint_id=checkpoint_id,
                process_group=process_group,
            )
            logger.info(f"Async checkpoint started: {checkpoint_id}")

    def _pinned_stager(self) -> DefaultStager:
        """The stager that copies each snapshot into pinned CPU memory.

        On torch 2.13 and later one stager serves every save, so its pinned copy is
        allocated once and refilled. Torch 2.12's stager drops that copy after each save
        and fails when used again, so there each save gets a new one.
        """
        reuse = _reuses_stager()
        if self._stager is None or not reuse:
            self._stager = DefaultStager(
                StagingOptions(
                    use_pinned_memory=True,
                    use_shared_memory=False,
                    use_async_staging=False,
                    use_non_blocking_copy=True,
                )
            )
            if reuse:
                self._release_stager = weakref.finalize(self, self._stager.close)
        return self._stager

    def _can_pin(self) -> bool:
        """Whether pinned staging can run here, warning once when it cannot.

        The stager pins through CUDA, so without a CUDA device the pinned mode saves
        as async instead of failing.
        """
        if torch.cuda.is_available():
            return True
        if not self._warned_unpinned:
            logger.warning(
                "async_with_pinned_mem needs a CUDA device to pin memory; saving as async"
            )
            self._warned_unpinned = True
        return False

    def close(self) -> None:
        """Wait for any pending save, then release the pinned copies the stager holds.

        The stager is released and the pending save dropped even when that save failed,
        whose error still propagates. A later save in the pinned mode builds a new stager.
        """
        try:
            self.wait()
        finally:
            self._pending_future = None
            if self._release_stager is not None:
                self._release_stager()
                self._release_stager = None
            elif self._stager is not None:
                self._stager.close()
            self._stager = None

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
