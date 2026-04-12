"""Distributed, stateful DataLoader for KempnerForge.

Wraps PyTorch DataLoader with:
  - Distributed-aware setup (correct worker count, pinned memory)
  - Stateful iteration tracking for checkpoint/resume
  - Integration with DistributedSampler for rank-partitioned data
"""

from __future__ import annotations

import logging

import torch
from torch.utils.data import DataLoader, Dataset

from kempnerforge.config.schema import DataConfig
from kempnerforge.data.sampler import DistributedSampler, MixtureSampler

logger = logging.getLogger(__name__)


class StatefulDataLoader:
    """Stateful wrapper around PyTorch DataLoader.

    Tracks iteration progress so training can resume from the exact
    position after a checkpoint load.

    Args:
        dataset: Dataset to load from.
        batch_size: Per-device micro-batch size.
        sampler: Distributed sampler (created automatically if None).
        config: Data pipeline configuration.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        sampler: DistributedSampler | MixtureSampler | None = None,
        config: DataConfig | None = None,
    ) -> None:
        config = config or DataConfig()
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampler = sampler or DistributedSampler(dataset)

        self._dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=self.sampler,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
            persistent_workers=config.num_workers > 0,
            drop_last=True,
        )

        # State tracking
        self._epoch = 0
        self._batches_yielded = 0
        self._iterator = None

        logger.info(
            f"StatefulDataLoader: batch_size={batch_size}, "
            f"workers={config.num_workers}, pin_memory={config.pin_memory}"
        )

    def __iter__(self):
        self.sampler.set_epoch(self._epoch)
        self._iterator = iter(self._dataloader)
        self._batches_yielded = 0
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        if self._iterator is None:
            raise StopIteration

        try:
            batch = next(self._iterator)
            self._batches_yielded += 1
            return batch
        except StopIteration:
            self._epoch += 1
            self._batches_yielded = 0
            self._iterator = None
            raise

    def __len__(self) -> int:
        return len(self._dataloader)

    def state_dict(self) -> dict:
        """Return checkpoint state. Keys: ``epoch``, ``batches_yielded``, ``sampler``."""
        return {
            "epoch": self._epoch,
            "batches_yielded": self._batches_yielded,
            "sampler": self.sampler.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore from checkpoint. Restores sampler state and skips to saved batch position."""
        self._epoch = state.get("epoch", 0)
        batches_yielded = state.get("batches_yielded", 0)

        # Set sampler state for resumption
        if "sampler" in state:
            self.sampler.load_state_dict(state["sampler"])

        # Skip ahead to the correct position in the current epoch
        if batches_yielded > 0:
            self.sampler.set_skip(batches_yielded * self.batch_size)

        logger.info(f"Resumed DataLoader: epoch={self._epoch}, skip_batches={batches_yielded}")
