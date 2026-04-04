"""Distributed sampler with deterministic shuffling and skip-ahead.

Correctly partitions data across data-parallel ranks with:
  - Epoch-based re-shuffling with deterministic seeds
  - Skip-ahead for exact resumption after checkpoint load
  - Handling of uneven dataset sizes (drop last partial batch)
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler


class DistributedSampler(Sampler[int]):
    """Deterministic distributed sampler with skip-ahead support.

    Partitions dataset indices across data-parallel ranks. Each rank
    sees a unique, non-overlapping subset of the data.

    Args:
        dataset: Dataset to sample from.
        num_replicas: Number of data-parallel ranks (default: world_size).
        rank: Current rank (default: from dist).
        shuffle: Whether to shuffle indices.
        seed: Base random seed for deterministic shuffling.
        drop_last: Drop samples that don't divide evenly across ranks.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

        # Compute per-rank sample count
        total = len(dataset)
        if drop_last:
            # Drop remainder so all ranks get the same count
            self.num_samples = total // num_replicas
            self.total_size = self.num_samples * num_replicas
        else:
            # Pad to make evenly divisible
            self.num_samples = math.ceil(total / num_replicas)
            self.total_size = self.num_samples * num_replicas

        # State for resumption
        self._epoch = 0
        self._skip = 0  # Number of samples to skip (for resumption)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic re-shuffling."""
        self._epoch = epoch

    def set_skip(self, skip: int) -> None:
        """Set number of samples to skip (for resumption after checkpoint)."""
        self._skip = skip

    def __iter__(self):
        # Generate deterministic permutation
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self._epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Handle uneven sizes
        if self.drop_last:
            indices = indices[: self.total_size]
        else:
            # Pad by wrapping around
            padding = self.total_size - len(indices)
            indices += indices[:padding]

        # Partition: each rank gets every num_replicas-th element
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        # Skip-ahead for resumption
        if self._skip > 0:
            indices = indices[self._skip :]
            self._skip = 0  # Reset after applying

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def state_dict(self) -> dict:
        return {
            "epoch": self._epoch,
            "seed": self.seed,
            "num_replicas": self.num_replicas,
            "rank": self.rank,
        }

    def load_state_dict(self, state: dict) -> None:
        self._epoch = state.get("epoch", 0)
