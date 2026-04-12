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
        total = len(dataset)  # type: ignore[reportArgumentType]
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
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[reportArgumentType]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[reportArgumentType]

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
        """Return checkpoint state. Keys: ``epoch``, ``seed``, ``num_replicas``, ``rank``."""
        return {
            "epoch": self._epoch,
            "seed": self.seed,
            "num_replicas": self.num_replicas,
            "rank": self.rank,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore from checkpoint. Only ``epoch`` is restored; rank info is local."""
        self._epoch = state.get("epoch", 0)


class MixtureSampler(Sampler[int]):
    """Weighted sampler over a :class:`MixtureDataset`.

    Each sub-dataset's indices are partitioned across ranks (like
    ``DistributedSampler``). The ``weights`` control what fraction of
    the epoch is drawn from each dataset — datasets with higher weight
    are oversampled.

    Args:
        cumulative_sizes: ``[0, len(ds0), len(ds0)+len(ds1), ...]``
            from ``MixtureDataset.cumulative_sizes``.
        weights: Per-dataset sampling weights (normalized internally).
        num_replicas: Number of data-parallel ranks.
        rank: Current rank.
        shuffle: Whether to shuffle indices.
        seed: Base random seed.
        drop_last: Drop samples that don't divide evenly across ranks.
        temperature: Weight temperature (1.0 = as-is, >1 → more uniform).
    """

    def __init__(
        self,
        cumulative_sizes: list[int],
        weights: list[float],
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
        temperature: float = 1.0,
    ) -> None:
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0

        n = len(cumulative_sizes) - 1
        self._dataset_sizes = [cumulative_sizes[i + 1] - cumulative_sizes[i] for i in range(n)]
        self._offsets = list(cumulative_sizes[:n])

        # Apply temperature scaling and normalize
        if temperature != 1.0:
            import math as _math

            log_w = [_math.log(max(w, 1e-12)) / temperature for w in weights]
            max_lw = max(log_w)
            scaled = [_math.exp(lw - max_lw) for lw in log_w]
            total = sum(scaled)
            self._probs = [s / total for s in scaled]
        else:
            total_w = sum(weights)
            self._probs = [w / total_w for w in weights]

        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

        # Per-dataset per-rank available count
        per_rank_avail = []
        for size in self._dataset_sizes:
            if drop_last:
                per_rank_avail.append(size // num_replicas)
            else:
                per_rank_avail.append(math.ceil(size / num_replicas))

        # Weighted allocation: how many samples from each dataset per epoch
        total_per_rank = sum(per_rank_avail)
        self._target_counts = [round(p * total_per_rank) for p in self._probs]

        # Fix rounding to match total exactly
        diff = total_per_rank - sum(self._target_counts)
        sorted_idx = sorted(range(n), key=lambda i: -self._probs[i])
        for i in range(abs(diff)):
            idx = sorted_idx[i % n]
            self._target_counts[idx] += 1 if diff > 0 else -1

        self.num_samples = sum(self._target_counts)
        self._epoch = 0
        self._skip = 0

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic re-shuffling."""
        self._epoch = epoch

    def set_skip(self, skip: int) -> None:
        """Set number of samples to skip (for resumption after checkpoint)."""
        self._skip = skip

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self._epoch)

        result: list[int] = []
        for ds_i in range(len(self._dataset_sizes)):
            size = self._dataset_sizes[ds_i]
            offset = self._offsets[ds_i]
            target = self._target_counts[ds_i]

            if target <= 0 or size == 0:
                continue

            # Shuffled local indices for this dataset
            if self.shuffle:
                indices = torch.randperm(size, generator=g).tolist()
            else:
                indices = list(range(size))

            # Partition for this rank (stride-based, like DistributedSampler)
            if self.drop_last:
                usable = size - (size % self.num_replicas)
                indices = indices[:usable]
            else:
                padding = (self.num_replicas - len(indices) % self.num_replicas) % self.num_replicas
                if padding:
                    indices = indices + indices[:padding]

            rank_indices = indices[self.rank :: self.num_replicas]

            if not rank_indices:
                continue

            # Draw target samples (wrap around for oversampling)
            if target <= len(rank_indices):
                drawn = rank_indices[:target]
            else:
                reps = target // len(rank_indices) + 1
                drawn = (rank_indices * reps)[:target]

            # Convert to global MixtureDataset indices
            result.extend(idx + offset for idx in drawn)

        # Shuffle all indices together for random interleaving
        if self.shuffle:
            perm = torch.randperm(len(result), generator=g)
            result = [result[p] for p in perm.tolist()]

        # Skip for resumption
        if self._skip > 0:
            result = result[self._skip :]
            self._skip = 0

        return iter(result)

    def state_dict(self) -> dict:
        """Return checkpoint state."""
        return {
            "epoch": self._epoch,
            "seed": self.seed,
            "num_replicas": self.num_replicas,
            "rank": self.rank,
        }

    def update_weights(self, weights: list[float], temperature: float = 1.0) -> None:
        """Update sampling weights for phase transitions.

        Recomputes internal probabilities and per-dataset target counts.
        Takes effect on the next ``__iter__()`` call.
        """
        n = len(self._dataset_sizes)
        if len(weights) != n:
            raise ValueError(f"Expected {n} weights, got {len(weights)}")

        # Apply temperature scaling and normalize (same logic as __init__)
        if temperature != 1.0:
            import math as _math

            log_w = [_math.log(max(w, 1e-12)) / temperature for w in weights]
            max_lw = max(log_w)
            scaled = [_math.exp(lw - max_lw) for lw in log_w]
            total = sum(scaled)
            self._probs = [s / total for s in scaled]
        else:
            total_w = sum(weights)
            self._probs = [w / total_w for w in weights]

        # Recompute per-dataset per-rank available count
        per_rank_avail = []
        for size in self._dataset_sizes:
            if self.drop_last:
                per_rank_avail.append(size // self.num_replicas)
            else:
                per_rank_avail.append(math.ceil(size / self.num_replicas))

        total_per_rank = sum(per_rank_avail)
        self._target_counts = [round(p * total_per_rank) for p in self._probs]

        # Fix rounding to match total exactly
        diff = total_per_rank - sum(self._target_counts)
        sorted_idx = sorted(range(n), key=lambda i: -self._probs[i])
        for i in range(abs(diff)):
            idx = sorted_idx[i % n]
            self._target_counts[idx] += 1 if diff > 0 else -1

        self.num_samples = sum(self._target_counts)

    def load_state_dict(self, state: dict) -> None:
        """Restore from checkpoint."""
        self._epoch = state.get("epoch", 0)
