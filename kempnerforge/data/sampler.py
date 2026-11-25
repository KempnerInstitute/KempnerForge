"""Distributed sampler with deterministic shuffling and skip-ahead.

Correctly partitions data across data-parallel ranks with:
  - Epoch-based re-shuffling with deterministic seeds
  - Skip-ahead for exact resumption after checkpoint load
  - Handling of uneven dataset sizes (drop last partial batch)
"""

from __future__ import annotations

import logging
import math

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)

_REDRAW_SEED_OFFSET = 1 << 32
_SEED_RANGE = 1 << 64


def _validate_weights(weights: list[float], context: str) -> None:
    """Fail fast on empty, negative, or all-zero weight lists.

    The two normalization branches in ``MixtureSampler`` disagree on all-zero
    input: the ``temperature == 1.0`` branch divides by ``sum(weights)`` and
    raises ``ZeroDivisionError``; the ``temperature != 1.0`` branch clamps via
    ``max(w, 1e-12)`` and silently produces uniform sampling. Reject both
    cases up-front with a clear error so misconfigured phase transitions
    surface immediately instead of crashing mid-run or drifting silently.
    """
    if not weights:
        raise ValueError(f"{context}: weights list is empty")
    if any(w < 0 for w in weights):
        raise ValueError(f"{context}: weights must be non-negative, got {weights}")
    if sum(weights) <= 0:
        raise ValueError(
            f"{context}: weights must sum to > 0 (at least one dataset must have "
            f"weight > 0), got {weights}"
        )


def _mixture_probs(weights: list[float], temperature: float) -> list[float]:
    """Normalize ``weights`` into sampling probabilities, flattened by ``temperature``."""
    if temperature != 1.0:
        log_w = [math.log(max(w, 1e-12)) / temperature for w in weights]
        max_lw = max(log_w)
        scaled = [math.exp(lw - max_lw) for lw in log_w]
        total = sum(scaled)
        return [s / total for s in scaled]
    total_w = sum(weights)
    return [w / total_w for w in weights]


def _apportion(total: int, probs: list[float]) -> list[int]:
    """Split ``total`` draws across datasets in proportion to ``probs``.

    Each share is rounded, then the rounding difference moves one draw at a
    time across the datasets in order of decreasing probability.
    """
    counts = [round(p * total) for p in probs]
    diff = total - sum(counts)
    order = sorted(range(len(probs)), key=lambda i: -probs[i])
    for i in range(abs(diff)):
        counts[order[i % len(probs)]] += 1 if diff > 0 else -1
    return counts


def _capped_shares(total: int, probs: list[float], caps: list[int]) -> list[int]:
    """Split ``total`` draws in proportion to ``probs`` with no dataset above its cap.

    The draws a cap turns away go to the datasets with room left, in order of
    decreasing probability. The caps must add up to at least ``total``.
    """
    counts = [min(share, cap) for share, cap in zip(_apportion(total, probs), caps, strict=True)]
    left = total - sum(counts)
    for i in sorted(range(len(probs)), key=lambda i: -probs[i]):
        more = min(left, caps[i] - counts[i])
        counts[i] += more
        left -= more
    return counts


def _pick(count: int, unseen: list[int], pool: list[int]) -> list[int]:
    """``count`` samples: ``unseen`` in order, then ``pool`` from its start, repeated as needed."""
    if count <= len(unseen):
        return unseen[:count]
    extra = count - len(unseen)
    return unseen + (pool * (extra // len(pool) + 1))[:extra]


class DistributedSampler(Sampler[int]):
    """Deterministic distributed sampler with skip-ahead support.

    Partitions dataset indices across data-parallel ranks. Each rank
    sees a unique, non-overlapping subset of the data. Every epoch has
    one global order, the same at any ``num_replicas``, and rank ``r``
    reads positions ``r``, ``r + num_replicas``, ... of it, so a resume
    position counted in that order (``set_position``) carries over to
    another number of ranks.

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

        total = len(dataset)  # type: ignore[reportArgumentType]
        if drop_last:
            self.num_samples = total // num_replicas
            self.total_size = self.num_samples * num_replicas
        else:
            self.num_samples = math.ceil(total / num_replicas)
            self.total_size = self.num_samples * num_replicas

        self._epoch = 0
        self._skip = 0
        self._position = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic re-shuffling."""
        self._epoch = epoch

    def set_skip(self, skip: int) -> None:
        """Set number of samples to skip (for resumption after checkpoint)."""
        self._skip = skip
        self._position = 0

    def set_position(self, position: int) -> None:
        """Start the next pass at sample ``position`` of the epoch's global order.

        ``position`` counts the samples all ranks have read together, so it
        means the same at any ``num_replicas``. Rank ``r`` starts at its first
        sample at or after ``position``, and every rank yields
        ``samples_after(position)`` samples, so the ranks stay in step and none
        reads a sample before ``position`` again. Applies to the next pass only.
        """
        self._position = position
        self._skip = 0

    def samples_after(self, position: int) -> int:
        """How many samples each rank yields in a pass that starts at ``position``."""
        q, m = divmod(position, self.num_replicas)
        return max(0, self.num_samples - q - (1 if m else 0))

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self._epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[reportArgumentType]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[reportArgumentType]

        if self.drop_last:
            indices = indices[: self.total_size]
        else:
            padding = self.total_size - len(indices)
            indices += indices[:padding]

        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        if self._position > 0:
            q, m = divmod(self._position, self.num_replicas)
            start = q + (1 if self.rank < m else 0)
            indices = indices[start : start + self.samples_after(self._position)]
            self._position = 0
        elif self._skip > 0:
            indices = indices[self._skip :]
            self._skip = 0

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
    are oversampled. Each rank draws and shuffles its own list, so a
    resume position is a count of this rank's samples and carries over
    only to the same ``num_replicas``.

    Weights set by ``update_weights`` apply from the next pass. A pass that
    starts inside the epoch, at a skip set after the change, draws the rest of
    the epoch with the new weights, from the samples this rank has not drawn
    in the epoch first, so a dataset that no weighting of the epoch
    oversamples draws no sample twice in the epoch. The state dict records
    the weights the epoch drew with from each such position, so a resume
    draws what the uninterrupted run draws.

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

        _validate_weights(weights, "MixtureSampler(weights=...)")

        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

        self._total_per_rank = sum(
            size // num_replicas if drop_last else math.ceil(size / num_replicas)
            for size in self._dataset_sizes
        )
        self._set_probs(weights, temperature)
        self._epoch = 0
        self._skip = 0
        self._epoch_weights: list[tuple[int, list[float]]] = []
        self._weights_epoch: int | None = None

    def _set_probs(self, weights: list[float], temperature: float) -> None:
        self._probs = _mixture_probs(weights, temperature)
        self._target_counts = _apportion(self._total_per_rank, self._probs)
        self.num_samples = sum(self._target_counts)

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic re-shuffling."""
        self._epoch = epoch

    def set_skip(self, skip: int) -> None:
        """Set number of samples to skip (for resumption after checkpoint)."""
        self._skip = skip

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        skip, self._skip = self._skip, 0
        self._note_weights(skip)
        return iter(self._epoch_order()[skip:])

    def _note_weights(self, position: int) -> None:
        """Record that the epoch draws with the current weights from ``position`` on."""
        if position == 0 or self._weights_epoch != self._epoch:
            self._epoch_weights = [(0, list(self._probs))]
            self._weights_epoch = self._epoch
            return
        kept = [entry for entry in self._epoch_weights if entry[0] < position]
        if kept[-1][1] != self._probs:
            kept.append((position, list(self._probs)))
        self._epoch_weights = kept

    def _partition(self, size: int, g: torch.Generator) -> list[int]:
        """This rank's share of a sub-dataset of ``size`` samples, in the epoch's order."""
        indices = torch.randperm(size, generator=g).tolist() if self.shuffle else list(range(size))
        if self.drop_last:
            indices = indices[: size - size % self.num_replicas]
        else:
            padding = (self.num_replicas - len(indices) % self.num_replicas) % self.num_replicas
            indices = indices + indices[:padding]
        return indices[self.rank :: self.num_replicas]

    def _first_order(
        self, targets: list[int], g: torch.Generator
    ) -> tuple[list[int], list[list[int] | None]]:
        """The epoch's list under ``targets``, and the rank share of each dataset it drew from."""
        order: list[int] = []
        pools: list[list[int] | None] = [None] * len(targets)
        for ds_i, (size, offset, target) in enumerate(
            zip(self._dataset_sizes, self._offsets, targets, strict=True)
        ):
            if target <= 0 or size == 0:
                continue
            pool = self._partition(size, g)
            pools[ds_i] = pool
            if not pool:
                continue
            order.extend(idx + offset for idx in _pick(target, pool, pool))
        if self.shuffle:
            perm = torch.randperm(len(order), generator=g)
            order = [order[p] for p in perm.tolist()]
        return order, pools

    def _rest(
        self,
        drawn: list[int],
        length: int,
        probs: list[float],
        pools: list[list[int]],
        oversampled: list[bool],
        g: torch.Generator,
    ) -> list[int]:
        """``length`` more samples under ``probs``, those not in ``drawn`` first."""
        seen = set(drawn)
        unseen = [
            [idx for idx in pool if idx + offset not in seen]
            for pool, offset in zip(pools, self._offsets, strict=True)
        ]
        caps = [
            length if over and pool else len(left)
            for over, pool, left in zip(oversampled, pools, unseen, strict=True)
        ]
        rest: list[int] = []
        for pool, left, offset, count in zip(
            pools, unseen, self._offsets, _capped_shares(length, probs, caps), strict=True
        ):
            rest.extend(idx + offset for idx in _pick(count, left, pool))
        if self.shuffle:
            perm = torch.randperm(len(rest), generator=g)
            rest = [rest[p] for p in perm.tolist()]
        return rest

    def _epoch_order(self) -> list[int]:
        """This rank's samples for the epoch, in order, under the weights it recorded.

        The draws after a change of weights use a generator of their own, seeded
        by the seed and epoch alone, so the rank shares of the datasets the
        epoch's start did not draw from are the same split on every rank, even
        when the ranks' lists differ in length.
        """
        g = torch.Generator().manual_seed(self.seed + self._epoch)
        targets = _apportion(self._total_per_rank, self._epoch_weights[0][1])
        order, first_pools = self._first_order(targets, g)
        if len(self._epoch_weights) == 1:
            return order
        g = torch.Generator().manual_seed(
            (self.seed + self._epoch + _REDRAW_SEED_OFFSET) % _SEED_RANGE
        )
        pools = [
            pool if pool is not None else self._partition(size, g)
            for pool, size in zip(first_pools, self._dataset_sizes, strict=True)
        ]
        oversampled = [target > len(pool) for target, pool in zip(targets, pools, strict=True)]
        for position, probs in self._epoch_weights[1:]:
            targets = _apportion(self._total_per_rank, probs)
            oversampled = [
                over or target > len(pool)
                for over, target, pool in zip(oversampled, targets, pools, strict=True)
            ]
            drawn = order[:position]
            length = max(0, len(order) - position)
            order = drawn + self._rest(drawn, length, probs, pools, oversampled, g)
        return order

    def state_dict(self) -> dict:
        """Return checkpoint state.

        Keys: ``epoch``, ``seed``, ``num_replicas``, ``rank``, and, once the
        weights changed inside the epoch, ``epoch_weights``: the sampling
        probabilities the epoch drew with, as ``[position, probabilities]``
        pairs from position 0.
        """
        state: dict = {
            "epoch": self._epoch,
            "seed": self.seed,
            "num_replicas": self.num_replicas,
            "rank": self.rank,
        }
        if self._weights_epoch == self._epoch and self._epoch_weights != [(0, self._probs)]:
            state["epoch_weights"] = [[pos, list(probs)] for pos, probs in self._epoch_weights]
        return state

    def update_weights(self, weights: list[float], temperature: float = 1.0) -> None:
        """Update sampling weights for phase transitions.

        Recomputes internal probabilities and per-dataset target counts.
        Takes effect on the next ``__iter__()`` call: a pass at the start of an
        epoch draws the epoch with the new weights, and a pass that starts
        inside the epoch draws the rest of it with them (see the class).
        """
        n = len(self._dataset_sizes)
        if len(weights) != n:
            raise ValueError(f"Expected {n} weights, got {len(weights)}")

        _validate_weights(weights, "MixtureSampler.update_weights")
        self._set_probs(weights, temperature)

    def load_state_dict(self, state: dict) -> None:
        """Restore from checkpoint: the epoch and the weights it recorded.

        A record of ``epoch_weights`` for another number of datasets is
        dropped with a warning; the epoch then continues as one drawn with the
        current weights from its start, as a state without the record does.
        """
        self._epoch = state.get("epoch", 0)
        saved = state.get("epoch_weights")
        if saved and any(len(probs) != len(self._dataset_sizes) for _, probs in saved):
            logger.warning(
                f"The checkpoint records the mixture weights of epoch {self._epoch} for "
                f"{len(saved[0][1])} datasets but this mixture has {len(self._dataset_sizes)}, "
                f"so the rest of the epoch is drawn as if the current weights had held from "
                f"its start, which can skip or repeat samples"
            )
            saved = None
        self._epoch_weights = [(position, list(probs)) for position, probs in saved or []]
        self._weights_epoch = self._epoch if saved else None
