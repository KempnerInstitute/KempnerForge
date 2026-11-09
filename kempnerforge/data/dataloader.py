"""Distributed, stateful DataLoader for KempnerForge.

Wraps PyTorch DataLoader with:
  - Distributed-aware setup (correct worker count, pinned memory)
  - Stateful iteration tracking for checkpoint/resume
  - Integration with DistributedSampler for rank-partitioned data
  - A resume position that survives a change of data-parallel size or batch size
  - A streaming loader that resumes by replaying its stream to the saved position
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch
from torch.utils.data import DataLoader, Dataset

from kempnerforge.config.schema import DataConfig
from kempnerforge.data.dataset import StreamingHuggingFaceDataset
from kempnerforge.data.sampler import DistributedSampler, MixtureSampler
from kempnerforge.resilience.signal_handler import ignore_shutdown_signals_in_worker

logger = logging.getLogger(__name__)


class DataPositionError(ValueError):
    """A saved data position that this loader cannot resume from.

    ``load_state_dict`` raises it when the saved position has no exact
    counterpart for this loader, such as a mixture position saved on another
    data-parallel size. The saved state and the loader settings are the same on
    every rank, so every rank raises it together.
    """


class StatefulDataLoader:
    """Stateful wrapper around PyTorch DataLoader.

    Tracks iteration progress so training can resume from the exact
    position after a checkpoint load. With a ``DistributedSampler`` the
    position is the number of samples of the epoch's global order that all
    data-parallel ranks consumed together, so it resumes exactly on another
    data-parallel size or batch size. With any other sampler it is this
    rank's sample count, which resumes on the data-parallel size that saved it.

    Args:
        dataset: Dataset to load from.
        batch_size: Per-device micro-batch size.
        sampler: Distributed sampler (created automatically if None).
        config: Data pipeline configuration.
        collate_fn: Optional custom batch collator. When None, uses
            PyTorch's default collation. VLM training passes
            ``VLMCollator`` so that the fixed-length padding and
            ``image_positions`` slot reach the batch.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        sampler: DistributedSampler | MixtureSampler | None = None,
        config: DataConfig | None = None,
        collate_fn: Callable | None = None,
    ) -> None:
        config = config or DataConfig()
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampler = sampler or DistributedSampler(dataset)

        loader_kwargs: dict = {
            "batch_size": batch_size,
            "sampler": self.sampler,
            "num_workers": config.num_workers,
            "pin_memory": config.pin_memory,
            "prefetch_factor": config.prefetch_factor if config.num_workers > 0 else None,
            "persistent_workers": config.num_workers > 0,
            "drop_last": True,
            "worker_init_fn": ignore_shutdown_signals_in_worker,
        }
        if collate_fn is not None:
            loader_kwargs["collate_fn"] = collate_fn
        self._dataloader = DataLoader(dataset, **loader_kwargs)

        self._epoch = 0
        self._batches_yielded = 0
        self._start = 0
        self._iterator = None

        logger.info(
            f"StatefulDataLoader: batch_size={batch_size}, "
            f"workers={config.num_workers}, pin_memory={config.pin_memory}"
        )

    def _consumed(self) -> int:
        """Samples consumed this epoch.

        All ranks' samples for a ``DistributedSampler``, else this rank's.
        """
        if isinstance(self.sampler, DistributedSampler):
            per_batch = self.batch_size * self.sampler.num_replicas
        else:
            per_batch = self.batch_size
        return self._start + self._batches_yielded * per_batch

    def __iter__(self):
        self.sampler.set_epoch(self._epoch)
        consumed = self._consumed()
        if consumed > 0 and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_position(consumed)
        elif consumed > 0:
            self.sampler.set_skip(consumed)
        self._iterator = iter(self._dataloader)
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
            self._start = 0
            self._iterator = None
            raise

    def __len__(self) -> int:
        return len(self._dataloader)

    def state_dict(self) -> dict:
        """Return checkpoint state.

        Keys: ``epoch``, ``batch_size``, ``sampler``, the position, and
        ``batches_yielded`` (this rank's batches this epoch, read by older
        versions). The position is ``global_position``, the samples all ranks
        consumed this epoch, for a ``DistributedSampler``, and
        ``samples_consumed``, this rank's samples, for any other sampler.
        """
        consumed = self._consumed()
        state: dict = {
            "epoch": self._epoch,
            "batch_size": self.batch_size,
            "sampler": self.sampler.state_dict(),
        }
        if isinstance(self.sampler, DistributedSampler):
            state["global_position"] = consumed
            state["batches_yielded"] = consumed // (self.batch_size * self.sampler.num_replicas)
        else:
            state["samples_consumed"] = consumed
            state["batches_yielded"] = consumed // self.batch_size
        return state

    def load_state_dict(self, state: dict) -> None:
        """Restore from checkpoint, on another data-parallel size or batch size where exact.

        A ``global_position`` resumes exactly on any data-parallel size and batch
        size. A per-rank position resumes exactly on the data-parallel size that
        saved it, at any batch size. A checkpoint that predates the saved batch
        size counts batches, so it resumes as before, on the saved data-parallel
        size only. A position with no full batch left in its epoch starts the
        next epoch. ``__iter__`` applies the position, so a later ``iter()`` in
        the same epoch stays aligned.

        Raises:
            DataPositionError: The data-parallel size changed and the saved
                position is per rank (a mixture, or a checkpoint that predates
                the saved batch size), or the state is a streaming position.
        """
        if "stream" in state:
            raise DataPositionError(
                "The checkpoint holds a streaming data position, but this run reads a "
                "map-style dataset (data.hf_streaming changed), so it has no sample to "
                "resume at."
            )
        sampler_state = state.get("sampler", {})
        replicas = self.sampler.num_replicas
        saved_replicas = sampler_state.get("num_replicas", replicas)
        saved_batch = state.get("batch_size")
        epoch = state.get("epoch", 0)
        global_order = isinstance(self.sampler, DistributedSampler)
        if global_order and "global_position" in state:
            start = state["global_position"]
        elif saved_replicas == replicas:
            per_rank = state.get("samples_consumed")
            if per_rank is None:
                per_rank = state.get("batches_yielded", 0) * self.batch_size
            start = per_rank * replicas if global_order else per_rank
        else:
            if global_order:
                why = (
                    "the checkpoint predates the saved batch size, so the global position it "
                    "reached is unknown; a checkpoint saved after resuming there records it"
                )
            else:
                why = (
                    "a mixture draws each rank's samples separately, so a position on one "
                    "data-parallel size has no counterpart on another"
                )
            raise DataPositionError(
                f"Cannot resume the data position saved on data-parallel size "
                f"{saved_replicas} on data-parallel size {replicas}: {why}. Resume on "
                f"{saved_replicas} data-parallel ranks."
            )

        self._epoch = epoch
        self.sampler.load_state_dict(sampler_state)
        self._batches_yielded = 0
        self._start = start
        sampler = self.sampler
        if (
            isinstance(sampler, DistributedSampler)
            and start > 0
            and sampler.samples_after(start) < self.batch_size
        ):
            logger.info(
                f"Saved data position {start} leaves no full batch in epoch {epoch} on "
                f"data-parallel size {replicas}; resuming at the start of epoch {epoch + 1}"
            )
            self._epoch = epoch + 1
            self._start = 0
        batch_changed = saved_batch is not None and saved_batch != self.batch_size
        if saved_replicas != replicas or batch_changed:
            logger.warning(
                f"Data position carried over: saved on data-parallel size {saved_replicas} "
                f"with batch size {saved_batch}, resuming epoch {self._epoch} at sample "
                f"{self._start} on data-parallel size {replicas} with batch size "
                f"{self.batch_size}"
            )

        logger.info(f"Resumed DataLoader: epoch={self._epoch}, position={self._start}")


class StreamingDataLoader:
    """DataLoader over a ``StreamingHuggingFaceDataset`` that resumes by replaying.

    A stream has no index to seek to, so the position is the number of batches
    training took this epoch. A resume rebuilds the epoch's stream and discards
    that many batches before yielding, which lands on the exact next sample
    because the shuffle, the split across ranks and workers, and the packing are
    deterministic. ``load_state_dict`` refuses a position saved under anything
    that changes the stream: the dataset's ``stream_identity``, the batch size or
    the worker count. The replay rereads and retokenizes the part of the epoch
    already trained on, so a resume deep into an epoch takes time.

    Args:
        dataset: The streaming dataset.
        batch_size: Per-device micro-batch size.
        config: Data pipeline configuration.
    """

    def __init__(
        self,
        dataset: StreamingHuggingFaceDataset,
        batch_size: int,
        config: DataConfig | None = None,
    ) -> None:
        config = config or DataConfig()
        self.dataset = dataset
        self.batch_size = batch_size
        self._num_workers = config.num_workers
        self._dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
            persistent_workers=config.num_workers > 0,
            worker_init_fn=ignore_shutdown_signals_in_worker,
        )
        self._epoch = 0
        self._batches_yielded = 0
        self._iterator = None

    def _identity(self) -> dict:
        return {
            **self.dataset.stream_identity(),
            "batch_size": self.batch_size,
            "num_workers": self._num_workers,
        }

    def __iter__(self):
        self.dataset.set_epoch(self._epoch)
        self._iterator = iter(self._dataloader)
        if self._batches_yielded:
            self._replay()
        return self

    def _replay(self) -> None:
        """Discard the batches this epoch already gave training."""
        logger.info(
            f"Replaying {self._batches_yielded} batches of epoch {self._epoch} to restore "
            f"the stream position"
        )
        assert self._iterator is not None
        for replayed in range(self._batches_yielded):
            try:
                next(self._iterator)
            except StopIteration:
                logger.warning(
                    f"The stream ended after {replayed} of {self._batches_yielded} replayed "
                    f"batches; starting epoch {self._epoch + 1}"
                )
                self._epoch += 1
                self._batches_yielded = 0
                self.dataset.set_epoch(self._epoch)
                self._iterator = iter(self._dataloader)
                return

    def __next__(self) -> dict[str, torch.Tensor]:
        if self._iterator is None:
            raise StopIteration
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._epoch += 1
            self._batches_yielded = 0
            self._iterator = None
            raise
        self._batches_yielded += 1
        return batch

    def state_dict(self) -> dict:
        """Return checkpoint state: ``epoch``, ``batches_yielded`` and the ``stream`` it counts."""
        return {
            "epoch": self._epoch,
            "batches_yielded": self._batches_yielded,
            "stream": self._identity(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore the position; the next ``iter()`` replays the stream to it.

        Raises:
            DataPositionError: The state is a map-style position, or something
                that changes the stream differs from when it was saved.
        """
        saved = state.get("stream")
        if saved is None:
            raise DataPositionError(
                "The checkpoint holds a map-style data position, but this run streams its "
                "data (data.hf_streaming changed), so the position cannot be replayed."
            )
        current = self._identity()
        changed = sorted(k for k in set(saved) | set(current) if saved.get(k) != current.get(k))
        if changed:
            detail = "; ".join(
                f"{k}: saved {saved.get(k)!r}, now {current.get(k)!r}" for k in changed
            )
            raise DataPositionError(
                f"Cannot replay the stream to its saved position because the stream "
                f"changed ({detail}). Resume with the saved values."
            )
        self._epoch = state.get("epoch", 0)
        self._batches_yielded = state.get("batches_yielded", 0)
        logger.info(
            f"Resumed streaming DataLoader: epoch={self._epoch}, replaying "
            f"{self._batches_yielded} batches on the next iteration"
        )
