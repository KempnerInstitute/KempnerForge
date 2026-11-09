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
import time
from collections.abc import Callable, Sized
from typing import Any

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


def _stream_change(key: str, saved: Any, now: Any) -> str:
    """One changed stream setting, with the data file list reduced to its count."""
    if key != "source" or not isinstance(saved, dict) or not isinstance(now, dict):
        return f"{key}: saved {saved!r}, now {now!r}"
    parts = []
    for part in sorted(set(saved) | set(now)):
        before, after = saved.get(part), now.get(part)
        if before == after:
            continue
        if part == "files":
            parts.append(
                f"data files differ ({len(before or [])} saved, {len(after or [])} now, "
                f"compared by name and size)"
            )
        else:
            parts.append(f"{part}: saved {before!r}, now {after!r}")
    return "source " + ", ".join(parts)


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

        Keys: ``epoch``, ``batch_size``, ``dataset_length``, ``sampler`` (which
        holds the shuffle seed), the position, and ``batches_yielded``. The
        position is ``global_position``, the samples all ranks consumed this
        epoch, for a ``DistributedSampler``, and ``samples_consumed``, this rank's
        samples, for any other sampler. ``batches_yielded`` is the position in
        whole batches of this loader's size, for versions that predate the
        position: the batches this rank took this epoch in a run that never
        changed size, and the position divided by the global batch, rounded down,
        after a resize.
        """
        consumed = self._consumed()
        state: dict = {
            "epoch": self._epoch,
            "batch_size": self.batch_size,
            "dataset_length": len(self.dataset) if isinstance(self.dataset, Sized) else None,
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
        size counts batches, so it resumes as before, only on the saved
        data-parallel size and exact only at the saved batch size, which it does
        not record. A position with no full batch left in its epoch starts the
        next epoch. A changed shuffle seed or dataset length is logged, since the
        position counts samples of the saved order. ``__iter__`` applies the
        position, so a later ``iter()`` in the same epoch stays aligned.

        Raises:
            DataPositionError: The data-parallel size changed and the saved
                position is per rank (a mixture's, or one that predates the saved
                batch size), or the state is a streaming position.
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
                if per_rank and saved_batch is None:
                    logger.warning(
                        f"This checkpoint predates the saved batch size, so its data position "
                        f"assumes the batch size is unchanged ({self.batch_size}); with "
                        f"another batch size at the save, samples are skipped or repeated"
                    )
            start = per_rank * replicas if global_order else per_rank
        else:
            raise DataPositionError(self._refusal(state, saved_replicas, global_order))

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
        self._warn_if_reordered(state, sampler_state)

        logger.info(f"Resumed DataLoader: epoch={self._epoch}, position={self._start}")

    def _refusal(self, state: dict, saved_replicas: int, global_order: bool) -> str:
        """Why a position saved on ``saved_replicas`` ranks cannot resume here, and the way out."""
        replicas = self.sampler.num_replicas
        ranks = "rank" if saved_replicas == 1 else "ranks"
        if not global_order:
            why = (
                "a mixture draws each rank's samples separately, so a position on one "
                "data-parallel size has no counterpart on another"
            )
            remedy = f"Resume on {saved_replicas} data-parallel {ranks}."
        elif "samples_consumed" in state:
            why = (
                "it counts one rank's samples of a mixture, which have no counterpart on "
                "another data-parallel size"
            )
            remedy = f"Resume on {saved_replicas} data-parallel {ranks}."
        else:
            why = (
                "the checkpoint predates the saved batch size, so the global position it "
                "reached is unknown"
            )
            remedy = (
                f"Resume on {saved_replicas} data-parallel {ranks} with the batch size it was "
                f"saved with; a checkpoint saved from there records the position and resumes "
                f"on any size."
            )
        return (
            f"Cannot resume the data position saved on data-parallel size {saved_replicas} "
            f"on data-parallel size {replicas}: {why}. {remedy}"
        )

    def _warn_if_reordered(self, state: dict, sampler_state: dict) -> None:
        """Log a shuffle seed or dataset length that differs from the checkpoint's."""
        saved_seed = sampler_state.get("seed")
        if saved_seed is not None and saved_seed != self.sampler.seed:
            logger.warning(
                f"The data seed is {self.sampler.seed} but the checkpoint was saved with "
                f"{saved_seed}: the position counts samples of the saved order, so samples "
                f"are skipped or repeated"
            )
        saved_length = state.get("dataset_length")
        length = len(self.dataset) if isinstance(self.dataset, Sized) else None
        if saved_length is not None and length is not None and saved_length != length:
            logger.warning(
                f"The dataset holds {length} samples but the checkpoint was saved with "
                f"{saved_length}: the position counts samples of the saved order, so samples "
                f"are skipped or repeated"
            )


class StreamingDataLoader:
    """DataLoader over a ``StreamingHuggingFaceDataset`` that resumes by replaying.

    A stream has no index to seek to, so the position is the number of batches
    training took this epoch. A resume rebuilds the epoch's stream and discards
    that many batches before yielding, which lands on the exact next sample
    because the shuffle, the split across ranks and workers, and the packing are
    deterministic. The replay rereads and retokenizes the part of the epoch
    already trained on, so a resume deep into an epoch takes time; it logs its
    progress.

    ``load_state_dict`` refuses a position saved under different values of what
    it checks: the dataset's ``stream_identity`` (the data files by name and size,
    or the Hub name and config, with the shard count; the data-parallel size, the
    seed, the shuffle buffer, ``seq_len``, the tokenizer as named and
    ``pack_sequences``), the batch size and the worker count. It does not see a
    file rewritten at the same size, a new revision of a Hub dataset with the same
    shard count, or a new version of ``datasets`` or of the tokenizer files; any
    of those can shift the replay.

    A checkpoint holds one rank's position, and every rank replays its own stream
    by it. That is exact while every rank started its current pass on the same
    step, so in practice until the first rank finishes a pass.

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
        self._loader_kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "num_workers": config.num_workers,
            "pin_memory": config.pin_memory,
            "prefetch_factor": config.prefetch_factor if config.num_workers > 0 else None,
            "persistent_workers": config.num_workers > 0,
            "worker_init_fn": ignore_shutdown_signals_in_worker,
        }
        self._dataloader = DataLoader(dataset, **self._loader_kwargs)
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
        if self._iterator is not None and self._num_workers > 0:
            self._restart_workers()
        self.dataset.set_epoch(self._epoch)
        self._iterator = iter(self._dataloader)
        if self._batches_yielded:
            self._replay()
        return self

    def _restart_workers(self) -> None:
        """Replace the persistent workers before a pass restarts midway.

        A worker keeps its own copy of the dataset, and one that already finished
        its part of the pass has moved that copy to the next epoch, so it would
        read the next epoch's shuffle. New workers copy the dataset at this epoch.
        """
        running = getattr(self._dataloader, "_iterator", None)
        if running is not None and hasattr(running, "_shutdown_workers"):
            running._shutdown_workers()
            for worker in getattr(running, "_workers", ()):
                if worker.is_alive():
                    worker.kill()
                    worker.join(timeout=10)
        self._dataloader = DataLoader(self.dataset, **self._loader_kwargs)

    def _replay(self) -> None:
        """Discard the batches this epoch already gave training, logging progress."""
        total = self._batches_yielded
        tokens_per_batch = self.batch_size * self.dataset.seq_len
        logger.info(
            f"Replaying {total} batches ({total * tokens_per_batch:,} tokens) of epoch "
            f"{self._epoch} to restore the stream position"
        )
        assert self._iterator is not None
        every = max(1, total // 10)
        started = time.monotonic()
        for replayed in range(1, total + 1):
            try:
                next(self._iterator)
            except StopIteration:
                logger.warning(
                    f"The stream ended after {replayed - 1} of {total} replayed batches; "
                    f"starting epoch {self._epoch + 1}"
                )
                self._epoch += 1
                self._batches_yielded = 0
                self.dataset.set_epoch(self._epoch)
                self._iterator = iter(self._dataloader)
                return
            if replayed % every == 0:
                rate = replayed / max(time.monotonic() - started, 1e-9)
                logger.info(
                    f"Replayed {replayed} of {total} batches, "
                    f"{rate * tokens_per_batch / 1e6:.2f}M tokens/s, about "
                    f"{(total - replayed) / rate:.0f} s left"
                )

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
            detail = "; ".join(_stream_change(k, saved.get(k), current.get(k)) for k in changed)
            raise DataPositionError(
                f"Cannot replay the stream to its saved position because the stream "
                f"changed ({detail}). Resume with the saved values."
            )
        self._epoch = state.get("epoch", 0)
        self._batches_yielded = state.get("batches_yielded", 0)
        if self._epoch > 0 and self.dataset.world_size > 1:
            logger.warning(
                f"Resuming the stream in pass {self._epoch} on {self.dataset.world_size} "
                f"data-parallel ranks: the checkpoint holds one rank's position, which is "
                f"right for every rank only while all ranks started this pass on the same "
                f"step. Ranks whose parts of the stream differ in length start later passes "
                f"on different steps, so a rank may skip or repeat up to that difference "
                f"until checkpoints hold each rank's own loader state"
            )
        logger.info(
            f"Resumed streaming DataLoader: epoch={self._epoch}, replaying "
            f"{self._batches_yielded} batches on the next iteration"
        )
