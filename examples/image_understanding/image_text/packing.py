"""Pack rendered documents, in order, into one row of a fixed length per batch.

``PackedLoader`` packs a map-style dataset and ``PackedStreamLoader`` a resumable stream.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sized
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from kempnerforge.data.vlm_dataset import pack_vlm_samples
from kempnerforge.resilience.signal_handler import ignore_shutdown_signals_in_worker

from .stream import ResumableStream

logger = logging.getLogger(__name__)


class _Order:
    """The dataset indices a pass reads, from a given position on."""

    def __init__(self, order: np.ndarray) -> None:
        self.order = order

    def __iter__(self) -> Iterator[int]:
        return map(int, self.order)

    def __len__(self) -> int:
        return len(self.order)


def footprint(sample: dict[str, torch.Tensor], tokens_per_image: int) -> int:
    """The positions ``sample`` takes in a packed row: its views' tokens, then its text."""
    pixels = sample["pixel_values"]
    views = 1 if pixels.dim() == 3 else pixels.shape[0]
    return views * tokens_per_image + int(sample["text_len"])


def pack_documents(
    documents: list[dict[str, torch.Tensor]],
    *,
    seq_len: int,
    tokens_per_image: int,
    pad_id: int,
    view_groups: bool = False,
) -> dict[str, torch.Tensor]:
    """``pack_vlm_samples`` over one row's documents, with ``attention_group_ids`` per view."""
    batch = pack_vlm_samples(
        documents, seq_len=seq_len, tokens_per_image=tokens_per_image, pad_id=pad_id
    )
    if view_groups:
        groups = torch.full_like(batch["input_ids"], -1)
        tokens = int(batch["image_tokens"])
        starts = batch["image_starts"].tolist()
        for view, (row, start) in enumerate(zip(batch["image_rows"].tolist(), starts, strict=True)):
            groups[row, start : start + tokens] = view
        batch["attention_group_ids"] = groups
    return batch


def _shut_down(iterator: Any) -> None:
    """Shut down a ``DataLoader`` iterator's workers, killing any that do not exit."""
    if iterator is None or not hasattr(iterator, "_shutdown_workers"):
        return
    iterator._shutdown_workers()
    for worker in getattr(iterator, "_workers", ()):
        if worker.is_alive():
            worker.kill()
            worker.join(timeout=5)


def _note_oversize(count: int, size: int, seq_len: int) -> int:
    """``count`` plus the document of ``size`` positions passed over, the first one logged."""
    if count == 0:
        logger.warning(
            "Passing over a document of %d positions, longer than seq_len=%d; "
            "the loader's oversize counts the rest",
            size,
            seq_len,
        )
    return count + 1


class PackedLoader:
    """Packs a dataset's samples into one row of ``seq_len`` per batch.

    Each pass shuffles every index of ``dataset`` from ``seed`` and the pass number,
    cuts the order to a multiple of ``world_size`` and gives ``rank`` every
    ``world_size``-th index. The samples are read by a ``DataLoader`` with
    ``options`` (workers, prefetching; the workers ignore shutdown signals unless
    ``worker_init_fn`` says otherwise) and packed here in that order: a row takes
    documents until the next one would overflow it, and that document opens the next
    row. Each batch is ``pack_vlm_samples`` over one row's documents. A document
    longer than ``seq_len`` is passed over and counted in ``oversize``, the first
    one logged. The ``DataLoader`` of a pass is seeded by its own generator, drawn
    from ``seed``, ``rank``, the pass and its start, so starting it leaves the
    global random state alone; ``in_order=False`` is refused, as a resume needs the
    documents in order. The workers ignore SIGTERM, so ``close`` shuts them down
    when done, as the training loop's ``BatchStream.close`` does. With
    ``view_groups`` each batch also carries ``attention_group_ids``, one group per
    image view's tokens and ``-1`` elsewhere, so the tokens of a view attend to each
    other in both directions. Passes follow
    one another without end, so every rank draws as many batches as training asks
    for. ``state_dict`` holds the pass and the position in it of the next row's
    first document, so a resume packs the same rows a run without the interruption
    would have.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        seq_len: int,
        tokens_per_image: int,
        pad_id: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        view_groups: bool = False,
        **options: Any,
    ) -> None:
        if seq_len <= 0 or tokens_per_image <= 0:
            raise ValueError("seq_len and tokens_per_image must be positive")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} is outside world_size {world_size}")
        if not isinstance(dataset, Sized) or len(dataset) < world_size:
            raise ValueError(f"the dataset needs at least {world_size} samples, one per rank")
        for name in (
            "batch_size",
            "sampler",
            "shuffle",
            "batch_sampler",
            "collate_fn",
            "generator",
        ):
            if name in options:
                raise ValueError(f"PackedLoader sets {name} itself")
        if options.get("in_order", True) is not True:
            raise ValueError("PackedLoader needs in_order=True to resume at its recorded position")
        self.dataset = dataset
        self.seq_len = seq_len
        self.tokens_per_image = tokens_per_image
        self.pad_id = pad_id
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.view_groups = view_groups
        self.options = {"worker_init_fn": ignore_shutdown_signals_in_worker, **options}
        self.oversize = 0
        self._iterator: Any = None
        self._pass = 0
        self._start = 0

    def order(self, pass_index: int) -> np.ndarray:
        """This rank's indices for pass ``pass_index``."""
        n = len(self.dataset)
        shuffled = np.random.default_rng([self.seed, pass_index]).permutation(n)
        return shuffled[: n - n % self.world_size][self.rank :: self.world_size]

    def _pack(self, documents: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return pack_documents(
            documents,
            seq_len=self.seq_len,
            tokens_per_image=self.tokens_per_image,
            pad_id=self.pad_id,
            view_groups=self.view_groups,
        )

    def _documents(self, pass_index: int, start: int) -> Iterator[dict[str, torch.Tensor]]:
        draw = np.random.default_rng([self.seed, self.rank, pass_index, start]).integers(2**62)
        loader = DataLoader(
            self.dataset,
            batch_size=None,
            sampler=_Order(self.order(pass_index)[start:]),
            generator=torch.Generator().manual_seed(int(draw)),
            **self.options,
        )
        self._iterator = iter(loader)
        return self._iterator

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        while True:
            pass_index, start = self._pass, self._start
            rows: list[dict[str, torch.Tensor]] = []
            fill = 0
            packed = False
            for position, sample in enumerate(self._documents(pass_index, start), start):
                size = footprint(sample, self.tokens_per_image)
                if size > self.seq_len:
                    self.oversize = _note_oversize(self.oversize, size, self.seq_len)
                    continue
                if rows and fill + size > self.seq_len:
                    self._start = position
                    packed = True
                    yield self._pack(rows)
                    rows, fill = [], 0
                rows.append(sample)
                fill += size
            self._pass, self._start = pass_index + 1, 0
            if rows:
                packed = True
                yield self._pack(rows)
            if not packed and start == 0:
                raise RuntimeError(f"pass {pass_index} packed no document for rank {self.rank}")

    def close(self) -> None:
        """Shut the loader workers down; an open loader with workers blocks interpreter exit."""
        iterator, self._iterator = self._iterator, None
        _shut_down(iterator)

    def state_dict(self) -> dict[str, Any]:
        return {"pass": self._pass, "start": self._start, "oversize": self.oversize}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._pass = int(state["pass"])
        self._start = int(state["start"])
        self.oversize = int(state.get("oversize", 0))


class PackedStreamLoader:
    """Packs a resumable stream's samples into one row of ``seq_len`` per batch.

    ``stream`` is an ``IterableDataset`` of ``(position, sample)`` pairs that records
    positions with ``advance`` and resumes from ``state_dict``, as ``RenderedStream``
    over ``DataCompStream`` does. Its samples are read by a ``DataLoader`` with
    ``options`` and packed here in order, as ``PackedLoader`` packs them, with
    oversize documents passed over and counted in ``oversize``. A document's
    position is recorded once it is placed in a row, after the rows before it are
    handed out, so ``state_dict`` resumes at the first document of the next row.
    The loader workers start afresh from the recorded positions on each ``iter``,
    so ``persistent_workers`` is refused; ``close`` shuts them down. A stream that
    ends yields its last, partly filled row; one that ends before any document
    fits, or passes over ``max_oversize_run`` documents in a row, raises
    ``RuntimeError``, as nothing it holds fits ``seq_len``.
    """

    def __init__(
        self,
        stream: ResumableStream,
        *,
        seq_len: int,
        tokens_per_image: int,
        pad_id: int,
        view_groups: bool = False,
        max_oversize_run: int = 10_000,
        **options: Any,
    ) -> None:
        if not isinstance(stream, IterableDataset):
            raise TypeError(
                f"PackedStreamLoader needs an IterableDataset, got {type(stream).__name__}"
            )
        if seq_len <= 0 or tokens_per_image <= 0 or max_oversize_run <= 0:
            raise ValueError("seq_len, tokens_per_image and max_oversize_run must be positive")
        for name in ("batch_size", "sampler", "shuffle", "batch_sampler", "collate_fn"):
            if name in options:
                raise ValueError(f"PackedStreamLoader sets {name} itself")
        if options.get("persistent_workers"):
            raise ValueError(
                "PackedStreamLoader starts its workers afresh, so persistent_workers is refused"
            )
        self.stream = stream
        self.seq_len = seq_len
        self.tokens_per_image = tokens_per_image
        self.pad_id = pad_id
        self.view_groups = view_groups
        self.max_oversize_run = max_oversize_run
        self.options = {"worker_init_fn": ignore_shutdown_signals_in_worker, **options}
        self.oversize = 0
        self._iterator: Any = None

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        _shut_down(self._iterator)
        self._iterator = iter(DataLoader(self.stream, batch_size=None, **self.options))
        rows: list[dict[str, torch.Tensor]] = []
        fill = 0
        placed = False
        run = 0
        for position, sample in self._iterator:
            size = footprint(sample, self.tokens_per_image)
            if size > self.seq_len:
                self.stream.advance(position)
                self.oversize = _note_oversize(self.oversize, size, self.seq_len)
                run += 1
                if run >= self.max_oversize_run:
                    raise RuntimeError(
                        f"{run} documents in a row are longer than seq_len={self.seq_len}"
                    )
                continue
            run = 0
            placed = True
            if rows and fill + size > self.seq_len:
                yield pack_documents(
                    rows,
                    seq_len=self.seq_len,
                    tokens_per_image=self.tokens_per_image,
                    pad_id=self.pad_id,
                    view_groups=self.view_groups,
                )
                rows, fill = [], 0
            self.stream.advance(position)
            rows.append(sample)
            fill += size
        if not placed:
            raise RuntimeError(f"the stream ended before a document fit seq_len={self.seq_len}")
        if rows:
            yield pack_documents(
                rows,
                seq_len=self.seq_len,
                tokens_per_image=self.tokens_per_image,
                pad_id=self.pad_id,
                view_groups=self.view_groups,
            )

    def close(self) -> None:
        """Shut the loader workers down; an open loader with workers blocks interpreter exit."""
        iterator, self._iterator = self._iterator, None
        _shut_down(iterator)

    def state_dict(self) -> dict[str, Any]:
        return {"stream": self.stream.state_dict(), "oversize": self.oversize}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.stream.load_state_dict(state["stream"])
        self.oversize = int(state.get("oversize", 0))
