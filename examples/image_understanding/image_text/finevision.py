"""FineVision subsets read from their memory-mappable Arrow copy.

``scripts/convert_finevision_to_arrow.py`` writes each parquet file of a subset as
an Arrow IPC file with one record batch per row group, plus ``index.npz``: the
file list, record batches per file, rows per batch, a per-row keep flag, per-row
image counts and the rating columns. Rows are numbered across the subset in that
file and batch order.
"""

from __future__ import annotations

import bisect
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
from torch.utils.data import Dataset

from .decode import decode_rgb
from .document import Document, Turn
from .skip import MAX_ATTEMPTS, SkipLog, UnusableRowError, check_max_attempts, first_usable

RATING_COLUMNS = (
    "relevance_min",
    "visual_dependency_min",
    "image_correspondence_min",
    "formatting_min",
)


def _is_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


class FineVisionArrow(Dataset):
    """The kept conversations of one FineVision subset.

    Rows pass when their keep flag is set, every ``min_ratings`` column meets its
    threshold (a missing rating, stored as ``-1``, fails), and the row has between
    1 and ``max_images`` images, which ``None`` lifts; a ``text_only`` reader takes
    no image filter and its documents carry no image, for a renderer that accepts
    them. ``num_images`` overrides the per-row image counts stored in ``index.npz``.
    Turns missing their user message or their reply are dropped; a row left
    without a turn, or whose image does not decode, is passed over for the next
    row, up to ``max_attempts`` in a row, and ``skipped`` counts them.
    """

    def __init__(
        self,
        subset_dir: str | Path,
        *,
        max_images: int | None = 1,
        num_images: np.ndarray | None = None,
        min_ratings: dict[str, int] | None = None,
        text_only: bool = False,
        max_attempts: int = MAX_ATTEMPTS,
        max_open_files: int = 4096,
    ) -> None:
        if max_open_files < 1:
            raise ValueError(f"max_open_files must be positive, got {max_open_files}")
        self.subset_dir = Path(subset_dir)
        self.text_only = text_only
        self.max_attempts = check_max_attempts(max_attempts)
        self.max_open_files = max_open_files
        self._skips = SkipLog()
        index = np.load(self.subset_dir / "index.npz")
        self.files = [str(name) for name in index["files"]]
        batch_rows = index["batch_rows"]
        self._batch_starts = np.concatenate([[0], np.cumsum(batch_rows)[:-1]]).tolist()
        self._batches = [
            (file, batch)
            for file, count in enumerate(index["file_batches"].tolist())
            for batch in range(count)
        ]
        keep = index["keep"].astype(bool).copy()
        for name, threshold in (min_ratings or {}).items():
            if name not in RATING_COLUMNS:
                raise ValueError(f"unknown rating {name!r}; expected one of {RATING_COLUMNS}")
            keep &= index[name] >= threshold
        if max_images is not None and not text_only:
            counts = num_images if num_images is not None else index.get("num_images")
            if counts is None:
                raise ValueError(
                    f"{self.subset_dir / 'index.npz'} has no num_images; pass num_images "
                    "or rebuild the index with scripts/convert_finevision_to_arrow.py"
                )
            if len(counts) != len(keep):
                raise ValueError(f"num_images has {len(counts)} rows, the index {len(keep)}")
            keep &= (counts >= 1) & (counts <= max_images)
        self.rows = np.flatnonzero(keep).astype(np.int64)
        self._readers: OrderedDict[int, Any] = OrderedDict()

    @property
    def skipped(self) -> int:
        return self._skips.count

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_readers"] = OrderedDict()
        return state

    def __len__(self) -> int:
        return len(self.rows)

    def _reader(self, file: int) -> Any:
        reader = self._readers.pop(file, None)
        if reader is None:
            if len(self._readers) >= self.max_open_files:
                self._readers.popitem(last=False)
            path = self.subset_dir / self.files[file]
            reader = pa.ipc.open_file(pa.memory_map(str(path), "r"))
        self._readers[file] = reader
        return reader

    def row(self, index: int) -> dict[str, Any]:
        """The ``images`` (unless text-only), ``texts`` and ``source`` of kept row ``index``."""
        global_row = int(self.rows[index])
        position = bisect.bisect_right(self._batch_starts, global_row) - 1
        file, batch = self._batches[position]
        offset = global_row - self._batch_starts[position]
        record = self._reader(file).get_batch(batch).slice(offset, 1)
        columns = ["texts", "source"] if self.text_only else ["images", "texts", "source"]
        return {name: record.column(name)[0].as_py() for name in columns}

    def _read(self, index: int) -> Document:
        row = self.row(index)
        images = (
            () if self.text_only else tuple(decode_rgb(item["bytes"]) for item in row["images"])
        )
        turns = tuple(
            Turn(t["user"], (t["assistant"],))
            for t in row["texts"] or []
            if _is_text(t.get("user")) and _is_text(t.get("assistant"))
        )
        if not turns:
            raise UnusableRowError("no turn has both a user message and a reply")
        return Document(images=images, turns=turns, kind="conversation")

    def __getitem__(self, index: int) -> Document:
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} is out of range for {len(self)} rows")
        return first_usable(index, len(self), self.max_attempts, self._read, self._skips)
