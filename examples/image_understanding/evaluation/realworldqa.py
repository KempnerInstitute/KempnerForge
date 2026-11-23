"""RealWorldQA, read from the parquet files its test split is published in.

The Hugging Face Hub publishes RealWorldQA (``xai-org/RealworldQA``, CC BY-ND
4.0) as one ``test`` split, in shards named ``data/test-<i>-of-<n>.parquet``.
A row holds one image, encoded in the ``bytes`` field of its ``image`` struct,
a ``question`` that ends with its own answer instruction (a single word or
number, or the letter of an option), and one reference ``answer``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from image_text.decode import decode_rgb
from image_text.document import Document, Turn, is_text
from image_text.skip import (
    MAX_ATTEMPTS,
    SkipLog,
    UnusableRowError,
    check_max_attempts,
    first_usable,
)
from torch.utils.data import Dataset

COLUMNS = ("image", "question", "answer")
_SHARD = re.compile(r"test-(\d+)-of-(\d+)\.parquet")


def _is_text_type(kind: pa.DataType) -> bool:
    return pa.types.is_string(kind) or pa.types.is_large_string(kind)


def _is_image_type(kind: pa.DataType) -> bool:
    if not pa.types.is_struct(kind) or kind.get_field_index("bytes") < 0:
        return False
    encoded = kind.field("bytes").type
    return pa.types.is_binary(encoded) or pa.types.is_large_binary(encoded)


def _read_table(path: Path) -> pa.Table:
    """The columns of ``COLUMNS`` in the file at ``path``, the image struct flattened.

    The struct's fields become columns named ``image.<field>``, null where the
    struct is null. A file whose columns lack their types raises ``ValueError``.
    """
    schema = pq.read_schema(path)
    kinds = {field.name: field.type for field in schema}
    usable = (
        all(name in kinds for name in COLUMNS)
        and _is_image_type(kinds["image"])
        and _is_text_type(kinds["question"])
        and _is_text_type(kinds["answer"])
    )
    if not usable:
        found = {name: str(kind) for name, kind in kinds.items()}
        raise ValueError(
            f"{path}: a RealWorldQA file has an image struct with the encoded image in "
            f"its bytes field, and question and answer strings; its columns are {found}"
        )
    return pq.read_table(path, columns=list(COLUMNS)).flatten()


class RealWorldQARows(Dataset):
    """The questions of RealWorldQA parquet files, each with its image; images are read on access.

    Every file needs the columns of ``COLUMNS``: ``image``, a struct whose
    ``bytes`` field holds the encoded image, and the ``question`` and ``answer``
    strings; a file without them raises ``ValueError``. Rows follow the order of
    ``files``. The question is the user message as written, and the answer,
    stripped of surrounding whitespace, its one reference; a row with a blank
    question or answer is dropped. The encoded images are held in memory and
    decoded upright as RGB when their row is read; a row whose image holds no
    bytes or does not decode is passed over for the next one, up to
    ``max_attempts`` in a row, and ``skipped`` counts them.
    """

    def __init__(self, files: Sequence[str | Path], max_attempts: int = MAX_ATTEMPTS) -> None:
        self.max_attempts = check_max_attempts(max_attempts)
        self._skips = SkipLog()
        self._images: list[pa.ChunkedArray] = []
        self._rows: list[tuple[int, int, str, str]] = []
        for number, path in enumerate(files):
            table = _read_table(Path(path))
            self._images.append(table["image.bytes"])
            pairs = zip(table["question"].to_pylist(), table["answer"].to_pylist(), strict=True)
            for row, (question, answer) in enumerate(pairs):
                if is_text(question) and is_text(answer):
                    self._rows.append((number, row, question, answer.strip()))

    @property
    def skipped(self) -> int:
        return self._skips.count

    def __len__(self) -> int:
        return len(self._rows)

    def _read(self, index: int) -> Document:
        number, row, question, answer = self._rows[index]
        encoded = self._images[number][row].as_py()
        if encoded is None:
            raise UnusableRowError("the image holds no bytes")
        return Document(images=(decode_rgb(encoded),), turns=(Turn(question, (answer,)),))

    def read_with_index(self, index: int) -> tuple[int, Document]:
        """The actual row index and document after passing over unusable rows."""
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} is out of range for {len(self)} rows")
        return first_usable(
            index, len(self), self.max_attempts, lambda row: (row, self._read(row)), self._skips
        )

    def __getitem__(self, index: int) -> Document:
        return self.read_with_index(index)[1]


def _test_shards(dataset_dir: Path) -> list[Path]:
    """The shards of the test split under ``dataset_dir``, in shard order."""
    folder = dataset_dir / "data"
    numbers: dict[Path, tuple[int, int]] = {}
    for path in folder.glob("test-*.parquet"):
        match = _SHARD.fullmatch(path.name)
        if match is not None:
            numbers[path] = (int(match[1]), int(match[2]))
    if not numbers:
        raise FileNotFoundError(
            f"{folder} holds no shard of the test split, test-<i>-of-<n>.parquet"
        )
    counts = sorted({count for _, count in numbers.values()})
    if len(counts) != 1:
        raise ValueError(f"{folder}: the test shards disagree on their count: {counts}")
    found = sorted(index for index, _ in numbers.values())
    if found != list(range(counts[0])):
        raise ValueError(
            f"{folder}: the test split has {counts[0]} shards, numbered from 0; found {found}"
        )
    return sorted(numbers, key=lambda path: numbers[path][0])


def realworldqa(dataset_dir: str | Path) -> RealWorldQARows:
    """RealWorldQA's test split, from the dataset as downloaded from the Hub to ``dataset_dir``.

    ``dataset_dir`` holds the shards as the Hub lays them out,
    ``data/test-<i>-of-<n>.parquet`` with ``i`` from 0 to ``n - 1``, as
    ``hf download xai-org/RealworldQA --repo-type dataset --local-dir <dataset_dir>``
    writes them, and they are read in shard order. A directory without a shard
    raises ``FileNotFoundError``; shards that disagree on ``n``, or that leave
    one of the ``n`` out, raise ``ValueError``.
    """
    return RealWorldQARows(_test_shards(Path(dataset_dir)))
