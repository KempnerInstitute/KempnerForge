"""Readers of evaluation splits that no training source reads.

PixMo pointing and counting take the PixMo evaluation files, and GSM8K and MMLU
are text-only, each read from its test split as a parquet file. Each row is a
one-turn document whose question is the one the board's evaluation asks.
MathVista reads a split's parquet file as published, its images inside it, and
asks each problem's query.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from image_text.decode import decode_rgb
from image_text.document import Document, Turn
from image_text.points import format_point_answer
from image_text.skip import (
    MAX_ATTEMPTS,
    SkipLog,
    UnusableRowError,
    check_max_attempts,
    first_usable,
)
from image_text.testbed import ImageTextRows
from torch.utils.data import Dataset

from .extraction import bigbird_extractor
from .scorers import MathVistaReference

MMLU_LETTERS = ("A", "B", "C", "D")
MATHVISTA_COLUMNS = (
    "pid",
    "query",
    "answer",
    "question_type",
    "answer_type",
    "choices",
    "precision",
    "decoded_image",
)
_STRING = r'"((?:[^"\\]|\\.)*)"'
_IMAGE = re.compile(rf'"image"\s*:\s*{_STRING}')
_LABEL = re.compile(rf'"label"\s*:\s*{_STRING}')


def _string(pattern: re.Pattern[str], line: str, name: str) -> str:
    """The JSON string value of field ``name``, which ``pattern`` finds last in ``line``."""
    found = pattern.findall(line)
    if not found:
        raise ValueError(f"a row has no {name!r} field")
    return json.loads(f'"{found[-1]}"')


class PixmoPointsEval(Dataset):
    """PixMo pointing questions, each with its objects' segmentation masks for ``reference``.

    A row asks "Point to the {label}." of its image. Its answer is the
    annotated points as a point tag, kept for the predictions file; the row is
    scored against its masks, which ``reference`` reads from the row's line only
    when asked, as the file is large. An image that does not decode is passed
    over as ``ImageTextRows`` does.
    """

    def __init__(
        self,
        annotation_file: str | Path,
        image_root: str | Path,
        max_attempts: int = MAX_ATTEMPTS,
    ) -> None:
        self.annotation_file = Path(annotation_file)
        self.image_root = Path(image_root)
        self.max_attempts = check_max_attempts(max_attempts)
        self._skips = SkipLog()
        self._offsets: list[int] = []
        self._images: list[str] = []
        self._labels: list[str] = []
        with open(self.annotation_file, "rb") as file:
            offset = 0
            for raw in file:
                line = raw.decode()
                if line.strip():
                    self._offsets.append(offset)
                    self._images.append(_string(_IMAGE, line, "image"))
                    self._labels.append(_string(_LABEL, line, "label"))
                offset += len(raw)

    @property
    def skipped(self) -> int:
        return self._skips.count

    def __len__(self) -> int:
        return len(self._offsets)

    def _row(self, index: int) -> dict[str, Any]:
        with open(self.annotation_file, "rb") as file:
            file.seek(self._offsets[index])
            return json.loads(file.readline())

    def reference(self, index: int) -> np.ndarray:
        """Row ``index``'s per-object masks, ``(objects, H, W)`` booleans."""
        masks = np.asarray(self._row(index)["masks"], dtype=bool)
        return masks.reshape(0, 0, 0) if masks.size == 0 else masks

    def _read(self, index: int) -> Document:
        image = decode_rgb(self.image_root / self._images[index])
        label = self._labels[index]
        points = self._row(index).get("points") or []
        answer = format_point_answer(
            [(float(p["x"]), float(p["y"])) for p in points], label, absent_text="none"
        )
        return Document(images=(image,), turns=(Turn(f"Point to the {label}.", (answer,)),))

    def read_with_index(self, index: int) -> tuple[int, Document]:
        """The actual row index and document after passing over unusable rows."""
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} is out of range for {len(self)} rows")
        return first_usable(
            index, len(self), self.max_attempts, lambda row: (row, self._read(row)), self._skips
        )

    def __getitem__(self, index: int) -> Document:
        return self.read_with_index(index)[1]


class TextRows(Dataset):
    """Text-only one-turn documents: a question and its reference answer per row."""

    def __init__(self, questions: list[str], answers: list[str]) -> None:
        if len(questions) != len(answers):
            raise ValueError(f"{len(questions)} questions but {len(answers)} answers")
        self._questions = questions
        self._answers = answers

    def __len__(self) -> int:
        return len(self._questions)

    def __getitem__(self, index: int) -> Document:
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} is out of range for {len(self)} rows")
        return Document(images=(), turns=(Turn(self._questions[index], (self._answers[index],)),))


class MathVistaRows(Dataset):
    """MathVista problems from a split's parquet file, as the dataset publishes it.

    A row asks its ``query`` (the question with its hint and any lettered
    options) of its ``decoded_image`` and is answered by ``answer``.
    ``reference(row)`` is the row's ``MathVistaReference``, whose ``extract`` is
    ``extractor``. Every row is checked as the file is read, so a split the
    scorer cannot score is refused before anything is decoded (the test split
    publishes no answers). An image that does not decode, or a row without
    image bytes, is passed over as ``ImageTextRows`` does.

    Raises:
        ValueError: the file lacks a column of ``MATHVISTA_COLUMNS``, or a row
            is one ``MathVistaReference`` refuses.
    """

    def __init__(
        self,
        annotation_file: str | Path,
        extractor: Callable[[str, str], str],
        max_attempts: int = MAX_ATTEMPTS,
    ) -> None:
        self.annotation_file = Path(annotation_file)
        self.max_attempts = check_max_attempts(max_attempts)
        self._skips = SkipLog()
        missing = sorted(set(MATHVISTA_COLUMNS) - set(pq.read_schema(self.annotation_file).names))
        if missing:
            raise ValueError(
                f"{self.annotation_file} has no {missing} column; expected a MathVista split "
                f"with the columns {list(MATHVISTA_COLUMNS)}"
            )
        table = pq.read_table(self.annotation_file, columns=list(MATHVISTA_COLUMNS))
        self._images = table.column("decoded_image")
        self._pids: list[str] = table.column("pid").to_pylist()
        problems = table.drop_columns(["decoded_image"]).to_pylist()
        self._references = [self._reference(problem, extractor) for problem in problems]

    def _reference(
        self, problem: dict[str, Any], extractor: Callable[[str, str], str]
    ) -> MathVistaReference:
        precision = problem["precision"]
        try:
            return MathVistaReference(
                answer=problem["answer"],
                query=problem["query"],
                question_type=problem["question_type"],
                answer_type=problem["answer_type"],
                extract=extractor,
                choices=tuple(problem["choices"] or ()),
                precision=None if precision is None else int(precision),
            )
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"{self.annotation_file}: problem {problem['pid']}: {exc}") from exc

    @property
    def skipped(self) -> int:
        return self._skips.count

    def __len__(self) -> int:
        return len(self._references)

    def reference(self, index: int) -> MathVistaReference:
        """Row ``index``'s problem, which its reply is scored against."""
        return self._references[index]

    def _read(self, index: int) -> Document:
        image = self._images[index].as_py()
        if image is None or image.get("bytes") is None:
            raise UnusableRowError(f"problem {self._pids[index]} has no image bytes")
        problem = self._references[index]
        return Document(
            images=(decode_rgb(image["bytes"]),),
            turns=(Turn(problem.query, (problem.answer,)),),
        )

    def read_with_index(self, index: int) -> tuple[int, Document]:
        """The actual row index and document after passing over unusable rows."""
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} is out of range for {len(self)} rows")
        return first_usable(
            index, len(self), self.max_attempts, lambda row: (row, self._read(row)), self._skips
        )

    def __getitem__(self, index: int) -> Document:
        return self.read_with_index(index)[1]


def _jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path) as file:
        return [json.loads(line) for line in file if line.strip()]


def _parquet(path: str | Path, rows: Callable[[dict[str, Any]], tuple[str, str]]) -> TextRows:
    pairs = [rows(row) for row in pq.read_table(path).to_pylist()]
    return TextRows([q for q, _ in pairs], [a for _, a in pairs])


def pixmo_points_eval(annotation_file: str | Path, image_root: str | Path) -> PixmoPointsEval:
    """PixMo pointing: "Point to the {label}.", scored against the objects' masks."""
    return PixmoPointsEval(annotation_file, image_root)


def pixmo_count_eval(annotation_file: str | Path, image_root: str | Path) -> ImageTextRows:
    """PixMo counting: "How many {label} are in the image?", answered by the count."""
    rows = _jsonl(annotation_file)
    return ImageTextRows(
        image_root,
        [row["image"] for row in rows],
        [f"How many {row['label']} are in the image?" for row in rows],
        [[str(row["count"])] for row in rows],
    )


def gsm8k(annotation_file: str | Path) -> TextRows:
    """GSM8K: the question, answered by the number after the solution's ``####``."""

    def row(record: dict[str, Any]) -> tuple[str, str]:
        gold = record["answer"].split("####")[-1].strip().replace(",", "")
        return record["question"], gold

    return _parquet(annotation_file, row)


def mmlu(annotation_file: str | Path) -> TextRows:
    """MMLU: the question with its options lettered A-D, answered by the correct letter."""

    def row(record: dict[str, Any]) -> tuple[str, str]:
        options = "\n".join(
            f"{MMLU_LETTERS[i]}. {choice}" for i, choice in enumerate(record["choices"])
        )
        question = f"{record['question']}\n{options}\nAnswer with the letter of the correct option."
        return question, MMLU_LETTERS[int(record["answer"])]

    return _parquet(annotation_file, row)


def mathvista(annotation_file: str | Path, extractor_device: str = "cpu") -> MathVistaRows:
    """MathVista: each problem's query asked of its image, scored by ``mathvista_accuracy``.

    ``annotation_file`` is a split's parquet file as published (``testmini``,
    the split with answers). A reply that needs its answer extracted is read by
    ``bigbird_extractor`` on ``extractor_device``, whose files are fetched when
    the reader is made and whose model is loaded when a row first needs it.
    """
    return MathVistaRows(annotation_file, bigbird_extractor(extractor_device))
