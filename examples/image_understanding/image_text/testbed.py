"""Readers for single-image question-answer and caption annotations.

Each loader takes an annotation file in its dataset's published layout and the
directory its image paths resolve against.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
from torch.utils.data import Dataset

from .decode import decode_rgb
from .document import Document, Turn, is_text
from .skip import MAX_ATTEMPTS, SkipLog, check_max_attempts, first_usable


class ImageTextRows(Dataset):
    """Single-image, single-turn rows held in an Arrow table; images are read on access.

    The table is shared with forked loader workers without the per-row copies that
    touching Python objects would cause. Blank answers are dropped, and with them
    any row left without an answer or without a user message; answers lose their
    surrounding whitespace, which would otherwise be trained. Images are decoded
    upright; a row whose image
    does not decode is passed over for the next one, up to ``max_attempts`` in a
    row, and ``skipped`` counts them. A missing image file is an error.
    """

    def __init__(
        self,
        image_root: str | Path,
        images: list[str],
        users: list[str],
        answers: list[list[str]],
        max_attempts: int = MAX_ATTEMPTS,
    ) -> None:
        if not len(images) == len(users) == len(answers):
            raise ValueError(
                f"column lengths differ: {len(images)} images, {len(users)} users, "
                f"{len(answers)} answer lists"
            )
        usable = [[answer.strip() for answer in row if is_text(answer)] for row in answers]
        kept = [i for i, row in enumerate(usable) if row and is_text(users[i])]
        self.image_root = Path(image_root)
        self.max_attempts = check_max_attempts(max_attempts)
        self._skips = SkipLog()
        self._rows = pa.table(
            {
                "image": [images[i] for i in kept],
                "user": [users[i] for i in kept],
                "answers": [usable[i] for i in kept],
            }
        )

    @property
    def skipped(self) -> int:
        return self._skips.count

    def __len__(self) -> int:
        return self._rows.num_rows

    def _read(self, index: int) -> Document:
        row = self._rows.slice(index, 1).to_pylist()[0]
        image = decode_rgb(self.image_root / row["image"])
        return Document(images=(image,), turns=(Turn(row["user"], tuple(row["answers"])),))

    def read_with_index(self, index: int) -> tuple[int, Document]:
        """The actual row index and document after passing over unusable rows."""
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} is out of range for {len(self)} rows")
        return first_usable(
            index,
            len(self),
            self.max_attempts,
            lambda row: (row, self._read(row)),
            self._skips,
        )

    def __getitem__(self, index: int) -> Document:
        return self.read_with_index(index)[1]


def _load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _answers(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value]
    return [str(v) for v in values if v is not None]


def coco_captions(
    annotation_file: str | Path, image_root: str | Path, prompt: str
) -> ImageTextRows:
    """COCO captions in the Karpathy layout, one row per caption, asked with ``prompt``."""
    rows = _load_json(annotation_file)
    return ImageTextRows(
        image_root,
        [row["image"] for row in rows],
        [prompt] * len(rows),
        [_answers(row["caption"]) for row in rows],
    )


def vqav2(annotation_file: str | Path, image_root: str | Path) -> ImageTextRows:
    """VQAv2 in the LAVIS layout, every annotator answer kept as a candidate."""
    rows = _load_json(annotation_file)
    return ImageTextRows(
        image_root,
        [row["image"] for row in rows],
        [row["question"] for row in rows],
        [_answers(row["answer"]) for row in rows],
    )


def textvqa(annotation_file: str | Path, image_root: str | Path) -> ImageTextRows:
    """TextVQA, whose images are named by ``image_id``."""
    rows = _load_json(annotation_file)["data"]
    return ImageTextRows(
        image_root,
        [f"{row['image_id']}.jpg" for row in rows],
        [row["question"] for row in rows],
        [_answers(row["answers"]) for row in rows],
    )


def docvqa(annotation_file: str | Path, image_root: str | Path) -> ImageTextRows:
    """DocVQA single-page questions."""
    rows = _load_json(annotation_file)["data"]
    return ImageTextRows(
        image_root,
        [row["image"] for row in rows],
        [row["question"] for row in rows],
        [_answers(row["answers"]) for row in rows],
    )


def chartqa(annotation_file: str | Path, image_root: str | Path) -> ImageTextRows:
    """ChartQA questions, human-written or augmented."""
    rows = _load_json(annotation_file)
    return ImageTextRows(
        image_root,
        [row["imgname"] for row in rows],
        [row["query"] for row in rows],
        [_answers(row["label"]) for row in rows],
    )
