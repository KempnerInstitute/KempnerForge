"""Readers for single-image question-answer and caption annotations.

Each loader takes an annotation file in its dataset's published layout and the
directory its image paths resolve against.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
from PIL import Image
from torch.utils.data import Dataset

from .document import Document, Turn


class ImageTextRows(Dataset):
    """Single-image, single-turn rows held in an Arrow table; images are read on access.

    Empty answers are dropped, and with them any row left without an answer.
    """

    def __init__(
        self,
        image_root: str | Path,
        images: list[str],
        users: list[str],
        answers: list[list[str]],
    ) -> None:
        if not len(images) == len(users) == len(answers):
            raise ValueError(
                f"column lengths differ: {len(images)} images, {len(users)} users, "
                f"{len(answers)} answer lists"
            )
        usable = [[answer for answer in row if answer] for row in answers]
        kept = [i for i, row in enumerate(usable) if row]
        self.image_root = Path(image_root)
        self._rows = pa.table(
            {
                "image": [images[i] for i in kept],
                "user": [users[i] for i in kept],
                "answers": [usable[i] for i in kept],
            }
        )

    def __len__(self) -> int:
        return self._rows.num_rows

    def __getitem__(self, index: int) -> Document:
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} is out of range for {len(self)} rows")
        row = self._rows.slice(index, 1).to_pylist()[0]
        with Image.open(self.image_root / row["image"]) as image:
            rgb = image.convert("RGB")
        return Document(images=(rgb,), turns=(Turn(row["user"], tuple(row["answers"])),))


def _load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _answers(value: Any) -> list[str]:
    return [str(v) for v in value] if isinstance(value, list) else [str(value)]


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
