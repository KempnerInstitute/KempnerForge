"""Readers for the PixMo annotations, whose images share one pool named by URL hash.

A few images of the pool do not decode (truncated files, or ones over PIL's
decompression-bomb limit), so a read whose image does not decode moves on to the
next image.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
from PIL import Image
from torch.utils.data import Dataset

from kempnerforge.metrics.logger import get_logger

from .decode import UnreadableImageError, decode_rgb, first_readable
from .document import Document, Turn
from .points import check_max_points, format_point_answer, pixels_to_percent

logger = get_logger(__name__)

MAX_POINTS = 30
MAX_ATTEMPTS = 16
POINT_PLACEHOLDER = "<|POINT|>"


def _jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _is_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


class PixmoRows(Dataset):
    """Annotation rows over the PixMo image pool, skipping rows whose image does not decode.

    A read tries ``index`` and then the rows after it, ``max_attempts`` images in
    all: the rows that follow a failed one with the same image are passed over
    without counting as attempts.
    """

    def __init__(
        self,
        image_root: str | Path,
        columns: dict[str, list[Any]],
        max_attempts: int = MAX_ATTEMPTS,
    ) -> None:
        if "image" not in columns:
            raise ValueError("PixMo rows need an 'image' column")
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be positive, got {max_attempts}")
        self.image_root = Path(image_root)
        self.max_attempts = max_attempts
        self.skipped = 0
        self._rows = pa.table(columns)

    def __len__(self) -> int:
        return self._rows.num_rows

    def document(self, row: dict[str, Any], image: Image.Image) -> Document:
        """The document for one row and its decoded image."""
        raise NotImplementedError

    def _read(self, index: int) -> Document:
        row = self._rows.slice(index, 1).to_pylist()[0]
        return self.document(row, decode_rgb(self.image_root / row["image"]))

    def _skip(self, index: int, error: UnreadableImageError) -> None:
        if not self.skipped:
            logger.warning(f"skipping row {index}: {error}; later skips are counted, not logged")
        self.skipped += 1

    def _image(self, index: int) -> str:
        return self._rows["image"][index].as_py()

    def _next_image(self, index: int) -> int:
        """The first row after ``index`` whose image differs from its own."""
        image = self._image(index)
        for step in range(1, len(self)):
            if self._image((index + step) % len(self)) != image:
                return index + step
        return index + 1

    def __getitem__(self, index: int) -> Document:
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} is out of range for {len(self)} rows")
        return first_readable(
            index, len(self), self.max_attempts, self._read, self._skip, self._next_image
        )


class PixmoCaptions(PixmoRows):
    """Dense captions, asked with a fixed prompt."""

    def __init__(
        self,
        image_root: str | Path,
        columns: dict[str, list[Any]],
        prompt: str,
        max_attempts: int = MAX_ATTEMPTS,
    ) -> None:
        super().__init__(image_root, columns, max_attempts)
        self.prompt = prompt

    def document(self, row: dict[str, Any], image: Image.Image) -> Document:
        return Document(images=(image,), turns=(Turn(self.prompt, (row["caption"],)),))


class PixmoConversations(PixmoRows):
    """Multi-turn conversations about one image."""

    def document(self, row: dict[str, Any], image: Image.Image) -> Document:
        turns = tuple(
            Turn(user, (assistant,))
            for user, assistant in zip(row["users"], row["assistants"], strict=True)
        )
        return Document(images=(image,), turns=turns, kind="conversation")


class PixmoPointing(PixmoRows):
    """Pointing questions answered with point tags, or a not-present sentence."""

    def __init__(
        self,
        image_root: str | Path,
        columns: dict[str, list[Any]],
        max_points: int = MAX_POINTS,
        max_attempts: int = MAX_ATTEMPTS,
    ) -> None:
        super().__init__(image_root, columns, max_attempts)
        self.max_points = check_max_points(max_points)

    def document(self, row: dict[str, Any], image: Image.Image) -> Document:
        label = row["label"]
        answer = format_point_answer(
            list(zip(row["xs"], row["ys"], strict=True)),
            label,
            absent_text=f"There are no {label} in the image.",
            max_points=self.max_points,
        )
        return Document(images=(image,), turns=(Turn(f"Point to the {label}.", (answer,)),))


class PixmoCounting(PixmoRows):
    """Counting questions answered by pointing, then counting.

    ``pixels`` says the points are stored in pixels and converts them to percent
    of the decoded image's size. Every point is kept unless
    ``max_points`` caps them, so the points agree with the stated count; a cap
    keeps the first points and still states the full count. The answer lists the
    points before the count, so when it outruns the renderer's text budget (at
    256 tokens, past about 13 points) the count is cut away and only the leading
    points train; ``max_points`` bounds the answer so the count fits.
    """

    def __init__(
        self,
        image_root: str | Path,
        columns: dict[str, list[Any]],
        *,
        pixels: bool,
        max_points: int | None = None,
        max_attempts: int = MAX_ATTEMPTS,
    ) -> None:
        super().__init__(image_root, columns, max_attempts)
        self.pixels = pixels
        self.max_points = check_max_points(max_points)

    def document(self, row: dict[str, Any], image: Image.Image) -> Document:
        label = row["label"]
        count = row["count"] or 0
        points = list(zip(row["xs"], row["ys"], strict=True))
        if self.pixels:
            points = pixels_to_percent(points, *image.size)
        if points:
            tag = format_point_answer(points, label, absent_text="", max_points=self.max_points)
            answer = f"{tag}\nCounting the {label}, I count {count}."
        else:
            answer = f"I count {count} {label} in the image."
        user = f"How many {label} are in the image?"
        return Document(images=(image,), turns=(Turn(user, (answer,)),))


def pixmo_captions(
    annotation_file: str | Path,
    image_root: str | Path,
    prompt: str,
    max_attempts: int = MAX_ATTEMPTS,
) -> PixmoCaptions:
    """PixMo-Cap dense captions."""
    images, captions = [], []
    for row in _jsonl(annotation_file):
        images.append(row["image"])
        captions.append(row["caption"])
    return PixmoCaptions(image_root, {"image": images, "caption": captions}, prompt, max_attempts)


def _conversations(
    image_root: str | Path,
    rows: Iterator[tuple[str, list[tuple[Any, Any]]]],
    max_attempts: int,
) -> PixmoConversations:
    images, users, assistants = [], [], []
    for image, pairs in rows:
        kept = [(u, a) for u, a in pairs if _is_text(u) and _is_text(a)]
        if kept:
            images.append(image)
            users.append([u for u, _ in kept])
            assistants.append([a for _, a in kept])
    return PixmoConversations(
        image_root, {"image": images, "users": users, "assistants": assistants}, max_attempts
    )


def pixmo_cap_qa(
    annotation_file: str | Path, image_root: str | Path, max_attempts: int = MAX_ATTEMPTS
) -> PixmoConversations:
    """PixMo-CapQA, whose ``messages[i]`` ends with turn ``i``'s question and answer."""
    return _conversations(
        image_root,
        (
            (
                row["image"],
                [
                    (element[-2], element[-1])
                    for element in row.get("messages") or []
                    if isinstance(element, list) and len(element) >= 2
                ],
            )
            for row in _jsonl(annotation_file)
        ),
        max_attempts,
    )


def _in_percent(points: list[tuple[float, float]]) -> bool:
    return all(0.0 <= x <= 100.0 and 0.0 <= y <= 100.0 for x, y in points)


def explanation_answer(
    parsed: str,
    points: Sequence[Sequence[Sequence[float]]],
    inline_text: Sequence[str],
    alt_text: Sequence[str],
) -> str | None:
    """The reply with each placeholder of ``parsed`` rendered as a point tag.

    The placeholders pair with ``points`` (a list of ``(x, y)`` per placeholder),
    ``inline_text`` (the tag's text) and ``alt_text`` (its ``alt``), the way
    Molmo rebuilds these answers. A reply whose counts disagree or whose points
    leave the image is ``None``.
    """
    parts = parsed.split(POINT_PLACEHOLDER)
    if not len(parts) - 1 == len(points) == len(inline_text) == len(alt_text):
        return None
    tags = []
    for group, text, alt in zip(points, inline_text, alt_text, strict=True):
        group_points = [(float(x), float(y)) for x, y in group]
        if not group_points or not _in_percent(group_points):
            return None
        tags.append(format_point_answer(group_points, text, absent_text="", alt=alt))
    return "".join(part + tag for part, tag in zip(parts, tags + [""], strict=True))


def pixmo_point_explanations(
    annotation_file: str | Path, image_root: str | Path, max_attempts: int = MAX_ATTEMPTS
) -> PixmoConversations:
    """PixMo point-explanations, replies rebuilt from the parsed response and its points.

    The raw ``response`` writes some multi-point tags in a second syntax, so the
    replies are rebuilt from ``parsed_response`` with the pointing syntax;
    turns whose points are malformed or outside the image are dropped.
    """

    def turns(row: dict[str, Any]) -> list[tuple[Any, Any]]:
        fields = (
            row.get("question") or [],
            row.get("parsed_response") or [],
            row.get("points") or [],
            row.get("inline_text") or [],
            row.get("alt_text") or [],
        )
        return [
            (user, explanation_answer(parsed, pts, inline, alt))
            for user, parsed, pts, inline, alt in zip(*fields, strict=False)
        ]

    return _conversations(
        image_root,
        ((row["image"], turns(row)) for row in _jsonl(annotation_file)),
        max_attempts,
    )


def pixmo_pointing(
    annotation_file: str | Path,
    image_root: str | Path,
    max_points: int = MAX_POINTS,
    max_attempts: int = MAX_ATTEMPTS,
) -> PixmoPointing:
    """PixMo-Points pointing, one example per labelled object kind."""
    columns: dict[str, list[Any]] = {"image": [], "label": [], "xs": [], "ys": []}
    for row in _jsonl(annotation_file):
        for label, points in zip(row["label"], row["points"], strict=False):
            columns["image"].append(row["image"])
            columns["label"].append(label)
            columns["xs"].append([float(p["x"]) for p in points])
            columns["ys"].append([float(p["y"]) for p in points])
    return PixmoPointing(image_root, columns, max_points, max_attempts)


def pixmo_counting(
    annotation_file: str | Path,
    image_root: str | Path,
    *,
    multi_label: bool,
    max_points: int | None = None,
    max_attempts: int = MAX_ATTEMPTS,
) -> PixmoCounting:
    """PixMo counting, keeping every point of a row unless ``max_points`` caps them.

    ``multi_label`` rows hold parallel ``label``, ``count`` and ``points`` lists
    in percent (PixMo-Points counting); otherwise a row is one label whose
    ``points`` are an ``{"x": [...], "y": [...]}`` mapping in pixels (PixMo-Count),
    converted to percent of the decoded image.
    """
    columns: dict[str, list[Any]] = {"image": [], "label": [], "count": [], "xs": [], "ys": []}
    for row in _jsonl(annotation_file):
        if multi_label:
            entries = [
                (label, count, [float(p["x"]) for p in pts], [float(p["y"]) for p in pts])
                for label, count, pts in zip(
                    row["label"], row["count"], row["points"], strict=False
                )
            ]
        else:
            pts = row["points"]
            entries = [
                (
                    row["label"],
                    row["count"],
                    [float(x) for x in pts["x"]],
                    [float(y) for y in pts["y"]],
                )
            ]
        for label, count, xs, ys in entries:
            columns["image"].append(row["image"])
            columns["label"].append(label)
            columns["count"].append(count)
            columns["xs"].append(xs)
            columns["ys"].append(ys)
    return PixmoCounting(
        image_root, columns, pixels=not multi_label, max_points=max_points, max_attempts=max_attempts
    )
