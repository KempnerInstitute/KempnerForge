"""Image decoding, and reading past rows whose image does not decode."""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path

from PIL import Image


class UnreadableImageError(Exception):
    """An image file or byte string that does not decode."""


def decode_rgb(source: str | Path | bytes) -> Image.Image:
    """Decode an image file or byte string as RGB."""
    try:
        with Image.open(io.BytesIO(source) if isinstance(source, bytes) else source) as image:
            return image.convert("RGB")
    except Exception as exc:
        what = f"{len(source)} bytes" if isinstance(source, bytes) else str(source)
        raise UnreadableImageError(f"{what}: {type(exc).__name__}: {exc}") from exc


def first_readable[T](
    index: int,
    length: int,
    max_attempts: int,
    read: Callable[[int], T],
    on_skip: Callable[[int, UnreadableImageError], None],
) -> T:
    """``read(index)``, moving on to the following rows while their image does not decode.

    Rows wrap around at ``length``; after ``max_attempts`` unreadable rows in a row
    it raises ``RuntimeError``.
    """
    for attempt in range(max_attempts):
        row = (index + attempt) % length
        try:
            return read(row)
        except UnreadableImageError as exc:
            on_skip(row, exc)
    raise RuntimeError(f"{max_attempts} consecutive rows from index {index} have no readable image")
