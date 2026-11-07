"""Image decoding, and reading past rows whose image does not decode."""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path

from PIL import Image, ImageOps

DECODE_ERRORS = (OSError, ValueError, SyntaxError, Image.DecompressionBombError)


class UnreadableImageError(Exception):
    """An image file or byte string that does not decode."""


def decode_rgb(source: str | Path | bytes) -> Image.Image:
    """Decode an image file or byte string as RGB, turned upright by its EXIF orientation.

    Data that does not decode raises ``UnreadableImageError``; a file the system
    cannot open (missing, unreadable) raises its own ``OSError``, which carries an
    errno where a decoding failure does not. Any other error propagates.
    """
    try:
        with Image.open(io.BytesIO(source) if isinstance(source, bytes) else source) as image:
            upright = ImageOps.exif_transpose(image)
            return upright.convert("RGB")  # type: ignore[union-attr]
    except DECODE_ERRORS as exc:
        if isinstance(exc, OSError) and exc.errno is not None:
            raise
        what = f"{len(source)} bytes" if isinstance(source, bytes) else str(source)
        raise UnreadableImageError(f"{what}: {type(exc).__name__}: {exc}") from exc


def first_readable[T](
    index: int,
    length: int,
    max_attempts: int,
    read: Callable[[int], T],
    on_skip: Callable[[int, UnreadableImageError], None],
    next_row: Callable[[int], int] | None = None,
) -> T:
    """``read(index)``, moving on past rows whose image does not decode.

    After an unreadable row the next one tried is ``next_row(row)`` (by default
    the following row), wrapping around at ``length``; after ``max_attempts``
    unreadable rows in a row it raises ``RuntimeError``.
    """
    row = index
    for _ in range(max_attempts):
        try:
            return read(row)
        except UnreadableImageError as exc:
            on_skip(row, exc)
            row = (next_row(row) if next_row is not None else row + 1) % length
    raise RuntimeError(f"{max_attempts} consecutive rows from index {index} have no readable image")
