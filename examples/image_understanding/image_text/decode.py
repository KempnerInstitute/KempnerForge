"""Image decoding."""

from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

from PIL import Image, ImageOps

from .points import pixels_to_percent

DECODE_ERRORS = (OSError, ValueError, SyntaxError, Image.DecompressionBombError)
EXIF_ORIENTATION = 0x0112


class UnreadableImageError(Exception):
    """An image file or byte string that does not decode."""


def upright_points(
    points: Sequence[tuple[float, float]], orientation: int, width: int, height: int
) -> list[tuple[float, float]]:
    """Pixel points of a stored ``width`` by ``height`` grid, moved as the orientation is undone."""
    moves = {
        2: lambda x, y: (width - x, y),
        3: lambda x, y: (width - x, height - y),
        4: lambda x, y: (x, height - y),
        5: lambda x, y: (y, x),
        6: lambda x, y: (height - y, x),
        7: lambda x, y: (height - y, width - x),
        8: lambda x, y: (y, width - x),
    }
    move = moves.get(orientation)
    return [move(x, y) for x, y in points] if move else list(points)


class Decoded(NamedTuple):
    """An image decoded upright, with the orientation and size of its stored pixel grid."""

    image: Image.Image
    orientation: int
    stored_size: tuple[int, int]

    def percent(self, points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
        """Pixel points of the stored grid as percent of the upright image."""
        moved = upright_points(points, self.orientation, *self.stored_size)
        return pixels_to_percent(moved, *self.image.size)


def decode_upright(source: str | Path | bytes) -> Decoded:
    """Decode an image file or byte string as RGB, turned upright by its EXIF orientation.

    Data that does not decode raises ``UnreadableImageError``; a file the system
    cannot open (missing, unreadable) raises its own ``OSError``, which carries an
    errno where a decoding failure does not. Any other error propagates.
    """
    try:
        with Image.open(io.BytesIO(source) if isinstance(source, bytes) else source) as image:
            orientation = image.getexif().get(EXIF_ORIENTATION, 1)
            stored_size = image.size
            upright = ImageOps.exif_transpose(image)
            return Decoded(upright.convert("RGB"), orientation, stored_size)  # type: ignore[union-attr]
    except DECODE_ERRORS as exc:
        if isinstance(exc, OSError) and exc.errno is not None:
            raise
        what = f"{len(source)} bytes" if isinstance(source, bytes) else str(source)
        raise UnreadableImageError(f"{what}: {type(exc).__name__}: {exc}") from exc


def decode_rgb(source: str | Path | bytes) -> Image.Image:
    """The upright RGB image of ``decode_upright``."""
    return decode_upright(source).image
