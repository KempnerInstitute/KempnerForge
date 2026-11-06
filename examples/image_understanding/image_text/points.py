"""Molmo-style point answers: coordinates in percent of the image, as XML-like tags."""

from __future__ import annotations

from collections.abc import Sequence
from xml.sax.saxutils import escape


def _coordinate(value: float) -> str:
    return f"{min(100.0, max(0.0, value)):.1f}"


def format_point_answer(
    points: Sequence[tuple[float, float]],
    label: str,
    *,
    absent_text: str,
    max_points: int | None = None,
) -> str:
    """Render percent ``(x, y)`` points for ``label`` as a ``<point>`` or ``<points>`` tag.

    No points give ``absent_text``. Coordinates are clamped to ``[0, 100]`` with
    one decimal, and at most ``max_points`` of them are kept.
    """
    if not points:
        return absent_text
    if max_points is not None:
        points = points[:max_points]
    label_attr = escape(label, {'"': "&quot;"})
    label_text = escape(label)
    if len(points) == 1:
        x, y = points[0]
        return (
            f'<point x="{_coordinate(x)}" y="{_coordinate(y)}" '
            f'alt="{label_attr}">{label_text}</point>'
        )
    coords = " ".join(
        f'x{i}="{_coordinate(x)}" y{i}="{_coordinate(y)}"'
        for i, (x, y) in enumerate(points, start=1)
    )
    return f'<points {coords} alt="{label_attr}">{label_text}</points>'


def pixels_to_percent(
    points: Sequence[tuple[float, float]], width: int, height: int
) -> list[tuple[float, float]]:
    """Convert pixel coordinates to percent of an image of ``width`` by ``height``."""
    if width <= 0 or height <= 0:
        raise ValueError(f"image size must be positive, got {width}x{height}")
    return [(100.0 * x / width, 100.0 * y / height) for x, y in points]
