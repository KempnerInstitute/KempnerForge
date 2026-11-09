"""Reading past rows a reader cannot use."""

from __future__ import annotations

from collections.abc import Callable

from kempnerforge.metrics.logger import get_logger

logger = get_logger(__name__)

MAX_ATTEMPTS = 16


class UnusableRowError(Exception):
    """A row a reader cannot turn into a usable document."""


def check_max_attempts(max_attempts: int) -> int:
    """``max_attempts``, refused unless positive."""
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be positive, got {max_attempts}")
    return max_attempts


class SkipLog:
    """Counts the rows a reader passed over, logging the first."""

    def __init__(self, what: str = "row") -> None:
        self.what = what
        self.count = 0

    def __call__(self, index: int, error: Exception) -> None:
        if not self.count:
            logger.warning(
                f"skipping {self.what} {index}: {error}; later skips are counted, not logged"
            )
        self.count += 1


def first_usable[T](
    index: int,
    length: int,
    max_attempts: int,
    read: Callable[[int], T],
    on_skip: Callable[[int, Exception], None],
    next_row: Callable[[int, UnusableRowError], int] | None = None,
) -> T:
    """``read(index)``, moving on past rows whose read raises ``UnusableRowError``.

    After a skipped row the next one tried is ``next_row(row, error)`` (by default
    the following row), wrapping around at ``length``; after ``max_attempts`` skipped
    rows in a row it raises ``RuntimeError``. Any other error propagates.
    """
    row = index
    for _ in range(max_attempts):
        try:
            return read(row)
        except UnusableRowError as exc:
            on_skip(row, exc)
            row = (next_row(row, exc) if next_row is not None else row + 1) % length
    raise RuntimeError(f"{max_attempts} consecutive rows from index {index} cannot be used")
