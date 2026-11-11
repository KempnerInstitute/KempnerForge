"""Reading past rows a reader cannot use."""

from __future__ import annotations

import multiprocessing
from collections.abc import Callable
from typing import Any

from kempnerforge.metrics.logger import get_logger

logger = get_logger(__name__)

MAX_ATTEMPTS = 16


class UnusableRowError(Exception):
    """A row a reader cannot turn into a usable document."""


class NoUsableRowError(RuntimeError):
    """A reader gave up after a run of rows it could not use."""


def check_max_attempts(max_attempts: int) -> int:
    """``max_attempts``, refused unless positive."""
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be positive, got {max_attempts}")
    return max_attempts


class SkipLog:
    """Counts the rows a reader passed over, logging the first in each process.

    The count lives in shared memory, so loader workers forked from the process
    that built the reader add to the count that process reads; a copy sent to
    another process by pickling starts from the count it carried.
    """

    def __init__(self, what: str = "row", count: int = 0) -> None:
        self.what = what
        self._count = multiprocessing.Value("q", count)
        self._logged = False

    @property
    def count(self) -> int:
        return self._count.value

    def __call__(self, index: int, error: Exception) -> None:
        if not self._logged:
            logger.warning(
                f"skipping {self.what} {index}: {error}; later skips are counted, not logged"
            )
            self._logged = True
        with self._count.get_lock():
            self._count.value += 1

    def __getstate__(self) -> dict[str, Any]:
        return {"what": self.what, "count": self.count}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(state["what"], state["count"])


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
    rows in a row it raises ``NoUsableRowError``. Any other error propagates.
    """
    row = index
    for _ in range(max_attempts):
        try:
            return read(row)
        except UnusableRowError as exc:
            on_skip(row, exc)
            row = (next_row(row, exc) if next_row is not None else row + 1) % length
    raise NoUsableRowError(f"{max_attempts} consecutive rows from index {index} cannot be used")
