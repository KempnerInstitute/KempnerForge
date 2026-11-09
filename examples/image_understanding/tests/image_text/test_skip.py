"""Tests for reading past unusable rows."""

from __future__ import annotations

import pytest
from image_text import skip
from image_text.skip import SkipLog, UnusableRowError, check_max_attempts, first_usable


class Unusable(UnusableRowError):
    pass


def _reader(bad: set[int]):
    def read(i: int) -> str:
        if i in bad:
            raise Unusable(f"row {i}")
        return f"row {i}"

    return read


class TestFirstUsable:
    def test_returns_the_first_usable_row(self) -> None:
        skips: list[int] = []
        read = _reader({0, 1})
        assert first_usable(0, 4, 3, read, lambda i, _: skips.append(i)) == "row 2"
        assert skips == [0, 1]

    def test_wraps_around(self) -> None:
        assert first_usable(3, 4, 2, _reader({3}), lambda *_: None) == "row 0"

    def test_raises_after_max_attempts(self) -> None:
        with pytest.raises(RuntimeError, match="2 consecutive rows from index 1"):
            first_usable(1, 4, 2, _reader({1, 2}), lambda *_: None)

    def test_other_errors_propagate(self) -> None:
        def read(i: int) -> str:
            raise KeyError(i)

        with pytest.raises(KeyError):
            first_usable(0, 4, 2, read, lambda *_: None)

    def test_next_row_chooses_where_to_continue(self) -> None:
        tried: list[int] = []

        def read(i: int) -> str:
            tried.append(i)
            if i < 3:
                raise Unusable(i)
            return "ok"

        assert first_usable(0, 4, 2, read, lambda *_: None, next_row=lambda i: i + 3) == "ok"
        assert tried == [0, 3]


class TestSkipLog:
    def test_logs_the_first_skip_and_counts_the_rest(self, monkeypatch: pytest.MonkeyPatch) -> None:
        warnings: list[str] = []
        monkeypatch.setattr(skip.logger, "warning", warnings.append)
        skips = SkipLog("document")
        skips(3, Unusable("bad"))
        skips(4, Unusable("bad"))
        assert skips.count == 2
        assert len(warnings) == 1 and warnings[0].startswith("skipping document 3: bad")


@pytest.mark.parametrize("max_attempts", [0, -1])
def test_max_attempts_must_be_positive(max_attempts: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        check_max_attempts(max_attempts)
