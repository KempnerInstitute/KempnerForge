"""Tests for reading past unusable rows."""

from __future__ import annotations

import pickle

import pytest
from image_text import skip
from image_text.skip import (
    NoUsableRowError,
    SkipLog,
    UnusableRowError,
    check_max_attempts,
    first_usable,
)
from torch.utils.data import DataLoader, Dataset


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
        with pytest.raises(NoUsableRowError, match="2 consecutive rows from index 1"):
            first_usable(1, 4, 2, _reader({1, 2}), lambda *_: None)

    def test_a_give_up_is_a_runtime_error(self) -> None:
        assert issubclass(NoUsableRowError, RuntimeError)

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

        assert first_usable(0, 4, 2, read, lambda *_: None, next_row=lambda i, _: i + 3) == "ok"
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

    def test_the_count_survives_pickling(self) -> None:
        skips = SkipLog()
        skips(0, Unusable("bad"))
        copy = pickle.loads(pickle.dumps(skips))
        copy(1, Unusable("bad"))
        assert (skips.count, copy.count) == (1, 2)

    def test_forked_loader_workers_add_to_the_count(self) -> None:
        rows = _Rows()
        loader = DataLoader(rows, batch_size=None, num_workers=2, multiprocessing_context="fork")
        assert sorted(int(x) for x in loader) == [0, 0, 2, 2]
        assert rows.skips.count == 2


class _Rows(Dataset):
    def __init__(self) -> None:
        self.skips = SkipLog()

    def __len__(self) -> int:
        return 4

    def _read(self, i: int) -> int:
        if i % 2:
            raise Unusable(i)
        return i

    def __getitem__(self, i: int) -> int:
        return first_usable(i, 4, 2, self._read, self.skips)


@pytest.mark.parametrize("max_attempts", [0, -1])
def test_max_attempts_must_be_positive(max_attempts: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        check_max_attempts(max_attempts)
