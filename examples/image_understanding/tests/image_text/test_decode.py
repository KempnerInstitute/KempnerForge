"""Tests for image decoding and reading past unreadable rows."""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path

import pytest
from image_text.decode import UnreadableImageError, decode_rgb, first_readable
from PIL import Image


class TestDecodeRgb:
    def test_decodes_a_file(self, write_image: Callable) -> None:
        image = decode_rgb(write_image("a.png", size=(4, 3), mode="L"))
        assert image.mode == "RGB" and image.size == (4, 3)

    def test_decodes_bytes(self) -> None:
        buffer = io.BytesIO()
        Image.new("RGBA", (2, 5)).save(buffer, format="PNG")
        assert decode_rgb(buffer.getvalue()).size == (2, 5)

    def test_a_saved_error_page_is_unreadable(self, tmp_path: Path) -> None:
        stub = tmp_path / "stub"
        stub.write_text("404 Client Error: Not Found for url")
        with pytest.raises(UnreadableImageError, match="stub"):
            decode_rgb(stub)

    def test_garbage_bytes_are_unreadable(self) -> None:
        with pytest.raises(UnreadableImageError, match="4 bytes"):
            decode_rgb(b"\x00\x01\x02\x03")

    def test_truncated_data_is_unreadable(self) -> None:
        buffer = io.BytesIO()
        Image.new("RGB", (64, 64), (1, 2, 3)).save(buffer, format="PNG")
        with pytest.raises(UnreadableImageError):
            decode_rgb(buffer.getvalue()[: len(buffer.getvalue()) // 2])

    def test_a_missing_file_raises_its_own_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            decode_rgb(tmp_path / "absent.png")


class TestFirstReadable:
    def _read(self, unreadable: set[int]) -> Callable[[int], int]:
        def read(row: int) -> int:
            if row in unreadable:
                raise UnreadableImageError(f"row {row}")
            return row

        return read

    def test_returns_the_first_readable_row(self) -> None:
        skipped: list[int] = []
        row = first_readable(1, 5, 4, self._read({1, 2}), lambda i, _: skipped.append(i))
        assert row == 3 and skipped == [1, 2]

    def test_wraps_around(self) -> None:
        assert first_readable(4, 5, 3, self._read({4}), lambda *_: None) == 0

    def test_raises_after_max_attempts(self) -> None:
        with pytest.raises(RuntimeError, match="2 consecutive rows from index 0"):
            first_readable(0, 5, 2, self._read({0, 1}), lambda *_: None)

    def test_other_errors_propagate(self) -> None:
        def read(row: int) -> int:
            raise KeyError("bug")

        with pytest.raises(KeyError):
            first_readable(0, 5, 3, read, lambda *_: None)
