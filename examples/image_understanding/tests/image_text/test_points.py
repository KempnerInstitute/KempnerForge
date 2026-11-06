"""Tests for the Molmo-style point answer format."""

from __future__ import annotations

import pytest
from image_text.points import format_point_answer, pixels_to_percent


class TestFormatPointAnswer:
    def test_no_points_gives_the_absent_text(self) -> None:
        assert format_point_answer([], "cat", absent_text="none") == "none"

    def test_one_point_is_a_point_tag(self) -> None:
        assert format_point_answer([(12.34, 56.78)], "cat", absent_text="") == (
            '<point x="12.3" y="56.8" alt="cat">cat</point>'
        )

    def test_many_points_are_numbered(self) -> None:
        assert format_point_answer([(1, 2), (3, 4)], "dog", absent_text="") == (
            '<points x1="1.0" y1="2.0" x2="3.0" y2="4.0" alt="dog">dog</points>'
        )

    def test_coordinates_are_clamped_to_percent(self) -> None:
        assert format_point_answer([(-0.2, 140.0)], "cat", absent_text="") == (
            '<point x="0.0" y="100.0" alt="cat">cat</point>'
        )

    def test_max_points_keeps_the_first(self) -> None:
        answer = format_point_answer([(1, 1), (2, 2), (3, 3)], "x", absent_text="", max_points=2)
        assert 'x2="2.0"' in answer and "x3" not in answer

    def test_label_is_escaped(self) -> None:
        answer = format_point_answer([(1, 1)], 'a "b" & <c>', absent_text="")
        assert 'alt="a &quot;b&quot; &amp; &lt;c&gt;"' in answer
        assert '>a "b" &amp; &lt;c&gt;</point>' in answer


class TestPixelsToPercent:
    def test_scales_by_the_image_size(self) -> None:
        assert pixels_to_percent([(100, 25), (0, 100)], 200, 100) == [(50.0, 25.0), (0.0, 100.0)]

    def test_refuses_an_empty_image(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            pixels_to_percent([(1, 1)], 0, 10)
