"""Tests for the PixMo readers."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from image_text.pixmo import (
    PixmoRows,
    pixmo_cap_qa,
    pixmo_captions,
    pixmo_counting,
    pixmo_point_explanations,
    pixmo_pointing,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def _write_stub(tmp_path: Path, relative: str) -> None:
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("404 Client Error: Not Found for url")


class TestCaptionsAndConversations:
    def test_captions_are_asked_with_the_prompt(
        self, tmp_path: Path, write_image: Callable
    ) -> None:
        write_image("pixmo_images/aa")
        ann = _write_jsonl(
            tmp_path / "cap.jsonl",
            [{"image": "pixmo_images/aa", "caption": "A long caption.", "transcripts": ["t"]}],
        )
        doc = pixmo_captions(ann, tmp_path, prompt="Describe.")[0]
        assert doc.kind == "qa"
        assert doc.turns[0].user == "Describe."
        assert doc.turns[0].answers == ("A long caption.",)

    def test_cap_qa_takes_the_last_pair_of_each_message(
        self, tmp_path: Path, write_image: Callable
    ) -> None:
        write_image("pixmo_images/bb")
        messages = [["Q1", "A1"], ["Q1", "A1", "Q2", "A2"], ["Q3", ""], "junk", ["only"]]
        ann = _write_jsonl(
            tmp_path / "capqa.jsonl", [{"image": "pixmo_images/bb", "messages": messages}]
        )
        doc = pixmo_cap_qa(ann, tmp_path)[0]
        assert doc.kind == "conversation"
        assert [(t.user, t.answers) for t in doc.turns] == [("Q1", ("A1",)), ("Q2", ("A2",))]

    def test_rows_without_turns_are_dropped(self, tmp_path: Path) -> None:
        ann = _write_jsonl(
            tmp_path / "capqa.jsonl",
            [{"image": "pixmo_images/cc", "messages": [["", "A"]]}, {"image": "x", "messages": []}],
        )
        assert len(pixmo_cap_qa(ann, tmp_path)) == 0

    def test_point_explanations_pair_questions_and_responses(
        self, tmp_path: Path, write_image: Callable
    ) -> None:
        write_image("pixmo_images/dd")
        ann = _write_jsonl(
            tmp_path / "expl.jsonl",
            [
                {
                    "image": "pixmo_images/dd",
                    "question": ["Where?", "Why?", " "],
                    "response": ['<point x="1.0" y="2.0" alt="a">a</point> here', "Because.", "x"],
                }
            ],
        )
        doc = pixmo_point_explanations(ann, tmp_path)[0]
        assert [t.user for t in doc.turns] == ["Where?", "Why?"]
        assert doc.turns[0].answers == ('<point x="1.0" y="2.0" alt="a">a</point> here',)


class TestPointingAndCounting:
    def test_pointing_unrolls_one_example_per_label(
        self, tmp_path: Path, write_image: Callable
    ) -> None:
        write_image("pixmo_images/ee")
        ann = _write_jsonl(
            tmp_path / "pointing.jsonl",
            [
                {
                    "image": "pixmo_images/ee",
                    "label": ["cat", "dog"],
                    "points": [[{"x": 10.0, "y": 20.0}], []],
                }
            ],
        )
        reader = pixmo_pointing(ann, tmp_path)
        assert len(reader) == 2
        assert reader[0].turns[0].user == "Point to the cat."
        assert reader[0].turns[0].answers == ('<point x="10.0" y="20.0" alt="cat">cat</point>',)
        assert reader[1].turns[0].answers == ("There are no dog in the image.",)

    def test_multi_label_counting_keeps_percent_points(
        self, tmp_path: Path, write_image: Callable
    ) -> None:
        write_image("pixmo_images/ff", size=(400, 200))
        ann = _write_jsonl(
            tmp_path / "points_counting.jsonl",
            [
                {
                    "image": "pixmo_images/ff",
                    "label": ["cup", "pen"],
                    "count": [2, 0],
                    "points": [[{"x": 10.0, "y": 20.0}, {"x": 30.0, "y": 40.0}], []],
                }
            ],
        )
        reader = pixmo_counting(ann, tmp_path, multi_label=True, coordinates="percent")
        assert reader[0].turns[0].user == "How many cup are in the image?"
        assert reader[0].turns[0].answers == (
            '<points x1="10.0" y1="20.0" x2="30.0" y2="40.0" alt="cup">cup</points>\n'
            "Counting the cup, I count 2.",
        )
        assert reader[1].turns[0].answers == ("I count 0 pen in the image.",)

    def test_pixel_points_are_converted_to_percent(
        self, tmp_path: Path, write_image: Callable
    ) -> None:
        write_image("pixmo_images/gg", size=(400, 200))
        ann = _write_jsonl(
            tmp_path / "count.jsonl",
            [
                {
                    "image": "pixmo_images/gg",
                    "label": "bird",
                    "count": 2,
                    "points": {"x": [200.0, 400.0], "y": [50.0, 200.0]},
                }
            ],
        )
        doc = pixmo_counting(ann, tmp_path, multi_label=False, coordinates="pixels")[0]
        assert doc.turns[0].answers == (
            '<points x1="50.0" y1="25.0" x2="100.0" y2="100.0" alt="bird">bird</points>\n'
            "Counting the bird, I count 2.",
        )

    def test_missing_count_reads_as_zero(self, tmp_path: Path, write_image: Callable) -> None:
        write_image("pixmo_images/hh")
        ann = _write_jsonl(
            tmp_path / "count.jsonl",
            [
                {
                    "image": "pixmo_images/hh",
                    "label": "ant",
                    "count": None,
                    "points": {"x": [], "y": []},
                }
            ],
        )
        doc = pixmo_counting(ann, tmp_path, multi_label=False, coordinates="pixels")[0]
        assert doc.turns[0].answers == ("I count 0 ant in the image.",)

    def test_unknown_coordinate_unit_is_refused(self, tmp_path: Path) -> None:
        ann = _write_jsonl(tmp_path / "count.jsonl", [])
        with pytest.raises(ValueError, match="'percent' or 'pixels'"):
            pixmo_counting(ann, tmp_path, multi_label=False, coordinates="inches")


class TestUnreadableImages:
    def _captions(self, tmp_path: Path, names: list[str]) -> Path:
        return _write_jsonl(
            tmp_path / "cap.jsonl",
            [{"image": f"pixmo_images/{n}", "caption": f"caption {n}"} for n in names],
        )

    def test_a_stub_steps_to_the_next_row(self, tmp_path: Path, write_image: Callable) -> None:
        _write_stub(tmp_path, "pixmo_images/s1")
        write_image("pixmo_images/ok")
        reader = pixmo_captions(self._captions(tmp_path, ["s1", "ok"]), tmp_path, prompt="P")
        assert reader[0].turns[0].answers == ("caption ok",)
        assert reader.skipped == 1

    def test_the_step_wraps_to_the_first_row(self, tmp_path: Path, write_image: Callable) -> None:
        write_image("pixmo_images/ok")
        _write_stub(tmp_path, "pixmo_images/s1")
        reader = pixmo_captions(self._captions(tmp_path, ["ok", "s1"]), tmp_path, prompt="P")
        assert reader[1].turns[0].answers == ("caption ok",)

    def test_gives_up_after_max_attempts(self, tmp_path: Path) -> None:
        for name in ("s1", "s2", "s3"):
            _write_stub(tmp_path, f"pixmo_images/{name}")
        reader = pixmo_captions(self._captions(tmp_path, ["s1", "s2", "s3"]), tmp_path, prompt="P")
        reader.max_attempts = 3
        with pytest.raises(RuntimeError, match="3 consecutive rows"):
            reader[0]

    def test_rows_need_an_image_column(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="'image' column"):
            PixmoRows(tmp_path, {"caption": ["x"]})

    def test_out_of_range_index_raises(self, tmp_path: Path) -> None:
        reader = pixmo_captions(self._captions(tmp_path, ["a"]), tmp_path, prompt="P")
        with pytest.raises(IndexError):
            reader[1]
