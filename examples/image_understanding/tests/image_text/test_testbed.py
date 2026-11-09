"""Tests for the single-image question-answer and caption readers."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from image_text.testbed import (
    ImageTextRows,
    chartqa,
    coco_captions,
    docvqa,
    textvqa,
    vqav2,
)


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value))
    return path


class TestLoaders:
    def test_coco_captions_asks_the_prompt(self, tmp_path: Path, write_image: Callable) -> None:
        write_image("train2014/a.jpg", size=(10, 4))
        ann = _write_json(
            tmp_path / "coco.json",
            [{"caption": "A red square.", "image": "train2014/a.jpg", "image_id": "a"}],
        )
        rows = coco_captions(ann, tmp_path, prompt="Describe it.")
        doc = rows[0]
        assert len(rows) == 1
        assert doc.kind == "qa"
        assert doc.turns[0].user == "Describe it."
        assert doc.turns[0].answers == ("A red square.",)
        assert doc.images[0].size == (10, 4)

    def test_vqav2_keeps_every_annotator_answer(
        self, tmp_path: Path, write_image: Callable
    ) -> None:
        write_image("train2014/b.jpg")
        answers = ["two", "2", "two"]
        ann = _write_json(
            tmp_path / "vqa.json",
            [{"question": "How many?", "answer": answers, "image": "train2014/b.jpg"}],
        )
        doc = vqav2(ann, tmp_path)[0]
        assert doc.turns[0].user == "How many?"
        assert doc.turns[0].answers == ("two", "2", "two")

    def test_textvqa_names_images_by_id(self, tmp_path: Path, write_image: Callable) -> None:
        write_image("train_images/abc123.jpg", size=(5, 5))
        ann = _write_json(
            tmp_path / "textvqa.json",
            {"data": [{"question": "What brand?", "answers": ["acme"], "image_id": "abc123"}]},
        )
        doc = textvqa(ann, tmp_path / "train_images")[0]
        assert doc.turns[0].answers == ("acme",)
        assert doc.images[0].size == (5, 5)

    def test_docvqa_reads_the_image_path(self, tmp_path: Path, write_image: Callable) -> None:
        write_image("documents/page_1.png")
        ann = _write_json(
            tmp_path / "docvqa.json",
            {"data": [{"question": "Date?", "answers": ["1999"], "image": "documents/page_1.png"}]},
        )
        doc = docvqa(ann, tmp_path)[0]
        assert doc.turns[0].user == "Date?"
        assert doc.turns[0].answers == ("1999",)

    def test_chartqa_label_becomes_a_string(self, tmp_path: Path, write_image: Callable) -> None:
        write_image("png/c.png")
        ann = _write_json(
            tmp_path / "chartqa.json", [{"imgname": "c.png", "query": "Peak?", "label": 42}]
        )
        doc = chartqa(ann, tmp_path / "png")[0]
        assert doc.turns[0].answers == ("42",)


class TestImageTextRows:
    @pytest.mark.parametrize("mode", ["L", "RGBA"])
    def test_images_are_converted_to_rgb(
        self, tmp_path: Path, write_image: Callable, mode: str
    ) -> None:
        write_image("x.png", mode=mode)
        rows = ImageTextRows(tmp_path, ["x.png"], ["Q"], [["A"]])
        assert rows[0].images[0].mode == "RGB"

    def test_empty_answers_and_answerless_rows_are_dropped(
        self, tmp_path: Path, write_image: Callable
    ) -> None:
        write_image("c.png")
        rows = ImageTextRows(
            tmp_path, ["a", "b", "c.png"], ["Q", "Q", "Q3"], [[], [" \n"], ["", "C"]]
        )
        assert len(rows) == 1
        assert (rows[0].turns[0].user, rows[0].turns[0].answers) == ("Q3", ("C",))

    def test_column_length_mismatch_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="column lengths differ"):
            ImageTextRows(tmp_path, ["a"], ["Q", "Q"], [["A"]])

    @pytest.mark.parametrize("index", [-1, 1])
    def test_out_of_range_index_raises(self, tmp_path: Path, index: int) -> None:
        rows = ImageTextRows(tmp_path, ["a"], ["Q"], [["A"]])
        with pytest.raises(IndexError):
            rows[index]

    def test_missing_image_raises(self, tmp_path: Path) -> None:
        rows = ImageTextRows(tmp_path, ["absent.jpg"], ["Q"], [["A"]])
        with pytest.raises(FileNotFoundError):
            rows[0]

    def test_an_undecodable_image_is_passed_over(
        self, tmp_path: Path, write_image: Callable
    ) -> None:
        (tmp_path / "bad.png").write_text("not an image")
        write_image("good.png")
        rows = ImageTextRows(tmp_path, ["bad.png", "good.png"], ["Q1", "Q2"], [["A1"], ["A2"]])
        assert rows[0].turns[0].user == "Q2"
        assert rows.skipped == 1
        with pytest.raises(RuntimeError, match="1 consecutive"):
            ImageTextRows(tmp_path, ["bad.png"], ["Q"], [["A"]], max_attempts=1)[0]
