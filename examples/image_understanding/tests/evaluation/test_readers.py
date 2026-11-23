"""Tests for the readers of the evaluation-only splits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from evaluation.readers import gsm8k, mmlu, pixmo_count_eval, pixmo_points_eval
from evaluation.tasks import build_reader, load_tasks
from image_text.testbed import aokvqa
from PIL import Image


def _image(path: Path, color: tuple[int, int, int] = (200, 10, 10)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), color).save(path, format="PNG")


@pytest.fixture
def points_file(tmp_path: Path) -> Path:
    _image(tmp_path / "pixmo" / "pixmo_images" / "a")
    _image(tmp_path / "pixmo" / "pixmo_images" / "b")
    mask = np.zeros((2, 3, 4), dtype=bool)
    mask[0, 0, 0] = mask[1, 2, 3] = True
    rows = [
        {
            "label": 'the "big" cat',
            "points": [{"x": 0.0, "y": 0.0}, {"x": 100.0, "y": 100.0}],
            "masks": mask.tolist(),
            "image": "pixmo_images/a",
        },
        {"label": "dog", "points": [], "masks": [], "image": "pixmo_images/b"},
    ]
    path = tmp_path / "base.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


class TestPixmoPointsEval:
    def test_rows_ask_to_point_and_keep_the_annotated_points(
        self, points_file: Path, tmp_path: Path
    ) -> None:
        reader = pixmo_points_eval(points_file, tmp_path / "pixmo")
        assert len(reader) == 2
        first, second = reader[0], reader[1]
        assert first.turns[0].user == 'Point to the the "big" cat.'
        assert first.turns[0].answers[0].startswith('<points x1="0.0" y1="0.0" x2="100.0"')
        assert first.images[0].size == (8, 6)
        assert second.turns[0].answers == ("none",)

    def test_the_reference_is_the_rows_masks(self, points_file: Path, tmp_path: Path) -> None:
        reader = pixmo_points_eval(points_file, tmp_path / "pixmo")
        masks = reader.reference(0)
        assert masks.shape == (2, 3, 4) and masks.dtype == bool
        assert masks[0, 0, 0] and masks[1, 2, 3] and masks.sum() == 2
        assert reader.reference(1).size == 0

    def test_an_image_that_does_not_decode_is_passed_over(
        self, points_file: Path, tmp_path: Path
    ) -> None:
        (tmp_path / "pixmo" / "pixmo_images" / "a").write_bytes(b"not an image")
        reader = pixmo_points_eval(points_file, tmp_path / "pixmo")
        assert reader.read_with_index(0)[0] == 1
        assert reader.skipped == 1


def test_counting_rows_ask_how_many(tmp_path: Path) -> None:
    _image(tmp_path / "pixmo" / "pixmo_images" / "a")
    path = tmp_path / "test.jsonl"
    path.write_text(json.dumps({"image": "pixmo_images/a", "label": "people", "count": 5}) + "\n")
    (document,) = list(pixmo_count_eval(path, tmp_path / "pixmo"))
    assert document.turns[0].user == "How many people are in the image?"
    assert document.turns[0].answers == ("5",)


def test_aokvqa_rows_keep_every_direct_answer(tmp_path: Path) -> None:
    _image(tmp_path / "coco" / "val2014" / "x.jpg")
    path = tmp_path / "aokvqa.json"
    row = {"image": "val2014/x.jpg", "question": "What is it?", "direct_answers": ["cat", "dog"]}
    path.write_text(json.dumps([row]))
    (document,) = list(aokvqa(path, tmp_path / "coco"))
    assert document.turns[0].user == "What is it?"
    assert document.turns[0].answers == ("cat", "dog")


def test_gsm8k_rows_are_text_answered_by_the_final_number(tmp_path: Path) -> None:
    path = tmp_path / "test.parquet"
    pq.write_table(
        pa.table({"question": ["How many?"], "answer": ["Work it out.\n#### 1,234"]}), path
    )
    (document,) = list(gsm8k(path))
    assert document.images == ()
    assert (document.turns[0].user, document.turns[0].answers) == ("How many?", ("1234",))


def test_mmlu_rows_letter_their_options(tmp_path: Path) -> None:
    path = tmp_path / "test.parquet"
    table = {"question": ["Which?"], "choices": [["w", "x", "y", "z"]], "answer": [2]}
    pq.write_table(pa.table(table), path)
    (document,) = list(mmlu(path))
    assert document.images == ()
    assert document.turns[0].user == (
        "Which?\nA. w\nB. x\nC. y\nD. z\nAnswer with the letter of the correct option."
    )
    assert document.turns[0].answers == ("C",)


def test_the_board_tasks_load_from_a_task_file(points_file: Path, tmp_path: Path) -> None:
    parquet = tmp_path / "gsm8k.parquet"
    pq.write_table(pa.table({"question": ["How many?"], "answer": ["#### 3"]}), parquet)
    path = tmp_path / "tasks.toml"
    path.write_text(
        '[[tasks]]\nname = "pixmo-pointing"\nreader = "pixmo_points_eval"\n'
        'scorer = "points_in_masks"\nmax_new_tokens = 75\n[tasks.reader_args]\n'
        f'annotation_file = "{points_file}"\nimage_root = "{tmp_path / "pixmo"}"\n'
        '[[tasks]]\nname = "gsm8k"\nreader = "gsm8k"\nscorer = "final_number"\n'
        f'max_new_tokens = 256\n[tasks.reader_args]\nannotation_file = "{parquet}"\n'
    )
    pointing, arithmetic = load_tasks(path)
    assert len(build_reader(pointing)) == 2
    assert len(build_reader(arithmetic)) == 1
