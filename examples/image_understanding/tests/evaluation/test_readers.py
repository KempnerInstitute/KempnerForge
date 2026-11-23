"""Tests for the readers of the evaluation-only splits."""

from __future__ import annotations

import io
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from evaluation.extraction import AnswerExtractor
from evaluation.readers import (
    MATHVISTA_COLUMNS,
    MathVistaRows,
    gsm8k,
    mathvista,
    mmlu,
    pixmo_count_eval,
    pixmo_points_eval,
)
from evaluation.scorers import MathVistaReference
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


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), (10, 200, 10)).save(buffer, format="PNG")
    return buffer.getvalue()


def _extract(question: str, reply: str) -> str:
    return reply


MATHVISTA_PROBLEMS: list[dict[str, Any]] = [
    {
        "pid": "1",
        "query": "Hint: pick an option.\nQuestion: Which fraction?\nChoices:\n(A) 3/11\n(B) 8/11",
        "answer": "8/11",
        "question_type": "multi_choice",
        "answer_type": "text",
        "choices": ["3/11", "8/11"],
        "precision": None,
    },
    {
        "pid": "2",
        "query": "Hint: answer with an integer.\nQuestion: How many cats?",
        "answer": "3",
        "question_type": "free_form",
        "answer_type": "integer",
        "choices": None,
        "precision": None,
    },
    {
        "pid": "3",
        "query": "Hint: two decimal places.\nQuestion: How long is it?",
        "answer": "1.25",
        "question_type": "free_form",
        "answer_type": "float",
        "choices": None,
        "precision": 2.0,
    },
]


def _changed(row: int, **fields: Any) -> list[dict[str, Any]]:
    """``MATHVISTA_PROBLEMS`` with ``fields`` of problem ``row`` changed."""
    problems = [dict(problem) for problem in MATHVISTA_PROBLEMS]
    problems[row].update(fields)
    return problems


class TestMathVista:
    def test_a_row_asks_its_query_of_its_image(self, write_mathvista: Callable[..., Path]) -> None:
        reader = MathVistaRows(write_mathvista(MATHVISTA_PROBLEMS), _extract)
        assert len(reader) == 3
        for row, problem in enumerate(MATHVISTA_PROBLEMS):
            document = reader[row]
            assert document.images[0].size == (8 + row, 6)
            assert document.images[0].mode == "RGB"
            assert document.turns[0].user == problem["query"]
            assert document.turns[0].answers == (problem["answer"],)

    def test_the_reference_is_the_problem_with_the_extractor(
        self, write_mathvista: Callable[..., Path]
    ) -> None:
        reader = MathVistaRows(write_mathvista(MATHVISTA_PROBLEMS), _extract)
        assert reader.reference(0) == MathVistaReference(
            answer="8/11",
            query=MATHVISTA_PROBLEMS[0]["query"],
            question_type="multi_choice",
            answer_type="text",
            extract=_extract,
            choices=("3/11", "8/11"),
        )
        assert (reader.reference(1).choices, reader.reference(1).precision) == ((), None)
        assert reader.reference(2).precision == 2
        assert isinstance(reader.reference(2).precision, int)

    @pytest.mark.parametrize("broken", [{"bytes": b"not an image", "path": "0.png"}, None])
    def test_a_row_whose_image_cannot_be_used_is_passed_over(
        self, write_mathvista: Callable[..., Path], broken: dict[str, Any] | None
    ) -> None:
        image = {"bytes": _png(), "path": "1.png"}
        path = write_mathvista(MATHVISTA_PROBLEMS, images=[broken, image, image])
        reader = MathVistaRows(path, _extract)
        assert reader.read_with_index(0)[0] == 1
        assert reader.skipped == 1

    @pytest.mark.parametrize("column", MATHVISTA_COLUMNS)
    def test_a_file_without_a_column_is_refused(
        self, write_mathvista: Callable[..., Path], column: str
    ) -> None:
        path = write_mathvista(MATHVISTA_PROBLEMS)
        pq.write_table(pq.read_table(path).drop_columns([column]), path)
        with pytest.raises(ValueError, match=re.escape(f"has no ['{column}'] column")):
            MathVistaRows(path, _extract)

    @pytest.mark.parametrize(
        ("problems", "error"),
        [
            (_changed(1, answer=""), "problem 2: answer must be a non-blank string"),
            (_changed(2, answer_type="fraction"), "problem 3: answer_type must be one of"),
            (_changed(2, precision=None), "problem 3: a float answer needs a precision"),
            (_changed(2, precision=float("nan")), "problem 3: "),
            (_changed(0, answer="1/2"), "problem 1: the answer '1/2' is not one of"),
        ],
    )
    def test_a_problem_that_could_never_be_scored_is_refused_by_its_pid(
        self, write_mathvista: Callable[..., Path], problems: list[dict[str, Any]], error: str
    ) -> None:
        path = write_mathvista(problems)
        with pytest.raises(ValueError, match=error):
            MathVistaRows(path, _extract)

    def test_the_factory_gives_each_row_an_extractor_on_the_device(
        self, write_mathvista: Callable[..., Path]
    ) -> None:
        reader = mathvista(write_mathvista(MATHVISTA_PROBLEMS), extractor_device="cpu")
        extract = reader.reference(0).extract
        assert isinstance(extract, AnswerExtractor) and extract.device == torch.device("cpu")
        assert all(reader.reference(row).extract is extract for row in range(len(reader)))

    def test_a_device_torch_cannot_read_is_refused_when_the_reader_is_built(
        self, write_mathvista: Callable[..., Path]
    ) -> None:
        with pytest.raises(ValueError, match="extractor_device"):
            mathvista(write_mathvista(MATHVISTA_PROBLEMS), extractor_device="gpu0")

    def test_a_task_file_reads_the_split_from_its_own_directory(
        self, write_mathvista: Callable[..., Path], tmp_path: Path
    ) -> None:
        write_mathvista(MATHVISTA_PROBLEMS, name="testmini.parquet")
        path = tmp_path / "tasks.toml"
        path.write_text(
            '[[tasks]]\nname = "mathvista-testmini"\nreader = "mathvista"\n'
            'scorer = "mathvista_accuracy"\nmax_new_tokens = 100\n[tasks.reader_args]\n'
            'annotation_file = "testmini.parquet"\nextractor_device = "cpu"\n'
        )
        (task,) = load_tasks(path)
        assert task.reader_args["annotation_file"] == str(tmp_path / "testmini.parquet")
        reader = build_reader(task)
        assert isinstance(reader, MathVistaRows) and len(reader) == 3
