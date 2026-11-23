"""Tests for the RealWorldQA reader and the evaluation of its task."""

from __future__ import annotations

import io
import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from evaluation.evaluator import UNREADABLE_ROW, evaluate_checkpoint
from evaluation.realworldqa import RealWorldQARows, realworldqa
from evaluation.scorers import trimmed_match
from evaluation.tasks import READERS, build_reader, load_tasks
from image_text.skip import NoUsableRowError
from PIL import Image

from kempnerforge.config.job import JobConfig
from kempnerforge.config.loader import load_config

IMAGE = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
LETTER = "Please answer directly with only the letter of the correct option and nothing else."
WORD = "Please answer directly with a single word or number."
Row = tuple[Any, Any, Any]


def _encoded(size: tuple[int, int] = (8, 6), image_format: str = "PNG", mode: str = "RGB") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, 90 if mode == "L" else (200, 30, 10, 255)[: len(mode)]).save(
        buffer, format=image_format
    )
    return buffer.getvalue()


def _image(encoded: bytes | None) -> dict[str, Any]:
    return {"bytes": encoded, "path": ""}


def _row(question: Any, answer: Any = "A", image: Any = None) -> Row:
    return (_image(_encoded()) if image is None else image), question, answer


def _write(path: Path, rows: Sequence[Row]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    images, questions, answers = zip(*rows, strict=True)
    table = pa.table(
        {
            "image": pa.array(images, type=IMAGE),
            "question": pa.array(questions, type=pa.string()),
            "answer": pa.array(answers, type=pa.string()),
        }
    )
    pq.write_table(table, path)
    return path


def _publish(root: Path, *shards: Sequence[Row]) -> Path:
    """Writes ``shards`` where the Hub puts the test split's shards, the last one first."""
    for index in reversed(range(len(shards))):
        _write(root / "data" / f"test-{index:05d}-of-{len(shards):05d}.parquet", shards[index])
    return root


def _questions(reader: RealWorldQARows) -> list[str]:
    return [reader[index].turns[0].user for index in range(len(reader))]


class TestPublishedLayout:
    def test_the_shards_are_read_in_shard_order(self, tmp_path: Path) -> None:
        root = _publish(tmp_path, [_row("Q1"), _row("Q2")], [_row("Q3")])
        assert _questions(realworldqa(str(root))) == ["Q1", "Q2", "Q3"]

    def test_a_directory_without_a_test_shard_is_refused(self, tmp_path: Path) -> None:
        _write(tmp_path / "data" / "test.parquet", [_row("Q")])
        _write(tmp_path / "data" / "test-notes.parquet", [_row("Q")])
        _write(tmp_path / "data" / "validation-00000-of-00001.parquet", [_row("Q")])
        with pytest.raises(FileNotFoundError, match="holds no shard of the test split"):
            realworldqa(tmp_path)

    @pytest.mark.parametrize(
        ("names", "message"),
        [
            (
                ["test-00000-of-00003", "test-00002-of-00003"],
                "has 3 shards, numbered from 0; found [0, 2]",
            ),
            (["test-00001-of-00002"], "has 2 shards, numbered from 0; found [1]"),
            (
                ["test-0-of-2", "test-00000-of-00002", "test-00001-of-00002"],
                "has 2 shards, numbered from 0; found [0, 0, 1]",
            ),
        ],
    )
    def test_a_missing_or_repeated_shard_is_refused(
        self, tmp_path: Path, names: list[str], message: str
    ) -> None:
        for name in names:
            _write(tmp_path / "data" / f"{name}.parquet", [_row("Q")])
        with pytest.raises(ValueError, match=re.escape(message)):
            realworldqa(tmp_path)

    def test_shards_that_disagree_on_their_count_are_refused(self, tmp_path: Path) -> None:
        _write(tmp_path / "data" / "test-00000-of-00002.parquet", [_row("Q")])
        _write(tmp_path / "data" / "test-00001-of-00003.parquet", [_row("Q")])
        with pytest.raises(ValueError, match=re.escape("disagree on their count: [2, 3]")):
            realworldqa(tmp_path)


class TestRows:
    def test_the_question_is_the_prompt_as_written_and_the_answer_its_one_reference(
        self, tmp_path: Path
    ) -> None:
        question = f" Which way is the car turning?\nA. Left\nB. Right\n{LETTER}\n"
        image = _image(_encoded((10, 4), image_format="WEBP"))
        document = realworldqa(_publish(tmp_path, [_row(question, " B\n", image)]))[0]
        assert document.kind == "qa"
        assert len(document.turns) == 1
        assert document.turns[0].user == question
        assert document.turns[0].answers == ("B",)
        assert len(document.images) == 1
        assert (document.images[0].size, document.images[0].mode) == ((10, 4), "RGB")

    @pytest.mark.parametrize("mode", ["L", "RGBA"])
    def test_images_are_read_as_rgb(self, tmp_path: Path, mode: str) -> None:
        reader = realworldqa(_publish(tmp_path, [_row("Q", image=_image(_encoded(mode=mode)))]))
        assert reader[0].images[0].mode == "RGB"

    def test_rows_with_a_blank_question_or_answer_are_dropped(self, tmp_path: Path) -> None:
        rows = [
            _row("Q1"),
            _row(" \n"),
            _row(None),
            _row("Q4", ""),
            _row("Q5", " "),
            _row("Q6", None),
            _row("Q7", " Yes "),
        ]
        reader = realworldqa(_publish(tmp_path, rows))
        assert [(reader[i].turns[0].user, reader[i].turns[0].answers) for i in range(2)] == [
            ("Q1", ("A",)),
            ("Q7", ("Yes",)),
        ]
        assert len(reader) == 2

    @pytest.mark.parametrize(
        "image",
        [
            pytest.param(_image(b"not an image"), id="undecodable"),
            pytest.param(_image(None), id="no-bytes"),
            pytest.param(None, id="null"),
        ],
    )
    def test_a_row_whose_image_cannot_be_read_is_passed_over(
        self, tmp_path: Path, image: Any
    ) -> None:
        root = tmp_path / "data" / "test-00000-of-00001.parquet"
        _write(root, [(image, "Q1", "A"), _row("Q2")])
        reader = realworldqa(tmp_path)
        row, document = reader.read_with_index(0)
        assert (row, document.turns[0].user, reader.skipped) == (1, "Q2", 1)

    def test_a_run_of_unreadable_images_gives_up(self, tmp_path: Path) -> None:
        bad = _image(b"not an image")
        path = _write(tmp_path / "rows.parquet", [_row("Q1", image=bad), _row("Q2", image=bad)])
        with pytest.raises(NoUsableRowError, match="2 consecutive rows"):
            RealWorldQARows([path], max_attempts=2)[0]

    @pytest.mark.parametrize("index", [-1, 1])
    def test_an_index_out_of_range_raises(self, tmp_path: Path, index: int) -> None:
        with pytest.raises(IndexError, match="out of range for 1 rows"):
            RealWorldQARows([_write(tmp_path / "rows.parquet", [_row("Q")])])[index]

    def test_max_attempts_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="max_attempts must be positive"):
            RealWorldQARows([], max_attempts=0)


class TestFiles:
    def test_files_are_read_in_the_order_given(self, tmp_path: Path) -> None:
        first = _write(tmp_path / "a.parquet", [_row("Q1")])
        second = _write(tmp_path / "b.parquet", [_row("Q2"), _row("Q3")])
        assert _questions(RealWorldQARows([second, first])) == ["Q2", "Q3", "Q1"]

    @pytest.mark.parametrize(
        "columns",
        [
            pytest.param({"image": None}, id="no-image"),
            pytest.param({"answer": None}, id="no-answer"),
            pytest.param({"image": pa.array([b"x"])}, id="image-not-a-struct"),
            pytest.param(
                {"image": pa.array([{"path": "x"}], type=pa.struct([("path", pa.string())]))},
                id="image-without-bytes",
            ),
            pytest.param(
                {"image": pa.array([{"bytes": "x"}], type=pa.struct([("bytes", pa.string())]))},
                id="image-bytes-as-text",
            ),
            pytest.param({"answer": pa.array([3])}, id="answer-not-a-string"),
            pytest.param({"question": pa.array([b"Q"])}, id="question-not-a-string"),
        ],
    )
    def test_a_file_without_the_columns_and_their_types_is_refused(
        self, tmp_path: Path, columns: dict[str, Any]
    ) -> None:
        table = {
            "image": pa.array([_image(_encoded())], type=IMAGE),
            "question": pa.array(["Q"]),
            "answer": pa.array(["A"]),
        }
        table.update(columns)
        path = tmp_path / "rows.parquet"
        pq.write_table(pa.table({k: v for k, v in table.items() if v is not None}), path)
        with pytest.raises(ValueError, match=re.escape(f"{path}: a RealWorldQA file has ")):
            RealWorldQARows([path])

    def test_large_types_and_other_columns_are_read(self, tmp_path: Path) -> None:
        image = pa.struct([("bytes", pa.large_binary()), ("path", pa.string())])
        table = pa.table(
            {
                "id": [7],
                "image": pa.array([_image(_encoded())], type=image),
                "question": pa.array(["Q"], type=pa.large_string()),
                "answer": pa.array(["A"], type=pa.large_string()),
            }
        )
        pq.write_table(table, tmp_path / "rows.parquet")
        document = RealWorldQARows([tmp_path / "rows.parquet"])[0]
        assert (document.turns[0].user, document.turns[0].answers) == ("Q", ("A",))


def _task_file(path: Path, dataset_dir: Path | str | None, max_new_tokens: int = 10) -> Path:
    text = (
        '[[tasks]]\nname = "realworldqa"\nreader = "realworldqa"\nscorer = "trimmed_match"\n'
        f"max_new_tokens = {max_new_tokens}\n"
    )
    if dataset_dir is not None:
        text += f'\n[tasks.reader_args]\ndataset_dir = "{dataset_dir}"\n'
    path.write_text(text)
    return path


class TestTask:
    def test_the_reader_is_registered(self) -> None:
        assert READERS["realworldqa"] is realworldqa

    def test_a_relative_dataset_dir_is_taken_from_the_task_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _publish(tmp_path / "run" / "realworldqa", [_row("Q1", "Yes")])
        path = _task_file(tmp_path / "run" / "tasks.toml", "realworldqa")
        (tmp_path / "elsewhere").mkdir()
        monkeypatch.chdir(tmp_path / "elsewhere")
        task = load_tasks(path)[0]
        assert task.reader_args == {"dataset_dir": str(tmp_path.resolve() / "run" / "realworldqa")}
        assert build_reader(task)[0].turns[0].answers == ("Yes",)

    def test_the_task_needs_its_dataset_dir(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=re.escape("needs ['dataset_dir'] in reader_args")):
            load_tasks(_task_file(tmp_path / "tasks.toml", None))


@pytest.fixture
def run(
    write_config: Callable[..., Path], save_checkpoint: Callable[..., Path]
) -> tuple[JobConfig, Path]:
    config = load_config(write_config(), cli_args=[])
    return config, save_checkpoint(config)


def _evaluate(
    run: tuple[JobConfig, Path], tmp_path: Path, name: str, answers: Sequence[str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    questions = ["how many ?", "which color is it ?", "is it a cat ?", "what is it ?", "big ?"]
    rows = [
        _row(f"{question}\n{WORD}", answer)
        for question, answer in zip(questions, answers, strict=True)
    ]
    rows[2] = _row(rows[2][1], rows[2][2], _image(b"not an image"))
    root = _publish(tmp_path / name, rows[:3], rows[3:])
    config, step_dir = run
    output = tmp_path / f"{name}-results"
    tasks = load_tasks(_task_file(tmp_path / f"{name}.toml", root, max_new_tokens=4))
    results = evaluate_checkpoint(config, tasks, step_dir, output, publish=False)
    assert results is not None
    lines = (output / "predictions" / "realworldqa.jsonl").read_text().splitlines()
    return results["tasks"]["realworldqa"], [json.loads(line) for line in lines]


class TestEvaluation:
    def test_the_split_is_decoded_and_scored_with_the_trimmed_match(
        self, run: tuple[JobConfig, Path], tmp_path: Path
    ) -> None:
        summary, records = _evaluate(run, tmp_path, "first", ["zebra"] * 5)
        assert (summary["scorer"], summary["rows"], summary["samples"]) == ("trimmed_match", 5, 4)
        assert summary["skipped"] == {UNREADABLE_ROW: 1}
        assert records[2] == {"row": 2, "skipped": UNREADABLE_ROW}
        scored = [record for record in records if "skipped" not in record]
        assert [record["score"] for record in scored] == [
            trimmed_match(record["prediction"], record["answers"]) for record in scored
        ]
        predictions = {record["row"]: record["prediction"] for record in scored}
        matched = [row for row, prediction in predictions.items() if prediction.strip(" ?.!")]
        assert matched
        answers = [
            f" {predictions[row].upper()}. " if row in matched else "zebra crossing"
            for row in range(5)
        ]
        summary, records = _evaluate(run, tmp_path, "second", answers)
        assert {r["row"]: r["prediction"] for r in records if "skipped" not in r} == predictions
        assert {r["row"]: r["score"] for r in records if "skipped" not in r} == {
            row: float(row in matched) for row in predictions
        }
        assert summary["metrics"] == {"score": len(matched) / 4}
