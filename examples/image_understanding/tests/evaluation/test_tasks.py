"""Tests for the evaluation task file."""

from __future__ import annotations

import json
import re
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest
from evaluation.scorers import SCORERS
from evaluation.tasks import READERS, EvalTask, build_reader, load_tasks
from image_text.testbed import ImageTextRows, chartqa, coco_captions, docvqa, textvqa, vqav2
from PIL import Image

SAMPLE = Path(__file__).resolve().parents[2] / "configs" / "eval_tasks.toml"

TASK = """\
[[tasks]]
name = "docvqa"
reader = "docvqa"
scorer = "anls"
max_new_tokens = 50

[tasks.reader_args]
annotation_file = "ann.json"
image_root = "images"
"""
HEAD = TASK.split("[tasks.reader_args]")[0]
TASK_KEYS = ["max_new_tokens", "name", "reader", "reader_args", "scorer"]


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "tasks.toml"
    path.write_text(text)
    return path


def _refused(tmp_path: Path, text: str, message: str) -> None:
    path = _write(tmp_path, text)
    with pytest.raises(ValueError, match=re.escape(message)) as error:
        load_tasks(path)
    assert str(error.value).startswith(f"{path}: ")


def _without(key: str) -> str:
    return "\n".join(line for line in TASK.splitlines() if not line.startswith(f"{key} ="))


def _task(reader: str = "rows", **reader_args: Any) -> EvalTask:
    return EvalTask("task", reader, "anls", 1, {"annotation_file": "ann.json", **reader_args})


def _rows(
    annotation_file: str | Path,
    limit: int | None = None,
    ratio: float = 1.0,
    shuffle: bool = False,
    ratings: dict[str, int] | None = None,
    extra=None,
    *,
    prompt: str = "",
) -> list[str]:
    return [str(annotation_file)]


class _Rows:
    def __init__(self, annotation_file: str | Path, limit: int = 0) -> None:
        self.annotation_file = annotation_file
        self.limit = limit


def _loose(annotation_file: str, *rest: str, **options: str) -> list[str]:
    return [annotation_file, *rest, *options]


def _nothing() -> list[str]:
    return []


class TestLoadTasks:
    def test_a_task_table_becomes_a_task(self, tmp_path: Path) -> None:
        assert load_tasks(_write(tmp_path, TASK)) == (
            EvalTask(
                name="docvqa",
                reader="docvqa",
                scorer="anls",
                max_new_tokens=50,
                reader_args={"annotation_file": "ann.json", "image_root": "images"},
            ),
        )

    def test_tasks_keep_the_file_order(self, tmp_path: Path) -> None:
        second = TASK.replace('name = "docvqa"', 'name = "docvqa-test"')
        loaded = load_tasks(str(_write(tmp_path, TASK + "\n" + second)))
        assert [task.name for task in loaded] == ["docvqa", "docvqa-test"]

    def test_a_missing_file_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_tasks(tmp_path / "absent.toml")

    def test_invalid_toml_names_the_file(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "[[tasks]\n")
        with pytest.raises(ValueError, match=re.escape(f"{path}: ")):
            load_tasks(path)


class TestSampleFile:
    def test_it_holds_the_five_tasks_and_their_caps(self) -> None:
        assert [
            (task.name, task.reader, task.scorer, task.max_new_tokens)
            for task in load_tasks(SAMPLE)
        ] == [
            ("vqav2", "vqav2", "vqa_accuracy", 10),
            ("textvqa", "textvqa", "vqa_accuracy", 75),
            ("docvqa", "docvqa", "anls", 50),
            ("chartqa-human", "chartqa", "relaxed_accuracy", 50),
            ("chartqa-machine", "chartqa", "relaxed_accuracy", 50),
        ]

    def test_its_paths_are_placeholders(self) -> None:
        for task in load_tasks(SAMPLE):
            assert set(task.reader_args) == {"annotation_file", "image_root"}
            assert all(value.startswith("path-to-") for value in task.reader_args.values())


class TestFileLayout:
    @pytest.mark.parametrize(
        ("text", "message"),
        [
            (
                "seed = 1\n" + TASK,
                "the task file: unknown keys ['seed']; expected some of ['tasks']",
            ),
            ("", "the task file: missing keys ['tasks']; expected ['tasks']"),
            ("tasks = []\n", "tasks must be one or more [[tasks]] tables"),
            ("tasks = [1, 2]\n", "tasks must be one or more [[tasks]] tables"),
            ('[tasks]\nname = "docvqa"\n', "tasks must be one or more [[tasks]] tables"),
            (
                HEAD + 'prompt = "Describe it."\n',
                f"tasks[0]: unknown keys ['prompt']; expected some of {TASK_KEYS}",
            ),
            (TASK + "\n" + TASK, "task names must be unique; repeated: ['docvqa']"),
        ],
    )
    def test_anything_but_task_tables_is_refused(
        self, tmp_path: Path, text: str, message: str
    ) -> None:
        _refused(tmp_path, text, message)

    @pytest.mark.parametrize("key", ["name", "reader", "scorer", "max_new_tokens"])
    def test_a_task_needs_every_key(self, tmp_path: Path, key: str) -> None:
        _refused(tmp_path, _without(key), f"tasks[0]: missing keys ['{key}']; expected {TASK_KEYS}")


class TestTaskValues:
    def test_an_unknown_reader_lists_the_valid_ones(self, tmp_path: Path) -> None:
        _refused(
            tmp_path,
            TASK.replace('reader = "docvqa"', 'reader = "docvqa2"'),
            f"task 'docvqa': unknown reader 'docvqa2'; expected one of {sorted(READERS)}",
        )

    def test_an_unknown_scorer_lists_the_valid_ones(self, tmp_path: Path) -> None:
        _refused(
            tmp_path,
            TASK.replace('scorer = "anls"', 'scorer = "bleu"'),
            f"task 'docvqa': unknown scorer 'bleu'; expected one of {sorted(SCORERS)}",
        )

    @pytest.mark.parametrize("key", ["reader", "scorer"])
    @pytest.mark.parametrize("value", ["3", '["anls"]'])
    def test_a_reader_or_scorer_must_be_a_name(self, tmp_path: Path, key: str, value: str) -> None:
        text = re.sub(rf'{key} = "\w+"', f"{key} = {value}", TASK)
        _refused(tmp_path, text, f"task 'docvqa': unknown {key} ")

    @pytest.mark.parametrize("value", ["0", "-1", '"10"', "1.5", "true"])
    def test_the_answer_cap_is_an_integer_of_at_least_one(self, tmp_path: Path, value: str) -> None:
        _refused(
            tmp_path,
            TASK.replace("max_new_tokens = 50", f"max_new_tokens = {value}"),
            "task 'docvqa': max_new_tokens must be an integer of at least 1, got ",
        )

    def test_a_cap_of_one_token_is_accepted(self, tmp_path: Path) -> None:
        text = TASK.replace("max_new_tokens = 50", "max_new_tokens = 1")
        assert load_tasks(_write(tmp_path, text))[0].max_new_tokens == 1

    @pytest.mark.parametrize("value", ['""', '"two words"', '"a/b"', '"-lead"', '"é"', "3"])
    def test_a_name_is_a_plain_identifier(self, tmp_path: Path, value: str) -> None:
        _refused(
            tmp_path,
            TASK.replace('name = "docvqa"', f"name = {value}"),
            "a task name starts with a letter or digit and continues with letters, digits, "
            "'.', '_' or '-'; got ",
        )

    @pytest.mark.parametrize("name", ["chartqa-human", "v2.1_val", "0shot", "A"])
    def test_letters_digits_and_a_few_marks_make_a_name(self, tmp_path: Path, name: str) -> None:
        text = TASK.replace('name = "docvqa"', f'name = "{name}"')
        assert load_tasks(_write(tmp_path, text))[0].name == name


class TestReaderArgs:
    def test_an_unknown_argument_lists_the_readers_own(self, tmp_path: Path) -> None:
        _refused(
            tmp_path,
            TASK + 'split = "val"\n',
            "task 'docvqa': reader 'docvqa' takes no ['split']; "
            "expected some of ['annotation_file', 'image_root']",
        )

    def test_a_missing_argument_is_named(self, tmp_path: Path) -> None:
        _refused(
            tmp_path,
            TASK.replace('image_root = "images"\n', ""),
            "task 'docvqa': reader 'docvqa' needs ['image_root'] in reader_args; "
            "it takes ['annotation_file', 'image_root']",
        )

    def test_a_task_without_the_table_lacks_every_argument(self, tmp_path: Path) -> None:
        _refused(tmp_path, HEAD, "needs ['annotation_file', 'image_root'] in reader_args")

    def test_an_argument_of_the_wrong_type_is_refused(self, tmp_path: Path) -> None:
        _refused(
            tmp_path,
            TASK.replace('image_root = "images"', "image_root = 5"),
            "task 'docvqa': reader_args.image_root must be a string, got 5",
        )

    def test_the_arguments_are_a_table(self, tmp_path: Path) -> None:
        _refused(
            tmp_path,
            HEAD + 'reader_args = "ann.json"\n',
            "task 'docvqa': reader_args must be a table of the reader's arguments, got 'ann.json'",
        )

    def test_a_caption_reader_needs_its_prompt(self, tmp_path: Path) -> None:
        captions = TASK.replace('reader = "docvqa"', 'reader = "coco_captions"')
        _refused(
            tmp_path,
            captions,
            "task 'docvqa': reader 'coco_captions' needs ['prompt'] in reader_args; "
            "it takes ['annotation_file', 'image_root', 'prompt']",
        )
        _refused(tmp_path, captions + "prompt = 3\n", "reader_args.prompt must be a string, got 3")
        loaded = load_tasks(_write(tmp_path, captions + 'prompt = "Describe the image."\n'))
        assert loaded[0].reader_args["prompt"] == "Describe the image."

    @pytest.mark.parametrize(
        "reader_args",
        [
            {"limit": 3},
            {"ratio": 2},
            {"ratio": 0.5},
            {"shuffle": True},
            {"ratings": {"relevance": 4}},
            {"extra": [1, "x"]},
            {"prompt": "Describe the image."},
        ],
    )
    def test_values_of_the_annotated_type_are_accepted(
        self, monkeypatch: pytest.MonkeyPatch, reader_args: dict[str, Any]
    ) -> None:
        monkeypatch.setitem(READERS, "rows", _rows)
        assert _task(**reader_args).reader_args == {"annotation_file": "ann.json", **reader_args}

    @pytest.mark.parametrize(
        ("reader_args", "message"),
        [
            ({"limit": True}, "reader_args.limit must be an integer, got True"),
            ({"limit": "3"}, "reader_args.limit must be an integer, got '3'"),
            ({"ratio": "half"}, "reader_args.ratio must be a number, got 'half'"),
            ({"ratio": True}, "reader_args.ratio must be a number, got True"),
            ({"shuffle": 1}, "reader_args.shuffle must be a boolean, got 1"),
            ({"annotation_file": 3}, "reader_args.annotation_file must be a string, got 3"),
            ({"prompt": 1}, "reader_args.prompt must be a string, got 1"),
        ],
    )
    def test_values_of_another_type_are_refused(
        self, monkeypatch: pytest.MonkeyPatch, reader_args: dict[str, Any], message: str
    ) -> None:
        monkeypatch.setitem(READERS, "rows", _rows)
        with pytest.raises(ValueError, match=re.escape(message)):
            _task(**reader_args)

    def test_a_class_is_checked_by_its_constructor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(READERS, "rows", _Rows)
        with pytest.raises(ValueError, match="reader_args.limit must be an integer"):
            _task(limit="3")
        reader = build_reader(_task(limit=3))
        assert isinstance(reader, _Rows) and reader.limit == 3

    def test_only_named_parameters_are_arguments(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(READERS, "rows", _loose)
        assert build_reader(_task()) == ["ann.json"]
        with pytest.raises(ValueError, match=re.escape("takes no ['options']; expected some of")):
            _task(options="x")


class TestEvalTask:
    @pytest.mark.parametrize("field", ["name", "reader", "scorer", "max_new_tokens", "reader_args"])
    def test_a_task_is_frozen(self, field: str) -> None:
        task = EvalTask("docvqa", "docvqa", "anls", 50, {"annotation_file": "a", "image_root": "b"})
        with pytest.raises(FrozenInstanceError):
            setattr(task, field, "other")

    def test_equal_tasks_hash_alike(self) -> None:
        reader_args = {"annotation_file": "a", "image_root": "b"}
        tasks = {EvalTask("docvqa", "docvqa", "anls", 50, dict(reader_args)) for _ in range(2)}
        assert len(tasks) == 1

    def test_a_task_keeps_its_own_copy_of_the_arguments(self) -> None:
        reader_args = {"annotation_file": "a", "image_root": "b"}
        task = EvalTask("docvqa", "docvqa", "anls", 50, reader_args)
        reader_args["image_root"] = "c"
        assert task.reader_args == {"annotation_file": "a", "image_root": "b"}

    def test_a_reader_without_arguments_needs_no_table(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(READERS, "empty", _nothing)
        task = EvalTask("empty", "empty", "anls", 1)
        assert task.reader_args == {} and build_reader(task) == []


class TestBuildReader:
    def test_the_testbed_loaders_are_registered(self) -> None:
        loaders = {
            "chartqa": chartqa,
            "coco_captions": coco_captions,
            "docvqa": docvqa,
            "textvqa": textvqa,
            "vqav2": vqav2,
        }
        assert all(READERS[name] is loader for name, loader in loaders.items())

    def test_the_reader_reads_the_split(self, tmp_path: Path) -> None:
        (tmp_path / "documents").mkdir()
        Image.new("RGB", (8, 6)).save(tmp_path / "documents" / "page.png")
        rows = [{"question": "Date?", "answers": ["1999", "99"], "image": "documents/page.png"}]
        annotation = tmp_path / "docvqa.json"
        annotation.write_text(json.dumps({"data": rows}))
        text = TASK.replace("ann.json", str(annotation)).replace("images", str(tmp_path))
        reader = build_reader(load_tasks(_write(tmp_path, text))[0])
        assert isinstance(reader, ImageTextRows) and len(reader) == 1
        assert reader[0].turns[0].answers == ("1999", "99")
