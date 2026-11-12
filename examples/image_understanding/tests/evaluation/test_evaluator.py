"""Tests for evaluating a saved checkpoint on evaluation tasks."""

from __future__ import annotations

import json
import os
import socket
import statistics
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from evaluation import evaluator
from evaluation.checkpoint import load_model
from evaluation.evaluator import (
    PROMPT_TOO_LONG,
    UNREADABLE_ROW,
    UNRENDERABLE_ROW,
    ClearedResults,
    Decoder,
    check_token_budgets,
    clear_results,
    evaluate_batch,
    evaluate_checkpoint,
    merge_records,
    read_row,
    shard_batches,
    summarize,
)
from evaluation.generation import Generation, stop_token_ids
from evaluation.inputs import UnsupportedDocumentError, build_renderer
from evaluation.scorers import SCORERS, Scorer, anls
from evaluation.tasks import READERS, EvalTask, build_reader, load_tasks
from image_text.document import Document, Turn
from image_text.render import ChatRenderer, UnrenderableDocumentError
from image_text.skip import NoUsableRowError
from PIL import Image

from kempnerforge.config.job import JobConfig
from kempnerforge.config.loader import load_config
from kempnerforge.metrics.mlflow.publish import PublishRefused

LONG_QUESTION_ROW = 4
LONG_PROMPT_TOKENS = 21
EMPTY_PROMPT_TOKENS = 5
MAX_TEXT_LEN = 32
GROUP_TIMEOUT_SEC = 300
RANKS_BOUND_SEC = 150


@dataclass(frozen=True)
class Run:
    """The tiny run: its config file, loaded config and saved step directory."""

    config_path: Path
    config: JobConfig
    step_dir: Path


@pytest.fixture
def run(write_config: Callable[..., Path], save_checkpoint: Callable[..., Path]) -> Run:
    config_path = write_config()
    config = load_config(config_path, cli_args=[])
    return Run(config_path, config, save_checkpoint(config))


def _evaluate(run: Run, tasks: Path, output: Path, **options: Any) -> dict[str, Any]:
    options.setdefault("publish", False)
    results = evaluate_checkpoint(run.config, load_tasks(tasks), run.step_dir, output, **options)
    assert results is not None
    return results


def _predictions(output: Path, task: str = "docvqa") -> list[dict[str, Any]]:
    lines = (output / "predictions" / f"{task}.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def _questions(tasks: Path) -> list[str]:
    annotations = json.loads((tasks.parent / tasks.stem / "annotations.json").read_text())
    return [row["question"] for row in annotations["data"]]


def _corrupt(tasks: Path, row: int) -> None:
    (tasks.parent / tasks.stem / f"{row}.png").write_bytes(b"not an image")


def _decoder(run: Run) -> Decoder:
    renderer = build_renderer(run.config)
    model = load_model(run.config, run.step_dir, torch.device("cpu"))
    return Decoder(model, renderer, stop_token_ids(renderer))


def _batch_records(run: Run, tasks: Path, batch: range) -> list[dict[str, Any]]:
    (task,) = load_tasks(tasks)
    return evaluate_batch(_decoder(run), task, build_reader(task), batch)


class _NamesItsRow(list):
    """A reader that says which row it read, moving on from the rows in ``unusable``."""

    unusable: frozenset[int] = frozenset()

    def read_with_index(self, index: int) -> tuple[int, Any]:
        actual = index
        while actual in self.unusable:
            actual += 1
        return actual, self[actual]


class _CountsSkips(list):
    """A reader that moves on from the rows in ``unusable`` and only counts that it did."""

    unusable: frozenset[int] = frozenset()
    skipped = 0

    def __getitem__(self, index: Any) -> Any:
        while index in self.unusable:
            self.skipped += 1
            index += 1
        return super().__getitem__(index)


class TestReadRow:
    @pytest.mark.parametrize("reader_type", [_NamesItsRow, _CountsSkips])
    def test_a_row_the_reader_passes_over_is_none(self, reader_type: type) -> None:
        reader = reader_type(["a", "b", "c", "d"])
        reader.unusable = frozenset({1, 2})
        assert [read_row(reader, index) for index in range(4)] == ["a", None, None, "d"]

    def test_a_reader_that_never_moves_on_is_read_by_index(self) -> None:
        reader: Any = ["a", "b"]
        assert [read_row(reader, index) for index in range(2)] == ["a", "b"]

    def test_threads_reading_at_once_each_learn_of_their_own_row(
        self, write_task: Callable[..., Path]
    ) -> None:
        """The skip count is shared, so a read must not take another read's skip for its own."""
        tasks = write_task(rows=120)
        unreadable = set(range(3, 120, 7))
        for row in unreadable:
            _corrupt(tasks, row)
        reader = build_reader(load_tasks(tasks)[0])
        with ThreadPoolExecutor(8) as pool:
            documents = list(pool.map(partial(read_row, reader), range(120)))
        assert {row for row, document in enumerate(documents) if document is None} == unreadable
        for row, document in enumerate(documents):
            assert document is None or document.turns[0].user == _questions(tasks)[row]

    def test_a_run_of_rows_the_reader_gives_up_on_is_none_for_each_row(
        self, write_task: Callable[..., Path]
    ) -> None:
        tasks = write_task(rows=24)
        for row in range(2, 19):
            _corrupt(tasks, row)
        reader = build_reader(load_tasks(tasks)[0])
        documents = [read_row(reader, row) for row in range(24)]
        assert [row for row, document in enumerate(documents) if document is None] == list(
            range(2, 19)
        )

    def test_a_give_up_is_none_with_a_reader_that_names_its_row(self) -> None:
        class GivesUp(_NamesItsRow):
            def read_with_index(self, index: int) -> tuple[int, Any]:
                raise NoUsableRowError("gave up after 3 unreadable rows")

        assert read_row(GivesUp(["a", "b"]), 1) is None

    def test_a_give_up_is_none_with_a_reader_read_by_index(self) -> None:
        class GivesUp(list):
            def __getitem__(self, index: Any) -> Any:
                raise NoUsableRowError("gave up after 3 unreadable rows")

        assert read_row(GivesUp(["a", "b"]), 1) is None

    def test_an_error_that_is_not_a_give_up_is_raised_though_the_skip_count_moved(self) -> None:
        class MovesThenFails(_NamesItsRow):
            skipped = 0

            def read_with_index(self, index: int) -> tuple[int, Any]:
                self.skipped += 1
                raise RuntimeError("the reader failed")

        with pytest.raises(RuntimeError, match="the reader failed"):
            read_row(MovesThenFails(["a"]), 0)

    def test_an_error_that_is_not_a_give_up_is_raised_by_a_reader_read_by_index(self) -> None:
        class MovesThenFails(list):
            skipped = 0

            def __getitem__(self, index: Any) -> Any:
                self.skipped += 1
                raise RuntimeError("the reader failed")

        with pytest.raises(RuntimeError, match="the reader failed"):
            read_row(MovesThenFails(["a"]), 0)

    def test_an_unrelated_error_is_raised_though_another_thread_moved_the_skip_count(
        self,
    ) -> None:
        entered = threading.Event()
        moved = threading.Event()

        class Shared(_NamesItsRow):
            skipped = 0

            def read_with_index(self, index: int) -> tuple[int, Any]:
                if index == 0:
                    entered.set()
                    assert moved.wait(timeout=30)
                    raise RuntimeError("the reader failed")
                self.skipped += 1
                moved.set()
                return index, self[index]

        reader = Shared(["a", "b"])
        outcome: list[BaseException] = []

        def failing_read() -> None:
            try:
                read_row(reader, 0)
            except RuntimeError as error:
                outcome.append(error)

        thread = threading.Thread(target=failing_read)
        thread.start()
        assert entered.wait(timeout=30)
        assert read_row(reader, 1) == "b"
        thread.join(timeout=30)
        assert [str(error) for error in outcome] == ["the reader failed"]

    def test_an_error_of_a_reader_that_names_its_row_is_raised_without_skips(self) -> None:
        class Broken(_NamesItsRow):
            skipped = 0

            def read_with_index(self, index: int) -> tuple[int, Any]:
                raise RuntimeError("the reader failed")

        class CountsNothing(_NamesItsRow):
            def read_with_index(self, index: int) -> tuple[int, Any]:
                raise RuntimeError("the reader failed")

        for reader in (Broken(["a"]), CountsNothing(["a"])):
            with pytest.raises(RuntimeError, match="the reader failed"):
                read_row(reader, 0)

    def test_an_error_that_follows_no_skipped_row_is_raised(self) -> None:
        class Broken(list):
            skipped = 0

            def __getitem__(self, index: Any) -> Any:
                raise RuntimeError("the reader failed")

        with pytest.raises(RuntimeError, match="the reader failed"):
            read_row(Broken(["a"]), 0)

    def test_an_image_that_does_not_decode_is_not_answered_by_the_next_row(
        self, write_task: Callable[..., Path]
    ) -> None:
        tasks = write_task(rows=4)
        _corrupt(tasks, 1)
        reader = build_reader(load_tasks(tasks)[0])
        assert read_row(reader, 1) is None
        for row in (0, 2, 3):
            document = read_row(reader, row)
            assert document is not None and document.turns[0].user == _questions(tasks)[row]


class TestShardBatches:
    @pytest.mark.parametrize("world_size", [1, 2, 4, 5])
    @pytest.mark.parametrize("batch_size", [1, 3, 32])
    @pytest.mark.parametrize("rows", [1, 7, 32, 33])
    def test_every_row_is_in_one_batch_of_one_rank(
        self, rows: int, batch_size: int, world_size: int
    ) -> None:
        shards = [shard_batches(rows, batch_size, rank, world_size) for rank in range(world_size)]
        batches = sorted((batch for shard in shards for batch in shard), key=lambda b: b.start)
        assert [row for batch in batches for row in batch] == list(range(rows))
        assert batches == shard_batches(rows, batch_size, 0, 1)
        assert max(len(shard) for shard in shards) - min(len(shard) for shard in shards) <= 1

    def test_ranks_take_the_batches_in_turn(self) -> None:
        assert shard_batches(11, 2, 1, 3) == [range(2, 4), range(8, 10)]
        assert shard_batches(11, 2, 2, 3) == [range(4, 6), range(10, 11)]


class TestEvaluateBatch:
    def test_a_row_records_its_generation_prediction_and_score(
        self, run: Run, write_task: Callable[..., Path]
    ) -> None:
        tasks = write_task(rows=3)
        records = _batch_records(run, tasks, range(3))
        assert [record["row"] for record in records] == [0, 1, 2]
        for record in records:
            assert set(record) == {
                "row",
                "question",
                "answers",
                "generation",
                "prediction",
                "stopped",
                "score",
            }
            assert record["question"] == _questions(tasks)[record["row"]]
            assert record["answers"] == ["cat", "a dog"]
            assert record["prediction"] == record["generation"].replace("<unk> ", "")
            assert "<unk>" not in record["prediction"]
            assert record["score"] == anls(record["prediction"], record["answers"])
            assert isinstance(record["stopped"], bool)
            assert "<|im_end|>" not in record["generation"]
        assert len({record["generation"] for record in records}) > 1

    def test_a_special_token_that_is_not_a_stop_token_is_not_scored(
        self, run: Run, write_task: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tasks = write_task(rows=1)
        tokenizer = _decoder(run).renderer.tokenizer
        ids = tokenizer.convert_tokens_to_ids(["cat", "<|im_start|>"])
        monkeypatch.setattr(
            evaluator, "greedy_decode", lambda *args, **kwargs: [Generation(tuple(ids), True)]
        )
        (record,) = _batch_records(run, tasks, range(1))
        assert record["prediction"] == "cat"
        assert record["score"] == 1.0
        assert record["generation"] == tokenizer.decode(ids, skip_special_tokens=False)
        assert "<|im_start|>" in record["generation"]

    def test_a_prompt_too_long_for_the_budget_is_counted_and_not_cut(
        self, run: Run, write_task: Callable[..., Path]
    ) -> None:
        tasks = write_task(rows=6, max_new_tokens=12)
        records = {record["row"]: record for record in _batch_records(run, tasks, range(6))}
        assert records[LONG_QUESTION_ROW] == {
            "row": LONG_QUESTION_ROW,
            "question": _questions(tasks)[LONG_QUESTION_ROW],
            "skipped": PROMPT_TOO_LONG,
            "prompt_tokens": LONG_PROMPT_TOKENS,
        }
        assert LONG_PROMPT_TOKENS + 12 > MAX_TEXT_LEN
        others = [row for row in range(6) if row != LONG_QUESTION_ROW]
        assert all("score" in records[row] for row in others)
        alone = _batch_records(run, tasks, range(5, 6))
        assert alone == [records[5]]

    def test_a_row_the_reader_passes_over_is_counted_and_not_replaced(
        self, run: Run, write_task: Callable[..., Path]
    ) -> None:
        tasks = write_task(rows=4)
        clean = _batch_records(run, tasks, range(4))
        _corrupt(tasks, 1)
        records = _batch_records(run, tasks, range(4))
        assert records[0] == {"row": 1, "skipped": UNREADABLE_ROW}
        assert records[1:] == [clean[0], clean[2], clean[3]]

    def test_a_row_the_chat_template_cannot_lay_out_is_counted_and_the_others_are_scored(
        self, run: Run, write_task: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tasks = write_task(rows=4)
        clean = _batch_records(run, tasks, range(4))
        refused = _questions(tasks)[2]
        text_and_reply_spans = ChatRenderer.text_and_reply_spans

        def lay_out(self: ChatRenderer, document: Document, answers: Any) -> Any:
            if document.turns[0].user == refused:
                raise UnrenderableDocumentError("a reply carries template text")
            return text_and_reply_spans(self, document, answers)

        monkeypatch.setattr(ChatRenderer, "text_and_reply_spans", lay_out)
        records = _batch_records(run, tasks, range(4))
        assert records[0] == {
            "row": 2,
            "question": refused,
            "skipped": UNRENDERABLE_ROW,
            "error": "a reply carries template text",
        }
        assert UNRENDERABLE_ROW == "unrenderable_row"
        assert records[1:] == [clean[0], clean[1], clean[3]]

    def test_a_batch_with_nothing_to_decode_only_counts(
        self, run: Run, write_task: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tasks = write_task(rows=2)
        _corrupt(tasks, 0)
        monkeypatch.setattr(evaluator, "greedy_decode", None)
        assert _batch_records(run, tasks, range(1)) == [{"row": 0, "skipped": UNREADABLE_ROW}]

    @pytest.mark.parametrize(
        ("document", "limit"),
        [
            (Document(images=(), turns=(Turn("how many ?", ("one",)),)), "one image, got 0"),
            (
                Document(
                    images=(Image.new("RGB", (4, 4)),) * 2, turns=(Turn("how many ?", ("one",)),)
                ),
                "one image, got 2",
            ),
            (
                Document(
                    images=(Image.new("RGB", (4, 4)),),
                    turns=(Turn("how many ?", ("one",)), Turn("which color ?", ("red",))),
                    kind="conversation",
                ),
                "one turn, got 2",
            ),
        ],
    )
    def test_a_document_outside_the_limits_fails_with_its_task_and_row(
        self, run: Run, monkeypatch: pytest.MonkeyPatch, document: Document, limit: str
    ) -> None:
        usable = Document(images=(Image.new("RGB", (4, 4)),), turns=(Turn("how many ?", ("one",)),))
        monkeypatch.setitem(READERS, "mixed", lambda: [usable, document])
        task = EvalTask("mixed-task", "mixed", "anls", 4)
        with pytest.raises(UnsupportedDocumentError, match=f"task 'mixed-task', row 1: .*{limit}"):
            evaluate_batch(_decoder(run), task, build_reader(task), range(2))

    def test_a_missing_image_file_fails(self, run: Run, write_task: Callable[..., Path]) -> None:
        tasks = write_task(rows=2)
        (tasks.parent / tasks.stem / "1.png").unlink()
        with pytest.raises(FileNotFoundError):
            _batch_records(run, tasks, range(2))

    @pytest.mark.parametrize("threads", [2, 8])
    def test_rows_read_by_several_threads_give_the_same_records(
        self, run: Run, write_task: Callable[..., Path], threads: int
    ) -> None:
        tasks = write_task(rows=12, max_new_tokens=12)
        _corrupt(tasks, 6)
        (task,) = load_tasks(tasks)
        one = _batch_records(run, tasks, range(12))
        with ThreadPoolExecutor(threads) as pool:
            several = evaluate_batch(_decoder(run), task, build_reader(task), range(12), pool=pool)
        assert several == one
        assert sorted(record["skipped"] for record in one if "skipped" in record) == [
            PROMPT_TOO_LONG,
            PROMPT_TOO_LONG,
            UNREADABLE_ROW,
        ]

    def test_a_failure_in_a_reading_thread_fails_the_batch(
        self, run: Run, write_task: Callable[..., Path]
    ) -> None:
        tasks = write_task(rows=6)
        (tasks.parent / tasks.stem / "4.png").unlink()
        (task,) = load_tasks(tasks)
        with ThreadPoolExecutor(4) as pool, pytest.raises(FileNotFoundError):
            evaluate_batch(_decoder(run), task, build_reader(task), range(6), pool=pool)


class TestMergeRecords:
    def test_ranks_merge_into_row_order(self) -> None:
        per_rank = [[{"row": 0}, {"row": 3}], [{"row": 1}], [], [{"row": 2}]]
        assert merge_records(per_rank, 4) == [{"row": 0}, {"row": 1}, {"row": 2}, {"row": 3}]

    @pytest.mark.parametrize(
        ("per_rank", "rows"),
        [
            ([[{"row": 0}], [{"row": 2}]], 3),
            ([[{"row": 0}, {"row": 1}], [{"row": 1}]], 2),
            ([[{"row": 0}, {"row": 1}]], 1),
        ],
    )
    def test_a_row_decoded_twice_never_or_past_the_split_is_refused(
        self, per_rank: list[list[dict[str, Any]]], rows: int
    ) -> None:
        with pytest.raises(RuntimeError, match="every row must be decoded exactly once"):
            merge_records(per_rank, rows)


class TestSummarize:
    TASK = EvalTask("docvqa", "docvqa", "anls", 4, {"annotation_file": "a", "image_root": "b"})

    def test_counts_skips_by_reason_and_aggregates_the_scored_rows(self) -> None:
        records = [
            {"row": 0, "score": 1.0, "stopped": True},
            {"row": 1, "skipped": UNREADABLE_ROW},
            {"row": 2, "score": 0.5, "stopped": False},
            {"row": 3, "skipped": PROMPT_TOO_LONG},
            {"row": 4, "skipped": UNREADABLE_ROW},
            {"row": 5, "score": 0.0, "stopped": False},
        ]
        assert summarize(self.TASK, records) == {
            "scorer": "anls",
            "max_new_tokens": 4,
            "rows": 6,
            "samples": 3,
            "skipped": {PROMPT_TOO_LONG: 1, UNREADABLE_ROW: 2},
            "reached_max_new_tokens": 2,
            "metrics": {"score": 0.5},
        }

    def test_a_scorer_with_components_reports_each(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(SCORERS, "parts", Scorer(lambda prediction, reference: {}))
        task = EvalTask("parts-task", "docvqa", "parts", 4, self.TASK.reader_args)
        records = [
            {"row": 0, "score": {"exact": 1.0, "close": 1.0}, "stopped": True},
            {"row": 1, "score": {"exact": 0.0, "close": 0.5}, "stopped": True},
        ]
        assert summarize(task, records)["metrics"] == {"exact": 0.5, "close": 0.75}

    def test_a_task_with_no_scored_row_is_refused(self) -> None:
        records = [{"row": 0, "skipped": UNREADABLE_ROW}, {"row": 1, "skipped": PROMPT_TOO_LONG}]
        with pytest.raises(ValueError, match="task 'docvqa': none of its 2 rows was scored"):
            summarize(self.TASK, records)


class TestClearResults:
    def test_rank_zero_removes_the_marker_and_other_ranks_leave_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "results.json").write_text("{}")
        monkeypatch.setattr(evaluator, "get_world_info", lambda: (1, 1, 2))
        assert clear_results(tmp_path) == ClearedResults(None)
        assert (tmp_path / "results.json").exists()
        monkeypatch.setattr(evaluator, "get_world_info", lambda: (0, 0, 2))
        assert clear_results(tmp_path) == ClearedResults(None)
        assert not (tmp_path / "results.json").exists()
        assert clear_results(tmp_path / "missing") == ClearedResults(None)

    def test_a_removal_that_fails_is_returned(self, tmp_path: Path) -> None:
        (tmp_path / "results.json").mkdir()
        cleared = clear_results(tmp_path)
        assert isinstance(cleared, ClearedResults)
        assert isinstance(cleared.failure, OSError)

    def test_every_file_of_an_earlier_evaluation_goes_and_nothing_else(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "predictions").mkdir()
        earlier = [
            tmp_path / "results.json",
            tmp_path / "metrics.json",
            tmp_path / "predictions" / "docvqa.jsonl",
            tmp_path / "predictions" / "vqav2.jsonl",
        ]
        kept = [
            tmp_path / "launched.json",
            tmp_path / "launch.log",
            tmp_path / "predictions" / "notes.txt",
        ]
        for path in (*earlier, *kept):
            path.write_text("x")
        assert clear_results(tmp_path) == ClearedResults(None)
        assert not any(path.exists() for path in earlier)
        assert all(path.exists() for path in kept)

    def test_the_marker_goes_before_the_other_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "predictions").mkdir()
        names = ["results.json", "metrics.json", "predictions/a.jsonl"]
        for name in names:
            (tmp_path / name).write_text("x")
        removed: list[str] = []
        unlink = Path.unlink

        def record(path: Path, missing_ok: bool = False) -> None:
            removed.append(str(path.relative_to(tmp_path)))
            unlink(path, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", record)
        clear_results(tmp_path)
        assert removed == names

    def test_a_failed_removal_does_not_stop_the_others(self, tmp_path: Path) -> None:
        (tmp_path / "predictions").mkdir()
        (tmp_path / "results.json").mkdir()
        (tmp_path / "metrics.json").write_text("x")
        (tmp_path / "predictions" / "a.jsonl").write_text("x")
        cleared = clear_results(tmp_path)
        assert isinstance(cleared.failure, OSError)
        assert not (tmp_path / "metrics.json").exists()
        assert not (tmp_path / "predictions" / "a.jsonl").exists()

    @pytest.mark.skipif(os.geteuid() == 0, reason="root lists a directory it cannot read")
    def test_a_predictions_directory_that_cannot_be_listed_is_returned_not_raised(
        self, tmp_path: Path
    ) -> None:
        predictions = tmp_path / "predictions"
        predictions.mkdir()
        (predictions / "a.jsonl").write_text("x")
        (tmp_path / "results.json").write_text("x")
        (tmp_path / "metrics.json").write_text("x")
        predictions.chmod(0)
        try:
            cleared = clear_results(tmp_path)
        finally:
            predictions.chmod(0o700)
        assert isinstance(cleared.failure, PermissionError)
        assert not (tmp_path / "results.json").exists()
        assert not (tmp_path / "metrics.json").exists()
        assert (predictions / "a.jsonl").exists()


class TestWriteWhole:
    def test_two_overlapping_writes_of_one_file_both_finish_and_leave_it_whole(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "results.json"
        replace = os.replace
        nested: list[bool] = []

        def interleaved(source: Any, destination: Any) -> None:
            if not nested:
                nested.append(True)
                evaluator._write_json(target, {"writer": "second"})
            replace(source, destination)

        monkeypatch.setattr(evaluator.os, "replace", interleaved)
        evaluator._write_json(target, {"writer": "first"})
        assert json.loads(target.read_text()) == {"writer": "first"}
        assert [path.name for path in tmp_path.iterdir()] == ["results.json"]

    def test_the_predictions_are_written_whole_too(
        self,
        run: Run,
        write_task: Callable[..., Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ours = ["docvqa.jsonl", "metrics.json", "results.json"]
        replaced: list[str] = []
        replace = os.replace

        def recorded(source: Any, destination: Any) -> None:
            replaced.append(Path(destination).name)
            replace(source, destination)

        monkeypatch.setattr(evaluator.os, "replace", recorded)
        output = tmp_path / "results"
        _evaluate(run, write_task(rows=2), output)
        assert [name for name in replaced if name in ours] == ours
        assert sorted(path.name for path in (output / "predictions").iterdir()) == ["docvqa.jsonl"]


class TestEvaluateCheckpoint:
    def test_results_metrics_and_predictions_are_written(
        self, run: Run, write_task: Callable[..., Path], tmp_path: Path
    ) -> None:
        output = tmp_path / "results"
        results = _evaluate(run, write_task(rows=7), output)
        rows = _predictions(output)
        score = statistics.fmean(anls(row["prediction"], row["answers"]) for row in rows)
        assert [row["row"] for row in rows] == list(range(7))
        assert results == {
            "checkpoint": str(run.step_dir.resolve()),
            "step": 3,
            "tasks": {
                "docvqa": {
                    "scorer": "anls",
                    "max_new_tokens": 4,
                    "rows": 7,
                    "samples": 7,
                    "skipped": {},
                    "reached_max_new_tokens": sum(not row["stopped"] for row in rows),
                    "metrics": {"score": score},
                    "predictions": "predictions/docvqa.jsonl",
                }
            },
        }
        assert json.loads((output / "results.json").read_text()) == results
        assert json.loads((output / "metrics.json").read_text()) == {"docvqa/score": score}
        assert sorted(path.name for path in output.iterdir()) == [
            "metrics.json",
            "predictions",
            "results.json",
        ]

    def test_a_prediction_that_matches_a_reference_scores(
        self, run: Run, write_task: Callable[..., Path], tmp_path: Path
    ) -> None:
        _evaluate(run, write_task(rows=6, name="first"), tmp_path / "first")
        first = [row["prediction"] for row in _predictions(tmp_path / "first", "first")]
        matched = [row for row in range(6) if row % 2 == 0 and first[row]]
        assert matched
        tasks = write_task(
            rows=6, answers=lambda row: [first[row] if row in matched else "zebra crossing"]
        )
        results = _evaluate(run, tasks, tmp_path / "second")
        rows = _predictions(tmp_path / "second")
        assert [row["prediction"] for row in rows] == first
        assert [row["score"] for row in rows] == [float(row in matched) for row in range(6)]
        assert results["tasks"]["docvqa"]["metrics"] == {"score": len(matched) / 6}

    def test_a_run_directory_evaluates_its_latest_checkpoint(
        self,
        run: Run,
        save_checkpoint: Callable[..., Path],
        write_task: Callable[..., Path],
        tmp_path: Path,
    ) -> None:
        tasks = load_tasks(write_task(rows=3))
        run_dir = run.config.checkpoint.dir
        first = evaluate_checkpoint(run.config, tasks, run_dir, tmp_path / "a", publish=False)
        later = save_checkpoint(run.config, step=8, seed=1)
        second = evaluate_checkpoint(run.config, tasks, run_dir, tmp_path / "b", publish=False)
        assert first is not None and second is not None
        assert (first["checkpoint"], first["step"]) == (str(run.step_dir.resolve()), 3)
        assert (second["checkpoint"], second["step"]) == (str(later.resolve()), 8)
        assert _predictions(tmp_path / "a") != _predictions(tmp_path / "b")

    def test_every_task_of_the_file_is_evaluated_in_order(
        self, run: Run, write_task: Callable[..., Path], tmp_path: Path
    ) -> None:
        texts = [
            write_task(rows=3, name="zeta").read_text(),
            write_task(rows=5, name="alpha", max_new_tokens=2).read_text(),
        ]
        both = tmp_path / "both.toml"
        both.write_text("\n".join(texts))
        results = _evaluate(run, both, tmp_path / "results")
        assert list(results["tasks"]) == ["zeta", "alpha"]
        assert [task["rows"] for task in results["tasks"].values()] == [3, 5]
        assert list(json.loads((tmp_path / "results" / "metrics.json").read_text())) == [
            "zeta/score",
            "alpha/score",
        ]
        assert len(_predictions(tmp_path / "results", "alpha")) == 5

    def _mixed_task(self, write_task: Callable[..., Path]) -> Path:
        """Eleven rows of which one does not decode and two have prompts past the budget."""
        tasks = write_task(rows=11, max_new_tokens=12)
        _corrupt(tasks, 6)
        return tasks

    @pytest.mark.parametrize("batch_size", [1, 2, 3, 5, 32])
    def test_scores_do_not_depend_on_the_batch_size(
        self, run: Run, write_task: Callable[..., Path], tmp_path: Path, batch_size: int
    ) -> None:
        tasks = self._mixed_task(write_task)
        reference = _evaluate(run, tasks, tmp_path / "reference", batch_size=4)
        results = _evaluate(run, tasks, tmp_path / "results", batch_size=batch_size)
        assert results == reference
        assert reference["tasks"]["docvqa"]["skipped"] == {PROMPT_TOO_LONG: 2, UNREADABLE_ROW: 1}
        assert reference["tasks"]["docvqa"]["samples"] == 8
        for name in ("results.json", "metrics.json", "predictions/docvqa.jsonl"):
            assert (tmp_path / "results" / name).read_bytes() == (
                tmp_path / "reference" / name
            ).read_bytes()

    @pytest.mark.parametrize("threads", [0, 1, 8])
    def test_scores_do_not_depend_on_the_reading_threads(
        self,
        run: Run,
        write_config: Callable[..., Path],
        write_task: Callable[..., Path],
        tmp_path: Path,
        threads: int,
    ) -> None:
        tasks = self._mixed_task(write_task)
        assert run.config.data.num_workers == 4
        reference = _evaluate(run, tasks, tmp_path / "reference")
        config = load_config(write_config(), cli_args=[f"--data.num_workers={threads}"])
        results = evaluate_checkpoint(
            config, load_tasks(tasks), run.step_dir, tmp_path / "results", publish=False
        )
        assert results == reference
        assert _predictions(tmp_path / "results") == _predictions(tmp_path / "reference")

    def test_progress_is_logged_about_ten_times_a_task(
        self,
        run: Run,
        write_task: Callable[..., Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        lines: list[str] = []
        monkeypatch.setattr(evaluator.logger, "info", lines.append)
        _evaluate(run, write_task(rows=25), tmp_path / "results", batch_size=1)
        progress = [line for line in lines if "batches decoded" in line]
        assert progress[0] == "docvqa: 2 of this rank's 25 batches decoded"
        assert len(progress) == 12

    @pytest.mark.parametrize("world_size", [2, 3, 4, 5])
    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_the_shards_of_any_number_of_ranks_merge_into_one_ranks_records(
        self,
        run: Run,
        write_task: Callable[..., Path],
        tmp_path: Path,
        batch_size: int,
        world_size: int,
    ) -> None:
        tasks = self._mixed_task(write_task)
        reference = _evaluate(run, tasks, tmp_path / "reference", batch_size=4)
        (task,) = load_tasks(tasks)
        per_rank = [
            [
                record
                for batch in shard_batches(11, batch_size, rank, world_size)
                for record in _batch_records(run, tasks, batch)
            ]
            for rank in range(world_size)
        ]
        merged = merge_records(per_rank, 11)
        assert merged == _predictions(tmp_path / "reference")
        summary = {**summarize(task, merged), "predictions": "predictions/docvqa.jsonl"}
        assert summary == reference["tasks"]["docvqa"]

    def test_an_unsupported_config_is_refused_before_anything_is_read(
        self, write_config: Callable[..., Path], write_task: Callable[..., Path], tmp_path: Path
    ) -> None:
        config = load_config(write_config(), cli_args=["--vlm.arch=mot"])
        tasks = load_tasks(write_task())
        with pytest.raises(ValueError, match="vlm.arch='mot' is not supported"):
            evaluate_checkpoint(config, tasks, tmp_path / "missing", tmp_path / "results")
        assert not (tmp_path / "results").exists()

    def test_a_task_that_leaves_no_room_for_a_prompt_is_refused(
        self, run: Run, write_task: Callable[..., Path], tmp_path: Path
    ) -> None:
        renderer = build_renderer(run.config)
        fits = load_tasks(
            write_task(name="fits", max_new_tokens=MAX_TEXT_LEN - EMPTY_PROMPT_TOKENS)
        )
        check_token_budgets(renderer, fits)
        tasks = write_task(max_new_tokens=MAX_TEXT_LEN - EMPTY_PROMPT_TOKENS + 1)
        message = (
            "task 'docvqa': max_new_tokens=28 leaves no room for a prompt in "
            "vlm.max_text_len=32; the prompt of an empty question is already 5 tokens"
        )
        with pytest.raises(ValueError, match=message):
            _evaluate(run, tasks, tmp_path / "results")
        assert not (tmp_path / "results").exists()

    def test_a_failure_from_before_the_evaluation_is_raised_before_anything_is_written(
        self, run: Run, write_task: Callable[..., Path], tmp_path: Path
    ) -> None:
        tasks = load_tasks(write_task(rows=3))
        with pytest.raises(OSError, match="cannot clear"):
            evaluate_checkpoint(
                run.config,
                tasks,
                run.step_dir,
                tmp_path / "results",
                publish=False,
                cleared=ClearedResults(OSError("cannot clear")),
            )
        assert not (tmp_path / "results").exists()

    def test_a_recorded_failure_fails_every_rank_in_the_first_agreed_step(
        self,
        run: Run,
        write_task: Callable[..., Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        steps: list[str] = []
        agree = evaluator.agree_across_ranks

        def agreed(failure: Exception | None, what: str) -> None:
            steps.append(what)
            agree(failure, what)

        monkeypatch.setattr(evaluator, "agree_across_ranks", agreed)
        tasks = load_tasks(write_task(rows=3))
        with pytest.raises(OSError, match="cannot clear"):
            evaluate_checkpoint(
                run.config,
                tasks,
                run.step_dir,
                tmp_path / "results",
                publish=False,
                cleared=ClearedResults(OSError("cannot clear")),
            )
        assert steps == ["Preparing the evaluation"]

    def test_the_evaluation_clears_unless_the_caller_did(
        self,
        run: Run,
        write_task: Callable[..., Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[Path] = []
        clear = evaluator.clear_results

        def counted(output: str | Path) -> ClearedResults:
            calls.append(Path(output))
            return clear(output)

        monkeypatch.setattr(evaluator, "clear_results", counted)
        tasks = write_task(rows=2)
        _evaluate(run, tasks, tmp_path / "a")
        assert calls == [tmp_path / "a"]
        _evaluate(run, tasks, tmp_path / "b", cleared=ClearedResults(None))
        assert calls == [tmp_path / "a"]

    def test_a_failed_evaluation_leaves_no_file_of_an_earlier_one(
        self, run: Run, write_task: Callable[..., Path], tmp_path: Path
    ) -> None:
        output = tmp_path / "results"
        tasks = write_task(rows=3)
        _evaluate(run, tasks, output)
        assert (output / "metrics.json").exists()
        assert (output / "predictions" / "docvqa.jsonl").exists()
        (tasks.parent / tasks.stem / "1.png").unlink()
        with pytest.raises(FileNotFoundError):
            _evaluate(run, tasks, output)
        assert not (output / "results.json").exists()
        assert not (output / "metrics.json").exists()
        assert list((output / "predictions").glob("*.jsonl")) == []

    def test_one_pool_reads_the_rows_of_every_batch_of_a_task(
        self,
        run: Run,
        write_task: Callable[..., Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        created: list[int] = []

        class Counted(ThreadPoolExecutor):
            def __init__(self, workers: int) -> None:
                created.append(workers)
                super().__init__(workers)

        monkeypatch.setattr(evaluator, "ThreadPoolExecutor", Counted)
        _evaluate(run, write_task(rows=6), tmp_path / "results", batch_size=2)
        assert created == [4]

    def test_a_task_whose_every_prompt_is_too_long_fails_without_results(
        self, run: Run, write_task: Callable[..., Path], tmp_path: Path
    ) -> None:
        tasks = write_task(rows=3, max_new_tokens=MAX_TEXT_LEN - EMPTY_PROMPT_TOKENS)
        with pytest.raises(ValueError, match="task 'docvqa': none of its 3 rows was scored"):
            _evaluate(run, tasks, tmp_path / "results")
        assert not (tmp_path / "results" / "results.json").exists()

    def test_an_empty_split_is_refused(
        self, run: Run, write_task: Callable[..., Path], tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="task 'docvqa': its split has no rows"):
            _evaluate(run, write_task(rows=0), tmp_path / "results")

    def test_bad_arguments_are_refused(
        self, run: Run, write_config: Callable[..., Path], write_task: Callable[..., Path]
    ) -> None:
        tasks = load_tasks(write_task())
        with pytest.raises(ValueError, match="batch_size must be at least 1, got 0"):
            evaluate_checkpoint(run.config, tasks, run.step_dir, "unused", batch_size=0)
        with pytest.raises(ValueError, match="no task to evaluate"):
            evaluate_checkpoint(run.config, (), run.step_dir, "unused")
        no_tokenizer = load_config(write_config(), cli_args=["--data.tokenizer_path="])
        with pytest.raises(ValueError, match="data.tokenizer_path is not set"):
            evaluate_checkpoint(no_tokenizer, tasks, run.step_dir, "unused")

    def test_a_config_refusal_comes_before_the_ranks_are_joined(
        self,
        write_config: Callable[..., Path],
        write_task: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def joined(config: JobConfig) -> None:
            pytest.fail("the ranks were joined before the config was refused")

        monkeypatch.setattr(evaluator, "_join_ranks", joined)
        no_tokenizer = load_config(write_config(), cli_args=["--data.tokenizer_path="])
        with pytest.raises(ValueError, match="data.tokenizer_path is not set"):
            evaluate_checkpoint(no_tokenizer, load_tasks(write_task()), "unused", "unused")

    def test_a_directory_without_a_checkpoint_is_refused(
        self, run: Run, write_task: Callable[..., Path], tmp_path: Path
    ) -> None:
        tasks = load_tasks(write_task())
        with pytest.raises(FileNotFoundError, match="holds no saved weights"):
            evaluate_checkpoint(run.config, tasks, tmp_path, tmp_path / "results")

    def test_a_run_of_unreadable_rows_is_counted_row_by_row(
        self, run: Run, write_task: Callable[..., Path], tmp_path: Path
    ) -> None:
        tasks = write_task(rows=20)
        for row in range(2, 19):
            _corrupt(tasks, row)
        results = _evaluate(run, tasks, tmp_path / "results")
        summary = results["tasks"]["docvqa"]
        assert (summary["rows"], summary["samples"], summary["skipped"]) == (
            20,
            3,
            {UNREADABLE_ROW: 17},
        )

    def test_a_refused_run_leaves_no_results_json_of_an_earlier_one(
        self, run: Run, write_task: Callable[..., Path], tmp_path: Path
    ) -> None:
        output = tmp_path / "results"
        tasks = write_task(rows=3)
        _evaluate(run, tasks, output)
        assert (output / "results.json").exists()
        with pytest.raises(ValueError, match="batch_size must be at least 1"):
            _evaluate(run, tasks, output, batch_size=0)
        assert not (output / "results.json").exists()

    def test_results_json_exists_only_once_an_evaluation_has_finished(
        self, run: Run, write_task: Callable[..., Path], tmp_path: Path
    ) -> None:
        output = tmp_path / "results"
        tasks = write_task(rows=3)
        _evaluate(run, tasks, output)
        assert (output / "results.json").exists()
        (tasks.parent / "docvqa" / "2.png").unlink()
        with pytest.raises(FileNotFoundError):
            _evaluate(run, tasks, output)
        assert not (output / "results.json").exists()


def _published(monkeypatch: pytest.MonkeyPatch, error: Exception | None = None) -> list[Any]:
    calls: list[Any] = []

    def publish(checkpoint_dir: Path, metrics: dict[str, float]) -> Any:
        calls.append((checkpoint_dir, dict(metrics)))
        if error is not None:
            raise error
        return SimpleNamespace(run_id="run-1", step=3)

    monkeypatch.setattr(evaluator, "publish_metrics", publish)
    return calls


class TestJoiningRanks:
    def test_a_launch_of_several_processes_starts_and_ends_its_group(
        self,
        run: Run,
        write_task: Callable[..., Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import torch.distributed as dist

        started: list[Any] = []

        def start_group(config: Any, seed: int) -> None:
            started.append((config.backend, seed))
            dist.init_process_group(
                "gloo", rank=0, world_size=1, init_method=f"tcp://127.0.0.1:{_free_port()}"
            )

        def end_group() -> None:
            started.append("ended")
            dist.destroy_process_group()

        monkeypatch.setattr(evaluator, "get_world_info", lambda: (0, 0, 2))
        monkeypatch.setattr(evaluator, "init_distributed", start_group)
        monkeypatch.setattr(evaluator, "destroy_distributed", end_group)
        _evaluate(run, write_task(rows=3), tmp_path / "results")
        assert started == [(run.config.distributed.backend, run.config.train.seed), "ended"]
        assert not dist.is_initialized()


class TestPublishing:
    def test_the_task_metrics_are_published_once_for_the_checkpoint(
        self,
        run: Run,
        write_task: Callable[..., Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = _published(monkeypatch)
        results = _evaluate(run, write_task(rows=3), tmp_path / "results", publish=True)
        score = results["tasks"]["docvqa"]["metrics"]["score"]
        assert calls == [(run.step_dir.resolve(), {"docvqa/score": score})]

    def test_nothing_is_published_when_turned_off(
        self,
        run: Run,
        write_task: Callable[..., Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = _published(monkeypatch)
        _evaluate(run, write_task(rows=3), tmp_path / "results", publish=False)
        assert calls == []

    @pytest.mark.parametrize(
        ("error", "reason"),
        [
            (PublishRefused("metadata.json: no 'tracking' block"), "no 'tracking' block"),
            (RuntimeError("POST https://user:secret@host/api/2.0 failed"), "https://<redacted>"),
            (ConnectionError("unreachable"), "ConnectionError: unreachable"),
        ],
    )
    def test_a_failed_publish_warns_and_leaves_the_results(
        self,
        run: Run,
        write_task: Callable[..., Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        error: Exception,
        reason: str,
    ) -> None:
        warnings: list[str] = []
        monkeypatch.setattr(evaluator.logger, "warning", warnings.append)
        calls = _published(monkeypatch, error)
        output = tmp_path / "results"
        results = _evaluate(run, write_task(rows=3), output, publish=True)
        assert len(calls) == 1
        assert json.loads((output / "results.json").read_text()) == results
        (warning,) = warnings
        assert "were not published" in warning and reason in warning and "secret" not in warning
        assert f"{run.step_dir.resolve()} --metrics {output / 'metrics.json'}" in warning


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _evaluate_on_rank(
    rank: int,
    world: int,
    port: int,
    config: str,
    checkpoint: str,
    tasks: str,
    output: str,
    broken_rank: int,
    broken: str,
) -> None:
    """One rank of a launch that has no group yet; it notes how it ended and what it published."""
    import os
    from datetime import timedelta

    import torch.distributed as dist

    notes = Path(output).parent

    def publish(checkpoint_dir: Path, metrics: dict[str, float]) -> Any:
        (notes / f"published-{rank}").write_text(json.dumps([str(checkpoint_dir), metrics]))
        return SimpleNamespace(run_id="run-1", step=3)

    def open_split(task: EvalTask) -> Any:
        assert dist.is_initialized(), "the split was opened before the ranks were joined"
        raise OSError("the split is not on this node")

    def load_tokenizer(config: Any) -> Any:
        assert dist.is_initialized(), "the tokenizer was loaded before the ranks were joined"
        raise OSError("the tokenizer is not on this node")

    def start_group(config: Any, seed: int) -> None:
        dist.init_process_group(
            "gloo",
            rank=rank,
            world_size=world,
            init_method=f"tcp://127.0.0.1:{port}",
            timeout=timedelta(seconds=GROUP_TIMEOUT_SEC),
        )

    os.environ.update(RANK=str(rank), LOCAL_RANK="0", WORLD_SIZE=str(world))
    evaluator.publish_metrics = publish
    evaluator.init_distributed = start_group
    if rank == broken_rank:
        if broken == "split":
            evaluator.build_reader = open_split
        else:
            evaluator.build_renderer = load_tokenizer
    torch.set_num_threads(1)
    ended = "finished"
    try:
        evaluate_checkpoint(
            load_config(config, cli_args=[]),
            load_tasks(tasks),
            checkpoint,
            output,
            batch_size=2,
        )
    except Exception as exc:
        ended = f"{type(exc).__name__}: {exc}"
    (notes / f"ended-{rank}").write_text(ended)
    if dist.is_initialized():
        dist.destroy_process_group()


def _spawn_ranks(
    world: int, run: Run, tasks: Path, output: Path, broken_rank: int = -1, broken: str = "split"
) -> list[str]:
    """Evaluate on ``world`` ranks with no group yet; how each rank ended, in rank order.

    Every rank must end within ``RANKS_BOUND_SEC``, well under the time a rank
    left waiting in the group's start-up would be released by its timeout.
    """
    import torch.multiprocessing as mp

    args = (
        world,
        _free_port(),
        str(run.config_path),
        str(run.step_dir),
        str(tasks),
        str(output),
        broken_rank,
        broken,
    )
    context = mp.start_processes(
        _evaluate_on_rank, args=args, nprocs=world, join=False, start_method="spawn"
    )
    deadline = time.monotonic() + RANKS_BOUND_SEC
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail(f"the ranks did not all end within {RANKS_BOUND_SEC} s")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.kill()
    return [(output.parent / f"ended-{rank}").read_text() for rank in range(world)]


class TestSeveralRanks:
    @pytest.mark.parametrize("world", [2, 3])
    def test_several_ranks_write_what_one_process_writes(
        self, run: Run, write_task: Callable[..., Path], tmp_path: Path, world: int
    ) -> None:
        tasks = write_task(rows=11, max_new_tokens=12)
        _corrupt(tasks, 6)
        _evaluate(run, tasks, tmp_path / "one" / "results", batch_size=2)
        output = tmp_path / "ranks" / "results"
        output.parent.mkdir()
        assert _spawn_ranks(world, run, tasks, output) == ["finished"] * world
        for name in ("results.json", "metrics.json", "predictions/docvqa.jsonl"):
            assert (output / name).read_bytes() == (
                tmp_path / "one" / "results" / name
            ).read_bytes()
        assert sorted(path.name for path in output.parent.glob("published-*")) == ["published-0"]
        published = json.loads((output.parent / "published-0").read_text())
        assert published == [
            str(run.step_dir.resolve()),
            json.loads((output / "metrics.json").read_text()),
        ]

    def test_a_failure_on_one_rank_fails_every_rank(
        self, run: Run, write_task: Callable[..., Path], tmp_path: Path
    ) -> None:
        """Row 2 is in the second batch, which rank 1 decodes; rank 0 must not wait for it."""
        tasks = write_task(rows=8)
        (tasks.parent / "docvqa" / "2.png").unlink()
        output = tmp_path / "ranks" / "results"
        output.parent.mkdir()
        first, second = _spawn_ranks(2, run, tasks, output)
        assert first == "RuntimeError: Evaluating task 'docvqa' failed on ranks [1]"
        assert second.startswith("FileNotFoundError")
        assert not (output / "results.json").exists()
        assert not list(output.parent.glob("published-*"))

    def test_a_split_one_rank_cannot_open_fails_every_rank_before_any_decoding(
        self, run: Run, write_task: Callable[..., Path], tmp_path: Path
    ) -> None:
        tasks = write_task(rows=8)
        output = tmp_path / "ranks" / "results"
        output.parent.mkdir()
        first, second = _spawn_ranks(2, run, tasks, output, broken_rank=1)
        assert first == "RuntimeError: Preparing the evaluation failed on ranks [1]"
        assert second == "OSError: the split is not on this node"
        assert not output.exists()

    def test_a_tokenizer_one_rank_cannot_load_fails_every_rank_with_nothing_written(
        self, run: Run, write_task: Callable[..., Path], tmp_path: Path
    ) -> None:
        tasks = write_task(rows=8)
        output = tmp_path / "ranks" / "results"
        output.parent.mkdir()
        first, second = _spawn_ranks(2, run, tasks, output, broken_rank=1, broken="tokenizer")
        assert first == "RuntimeError: Preparing the evaluation failed on ranks [1]"
        assert second == "OSError: the tokenizer is not on this node"
        assert not output.exists()
        assert not list(output.parent.glob("published-*"))

    def test_a_marker_rank_zero_cannot_remove_fails_every_rank(
        self, run: Run, write_task: Callable[..., Path], tmp_path: Path
    ) -> None:
        tasks = write_task(rows=8)
        output = tmp_path / "ranks" / "results"
        (output / "results.json").mkdir(parents=True)
        first, second = _spawn_ranks(2, run, tasks, output)
        assert first.split(":")[0] in {"IsADirectoryError", "PermissionError"}
        assert second == "RuntimeError: Preparing the evaluation failed on ranks [0]"
