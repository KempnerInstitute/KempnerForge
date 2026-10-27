"""MLflow publishing of the VLM eval harness: synthetic sidecars, fake mlflow and pandas.

``kempnerforge.metrics.publish`` is the real one; only ``mlflow`` and ``pandas`` are fakes
(``_fake_mlflow``), so the tests need neither package.
"""

from __future__ import annotations

import json
import logging
import os
import re
import runpy
import shlex
import subprocess
import sys
import tempfile
import types
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import eval_publish
import pytest
from eval_publish import collect_scores, preflight, publish_results, scores_path_for
from lmms_adapter import resolve_checkpoint

from kempnerforge.metrics.publish import PublishRefused, read_mlflow_target
from kempnerforge.metrics.publish import main as publish_cli

from ._fake_mlflow import break_import, install_mlflow, install_pandas, uninstall
from .test_adapter import _patch_loaders, _vlm_job_config

EXAMPLE_ROOT = Path(__file__).resolve().parents[3]
URI = "tracking-uri-placeholder"
MISSING = object()  # drops the key from the synthetic sidecar

FIRST = datetime(2026, 10, 21, 9, 5, 7, 250, tzinfo=UTC)
FIRST_STAMP = "20261021T090507.000250Z"
SECOND = datetime(2026, 10, 22, tzinfo=UTC)
SECOND_STAMP = "20261022T000000.000000Z"

SCORES = {
    "mmbench_en_dev_gpt_eval_score": 41.5,
    "mmmu_val_mmmu_acc": 0.31,
    "scienceqa_img_exact_match": 0.45,
    "scienceqa_img_exact_match_flexible-extract": 0.5,
}
# (step, benchmark, metric, filter, score, stderr, n_samples, name), ordered by task, metric
# and filter.
ROWS = [
    (
        4000,
        "mmbench_en_dev",
        "gpt_eval_score",
        "none",
        41.5,
        None,
        4,
        "mmbench_en_dev_gpt_eval_score",
    ),
    (4000, "mmmu_val", "mmmu_acc", "none", 0.31, 0.015, 900, "mmmu_val_mmmu_acc"),
    (
        4000,
        "scienceqa_img",
        "exact_match",
        "flexible-extract",
        0.5,
        0.02,
        2017,
        "scienceqa_img_exact_match_flexible-extract",
    ),
    (4000, "scienceqa_img", "exact_match", "none", 0.45, None, 2017, "scienceqa_img_exact_match"),
]


def _results() -> dict[str, Any]:
    """The shape of ``simple_evaluate``'s return value for three tasks."""
    return {
        "results": {
            "mmmu_val": {
                "alias": "mmmu_val",
                "mmmu_acc,none": 0.31,
                "mmmu_acc_stderr,none": 0.015,
                "samples": 900,
            },
            "mmbench_en_dev": {
                "alias": "mmbench_en_dev",
                "gpt_eval_score,none": 41.5,
                "gpt_eval_score_stderr,none": "N/A",
                "submission,none": "N/A",
                "submission_stderr,none": "N/A",
            },
            "scienceqa_img": {
                "alias": "scienceqa_img",
                "exact_match,none": 0.45,
                "exact_match,flexible-extract": 0.5,
                "exact_match_stderr,flexible-extract": 0.02,
            },
        },
        "higher_is_better": {
            "mmmu_val": {"mmmu_acc": True},
            "mmbench_en_dev": {"gpt_eval_score": True, "submission": True},
            "scienceqa_img": {"exact_match": True},
        },
        "n-samples": {
            "mmmu_val": {"original": 900, "effective": 900},
            "mmbench_en_dev": {"original": 4377, "effective": 4},
            "scienceqa_img": {"original": 2017, "effective": 2017},
        },
    }


def _entry(**overrides: Any) -> dict[str, Any]:
    entry = {"backend": "mlflow", "run_id": "run-1", "experiment": "exp", "tracking_uri": URI}
    return {k: v for k, v in {**entry, **overrides}.items() if v is not MISSING}


def _checkpoint(root: Path, name: str = "step_4000", **meta: Any) -> Path:
    """A synthetic checkpoint directory; ``meta`` overrides or (with MISSING) drops a key."""
    body = {"step": 4000, "tokens_seen": 1_000_000, "tracking": [_entry()], **meta}
    ckpt = root / name
    ckpt.mkdir(parents=True)
    (ckpt / "metadata.json").write_text(
        json.dumps({k: v for k, v in body.items() if v is not MISSING})
    )
    return ckpt


def _clock(monkeypatch, *moments: datetime) -> None:
    """Make the timestamps of the next evaluations the given moments, in order."""
    remaining = iter(moments)
    clock = types.SimpleNamespace(now=lambda tz: next(remaining))
    monkeypatch.setattr(eval_publish, "datetime", clock)


def _raiser(error: Exception):
    def raise_it(*args: Any, **kwargs: Any) -> None:
        raise error

    return raise_it


def _hint(caplog) -> list[str]:
    """The arguments of the re-publish command the log offers."""
    (line,) = [m for m in caplog.messages if "python -m kempnerforge.metrics.publish" in m]
    words = shlex.split(line.split("python -m kempnerforge.metrics.publish ")[1])
    assert words[1] == "--metrics"
    return words


def _warnings(caplog) -> str:
    return "\n".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)


@pytest.fixture(autouse=True)
def _isolated(caplog, tmp_path, monkeypatch):
    """Capture INFO logs, and keep the scores file written without ``--output`` in tmp."""
    caplog.set_level(logging.INFO)
    (tmp_path / "cwd").mkdir()
    monkeypatch.chdir(tmp_path / "cwd")


class TestCollectScores:
    def test_the_declared_metrics_are_the_scores(self):
        scores = collect_scores(_results())
        assert {s.name: s.value for s in scores} == SCORES
        assert [
            (s.benchmark, s.metric, s.filter, s.value, s.stderr, s.n_samples) for s in scores
        ] == [row[1:7] for row in ROWS]

    def test_a_filter_other_than_none_extends_the_name(self):
        by_name = {s.name: s for s in collect_scores(_results())}
        assert by_name["scienceqa_img_exact_match"].filter == "none"
        assert by_name["scienceqa_img_exact_match_flexible-extract"].filter == "flexible-extract"

    def test_names_hold_only_characters_mlflow_accepts(self):
        results = {
            "results": {"refcoco+_bbox_val": {"Bleu_4,none": 1.0, "avg@4,none": 2.0}},
            "higher_is_better": {"refcoco+_bbox_val": {"Bleu_4": True, "avg@4": True}},
        }
        names = [s.name for s in collect_scores(results)]
        assert names == ["refcoco_bbox_val_Bleu_4", "refcoco_bbox_val_avg_4"]
        assert all(re.fullmatch(r"[A-Za-z0-9_.-]+", n) for n in names)

    def test_leaves_out_what_is_not_a_score(self):
        results = {
            "results": {
                "t": {
                    "alias": "t",
                    "samples": 10,
                    "acc,none": 0.5,
                    "acc_stderr,none": 0.1,
                    "acc_stderr_clt,none": 0.1,
                    "acc_expected_accuracy,none": 0.4,
                    "undeclared,none": 3.0,
                    "text,none": "N/A",
                    "flag,none": True,
                    "no_filter": 1.0,
                },
                "untracked": {"acc,none": 0.9},
            },
            "higher_is_better": {"t": {"acc": True, "text": True, "flag": True, "no_filter": True}},
        }
        assert [(s.benchmark, s.name) for s in collect_scores(results)] == [("t", "t_acc")]

    def test_a_missing_declaration_leaves_nothing(self):
        assert collect_scores({"results": {"t": {"acc,none": 0.5}}}) == []
        assert collect_scores({}) == []

    def test_a_score_that_is_not_finite_is_left_out_with_a_warning(self, caplog):
        results = {
            "results": {"t": {"a,none": float("nan"), "b,none": float("inf"), "c,none": 0.5}},
            "higher_is_better": {"t": {"a": True, "b": True, "c": True}},
        }
        assert [s.name for s in collect_scores(results)] == ["t_c"]
        assert "t/a,none" in _warnings(caplog) and "t/b,none" in _warnings(caplog)

    def test_an_integer_too_large_for_a_float_is_left_out(self):
        results = {
            "results": {"t": {"a,none": 10**400, "b,none": 1}},
            "higher_is_better": {"t": {"a": True, "b": True}},
        }
        assert [s.name for s in collect_scores(results)] == ["t_b"]

    def test_a_standard_error_that_is_not_finite_is_dropped(self):
        results = {
            "results": {"t": {"a,none": 0.5, "a_stderr,none": float("nan")}},
            "higher_is_better": {"t": {"a": True}},
        }
        (score,) = collect_scores(results)
        assert score.stderr is None

    def test_a_bypass_metric_is_a_placeholder_and_not_a_score(self):
        results = {
            "results": {"t": {"exact_match,none": 0.5, "extracted_answers,none": 999}},
            "higher_is_better": {"t": {"exact_match": True, "extracted_answers": True}},
            "configs": {
                "t": {
                    "metric_list": [
                        {"metric": "exact_match", "aggregation": "mean"},
                        {"metric": "extracted_answers", "aggregation": "bypass"},
                    ]
                }
            },
        }
        assert [s.name for s in collect_scores(results)] == ["t_exact_match"]

    def test_a_group_inherits_the_bypass_metrics_of_its_subtasks(self):
        results = {
            "results": {
                "g": {"exact_match,none": 0.4, "extracted_answers,none": 999},
                "t": {"exact_match,none": 0.5, "extracted_answers,none": 999},
            },
            "higher_is_better": {
                "g": {"exact_match": True, "extracted_answers": True},
                "t": {"exact_match": True, "extracted_answers": True},
            },
            "group_subtasks": {"g": ["h"], "h": ["t"], "t": []},
            "configs": {
                "t": {"metric_list": [{"metric": "extracted_answers", "aggregation": "bypass"}]}
            },
        }
        assert [s.name for s in collect_scores(results)] == ["g_exact_match", "t_exact_match"]

    def test_malformed_configs_and_subtask_cycles_do_not_matter(self):
        results = {
            "results": {"t": {"a,none": 1.0}},
            "higher_is_better": {"t": {"a": True}},
            "configs": {"t": {"metric_list": ["junk", {"metric": "a", "aggregation": 3}]}},
            "group_subtasks": {"t": ["t", 7]},
        }
        assert [s.name for s in collect_scores(results)] == ["t_a"]

    def test_two_scores_under_one_name_are_refused(self):
        results = {
            "results": {"a_b": {"c,none": 1.0}, "a": {"b_c,none": 2.0}},
            "higher_is_better": {"a_b": {"c": True}, "a": {"b_c": True}},
        }
        with pytest.raises(ValueError, match="both be published as 'a_b_c'"):
            collect_scores(results)


class TestScoresPath:
    def test_beside_the_output_file(self, tmp_path):
        path = scores_path_for("results/run.json", tmp_path / "step_1", FIRST_STAMP)
        assert path == Path(f"results/run.{FIRST_STAMP}.scores.json")

    def test_named_for_the_run_and_step_without_an_output_file(self, tmp_path):
        path = scores_path_for(None, tmp_path / "run" / "step_4000", FIRST_STAMP)
        assert path == Path(f"run_step_4000.{FIRST_STAMP}.scores.json")


class TestPublishResults:
    @pytest.fixture
    def mlflow(self, monkeypatch):
        install_pandas(monkeypatch)
        return install_mlflow(monkeypatch)

    def test_publishes_eval_metrics_at_the_sidecar_step(self, tmp_path, mlflow):
        assert publish_results(_checkpoint(tmp_path), _results(), str(tmp_path / "o.json"))
        (batch,) = mlflow.batches
        assert (batch.run_id, batch.tracking_uri) == ("run-1", URI)
        assert {m.key: (m.value, m.step) for m in batch.metrics} == {
            f"eval/{name}": (value, 4000) for name, value in SCORES.items()
        }

    def test_the_summary_table_has_one_row_per_scored_metric(self, tmp_path, mlflow):
        publish_results(_checkpoint(tmp_path), _results(), None)
        (table,) = mlflow.tables
        assert table.run_id == "run-1"
        assert table.data.columns == [
            "step",
            "benchmark",
            "metric",
            "filter",
            "score",
            "stderr",
            "n_samples",
            "name",
        ]
        assert table.data.rows == ROWS

    def test_scores_json_holds_the_published_scores(self, tmp_path, mlflow):
        publish_results(_checkpoint(tmp_path), _results(), None)
        (uploaded,) = mlflow.dicts
        assert uploaded.run_id == "run-1"
        assert uploaded.dictionary == SCORES

    def test_the_table_and_scores_json_share_a_directory_per_evaluation(
        self, tmp_path, monkeypatch, mlflow
    ):
        _clock(monkeypatch, FIRST, SECOND)
        ckpt = _checkpoint(tmp_path)
        publish_results(ckpt, _results(), None)
        publish_results(ckpt, _results(), None)
        assert [t.artifact_file for t in mlflow.tables] == [
            f"eval/step_4000/{FIRST_STAMP}/summary.json",
            f"eval/step_4000/{SECOND_STAMP}/summary.json",
        ]
        assert [d.artifact_file for d in mlflow.dicts] == [
            f"eval/step_4000/{FIRST_STAMP}/scores.json",
            f"eval/step_4000/{SECOND_STAMP}/scores.json",
        ]

    def test_only_logs_to_the_run_and_never_starts_ends_or_edits_one(self, tmp_path, mlflow):
        publish_results(_checkpoint(tmp_path), _results(), None)
        assert mlflow.events == ["client", "log_batch", "client", "log_table", "log_dict"]
        assert set(mlflow.client_uris) == {URI}

    def test_the_scores_are_saved_before_anything_is_sent(self, tmp_path, monkeypatch):
        install_pandas(monkeypatch)
        _clock(monkeypatch, FIRST)
        saved = tmp_path / "out" / f"run.{FIRST_STAMP}.scores.json"
        on_disk: list[object] = []

        def on_call(method: str) -> None:
            if method == "log_batch":
                on_disk.append(json.loads(saved.read_text()) if saved.exists() else None)

        install_mlflow(monkeypatch, on_call=on_call)
        publish_results(_checkpoint(tmp_path), _results(), str(tmp_path / "out" / "run.json"))
        assert on_disk == [SCORES]

    def test_the_scores_are_saved_next_to_the_output_file(self, tmp_path, monkeypatch, mlflow):
        _clock(monkeypatch, FIRST)
        publish_results(_checkpoint(tmp_path), _results(), str(tmp_path / "out" / "run.json"))
        saved = tmp_path / "out" / f"run.{FIRST_STAMP}.scores.json"
        assert json.loads(saved.read_text()) == SCORES

    def test_without_an_output_file_the_scores_are_saved_in_the_working_directory(
        self, tmp_path, monkeypatch, mlflow
    ):
        _clock(monkeypatch, FIRST)
        publish_results(_checkpoint(tmp_path / "run"), _results(), None)
        saved = tmp_path / "cwd" / f"run_step_4000.{FIRST_STAMP}.scores.json"
        assert json.loads(saved.read_text()) == SCORES

    def test_every_evaluation_keeps_its_own_scores_and_its_own_recovery_command(
        self, tmp_path, monkeypatch, caplog
    ):
        install_pandas(monkeypatch)
        install_mlflow(monkeypatch, fail={"log_batch": RuntimeError("down")})
        _clock(monkeypatch, FIRST, SECOND)
        ckpt = _checkpoint(tmp_path)
        first, second = _results(), _results()
        second["results"]["mmmu_val"]["mmmu_acc,none"] = 0.99

        publish_results(ckpt, first, None)
        first_hint = _hint(caplog)
        caplog.clear()
        publish_results(ckpt, second, None)
        second_hint = _hint(caplog)

        assert first_hint[2] != second_hint[2]
        assert json.loads(Path(first_hint[2]).read_text())["mmmu_val_mmmu_acc"] == 0.31
        assert json.loads(Path(second_hint[2]).read_text())["mmmu_val_mmmu_acc"] == 0.99

    @pytest.mark.parametrize(
        "meta",
        [{"tracking": MISSING}, {"tracking": [{"backend": "wandb", "run_id": "w-1"}]}],
        ids=["no-tracking-block", "no-mlflow-entry"],
    )
    def test_a_refusal_is_surfaced_verbatim_and_the_scores_are_kept(
        self, tmp_path, monkeypatch, mlflow, caplog, meta
    ):
        _clock(monkeypatch, FIRST)
        ckpt = _checkpoint(tmp_path, **meta)
        with pytest.raises(PublishRefused) as refusal:
            read_mlflow_target(ckpt)
        out = tmp_path / "out" / "run.json"

        assert publish_results(ckpt, _results(), str(out)) is False

        assert f"The scores were NOT published to MLflow. Refused: {refusal.value}" in _warnings(
            caplog
        )
        saved = tmp_path / "out" / f"run.{FIRST_STAMP}.scores.json"
        assert json.loads(saved.read_text()) == SCORES
        assert mlflow.events == [] and _hint(caplog)[0] == str(ckpt.resolve())

    def test_a_server_failure_warns_with_every_url_hidden(self, tmp_path, monkeypatch, caplog):
        install_pandas(monkeypatch)
        error = RuntimeError("request to https://user:token@tracking.invalid/api failed")
        mlflow = install_mlflow(monkeypatch, fail={"log_batch": error})

        assert publish_results(_checkpoint(tmp_path), _results(), None) is False

        text = _warnings(caplog)
        assert "Failed: RuntimeError: request to https://<redacted> failed" in text
        assert "token" not in text and "tracking.invalid" not in text
        assert mlflow.tables == [] and mlflow.dicts == []

    def test_the_hint_republishes_the_saved_scores_through_the_core_cli(
        self, tmp_path, monkeypatch, caplog
    ):
        install_pandas(monkeypatch)
        install_mlflow(monkeypatch, fail={"log_batch": RuntimeError("down")})
        _clock(monkeypatch, FIRST)
        ckpt = _checkpoint(tmp_path)
        publish_results(ckpt, _results(), str(tmp_path / "out" / "run.json"))
        hint = _hint(caplog)
        saved = tmp_path / "out" / f"run.{FIRST_STAMP}.scores.json"
        assert hint == [str(ckpt.resolve()), "--metrics", str(saved.resolve())]

        retried = install_mlflow(monkeypatch)
        assert publish_cli(hint) == 0
        (batch,) = retried.batches
        assert {m.key: m.value for m in batch.metrics} == {
            f"eval/{n}": v for n, v in SCORES.items()
        }

    def test_a_failed_table_upload_keeps_the_metrics_and_still_uploads_scores_json(
        self, tmp_path, monkeypatch, caplog
    ):
        install_pandas(monkeypatch)
        mlflow = install_mlflow(monkeypatch, fail={"log_table": ValueError("table refused")})

        assert publish_results(_checkpoint(tmp_path), _results(), None) is False

        assert "published to MLflow, but the upload of the summary table failed" in _warnings(
            caplog
        )
        assert "ValueError: table refused" in _warnings(caplog)
        assert len(mlflow.batches) == 1 and len(mlflow.dicts) == 1
        assert "nothing re-uploads the table or scores.json" in _warnings(caplog)
        assert _hint(caplog)[1] == "--metrics"

    def test_a_failed_scores_json_upload_is_reported(self, tmp_path, monkeypatch, caplog):
        install_pandas(monkeypatch)
        mlflow = install_mlflow(monkeypatch, fail={"log_dict": OSError("no space")})

        assert publish_results(_checkpoint(tmp_path), _results(), None) is False

        assert "the upload of the scores.json failed: OSError: no space" in _warnings(caplog)
        assert len(mlflow.tables) == 1
        assert _hint(caplog)[1] == "--metrics"

    def test_a_client_that_cannot_be_built_fails_both_uploads(self, tmp_path, monkeypatch, caplog):
        install_pandas(monkeypatch)
        clients: list[str] = []

        def on_call(method: str) -> None:
            if method == "client":
                clients.append(method)
                if len(clients) == 2:  # the first client belongs to the metrics publisher
                    raise ConnectionError("no route")

        install_mlflow(monkeypatch, on_call=on_call)
        assert publish_results(_checkpoint(tmp_path), _results(), None) is False
        assert "upload of the summary table and scores.json failed: ConnectionError" in _warnings(
            caplog
        )

    def test_without_pandas_only_the_table_is_missing(self, tmp_path, monkeypatch, caplog):
        mlflow = install_mlflow(monkeypatch)
        uninstall(monkeypatch, "pandas")

        assert publish_results(_checkpoint(tmp_path), _results(), None) is False

        assert "pandas could not be imported" in _warnings(caplog)
        assert "uv sync --group eval" in _warnings(caplog)
        assert len(mlflow.batches) == 1 and mlflow.tables == [] and len(mlflow.dicts) == 1
        assert _hint(caplog)[1] == "--metrics"

    def test_a_dependency_that_breaks_when_imported_only_costs_its_upload(
        self, tmp_path, monkeypatch, caplog
    ):
        mlflow = install_mlflow(monkeypatch)
        break_import(monkeypatch, "pandas", ValueError("numpy.dtype size changed"))

        assert publish_results(_checkpoint(tmp_path), _results(), None) is False

        assert "pandas could not be imported (ValueError: numpy.dtype size changed)" in _warnings(
            caplog
        )
        assert len(mlflow.batches) == 1 and mlflow.tables == [] and len(mlflow.dicts) == 1

    def test_without_mlflow_the_core_refusal_is_surfaced(self, tmp_path, monkeypatch, caplog):
        install_pandas(monkeypatch)
        uninstall(monkeypatch, "mlflow")

        assert publish_results(_checkpoint(tmp_path), _results(), None) is False

        assert "Refused: mlflow could not be imported" in _warnings(caplog)
        assert "the environment is incomplete, run: uv sync" in _warnings(caplog)
        assert "--group mlflow" not in _warnings(caplog)
        assert _hint(caplog)

    def test_an_unwritable_location_falls_back_to_a_temporary_file(
        self, tmp_path, monkeypatch, caplog
    ):
        install_pandas(monkeypatch)
        install_mlflow(monkeypatch, fail={"log_batch": RuntimeError("down")})
        (tmp_path / "tmp").mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
        (tmp_path / "blocked").write_text("a file where a directory is needed")

        publish_results(_checkpoint(tmp_path), _results(), str(tmp_path / "blocked" / "r.json"))

        (fallback,) = (tmp_path / "tmp").glob("eval-scores-*.json")
        assert json.loads(fallback.read_text()) == SCORES
        assert f"Saved the scores to {fallback} instead" in _warnings(caplog)
        assert _hint(caplog)[2] == str(fallback)

    def test_a_symlink_loop_in_the_location_falls_back_too(self, tmp_path, monkeypatch, mlflow):
        (tmp_path / "tmp").mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
        (tmp_path / "loop").symlink_to("loop")  # Path.resolve raises RuntimeError for a loop

        assert publish_results(_checkpoint(tmp_path), _results(), str(tmp_path / "loop" / "r.json"))

        assert len(list((tmp_path / "tmp").glob("eval-scores-*.json"))) == 1
        assert len(mlflow.batches) == 1

    def test_a_scores_name_too_long_for_the_file_system_falls_back_too(
        self, tmp_path, monkeypatch, mlflow
    ):
        (tmp_path / "tmp").mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
        limit = os.pathconf(tmp_path, "PC_NAME_MAX")
        output = tmp_path / ("a" * limit + ".json")  # the scores name outgrows the limit

        assert publish_results(_checkpoint(tmp_path), _results(), str(output))

        assert len(list((tmp_path / "tmp").glob("eval-scores-*.json"))) == 1
        assert len(mlflow.batches) == 1

    def test_publishing_goes_on_with_the_temporary_file_as_the_local_copy(
        self, tmp_path, monkeypatch, mlflow
    ):
        (tmp_path / "tmp").mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
        (tmp_path / "blocked").write_text("x")
        assert publish_results(
            _checkpoint(tmp_path), _results(), str(tmp_path / "blocked" / "r.json")
        )
        assert len(mlflow.batches) == 1

    def test_with_no_local_copy_nothing_is_sent_and_the_exact_scores_are_logged(
        self, tmp_path, monkeypatch, mlflow, caplog
    ):
        (tmp_path / "blocked").write_text("x")
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "blocked"))
        results = _results()
        results["results"]["mmmu_val"]["mmmu_acc,none"] = 0.123456789

        assert (
            publish_results(_checkpoint(tmp_path), results, str(tmp_path / "blocked" / "r.json"))
            is False
        )

        assert mlflow.events == []
        text = _warnings(caplog)
        assert "nothing is sent without a local copy" in text
        assert '"mmmu_val_mmmu_acc": 0.123456789' in text
        assert not [m for m in caplog.messages if "python -m kempnerforge" in m]

    def test_the_recovery_command_needs_no_lookup_of_the_working_directory(
        self, tmp_path, monkeypatch, caplog
    ):
        install_pandas(monkeypatch)

        def on_call(method: str) -> None:
            if method == "log_table":
                monkeypatch.setattr(os, "getcwd", _raiser(FileNotFoundError("cwd is gone")))

        install_mlflow(monkeypatch, on_call=on_call, fail={"log_table": RuntimeError("down")})

        assert publish_results(_checkpoint(tmp_path), _results(), None) is False

        assert _hint(caplog)[1] == "--metrics"

    def test_results_without_scores_publish_nothing(self, tmp_path, mlflow, caplog):
        results = {"results": {"t": {"alias": "t"}}, "higher_is_better": {"t": {}}}
        assert publish_results(_checkpoint(tmp_path), results, str(tmp_path / "r.json")) is False
        assert "No scores to publish" in _warnings(caplog)
        assert mlflow.events == [] and not (tmp_path / "r.scores.json").exists()

    def test_a_name_collision_publishes_nothing(self, tmp_path, mlflow, caplog):
        results = {
            "results": {"a_b": {"c,none": 1.0}, "a": {"b_c,none": 2.0}},
            "higher_is_better": {"a_b": {"c": True}, "a": {"b_c": True}},
        }
        assert publish_results(_checkpoint(tmp_path), results, str(tmp_path / "r.json")) is False
        assert "would both be published as 'a_b_c'" in _warnings(caplog)
        assert mlflow.events == []


class TestPreflight:
    @pytest.fixture
    def fakes(self, monkeypatch):
        install_pandas(monkeypatch)
        return install_mlflow(monkeypatch)

    def test_a_usable_checkpoint_only_says_where_the_scores_go(self, tmp_path, fakes, caplog):
        preflight(_checkpoint(tmp_path), None)
        assert _warnings(caplog) == ""
        assert "run run-1 at step 4000" in caplog.text
        assert fakes.events == []

    def test_a_limit_is_called_out_before_the_scoring_starts(self, tmp_path, fakes, caplog):
        preflight(_checkpoint(tmp_path), 4)
        assert "--limit 4" in _warnings(caplog)

    def test_a_refusal_is_surfaced_verbatim(self, tmp_path, fakes, caplog):
        ckpt = _checkpoint(tmp_path, tracking=MISSING)
        with pytest.raises(PublishRefused) as refusal:
            read_mlflow_target(ckpt)
        preflight(ckpt, 4)
        text = _warnings(caplog)
        assert f"will not work for this run: {refusal.value}" in text
        assert "pass --no-publish to silence this" in text
        assert "--limit" not in text

    def test_missing_packages_are_named_with_how_to_install_them(
        self, tmp_path, monkeypatch, caplog
    ):
        uninstall(monkeypatch, "mlflow")
        uninstall(monkeypatch, "pandas")
        preflight(_checkpoint(tmp_path), None)
        text = _warnings(caplog)
        assert "the environment is incomplete, run: uv sync" in text
        assert "uv sync --group eval" in text and "--group mlflow" not in text

    def test_a_dependency_that_breaks_when_imported_is_a_warning_not_an_abort(
        self, tmp_path, monkeypatch, caplog
    ):
        install_mlflow(monkeypatch)
        break_import(monkeypatch, "pandas", ValueError("numpy.dtype size changed"))
        preflight(_checkpoint(tmp_path), None)
        text = _warnings(caplog)
        assert "pandas could not be imported (ValueError: numpy.dtype size changed)" in text
        assert "uv sync --group eval" in text

    def test_an_unreadable_sidecar_is_a_warning_not_an_abort(
        self, tmp_path, monkeypatch, fakes, caplog
    ):
        monkeypatch.setattr(eval_publish, "read_mlflow_target", _raiser(PermissionError("denied")))
        preflight(_checkpoint(tmp_path), None)
        assert "could not be read (PermissionError: denied)" in _warnings(caplog)


class TestResolveCheckpoint:
    def test_a_run_directory_resolves_to_its_latest_step(self, tmp_path):
        run = tmp_path / "run"
        (run / "step_100").mkdir(parents=True)
        (run / "step_200").mkdir()
        (run / "latest").symlink_to("step_100")
        assert resolve_checkpoint(str(run)) == (run / "step_100").resolve()

    def test_a_run_directory_without_a_link_resolves_to_its_highest_durable_step(self, tmp_path):
        run = tmp_path / "run"
        for step in (100, 200):
            (run / f"step_{step}").mkdir(parents=True, exist_ok=True)
            (run / f"step_{step}" / ".metadata").write_text("")
        assert resolve_checkpoint(str(run)) == (run / "step_200").resolve()

    def test_a_step_directory_is_used_as_is(self, tmp_path):
        step = tmp_path / "run" / "step_100"
        step.mkdir(parents=True)
        assert resolve_checkpoint(str(step)) == step.resolve()

    def test_a_symlink_to_a_step_resolves_to_the_step(self, tmp_path):
        run = tmp_path / "run"
        (run / "step_100").mkdir(parents=True)
        (run / "latest").symlink_to("step_100")
        assert resolve_checkpoint(str(run / "latest")) == (run / "step_100").resolve()


class TestEvalScript:
    @pytest.fixture
    def run_eval(self, monkeypatch, tiny_vlm_configs, tiny_vlm_wrapper):
        """Run eval.py as ``__main__`` with ``simple_evaluate`` stubbed to ``scoring()``."""

        def run(checkpoint: Path, *flags: str, scoring=_results) -> list[str]:
            loaded: list[str] = []
            evaluator = types.ModuleType("lmms_eval.evaluator")
            evaluator.simple_evaluate = lambda **kwargs: scoring()  # type: ignore[attr-defined]
            monkeypatch.setitem(sys.modules, "lmms_eval.evaluator", evaluator)
            _patch_loaders(monkeypatch, _vlm_job_config(tiny_vlm_configs), tiny_vlm_wrapper)
            monkeypatch.setattr(
                "lmms_adapter._load_weights",
                lambda config, path, *args: loaded.append(path) or tiny_vlm_wrapper,
            )
            argv = ["eval.py", "--config", "x", "--checkpoint", str(checkpoint), "--tasks", "t"]
            monkeypatch.setattr(
                sys, "argv", argv + ["--device", "cpu", "--dtype", "float32"] + list(flags)
            )
            runpy.run_path(str(EXAMPLE_ROOT / "eval.py"), run_name="__main__")
            return loaded

        return run

    def test_publishes_after_scoring_by_default(self, tmp_path, monkeypatch, run_eval, caplog):
        install_pandas(monkeypatch)
        mlflow = install_mlflow(monkeypatch)
        _clock(monkeypatch, FIRST)
        out = tmp_path / "out" / "run.json"

        run_eval(_checkpoint(tmp_path), "--output", str(out))

        assert json.loads(out.read_text())["results"]["mmmu_val"]["mmmu_acc,none"] == 0.31
        saved = tmp_path / "out" / f"run.{FIRST_STAMP}.scores.json"
        assert json.loads(saved.read_text()) == SCORES
        assert {m.key for m in mlflow.batches[0].metrics} == {f"eval/{n}" for n in SCORES}
        assert len(mlflow.tables) == 1 and len(mlflow.dicts) == 1

    def test_no_publish_opts_out_entirely(self, tmp_path, monkeypatch, run_eval, caplog):
        install_pandas(monkeypatch)
        mlflow = install_mlflow(monkeypatch)
        out = tmp_path / "out" / "run.json"

        run_eval(_checkpoint(tmp_path), "--output", str(out), "--no-publish")

        assert out.exists() and list(out.parent.glob("*.scores.json")) == []
        assert mlflow.events == [] and "MLflow" not in caplog.text

    def test_a_missing_tracking_block_leaves_the_run_successful(
        self, tmp_path, monkeypatch, run_eval, caplog
    ):
        mlflow = install_mlflow(monkeypatch)
        install_pandas(monkeypatch)
        out = tmp_path / "out" / "run.json"

        run_eval(_checkpoint(tmp_path, tracking=MISSING), "--output", str(out))

        assert out.exists() and len(list(out.parent.glob("*.scores.json"))) == 1
        assert caplog.text.count("no 'tracking' block") == 2  # before the scoring and after it
        assert mlflow.events == []

    def test_a_server_failure_leaves_the_run_successful(self, tmp_path, monkeypatch, run_eval):
        install_pandas(monkeypatch)
        install_mlflow(monkeypatch, fail={"log_batch": RuntimeError("down")})
        out = tmp_path / "run.json"
        run_eval(_checkpoint(tmp_path), "--output", str(out))
        assert out.exists() and len(list(tmp_path.glob("run.*.scores.json"))) == 1

    def test_a_broken_dependency_does_not_stop_the_scoring(
        self, tmp_path, monkeypatch, run_eval, caplog
    ):
        install_mlflow(monkeypatch)
        break_import(monkeypatch, "pandas", ValueError("numpy.dtype size changed"))
        scored: list[int] = []
        out = tmp_path / "out" / "run.json"

        run_eval(
            _checkpoint(tmp_path),
            "--output",
            str(out),
            scoring=lambda: scored.append(1) or _results(),
        )

        assert scored == [1] and out.exists()
        assert "pandas could not be imported (ValueError" in caplog.text

    def test_scores_belong_to_the_step_scored_even_if_latest_moves(
        self, tmp_path, monkeypatch, run_eval
    ):
        install_pandas(monkeypatch)
        mlflow = install_mlflow(monkeypatch)
        run = tmp_path / "run"
        _checkpoint(run, "step_100", step=100)
        _checkpoint(run, "step_200", step=200)
        (run / "latest").symlink_to("step_100")

        def scoring():
            (run / "latest").unlink()
            (run / "latest").symlink_to("step_200")
            return _results()

        loaded = run_eval(run, "--output", str(tmp_path / "r.json"), scoring=scoring)

        assert loaded == [str((run / "step_100").resolve())]
        assert {m.step for m in mlflow.batches[0].metrics} == {100}

    def test_nothing_is_published_without_results(self, tmp_path, monkeypatch, run_eval):
        install_pandas(monkeypatch)
        mlflow = install_mlflow(monkeypatch)
        run_eval(_checkpoint(tmp_path), scoring=lambda: None)
        assert mlflow.events == []

    def test_help_lists_the_opt_out(self):
        result = subprocess.run(
            [sys.executable, str(EXAMPLE_ROOT / "eval.py"), "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "--no-publish" in result.stdout
