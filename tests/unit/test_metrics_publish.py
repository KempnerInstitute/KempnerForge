"""Tests for the MLflow eval publisher: synthetic sidecars and a fake mlflow, no server."""

from __future__ import annotations

import json
import runpy
import sys
import threading
import time
import types
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import kempnerforge.metrics.publish as publish_mod
from kempnerforge.checkpoint import manager as checkpoint_manager
from kempnerforge.metrics.publish import (
    MlflowTarget,
    PublishRefused,
    main,
    publish_metrics,
    read_mlflow_target,
)

URI = "http://localhost:5000"
MISSING = object()  # drop the key from the synthetic sidecar


def _drop_missing(mapping: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in mapping.items() if v is not MISSING}


def _entry(**overrides: Any) -> dict[str, Any]:
    entry = {"backend": "mlflow", "run_id": "run-1", "experiment": "exp", "tracking_uri": URI}
    return _drop_missing({**entry, **overrides})


def _sidecar(tmp_path: Path, name: str = "step_120", **meta: Any) -> Path:
    """Write a synthetic checkpoint dir; ``meta`` overrides or (with MISSING) drops a key."""
    body = {"step": 120, "tokens_seen": 1_000_000, "tracking": [_entry()], **meta}
    ckpt = tmp_path / name
    ckpt.mkdir()
    (ckpt / "metadata.json").write_text(json.dumps(_drop_missing(body)))
    return ckpt


class _Metric:
    """Mirrors mlflow.entities.Metric, logged-model linkage included."""

    def __init__(
        self,
        key,
        value,
        timestamp,
        step,
        model_id=None,
        dataset_name=None,
        dataset_digest=None,
        run_id=None,
    ):
        self.key, self.value, self.timestamp, self.step = key, value, timestamp, step
        self.model_id = model_id


class _MetricWithoutModelId:
    """An mlflow whose Metric predates logged-model linkage."""

    def __init__(self, key, value, timestamp, step):
        self.key, self.value, self.timestamp, self.step = key, value, timestamp, step
        self.model_id = None


class _FakeMlflow(types.ModuleType):
    """Stand-in `mlflow` recording what a publisher does.

    Only ``MlflowClient(tracking_uri=...)`` and ``log_batch`` exist: touching anything
    else (``start_run``, ``set_tracking_uri``, ``end_run``, ...) raises AssertionError.
    """

    def __init__(self, metric_cls: type = _Metric, on_log_batch=None):
        super().__init__("mlflow")
        self.events: list[str] = []
        self.batches: list[types.SimpleNamespace] = []
        entities = types.ModuleType("mlflow.entities")
        entities.Metric = metric_cls
        self.entities_module = entities
        fake = self

        class MlflowClient:
            def __init__(self, tracking_uri=None, registry_uri=None):
                fake.events.append("client")
                self._uri = tracking_uri

            def log_batch(self, run_id, metrics=(), params=(), tags=(), synchronous=None):
                fake.events.append("log_batch")
                call = types.SimpleNamespace(
                    tracking_uri=self._uri,
                    run_id=run_id,
                    metrics=list(metrics),
                    params=list(params),
                    tags=list(tags),
                    synchronous=synchronous,
                )
                fake.batches.append(call)
                if on_log_batch is not None:
                    on_log_batch(call)

            def __getattr__(self, name):
                raise AssertionError(f"publisher touched MlflowClient.{name}")

        self.MlflowClient = MlflowClient

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        raise AssertionError(f"publisher touched mlflow.{name}")


def _install_mlflow(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> _FakeMlflow:
    fake = _FakeMlflow(**kwargs)
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.setitem(sys.modules, "mlflow.entities", fake.entities_module)
    return fake


# ---------------------------------------------------------------------------
# Reading the sidecar
# ---------------------------------------------------------------------------


class TestReadMlflowTarget:
    def test_reads_the_mlflow_entry_and_the_step(self, tmp_path):
        target = read_mlflow_target(_sidecar(tmp_path))
        assert target == MlflowTarget(
            run_id="run-1", experiment="exp", tracking_uri=URI, step=120, model_id=None
        )

    def test_repr_keeps_the_tracking_uri_out_of_logs(self, tmp_path):
        target = read_mlflow_target(_sidecar(tmp_path, tracking=[_entry(tracking_uri=URI)]))
        assert URI not in repr(target) and "run-1" in repr(target)

    def test_accepts_a_path_object_or_a_string(self, tmp_path):
        ckpt = _sidecar(tmp_path)
        assert read_mlflow_target(str(ckpt)) == read_mlflow_target(ckpt)

    def test_picks_mlflow_among_other_backends_and_ignores_their_fields(self, tmp_path):
        tracking = [
            {"backend": "wandb", "run_id": 5, "project": "p"},
            _entry(model_id="m-1", extra_field="ignored"),
        ]
        target = read_mlflow_target(_sidecar(tmp_path, tracking=tracking))
        assert (target.run_id, target.model_id) == ("run-1", "m-1")

    def test_null_model_id_means_no_model(self, tmp_path):
        ckpt = _sidecar(tmp_path, tracking=[_entry(model_id=None)])
        assert read_mlflow_target(ckpt).model_id is None

    def test_sidecar_filename_matches_the_checkpoint_writer(self):
        assert publish_mod._SIDECAR_FILE == checkpoint_manager._METADATA_FILE


class TestSidecarRefusals:
    def test_directory_that_does_not_exist(self, tmp_path):
        with pytest.raises(PublishRefused, match="is not a directory"):
            read_mlflow_target(tmp_path / "absent")

    def test_path_that_is_a_file(self, tmp_path):
        (tmp_path / "plain").write_text("x")
        with pytest.raises(PublishRefused, match="is not a directory"):
            read_mlflow_target(tmp_path / "plain")

    def test_missing_sidecar(self, tmp_path):
        (tmp_path / "step_1").mkdir()
        with pytest.raises(PublishRefused, match=r"metadata\.json not found"):
            read_mlflow_target(tmp_path / "step_1")

    @pytest.mark.parametrize("raw", [b"{ not json", b"\x80\x81\x82", b""])
    def test_unparseable_sidecar(self, tmp_path, raw):
        ckpt = tmp_path / "step_1"
        ckpt.mkdir()
        (ckpt / "metadata.json").write_bytes(raw)
        with pytest.raises(PublishRefused, match="not valid JSON"):
            read_mlflow_target(ckpt)

    def test_unreadable_sidecar(self, tmp_path, monkeypatch):
        ckpt = _sidecar(tmp_path)

        def _deny(self, *args, **kwargs):
            raise PermissionError("denied")

        monkeypatch.setattr(Path, "read_bytes", _deny)
        with pytest.raises(PublishRefused, match="unreadable or not valid JSON: denied"):
            read_mlflow_target(ckpt)

    @pytest.mark.parametrize("body", [[], "text", 3, None])
    def test_top_level_must_be_an_object(self, tmp_path, body):
        ckpt = tmp_path / "step_1"
        ckpt.mkdir()
        (ckpt / "metadata.json").write_text(json.dumps(body))
        with pytest.raises(PublishRefused, match="top level must be an object"):
            read_mlflow_target(ckpt)

    def test_old_sidecar_without_a_tracking_block(self, tmp_path):
        ckpt = _sidecar(tmp_path, tracking=MISSING)
        with pytest.raises(PublishRefused, match="no 'tracking' block"):
            read_mlflow_target(ckpt)

    @pytest.mark.parametrize("tracking", [{}, "mlflow", None, 7])
    def test_tracking_must_be_a_list(self, tmp_path, tracking):
        with pytest.raises(PublishRefused, match="'tracking' must be a list"):
            read_mlflow_target(_sidecar(tmp_path, tracking=tracking))

    def test_empty_tracking_list_has_no_mlflow_entry(self, tmp_path):
        with pytest.raises(PublishRefused, match=r"no mlflow entry \(backends recorded: none\)"):
            read_mlflow_target(_sidecar(tmp_path, tracking=[]))

    def test_only_other_backends_have_no_mlflow_entry(self, tmp_path):
        tracking = [{"backend": "wandb"}, {"backend": "tensorboard"}]
        with pytest.raises(PublishRefused, match="no mlflow entry.*wandb, tensorboard"):
            read_mlflow_target(_sidecar(tmp_path, tracking=tracking))

    @pytest.mark.parametrize("entry", ["mlflow", 4, None, ["mlflow"]])
    def test_entry_must_be_an_object(self, tmp_path, entry):
        with pytest.raises(PublishRefused, match=r"tracking\[1\] must be an object"):
            read_mlflow_target(_sidecar(tmp_path, tracking=[_entry(), entry]))

    @pytest.mark.parametrize(
        ("backend", "message"),
        [
            (MISSING, "'backend' is missing"),
            ("", "'backend' is empty"),
            (5, "'backend' must be a string, got int"),
        ],
    )
    def test_entry_needs_a_backend_name(self, tmp_path, backend, message):
        tracking = [_entry(backend=backend)]
        with pytest.raises(PublishRefused, match=rf"tracking\[0\]: {message}"):
            read_mlflow_target(_sidecar(tmp_path, tracking=tracking))

    def test_two_mlflow_entries_are_ambiguous(self, tmp_path):
        tracking = [_entry(run_id="a"), {"backend": "wandb"}, _entry(run_id="b")]
        with pytest.raises(PublishRefused, match=r"2 mlflow entries \(tracking\[0, 2\]\)"):
            read_mlflow_target(_sidecar(tmp_path, tracking=tracking))

    @pytest.mark.parametrize("field", ["run_id", "experiment", "tracking_uri"])
    @pytest.mark.parametrize(
        ("value", "message"),
        [
            (MISSING, "is missing"),
            ("", "is empty"),
            ("   ", "is empty"),
            (None, "must be a string, got NoneType"),
            (12, "must be a string, got int"),
            (["x"], "must be a string, got list"),
        ],
    )
    def test_required_field_absent_empty_or_mistyped(self, tmp_path, field, value, message):
        ckpt = _sidecar(tmp_path, tracking=[_entry(**{field: value})])
        with pytest.raises(PublishRefused, match=rf"tracking\[0\]: '{field}' {message}"):
            read_mlflow_target(ckpt)

    @pytest.mark.parametrize(
        ("value", "message"),
        [("", "is empty"), (7, "must be a string, got int"), ([], "must be a string, got list")],
    )
    def test_model_id_when_present_must_be_a_non_empty_string(self, tmp_path, value, message):
        ckpt = _sidecar(tmp_path, tracking=[_entry(model_id=value)])
        with pytest.raises(PublishRefused, match=rf"'model_id' {message}"):
            read_mlflow_target(ckpt)

    @pytest.mark.parametrize(
        ("step", "message"),
        [
            (MISSING, "'step' is missing"),
            ("7", "'step' must be an integer, got str"),
            (7.0, "'step' must be an integer, got float"),
            (True, "'step' must be an integer, got bool"),
            (None, "'step' must be an integer, got NoneType"),
            (-1, "'step' must be non-negative, got -1"),
        ],
    )
    def test_step_must_be_a_non_negative_integer(self, tmp_path, step, message):
        with pytest.raises(PublishRefused, match=message):
            read_mlflow_target(_sidecar(tmp_path, step=step))

    def test_step_zero_is_valid(self, tmp_path):
        assert read_mlflow_target(_sidecar(tmp_path, step=0)).step == 0

    def test_every_problem_is_named_in_one_refusal(self, tmp_path):
        tracking = [_entry(run_id=MISSING, tracking_uri="", experiment=3)]
        ckpt = _sidecar(tmp_path, step=MISSING, tracking=tracking)
        with pytest.raises(PublishRefused) as err:
            read_mlflow_target(ckpt)
        message = str(err.value)
        for fragment in (
            "'step' is missing",
            "'run_id' is missing",
            "'experiment' must be a string, got int",
            "'tracking_uri' is empty",
        ):
            assert fragment in message

    def test_refusals_never_echo_field_values(self, tmp_path):
        marker = "http://localhost:5000/marker-not-to-be-echoed"
        ckpt = _sidecar(tmp_path, tracking=[_entry(tracking_uri=marker, run_id=MISSING)])
        with pytest.raises(PublishRefused) as err:
            read_mlflow_target(ckpt)
        assert "marker-not-to-be-echoed" not in str(err.value)


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------


def _now_ms() -> int:
    return int(time.time() * 1000)


class TestPublishMetrics:
    def test_appends_prefixed_metrics_at_the_sidecar_step(self, tmp_path, monkeypatch):
        fake = _install_mlflow(monkeypatch)
        ckpt = _sidecar(tmp_path, step=1500)

        before = _now_ms()
        target = publish_metrics(ckpt, {"mmmu": 0.5, "mmbench": 2})
        after = _now_ms()

        (batch,) = fake.batches
        assert batch.run_id == "run-1"
        assert batch.synchronous is True
        assert batch.params == [] and batch.tags == []
        assert [(m.key, m.value, m.step) for m in batch.metrics] == [
            ("eval/mmmu", 0.5, 1500),
            ("eval/mmbench", 2.0, 1500),
        ]
        assert all(isinstance(m.value, float) for m in batch.metrics)
        assert all(before <= m.timestamp <= after for m in batch.metrics)
        assert target == read_mlflow_target(ckpt)

    @pytest.mark.parametrize(
        ("prefix", "expected"), [("probe/", "probe/acc"), ("", "acc"), ("eval_", "eval_acc")]
    )
    def test_prefix_is_prepended_verbatim(self, tmp_path, monkeypatch, prefix, expected):
        fake = _install_mlflow(monkeypatch)
        publish_metrics(_sidecar(tmp_path), {"acc": 1.0}, prefix=prefix)
        assert [m.key for m in fake.batches[0].metrics] == [expected]

    def test_only_one_client_is_built_and_only_log_batch_is_called(self, tmp_path, monkeypatch):
        fake = _install_mlflow(monkeypatch)
        publish_metrics(_sidecar(tmp_path), {"acc": 1.0})
        assert fake.events == ["client", "log_batch"]

    def test_client_is_bound_to_the_sidecar_uri_without_setting_the_global_one(
        self, tmp_path, monkeypatch
    ):
        fake = _install_mlflow(monkeypatch)
        uri = "http://localhost:5999"
        publish_metrics(_sidecar(tmp_path, tracking=[_entry(tracking_uri=uri)]), {"acc": 1.0})
        assert fake.batches[0].tracking_uri == uri

    def test_numeric_types_beyond_float_are_accepted(self, tmp_path, monkeypatch):
        fake = _install_mlflow(monkeypatch)
        metrics = {"a": 3, "b": np.float32(0.25), "c": np.int64(4), "d": Fraction(1, 2)}
        publish_metrics(_sidecar(tmp_path), metrics)
        assert [m.value for m in fake.batches[0].metrics] == [3.0, 0.25, 4.0, 0.5]

    def test_links_metrics_to_the_model_when_the_entry_has_one(self, tmp_path, monkeypatch):
        fake = _install_mlflow(monkeypatch)
        ckpt = _sidecar(tmp_path, tracking=[_entry(model_id="m-abc")])
        publish_metrics(ckpt, {"a": 1.0, "b": 2.0})
        assert [m.model_id for m in fake.batches[0].metrics] == ["m-abc", "m-abc"]

    @pytest.mark.parametrize("tracking", [[_entry()], [_entry(model_id=None)]])
    def test_no_model_id_means_no_linkage(self, tmp_path, monkeypatch, tracking):
        fake = _install_mlflow(monkeypatch)
        publish_metrics(_sidecar(tmp_path, tracking=tracking), {"a": 1.0})
        assert [m.model_id for m in fake.batches[0].metrics] == [None]

    def test_model_id_is_ignored_with_a_debug_note_when_mlflow_cannot_link(
        self, tmp_path, monkeypatch
    ):
        fake = _install_mlflow(monkeypatch, metric_cls=_MetricWithoutModelId)
        notes: list[str] = []
        monkeypatch.setattr(
            publish_mod, "logger", types.SimpleNamespace(debug=lambda msg, *a: notes.append(msg))
        )
        ckpt = _sidecar(tmp_path, tracking=[_entry(model_id="m-abc")])

        publish_metrics(ckpt, {"a": 1.0})

        assert [(m.key, m.model_id) for m in fake.batches[0].metrics] == [("eval/a", None)]
        assert len(notes) == 1 and "model_id" in notes[0]

    def test_server_errors_propagate_unchanged(self, tmp_path, monkeypatch):
        def _down(call):
            raise RuntimeError("server unreachable")

        _install_mlflow(monkeypatch, on_log_batch=_down)
        with pytest.raises(RuntimeError, match="server unreachable") as err:
            publish_metrics(_sidecar(tmp_path), {"a": 1.0})
        assert not isinstance(err.value, PublishRefused)

    def test_refused_when_mlflow_is_not_installed(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "mlflow", None)
        with pytest.raises(PublishRefused, match="uv sync --group mlflow"):
            publish_metrics(_sidecar(tmp_path), {"a": 1.0})

    def test_concurrent_publishers_to_one_run_do_not_share_state(self, tmp_path, monkeypatch):
        barrier = threading.Barrier(2, timeout=10)
        fake = _install_mlflow(monkeypatch, on_log_batch=lambda call: barrier.wait())
        first = _sidecar(
            tmp_path, "step_1", tracking=[_entry(tracking_uri="http://localhost:5001")]
        )
        second = _sidecar(
            tmp_path, "step_2", tracking=[_entry(tracking_uri="http://localhost:5002")]
        )
        failures: list[BaseException] = []

        def _publish(ckpt: Path, prefix: str) -> None:
            try:
                publish_metrics(ckpt, {"acc": 1.0}, prefix=prefix)
            except BaseException as e:  # BrokenBarrierError if the calls never overlapped
                failures.append(e)

        threads = [
            threading.Thread(target=_publish, args=(first, "a/")),
            threading.Thread(target=_publish, args=(second, "b/")),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert failures == []
        seen = sorted((b.tracking_uri, b.metrics[0].key) for b in fake.batches)
        assert seen == [("http://localhost:5001", "a/acc"), ("http://localhost:5002", "b/acc")]


class TestRefusalsPrecedeThePublish:
    """Bad input is refused before any client exists, so nothing is ever half-published."""

    @pytest.mark.parametrize(
        ("metrics", "message"),
        [
            ({}, "no metrics to publish"),
            ([("acc", 1.0)], "metrics must be a mapping of name to number, got list"),
            (None, "metrics must be a mapping of name to number, got NoneType"),
            ({"": 1.0}, "metric name '' must be a non-empty string"),
            ({"  ": 1.0}, "must be a non-empty string"),
            ({3: 1.0}, "metric name 3 must be a non-empty string"),
            ({"acc": "high"}, "metric 'acc' must be a number, got str"),
            ({"acc": None}, "metric 'acc' must be a number, got NoneType"),
            ({"acc": [1.0]}, "metric 'acc' must be a number, got list"),
            ({"acc": True}, "metric 'acc' must be a number, got bool"),
            ({"acc": np.bool_(True)}, "metric 'acc' must be a number, got bool"),
            ({"acc": float("nan")}, "metric 'acc' must be finite, got nan"),
            ({"acc": float("inf")}, "metric 'acc' must be finite, got inf"),
            ({"acc": float("-inf")}, "metric 'acc' must be finite, got -inf"),
        ],
    )
    def test_bad_metrics(self, tmp_path, monkeypatch, metrics, message):
        fake = _install_mlflow(monkeypatch)
        with pytest.raises(PublishRefused, match=message):
            publish_metrics(_sidecar(tmp_path), metrics)
        assert fake.events == []

    @pytest.mark.parametrize("prefix", [None, 5, b"eval/"])
    def test_prefix_must_be_a_string(self, tmp_path, monkeypatch, prefix):
        fake = _install_mlflow(monkeypatch)
        with pytest.raises(PublishRefused, match="prefix must be a string"):
            publish_metrics(_sidecar(tmp_path), {"acc": 1.0}, prefix=prefix)
        assert fake.events == []

    def test_one_bad_metric_blocks_the_whole_batch_and_all_are_named(self, tmp_path, monkeypatch):
        fake = _install_mlflow(monkeypatch)
        metrics = {"good": 1.0, "bad": "x", "worse": float("nan")}
        with pytest.raises(PublishRefused) as err:
            publish_metrics(_sidecar(tmp_path), metrics)
        message = str(err.value)
        assert "'bad'" in message and "'worse'" in message and "'good'" not in message
        assert fake.events == []

    def test_unusable_sidecar_builds_no_client(self, tmp_path, monkeypatch):
        fake = _install_mlflow(monkeypatch)
        with pytest.raises(PublishRefused, match="no 'tracking' block"):
            publish_metrics(_sidecar(tmp_path, tracking=MISSING), {"acc": 1.0})
        assert fake.events == []


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _metrics_file(tmp_path: Path, body: Any = None, raw: str | None = None) -> Path:
    path = tmp_path / "scores.json"
    path.write_text(raw if raw is not None else json.dumps({"mmmu": 0.5} if body is None else body))
    return path


class TestCli:
    def test_publishes_and_reports_run_and_step(self, tmp_path, monkeypatch, capsys):
        fake = _install_mlflow(monkeypatch)
        ckpt = _sidecar(tmp_path)

        code = main([str(ckpt), "--metrics", str(_metrics_file(tmp_path))])

        out = capsys.readouterr()
        assert code == 0
        assert out.out.strip() == "published 1 metric(s) to run run-1 at step 120"
        assert out.err == ""
        assert [(m.key, m.value) for m in fake.batches[0].metrics] == [("eval/mmmu", 0.5)]

    @pytest.mark.parametrize(("flag", "expected"), [("probe/", "probe/mmmu"), ("", "mmmu")])
    def test_prefix_flag(self, tmp_path, monkeypatch, flag, expected):
        fake = _install_mlflow(monkeypatch)
        ckpt = _sidecar(tmp_path)
        assert main([str(ckpt), "--metrics", str(_metrics_file(tmp_path)), "--prefix", flag]) == 0
        assert [m.key for m in fake.batches[0].metrics] == [expected]

    def test_refusal_exits_2_and_names_the_problem_on_stderr(self, tmp_path, monkeypatch, capsys):
        fake = _install_mlflow(monkeypatch)
        ckpt = _sidecar(tmp_path, tracking=MISSING)

        code = main([str(ckpt), "--metrics", str(_metrics_file(tmp_path))])

        out = capsys.readouterr()
        assert code == 2
        assert out.out == ""
        assert out.err.startswith("refused: ") and "no 'tracking' block" in out.err
        assert fake.events == []

    def test_missing_metrics_file(self, tmp_path, monkeypatch, capsys):
        _install_mlflow(monkeypatch)
        code = main([str(_sidecar(tmp_path)), "--metrics", str(tmp_path / "absent.json")])
        assert code == 2
        assert "cannot read metrics file" in capsys.readouterr().err

    def test_metrics_file_that_is_not_json(self, tmp_path, monkeypatch, capsys):
        _install_mlflow(monkeypatch)
        metrics = _metrics_file(tmp_path, raw="{ nope")
        assert main([str(_sidecar(tmp_path)), "--metrics", str(metrics)]) == 2
        assert "cannot read metrics file" in capsys.readouterr().err

    def test_metrics_file_that_is_not_an_object(self, tmp_path, monkeypatch, capsys):
        fake = _install_mlflow(monkeypatch)
        metrics = _metrics_file(tmp_path, body=[0.5])
        assert main([str(_sidecar(tmp_path)), "--metrics", str(metrics)]) == 2
        assert "metrics must be a mapping of name to number, got list" in capsys.readouterr().err
        assert fake.events == []

    def test_non_finite_literal_in_the_metrics_file_is_refused(self, tmp_path, monkeypatch, capsys):
        fake = _install_mlflow(monkeypatch)
        metrics = _metrics_file(tmp_path, raw='{"mmmu": NaN}')
        assert main([str(_sidecar(tmp_path)), "--metrics", str(metrics)]) == 2
        assert "must be finite" in capsys.readouterr().err
        assert fake.events == []

    def test_metrics_flag_is_required(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as err:
            main([str(_sidecar(tmp_path))])
        assert err.value.code == 2
        assert "--metrics" in capsys.readouterr().err

    def test_help_lists_the_arguments(self, capsys):
        with pytest.raises(SystemExit) as err:
            main(["--help"])
        assert err.value.code == 0
        text = capsys.readouterr().out
        assert "checkpoint_dir" in text and "--metrics" in text and "--prefix" in text

    def test_python_dash_m_exits_with_the_status_of_main(self, tmp_path, monkeypatch, capsys):
        _install_mlflow(monkeypatch)
        argv = ["publish", str(_sidecar(tmp_path)), "--metrics", str(_metrics_file(tmp_path))]
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.delitem(sys.modules, "kempnerforge.metrics.publish")
        with pytest.raises(SystemExit) as err:
            runpy.run_module("kempnerforge.metrics.publish", run_name="__main__")
        assert err.value.code == 0
        assert "published 1 metric(s)" in capsys.readouterr().out
