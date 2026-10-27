"""Tests for the MLflow eval publisher: synthetic sidecars and a fake mlflow, no server."""

from __future__ import annotations

import json
import os
import runpy
import sys
import threading
import types
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import kempnerforge.metrics.mlflow.publish as publish_mod
from kempnerforge.checkpoint import manager as checkpoint_manager
from kempnerforge.metrics.mlflow.publish import (
    MlflowTarget,
    PublishRefused,
    main,
    publish_metrics,
    read_mlflow_target,
)

URI = "tracking-uri-placeholder"
MISSING = object()  # drop the key from the synthetic sidecar


def _drop_missing(mapping: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in mapping.items() if v is not MISSING}


@pytest.fixture(autouse=True)
def _no_databricks_auth_env(monkeypatch):
    """Start without Databricks auth variables; whatever the publisher exports is undone."""
    for name in ("DATABRICKS_CONFIG_PROFILE", "DATABRICKS_TOKEN", "DATABRICKS_API_TOKEN"):
        monkeypatch.setenv(name, "")  # registers the restore even when the variable was unset
        monkeypatch.delenv(name)


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


def _raiser(error: Exception):
    def _raise(*args: Any, **kwargs: Any) -> None:
        raise error

    return _raise


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


_MODEL_CALLS = (
    "get_run",
    "search_logged_models",
    "create_logged_model",
    "log_model_artifacts",
    "finalize_logged_model",
)


class _FakeMlflow(types.ModuleType):
    """Stand-in `mlflow` recording what a publisher does.

    Only ``MlflowClient(tracking_uri=...)``, ``log_batch`` and ``get_active_model_id``
    exist, plus the logged-model calls of ``models`` (a FakeModelStore) when one is given:
    touching anything else (``start_run``, ``set_tracking_uri``, ``end_run``, ...) raises
    AssertionError. ``log_batch`` fills a missing metric model_id from the active
    model, as mlflow's client does. ``reject`` fails a request before the fake stores it
    (a real store may keep part of a failed batch), ``on_log_batch`` runs after a request
    is stored, and ``init_error`` fails the client.
    """

    def __init__(
        self,
        metric_cls: type = _Metric,
        on_log_batch=None,
        reject=None,
        init_error: Exception | None = None,
        active_model_id: str | None = None,
        models=None,
    ):
        super().__init__("mlflow")
        self.active_model_id = active_model_id
        self.events: list[str] = []
        self.batches: list[types.SimpleNamespace] = []
        entities = types.ModuleType("mlflow.entities")
        entities.Metric = metric_cls
        self.entities_module = entities
        fake = self

        class MlflowClient:
            def __init__(self, tracking_uri=None, registry_uri=None):
                if init_error is not None:
                    raise init_error
                fake.events.append("client")
                self._uri = tracking_uri

            def log_batch(self, run_id, metrics=(), params=(), tags=(), synchronous=None):
                fake.events.append("log_batch")
                for m in metrics:  # mlflow: model_id=metric.model_id or get_active_model_id()
                    if getattr(m, "model_id", None) is None:
                        m.model_id = fake.active_model_id
                if reject is not None:
                    reject(list(metrics))
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
                if models is not None and name in _MODEL_CALLS:
                    fake.events.append(name)
                    return getattr(models.client(self._uri), name)
                raise AssertionError(f"publisher touched MlflowClient.{name}")

        self.MlflowClient = MlflowClient

    def get_active_model_id(self):
        return self.active_model_id

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
        target = read_mlflow_target(_sidecar(tmp_path))
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

    def test_reads_the_model_name_when_the_entry_has_one(self, tmp_path):
        assert read_mlflow_target(_sidecar(tmp_path)).model_name is None
        named = _sidecar(tmp_path, "step_121", tracking=[_entry(model_name="run-step_121")])
        assert read_mlflow_target(named).model_name == "run-step_121"
        unnamed = _sidecar(tmp_path, "step_122", tracking=[_entry(model_name=None)])
        assert read_mlflow_target(unnamed).model_name is None

    def test_profile_uri_selects_the_profile_for_artifact_clients(self, tmp_path):
        """Artifact clients read only the environment, so the profile in a
        databricks://<profile> URI must land in DATABRICKS_CONFIG_PROFILE."""
        ckpt = _sidecar(tmp_path, tracking=[_entry(tracking_uri="databricks://prof-a")])
        read_mlflow_target(ckpt)
        assert os.environ["DATABRICKS_CONFIG_PROFILE"] == "prof-a"

    def test_profile_export_never_overrides_the_operator(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "operator")
        monkeypatch.setenv("DATABRICKS_API_TOKEN", "alias-token")
        ckpt = _sidecar(tmp_path, tracking=[_entry(tracking_uri="databricks://prof-a")])
        read_mlflow_target(ckpt)
        assert os.environ["DATABRICKS_CONFIG_PROFILE"] == "operator"
        assert os.environ["DATABRICKS_TOKEN"] == "alias-token"  # auth preparation ran

    @pytest.mark.parametrize("uri", ["databricks://", "databricks://scope:prefix"])
    def test_non_profile_uris_select_nothing(self, tmp_path, monkeypatch, uri):
        monkeypatch.setenv("DATABRICKS_API_TOKEN", "alias-token")
        read_mlflow_target(_sidecar(tmp_path, tracking=[_entry(tracking_uri=uri)]))
        assert "DATABRICKS_CONFIG_PROFILE" not in os.environ
        assert os.environ["DATABRICKS_TOKEN"] == "alias-token"  # auth preparation ran

    @pytest.mark.parametrize(
        ("uri", "profile"),
        [("databricks://prof-a/", "prof-a"), ("databricks://prof-a/x:y", "prof-a")],
    )
    def test_profile_is_parsed_like_the_training_backend(self, tmp_path, uri, profile):
        """Training takes the first path segment as the profile; publishing must agree."""
        read_mlflow_target(_sidecar(tmp_path, tracking=[_entry(tracking_uri=uri)]))
        assert os.environ["DATABRICKS_CONFIG_PROFILE"] == profile

    @pytest.mark.parametrize(
        "uri",
        [
            "databricks://profile-a",
            "databricks://profile-a/",
            "databricks://profile-a/x:y",
            "databricks",
            "databricks://",
            "databricks://scope:prefix",
            "databricks-uc://profile-a",
            "http://localhost:5000",
        ],
    )
    def test_profile_matches_the_training_parser(self, tmp_path, uri):
        """The publisher exports exactly the profile the trainer's parser names, for every URI."""
        from kempnerforge.metrics.mlflow.auth import _databricks_profile

        read_mlflow_target(_sidecar(tmp_path, tracking=[_entry(tracking_uri=uri)]))
        assert os.environ.get("DATABRICKS_CONFIG_PROFILE") == _databricks_profile(uri)

    def test_token_alias_stands_in_like_the_training_backend(self, tmp_path, monkeypatch):
        """A DATABRICKS_API_TOKEN that lets training authenticate must let publishing too."""
        monkeypatch.setenv("DATABRICKS_API_TOKEN", "alias-token")
        ckpt = _sidecar(tmp_path, tracking=[_entry(tracking_uri="databricks://prof-a")])
        read_mlflow_target(ckpt)
        assert os.environ["DATABRICKS_TOKEN"] == "alias-token"

    def test_token_alias_never_overrides_databricks_token(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATABRICKS_TOKEN", "primary")
        monkeypatch.setenv("DATABRICKS_API_TOKEN", "alias-token")
        read_mlflow_target(_sidecar(tmp_path, tracking=[_entry(tracking_uri="databricks://p")]))
        assert os.environ["DATABRICKS_TOKEN"] == "primary"
        assert os.environ["DATABRICKS_CONFIG_PROFILE"] == "p"  # auth preparation ran

    def test_auth_is_prepared_only_for_databricks_uris(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATABRICKS_API_TOKEN", "alias-token")
        off = _sidecar(tmp_path, name="off", tracking=[_entry(tracking_uri="http://h:5000/x")])
        read_mlflow_target(off)
        assert "DATABRICKS_TOKEN" not in os.environ
        assert "DATABRICKS_CONFIG_PROFILE" not in os.environ
        read_mlflow_target(
            _sidecar(tmp_path, name="on", tracking=[_entry(tracking_uri="databricks")])
        )
        assert os.environ["DATABRICKS_TOKEN"] == "alias-token"

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
        with pytest.raises(PublishRefused, match="malformed JSON"):
            read_mlflow_target(ckpt)

    def test_unreadable_sidecar(self, tmp_path, monkeypatch):
        ckpt = _sidecar(tmp_path)
        monkeypatch.setattr(Path, "read_bytes", _raiser(PermissionError("denied")))
        with pytest.raises(PublishRefused, match="unreadable or malformed JSON: denied"):
            read_mlflow_target(ckpt)

    @pytest.mark.parametrize(
        ("raw", "key"),
        [
            ('{"step": 1, "step": 2, "tracking": []}', "step"),
            (
                '{"step": 1, "tracking": [{"backend": "mlflow", "run_id": "a", "run_id": "b"}]}',
                "run_id",
            ),
        ],
    )
    def test_repeated_key_is_refused_instead_of_the_last_one_winning(self, tmp_path, raw, key):
        ckpt = tmp_path / "step_1"
        ckpt.mkdir()
        (ckpt / "metadata.json").write_text(raw)
        with pytest.raises(PublishRefused, match=f"duplicate key '{key}'"):
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
    @pytest.mark.parametrize("field", ["model_id", "model_name"])
    def test_model_fields_when_present_must_be_non_empty_strings(
        self, tmp_path, field, value, message
    ):
        ckpt = _sidecar(tmp_path, tracking=[_entry(**{field: value})])
        with pytest.raises(PublishRefused, match=rf"'{field}' {message}"):
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
        marker = "tracking-uri-marker-not-to-be-echoed"
        ckpt = _sidecar(tmp_path, tracking=[_entry(tracking_uri=marker, run_id=MISSING)])
        with pytest.raises(PublishRefused) as err:
            read_mlflow_target(ckpt)
        assert marker not in str(err.value)


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------


class TestPublishMetrics:
    def test_appends_prefixed_metrics_at_the_sidecar_step(self, tmp_path, monkeypatch):
        fake = _install_mlflow(monkeypatch)
        ckpt = _sidecar(tmp_path, step=1500)

        target = publish_metrics(ckpt, {"mmmu": 0.5, "mmbench": 2})

        (batch,) = fake.batches
        assert batch.run_id == "run-1"
        assert batch.synchronous is True
        assert batch.params == [] and batch.tags == []
        assert [(m.key, m.value, m.step) for m in batch.metrics] == [
            ("eval/mmmu", 0.5, 1500),
            ("eval/mmbench", 2.0, 1500),
        ]
        assert all(isinstance(m.value, float) for m in batch.metrics)
        assert all(isinstance(m.timestamp, int) and m.timestamp > 0 for m in batch.metrics)
        assert target == read_mlflow_target(ckpt)

    def test_every_metric_carries_the_one_publish_time(self, tmp_path, monkeypatch):
        fake = _install_mlflow(monkeypatch)
        monkeypatch.setattr(publish_mod, "_now_ms", lambda: 1_700_000_000_123)
        publish_metrics(_sidecar(tmp_path), {"a": 1.0, "b": 2.0})
        assert [m.timestamp for m in fake.batches[0].metrics] == [1_700_000_000_123] * 2

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
        uri = "tracking-uri-bound"
        publish_metrics(_sidecar(tmp_path, tracking=[_entry(tracking_uri=uri)]), {"acc": 1.0})
        assert fake.batches[0].tracking_uri == uri

    def test_numeric_types_beyond_float_are_accepted(self, tmp_path, monkeypatch):
        fake = _install_mlflow(monkeypatch)
        metrics = {"a": 3, "b": np.float32(0.25), "c": np.int64(4), "d": Fraction(1, 2)}
        publish_metrics(_sidecar(tmp_path), metrics)
        assert [m.value for m in fake.batches[0].metrics] == [3.0, 0.25, 4.0, 0.5]

    def test_the_request_limit_is_one_batch(self, tmp_path, monkeypatch):
        fake = _install_mlflow(monkeypatch)
        metrics = {f"m{i}": float(i) for i in range(publish_mod._MAX_METRICS_PER_CALL)}
        publish_metrics(_sidecar(tmp_path), metrics)
        assert [len(b.metrics) for b in fake.batches] == [1000]

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

    def test_ambient_active_model_refused_when_sidecar_has_no_model_id(self, tmp_path, monkeypatch):
        """log_batch fills model_id-less metrics with the process's active model; the
        publisher refuses rather than mislink them to an unrelated model."""
        fake = _install_mlflow(monkeypatch, active_model_id="m-ambient")
        ckpt = _sidecar(tmp_path, tracking=[_entry(model_id=None)])
        with pytest.raises(PublishRefused, match="m-ambient"):
            publish_metrics(ckpt, {"a": 1.0})
        assert fake.events == []  # refused before any client was built

    def test_ambient_active_model_is_overridden_by_the_sidecar_model_id(
        self, tmp_path, monkeypatch
    ):
        fake = _install_mlflow(monkeypatch, active_model_id="m-ambient")
        ckpt = _sidecar(tmp_path, tracking=[_entry(model_id="m-abc")])
        publish_metrics(ckpt, {"a": 1.0})
        assert [m.model_id for m in fake.batches[0].metrics] == ["m-abc"]

    def test_ambient_active_model_refused_when_mlflow_cannot_link_the_sidecar_model_id(
        self, tmp_path, monkeypatch
    ):
        fake = _install_mlflow(
            monkeypatch, metric_cls=_MetricWithoutModelId, active_model_id="m-ambient"
        )
        ckpt = _sidecar(tmp_path, tracking=[_entry(model_id="m-abc")])
        with pytest.raises(PublishRefused, match="m-ambient"):
            publish_metrics(ckpt, {"a": 1.0})
        assert fake.events == [] and fake.batches == []

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
        _install_mlflow(monkeypatch, on_log_batch=_raiser(RuntimeError("server unreachable")))
        with pytest.raises(RuntimeError, match="server unreachable") as err:
            publish_metrics(_sidecar(tmp_path), {"a": 1.0})
        assert not isinstance(err.value, PublishRefused)

    def test_a_rejected_request_propagates_the_stores_error(self, tmp_path, monkeypatch):
        def reject(metrics):
            if any("?" in m.key for m in metrics):
                raise ValueError("invalid metric name")

        fake = _install_mlflow(monkeypatch, reject=reject)
        with pytest.raises(ValueError, match="invalid metric name"):
            publish_metrics(_sidecar(tmp_path), {"ok": 1.0, "bad?": 2.0})
        assert fake.events == ["client", "log_batch"] and fake.batches == []

    def test_refused_when_mlflow_is_not_installed(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "mlflow", None)
        with pytest.raises(PublishRefused, match="the environment is incomplete, run: uv sync$"):
            publish_metrics(_sidecar(tmp_path), {"a": 1.0})

    def test_concurrent_publishers_to_one_run_do_not_share_state(self, tmp_path, monkeypatch):
        barrier = threading.Barrier(2, timeout=10)
        fake = _install_mlflow(monkeypatch, on_log_batch=lambda call: barrier.wait())
        first = _sidecar(tmp_path, "step_1", tracking=[_entry(tracking_uri="tracking-uri-1")])
        second = _sidecar(tmp_path, "step_2", tracking=[_entry(tracking_uri="tracking-uri-2")])
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
        assert seen == [("tracking-uri-1", "a/acc"), ("tracking-uri-2", "b/acc")]


class TestPublishLinksTheCheckpointModel:
    """A sidecar that names its model but has no model_id: the publisher finds or makes it."""

    NAME = "run-a-step_120"

    @staticmethod
    def _install(monkeypatch, model_store, **kwargs):
        model_store.runs["run-1"] = "7"
        return _install_mlflow(monkeypatch, models=model_store, **kwargs)

    def _named(self, tmp_path, **entry):
        return _sidecar(tmp_path, tracking=[_entry(model_name=self.NAME, **entry)])

    def test_links_the_runs_model_of_that_name_without_creating_one(
        self, tmp_path, monkeypatch, model_store
    ):
        fake = self._install(monkeypatch, model_store)
        found = model_store.add(self.NAME, run_id="run-1")
        target = publish_metrics(self._named(tmp_path), {"a": 1.0, "b": 2.0})
        assert [m.model_id for m in fake.batches[0].metrics] == [found.model_id] * 2
        assert target.model_id == found.model_id
        assert fake.events == ["client", "get_run", "search_logged_models", "log_batch"]
        assert len(model_store.models) == 1

    def test_creates_the_model_when_the_run_has_none_and_links_it(
        self, tmp_path, monkeypatch, model_store
    ):
        fake = self._install(monkeypatch, model_store, active_model_id="m-ambient")
        ckpt = self._named(tmp_path)
        sidecar = (ckpt / "metadata.json").read_bytes()
        monkeypatch.chdir(tmp_path)
        target = publish_metrics(ckpt.name, {"a": 1.0})
        (model,) = model_store.models
        assert [m.model_id for m in fake.batches[0].metrics] == [model.model_id]
        assert target.model_id == model.model_id
        assert (model.name, model.source_run_id, model.status) == (self.NAME, "run-1", "READY")
        assert model.params == {
            "step": "120",
            "checkpoint_dir": str((tmp_path / "step_120").resolve()),
        }
        assert fake.events.count("client") == 1
        assert {uri for _, uri, _ in model_store.calls} == {URI}
        assert (ckpt / "metadata.json").read_bytes() == sidecar

    def test_a_model_made_through_a_symlink_records_the_directory_it_points_at(
        self, tmp_path, monkeypatch, model_store
    ):
        self._install(monkeypatch, model_store)
        ckpt = self._named(tmp_path)
        latest = tmp_path / "latest"
        latest.symlink_to(ckpt.name)
        publish_metrics(latest, {"a": 1.0})
        (model,) = model_store.models
        assert model.params["checkpoint_dir"] == str(ckpt.resolve())

    def test_a_model_id_wins_over_the_model_name(self, tmp_path, monkeypatch, model_store):
        fake = self._install(monkeypatch, model_store)
        target = publish_metrics(self._named(tmp_path, model_id="m-abc"), {"a": 1.0})
        assert [m.model_id for m in fake.batches[0].metrics] == ["m-abc"]
        assert target.model_id == "m-abc" and model_store.calls == []

    def test_without_a_model_name_nothing_is_searched_or_created(
        self, tmp_path, monkeypatch, model_store
    ):
        fake = self._install(monkeypatch, model_store)
        target = publish_metrics(_sidecar(tmp_path), {"a": 1.0})
        assert fake.events == ["client", "log_batch"] and model_store.calls == []
        assert [m.model_id for m in fake.batches[0].metrics] == [None]
        assert target.model_id is None

    @pytest.mark.parametrize("model_id", [None, "m-abc"])
    def test_an_mlflow_that_cannot_link_ignores_the_model(
        self, tmp_path, monkeypatch, model_store, model_id
    ):
        fake = self._install(monkeypatch, model_store, metric_cls=_MetricWithoutModelId)
        target = publish_metrics(self._named(tmp_path, model_id=model_id), {"a": 1.0})
        assert fake.events == ["client", "log_batch"] and model_store.calls == []
        assert [m.model_id for m in fake.batches[0].metrics] == [None]
        assert target.model_id is None

    def test_a_server_error_finding_the_model_propagates_before_any_metric_is_sent(
        self, tmp_path, monkeypatch, model_store
    ):
        fake = self._install(monkeypatch, model_store)
        model_store.fail("search_logged_models", RuntimeError("server unreachable"))
        with pytest.raises(RuntimeError, match="server unreachable") as err:
            publish_metrics(self._named(tmp_path), {"a": 1.0})
        assert not isinstance(err.value, PublishRefused)
        assert fake.batches == []


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
            ({"huge": 10**1000}, "metric 'huge' is too large for a float"),
        ],
    )
    def test_bad_metrics(self, tmp_path, monkeypatch, metrics, message):
        fake = _install_mlflow(monkeypatch)
        with pytest.raises(PublishRefused, match=message):
            publish_metrics(_sidecar(tmp_path), metrics)
        assert fake.events == []

    def test_more_metrics_than_one_request_carries(self, tmp_path, monkeypatch):
        fake = _install_mlflow(monkeypatch)
        metrics = {f"m{i}": 1.0 for i in range(publish_mod._MAX_METRICS_PER_CALL + 1)}
        with pytest.raises(PublishRefused, match="1001 metrics exceed the 1000 one request"):
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

    def test_metrics_file_with_a_repeated_name(self, tmp_path, monkeypatch, capsys):
        fake = _install_mlflow(monkeypatch)
        metrics = _metrics_file(tmp_path, raw='{"acc": 0.5, "acc": 0.9}')
        assert main([str(_sidecar(tmp_path)), "--metrics", str(metrics)]) == 2
        assert "duplicate key 'acc'" in capsys.readouterr().err
        assert fake.events == []

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

    @pytest.mark.parametrize("failing_step", ["client", "log_batch"])
    @pytest.mark.parametrize(
        ("raised", "shown"),
        [
            (
                "cannot reach placeholder://user:secret-marker@tracking-placeholder/",
                "cannot reach placeholder://<redacted>",
            ),
            (
                "unsupported URI 'foo://tracking-placeholder/x?token=token-marker', retry",
                "unsupported URI 'foo://<redacted>', retry",
            ),
        ],
    )
    def test_mlflow_failure_exits_1_without_traceback_or_urls(
        self, tmp_path, monkeypatch, capsys, failing_step, raised, shown
    ):
        error = RuntimeError(raised)
        hook = {"init_error": error} if failing_step == "client" else {"reject": _raiser(error)}
        _install_mlflow(monkeypatch, **hook)

        code = main([str(_sidecar(tmp_path)), "--metrics", str(_metrics_file(tmp_path))])

        out = capsys.readouterr()
        assert code == 1
        assert out.out == ""
        assert out.err == f"failed: RuntimeError: {shown}\n"

    def test_refusal_text_never_shows_a_url(self, tmp_path, monkeypatch, capsys):
        _install_mlflow(monkeypatch)
        key = "placeholder://user:secret-marker@tracking-placeholder/"
        ckpt = tmp_path / "step_1"
        ckpt.mkdir()
        (ckpt / "metadata.json").write_text(f'{{"step": 1, "{key}": 1, "{key}": 2}}')

        code = main([str(ckpt), "--metrics", str(_metrics_file(tmp_path))])

        err = capsys.readouterr().err
        assert code == 2
        assert "duplicate key 'placeholder://<redacted>'" in err and "secret-marker" not in err

    def test_mlflow_failure_text_without_credentials_is_reported_as_is(
        self, tmp_path, monkeypatch, capsys
    ):
        _install_mlflow(monkeypatch, reject=_raiser(RuntimeError("run not found")))
        code = main([str(_sidecar(tmp_path)), "--metrics", str(_metrics_file(tmp_path))])
        assert code == 1
        assert capsys.readouterr().err == "failed: RuntimeError: run not found\n"

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
        monkeypatch.delitem(sys.modules, "kempnerforge.metrics.mlflow.publish")
        with pytest.raises(SystemExit) as err:
            runpy.run_module("kempnerforge.metrics.mlflow.publish", run_name="__main__")
        assert err.value.code == 0
        assert "published 1 metric(s)" in capsys.readouterr().out
