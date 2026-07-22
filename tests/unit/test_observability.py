"""Unit tests for KempnerForge observability modules (tracker, memory monitor, backends)."""

from __future__ import annotations

import logging
import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist

import kempnerforge.metrics.logger as log_mod
import kempnerforge.metrics.tracker as tracker_mod
from kempnerforge.config.schema import (
    DataConfig,
    DatasetSource,
    JobConfig,
    MetricsConfig,
    ModelConfig,
)
from kempnerforge.metrics.logger import (
    _format_number,
    _RankFilter,
    _RankFormatter,
    _supports_color,
    format_metrics,
    get_logger,
)
from kempnerforge.metrics.memory import (
    DeviceMemoryMonitor,
    get_memory_stats,
    get_memory_utilization,
    reset_peak_memory,
)
from kempnerforge.metrics.tracker import (
    MetricsTracker,
    MLflowBackend,
    StepMetrics,
    TensorBoardBackend,
    WandBBackend,
    _flatten_config_params,
    _resolve_mlflow_experiment,
)

# ---------------------------------------------------------------------------
# MetricsTracker
# ---------------------------------------------------------------------------


class TestMetricsTracker:
    def _make_tracker(self, log_interval: int = 1) -> MetricsTracker:
        config = JobConfig(
            model=ModelConfig(dim=128, n_layers=2, n_heads=2, vocab_size=256),
            metrics=MetricsConfig(log_interval=log_interval),
        )
        return MetricsTracker(config, num_gpus=1)

    def test_end_step_returns_metrics_on_log_step(self):
        tracker = self._make_tracker(log_interval=1)
        tracker.start_step()
        result = tracker.end_step(step=1, loss=2.5, grad_norm=1.0, lr=3e-4, tokens_in_step=1024)
        assert result is not None
        assert isinstance(result, StepMetrics)
        assert result.loss == 2.5

    def test_end_step_returns_none_on_non_log_step(self):
        tracker = self._make_tracker(log_interval=10)
        tracker.start_step()
        result = tracker.end_step(step=3, loss=2.5, grad_norm=1.0, lr=3e-4, tokens_in_step=1024)
        assert result is None

    def test_step_1_always_logs(self):
        tracker = self._make_tracker(log_interval=100)
        tracker.start_step()
        result = tracker.end_step(step=1, loss=2.5, grad_norm=1.0, lr=3e-4, tokens_in_step=1024)
        assert result is not None

    def test_metrics_fields(self):
        tracker = self._make_tracker()
        tracker.start_step()
        result = tracker.end_step(step=1, loss=3.0, grad_norm=0.5, lr=1e-4, tokens_in_step=2048)
        assert result.loss == 3.0
        assert result.grad_norm == 0.5
        assert result.lr == 1e-4
        assert result.tokens_per_sec > 0
        assert result.step_time_sec > 0

    def test_mfu_computed(self):
        tracker = self._make_tracker()
        tracker.start_step()
        result = tracker.end_step(step=1, loss=2.0, grad_norm=1.0, lr=3e-4, tokens_in_step=100000)
        # MFU is computed (may be > 1 in tests due to near-zero step time)
        assert result.mfu > 0.0

    def test_smoothed_metrics_updated(self):
        tracker = self._make_tracker()
        for i in range(5):
            tracker.start_step()
            tracker.end_step(
                step=i + 1, loss=2.0 - i * 0.1, grad_norm=1.0, lr=3e-4, tokens_in_step=1024
            )
        assert "loss" in tracker._smoothed
        assert "tokens_per_sec" in tracker._smoothed

    def test_close_without_backends(self):
        tracker = self._make_tracker()
        tracker.close()  # Should not raise

    def test_init_backends_rank_zero_appends_wandb(self, monkeypatch):
        """When rank-0 and enable_wandb=True, init_backends appends a WandBBackend."""
        monkeypatch.setattr(tracker_mod, "WandBBackend", MagicMock(name="FakeWandB"))
        config = JobConfig(
            model=ModelConfig(dim=128, n_layers=2, n_heads=2, vocab_size=256),
            metrics=MetricsConfig(enable_wandb=True),
        )
        tracker = MetricsTracker(config, num_gpus=1)
        tracker.init_backends(config)
        assert len(tracker._backends) == 1

    def test_init_backends_rank_zero_appends_tensorboard(self, monkeypatch):
        """When rank-0 and enable_tensorboard=True, init_backends appends a TBBackend."""
        monkeypatch.setattr(tracker_mod, "TensorBoardBackend", MagicMock(name="FakeTB"))
        config = JobConfig(
            model=ModelConfig(dim=128, n_layers=2, n_heads=2, vocab_size=256),
            metrics=MetricsConfig(enable_tensorboard=True),
        )
        tracker = MetricsTracker(config, num_gpus=1)
        tracker.init_backends(config)
        assert len(tracker._backends) == 1

    def test_init_backends_rank_zero_appends_mlflow(self, monkeypatch):
        """When rank-0 and enable_mlflow=True, init_backends appends an MLflowBackend."""
        monkeypatch.setattr(tracker_mod, "MLflowBackend", MagicMock(name="FakeMLflow"))
        config = JobConfig(
            model=ModelConfig(dim=128, n_layers=2, n_heads=2, vocab_size=256),
            metrics=MetricsConfig(enable_mlflow=True),
        )
        tracker = MetricsTracker(config, num_gpus=1)
        tracker.init_backends(config)
        assert len(tracker._backends) == 1

    def test_init_backends_skips_non_rank_zero(self, monkeypatch):
        """Non-rank-0 ranks must not initialize backends even if enabled."""
        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_rank", lambda: 1)
        config = JobConfig(
            model=ModelConfig(dim=128, n_layers=2, n_heads=2, vocab_size=256),
            metrics=MetricsConfig(enable_wandb=True),
        )
        tracker = MetricsTracker(config, num_gpus=1)
        tracker.init_backends(config)
        assert tracker._backends == []

    def test_init_backends_idempotent(self, monkeypatch):
        """Calling init_backends twice must not double-append backends."""
        fake = MagicMock(name="FakeWandB")
        monkeypatch.setattr(tracker_mod, "WandBBackend", fake)
        config = JobConfig(
            model=ModelConfig(dim=128, n_layers=2, n_heads=2, vocab_size=256),
            metrics=MetricsConfig(enable_wandb=True),
        )
        tracker = MetricsTracker(config, num_gpus=1)
        tracker.init_backends(config)
        tracker.init_backends(config)  # second call is a no-op
        assert fake.call_count == 1

    def test_end_step_dispatches_to_backend(self):
        """end_step must forward the metrics dict to every registered backend."""
        tracker = self._make_tracker(log_interval=1)
        fake = _FakeBackend()
        tracker._backends.append(fake)
        tracker.start_step()
        tracker.end_step(step=1, loss=2.5, grad_norm=1.0, lr=3e-4, tokens_in_step=1024)
        assert len(fake.log_calls) == 1
        metrics_dict, step = fake.log_calls[0]
        assert step == 1
        assert "train/loss" in metrics_dict

    def test_log_eval_dispatches_to_backends(self):
        """log_eval must forward the metrics dict verbatim to every backend."""
        tracker = self._make_tracker()
        fake = _FakeBackend()
        tracker._backends.append(fake)
        tracker.log_eval({"eval/loss": 2.3}, step=10)
        assert fake.log_calls == [({"eval/loss": 2.3}, 10)]

    def test_close_with_backends(self):
        """tracker.close() must call close() on every registered backend."""
        tracker = self._make_tracker()
        fake = _FakeBackend()
        tracker._backends.append(fake)
        tracker.close()
        assert fake.close_calls == 1


class _FakeBackend:
    """Recording backend used by tracker dispatch tests."""

    def __init__(self) -> None:
        self.log_calls: list[tuple[dict, int]] = []
        self.close_calls = 0

    def log(self, metrics: dict, step: int) -> None:
        self.log_calls.append((metrics, step))

    def close(self) -> None:
        self.close_calls += 1


# ---------------------------------------------------------------------------
# StepMetrics
# ---------------------------------------------------------------------------


class TestStepMetrics:
    def test_defaults(self):
        m = StepMetrics()
        assert m.loss == 0.0
        assert m.mfu == 0.0
        assert m.mem_utilization == 0.0


# ---------------------------------------------------------------------------
# DeviceMemoryMonitor
# ---------------------------------------------------------------------------


class TestDeviceMemoryMonitor:
    def test_report_returns_stats(self):
        monitor = DeviceMemoryMonitor()
        stats = monitor.report(step=1)
        assert "allocated_gb" in stats
        assert "peak_gb" in stats
        assert "mem_utilization" in stats

    def test_report_on_gpu(self):
        if not torch.cuda.is_available():
            return
        monitor = DeviceMemoryMonitor()
        stats = monitor.report(step=1)
        assert stats["total_gb"] > 0
        assert 0.0 <= stats["mem_utilization"] <= 1.0

    def test_snapshot_not_triggered_by_default(self):
        monitor = DeviceMemoryMonitor(snapshot_step=None)
        monitor.report(step=5)
        assert not monitor._snapshot_taken

    def test_snapshot_triggered_at_step(self, tmp_path):
        if not torch.cuda.is_available():
            return
        monitor = DeviceMemoryMonitor(snapshot_step=3, snapshot_dir=str(tmp_path / "snapshots"))
        monitor.report(step=1)
        assert not monitor._snapshot_taken
        monitor.report(step=3)
        assert monitor._snapshot_taken

    def test_snapshot_only_once(self, tmp_path):
        if not torch.cuda.is_available():
            return
        monitor = DeviceMemoryMonitor(snapshot_step=3, snapshot_dir=str(tmp_path / "snapshots"))
        monitor.report(step=3)
        assert monitor._snapshot_taken
        # Second report at same step should not re-snapshot
        monitor.report(step=3)
        assert monitor._snapshot_taken

    def test_capture_snapshot_cpu_only(self, monkeypatch):
        """Without CUDA, capture_snapshot returns None immediately."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monitor = DeviceMemoryMonitor()
        assert monitor.capture_snapshot(step=10) is None

    def test_capture_snapshot_handles_exception(self, monkeypatch, tmp_path):
        """Any exception inside capture_snapshot is swallowed; returns None."""
        # Bypass the CPU-only early return so the try-block runs.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated _record_memory_history failure")

        monkeypatch.setattr(torch.cuda.memory, "_record_memory_history", _boom)
        monitor = DeviceMemoryMonitor(snapshot_dir=str(tmp_path))
        assert monitor.capture_snapshot(step=1) is None


class TestMemoryHelpers:
    def test_get_memory_stats_cpu_only(self, monkeypatch):
        """Without CUDA, get_memory_stats returns all-zero values."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert get_memory_stats() == {
            "allocated_gb": 0,
            "peak_gb": 0,
            "reserved_gb": 0,
            "total_gb": 0,
        }

    def test_get_memory_utilization_zero_total(self, monkeypatch):
        """When total_gb == 0 (no GPU), utilization is 0.0 to avoid div-by-zero."""
        monkeypatch.setattr(
            "kempnerforge.metrics.memory.get_memory_stats",
            lambda d=0: {"allocated_gb": 0, "peak_gb": 5, "reserved_gb": 0, "total_gb": 0},
        )
        assert get_memory_utilization() == 0.0

    def test_reset_peak_memory_cpu_only(self, monkeypatch):
        """Without CUDA, reset_peak_memory is a no-op (must not raise)."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        reset_peak_memory(device=0)  # Should not raise


# ---------------------------------------------------------------------------
# Logging backends
# ---------------------------------------------------------------------------


class TestWandBBackend:
    def test_init_no_crash(self):
        config = MetricsConfig(enable_wandb=True)
        backend = WandBBackend(config)
        assert backend._run is None

    def test_log_without_init_triggers_init(self):
        config = MetricsConfig(enable_wandb=True)
        backend = WandBBackend(config)
        # This will try to import wandb — if not installed, sets sentinel
        backend.log({"test": 1.0}, step=1)
        # Either initialized or set to False sentinel
        assert backend._run is not None

    def test_wandb_handles_import_error(self, monkeypatch):
        """ImportError inside _ensure_init flips _run to the False sentinel."""
        monkeypatch.setitem(sys.modules, "wandb", None)
        backend = WandBBackend(MetricsConfig(enable_wandb=True))
        backend.log({"loss": 1.0}, step=1)
        assert backend._run is False

    def test_wandb_handles_init_exception(self, monkeypatch):
        """Non-ImportError exception from wandb.init() also flips _run = False."""
        import wandb

        def _boom(**kwargs):
            raise RuntimeError("simulated auth failure")

        monkeypatch.setattr(wandb, "init", _boom)
        backend = WandBBackend(MetricsConfig(enable_wandb=True))
        backend.log({"loss": 1.0}, step=1)
        assert backend._run is False


class _FakeMlflow:
    """Stand-in `mlflow` module for MLflowBackend tests (no real tracking server)."""

    def __init__(self):
        self._run = types.SimpleNamespace(info=types.SimpleNamespace(run_id="run-xyz"))
        self.log_steps: list = []
        self.start_run_ids: list = []
        self.end_run_calls = 0
        self.logged_params: list = []
        self.logged_tags: list = []
        self.history_steps: list = []  # steps returned by MlflowClient().get_metric_history

    def set_tracking_uri(self, *a, **k):
        pass

    def set_experiment(self, *a, **k):
        pass

    def set_system_metrics_sampling_interval(self, *a, **k):
        pass

    def start_run(self, run_id=None, run_name=None, log_system_metrics=False):
        self.start_run_ids.append(run_id)
        return self._run

    def log_metrics(self, metrics, step=None):
        self.log_steps.append(step)

    def log_params(self, params):
        self.logged_params.append(params)

    def set_tags(self, tags):
        self.logged_tags.append(tags)

    def end_run(self):
        self.end_run_calls += 1

    def MlflowClient(self, *a, **k):
        steps = [types.SimpleNamespace(step=s) for s in self.history_steps]
        return types.SimpleNamespace(get_metric_history=lambda run_id, key: steps)


def _mlflow_cfg(**kw):
    base = dict(
        enable_mlflow=True,
        mlflow_tracking_uri="http://localhost:5000",  # non-databricks: skip readiness gate
        mlflow_experiment="exp",
        mlflow_log_system_metrics=False,
    )
    base.update(kw)
    return MetricsConfig(**base)


class TestMLflowBackend:
    def test_init_no_crash(self):
        backend = MLflowBackend(MetricsConfig(enable_mlflow=True))
        assert backend._active is None

    def test_disabled_without_databricks_creds(self, monkeypatch):
        """tracking_uri=databricks with no creds disables the backend before importing mlflow."""
        for var in ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_API_TOKEN"):
            monkeypatch.delenv(var, raising=False)
        backend = MLflowBackend(MetricsConfig(enable_mlflow=True))  # default uri = databricks
        backend.log({"train/loss": 1.0}, step=1)
        assert backend._active is False

    def test_mlflow_handles_import_error(self, monkeypatch):
        """ImportError inside _ensure_init flips _active to the False sentinel."""
        monkeypatch.setitem(sys.modules, "mlflow", None)
        # Non-databricks URI so the readiness gate is skipped and the import is attempted.
        backend = MLflowBackend(
            MetricsConfig(enable_mlflow=True, mlflow_tracking_uri="http://localhost:5000")
        )
        backend.log({"train/loss": 1.0}, step=1)
        assert backend._active is False

    def test_mlflow_handles_init_exception(self, monkeypatch):
        """A non-ImportError from mlflow (e.g. tracking/auth) flips _active = False."""
        mlflow = pytest.importorskip("mlflow")

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated tracking failure")

        monkeypatch.setattr(mlflow, "set_tracking_uri", _boom)
        backend = MLflowBackend(
            MetricsConfig(enable_mlflow=True, mlflow_tracking_uri="sqlite:///x.db")
        )
        backend.log({"train/loss": 1.0}, step=1)
        assert backend._active is False

    def test_resume_falls_back_to_new_run_when_id_missing(self, monkeypatch):
        """A stale saved run_id (run deleted) starts a fresh run, like wandb resume='allow'."""
        import types

        tried: list = []
        fake_run = types.SimpleNamespace(info=types.SimpleNamespace(run_id="new-run-123"))

        def fake_start_run(run_id=None, run_name=None, log_system_metrics=False):
            tried.append(run_id)
            if run_id is not None:
                raise RuntimeError("RESOURCE_DOES_NOT_EXIST")
            return fake_run

        fake_mlflow = types.SimpleNamespace(
            set_tracking_uri=lambda *a, **k: None,
            set_experiment=lambda *a, **k: None,
            set_system_metrics_sampling_interval=lambda *a, **k: None,
            start_run=fake_start_run,
            log_params=lambda *a, **k: None,
            set_tags=lambda *a, **k: None,
            log_metrics=lambda *a, **k: None,
            end_run=lambda *a, **k: None,
        )
        monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
        cfg = MetricsConfig(
            enable_mlflow=True,
            mlflow_tracking_uri="http://localhost:5000",  # skip the databricks readiness gate
            mlflow_run_id="stale-id",
            mlflow_log_system_metrics=False,
        )
        backend = MLflowBackend(cfg)
        backend.log({"train/loss": 1.0}, step=1)
        assert backend._active is True
        assert cfg.mlflow_run_id == "new-run-123"  # wrote back the new id
        assert tried == ["stale-id", None]  # tried resume, then started fresh

    def test_log_retries_after_transient_failure(self, monkeypatch):
        """A transient log error is swallowed (warned) but does NOT disable the backend;
        the next step retries and succeeds. (Finding #1)"""
        fm = _FakeMlflow()
        state = {"n": 0}

        def flaky_log(metrics, step=None):
            state["n"] += 1
            if state["n"] == 1:
                raise RuntimeError("transient network blip")
            fm.log_steps.append(step)

        fm.log_metrics = flaky_log
        monkeypatch.setitem(sys.modules, "mlflow", fm)
        backend = MLflowBackend(_mlflow_cfg())
        backend.log({"train/loss": 2.0}, step=1)  # fails internally, swallowed
        backend.log({"train/loss": 1.5}, step=2)  # must retry, not stay disabled
        assert backend._active is True
        assert state["n"] == 2  # second call actually attempted (no permanent disable)
        assert fm.log_steps == [2]

    def test_resume_skips_already_logged_steps(self, monkeypatch):
        """On resume, steps already recorded in the run are not re-logged. (Finding #2)"""
        fm = _FakeMlflow()
        fm.history_steps = [1, 2, 3, 4]  # run already has metrics up to step 4
        monkeypatch.setitem(sys.modules, "mlflow", fm)
        backend = MLflowBackend(_mlflow_cfg(mlflow_run_id="existing-run"))
        backend.log({"train/loss": 1.2}, step=3)  # already logged -> skip
        backend.log({"train/loss": 1.0}, step=4)  # already logged -> skip
        backend.log({"train/loss": 0.8}, step=5)  # new -> log
        assert fm.log_steps == [5]

    def test_bad_env_experiment_disables_with_clear_message(self, monkeypatch):
        """Non-absolute $MLFLOW_EXPERIMENT on Databricks disables with a specific message,
        not the generic 'MLflow init failed'. (Finding #3)"""
        monkeypatch.setenv("DATABRICKS_HOST", "https://x.cloud.databricks.com")
        monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-x")
        monkeypatch.setenv("MLFLOW_EXPERIMENT", "bare-name")  # not absolute
        warnings_seen: list = []
        monkeypatch.setattr(
            tracker_mod.logger, "warning", lambda msg, *a, **k: warnings_seen.append(str(msg))
        )
        fm = _FakeMlflow()
        monkeypatch.setitem(sys.modules, "mlflow", fm)
        backend = MLflowBackend(MetricsConfig(enable_mlflow=True))  # default uri = databricks
        backend.log({"train/loss": 1.0}, step=1)
        assert backend._active is False
        assert fm.start_run_ids == []  # never started a run
        text = " ".join(warnings_seen).lower()
        assert "absolute" in text and "disabled" in text
        assert "init failed" not in text  # a deliberate config error, not an unexpected failure

    def test_null_experiment_disables_on_databricks(self, monkeypatch):
        """On Databricks, an unresolvable experiment (None) disables the backend instead of
        logging into the default experiment. (Finding #7)"""
        monkeypatch.setenv("DATABRICKS_HOST", "https://x.cloud.databricks.com")
        monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-x")
        monkeypatch.delenv("MLFLOW_EXPERIMENT", raising=False)
        monkeypatch.setattr(tracker_mod, "_resolve_mlflow_experiment", lambda cfg, uri: None)
        fm = _FakeMlflow()
        monkeypatch.setitem(sys.modules, "mlflow", fm)
        backend = MLflowBackend(MetricsConfig(enable_mlflow=True))  # uri = databricks
        backend.log({"train/loss": 1.0}, step=1)
        assert backend._active is False
        assert fm.start_run_ids == []  # never started a run in the default experiment

    def test_resume_transient_error_does_not_fork(self, monkeypatch):
        """A transient error resuming a still-valid run must NOT fork a new run; it disables
        instead so history isn't split. (Finding #8)"""
        fm = _FakeMlflow()

        def start_run(run_id=None, run_name=None, log_system_metrics=False):
            fm.start_run_ids.append(run_id)
            if run_id is not None:
                raise RuntimeError("503 Service Unavailable")  # transient; run still exists
            return fm._run

        fm.start_run = start_run
        monkeypatch.setitem(sys.modules, "mlflow", fm)
        backend = MLflowBackend(_mlflow_cfg(mlflow_run_id="live-run"))
        backend.log({"train/loss": 1.0}, step=1)
        assert backend._active is False  # disabled, not forked
        assert fm.start_run_ids == ["live-run"]  # only the resume attempt, no fresh start_run()

    def test_close_ends_run_even_when_inactive(self, monkeypatch):
        """Once a run is started, close() ends it even if the backend was later marked
        inactive, so the run isn't left RUNNING. (Finding #11)"""
        fm = _FakeMlflow()
        monkeypatch.setitem(sys.modules, "mlflow", fm)
        backend = MLflowBackend(_mlflow_cfg())
        backend.log({"train/loss": 1.0}, step=1)  # starts the run
        backend._active = False  # simulate a later disable
        backend.close()
        assert fm.end_run_calls == 1

    def test_close_survives_end_run_error(self, monkeypatch):
        """A failing end_run at teardown warns but does not raise."""
        fm = _FakeMlflow()

        def boom_end():
            raise RuntimeError("server gone")

        fm.end_run = boom_end
        monkeypatch.setitem(sys.modules, "mlflow", fm)
        backend = MLflowBackend(_mlflow_cfg())
        backend.log({"train/loss": 1.0}, step=1)
        backend.close()  # must not raise


class TestResolveMlflowExperiment:
    def test_env_experiment_must_be_absolute_on_databricks(self, monkeypatch):
        """Non-absolute $MLFLOW_EXPERIMENT is rejected on Databricks (config-guard parity)."""
        monkeypatch.setenv("MLFLOW_EXPERIMENT", "bare-name")
        with pytest.raises(ValueError, match="absolute workspace path"):
            _resolve_mlflow_experiment(MetricsConfig(enable_mlflow=True), "databricks")

    def test_env_experiment_bare_ok_for_non_databricks(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_EXPERIMENT", "bare-name")
        cfg = MetricsConfig(enable_mlflow=True, mlflow_tracking_uri="http://localhost:5000")
        assert _resolve_mlflow_experiment(cfg, "http://localhost:5000") == "bare-name"


class TestTensorBoardBackend:
    def test_init_no_crash(self):
        config = MetricsConfig(enable_tensorboard=True)
        backend = TensorBoardBackend(config)
        assert backend._writer is None

    def test_log_creates_writer(self, tmp_path):
        config = MetricsConfig(enable_tensorboard=True, tensorboard_dir=str(tmp_path / "tb"))
        backend = TensorBoardBackend(config)
        backend.log({"train/loss": 2.5}, step=1)
        # Writer should be initialized (or False if tensorboard not installed)
        assert backend._writer is not None
        backend.close()

    def test_tb_handles_import_error(self, monkeypatch):
        """ImportError inside _ensure_init flips _writer to the False sentinel."""
        monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", None)
        backend = TensorBoardBackend(MetricsConfig(enable_tensorboard=True))
        backend.log({"loss": 1.0}, step=1)
        assert backend._writer is False


class TestFlattenConfigParams:
    def test_flatten_produces_dotted_scalar_string_keys(self):
        config = JobConfig(
            model=ModelConfig(dim=128, n_layers=2, n_heads=2, vocab_size=256),
            metrics=MetricsConfig(log_interval=7),
        )
        params = _flatten_config_params(config)
        assert params["model.dim"] == "128"
        assert params["metrics.log_interval"] == "7"
        # mlflow.log_params requires string values.
        assert all(isinstance(v, str) for v in params.values())
        # Optional None sub-configs (vlm, adapter, vision_encoder) are skipped.
        assert not any(k.startswith("vlm") for k in params)

    def test_flatten_truncates_long_values(self):
        config = JobConfig(model=ModelConfig(dim=128, n_layers=2, n_heads=2, vocab_size=256))
        params = _flatten_config_params(config, max_len=4)
        assert all(len(v) <= 4 for v in params.values())

    def test_flatten_recurses_into_lists(self):
        """list-of-dataclass config (data.datasets) expands into indexed dotted keys,
        not one truncated str(). (Finding #5)"""
        config = JobConfig(
            model=ModelConfig(dim=128, n_layers=2, n_heads=2, vocab_size=256),
            data=DataConfig(
                datasets=[
                    DatasetSource(path="/d/a", name="ds_a", weight=0.7),
                    DatasetSource(path="/d/b", name="ds_b", weight=0.3),
                ]
            ),
        )
        params = _flatten_config_params(config)
        assert params["data.datasets.0.name"] == "ds_a"
        assert params["data.datasets.1.name"] == "ds_b"
        assert params["data.datasets.0.path"] == "/d/a"
        assert "data.datasets" not in params  # not collapsed into one opaque key


class TestMetricsConfigMLflow:
    def test_databricks_requires_absolute_experiment_path(self):
        with pytest.raises(ValueError, match="absolute workspace path"):
            MetricsConfig(enable_mlflow=True, mlflow_experiment="bare-name")

    def test_absolute_experiment_path_ok(self):
        cfg = MetricsConfig(enable_mlflow=True, mlflow_experiment="/Users/me/proj")
        assert cfg.mlflow_experiment == "/Users/me/proj"

    def test_bare_name_ok_for_local_uri(self):
        cfg = MetricsConfig(
            enable_mlflow=True,
            mlflow_tracking_uri="sqlite:///mlflow.db",
            mlflow_experiment="bare-name",
        )
        assert cfg.mlflow_experiment == "bare-name"


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Rank-aware logging
# ---------------------------------------------------------------------------


class TestRankLogger:
    def test_get_logger_returns_logger(self):
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "kempnerforge.test_module"

    def test_rank_filter_allows_rank_zero(self):
        f = _RankFilter(rank=0)
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        with patch.dict(os.environ, {"RANK": "0"}):
            assert f.filter(record) is True

    def test_rank_filter_blocks_non_zero(self):
        f = _RankFilter(rank=0)
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        with patch.dict(os.environ, {"RANK": "3"}):
            assert f.filter(record) is False

    def test_rank_filter_custom_rank(self):
        """Filter allowing rank 2 should pass rank 2 and block others."""
        f = _RankFilter(rank=2)
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        with patch.dict(os.environ, {"RANK": "2"}):
            assert f.filter(record) is True
        with patch.dict(os.environ, {"RANK": "0"}):
            assert f.filter(record) is False

    def test_rank_formatter_includes_rank(self):
        fmt = _RankFormatter(use_color=False)
        record = logging.LogRecord("test", logging.INFO, "", 0, "hello world", (), None)
        with patch.dict(os.environ, {"RANK": "5"}):
            output = fmt.format(record)
        assert "[rank 5]" in output
        assert "INFO" in output
        assert "hello world" in output

    def test_supports_color_no_color_env(self, monkeypatch):
        """NO_COLOR=1 disables color output regardless of TTY status."""
        monkeypatch.setenv("NO_COLOR", "1")
        assert _supports_color() is False

    def test_supports_color_no_isatty(self, monkeypatch):
        """A stdout object without isatty disables color output.

        io.StringIO HAS isatty (returns False) so cannot be used here;
        use a custom stub that simply lacks the attribute.
        """

        class _NoIsattyStdout:
            def write(self, s):
                pass

            def flush(self):
                pass

        monkeypatch.setattr(sys, "stdout", _NoIsattyStdout())
        assert _supports_color() is False

    def test_rank_formatter_with_color(self, monkeypatch):
        """When use_color=True and _supports_color()=True, output includes ANSI."""
        monkeypatch.setattr(log_mod, "_supports_color", lambda: True)
        monkeypatch.setenv("RANK", "0")
        fmt = _RankFormatter(use_color=True)
        record = logging.LogRecord("test", logging.INFO, "", 0, "hello", (), None)
        output = fmt.format(record)
        assert "\x1b[" in output

    def test_configure_root_no_rank_filter(self, monkeypatch):
        """When rank_zero_only=False, _configure_root attaches no _RankFilter.

        _configure_root has a module-level _configured idempotency guard; any
        earlier get_logger() call will have set it to True. We reset it via
        monkeypatch so the function body actually runs.
        """
        monkeypatch.setattr(log_mod, "_configured", False)
        root = logging.getLogger("kempnerforge")
        orig_handlers = list(root.handlers)
        try:
            root.handlers.clear()
            log_mod._configure_root(rank_zero_only=False)
            for h in root.handlers:
                rank_filters = [f for f in h.filters if isinstance(f, log_mod._RankFilter)]
                assert rank_filters == []
        finally:
            root.handlers.clear()
            root.handlers.extend(orig_handlers)


# ---------------------------------------------------------------------------
# Format metrics
# ---------------------------------------------------------------------------


class TestFormatMetrics:
    def test_format_metrics_basic(self):
        s = format_metrics(100, {"loss": "2.34", "lr": "3.00e-04"})
        assert "100" in s
        assert "loss" in s

    def test_format_number_large_int(self):
        assert "125k" in _format_number(125000)
        assert "1.5M" in _format_number(1500000)
        assert "1.0B" in _format_number(1000000000)

    def test_format_number_small_float(self):
        result = _format_number(3e-4)
        assert "e" in result  # Scientific notation

    def test_format_number_regular_float(self):
        result = _format_number(2.34)
        assert "2.34" in result

    def test_format_number_small_int(self):
        """Integers below 1000 return as plain str without unit suffix."""
        assert _format_number(42) == "42"

    def test_format_number_non_numeric_fallback(self):
        """Defensive final return: non-numeric input passes through as str(val)."""
        assert _format_number("foo") == "foo"
