"""Unit tests for KempnerForge observability modules (tracker, memory monitor, backends)."""

from __future__ import annotations

import glob
import json
import logging
import os
import sys
import types
from contextlib import closing
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist

import kempnerforge.metrics.logger as log_mod
import kempnerforge.metrics.tracker as tracker_mod
from kempnerforge.config.schema import (
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
    StepMetrics,
    TensorBoardBackend,
    WandBBackend,
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
        monkeypatch.setattr(
            "kempnerforge.metrics.mlflow_tracking.backend.MLflowBackend", MagicMock(name="FakeMLflow")
        )
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


# ---------------------------------------------------------------------------
# Tracking entries: what a checkpoint's metadata.json records about the live runs
# ---------------------------------------------------------------------------


class _FakeWandbRun:
    """Stand-in wandb run; ``url`` counts its reads, since asking an offline run warns."""

    def __init__(self, *, entity="team", offline=False, disabled=False):
        self.id = "run-1"
        self.project = "proj"
        self.entity = entity
        self.offline = offline
        self.disabled = disabled
        self.url_reads = 0

    @property
    def url(self):
        self.url_reads += 1
        return None if self.offline else "https://tracking.example/team/proj/runs/run-1"


def _wandb_backend(monkeypatch, run) -> WandBBackend:
    """A WandBBackend whose lazy init (the first log) returns ``run`` instead of calling wandb."""
    import wandb

    monkeypatch.setattr(wandb, "init", lambda **kwargs: run)
    monkeypatch.setattr(wandb, "log", lambda *args, **kwargs: None)
    backend = WandBBackend(MetricsConfig(enable_wandb=True, wandb_project="proj"))
    backend.log({"train/loss": 1.0}, step=1)
    return backend


def _capture_formatted_warnings(monkeypatch) -> list[str]:
    """Tracker warnings as a handler would format them (messages logged with ``%s`` arguments)."""
    seen: list[str] = []
    monkeypatch.setattr(
        tracker_mod.logger, "warning", lambda msg, *args, **kw: seen.append(msg % args)
    )
    return seen


class _OnceItems(dict):
    """A mapping whose items can be listed once."""

    def __init__(self, **items):
        super().__init__(**items)
        self._listed = False

    def items(self):
        if self._listed:
            raise RuntimeError("items listed twice")
        self._listed = True
        return super().items()


class _HostileEntry(dict):
    """A mapping that cannot be inspected."""

    def items(self):
        raise RuntimeError("cannot list items")


class _EntryBackend(tracker_mod._LoggingBackend):
    """Reports a fixed entry, or raises, so the tracker's aggregation can be driven directly.

    It never tracks its own closure, like a backend that leaves that to the tracker.
    """

    def __init__(self, entry=None, error=None, close_error=None):
        self._entry = entry
        self._error = error
        self._close_error = close_error

    def log(self, metrics, step):
        pass

    def tracking_entry(self):
        if self._error is not None:
            raise self._error
        return self._entry

    def close(self):
        if self._close_error is not None:
            raise self._close_error


class TestTrackingEntries:
    def test_a_backend_without_a_descriptor_contributes_nothing(self):
        assert tracker_mod._LoggingBackend().tracking_entry() is None

    # WandB: run id and project, plus the entity and URL when the run has them

    def test_wandb_entry_carries_the_run_project_entity_and_url(self, monkeypatch):
        backend = _wandb_backend(monkeypatch, _FakeWandbRun())
        assert backend.tracking_entry() == {
            "backend": "wandb",
            "run_id": "run-1",
            "experiment": "proj",
            "entity": "team",
            "url": "https://tracking.example/team/proj/runs/run-1",
        }

    def test_wandb_entry_leaves_out_what_the_run_does_not_have(self, monkeypatch):
        run = _FakeWandbRun(entity="", offline=True)
        backend = _wandb_backend(monkeypatch, run)
        reads = run.url_reads
        assert backend.tracking_entry() == {
            "backend": "wandb",
            "run_id": "run-1",
            "experiment": "proj",
        }
        assert run.url_reads == reads  # an offline run is never asked for its URL

    def test_wandb_has_no_entry_without_a_live_run(self, monkeypatch):
        assert WandBBackend(MetricsConfig(enable_wandb=True)).tracking_entry() is None
        assert _wandb_backend(monkeypatch, _FakeWandbRun(disabled=True)).tracking_entry() is None

        monkeypatch.setitem(sys.modules, "wandb", None)
        backend = WandBBackend(MetricsConfig(enable_wandb=True))
        backend.log({"train/loss": 1.0}, step=1)
        assert backend._run is False
        assert backend.tracking_entry() is None  # wandb is not installed

    def test_wandb_has_no_entry_once_it_is_closed(self, monkeypatch):
        import wandb

        finished: list = []
        monkeypatch.setattr(wandb, "finish", lambda *args, **kwargs: finished.append(1))
        backend = _wandb_backend(monkeypatch, _FakeWandbRun())
        assert backend.tracking_entry() is not None
        backend.close()
        assert finished == [1]
        assert backend.tracking_entry() is None  # the run has finished

    # TensorBoard: the log directory once the writer is open

    def test_tensorboard_entry_names_the_log_dir_once_the_writer_is_open(self, tmp_path):
        pytest.importorskip("tensorboard")
        log_dir = str(tmp_path / "tb")
        config = MetricsConfig(enable_tensorboard=True, tensorboard_dir=log_dir)
        with closing(TensorBoardBackend(config)) as backend:
            assert backend.tracking_entry() is None
            backend.log({"train/loss": 2.5}, step=1)
            assert backend.tracking_entry() == {"backend": "tensorboard", "log_dir": log_dir}
            backend.close()
            assert backend.tracking_entry() is None  # the writer is closed

    def test_tensorboard_entry_names_the_directory_the_writer_chose(self, tmp_path, monkeypatch):
        pytest.importorskip("tensorboard")
        monkeypatch.chdir(tmp_path)  # an empty setting makes the writer create runs/<stamp> here
        config = MetricsConfig(enable_tensorboard=True, tensorboard_dir="")
        with closing(TensorBoardBackend(config)) as backend:
            backend.log({"train/loss": 2.5}, step=1)
            entry = backend.tracking_entry()
            assert entry is not None
            assert entry["log_dir"] == backend._writer.get_logdir()
            assert entry["log_dir"].startswith("runs")
            assert os.path.isdir(entry["log_dir"])

    def test_tensorboard_entry_is_a_string_even_for_a_path_setting(self, tmp_path):
        pytest.importorskip("tensorboard")
        config = MetricsConfig(enable_tensorboard=True, tensorboard_dir=tmp_path / "tb")  # type: ignore[arg-type]
        with closing(TensorBoardBackend(config)) as backend:
            backend.log({"train/loss": 2.5}, step=1)
            assert backend.tracking_entry() == {
                "backend": "tensorboard",
                "log_dir": str(tmp_path / "tb"),
            }

    @pytest.mark.parametrize(
        "setting",
        ["rel_dir", Path("path_dir"), b"bytes_dir", 7],
        ids=["str", "path", "bytes", "number"],
    )
    def test_tensorboard_entry_is_where_the_event_files_are_written(
        self, tmp_path, monkeypatch, setting
    ):
        pytest.importorskip("tensorboard")
        monkeypatch.chdir(tmp_path)  # relative settings create their directory here
        config = MetricsConfig(enable_tensorboard=True, tensorboard_dir=setting)  # type: ignore[arg-type]
        with closing(TensorBoardBackend(config)) as backend:
            backend.log({"train/loss": 2.5}, step=1)
            entry = backend.tracking_entry()
            assert entry is not None
            written = {os.path.dirname(p) for p in glob.glob("**/*tfevents*", recursive=True)}
            assert written == {entry["log_dir"]}

    @pytest.mark.parametrize(
        ("reported", "log_dir"),
        [
            ("logs/run", "logs/run"),
            (Path("logs/run"), "logs/run"),
            (" ", " "),
            (b"logs/run", "b'logs/run'"),
            (7, "7"),
            ("", None),
            (None, None),
        ],
        ids=["str", "path", "blank", "bytes", "number", "empty", "none"],
    )
    def test_tensorboard_entry_keeps_only_what_is_a_directory(self, reported, log_dir):
        backend = TensorBoardBackend(MetricsConfig(enable_tensorboard=True))
        backend._writer = types.SimpleNamespace(get_logdir=lambda: reported)
        expected = None if log_dir is None else {"backend": "tensorboard", "log_dir": log_dir}
        assert backend.tracking_entry() == expected

    def test_tensorboard_has_no_entry_when_it_is_not_installed(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", None)
        backend = TensorBoardBackend(MetricsConfig(enable_tensorboard=True))
        backend.log({"train/loss": 1.0}, step=1)
        assert backend._writer is False
        assert backend.tracking_entry() is None

    # The tracker lists every backend that has one

    def _tracker(self) -> MetricsTracker:
        model = ModelConfig(dim=128, n_layers=2, n_heads=2, vocab_size=256)
        return MetricsTracker(JobConfig(model=model, metrics=MetricsConfig()), num_gpus=1)

    def test_tracker_lists_the_live_backends_in_order_and_skips_the_rest(self):
        tracker = self._tracker()
        first = {"backend": "first"}
        last = {"backend": "last"}
        tracker._backends = [_EntryBackend(first), _EntryBackend(), _EntryBackend(last)]
        assert tracker.tracking_entries() == [first, last]
        assert self._tracker().tracking_entries() == []

    def test_tracker_skips_a_backend_that_cannot_describe_itself(self, monkeypatch):
        seen = _capture_formatted_warnings(monkeypatch)
        tracker = self._tracker()
        kept = {"backend": "kept"}
        tracker._backends = [_EntryBackend(error=RuntimeError("boom")), _EntryBackend(kept)]
        assert tracker.tracking_entries() == [kept]
        assert len(seen) == 1 and "_EntryBackend" in seen[0] and "boom" in seen[0]

    def test_tracker_skips_a_backend_whose_error_cannot_be_printed(self, monkeypatch):
        class Unprintable(Exception):
            def __str__(self):
                raise RuntimeError("no message")

        monkeypatch.setattr(logging, "raiseExceptions", False)  # a logging error is not raised
        tracker = self._tracker()
        kept = {"backend": "kept"}
        tracker._backends = [_EntryBackend(error=Unprintable()), _EntryBackend(kept)]
        assert tracker.tracking_entries() == [kept]

    def test_tracker_scrubs_credentials_from_a_backends_error(self, monkeypatch):
        seen = _capture_formatted_warnings(monkeypatch)
        tracker = self._tracker()
        error = RuntimeError("no answer from https://u:tok@host/api")
        tracker._backends = [_EntryBackend(error=error)]
        assert tracker.tracking_entries() == []
        assert len(seen) == 1 and "u:tok@" not in seen[0] and "https://host/api" in seen[0]

    @pytest.mark.parametrize(
        "bad",
        [
            ["backend"],
            {},
            {"run_id": "r"},
            {"backend": ""},
            {"backend": "  "},
            {"backend": "x", "log_dir": Path("somewhere")},
            {"backend": "x", "step": 3},
            {"backend": "x", 1: "one"},
            _HostileEntry(backend="x"),
        ],
        ids=[
            "list",
            "empty",
            "no-backend",
            "empty-backend",
            "blank-backend",
            "path",
            "number",
            "key",
            "hostile",
        ],
    )
    def test_tracker_skips_an_entry_a_checkpoint_could_not_hold(self, monkeypatch, bad):
        seen = _capture_formatted_warnings(monkeypatch)
        tracker = self._tracker()
        kept = {"backend": "kept", "run_id": "r"}
        tracker._backends = [_EntryBackend(bad), _EntryBackend(kept)]
        assert tracker.tracking_entries() == [kept]  # one bad backend does not cost the others
        assert len(seen) == 1 and "_EntryBackend" in seen[0]

    def test_tracker_keeps_a_blank_field_that_is_not_the_backend_name(self):
        entry = {"backend": "tensorboard", "log_dir": "   "}  # a directory can be named so
        tracker = self._tracker()
        tracker._backends = [_EntryBackend(entry)]
        assert tracker.tracking_entries() == [entry]
        assert json.loads(json.dumps(tracker.tracking_entries())) == [entry]

    def test_tracker_copies_each_entry_once_into_a_plain_dict(self):
        once = _OnceItems(backend="lazy", run_id="r")  # a mapping that cannot be read twice
        tracker = self._tracker()
        tracker._backends = [_EntryBackend(once), _EntryBackend({"backend": "kept"})]
        entries = tracker.tracking_entries()
        assert entries == [{"backend": "lazy", "run_id": "r"}, {"backend": "kept"}]
        assert type(entries[0]) is dict
        json.dumps(entries)  # what the checkpoint writes cannot go back to the original

    def test_tracker_reports_nothing_once_closed(self):
        tracker = self._tracker()
        tracker._backends = [_EntryBackend({"backend": "live"})]
        assert tracker.tracking_entries() == [{"backend": "live"}]
        tracker.close()
        assert tracker.tracking_entries() == []  # whatever the backend itself reports

    def test_tracker_reports_nothing_after_a_close_that_fails_part_way(self):
        tracker = self._tracker()
        tracker._backends = [
            _EntryBackend({"backend": "first"}, close_error=RuntimeError("teardown")),
            _EntryBackend({"backend": "later"}),
        ]
        with pytest.raises(RuntimeError, match="teardown"):
            tracker.close()
        assert tracker.tracking_entries() == []

    def test_tracker_closed_before_its_backends_started_stays_empty(self, tmp_path):
        pytest.importorskip("tensorboard")
        config = JobConfig(
            model=ModelConfig(dim=128, n_layers=2, n_heads=2, vocab_size=256),
            metrics=MetricsConfig(enable_tensorboard=True, tensorboard_dir=str(tmp_path / "tb")),
        )
        tracker = MetricsTracker(config, num_gpus=1)
        tracker.init_backends(config)
        try:
            tracker.close()
            tracker.log_eval({"eval/loss": 1.0}, step=1)  # a late log still starts the writer
            assert tracker._backends[0].tracking_entry() is not None
            assert tracker.tracking_entries() == []
        finally:
            tracker._backends[0].close()

    def test_other_ranks_hold_no_backends_so_report_nothing(self, monkeypatch):
        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_rank", lambda: 1)
        config = JobConfig(
            model=ModelConfig(dim=128, n_layers=2, n_heads=2, vocab_size=256),
            metrics=MetricsConfig(enable_tensorboard=True),
        )
        tracker = MetricsTracker(config, num_gpus=1)
        tracker.init_backends(config)
        assert tracker.tracking_entries() == []


class _RegisteringBackend(_EntryBackend):
    """An _EntryBackend that also adds checkpoint fields and registers checkpoints, or raises."""

    def __init__(self, entry=None, fields=None, error=None):
        super().__init__(entry)
        self._fields = fields
        self._register_error = error
        self.registered: list = []

    def checkpoint_fields(self, step):
        if self._register_error is not None:
            raise self._register_error
        return self._fields

    def register_checkpoint(self, step, directory, record):
        self.registered.append((step, directory))
        if self._register_error is not None:
            raise self._register_error
        record({"registered": f"step_{step}"})


class TestTrackerRegistration:
    def _tracker(self) -> MetricsTracker:
        model = ModelConfig(dim=128, n_layers=2, n_heads=2, vocab_size=256)
        return MetricsTracker(JobConfig(model=model, metrics=MetricsConfig()), num_gpus=1)

    def test_a_backend_adds_and_registers_nothing_by_default(self):
        backend = tracker_mod._LoggingBackend()
        recorded: list = []
        assert backend.checkpoint_fields(1) is None
        assert backend.register_checkpoint(1, Path("step_1"), recorded.append) is None
        assert recorded == []

    def test_fields_are_keyed_by_the_backend_name_of_each_live_entry(self):
        tracker = self._tracker()
        tracker._backends = [
            _RegisteringBackend({"backend": "first"}, {"model_id": "m-1"}),
            _RegisteringBackend({"backend": "quiet"}),
            _RegisteringBackend(None, {"model_id": "m-3"}),
            _RegisteringBackend({"backend": "later"}, {"a": "1", "b": "2"}),
        ]
        assert tracker.checkpoint_fields(7) == {
            "first": {"model_id": "m-1"},
            "later": {"a": "1", "b": "2"},
        }

    def test_each_live_backend_registers_with_a_record_bound_to_its_name(self, tmp_path):
        tracker = self._tracker()
        first = _RegisteringBackend({"backend": "first"})
        no_run = _RegisteringBackend(None)
        later = _RegisteringBackend({"backend": "later"})
        tracker._backends = [first, no_run, later]
        recorded: list = []
        directory = tmp_path / "step_7"
        tracker.register_checkpoint(7, directory, lambda name, fields: recorded.append(name))
        assert recorded == ["first", "later"]
        assert first.registered == later.registered == [(7, directory)]
        assert no_run.registered == []

    def test_a_backend_that_fails_costs_only_its_own_fields_and_registration(
        self, monkeypatch, tmp_path
    ):
        seen = _capture_formatted_warnings(monkeypatch)
        tracker = self._tracker()
        tracker._backends = [
            _RegisteringBackend(
                {"backend": "bad"}, error=RuntimeError("POST http://user:token@host/api failed")
            ),
            _RegisteringBackend({"run_id": "r"}, {"model_id": "m-2"}),
            _RegisteringBackend({"backend": "kept"}, {"model_id": "m-1"}),
        ]
        assert tracker.checkpoint_fields(1) == {"kept": {"model_id": "m-1"}}
        recorded: list = []
        tracker.register_checkpoint(1, tmp_path, lambda name, fields: recorded.append(name))
        assert recorded == ["kept"]
        assert len(seen) == 4
        assert all("_RegisteringBackend" in warning for warning in seen)
        assert "fields skipped" in seen[0] and "registration skipped" in seen[2]
        assert "POST http://host/api failed" in seen[0] and "token" not in seen[0] + seen[2]
        assert "no backend name" in seen[1] and "no backend name" in seen[3]

    def test_nothing_once_closed_or_without_backends(self, tmp_path):
        tracker = self._tracker()
        backend = _RegisteringBackend({"backend": "live"}, {"model_id": "m-1"})
        tracker._backends = [backend]
        recorded: list = []

        def record(name, fields):
            recorded.append(name)

        assert tracker.checkpoint_fields(1) == {"live": {"model_id": "m-1"}}
        tracker.register_checkpoint(1, tmp_path, record)
        tracker.close()
        assert tracker.checkpoint_fields(2) == {}
        tracker.register_checkpoint(2, tmp_path, record)
        assert recorded == ["live"] and [step for step, _ in backend.registered] == [1]
        empty = self._tracker()
        assert empty.checkpoint_fields(1) == {}
        empty.register_checkpoint(1, tmp_path, record)
        assert recorded == ["live"]


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
