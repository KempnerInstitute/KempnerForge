"""Unit tests for KempnerForge observability modules (tracker, memory monitor, backends)."""

from __future__ import annotations

import logging
import os
import sys
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist

import kempnerforge.metrics.logger as log_mod
import kempnerforge.metrics.tracker as tracker_mod
from kempnerforge.config.schema import JobConfig, MetricsConfig, ModelConfig
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
