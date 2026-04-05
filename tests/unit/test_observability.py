"""Unit tests for KempnerForge observability modules (tracker, memory monitor, backends)."""

from __future__ import annotations

import torch

from kempnerforge.config.schema import JobConfig, MetricsConfig, ModelConfig
from kempnerforge.metrics.logger import _format_number, format_metrics
from kempnerforge.metrics.memory import DeviceMemoryMonitor
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


# ---------------------------------------------------------------------------
# Format helpers
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
