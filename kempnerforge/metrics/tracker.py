"""Metrics collection, accumulation, and reporting.

MetricsTracker aggregates per-step metrics (loss, grad norm, throughput,
MFU, memory) and dispatches them to configured logging backends (stdout,
WandB, TensorBoard) at a configurable interval.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from kempnerforge.config.schema import JobConfig, MetricsConfig
from kempnerforge.metrics.logger import format_metrics, get_logger
from kempnerforge.metrics.memory import get_memory_stats, get_memory_utilization
from kempnerforge.metrics.mfu import compute_mfu, get_gpu_peak_tflops

logger = get_logger(__name__)


@dataclass
class StepMetrics:
    """Metrics for a single training step."""

    loss: float = 0.0
    grad_norm: float = 0.0
    lr: float = 0.0
    tokens_per_sec: float = 0.0
    mfu: float = 0.0
    step_time_sec: float = 0.0
    allocated_gb: float = 0.0
    peak_gb: float = 0.0
    reserved_gb: float = 0.0
    total_gb: float = 0.0
    mem_utilization: float = 0.0


class MetricsTracker:
    """Collects, smooths, and reports training metrics.

    Timing is handled internally — call ``start_step()`` before and
    ``end_step()`` after each training step. Metrics are logged to
    all configured backends at the configured interval.

    Args:
        config: Full job config (used for MFU calculation and backend selection).
        num_gpus: Number of GPUs for MFU denominator.
        gpu_peak_tflops: Per-GPU peak TFLOPS. If None, auto-detected.
    """

    def __init__(
        self,
        config: JobConfig,
        num_gpus: int = 1,
        gpu_peak_tflops: float | None = None,
    ) -> None:
        self.metrics_config = config.metrics
        self.model_config = config.model
        self.num_gpus = num_gpus
        self.gpu_peak_tflops = gpu_peak_tflops or get_gpu_peak_tflops()

        # Smoothed metrics (exponential moving average)
        self._ema_alpha = 0.1
        self._smoothed: dict[str, float] = {}

        # Per-step timing
        self._step_start: float = 0.0

        # Logging backends (initialized lazily)
        self._backends: list[_LoggingBackend] = []
        self._backends_initialized = False

    def _init_backends(self, config: JobConfig) -> None:
        """Lazily initialize logging backends."""
        if self._backends_initialized:
            return
        self._backends_initialized = True

        mc = config.metrics
        if mc.enable_wandb:
            self._backends.append(WandBBackend(mc))
        if mc.enable_tensorboard:
            self._backends.append(TensorBoardBackend(mc))

    def start_step(self) -> None:
        """Mark the beginning of a training step."""
        self._step_start = time.perf_counter()

    def end_step(
        self,
        step: int,
        loss: float,
        grad_norm: float,
        lr: float,
        tokens_in_step: int,
    ) -> StepMetrics | None:
        """Mark the end of a training step and optionally log metrics.

        Args:
            step: Current training step number.
            loss: Loss value for this step.
            grad_norm: Gradient norm (after clipping).
            lr: Current learning rate.
            tokens_in_step: Total tokens processed in this step (across all GPUs).

        Returns:
            StepMetrics if this step was a logging step, None otherwise.
        """
        step_time = time.perf_counter() - self._step_start
        tokens_per_sec = tokens_in_step / step_time if step_time > 0 else 0.0

        # Compute MFU
        mfu = compute_mfu(
            self.model_config,
            tokens_per_sec=tokens_per_sec,
            num_gpus=self.num_gpus,
            gpu_peak_tflops=self.gpu_peak_tflops,
        )

        # Memory stats
        mem_stats = get_memory_stats()
        mem_util = get_memory_utilization()

        metrics = StepMetrics(
            loss=loss,
            grad_norm=grad_norm,
            lr=lr,
            tokens_per_sec=tokens_per_sec,
            mfu=mfu,
            step_time_sec=step_time,
            allocated_gb=mem_stats["allocated_gb"],
            peak_gb=mem_stats["peak_gb"],
            reserved_gb=mem_stats["reserved_gb"],
            total_gb=mem_stats["total_gb"],
            mem_utilization=mem_util,
        )

        # Update smoothed metrics
        self._update_smoothed("loss", loss)
        self._update_smoothed("tokens_per_sec", tokens_per_sec)
        self._update_smoothed("mfu", mfu)
        self._update_smoothed("step_time", step_time)

        # Log at interval
        if step % self.metrics_config.log_interval == 0 or step == 1:
            self._log_step(step, metrics)
            return metrics

        return None

    def _update_smoothed(self, key: str, value: float) -> None:
        """Update exponential moving average for a metric."""
        if key not in self._smoothed:
            self._smoothed[key] = value
        else:
            alpha = self._ema_alpha
            self._smoothed[key] = alpha * value + (1 - alpha) * self._smoothed[key]

    def _log_step(self, step: int, metrics: StepMetrics) -> None:
        """Log metrics to stdout and all backends."""
        # Stdout logging
        log_dict: dict[str, str | float | int] = {
            "loss": f"{metrics.loss:.4f}",
            "lr": f"{metrics.lr:.2e}",
            "grad_norm": f"{metrics.grad_norm:.3f}",
            "tok/s": f"{metrics.tokens_per_sec:,.0f}",
            "mfu": f"{metrics.mfu:.1%}",
            "mem": (f"{metrics.peak_gb:.1f}/{metrics.total_gb:.0f}GB"),
            "step_time": f"{metrics.step_time_sec:.2f}s",
        }
        logger.info(format_metrics(step, log_dict))

        # Backend logging (numeric dict)
        backend_dict = {
            "train/loss": metrics.loss,
            "train/grad_norm": metrics.grad_norm,
            "train/lr": metrics.lr,
            "train/tokens_per_sec": metrics.tokens_per_sec,
            "train/mfu": metrics.mfu,
            "train/step_time_sec": metrics.step_time_sec,
            "gpu/allocated_gb": metrics.allocated_gb,
            "gpu/peak_gb": metrics.peak_gb,
            "gpu/reserved_gb": metrics.reserved_gb,
            "gpu/mem_utilization": metrics.mem_utilization,
        }

        # Smoothed metrics
        for key, val in self._smoothed.items():
            backend_dict[f"smoothed/{key}"] = val

        for backend in self._backends:
            backend.log(backend_dict, step=step)

    def log_eval(self, metrics: dict[str, float], step: int) -> None:
        """Log eval metrics to all backends and stdout."""
        logger.info(format_metrics(step, metrics))
        for backend in self._backends:
            backend.log(metrics, step=step)

    def init_backends(self, config: JobConfig) -> None:
        """Initialize logging backends (call after distributed setup)."""
        self._init_backends(config)

    def close(self) -> None:
        """Flush and close all logging backends."""
        for backend in self._backends:
            backend.close()


# ---------------------------------------------------------------------------
# Logging backends
# ---------------------------------------------------------------------------


class _LoggingBackend:
    """Base class for metrics logging backends."""

    def log(self, metrics: dict[str, float], step: int) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class WandBBackend(_LoggingBackend):
    """Weights & Biases logging backend.

    Initializes a WandB run on first log call.
    """

    def __init__(self, config: MetricsConfig) -> None:
        self._config = config
        self._run = None

    def _ensure_init(self) -> None:
        if self._run is not None:
            return
        try:
            import wandb

            self._run = wandb.init(
                project=self._config.wandb_project,
                name=self._config.wandb_run_name,
                resume="allow",
            )
            logger.info(f"WandB initialized: {self._run.url}")
        except ImportError:
            logger.warning("wandb not installed — disabling WandB backend")
            self._run = False  # Sentinel: tried and failed
        except Exception as e:  # wandb.init() can raise many third-party errors (network, auth)
            logger.warning(f"WandB init failed: {e}")
            self._run = False

    def log(self, metrics: dict[str, float], step: int) -> None:
        self._ensure_init()
        if self._run is False:
            return
        import wandb

        wandb.log(metrics, step=step)

    def close(self) -> None:
        if self._run and self._run is not False:
            import wandb

            wandb.finish()


class TensorBoardBackend(_LoggingBackend):
    """TensorBoard logging backend."""

    def __init__(self, config: MetricsConfig) -> None:
        self._config = config
        self._writer = None

    def _ensure_init(self) -> None:
        if self._writer is not None:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(log_dir=self._config.tensorboard_dir)
            logger.info(f"TensorBoard writer → {self._config.tensorboard_dir}")
        except ImportError:
            logger.warning("tensorboard not installed — disabling TensorBoard backend")
            self._writer = False

    def log(self, metrics: dict[str, float], step: int) -> None:
        self._ensure_init()
        if self._writer is False:
            return
        for key, val in metrics.items():
            self._writer.add_scalar(key, val, global_step=step)

    def close(self) -> None:
        if self._writer and self._writer is not False:
            self._writer.close()
