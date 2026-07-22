# pyright: reportMissingImports=false
# ^ mlflow + databricks-sdk are an optional group; CI type-checks without them,
#   so the lazy imports below would otherwise raise reportMissingImports.
"""Metrics collection, accumulation, and reporting.

MetricsTracker aggregates per-step metrics (loss, grad norm, throughput,
MFU, memory) and dispatches them to configured logging backends (stdout,
WandB, TensorBoard, MLflow) at a configurable interval.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

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
        self.seq_len = config.train.seq_len
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
        """Lazily initialize logging backends (rank 0 only)."""
        if self._backends_initialized:
            return
        self._backends_initialized = True

        import torch.distributed as dist

        if dist.is_initialized() and dist.get_rank() != 0:
            return

        mc = config.metrics
        if mc.enable_wandb:
            self._backends.append(WandBBackend(mc))
        if mc.enable_tensorboard:
            self._backends.append(TensorBoardBackend(mc))
        if mc.enable_mlflow:
            self._backends.append(MLflowBackend(mc, job_config=config))

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
            seq_len=self.seq_len,
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
        logger.info(format_metrics(step, metrics))  # type: ignore[reportArgumentType]
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

            init_kwargs: dict[str, Any] = {
                "project": self._config.wandb_project,
                "name": self._config.wandb_run_name,
                "resume": "allow",
            }
            if self._config.wandb_run_id:
                init_kwargs["id"] = self._config.wandb_run_id
            self._run = wandb.init(**init_kwargs)
            self._config.wandb_run_id = self._run.id
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


def _mlflow_databricks_ready() -> bool:
    """True when Databricks credentials are present in the environment."""
    has_token = bool(os.environ.get("DATABRICKS_TOKEN") or os.environ.get("DATABRICKS_API_TOKEN"))
    return bool(os.environ.get("DATABRICKS_HOST")) and has_token


def _resolve_mlflow_experiment(config: MetricsConfig, uri: str) -> str | None:
    """Experiment name: mlflow_experiment -> $MLFLOW_EXPERIMENT -> auto
    (/Users/<user>/<project> via the SDK on Databricks; bare name locally).

    On Databricks a non-absolute $MLFLOW_EXPERIMENT raises ValueError; the backend
    catches it and disables MLflow with a clear message (the config-field equivalent
    is rejected at load time by MetricsConfig.__post_init__).
    """
    on_databricks = uri.startswith("databricks")
    if config.mlflow_experiment:
        return config.mlflow_experiment  # validated in MetricsConfig.__post_init__
    env_experiment = os.environ.get("MLFLOW_EXPERIMENT")
    if env_experiment:
        if on_databricks and not env_experiment.startswith("/"):
            raise ValueError(
                "$MLFLOW_EXPERIMENT must be an absolute workspace path on Databricks "
                f"(e.g. '/Users/you@example.com/proj'); got {env_experiment!r}"
            )
        return env_experiment
    project = config.wandb_project or "kempnerforge"
    if not on_databricks:
        return project
    try:
        from databricks.sdk import WorkspaceClient

        user = WorkspaceClient().current_user.me().user_name
        return f"/Users/{user}/{project}"
    except Exception as e:  # SDK missing, no creds, or API error
        logger.warning(f"Could not auto-resolve Databricks experiment path: {e}")
        return None


def _flatten_config_params(config: JobConfig, max_len: int = 250) -> dict[str, str]:
    """Flatten a JobConfig to dotted string keys (model.dim, ...) for mlflow.log_params.

    None is skipped; non-scalars are stringified; values truncated to max_len.
    """
    from dataclasses import asdict

    flat: dict[str, str] = {}

    def _walk(prefix: str, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, dict):
            for k, v in value.items():
                _walk(f"{prefix}.{k}" if prefix else str(k), v)
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                _walk(f"{prefix}.{i}" if prefix else str(i), v)
        else:
            flat[prefix] = str(value)[:max_len]

    _walk("", asdict(config))
    return flat


def _mlflow_run_gone(exc: Exception) -> bool:
    """True if the error means the run can't be resumed (deleted / does not exist), vs a
    transient network/server error that should NOT trigger a fresh-run fallback."""
    code = str(getattr(exc, "error_code", "") or "")
    if "RESOURCE_DOES_NOT_EXIST" in code or "NOT_FOUND" in code:
        return True
    msg = str(exc).lower()
    return any(
        s in msg for s in ("does not exist", "resource_does_not_exist", "not found", "deleted")
    )


def _mlflow_last_step(mlflow: Any, run_id: str) -> int | None:
    """Highest step already recorded for the watermark metric, so a resumed run can skip
    re-logging steps it already has (mlflow metric history is append-only)."""
    try:
        hist = mlflow.MlflowClient().get_metric_history(run_id, "train/loss")
        return max((m.step for m in hist), default=None)
    except Exception:
        return None


class MLflowBackend(_LoggingBackend):
    """MLflow logging backend (Databricks-hosted or any MLflow tracking server).

    Lazy init on first log; run ID written back to config for checkpoint resume.
    """

    def __init__(self, config: MetricsConfig, job_config: JobConfig | None = None) -> None:
        self._config = config
        self._job_config = job_config  # flattened to params lazily, only for a fresh run
        self._active: bool | None = None  # None = not tried, True = live, False = failed
        self._run_started = False  # a run exists that close() must end
        self._resume_skip_below: int | None = None  # skip re-logging steps already in the run
        self._log_failures = 0  # consecutive log failures (warn once, keep retrying)

    def _ensure_init(self) -> None:
        if self._active is not None:
            return
        uri = self._config.mlflow_tracking_uri
        if uri.startswith("databricks"):
            if not _mlflow_databricks_ready():
                logger.warning(
                    "enable_mlflow with tracking_uri='databricks' but DATABRICKS_HOST/token "
                    "not set — disabling MLflow backend"
                )
                self._active = False
                return
            # The Databricks SDK reads DATABRICKS_TOKEN; mirror the DATABRICKS_API_TOKEN
            # name many keep in ~/.bashrc so either works.
            if not os.environ.get("DATABRICKS_TOKEN") and os.environ.get("DATABRICKS_API_TOKEN"):
                os.environ["DATABRICKS_TOKEN"] = os.environ["DATABRICKS_API_TOKEN"]
        try:
            import mlflow

            mlflow.set_tracking_uri(uri)
            try:
                experiment = _resolve_mlflow_experiment(self._config, uri)
            except ValueError as e:  # bad $MLFLOW_EXPERIMENT: a deliberate config error
                logger.warning(f"MLflow disabled: {e}")
                self._active = False
                return
            if uri.startswith("databricks") and not experiment:
                logger.warning(
                    "MLflow disabled: could not resolve a Databricks experiment path "
                    "(set mlflow_experiment or $MLFLOW_EXPERIMENT to an absolute path)"
                )
                self._active = False
                return
            if experiment:
                mlflow.set_experiment(experiment)
            if self._config.mlflow_log_system_metrics:
                mlflow.set_system_metrics_sampling_interval(
                    self._config.mlflow_system_metrics_interval
                )
            run, resumed = self._start_run(mlflow)
            self._run_started = True
            self._config.mlflow_run_id = run.info.run_id
            self._active = True
            logger.info(f"MLflow initialized: experiment={experiment} run_id={run.info.run_id}")
            if resumed:
                self._resume_skip_below = _mlflow_last_step(mlflow, run.info.run_id)
            else:
                self._log_run_metadata(mlflow)
        except ImportError:
            logger.warning(
                "mlflow not installed — disabling MLflow backend (uv sync --group mlflow)"
            )
            self._active = False
        except Exception as e:  # mlflow talks to a remote server: network / auth / config errors
            logger.warning(f"MLflow init failed: {e}")
            self._active = False

    def _start_run(self, mlflow: Any) -> tuple[Any, bool]:
        """Start a run, or resume the saved run_id. If that run_id no longer exists, fall
        back to a fresh run (matching wandb's resume='allow'). Returns (run, resumed)."""
        kwargs: dict[str, Any] = {
            "run_name": self._config.mlflow_run_name,
            "log_system_metrics": self._config.mlflow_log_system_metrics,
        }
        run_id = self._config.mlflow_run_id or None
        if run_id:
            try:
                return mlflow.start_run(run_id=run_id, **kwargs), True
            except Exception as e:
                if not _mlflow_run_gone(e):
                    raise  # transient/other error — don't fork a new run; let init disable
                logger.warning(f"MLflow run_id={run_id} is gone ({e}); starting a new run")
        return mlflow.start_run(**kwargs), False

    def _log_run_metadata(self, mlflow: Any) -> None:
        """Log flattened config as params + host/slurm tags; non-fatal on error."""
        try:
            import socket

            hyperparams = _flatten_config_params(self._job_config) if self._job_config else {}
            items = list(hyperparams.items())
            # MLflow caps a single log_params batch at 100 entries.
            for i in range(0, len(items), 100):
                mlflow.log_params(dict(items[i : i + 100]))
            mlflow.set_tags(
                {"host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID", "")}
            )
        except Exception as e:
            logger.warning(f"MLflow metadata logging failed (continuing): {e}")

    def log(self, metrics: dict[str, float], step: int) -> None:
        self._ensure_init()
        if self._active is not True:
            return
        if self._resume_skip_below is not None and step <= self._resume_skip_below:
            return  # already logged before this resume — avoid duplicate history points
        import mlflow

        try:
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)
            self._log_failures = 0
        except Exception as e:  # transient network/token failure: warn once, keep retrying
            self._log_failures += 1
            if self._log_failures == 1:
                logger.warning(f"MLflow log failed ({e}); will keep retrying, points may be lost")

    def close(self) -> None:
        if not self._run_started:
            return
        try:
            import mlflow

            mlflow.end_run()
        except Exception as e:  # teardown must not fail the job
            logger.warning(f"MLflow end_run failed ({e})")
        finally:
            self._active = False
            self._run_started = False


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
            self._writer.add_scalar(key, val, global_step=step)  # type: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]

    def close(self) -> None:
        if self._writer and self._writer is not False:
            self._writer.close()  # type: ignore[reportAttributeAccessIssue]
