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
import re
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
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


# Tracking is a side channel: a slow or unreachable server may cost points, never a stalled rank.
_CALL_DEADLINE_S = 60.0  # one metric / param / tag call
_UPLOAD_DEADLINE_S = 180.0  # one artifact upload
_INIT_DEADLINE_S = 60.0  # experiment setup plus run start or resume, on the first log
_END_DEADLINE_S = 60.0  # ending the run in close(), after the drain
_DRAIN_DEADLINE_S = 30.0  # close() flush budget
_QUEUE_MAX = 256  # pending calls before new ones are dropped
_MAX_CALL_THREADS = 4  # live call threads, abandoned ones included, before new calls are dropped
_WARN_INTERVAL_S = 300.0  # minimum gap between repeats of a warning


# mlflow and SDK errors can quote a credentialed tracking URI; nothing logged may carry it.
_URI_USERINFO = re.compile(r"(?<=://)[^/?#\s]+@")


def _scrub_uris(text: str) -> str:
    return _URI_USERINFO.sub("", text)


class _RateLimitedWarning:
    """Warns on the first event, then at most once per interval, counting what it held back."""

    def __init__(self, interval: float, clock: Callable[[], float] = time.monotonic) -> None:
        self._interval = interval
        self._clock = clock
        self._lock = threading.Lock()
        self._last: float | None = None
        self._held_back = 0

    def __call__(self, message: str) -> None:
        with self._lock:
            now = self._clock()
            if self._last is not None and now - self._last < self._interval:
                self._held_back += 1
                return
            held_back, self._held_back, self._last = self._held_back, 0, now
        suffix = f" ({held_back} similar warnings suppressed)" if held_back else ""
        logger.warning(_scrub_uris(message) + suffix)


class _BestEffortWorker:
    """Runs tracking calls off the training thread so a slow server cannot stall it.

    One daemon thread takes calls in submission order. Each call runs on a short-lived thread
    that the worker waits on for at most the call's deadline; a thread cannot be cancelled, so a
    call past its deadline is abandoned and the next one proceeds. At most ``max_threads`` call
    threads are alive at once, abandoned ones included; beyond that, calls are dropped until one
    returns. ``submit`` never blocks (a full queue drops the new call), and failures, timeouts
    and drops only emit rate-limited warnings.
    """

    def __init__(
        self,
        name: str,
        *,
        max_queue: int,
        max_threads: int,
        call_deadline: float,
        drain_deadline: float,
    ) -> None:
        self._name = name
        self._max_queue = max_queue
        self._max_threads = max_threads
        self._call_deadline = call_deadline
        self._drain_deadline = drain_deadline
        self._calls: list[threading.Thread] = []  # call threads not yet seen to finish
        self._items: deque[tuple[str, Callable[[], Any], float]] = deque()
        self._cv = threading.Condition()
        self._busy = False  # a dequeued call is still being waited on
        self._closed = False
        self._thread = threading.Thread(target=self._consume, name=f"{name}-tracking", daemon=True)
        self._warn_call = _RateLimitedWarning(_WARN_INTERVAL_S)
        self._warn_drop = _RateLimitedWarning(_WARN_INTERVAL_S)

    def start(self) -> None:
        self._thread.start()

    def submit(self, what: str, fn: Callable[[], Any], deadline: float | None = None) -> bool:
        """Queue ``fn``; False if it was dropped (queue full or worker closed)."""
        with self._cv:
            if self._closed:
                return False
            accepted = len(self._items) < self._max_queue
            if accepted:
                limit = self._call_deadline if deadline is None else deadline
                self._items.append((what, fn, limit))
                self._cv.notify_all()
        if not accepted:
            self._warn_drop(
                f"{self._name} tracking queue full ({self._max_queue}); dropping {what}"
            )
        return accepted

    def flush(self, timeout: float) -> bool:
        """Wait until every queued call has been attempted; False if ``timeout`` ran out first."""
        end = time.monotonic() + timeout
        with self._cv:
            while self._items or self._busy:
                remaining = end - time.monotonic()
                if remaining <= 0:
                    return False
                self._cv.wait(remaining)
        return True

    def close(self) -> bool:
        """Stop accepting calls and deliver the queued ones within the drain deadline."""
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        drained = self.flush(self._drain_deadline)
        if not drained:
            with self._cv:
                lost = len(self._items) + int(self._busy)
                self._items.clear()
            logger.warning(f"{self._name}: {lost} tracking call(s) not delivered before shutdown")
        return drained

    def _consume(self) -> None:
        while True:
            with self._cv:
                while not self._items:
                    if self._closed:
                        return
                    self._cv.wait()
                what, fn, deadline = self._items.popleft()
                self._busy = True
            try:
                self._run(what, fn, deadline)
            except Exception as e:  # the worker must outlive any single call
                self._warn_call(f"{self._name} {what} not run ({e}); points may be lost")
            finally:
                with self._cv:
                    self._busy = False
                    self._cv.notify_all()

    def _run(self, what: str, fn: Callable[[], Any], deadline: float) -> None:
        self._calls = [thread for thread in self._calls if thread.is_alive()]
        if len(self._calls) >= self._max_threads:
            self._warn_drop(
                f"{self._name}: {len(self._calls)} abandoned tracking calls have not returned; "
                f"dropping {what}"
            )
            return
        failure: list[Exception] = []

        def call() -> None:
            try:
                fn()
            except Exception as e:  # reported by the worker, not raised in a stray thread
                failure.append(e)

        thread = threading.Thread(target=call, name=f"{self._name}-call", daemon=True)
        thread.start()
        self._calls.append(thread)
        thread.join(deadline)
        if thread.is_alive():
            self._warn_call(
                f"{self._name} {what} still running after {deadline:g}s; abandoning it, "
                "points may be lost"
            )
        elif failure:
            self._warn_call(f"{self._name} {what} failed ({failure[0]}); points may be lost")


class _RunOwner:
    """Starts and later ends a tracking run on one dedicated thread, waiting on each for a deadline.

    mlflow keeps the active run, and the system metrics monitor tied to it, per thread, so the end
    must run on the thread that ran the start. A caller that stops waiting is never blocked again:
    a start that returns after its deadline is ended right away, and a late end is abandoned.
    """

    def __init__(self, name: str, start: Callable[[], Any], end: Callable[[], Any]) -> None:
        self._start_fn = start
        self._end_fn = end
        self._started = threading.Event()  # start returned or raised
        self._end_requested = threading.Event()
        self._ended = threading.Event()  # end returned or raised, or there was nothing to end
        self._result: Any = None
        self._start_error: Exception | None = None
        self._end_error: Exception | None = None
        self._thread = threading.Thread(target=self._own, name=f"{name}-run", daemon=True)

    def start(self, deadline: float) -> Any:
        """Return what ``start`` returned; re-raise its error, or TimeoutError past ``deadline``."""
        self._thread.start()
        if not self._started.wait(deadline):
            self._end_requested.set()  # the start still running is ended as soon as it returns
            raise TimeoutError(f"no answer from the tracking server within {deadline:g}s")
        if self._start_error is not None:
            raise self._start_error
        return self._result

    def end(self, deadline: float) -> None:
        """Run ``end`` on the owner thread; re-raise its error, or TimeoutError past deadline."""
        self._end_requested.set()
        if not self._ended.wait(deadline):
            raise TimeoutError(f"no answer from the tracking server within {deadline:g}s")
        if self._end_error is not None:
            raise self._end_error

    def _own(self) -> None:
        try:
            self._result = self._start_fn()
        except Exception as e:  # raised again by start(), on the caller's thread
            self._start_error = e
        self._started.set()
        if self._start_error is None:
            self._end_requested.wait()
            try:
                self._end_fn()
            except Exception as e:  # raised again by end()
                self._end_error = e
        self._ended.set()


def _import_mlflow() -> Any:
    """Import mlflow with bounded HTTP requests; values already set in the environment win."""
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "30")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "3")
    # The Databricks SDK transport keeps its own request timeout; this caps how long it retries.
    os.environ.setdefault("MLFLOW_DATABRICKS_ENDPOINT_HTTP_RETRY_TIMEOUT", "60")
    import mlflow

    return mlflow


_DATABRICKS_AUTH_ROUTES = (
    "Databricks auth: env DATABRICKS_HOST and DATABRICKS_TOKEN; a ~/.databrickscfg profile via "
    "a databricks://<profile> URI or DATABRICKS_CONFIG_PROFILE; SDK OAuth"
)


def _auth_hint(error: Exception) -> str:
    """Where Databricks finds credentials, appended to a failure that may be a credentials one."""
    if isinstance(error, (ImportError, TimeoutError)):  # a missing package or a slow server
        return ""
    return f" ({_DATABRICKS_AUTH_ROUTES})"


def _databricks_profile(uri: str) -> str | None:
    """The profile a ``databricks://<profile>`` tracking URI names; None for any other URI."""
    scheme, separator, rest = uri.partition("://")
    profile = rest.partition("/")[0]
    if scheme != "databricks" or not separator or not profile or ":" in profile:
        return None  # "scope:prefix" names a secret scope, not a profile
    return profile


def _prepare_databricks_auth(uri: str) -> None:
    """Make SDK and artifact calls use the URI's profile, and let the token alias stand in.

    Credentials are never read here: the SDK resolves them from the environment, a profile file
    or OAuth. An operator-set DATABRICKS_CONFIG_PROFILE or DATABRICKS_TOKEN wins.
    """
    profile = _databricks_profile(uri)
    if profile:
        os.environ.setdefault("DATABRICKS_CONFIG_PROFILE", profile)
    # The Databricks SDK reads DATABRICKS_TOKEN; mirror the DATABRICKS_API_TOKEN
    # name many keep in ~/.bashrc so either works.
    if not os.environ.get("DATABRICKS_TOKEN") and os.environ.get("DATABRICKS_API_TOKEN"):
        os.environ["DATABRICKS_TOKEN"] = os.environ["DATABRICKS_API_TOKEN"]


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
        logger.warning(
            _scrub_uris(f"Could not auto-resolve Databricks experiment path: {e}{_auth_hint(e)}")
        )
        return None


# Run metadata, not params: a user-typed tag key can be MLflow-illegal and fail its whole batch.
_PARAM_SKIP = ("metrics.mlflow_tags", "metrics.mlflow_description")


def _flatten_config_params(config: JobConfig, max_len: int = 250) -> dict[str, str]:
    """Flatten a JobConfig to dotted string keys (model.dim, ...) for mlflow.log_params.

    None is skipped; non-scalars are stringified; values truncated to max_len.
    """
    flat: dict[str, str] = {}

    def _walk(prefix: str, value: Any) -> None:
        if prefix in _PARAM_SKIP or any(prefix.startswith(p + ".") for p in _PARAM_SKIP):
            return
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
    """Highest step already logged for "train/loss" (the metric _log_step always emits), so a
    resume skips steps it already has. Returns None if absent (dedup then skipped)."""
    try:
        hist = mlflow.MlflowClient().get_metric_history(run_id, "train/loss")
        return max((m.step for m in hist), default=None)
    except Exception:
        return None


# The worker thread has no active run (mlflow keeps it in thread-local state), so each call
# below names its run; an implicit call there would silently start a new run.
def _mlflow_log_metrics(mlflow: Any, run_id: str, metrics: dict[str, float], step: int) -> None:
    # synchronous=True: MLFLOW_ENABLE_ASYNC_LOGGING would otherwise divert these into
    # mlflow's own unbounded queue with a blocking flush at shutdown; bounding lives in
    # the worker, so the call itself must stay synchronous.
    mlflow.log_metrics(
        {k: float(v) for k, v in metrics.items()}, step=step, run_id=run_id, synchronous=True
    )


def _mlflow_set_tag(mlflow: Any, run_id: str, key: str, value: str) -> None:
    # fluent set_tags has no run_id; synchronous=True for the same reason as in log_metrics
    mlflow.MlflowClient().set_tag(run_id, key, value, synchronous=True)


# mlflow's standard run tags: the description shown in its UI, and the code version.
_NOTE_TAG = "mlflow.note.content"
_GIT_COMMIT_TAG = "mlflow.source.git.commit"
_GIT_DIRTY_TAG = "mlflow.source.git.dirty"
_CONFIG_ARTIFACT = "config.json"
_GIT_TIMEOUT_S = 10.0  # per git query; both fit inside the tag call's own deadline
_GIT_SHA = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


def _git_output(*args: str) -> str | None:
    """Stdout of ``git <args>`` in the working directory; None, noted at debug level, on failure."""
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}  # no index lock for a read-only query
    try:
        done = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            errors="replace",
            stdin=subprocess.DEVNULL,
            env=env,
            timeout=_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError) as e:  # no git binary, or no answer in time
        logger.debug(f"MLflow: git {args[0]} did not run ({e})")
        return None
    if done.returncode != 0:
        logger.debug(f"MLflow: git {args[0]} exited {done.returncode}: {done.stderr.strip()}")
        return None
    return done.stdout.strip()


def _git_state() -> tuple[str, bool | None] | None:
    """(commit sha, whether tracked files differ from it or None if unknown), else None.

    Describes the repository at the working directory. Every failure (git absent, not a
    repository, no commit yet, no answer in time) yields None or an unknown flag, never an error.
    """
    sha = _git_output("rev-parse", "HEAD")
    if sha is None:
        return None
    if not _GIT_SHA.fullmatch(sha):
        logger.debug(f"MLflow: git rev-parse printed {sha!r}, not a commit id")
        return None
    changes = _git_output("status", "--porcelain", "--untracked-files=no")
    return sha, None if changes is None else bool(changes)


def _mlflow_set_git_tags(mlflow: Any, run_id: str) -> None:
    state = _git_state()
    if state is None:
        return
    commit, dirty = state
    _mlflow_set_tag(mlflow, run_id, _GIT_COMMIT_TAG, commit)
    if dirty is not None:
        _mlflow_set_tag(mlflow, run_id, _GIT_DIRTY_TAG, str(dirty).lower())


def _mlflow_log_config(mlflow: Any, run_id: str, config: dict[str, Any]) -> None:
    mlflow.MlflowClient().log_dict(run_id, config, _CONFIG_ARTIFACT)  # stringifies non-JSON values


def _launch_toml_arg() -> str | None:
    """The ``.toml`` path in argv[1], where the entry points take their config file from."""
    argv = sys.argv
    return argv[1] if len(argv) > 1 and argv[1].lower().endswith(".toml") else None


def _mlflow_log_launch_toml(mlflow: Any, run_id: str, arg: str) -> None:
    path = os.path.abspath(arg)
    if not os.path.isfile(path):
        logger.debug(f"MLflow: launch config {arg!r} is not a file; not uploaded")
        return
    mlflow.MlflowClient().log_artifact(run_id, path)


class _MlflowDisabled(Exception):
    """A deliberate config error found during init: MLflow is disabled with this message."""


class MLflowBackend(_LoggingBackend):
    """MLflow logging backend (Databricks-hosted or any MLflow tracking server).

    Lazy init on first log; run ID written back to config for checkpoint resume. Every server
    call has a deadline: the run is set up and later ended on one owner thread (the active run is
    thread-local), and metric, param, tag and artifact calls go through a best-effort worker, so a
    slow server never stalls training. A new run (not a resumed one) also gets its description and
    tags, the git commit, and the resolved config (plus the launch TOML) as artifacts.

    The tracking URI resolves as config, then $MLFLOW_TRACKING_URI, then "databricks". Databricks
    credentials are not pre-checked: init attempts the server calls, and a failure disables the
    backend with a warning that names the auth routes.
    """

    def __init__(self, config: MetricsConfig, job_config: JobConfig | None = None) -> None:
        self._config = config
        self._job_config = job_config  # params and config.json come from it, for a fresh run only
        self._uri = ""  # the resolved tracking URI, set when init starts
        self._active: bool | None = None  # None = not tried, True = live, False = failed
        self._run_started = False  # a run exists that close() must end
        self._resume_skip_below: int | None = None  # skip re-logging steps already in the run
        self._mlflow: Any = None  # the imported module
        self._run_id = ""
        self._worker = _BestEffortWorker(
            "MLflow",
            max_queue=_QUEUE_MAX,
            max_threads=_MAX_CALL_THREADS,
            call_deadline=_CALL_DEADLINE_S,
            drain_deadline=_DRAIN_DEADLINE_S,
        )
        self._owner = _RunOwner("MLflow", start=self._open_run, end=lambda: self._mlflow.end_run())

    def _ensure_init(self) -> None:
        if self._active is not None:
            return
        self._uri = self._config.resolve_mlflow_tracking_uri()
        on_databricks = self._uri.startswith("databricks")
        try:
            if on_databricks:
                _prepare_databricks_auth(self._uri)
            self._mlflow = _import_mlflow()
            # The training thread waits here at most once, for at most the init deadline.
            experiment, run_id, resumed, last_step = self._owner.start(_INIT_DEADLINE_S)
            self._run_started = True
            self._run_id = run_id
            self._config.mlflow_run_id = run_id
            self._worker.start()
            self._active = True
            logger.info(f"MLflow initialized: experiment={experiment} run_id={run_id}")
            if resumed:
                self._resume_skip_below = last_step
            else:
                self._log_run_metadata(self._mlflow)
        except _MlflowDisabled as e:
            logger.warning(_scrub_uris(f"MLflow disabled: {e}"))
            self._active = False
        except ImportError:
            logger.warning(
                "mlflow not installed — disabling MLflow backend (uv sync --group mlflow)"
            )
            self._active = False
        except Exception as e:  # mlflow talks to a remote server: network / auth / config errors
            hint = _auth_hint(e) if on_databricks else ""
            logger.warning(_scrub_uris(f"MLflow init failed: {e}{hint}"))
            self._active = False

    def _open_run(self) -> tuple[str | None, str, bool, int | None]:
        """Every server step of init, run on the owner thread under the init deadline.

        Returns (experiment, run_id, resumed, last logged step); the caller applies it, so a
        start that finishes after the caller gave up changes nothing here.
        """
        mlflow, uri = self._mlflow, self._uri
        mlflow.set_tracking_uri(uri)
        try:
            experiment = _resolve_mlflow_experiment(self._config, uri)
        except ValueError as e:  # bad $MLFLOW_EXPERIMENT: a deliberate config error
            raise _MlflowDisabled(str(e)) from e
        if uri.startswith("databricks") and not experiment:
            raise _MlflowDisabled(
                "could not resolve a Databricks experiment path "
                "(set mlflow_experiment or $MLFLOW_EXPERIMENT to an absolute path)"
            )
        mlflow.set_experiment(experiment)  # never empty here: off Databricks a name always resolves
        if self._config.mlflow_log_system_metrics:
            mlflow.set_system_metrics_sampling_interval(self._config.mlflow_system_metrics_interval)
        run, resumed = self._start_run(mlflow)
        run_id = run.info.run_id
        last_step = _mlflow_last_step(mlflow, run_id) if resumed else None
        return experiment, run_id, resumed, last_step

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
                logger.warning(
                    _scrub_uris(f"MLflow run_id={run_id} is gone ({e}); starting a new run")
                )
        return mlflow.start_run(**kwargs), False

    def _log_run_metadata(self, mlflow: Any) -> None:
        """Queue a new run's params, tags and config artifacts; each call fails on its own.

        The training thread only snapshots values here. Everything slow (git, file access, the
        server) runs on the worker, so this never waits on any of it.
        """
        import socket

        hyperparams = _flatten_config_params(self._job_config) if self._job_config else {}
        items = list(hyperparams.items())
        # log_params caps at 100 per batch; one call per batch so a bad batch keeps the rest.
        for i in range(0, len(items), 100):
            batch = dict(items[i : i + 100])
            self._worker.submit(
                "log_params",
                partial(mlflow.log_params, batch, run_id=self._run_id, synchronous=True),
            )
        self._worker.submit("set_git_tags", partial(_mlflow_set_git_tags, mlflow, self._run_id))
        tags = {"host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID", "")}
        if self._config.mlflow_description:
            tags[_NOTE_TAG] = self._config.mlflow_description
        tags.update(self._config.mlflow_tags)  # configured tags override the automatic ones
        for key, value in tags.items():
            call = partial(_mlflow_set_tag, mlflow, self._run_id, key, value)
            self._worker.submit(f"set_tag {key}", call)
        if self._job_config is not None:
            call = partial(_mlflow_log_config, mlflow, self._run_id, asdict(self._job_config))
            self._worker.submit(f"upload {_CONFIG_ARTIFACT}", call, deadline=_UPLOAD_DEADLINE_S)
        toml_arg = _launch_toml_arg()
        if toml_arg is None:
            logger.debug("MLflow: no launch config on the command line; none uploaded")
        else:
            call = partial(_mlflow_log_launch_toml, mlflow, self._run_id, toml_arg)
            self._worker.submit("upload launch config", call, deadline=_UPLOAD_DEADLINE_S)

    def log(self, metrics: dict[str, float], step: int) -> None:
        self._ensure_init()
        if self._active is not True:
            return
        if self._resume_skip_below is not None and step <= self._resume_skip_below:
            return  # already logged before this resume — avoid duplicate history points
        call = partial(_mlflow_log_metrics, self._mlflow, self._run_id, dict(metrics), step)
        self._worker.submit("log_metrics", call)

    def close(self) -> None:
        if not self._run_started:
            return
        try:
            self._worker.close()
            self._owner.end(_END_DEADLINE_S)
        except Exception as e:  # teardown must not fail the job
            logger.warning(_scrub_uris(f"MLflow end_run failed ({e})"))
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
