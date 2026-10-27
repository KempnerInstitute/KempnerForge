"""The MLflow logging backend, for Databricks or any MLflow tracking server.

``MetricsTracker`` builds ``MLflowBackend`` on rank 0 when ``metrics.enable_mlflow`` is on. Its
server calls run on the threads of ``kempnerforge.metrics.mlflow.worker``, each under a deadline.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

from kempnerforge.config.schema import JobConfig, MetricsConfig
from kempnerforge.metrics.logger import get_logger
from kempnerforge.metrics.mlflow.auth import _auth_hint, _prepare_databricks_auth
from kempnerforge.metrics.mlflow.models import (
    checkpoint_model_name,
    find_or_create_checkpoint_model,
)
from kempnerforge.metrics.mlflow.worker import (
    _WARN_INTERVAL_S,
    _BestEffortWorker,
    _RateLimitedWarning,
    _RunOwner,
)
from kempnerforge.metrics.tracker import (
    _URI_USERINFO,
    BackendRecord,
    _error_text,
    _LoggingBackend,
    _scrub_uris,
)

logger = get_logger(__name__)

# Tracking is a side channel: a slow or unreachable server may cost points, never a stalled rank.
_CALL_DEADLINE_S = 60.0  # one metric / param / tag call
_UPLOAD_DEADLINE_S = 180.0  # one artifact upload
_INIT_DEADLINE_S = 60.0  # experiment setup plus run start or resume, on the first log
_END_DEADLINE_S = 60.0  # ending the run in close(), after the drain
_REGISTER_DEADLINE_S = 180.0
_DRAIN_DEADLINE_S = 30.0  # close() flush budget
_QUEUE_MAX = 256  # pending calls before new ones are dropped
_MAX_CALL_THREADS = 4  # live call threads, abandoned ones included, before new calls are dropped


def _import_mlflow() -> Any:
    """Import mlflow with bounded HTTP requests; values already set in the environment win."""
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "30")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "3")
    # The Databricks SDK transport keeps its own request timeout; this caps how long it retries.
    os.environ.setdefault("MLFLOW_DATABRICKS_ENDPOINT_HTTP_RETRY_TIMEOUT", "60")
    import mlflow

    return mlflow


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


_SIDECAR_FILE = "metadata.json"


class MLflowBackend(_LoggingBackend):
    """MLflow logging backend (Databricks-hosted or any MLflow tracking server).

    Lazy init on first log; run ID written back to config for checkpoint resume. Every server
    call has a deadline: the run is set up and later ended on one owner thread (the active run is
    thread-local), and metric, param, tag and artifact calls go through a best-effort worker, so a
    slow server never stalls training. A new run (not a resumed one) also gets its description and
    tags, the git commit, and the resolved config (plus the launch TOML) as artifacts. With
    ``mlflow_register_checkpoints``, the same worker registers each committed checkpoint as an
    external model, so a save never waits on the server for it.

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
        self._run_name = ""
        self._experiment = ""  # the experiment the run was set up in, set with the run id
        self._worker = _BestEffortWorker(
            "MLflow",
            max_queue=_QUEUE_MAX,
            max_threads=_MAX_CALL_THREADS,
            call_deadline=_CALL_DEADLINE_S,
            drain_deadline=_DRAIN_DEADLINE_S,
        )
        self._owner = _RunOwner("MLflow", start=self._open_run, end=lambda: self._mlflow.end_run())
        self._register_lock = threading.Lock()
        self._model_ids: dict[int, str] = {}
        self._registering: set[int] = set()
        self._warn_register = _RateLimitedWarning(_WARN_INTERVAL_S)

    def _ensure_init(self) -> None:
        if self._active is not None:
            return
        self._uri = self._config.resolve_mlflow_tracking_uri()
        on_databricks = self._uri.startswith("databricks")
        try:
            _prepare_databricks_auth(self._uri)
            self._mlflow = _import_mlflow()
            # The training thread waits here at most once, for at most the init deadline.
            experiment, run_id, run_name, resumed, last_step = self._owner.start(_INIT_DEADLINE_S)
            self._run_started = True
            self._run_id = run_id
            self._run_name = run_name or ""
            self._experiment = experiment or ""
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
                "mlflow could not be imported; disabling MLflow backend "
                "(the environment is incomplete, run: uv sync)"
            )
            self._active = False
        except Exception as e:  # mlflow talks to a remote server: network / auth / config errors
            hint = _auth_hint(e) if on_databricks else ""
            logger.warning(_scrub_uris(f"MLflow init failed: {e}{hint}"))
            self._active = False

    def _open_run(self) -> tuple[str | None, str, str | None, bool, int | None]:
        """Every server step of init, run on the owner thread under the init deadline.

        Returns (experiment, run_id, run name, resumed, last logged step); the caller applies it,
        so a start that finishes after the caller gave up changes nothing here.
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
        return experiment, run_id, run.info.run_name, resumed, last_step

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

    def tracking_entry(self) -> dict[str, str] | None:
        """The live run as its checkpoint reader needs it: all four fields non-blank, or none."""
        entry = {
            "backend": "mlflow",
            "run_id": self._run_id,
            "experiment": self._experiment,
            "tracking_uri": _URI_USERINFO.sub("", self._uri),
        }
        if self._active is not True or not all(value.strip() for value in entry.values()):
            return None
        return entry

    def _registers_checkpoints(self) -> bool:
        return self._config.mlflow_register_checkpoints and self.tracking_entry() is not None

    def checkpoint_fields(self, step: int) -> dict[str, str] | None:
        """The model name of the checkpoint saved at ``step``, plus its id once registered.

        Only with ``mlflow_register_checkpoints`` on and a live run; answered from memory.
        """
        if not self._registers_checkpoints():
            return None
        fields = {"model_name": checkpoint_model_name(self._run_name or self._run_id, step)}
        with self._register_lock:
            model_id = self._model_ids.get(step)
        if model_id is not None:
            fields["model_id"] = model_id
        return fields

    def register_checkpoint(self, step: int, directory: Path, record: BackendRecord) -> None:
        """Find or make the external model of a committed checkpoint, then record its id.

        Only with ``mlflow_register_checkpoints`` on and a live run, and it never waits: a step
        registered before is recorded at once from memory, and any other is queued on the worker,
        which records the id when the server answers. A failure warns (rate limited) and costs
        only the id; a later commit of the same step tries again.
        """
        if not self._registers_checkpoints():
            return
        with self._register_lock:
            model_id = self._model_ids.get(step)
        if model_id is not None:
            record({"model_id": model_id})
            return
        name = checkpoint_model_name(self._run_name or self._run_id, step)
        call = partial(self._register, step, name, directory.absolute(), record)
        if not self._worker.submit(f"register {name}", call, deadline=_REGISTER_DEADLINE_S):
            self._warn_register(
                f"MLflow checkpoint registration at step {step} was not queued; "
                "its tracking entry has no model_id"
            )

    def _register(self, step: int, name: str, directory: Path, record: BackendRecord) -> None:
        """Worker call: find or make the model and record its id; failures only warn.

        The step is marked in flight while this call runs, so a call queued behind it for the
        same step (a step committed twice) records its id, or leaves it to this one if it is
        still running past its deadline; a call the worker drops never leaves a mark.
        """
        with self._register_lock:
            model_id = self._model_ids.get(step)
            running = model_id is None and step in self._registering
            if model_id is None and not running:
                self._registering.add(step)
        if model_id is not None:
            record({"model_id": model_id})
            return
        if running:
            return
        try:
            if not (directory / _SIDECAR_FILE).exists():
                logger.debug(f"MLflow: {directory} was removed before it was registered")
                return
            client = self._mlflow.MlflowClient(tracking_uri=self._uri)
            model_id = find_or_create_checkpoint_model(
                client, self._run_id, name, step, str(directory)
            )
            with self._register_lock:
                self._model_ids[step] = model_id
            record({"model_id": model_id})
        except Exception as e:
            self._warn_register(
                f"MLflow checkpoint registration did not complete at step {step} "
                f"({_error_text(e)}); its tracking entry has no model_id"
            )
        finally:
            with self._register_lock:
                self._registering.discard(step)

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
