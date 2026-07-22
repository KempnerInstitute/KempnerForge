# MLflow backend

[`MLflowBackend`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/metrics/tracker.py)
logs metrics, params, and system utilization to a Databricks-hosted MLflow, so we
don't run a tracking server ourselves. It is opt-in and can run alongside WandB.

```toml
[metrics]
enable_mlflow     = true
mlflow_experiment = "/Experiments/my-project"   # absolute workspace path on Databricks
mlflow_run_name   = "7b-debug"                  # optional — MLflow auto-generates one
```

Credentials come from the environment, never from config:

```bash
export DATABRICKS_HOST="https://<workspace>.cloud.databricks.com"
export DATABRICKS_API_TOKEN="dapi..."           # or DATABRICKS_TOKEN
```

The backend is constructed by `MetricsTracker._init_backends` on rank 0 only;
other ranks never touch MLflow.

## Tracking URI and experiment

`mlflow_tracking_uri` defaults to `"databricks"`, which reads `DATABRICKS_HOST` +
a token from the environment. If it is `"databricks"` but no credentials are set,
the backend logs a warning and disables itself — it does not invent a local store.
Point `mlflow_tracking_uri` at an MLflow server (`http(s)://…`) to log elsewhere.

The client is `mlflow-skinny` (tracking-only; no server or database backends). To
log to a local SQLite store instead, install full `mlflow` and set
`mlflow_tracking_uri = "sqlite:///mlflow.db"`.

The experiment name resolves in order: `mlflow_experiment` → `$MLFLOW_EXPERIMENT`
→ auto. On Databricks the name **must be an absolute workspace path**
(`/Users/you@example.com/proj` or `/Experiments/proj`); a bare name fails, so a
non-absolute experiment is rejected — `MetricsConfig.__post_init__` aborts at config-load
time for the config field, and the backend disables with a clear message if
`$MLFLOW_EXPERIMENT` is not absolute at runtime. When none is set, auto derives
`/Users/<databricks-username>/<wandb_project>` via the Databricks SDK.

## Init is lazy

`__init__` just stashes the config — no network I/O. The first call to `log()`
runs `_ensure_init`, which sets the tracking URI + experiment and calls
`mlflow.start_run(...)`:

```python
# kempnerforge/metrics/tracker.py
run = mlflow.start_run(
    run_id=self._config.mlflow_run_id or None,   # resume if set
    run_name=self._config.mlflow_run_name,
    log_system_metrics=self._config.mlflow_log_system_metrics,
)
self._config.mlflow_run_id = run.info.run_id     # write back for checkpoint
```

Same two reasons as WandB: distributed setup and checkpoint resume finish before
the first step logs, and constructing a tracker never touches the network in
CI/tests.

## Run ID and resume

Mirrors [WandB](wandb.md#run-id-and-resume). On a fresh run, `start_run()` with
no `run_id` mints a new one; it is written back to `MetricsConfig` and the
training loop persists it in `ckpt_extra`:

```python
# scripts/train.py
if config.metrics.mlflow_run_id:
    ckpt_extra["mlflow_run_id"] = config.metrics.mlflow_run_id
```

On resume it is restored right after `ckpt_mgr.load(...)`:

```python
if ckpt_extra_loaded.get("mlflow_run_id"):
    config.metrics.mlflow_run_id = ckpt_extra_loaded["mlflow_run_id"]
```

`start_run(run_id=...)` then reattaches to the same run, so metrics continue on
one curve across Slurm restarts. If that run id no longer exists (e.g. it was
deleted), the backend falls back to a fresh run instead of erroring — matching
wandb's `resume="allow"`. On resume it also skips re-logging any step the run already
recorded, so re-executing the steps between the last checkpoint and the crash does not
duplicate metric points.

## What gets logged

- **Metrics** — the same backend dict as WandB/TensorBoard (`train/*`, `gpu/*`,
  `smoothed/*`), plus everything passed to `tracker.log_eval`, via
  `mlflow.log_metrics(dict, step=step)`.
- **Params** — the full `JobConfig` flattened to dotted keys (`model.dim`,
  `optimizer.lr`, …) once at run start, via `mlflow.log_params`. Cheap searchable
  metadata, not artifacts.
- **Tags** — `host` and `slurm_job_id`.
- **System metrics** — CPU / GPU / memory under `system/`, sampled on a
  background thread when `mlflow_log_system_metrics = true` (default). Lower
  `mlflow_system_metrics_interval` (seconds) for short test runs.

No checkpoints are uploaded as artifacts — they stay on the filesystem (artifacts
are the main MLflow storage cost).

## Failure modes

`_ensure_init` wraps setup in try/except and sets `self._active = False` on any
failure:

```python
except ImportError:
    logger.warning("mlflow not installed — disabling MLflow backend (uv sync --group mlflow)")
    self._active = False
except Exception as e:     # network, auth, experiment-path, etc.
    logger.warning(f"MLflow init failed: {e}")
    self._active = False
```

`log()` never propagates and does not disable the backend on a transient error: it
warns once and keeps retrying, so logging recovers after a blip. `close()` always ends
the run (even if logging was disabled) and swallows teardown errors. Param/tag logging
is separately guarded as well.

MLflow is an optional dependency: install with `uv sync --group mlflow` (pulls
`mlflow-skinny` + `databricks-sdk`). Without it, `enable_mlflow = true` logs a
warning and no-ops.

## `close()`

Called from `tracker.close()` at the end of training; runs `mlflow.end_run()`.
No-op unless the run is live.

## See also

- [Metrics tracker](metrics-tracker.md) — where backends are constructed.
- [WandB](wandb.md) — the sibling backend with the same metric dict and
  run-id-resume pattern.
- [Checkpointing § Train state](../checkpointing/train-state.md) — how
  `mlflow_run_id` travels with the checkpoint.
- [Configuration § `[metrics]`](../configuration/config-sections.md) — per-field
  reference for `MetricsConfig`.
