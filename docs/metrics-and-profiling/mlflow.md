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

Credentials never go in config. The Databricks SDK finds them in the environment, a
`~/.databrickscfg` profile, or OAuth; from the environment:

```bash
export DATABRICKS_HOST="https://<workspace>.cloud.databricks.com"
export DATABRICKS_API_TOKEN="dapi..."           # or DATABRICKS_TOKEN
```

The backend is constructed by `MetricsTracker._init_backends` on rank 0 only;
other ranks never touch MLflow.

## Tracking URI and experiment

The tracking URI resolves in order: `mlflow_tracking_uri` → `$MLFLOW_TRACKING_URI` →
`"databricks"`. `databricks://<profile>` selects a `~/.databrickscfg` profile and exports it
as `DATABRICKS_CONFIG_PROFILE` (an operator-set value wins), so SDK and artifact calls use
the same profile. `DATABRICKS_API_TOKEN` is mirrored into `DATABRICKS_TOKEN`, the name the
SDK reads, when that is unset. The SDK prefers environment variables over a profile file, so
unset a stale `DATABRICKS_API_TOKEN` when authenticating with a profile.

Credentials are not pre-checked: init attempts the connection, and if it fails the backend
logs a warning naming the auth routes (environment, profile, SDK OAuth) and disables itself;
it does not invent a local store. Point `mlflow_tracking_uri` at an MLflow server
(`http(s)://…`) to log elsewhere.

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
# training/loop.py::checkpoint_extra
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

## Finding a checkpoint's run

Each checkpoint's `metadata.json` names the live run in its `tracking` list
([Checkpointing](../checkpointing/index.md)):

```json
{"backend": "mlflow", "run_id": "<run id>", "experiment": "<experiment>", "tracking_uri": "<tracking uri>"}
```

All four fields are always present and non-empty, or the entry is left out. `tracking_uri` is
the URI the backend resolved (config, then `$MLFLOW_TRACKING_URI`, then `"databricks"`) and
`experiment` the name it set the run up under. The entry comes from
`MLflowBackend.tracking_entry()`; resume is unaffected and still restores `mlflow_run_id` from
`train_state.pt`.

## Registering checkpoints

`mlflow_register_checkpoints = true` (default `false`; requires `enable_mlflow`) makes each
committed checkpoint an MLflow external logged model, so the run lists its checkpoints:

- name `<run name>-step_<N>` (characters MLflow rejects in a model name become `_`, and the run
  id stands in for an empty run name), `model_type = "checkpoint"`, linked to the run;
- params `step` and `checkpoint_dir`, the absolute path the checkpoint was written to.

Nothing is uploaded but the small descriptor MLflow keeps with any external model. The weights
stay where they were saved, and `checkpoint.keep_last_n` can later delete the directory the
model points at.

Each save records the model's name in its `metadata.json`, in the `mlflow` entry of `tracking`
([Checkpointing](../checkpointing/index.md)). Registration starts once the checkpoint is
committed (`latest` points at it; with an async checkpoint mode, once its shards are durable),
runs on the background worker, and adds the `model_id` to that entry when the server answers,
replacing the file atomically:

```json
{"backend": "mlflow", "run_id": "<run id>", "experiment": "<experiment>", "tracking_uri": "<tracking uri>", "model_name": "<run name>-step_<N>", "model_id": "<model id>"}
```

A save never waits on the server for it, a preemption save included. Registration is best
effort: a server error leaves the entry without `model_id` and logs a rate-limited warning, and
the next commit of the same step tries again. A step saved twice (a shutdown save right after an
interval save, or a resumed run re-running it) keeps one model, and the last `metadata.json` of
that step still gets its `model_id`. A resume registers nothing by itself, and a checkpoint
deleted before its turn comes is skipped. At shutdown, `close()` delivers a registration still
queued within its drain budget, so the final checkpoint is registered too. A save before the
run's first log has no `mlflow` entry, so its `metadata.json` gets neither field. MLflow itself
logs two harmless "Failed to infer schema" warnings for each external model it creates.

The eval publisher (`python -m kempnerforge.metrics.publish`) links its metrics to the entry's
`model_id`. When the entry has only `model_name` (the registration failed or never finished),
it finds the run's model of that name, or creates it, and links to that; it never writes
`metadata.json`.

## What gets logged

- **Metrics** — the same backend dict as WandB/TensorBoard (`train/*`, `gpu/*`,
  `smoothed/*`), plus everything passed to `tracker.log_eval`, via
  `mlflow.log_metrics(dict, step=step)`.
- **Params** — the full `JobConfig` flattened to dotted keys (`model.dim`,
  `optimizer.lr`, …) once at run start, via `mlflow.log_params`. Cheap searchable
  metadata, not artifacts.
- **Tags**: `host` and `slurm_job_id`; the git commit of the working directory as
  `mlflow.source.git.commit`, with `mlflow.source.git.dirty` (`true` when tracked files differ
  from it), skipped quietly when git or a repository is missing; `mlflow_description` as the
  run description (`mlflow.note.content`); and every entry of `mlflow_tags`. Configured tags are
  applied last, so one replaces an automatic tag of the same key.
- **Artifacts**: `config.json`, the resolved `JobConfig` (TOML, CLI overrides and defaults
  applied), and the launch TOML when the first command-line argument names a `.toml` file
  (how `scripts/train.py` takes its config); without one that upload is skipped. Each upload
  has a 180 s deadline on the background worker.
- **System metrics** — CPU / GPU / memory under `system/`, sampled on a
  background thread when `mlflow_log_system_metrics = true` (default). Lower
  `mlflow_system_metrics_interval` (seconds) for short test runs.

Params, tags and artifacts are written once, when a new run starts; a resumed run keeps
what it has. No checkpoints are uploaded as artifacts: they stay on the filesystem
(artifacts are the main MLflow storage cost); [registering](#registering-checkpoints) records
where they are.

```toml
[metrics]
mlflow_description = "stage-2 video run"            # Markdown; the run description in the UI
mlflow_tags        = { team = "vlm", stage = "2" }  # non-empty strings, name to value
```

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
the run (even if logging was disabled) and swallows teardown errors. Param, tag and
artifact logging is separately guarded as well.

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
- [Checkpointing § `metadata.json`](../checkpointing/index.md): the `tracking` entry that
  names a checkpoint's run.
- [Configuration § `[metrics]`](../configuration/config-sections.md) — per-field
  reference for `MetricsConfig`.
