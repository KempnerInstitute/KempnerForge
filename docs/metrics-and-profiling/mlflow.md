# MLflow backend

[`MLflowBackend`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/metrics/tracker.py)
logs metrics, params, and system utilization to MLflow. The default is a Databricks-hosted
workspace, so we don't run a tracking server ourselves, but any MLflow tracking server works. It
is opt-in. The logging backends are independent toggles, so WandB, TensorBoard and MLflow can be
enabled together in one run config. Each receives the same metrics, and a checkpoint's
`metadata.json` lists every live run ([Checkpointing](../checkpointing/index.md)).

## Setup

```bash
uv sync --group mlflow    # mlflow-skinny, databricks-sdk, psutil, nvidia-ml-py
```

```toml
[metrics]
enable_mlflow     = true
mlflow_experiment = "/Experiments/my-project"   # absolute workspace path on Databricks
mlflow_run_name   = "7b-debug"                  # optional; MLflow auto-generates one
```

Then give the job credentials by one of the routes below. The backend is constructed by
`MetricsTracker._init_backends` on rank 0 only; other ranks never touch MLflow. Without the
`mlflow` group, `enable_mlflow = true` logs a warning and no-ops.

## Authentication

Credentials never go in config. The Databricks SDK finds them, so each user brings their own
by one of three routes:

| Route | You provide | Used when |
|-------|-------------|-----------|
| Environment | `DATABRICKS_HOST` and `DATABRICKS_TOKEN` (a personal access token) | both are set; they win over a profile |
| Profile | a `[<profile>]` section with `host` and `token` in `~/.databrickscfg` | the tracking URI is `databricks://<profile>`, or `$DATABRICKS_CONFIG_PROFILE` names it; otherwise the SDK's default profile, `[DEFAULT]` |
| SDK OAuth | a service principal (`DATABRICKS_HOST`, `DATABRICKS_CLIENT_ID`, `DATABRICKS_CLIENT_SECRET`), or a user login from the Databricks CLI (`databricks auth login`, with the CLI on `PATH` where the job runs) | no token is configured and OAuth is |

```bash
# environment route
export DATABRICKS_HOST="https://<workspace>.cloud.databricks.com"
export DATABRICKS_TOKEN="<personal-access-token>"
```

```ini
# ~/.databrickscfg, profile route
[<profile>]
host  = https://<workspace>.cloud.databricks.com
token = <personal-access-token>
```

The file holds tokens, so run `chmod 600 ~/.databrickscfg`. Select a profile with
`mlflow_tracking_uri = "databricks://<profile>"` (or the same value in `$MLFLOW_TRACKING_URI`),
or with `export DATABRICKS_CONFIG_PROFILE=<profile>`.

The SDK reads environment variables first and uses a profile only to fill in what they leave
unset, so an exported `DATABRICKS_TOKEN` overrides the profile's token. `databricks://<profile>`
also exports the profile as `DATABRICKS_CONFIG_PROFILE` for the SDK and artifact calls, unless
that variable is already set; if you set it yourself, keep it equal to the URI's profile.

```{warning}
`DATABRICKS_API_TOKEN` is not a name the SDK reads. The training backend mirrors it into
`DATABRICKS_TOKEN` when that is unset and the tracking URI is Databricks, and from then on it is
the environment route: it overrides a profile's token, so a stale export left in `~/.bashrc`
fails authentication even though the profile itself is fine. Unset stale exports
(`unset DATABRICKS_API_TOKEN DATABRICKS_TOKEN`) when you authenticate with a profile. Export
`DATABRICKS_TOKEN` rather than the alias for anything else that talks to the workspace: the
[publisher](#publishing-eval-results) does not mirror it.
```

`uv run python scripts/check_env.py --requires mlflow` reports which route a shell has. It
passes on the environment pair (`DATABRICKS_HOST` and a token under either name), or on a
profile section found in `~/.databrickscfg` (the profile named by a `databricks://<profile>`
`$MLFLOW_TRACKING_URI`, else `$DATABRICKS_CONFIG_PROFILE`, else `DEFAULT`). With neither it
warns without failing, naming the three routes, because it cannot verify OAuth offline. It
reads only section names from the profile file, never the tokens, and it looks at the
environment, not at `mlflow_tracking_uri` in a config. Add `--check-credentials` to also call
the workspace through the SDK.

Credentials are not pre-checked. Init, on the first `log()`, attempts the connection, and if it
fails the backend logs a warning, naming the auth routes when the failure may be a credentials
one, and disables itself; it does not invent a local store.

## Tracking URI and experiment

Both resolve the same way: an explicit config value, then the environment, then a default.

| | Config | Environment | Default |
|---|--------|-------------|---------|
| Tracking URI | `mlflow_tracking_uri` | `$MLFLOW_TRACKING_URI` | `"databricks"` |
| Experiment | `mlflow_experiment` | `$MLFLOW_EXPERIMENT` | auto (below) |

A config value, even `"databricks"`, wins over the environment, and an empty environment
variable counts as unset. The tracking URI is `databricks` (the workspace the SDK resolves),
`databricks://<profile>` (that `~/.databrickscfg` profile), or an `http(s)://…` MLflow server to
log elsewhere. The client is `mlflow-skinny` (tracking-only; no server or database backends); to
log to a local SQLite store instead, install full `mlflow` and set
`mlflow_tracking_uri = "sqlite:///mlflow.db"`.

On Databricks the experiment name **must be an absolute workspace path**
(`/Users/you@example.com/proj` or `/Experiments/proj`); a bare name fails, so a non-absolute
experiment is rejected. `MetricsConfig.__post_init__` aborts at config-load time for the config
field, and the backend disables with a clear message if `$MLFLOW_EXPERIMENT` is not absolute at
runtime. When none is set, auto derives `/Users/<databricks-username>/<wandb_project>` through
the Databricks SDK (off Databricks, the bare `wandb_project` name). It needs working
credentials: when the user cannot be resolved the backend disables itself and asks for an
explicit path.

## Checking the server from a compute node

Compute nodes can reach less of the network than a login node, so check the tracking server from
one before a long run. This calls it with the same client the backend uses, for any route and
any URI; run it on the partition the training job will use, with `$MLFLOW_TRACKING_URI`
exported if your config sets `mlflow_tracking_uri`:

```bash
srun -p <partition> -N 1 -n 1 -t 00:05:00 uv run --group mlflow python -c \
  'import os, mlflow; print(mlflow.MlflowClient(os.environ.get("MLFLOW_TRACKING_URI") or "databricks").search_experiments(max_results=1))'
```

A printed list (possibly empty) means the node reaches the server and the credentials work. A
connection error means the node cannot reach it; an authentication error points back at the
routes above.

## Init is lazy

`__init__` just stashes the config, so there is no network I/O. The first call to `log()` runs
`_ensure_init`, which resolves the tracking URI, prepares Databricks auth (the profile export
and the token alias above), imports `mlflow`, and starts the run on one dedicated owner thread,
waiting at most 60 s. On that thread it sets the tracking URI, resolves and sets the
experiment, sets the system-metrics sampling interval, starts the run (or resumes
`mlflow_run_id`), and, when resuming, reads the last step already logged. The same thread ends
the run in `close()`, because mlflow keeps the active run, and its system-metrics monitor, per
thread.

If the start does not answer in time, the backend disables itself with a warning, and a start
that returns late is ended at once. Same two reasons as WandB for laziness: distributed setup
and checkpoint resume finish before the first step logs, and constructing a tracker never
touches the network in CI/tests.

## Run ID and resume

Mirrors [WandB](wandb.md#run-id-and-resume). On a fresh run, `start_run()` with no `run_id`
mints a new one; `_ensure_init` writes it back to `MetricsConfig` and the training loop
persists it in `ckpt_extra`:

```python
# kempnerforge/metrics/tracker.py: MLflowBackend._ensure_init
experiment, run_id, resumed, last_step = self._owner.start(_INIT_DEADLINE_S)
...
self._config.mlflow_run_id = run_id      # write back for checkpoint
```

```python
# training/loop.py::checkpoint_extra
if config.metrics.mlflow_run_id:
    extra["mlflow_run_id"] = config.metrics.mlflow_run_id
```

On resume it is restored right after `ckpt_mgr.load(...)`:

```python
if ckpt_extra_loaded.get("mlflow_run_id"):
    config.metrics.mlflow_run_id = ckpt_extra_loaded["mlflow_run_id"]
```

`start_run(run_id=...)` then reattaches to the same run, so metrics continue on one curve across
Slurm restarts. If that run id no longer exists (e.g. it was deleted), the backend falls back to
a fresh run instead of erroring, matching wandb's `resume="allow"`; any other error (a network
or server problem) disables the backend rather than fork a new run. On resume it also skips
re-logging any step the run already recorded, so re-executing the steps between the last
checkpoint and the crash does not duplicate metric points, and it does not write the params and
tags again.

## Finding a checkpoint's run

Each checkpoint's `metadata.json` names the live run in its `tracking` list
([Checkpointing](../checkpointing/index.md)):

```json
{"backend": "mlflow", "run_id": "<run id>", "experiment": "<experiment>", "tracking_uri": "<tracking uri>"}
```

All four fields are always present and non-empty, or the entry is left out. `tracking_uri` is
the URI the backend resolved (config, then `$MLFLOW_TRACKING_URI`, then `"databricks"`) with
any `user:token@` credentials stripped, and `experiment` the name it set the run up under. The
entry comes from `MLflowBackend.tracking_entry()`; resume is unaffected and still restores
`mlflow_run_id` from `train_state.pt`. Offline tools select on it, as the
[publisher](#publishing-eval-results) does.

## Registering checkpoints

`mlflow_register_checkpoints = true` (default `false`; requires `enable_mlflow`) makes each
checkpoint save also record an MLflow external logged model, so the run lists its checkpoints:

- name `<run name>-step_<N>` (characters MLflow rejects in a model name become `_`, and the run
  id stands in for an empty run name), `model_type = "checkpoint"`, linked to the run;
- params `step` and `checkpoint_dir`, the absolute path the checkpoint was written to.

Nothing is uploaded but the small descriptor MLflow keeps with any external model. The weights
stay where they were saved, and `checkpoint.keep_last_n` can later delete the directory the
model points at. With an async checkpoint mode the model is made when the save is dispatched,
before its shards are durable; `latest` still marks the last complete checkpoint.

The model id goes into that save's `metadata.json`, in the `mlflow` entry of `tracking`
([Checkpointing](../checkpointing/index.md)):

```json
{"backend": "mlflow", "run_id": "<run id>", "experiment": "<experiment>", "tracking_uri": "<tracking uri>", "model_id": "<model id>"}
```

Registration is best effort and never fails a save. `save()` waits for it on rank 0 for at most
15 seconds, a preemption save included, while the other ranks wait at the save barrier. A server
error or a timeout leaves that save's entry without `model_id` and logs a rate-limited warning,
and while a timed-out call is still running, later saves skip registering. A step saved twice (an
emergency save, or a resumed run re-running it) keeps the model it already has, and a resume
registers nothing by itself. A save before the run's first log has no live run and registers
nothing. MLflow itself logs two harmless "Failed to infer schema" warnings for each external model
it creates.

**Models-tab leaderboard.** The id is also what lets eval scores be compared per checkpoint. The
[publisher](#publishing-eval-results) links every metric it appends to the model its checkpoint
registered, so the scores attach to per-checkpoint models: the experiment's Models tab can list
the checkpoints with their scores side by side and rank them, and
`MlflowClient.search_logged_models` filters and orders them by `metrics.<name>` from code.
Without registration the scores still land in the run at the checkpoint's step, but nothing ties
them to a per-checkpoint model.

## Publishing eval results

Evaluation runs on a checkpoint after training, so its scores belong on the run that produced it.
`kempnerforge.metrics.publish` appends metrics to the MLflow run a checkpoint's `metadata.json`
records, at the checkpoint's step:

```bash
uv run python -m kempnerforge.metrics.publish <checkpoint_dir> --metrics <scores.json>
```

`<checkpoint_dir>` is a step directory (the one holding `metadata.json`), and `<scores.json>` a
JSON object mapping a metric name to a number:

```json
{"heldout_loss": 2.31, "heldout_acc": 0.512}
```

Each name gets the `--prefix` (default `eval/`, may be empty), so the run gains
`eval/heldout_loss` and `eval/heldout_acc` at that step. The command prints
`published N metric(s) to run <run id> at step <step>` and exits 0. It exits 2 with `refused: …`
when nothing was sent because the input is unusable, and 1 with `failed: …` when MLflow or the
server errored; URLs in either message are masked.

From Python, `publish_metrics(checkpoint_dir, metrics, prefix="eval/")` does the same and returns
the target; `read_mlflow_target(checkpoint_dir)` returns it without publishing: the run id,
experiment, tracking URI, step and model id.

- **What it reads.** The single `mlflow` entry of the `tracking` list, whose `run_id`,
  `experiment` and `tracking_uri` must be non-empty strings, and `model_id` when present; plus a
  non-negative integer `step`. Nothing is inferred, and a key repeated anywhere in the file is
  refused.
- **What it sends.** One `log_batch` request through a client bound to the recorded tracking URI,
  after every metric has been checked: a non-empty name, a finite number, at most 1000 in all.
  When the entry has a `model_id`, the metrics are linked to that model. It never starts, ends or
  edits a run and keeps no state, so several publishers can write to one run at once. The call
  blocks on the server under MLflow's own HTTP timeouts, so run it as a separate process, never
  from the training step. `log_batch` is not atomic: a request that fails server side may still
  have stored some points.
- **When it refuses.** The directory has no `metadata.json`; there is no `tracking` list or no
  `mlflow` entry (a checkpoint saved before the run's first log, an older one, or training
  without MLflow); more than one `mlflow` entry; a field is missing or empty; the metrics are
  malformed; `mlflow` is not installed; or the sidecar has no `model_id` while the process has an
  active MLflow model, which `log_batch` would silently link the metrics to.
- **Credentials.** It authenticates like any MLflow client, from the
  [routes above](#authentication), and uses the recorded `tracking_uri` as written, so a
  `databricks://<profile>` entry selects that profile on the machine that publishes: give the
  publisher a profile of the same name. Export `DATABRICKS_TOKEN`; the publisher, unlike the
  training backend, does not mirror `DATABRICKS_API_TOKEN`.

A harness that scores checkpoints can publish through the same functions. Anything beyond
metrics, such as a summary table or a `scores.json`, goes to the same run through
`mlflow.MlflowClient(tracking_uri=target.tracking_uri)`: `log_dict` for a dict, `log_table` for a
table (it needs pandas, `uv sync --group eval`). Give each evaluation its own directory, for
example `eval/step_<N>/<UTC timestamp>/`, so repeated or concurrent evaluations never overwrite
each other. Treat publishing as best effort: write the scores to a local file first, and report a
refusal or a server error as a warning, so a publishing problem never costs a finished
evaluation.

## What gets logged

- **Metrics**: the same backend dict as WandB/TensorBoard (`train/*`, `gpu/*`,
  `smoothed/*`), plus everything passed to `tracker.log_eval`, queued to the background worker
  and sent with `mlflow.log_metrics(dict, step=step)`.
- **Params**: the full `JobConfig` flattened to dotted keys (`model.dim`, `optimizer.lr`,
  ...) once when a new run starts, via `mlflow.log_params` in batches of 100, with values cut
  at 250 characters. Cheap searchable metadata, not artifacts.
- **Tags**: `host` and `slurm_job_id`, also when a new run starts.
- **System metrics**: CPU / GPU / memory under `system/`, sampled on a
  background thread when `mlflow_log_system_metrics = true` (default). Lower
  `mlflow_system_metrics_interval` (seconds) for short test runs.

A resumed run keeps the params and tags it has. No checkpoints are uploaded as artifacts: they
stay on the filesystem (artifacts are the main MLflow storage cost);
[registering](#registering-checkpoints) records where they are.

## Bounded calls

Tracking is a side channel: a slow or unreachable server may cost points, never a stalled rank.
Every server call has a deadline.

| Call | Runs on | Bound |
|------|---------|-------|
| Run start or resume (first `log()`) | owner thread; the training thread waits | 60 s, then the backend disables itself |
| Metric, param and tag calls | background worker, in submission order; `log()` only queues | 60 s each, then abandoned |
| Checkpoint registration (opt-in) | its own thread; `save()` waits on rank 0 | 15 s |
| `close()` | drains the queue, then ends the run on the owner thread | 30 s drain, then 60 s to end |

At most 256 calls wait in the queue (a new call is dropped when it is full), and at most 4
call threads are alive at once, abandoned ones included (calls are dropped until one returns).
Warnings about failed or abandoned calls, about dropped calls, and about registration are each
rate limited to one per five minutes, with a count of what they held back. MLflow's own HTTP
layer is bounded too: requests default to a 30 s timeout with 3 retries, and Databricks SDK
requests to a 60 s retry budget (`MLFLOW_HTTP_REQUEST_TIMEOUT`,
`MLFLOW_HTTP_REQUEST_MAX_RETRIES` and `MLFLOW_DATABRICKS_ENDPOINT_HTTP_RETRY_TIMEOUT`; a value
already in the environment wins).

## Failure modes

`_ensure_init` disables the backend for the rest of the run on any failure, with one warning:

- `mlflow` is not installed: the warning says `uv sync --group mlflow`.
- A deliberate config error: `$MLFLOW_EXPERIMENT` is not absolute on Databricks, or no experiment
  path could be resolved. The warning reads `MLflow disabled: <reason>`.
- Anything else (network, auth, a start that outlasts its 60 s):
  `MLflow init failed: <error>`, followed on Databricks by the auth routes when the failure may
  be a credentials one.

Warnings that quote an error strip `user:token@` from any URI first, since MLflow and SDK
errors can quote a credentialed tracking URI.

After init, `log()` never raises and never waits: it queues the call. A call that fails or runs
past its deadline costs its points. The worker does not retry it and the backend stays enabled,
so logging recovers after a blip, and a full queue drops new calls. `close()` ends the run
whenever one was started and swallows teardown errors as warnings.

MLflow is an optional dependency: install with `uv sync --group mlflow` (pulls
`mlflow-skinny` + `databricks-sdk`, plus `psutil` and `nvidia-ml-py` for system metrics).
Without it, `enable_mlflow = true` logs a warning and no-ops.

## `close()`

Called from `tracker.close()` at the end of training, in a `finally`, so a run that raises
closes too. It stops accepting calls, gives the queued ones up to 30 s to be delivered, then
runs `mlflow.end_run()` on the owner thread within 60 s; calls still queued after the drain are
counted in a warning and dropped. No-op unless a run was started.

## See also

- [Metrics tracker](metrics-tracker.md): where backends are constructed.
- [WandB](wandb.md): the sibling backend with the same metric dict and
  run-id-resume pattern.
- [Checkpointing § Train state](../checkpointing/train-state.md): the `extra` dict that carries
  run ids across a resume.
- [Checkpointing § `metadata.json`](../checkpointing/index.md): the `tracking` entry that
  names a checkpoint's run.
- [Configuration § `[metrics]`](../configuration/config-sections.md): per-field
  reference for `MetricsConfig`.
