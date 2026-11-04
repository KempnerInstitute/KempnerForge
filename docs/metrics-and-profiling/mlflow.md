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
| Profile | a `[<profile>]` section with `host` and `token` in `~/.databrickscfg` | the tracking URI is `databricks://<profile>`, or `$DATABRICKS_CONFIG_PROFILE` names it; otherwise, when no `DATABRICKS_HOST`, `DATABRICKS_TOKEN` or other credential variable is set, the SDK's default profile (the `default_profile` of a `[__settings__]` section when the file has one, else `[DEFAULT]`) |
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

**Pin token auth on compute nodes.** With the environment or profile route, add
`export DATABRICKS_AUTH_TYPE=pat` to the job script. It makes the SDK use token auth only and
skip every other method. On a cluster whose compute nodes drop link-local metadata traffic, a
fresh client was measured at about a minute without the pin, long enough for the bounded init
to give up with `no answer from the tracking server within 60s`, and a few seconds with it. The
cause was not traced to a specific SDK step, and the pin costs nothing where there is no stall.
Leave it unset for the OAuth route, or pin that route's own type (`oauth-m2m` for a service
principal, `databricks-cli` for a CLI login).

When a profile is selected, by a `databricks://<profile>` URI or `$DATABRICKS_CONFIG_PROFILE`, the
SDK reads environment variables first and uses the profile only to fill in what they leave unset,
so an exported `DATABRICKS_TOKEN` overrides the profile's token. With none selected (a plain
`databricks` URI and no `$DATABRICKS_CONFIG_PROFILE`), an exported `DATABRICKS_HOST`,
`DATABRICKS_TOKEN` or other credential variable makes the SDK ignore `~/.databrickscfg`. To
combine a profile with exported variables, select it. For token auth with none selected, export
both `DATABRICKS_HOST` and a token, or neither. `databricks://<profile>` also exports the profile
as `DATABRICKS_CONFIG_PROFILE` for the SDK and artifact calls, unless that variable is already
set; if you set it yourself, keep it equal to the URI's profile.

```{warning}
`DATABRICKS_API_TOKEN` is not a name the SDK reads. The training backend and the
[publisher](#publishing-eval-results) mirror it into `DATABRICKS_TOKEN` when that is unset or
empty and the tracking URI is Databricks, and from then on it is the environment route: a stale
export left in `~/.bashrc` overrides a selected profile's token, and with none selected it makes
the SDK ignore `~/.databrickscfg`, even though the profile itself is fine. Unset stale exports
(`unset DATABRICKS_API_TOKEN DATABRICKS_TOKEN DATABRICKS_HOST`) when you authenticate with a
profile. Export `DATABRICKS_TOKEN` rather than the alias for any other tool that talks to the
workspace.
```

`uv run python scripts/check_env.py --requires mlflow` reports which route a shell has. It
passes on the environment pair (`DATABRICKS_HOST` and a token under either name), or on a
profile section found in `~/.databrickscfg` (the profile named by a `databricks://<profile>`
`$MLFLOW_TRACKING_URI`, else `$DATABRICKS_CONFIG_PROFILE`, else `DEFAULT`). With neither it
warns without failing, naming the three routes, because it cannot verify OAuth offline. It
reads only section names from the profile file, never the tokens, so it cannot see a
`[__settings__]` default: name the profile if you rely on one. It looks at the environment,
not at `mlflow_tracking_uri` in a config. Add `--check-credentials` to also call the workspace
through the SDK. A failure is reported by exception class only, because the text can quote a
token (a malformed profile line, a token with a stray newline); the connectivity check below
prints the full error.

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
any URI. Run it on the partition the training job will use, with the exports the job script
has (the auth pin above, and `$MLFLOW_TRACKING_URI` if your config sets `mlflow_tracking_uri`),
so it tests what the job will run:

```bash
export DATABRICKS_AUTH_TYPE=pat    # as in the job script; leave out for OAuth
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
the run in `close()`: mlflow keeps the active run per thread, and `end_run()` only ends, and
stops the system-metrics monitor of, the run active on the calling thread.

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
experiment, run_id, run_name, resumed, last_step = self._owner.start(_INIT_DEADLINE_S)
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
re-logging the steps the run already holds: every metric at or below the highest `train/loss`
step in the run's history is dropped, so re-executing the steps between the last checkpoint and
the crash does not duplicate points. If that history cannot be read, nothing is skipped. The
params, tags and artifacts are not written again.

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
committed checkpoint an MLflow external logged model, so the run lists its checkpoints:

- name `<run name>-step_<N>` (characters MLflow rejects in a model name become `_`, and the run
  id stands in for an empty run name), `model_type = "checkpoint"`, linked to the run;
- params `step` and `checkpoint_dir`, the absolute path of the step directory (publishing
  through the `latest` symlink records the directory it points at).

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
effort: a failure, such as a server error or a full queue, leaves the entry without `model_id`
and logs a rate-limited warning, and the next commit of the same step tries again. A step saved
twice (a shutdown save right after an interval save, or a resumed run re-running it) keeps one
model, and the last `metadata.json` of that step still gets its `model_id`; with an async
checkpoint mode the step is committed and registered once the second save's shards are
durable. A resume registers
nothing by itself, and a checkpoint deleted before its turn comes is skipped. At shutdown,
`close()` delivers a registration still queued within its 30 s drain budget, so the final
checkpoint is registered too. A save before the run's first log has no `mlflow` entry, so its
`metadata.json` gets neither field. MLflow itself logs two harmless "Failed to infer schema"
warnings for each external model it creates.

**Models-tab leaderboard.** The model is also what lets eval scores be compared per
checkpoint. The [publisher](#publishing-eval-results) links every metric it appends to the
entry's `model_id`. When the entry has only `model_name` (the registration failed or never
finished), it finds the run's model of that name, or creates it, and links to that; it never
writes `metadata.json`. The scores attach to per-checkpoint models: the experiment's Models tab
can list the checkpoints with their scores side by side and rank them, and
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
the target, with the `model_id` the metrics were linked to (`None` when unlinked);
`read_mlflow_target(checkpoint_dir)` returns it without publishing: the run id, experiment,
tracking URI, step, model id and model name. Both also prepare Databricks auth in the process
environment, as described under credentials below.

- **What it reads.** The single `mlflow` entry of the `tracking` list, whose `run_id`,
  `experiment` and `tracking_uri` must be non-empty strings, and `model_id` and `model_name`
  when present; plus a non-negative integer `step`. Nothing is inferred, and a key repeated
  anywhere in the file is refused.
- **What it sends.** One `log_batch` request through a client bound to the recorded tracking URI,
  after every metric has been checked: a non-empty name, a finite number, at most 1000 in all.
  The metrics are linked to the checkpoint's model: the entry's `model_id`, else the run's model
  named by `model_name`, found or created as [above](#registering-checkpoints). An entry with
  neither, or an MLflow too old to link metrics to a model, publishes them unlinked. It never starts, ends or edits a run and keeps no state, so
  several publishers can write to one run at once. The call blocks on the server under MLflow's
  own HTTP timeouts, so run it as a separate process, never from the training step. `log_batch`
  is not atomic: a request that fails server side may still have stored some points.
- **When it refuses.** The directory has no `metadata.json`; there is no `tracking` list or no
  `mlflow` entry (a checkpoint saved before the run's first log, an older one, or training
  without MLflow); more than one `mlflow` entry; a field is missing or empty; the metrics are
  malformed; `mlflow` is not installed; or the metrics have no model to link to while the
  process has an active MLflow model (`MLFLOW_ACTIVE_MODEL_ID` or
  `mlflow.set_active_model`), which `log_batch` would silently link the metrics to.
- **Credentials.** It prepares auth from the [routes above](#authentication) the way the
  training backend does: it uses the recorded `tracking_uri` as written, a
  `databricks://<profile>` entry selects that profile (exported as `DATABRICKS_CONFIG_PROFILE`
  unless already set, so artifact uploads a harness makes later in the same process
  authenticate too), and `DATABRICKS_API_TOKEN` stands in for an unset or empty
  `DATABRICKS_TOKEN`. `user:token@` credentials in a tracking URI are stripped before the
  sidecar is written, so that route does not carry over. Give the machine that publishes a profile of the
  same name.

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
  at 250 characters. Cheap searchable metadata, not artifacts. `metrics.mlflow_description`
  and `metrics.mlflow_tags` are left out, since they become the description and tags below.
- **Tags**, also when a new run starts: `host` and `slurm_job_id`; the git commit of the working
  directory as `mlflow.source.git.commit`, with `mlflow.source.git.dirty` (`true` when tracked
  files differ from it), skipped quietly when git or a repository is missing;
  `metrics.mlflow_description` as the run description (`mlflow.note.content`); and every entry
  of `metrics.mlflow_tags`. Configured tags are applied last, so one replaces an automatic tag
  of the same key.
- **Artifacts**, also when a new run starts: `config.json`, the resolved `JobConfig` (TOML, CLI
  overrides and defaults applied), and the launch TOML under its own file name when the first
  command-line argument names a `.toml` file (how `scripts/train.py` takes its config); without
  one that upload is skipped.
- **System metrics**: CPU / GPU / memory under `system/`, sampled on a
  background thread when `mlflow_log_system_metrics = true` (default). Lower
  `mlflow_system_metrics_interval` (seconds) for short test runs.

A resumed run keeps the params, tags and artifacts it has. No checkpoints are uploaded as
artifacts: they stay on the filesystem (artifacts are the main MLflow storage cost);
[registering](#registering-checkpoints) records where they are. The description and tags are
set in `[metrics]`:

```toml
[metrics]
mlflow_description = "stage-2 video run"            # Markdown; the run description in the UI
mlflow_tags        = { team = "vlm", stage = "2" }  # non-empty strings, name to value
```

## Bounded calls

Tracking is a side channel: a slow or unreachable server may cost points, never a stalled rank.
Every server call has a deadline.

| Call | Runs on | Bound |
|------|---------|-------|
| Run start or resume (first `log()`) | owner thread; the training thread waits | 60 s, then the backend disables itself |
| Metric, param and tag calls | background worker, in submission order; `log()` only queues | 60 s each, then abandoned |
| Artifact uploads (a new run's `config.json` and launch TOML) | background worker | 180 s each, then abandoned |
| Checkpoint registration (opt-in) | background worker, queued once the checkpoint is committed; `save()` never waits | 180 s, then abandoned |
| `close()` | drains the queue, then ends the run on the owner thread | 30 s drain, then 60 s to end |

At most 256 calls wait in the queue (a new call is dropped when it is full), and at most 4
call threads are alive at once, abandoned ones included (calls are dropped until one returns).
Warnings about failed or abandoned calls, about dropped calls, and about registration are each
rate limited to one per five minutes, with a count of what they held back.

MLflow's own HTTP layer is bounded too, by defaults the training backend sets when it imports
`mlflow` (a value already in the environment wins): requests time out after 30 s and retry 3
times (`MLFLOW_HTTP_REQUEST_TIMEOUT`, `MLFLOW_HTTP_REQUEST_MAX_RETRIES`), and MLflow's
Databricks SDK transport retries for at most 60 s
(`MLFLOW_DATABRICKS_ENDPOINT_HTTP_RETRY_TIMEOUT`). A `WorkspaceClient` the backend builds
itself, to resolve an automatic experiment path, keeps the SDK's own 300 s retry default, so
only the 60 s init deadline bounds it. Other MLflow clients, the publisher and the connectivity
check included, run with MLflow's own defaults.

## Failure modes

`_ensure_init` disables the backend for the rest of the run on any failure, with a warning:

- `mlflow` is not installed: the warning says `uv sync --group mlflow`.
- A deliberate config error: `$MLFLOW_EXPERIMENT` is not absolute on Databricks, or no experiment
  path could be resolved. The warning reads `MLflow disabled: <reason>`. For an unresolved path,
  an earlier warning gives the cause: `Could not auto-resolve Databricks experiment path: <error>`,
  followed by the auth routes when the error may be a credentials one.
- Anything else (network, auth, a start that outlasts its 60 s, which on compute nodes
  [pinning the auth type](#authentication) can avoid): `MLflow init failed: <error>`, followed
  on Databricks by the auth routes when the failure may be a credentials one.

Warnings that quote an error strip `user:token@` from any URI first, since MLflow and SDK
errors can quote a credentialed tracking URI. The checkpoint manager's warnings about tracking
entries and registration do the same.

After init, `log()` never raises and never waits: it queues the call. A call that fails or runs
past its deadline costs only what it carried: its points, a batch of params, a tag or an
upload. The worker does not retry it and the backend stays enabled, so logging recovers after a
blip, and a full queue drops new calls. `close()` ends the run whenever one was started and
swallows teardown errors as warnings.

MLflow is an optional dependency: install with `uv sync --group mlflow` (pulls
`mlflow-skinny` + `databricks-sdk`, plus `psutil` and `nvidia-ml-py` for system metrics).
Without it, `enable_mlflow = true` logs a warning and no-ops.

## `close()`

Called from `tracker.close()` at the end of training, in a `finally`, so a run that raises
closes too. It stops accepting calls, gives the queued ones (the final checkpoint's
registration included) up to 30 s to be delivered, then runs `mlflow.end_run()` on the owner
thread within 60 s; calls still queued after the drain are counted in a warning and dropped.
No-op unless a run was started.

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
