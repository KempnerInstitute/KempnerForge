# WandB backend

[`WandBBackend`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/metrics/tracker.py)
is opt-in. Enable it by setting two fields:

```toml
[metrics]
enable_wandb   = true
wandb_project  = "my-project"     # default "kempnerforge"
wandb_run_name = "7b-debug"       # optional — wandb auto-generates one
```

The backend is constructed by `MetricsTracker._init_backends` on rank 0
only; other ranks never touch wandb.

## Init is lazy

`__init__` just stashes the config — no network I/O. The first call to
`log()` runs `_ensure_init`, which calls `wandb.init(...)` with these
kwargs:

```python
# kempnerforge/metrics/tracker.py
init_kwargs = {
    "project": self._config.wandb_project,
    "name":    self._config.wandb_run_name,
    "resume":  "allow",
}
if self._config.wandb_run_id:
    init_kwargs["id"] = self._config.wandb_run_id
self._run = wandb.init(**init_kwargs)
self._config.wandb_run_id = self._run.id      # write back for checkpoint
```

Two reasons for lazy init:

- **Distributed setup finishes first.** By the time the first step logs,
  process groups are built, checkpoint resume has happened, and
  `wandb_run_id` may have been restored from the checkpoint `extra` dict.
- **No wandb call in CI / tests.** Constructing a tracker never touches
  the network unless a step actually logs.

## Run ID and resume

This is the key integration point with [Checkpointing](../checkpointing/index.md).
On a fresh run:

1. `wandb.init(...)` with no `id` → wandb mints a new run ID.
2. `self._config.wandb_run_id = self._run.id` stores it on the live
   `MetricsConfig`.
3. The training loop writes it into `ckpt_extra`:

   ```python
   # scripts/train.py
   if config.metrics.wandb_run_id:
       ckpt_extra["wandb_run_id"] = config.metrics.wandb_run_id
   ```

   This ends up in the `train_state.extra` dict inside the checkpoint.

On resume:

```python
# scripts/train.py — right after ckpt_mgr.load(...)
if ckpt_extra_loaded.get("wandb_run_id"):
    config.metrics.wandb_run_id = ckpt_extra_loaded["wandb_run_id"]
```

Then when tracker backends initialize lazily, `init_kwargs["id"]` carries
the restored ID and `resume="allow"` tells wandb to reattach to that run.
Metrics continue on the same chart rather than splitting across runs.

`resume="allow"` (not `"must"`) is intentional: if the old run was
deleted wandb-side, the flag falls back to a fresh run rather than
crashing the job.

## What gets logged

Exactly the backend dict from `MetricsTracker._log_step`:

```
train/loss, train/grad_norm, train/lr,
train/tokens_per_sec, train/mfu, train/step_time_sec,
gpu/allocated_gb, gpu/peak_gb, gpu/reserved_gb, gpu/mem_utilization,
smoothed/loss, smoothed/tokens_per_sec, smoothed/mfu, smoothed/step_time
```

Plus whatever dict is passed into `tracker.log_eval(...)` — eval loss,
MoE aux loss and router utilization, per-source losses for mixtures.

All scalars. No histograms, no images, no tables.

## Failure modes

`_ensure_init` wraps the `wandb.init(...)` call in a try/except and sets
`self._run = False` as a sentinel on any failure:

```python
except ImportError:
    logger.warning("wandb not installed — disabling WandB backend")
    self._run = False
except Exception as e:     # network, auth, quota, etc.
    logger.warning(f"WandB init failed: {e}")
    self._run = False
```

Subsequent `log()` calls check `if self._run is False: return` and silent
no-op. This is the intended behavior — a flaky network or expired login
should never crash training, just lose the logs.

If you see logs going to stdout but not wandb, grep the stderr for
`WandB init failed:` or `wandb not installed`.

## `close()`

Called from `tracker.close()` at the very end of training. Runs
`wandb.finish()` which flushes pending metrics, uploads any
`wandb.save`-queued files, and closes the run. No-op if `_run is False`.

## See also

- [Metrics tracker](metrics-tracker.md) — where `WandBBackend` is
  constructed and called.
- [TensorBoard](tensorboard.md) — the alternative backend with the same
  metric dict.
- [Checkpointing § Train state](../checkpointing/train-state.md) — how
  `wandb_run_id` travels with the checkpoint.
- [Configuration § `[metrics]`](../configuration/config-sections.md) —
  per-field reference for `MetricsConfig`.
