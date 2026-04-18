# Metrics and profiling

Everything the training loop reports about itself — throughput, MFU,
loss smoothing, GPU memory, kernel traces. Two modules:

- **`kempnerforge/metrics/`** — always-on per-step metrics with WandB /
  TensorBoard dispatch, MFU computation, memory stats.
- **`kempnerforge/profiling/`** — opt-in `torch.profiler` wrapper for
  kernel-level traces and ad-hoc CUDA event timing.

## At a glance

| Component | Type | Enabled by |
|-----------|------|-----------|
| `MetricsTracker` | always-on per-step metrics | created unconditionally in `scripts/train.py` |
| `WandBBackend` | cloud logging | `metrics.enable_wandb = true` |
| `TensorBoardBackend` | local event files | `metrics.enable_tensorboard = true` |
| `compute_mfu` | per-step MFU | always-on (part of `MetricsTracker`) |
| `get_memory_stats` | per-step GPU memory | always-on |
| `DeviceMemoryMonitor` | interval-scoped peak tracking + snapshots | opt-in, instantiate manually |
| `build_profiler` | `torch.profiler` trace | `profiling.enable = true` |
| `CUDATimer` | region-scoped GPU timing | opt-in, instantiate manually |

Everything on the left of that table is exported from
`kempnerforge.metrics` and `kempnerforge.profiling` — you can import and
use any of these outside the training script.

## What gets logged by default

Every `metrics.log_interval` steps (default 10), `MetricsTracker.end_step`
emits one stdout line and one backend update containing:

```
train/loss             train/grad_norm       train/lr
train/tokens_per_sec   train/mfu             train/step_time_sec
gpu/allocated_gb       gpu/peak_gb           gpu/reserved_gb
gpu/mem_utilization
smoothed/loss          smoothed/tokens_per_sec
smoothed/mfu           smoothed/step_time
```

Plus eval metrics (via `tracker.log_eval`): eval loss, MoE aux loss and
router stats, per-dataset losses for mixtures.

## Config

```toml
[metrics]
log_interval       = 10          # log every N steps
enable_wandb       = false
enable_tensorboard = false
wandb_project      = "kempnerforge"
wandb_run_name     = ""          # auto-generated if empty
wandb_run_id       = ""          # restored from checkpoint on resume
tensorboard_dir    = "tb_logs"

[profiling]
enable     = false
start_step = 5                   # begin recording here
end_step   = 8                   # stop after this step (exclusive)
trace_dir  = "profiler_traces"
```

See [Configuration § `[metrics]` and `[profiling]`](../configuration/config-sections.md)
for field-level notes.

## Pages

```{toctree}
:maxdepth: 1

metrics-tracker
wandb
tensorboard
mfu
memory-monitor
profiler
```

## See also

- [Training § Training loop](../training/training-loop.md) — where
  `tracker.start_step` / `end_step` bookend every step.
- [Checkpointing § Train state](../checkpointing/train-state.md) —
  `wandb_run_id` is saved in `ckpt_extra` so resume reattaches to the
  same run.
- [Configuration § `[metrics]` and `[profiling]`](../configuration/config-sections.md) —
  every knob on this page.
