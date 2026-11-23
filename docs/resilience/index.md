# Resilience

Long training runs die — preempted, NaN'd, a GPU falls off the bus.
KempnerForge's resilience module handles the recoverable failures so
the run picks up from the last checkpoint instead of burning a
re-submit.

Three failure modes the module addresses:

- **Preemption / manual stop** — SLURM sends a termination signal,
  KempnerForge catches it, writes an emergency checkpoint, and exits.
- **Numerical blow-up** — NaN / Inf loss or gradient norm detected;
  with the default `nan_action`, optimizer step skipped and gradients
  zeroed. If it persists, stop so a human can roll back.
- **Silent distributed hangs** — periodic NCCL liveness ping detects
  a dead peer before the training loop deadlocks.

## At a glance

| Component | Source | Wired into `train.py`? |
|-----------|--------|-------------------------|
| `ShutdownHandler` | `resilience/signal_handler.py` | always-on |
| `NaNDetector` | `resilience/health.py` | always-on; `train.nan_action`, `train.nan_max_consecutive` |
| `check_nccl_health` | `resilience/health.py` | opt-in via `train.nccl_health_check_interval` |
| `check_gpu_health` | `resilience/health.py` | no — manual utility |
| `SLURMInfo` / `get_slurm_info` / `log_job_info` | `resilience/elastic.py` | `log_job_info()` at startup |
| `resolve_resume_path` | `resilience/elastic.py` | at checkpoint load time |

Everything in the first column is importable from
`kempnerforge.resilience`.

## Config

```toml
[train]
shutdown_timeout_sec       = 600.0   # forced exit this long after a rank starts shutting down (0 = disabled);
                                     # fires before SLURM's SIGKILL only if below the signal lead
nccl_health_check_interval = 0       # NCCL ping every N steps (0 = disabled)
nan_action                 = "warn"  # on a NaN/Inf step: "warn" or "skip" (the loop skips it either way),
                                     # or "raise" (every rank raises RuntimeError)
nan_max_consecutive        = 10      # NaN/Inf steps in a row that stop the run for a manual rollback
```

Four knobs, all in `[train]`. The last two set the `NaNDetector` that
the training loop checks every step; see [NaN detection](nan-detection.md).

## SLURM launch

The reference preemption-resilient launch script is
[`scripts/slurm/7b_requeue.sh`](https://github.com/KempnerInstitute/KempnerForge/blob/main/scripts/slurm/7b_requeue.sh):

```bash
#SBATCH --signal=SIGTERM@120     # SIGTERM to the ranks before the time limit
#SBATCH --requeue                # auto-resubmit on preempt

srun --kill-on-bad-exit=1 uv run python scripts/train.py "${CONFIG}"
```

Pair with a checkpoint interval of a few hundred steps (~1.5 hours for
a 7B run on 16 H100s), so the emergency checkpoint never loses more
than that.

## Pages

```{toctree}
:maxdepth: 1

slurm-preemption
nan-detection
gpu-health
nccl-liveness
elastic
```

## See also

- [Checkpointing § Auto-resume](../checkpointing/auto-resume.md) —
  what a requeued job does on startup.
- [Training § Training loop](../training/training-loop.md) — where
  `shutdown_handler.should_shutdown()` and `nan_detector.check_loss()`
  are polled each step.
- [Configuration § `[train]`](../configuration/config-sections.md) —
  the resilience-related config fields.
