# Resilience

Long training runs die — preempted, NaN'd, a GPU falls off the bus.
KempnerForge's resilience module handles the recoverable failures so
the run picks up from the last checkpoint instead of burning a
re-submit.

Three failure modes the module addresses:

- **Preemption / manual stop** — SLURM sends a termination signal,
  KempnerForge catches it, writes an emergency checkpoint, and exits.
- **Numerical blow-up** — NaN / Inf loss detected, optimizer step
  skipped, gradient zeroed. If it persists, stop so a human can roll
  back.
- **Silent distributed hangs** — periodic NCCL liveness ping detects
  a dead peer before the training loop deadlocks.

## At a glance

| Component | Source | Wired into `train.py`? |
|-----------|--------|-------------------------|
| `ShutdownHandler` | `resilience/signal_handler.py` | always-on |
| `NaNDetector` | `resilience/health.py` | hardcoded `action="warn"` |
| `check_nccl_health` | `resilience/health.py` | opt-in via `train.nccl_health_check_interval` |
| `check_gpu_health` | `resilience/health.py` | no — manual utility |
| `SLURMInfo` / `get_slurm_info` / `log_job_info` | `resilience/elastic.py` | `log_job_info()` at startup |
| `resolve_resume_path` | `resilience/elastic.py` | at checkpoint load time |

Everything in the first column is importable from
`kempnerforge.resilience`.

## Config

```toml
[train]
shutdown_timeout_sec       = 600.0   # ShutdownHandler hard deadline (0 = disabled)
nccl_health_check_interval = 0       # NCCL ping every N steps (0 = disabled)
```

Two knobs. There is deliberately no `nan_detection` section — the
action and max-consecutive count are hardcoded in `scripts/train.py`
(`action="warn"`, `max_consecutive=10`). Edit the script if you need
different behavior; see [NaN detection](nan-detection.md).

## SLURM launch

The reference preemption-resilient launch script is
[`scripts/slurm/7b_requeue.sh`](https://github.com/KempnerInstitute/KempnerForge/blob/main/scripts/slurm/7b_requeue.sh):

```bash
#SBATCH --signal=B:SIGTERM@120   # SIGTERM 120s before hard kill
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
  the two resilience-related config fields.
