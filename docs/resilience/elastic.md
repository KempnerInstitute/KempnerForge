# SLURM elastic helpers

[`kempnerforge/resilience/elastic.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/resilience/elastic.py)
is a small pile of SLURM-environment helpers. Most of these are
read-only introspection — they pull variables out of the environment
and return typed data. The one with teeth is `resolve_resume_path`,
documented on its own page; this page covers the rest.

## `SLURMInfo` and `get_slurm_info`

```python
@dataclass
class SLURMInfo:
    job_id: str
    job_name: str
    node_list: str
    num_nodes: int
    ntasks_per_node: int
    restart_count: int
    partition: str
    array_task_id: str | None

    @property
    def is_requeued(self) -> bool:
        return self.restart_count > 0
```

`get_slurm_info()` reads these from env vars and returns `None` if not
running under SLURM (i.e. `SLURM_JOB_ID` is unset):

```python
# kempnerforge/resilience/elastic.py — get_slurm_info
SLURMInfo(
    job_id          = os.environ["SLURM_JOB_ID"],
    job_name        = os.environ.get("SLURM_JOB_NAME", ""),
    node_list       = os.environ.get("SLURM_JOB_NODELIST", ""),
    num_nodes       = int(os.environ.get("SLURM_NNODES", "1")),
    ntasks_per_node = int(os.environ.get("SLURM_NTASKS_PER_NODE", "1")),
    restart_count   = int(os.environ.get("SLURM_RESTART_COUNT", "0")),
    partition       = os.environ.get("SLURM_JOB_PARTITION", ""),
    array_task_id   = os.environ.get("SLURM_ARRAY_TASK_ID"),
)
```

Use it when you want to branch on SLURM-only behavior:

```python
from kempnerforge.resilience import get_slurm_info

info = get_slurm_info()
if info is not None and info.is_requeued:
    logger.info(f"Detected requeue #{info.restart_count}")
```

## `is_slurm_job` / `is_slurm_requeue`

Convenience booleans for the same data:

```python
is_slurm_job()     # "SLURM_JOB_ID" in os.environ
is_slurm_requeue() # int(os.environ.get("SLURM_RESTART_COUNT", "0")) > 0
```

Neither calls `get_slurm_info` — they just check the relevant env
var directly, so they're cheap to call frequently.

## `log_job_info`

Called once during training startup:

```python
# scripts/train.py
log_job_info()
```

Emits one info-level log line with the SLURM job details, plus a
second line if this is a requeue:

```
SLURM job: id=123456, name=kf-7b-adamw, nodes=4, tasks/node=4,
           partition=h200_preempt, restart_count=2
Job was requeued (restart #2) — will auto-resume
```

If not running under SLURM it logs `"Not running under SLURM"` and
returns.

## `resolve_resume_path`

Covered on its own page — see
[Checkpointing § Auto-resume](../checkpointing/auto-resume.md). This is
the function that decides whether the training run loads an existing
checkpoint at startup.

## Patterns these enable

### "First run vs requeue" branching

```python
info = get_slurm_info()
if info and info.is_requeued:
    # Skip LR warmup on requeue (schedule state restored from checkpoint)
    scheduler_state = "restored"
else:
    # Fresh run
    scheduler_state = "fresh"
```

In practice you don't need this — `scripts/train.py` handles warmup
restoration via the scheduler's own `state_dict`. The branch is useful
for one-off behaviors (e.g. "only print config diff on first run").

### Array job sharding

```python
info = get_slurm_info()
if info and info.array_task_id is not None:
    shard = int(info.array_task_id)
    # Use shard to pick a config variant, data shard, etc.
```

`scripts/train.py` doesn't read `array_task_id` — wire it through CLI
overrides if you want per-array-task config variants.

## See also

- [SLURM preemption](slurm-preemption.md) — where the requeue loop
  actually happens.
- [Checkpointing § Auto-resume](../checkpointing/auto-resume.md) —
  `resolve_resume_path` behavior and the `latest` symlink protocol.
- [`scripts/slurm/7b_requeue.sh`](https://github.com/KempnerInstitute/KempnerForge/blob/main/scripts/slurm/7b_requeue.sh)
  — the reference launch script that sets the SLURM env these helpers
  read.
