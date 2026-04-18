# NCCL liveness

[`check_nccl_health`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/resilience/health.py)
is a one-line all-reduce that verifies every rank is still talking to
every other rank. Cheap enough to run periodically from the training
loop.

## The check

```python
# kempnerforge/resilience/health.py — check_nccl_health
if not dist.is_initialized():
    return True

try:
    tensor = torch.ones(1, device="cuda")
    dist.all_reduce(tensor)
    torch.cuda.synchronize()
    expected = dist.get_world_size()
    return abs(tensor.item() - expected) < 1e-5
except RuntimeError as e:
    logger.error(f"NCCL health check failed: {e}")
    return False
```

A 4-byte all-reduce of ones. If every rank contributed, the result
equals `world_size`. If not — RuntimeError (mismatched group, network
drop) or a wrong sum (one rank contributed zeros).

`torch.cuda.synchronize()` forces the collective to complete before
reading `.item()`. Without it the check would short-circuit if the
collective hadn't actually started yet.

## Periodic check in the loop

```python
# scripts/train.py
if (
    tc.nccl_health_check_interval > 0
    and step % tc.nccl_health_check_interval == 0
    and not check_nccl_health()
):
    logger.error(f"NCCL health check failed at step {step} — stopping")
    break
```

`train.nccl_health_check_interval` defaults to **0** (disabled). Set it
to a positive integer to enable:

```toml
[train]
nccl_health_check_interval = 500   # every 500 steps
```

500 is a reasonable starting point: once every few minutes of
wall-clock, adds one extra all-reduce to that step. Setting it to 1
would sync every step — not useful because the gradient all-reduce in
the same step would already have failed.

## The timeout gotcha

The function signature advertises a timeout:

```python
def check_nccl_health(timeout_sec: float = 10.0) -> bool:
```

But `timeout_sec` is **not used** in the current implementation — the
function calls `dist.all_reduce` without a timeout and just catches
`RuntimeError`. If the collective hangs, the `try/except` won't trip;
the process blocks inside the C++ call.

In practice hangs are covered by two other layers:

- `NCCL_TIMEOUT` (set in `scripts/slurm/7b_requeue.sh` to 1800 = 30min)
  aborts the collective at the NCCL level and raises a `RuntimeError`,
  which `check_nccl_health` does catch.
- `ShutdownHandler`'s forced-exit timer kills the process if graceful
  shutdown stalls — see [SLURM preemption](slurm-preemption.md).

But a stuck all-reduce can still eat 30 min of wall-clock before the
NCCL-level timeout fires. If you need faster failure detection, lower
`NCCL_TIMEOUT` at the env level.

## When it fires

Most common causes of `check_nccl_health() == False`:

- **Rank crashed silently.** `world_size - 1` ranks participate, the
  all-reduce sum is off by one. Almost always the crashed rank's stdout
  has the real error — check per-rank logs.
- **Network partition.** Happens on Ethernet fabrics under heavy load;
  rare but not zero on InfiniBand. NCCL retries internally; repeated
  failures mean the fabric degraded.
- **Out-of-band process kill.** User ran `scancel` on one rank only.
  Don't do this — use `scancel --signal=USR1` instead so
  `ShutdownHandler` cleans up.

## Manual use

```python
from kempnerforge.resilience import check_nccl_health

# After a suspected partial failure
if not check_nccl_health():
    logger.error("NCCL unhealthy — aborting")
    sys.exit(1)
```

No-op if `dist.is_initialized()` is False — returns True so calling
code doesn't need to special-case single-process runs.

## See also

- [GPU health](gpu-health.md) — per-device checks; NCCL liveness tests
  the *group* instead.
- [SLURM preemption](slurm-preemption.md) — the forced-exit timer
  provides the ultimate bound on how long a hang can live.
- [Distributed § DeviceMesh](../distributed/device-mesh.md) — where
  the NCCL process groups come from.
- [Configuration § `[train]`](../configuration/config-sections.md) —
  `nccl_health_check_interval` and `shutdown_timeout_sec`.
