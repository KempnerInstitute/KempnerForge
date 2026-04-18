# GPU health

[`check_gpu_health`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/resilience/health.py)
runs a short smoke test against a CUDA device. The training loop
**does not** call it automatically — it's an opt-in diagnostic, most
useful at job startup or after a suspected hardware fault.

## What it checks

```python
# kempnerforge/resilience/health.py — check_gpu_health
result = {
    "cuda_available": torch.cuda.is_available(),
    "device_accessible": False,
    "compute_ok": False,
    "memory_ok": False,
    "error": "",
}
```

Four booleans + an error string. Each test must pass before the next
runs:

1. **`cuda_available`** — `torch.cuda.is_available()`. If this is
   False the rest short-circuits.
2. **`device_accessible`** — `torch.cuda.set_device(device)`. Catches
   stale CUDA contexts or permission errors that let `cuda_available`
   pass but block actual device use.
3. **`compute_ok`** — `x = torch.ones(16); y = x + x; assert y.sum() == 32`.
   A tiny elementwise + reduction. Catches fused-kernel or launcher
   failures that look fine to `set_device` but crash on first op.
4. **`memory_ok`** — allocate a 1 MB buffer
   (`torch.empty(256*1024, dtype=float32)`) and free it. Catches the
   case where the GPU is reachable and can launch kernels but OOMs on
   any new allocation (usually a stale allocator state).

## Usage

```python
from kempnerforge.resilience import check_gpu_health

health = check_gpu_health(device=0)
if not (health["cuda_available"] and health["compute_ok"] and health["memory_ok"]):
    raise RuntimeError(f"GPU unhealthy: {health['error']}")
```

Or as a pre-flight check before long runs:

```python
# At job start, before init_distributed
for device in range(torch.cuda.device_count()):
    h = check_gpu_health(device)
    if h["error"]:
        logger.error(f"Device {device}: {h['error']}")
```

## When to use it

- **After a hardware-suspected crash.** If a run died with a CUDA
  error, re-check the device before resuming.
- **On cluster nodes you don't own.** Mixed-tenant clusters sometimes
  leave GPUs in a partially-wedged state; this surfaces that before
  your training burns a training-step of compute.
- **Before expensive data loading.** Cheaper to fail at step 0 than
  30 minutes into HF dataset streaming.

## When it won't help

- **Transient NCCL failures.** `check_gpu_health` runs local ops only
  — no collective. For distributed liveness see
  [NCCL liveness](nccl-liveness.md).
- **Slow / degraded GPUs.** The test passes if the GPU *works*, not if
  it's running at spec. For throughput regressions use the
  [profiler](../metrics-and-profiling/profiler.md) or watch
  `step_time_sec` in the metrics.
- **Memory fragmentation under real load.** A 1 MB allocation doesn't
  trigger fragmentation; full training allocations might. Use
  [memory snapshots](../metrics-and-profiling/memory-monitor.md#memory-snapshot-export)
  for that.

## Return shape

```python
{
  "cuda_available": True,
  "device_accessible": True,
  "compute_ok": True,
  "memory_ok": True,
  "error": "",
}
```

`error` is the string of the last-raised `RuntimeError` / `AssertionError`
if any test failed. `cuda_available` gets a special case — if False,
`error` is set to `"CUDA not available"` before returning. Otherwise
errors only appear if a test actually raised.

## See also

- [NCCL liveness](nccl-liveness.md) — the distributed-side counterpart;
  covers "GPUs are alive but not talking to each other".
- [NaN detection](nan-detection.md) — model-level failures rather than
  device-level.
- [Memory monitor](../metrics-and-profiling/memory-monitor.md) — runtime
  memory tracking; complementary to the 1 MB allocation test here.
