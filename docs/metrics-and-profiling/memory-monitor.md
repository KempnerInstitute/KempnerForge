# Memory monitoring

Two layers in
[`kempnerforge/metrics/memory.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/metrics/memory.py):

- **Functional helpers** — `get_memory_stats`, `get_memory_utilization`,
  `format_memory_stats`, `reset_peak_memory`. These are used by the
  [metrics tracker](metrics-tracker.md) every step to populate
  `StepMetrics.peak_gb` / `mem_utilization`.
- **`DeviceMemoryMonitor`** — opt-in per-interval tracker that adds
  peak-reset semantics and an optional snapshot export for OOM debugging.

## Functional helpers

```python
from kempnerforge.metrics import (
    get_memory_stats, get_memory_utilization,
    format_memory_stats, reset_peak_memory,
)

stats = get_memory_stats(device=0)
# {"allocated_gb": 12.3, "peak_gb": 14.8,
#  "reserved_gb": 16.0, "total_gb": 80.0}

utilization = get_memory_utilization(device=0)   # peak_gb / total_gb

print(format_memory_stats(device=0))
# "GPU mem: 12.3GB allocated, 14.8GB peak, 16.0GB reserved / 80.0GB total (19%)"

reset_peak_memory(device=0)   # torch.cuda.reset_peak_memory_stats
```

Four quantities, all in GB:

- `allocated_gb` — bytes allocated to live tensors right now.
- `peak_gb` — maximum `allocated_gb` seen since the last reset.
- `reserved_gb` — bytes the PyTorch caching allocator has claimed from
  CUDA; always ≥ `allocated_gb`. Large gap means fragmentation.
- `total_gb` — total VRAM on the device.

All of these are cheap — `torch.cuda.memory_allocated` is a
thread-local counter, not a driver call. Safe to read every step.

## `DeviceMemoryMonitor`

Opt-in wrapper with per-interval peak reset + snapshot support. Not used
by `scripts/train.py` directly; instantiate it yourself when you want
window-scoped peak tracking:

```python
from kempnerforge.metrics import DeviceMemoryMonitor

mon = DeviceMemoryMonitor(
    device=0,
    snapshot_step=100,         # None to disable
    snapshot_dir="memory_snapshots",
)

for step in range(max_steps):
    train_step(...)
    if step % log_interval == 0:
        mon.report(step)       # logs + resets peak + optional snapshot
```

`report(step)` does four things:

1. Reads `get_memory_stats()` and `get_memory_utilization()`.
2. Logs a one-line summary: `[step N] GPU mem: ... (X%)`.
3. If `step == snapshot_step`, calls `capture_snapshot(step)` once.
4. Calls `reset_peak_memory()` so the next interval's `peak_gb` reflects
   *that interval*, not all-time.

The interval-reset is the reason to prefer this over
`get_memory_stats`: with the bare helper, `peak_gb` grows monotonically
and you can't see that step 200 spiked higher than step 100.

## Why `MetricsTracker` doesn't reset peak

`MetricsTracker.end_step` reads `get_memory_stats` every step but
**does not reset peak memory**. The `peak_gb` field in `StepMetrics`
therefore reports the all-time peak since the process started, not the
per-step peak. This matches what most people want in a training log:
"what's my worst-case memory footprint."

If you want per-step peak instead, add a `DeviceMemoryMonitor.report()`
call alongside the tracker — the `reset_peak_memory()` it issues is
global, so subsequent tracker reads reflect only the new interval.

## Memory snapshot export

`capture_snapshot(step)` dumps a CUDA allocator snapshot to a pickle file
for offline analysis:

```python
# kempnerforge/metrics/memory.py — capture_snapshot
torch.cuda.memory._record_memory_history()
torch.cuda.synchronize(self.device)
snapshot = torch.cuda.memory._snapshot()
torch.cuda.memory._record_memory_history(enabled=None)

with open(f"memory_snapshots/snapshot_step_{step}_device_{device}.pickle", "wb") as f:
    pickle.dump(snapshot, f)
```

Load the pickle at [pytorch.org/memory_viz](https://pytorch.org/memory_viz)
for a flamegraph-style timeline showing which allocator blocks are live,
how fragmentation accumulates, and which call sites are holding memory.

**Important caveat:** `_record_memory_history` + `_snapshot` are
underscore-prefixed PyTorch APIs — they're stable enough to rely on in
the short term but have changed shape between versions. The snapshot is
best-effort: any failure (CUDA error, disk full, pickle error) is caught
and logged as a warning, training continues.

The snapshot also only covers *this rank*. For distributed memory
analysis you need one snapshot per rank, saved to per-rank filenames
(the default path includes `device_{device}`, but if you're running
multiple ranks on separate devices via CUDA_VISIBLE_DEVICES you need to
differentiate them by rank instead).

## Use cases

- **OOM diagnosis.** Set `snapshot_step` to one step before the OOM, run
  again, load the pickle and see which allocation pushed over the edge.
- **Fragmentation.** Compare `peak_gb` to `reserved_gb` over time. A
  growing gap is fragmentation; the allocator can't reuse its reserved
  pool for new allocations of different size.
- **Activation checkpointing tuning.** Set `snapshot_step` inside a
  backward pass (e.g. step 5 after warmup) to see which activations are
  consuming the most.

## Integration with memory viz

Workflow:

```bash
# 1. Run with snapshot enabled
uv run python scripts/train.py configs/train/debug.toml
# [step 100] Memory snapshot saved: memory_snapshots/snapshot_step_100_device_0.pickle

# 2. Open https://pytorch.org/memory_viz
# 3. Drag-drop the .pickle file into the page
```

The visualizer runs entirely client-side — the pickle isn't uploaded
anywhere.

## See also

- [Metrics tracker](metrics-tracker.md) — consumer of `get_memory_stats`
  for the per-step `gpu/*` metrics.
- [Profiler](profiler.md) — the complementary `torch.profiler` path; it
  records `profile_memory=True` events inside the trace for Perfetto
  inspection.
