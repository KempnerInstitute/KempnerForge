# Memory monitoring

Two layers in
[`kempnerforge/metrics/memory.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/metrics/memory.py):

- **Functional helpers** — `get_memory_stats`, `get_memory_utilization`,
  `format_memory_stats`, `reset_peak_memory`. The
  [metrics tracker](metrics-tracker.md) calls `get_memory_stats` once on
  every logging step to fill the `StepMetrics` memory fields.
- **`DeviceMemoryMonitor`** — opt-in per-interval tracker that adds
  peak-reset semantics and an optional snapshot export for OOM debugging.

## Functional helpers

```python
from kempnerforge.metrics import (
    get_memory_stats, get_memory_utilization,
    format_memory_stats, reset_peak_memory,
)

stats = get_memory_stats()
# {"allocated_gb": 12.3, "peak_gb": 14.8,
#  "reserved_gb": 16.0, "total_gb": 80.0}

utilization = get_memory_utilization()   # peak_gb / total_gb

print(format_memory_stats())
# "GPU mem: 12.3GB allocated, 14.8GB peak, 16.0GB reserved / 80.0GB total (19%)"

reset_peak_memory()   # torch.cuda.reset_peak_memory_stats
```

Each helper reads one device. The default, `device=None`, is the current
CUDA device (`torch.cuda.current_device()`), which `init_distributed`
sets to the rank's own GPU; pass an index, such as `device=1`, to read
another.

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
    device=None,               # the current CUDA device (default)
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

`MetricsTracker.end_step` reads `get_memory_stats` once on each logging
step, from the current device, and **does not reset peak memory**.
`StepMetrics.peak_gb`, the `gpu/peak_gb` series and the `mem=` column of
the step log are therefore the high-water mark of the run: the most
memory allocated at any point since the process started, not the peak
of that step or of that logging interval. The value never goes down, so
a single spike, such as one in the first step, shows for the rest of the
run. `gpu/mem_utilization` is that peak over `total_gb`. This matches
what most people want in a training log: "what's my worst-case memory
footprint."

If you want a per-interval peak instead, call
`DeviceMemoryMonitor.report()` alongside the tracker. The
`reset_peak_memory()` it issues applies to the whole device, so from
then on the tracker's `peak_gb` covers only the time since the last
report.

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
