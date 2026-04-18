# Profiler

[`kempnerforge/profiling/profiler.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/profiling/profiler.py)
wraps `torch.profiler` with a schedule, a trace handler, and a
post-training analysis that classifies kernels into matmul / comm /
memory / other and writes a `summary.md` alongside the trace.

Opt-in via config:

```toml
[profiling]
enable     = false      # must flip to true
start_step = 5          # profile from step 5 inclusive
end_step   = 8          # through step 8 exclusive (3 active steps)
trace_dir  = "profiler_traces"
```

`ProfilingConfig.__post_init__` enforces `end_step > start_step`.

## Schedule

```python
# kempnerforge/profiling/profiler.py — build_profiler
wait_steps   = max(0, config.start_step - 1)
active_steps = config.end_step - config.start_step
schedule(
    wait=wait_steps,        # skip N steps
    warmup=1,               # one warmup step (stabilizes CUDA caches)
    active=active_steps,    # record this many
    repeat=1,               # one cycle, then idle
)
```

With the defaults (`start_step=5`, `end_step=8`):

- Steps 1–4: `wait` — profiler idle.
- Step 5: `warmup` — profiler running but trace discarded.
- Steps 6–8: `active` — trace recorded.
- Step 9+: done.

Why one warmup step: CUDA kernel launches, cudnn picks, and memory-pool
growth all stabilize in the first few steps. Without a warmup the trace
includes one-time costs that make it look like everything is slow.

Why not `repeat=N`: KempnerForge profiles once per run. If you need
periodic snapshots, either run multiple configs or instantiate
`torch.profiler.profile` directly.

## Activities

```python
profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=...,
    on_trace_ready=tensorboard_trace_handler(str(trace_dir)),
    record_shapes=True,
    profile_memory=True,
    with_stack=False,
    with_flops=True,
)
```

- **CPU + CUDA** — captures both sides. CPU traces catch Python overhead
  and CPU-bound preprocessing; CUDA traces catch kernel time.
- **`record_shapes=True`** — per-op input shapes in the trace. Required
  for the `with_flops=True` counter to work.
- **`profile_memory=True`** — allocations / frees appear in Perfetto as
  memory timeline events. Useful for spotting activation spikes.
- **`with_stack=False`** — skip Python stack capture. `True` adds
  significant overhead and makes the trace much larger; flip only if
  you're specifically hunting a Python-side hotspot.
- **`with_flops=True`** — populates `evt.flops` for matmul ops, which
  `_analyze_profiler` uses to compute achieved TFLOPS.

## Loop integration

```python
# scripts/train.py
prof = build_profiler(config.profiling, rank=rank)
...
if prof is not None:
    prof.start()

while step < tc.max_steps:
    ...
    if prof is not None:
        prof.step()        # advance schedule after each step

if prof is not None:
    prof.stop()
    if rank == 0:
        print_profiler_summary(prof, trace_dir=config.profiling.trace_dir)
```

Three conditionals because profiling is entirely optional — `build_profiler`
returns `None` when `enable=False`, and the rest of the loop no-ops.

**All ranks profile**, but only rank 0 prints the summary. Each rank
writes its own trace file to `trace_dir`, which is fine — trace
filenames include host + PID so they don't collide.

## Trace output

`tensorboard_trace_handler` writes one `.pt.trace.json` per rank per
scheduled trace to `trace_dir/`:

```
profiler_traces/
├── node01_12345.1714838500.pt.trace.json
├── node01_12346.1714838501.pt.trace.json
└── ...
```

View with:

```bash
# Perfetto UI (preferred for detailed exploration)
#   → https://ui.perfetto.dev/ → "Open trace file" → pick the .json

# TensorBoard (aggregated, per-op tables)
uv run tensorboard --logdir profiler_traces
```

Perfetto is better for zoom / search / timeline scrubbing. TensorBoard's
profiler plugin is better for aggregate tables and kernel-level stats
(when the plugin is installed, via `pip install torch_tb_profiler` — not
a hard dep of KempnerForge).

## `print_profiler_summary`

After `prof.stop()`, rank 0 runs `print_profiler_summary(prof, trace_dir)`
which prints three sections and writes a markdown report.

### Console output

- **Top CUDA kernels by total time** — `prof.key_averages().table(sort_by="cuda_time_total", row_limit=30)`.
- **Top CUDA kernels by FLOPS** — same but `sort_by="flops", row_limit=20`.
- **Aggregate GPU time breakdown** — totals for matmul / comm / memory /
  other, plus achieved vs peak TFLOPS.

### `summary.md`

Written to `trace_dir/summary.md`. Contains three tables:

- **GPU Time Breakdown** — matmul, communication, memory, other (%).
- **Efficiency** — total FLOPS, achieved TFLOPS, peak, kernel efficiency %.
- **Top CUDA Kernels** — top 20 by CUDA time with call count and GFLOPS.

Plus a note on how to view the traces. Designed to commit into an
experiment log.

## Kernel classification

`_analyze_profiler` classifies each kernel by substring matching on the
event key (lowercased):

| Category | Substrings matched |
|----------|--------------------|
| `matmul` | `gemm`, `mm`, `matmul`, `dot`, `bmm`, `cublas`, `nvjet`, `cutlass` |
| `comm`   | `nccl`, `allreduce`, `allgather`, `reduce_scatter` |
| `memory` | `memcpy`, `memset` |
| `other`  | everything else |

These are conservative — anything outside the lists (flash attention
kernels, softmax, element-wise ops) falls into `other`. That's intentional;
misclassifying attention as "matmul" would inflate the apparent matmul %.

The breakdown is a *rough* indicator:

- **High `matmul %` (>50%)** — compute-bound, MFU should be good.
- **High `comm %` (>20%)** — communication overhead; check
  [FSDP2](../distributed/fsdp2.md) overlap settings, tensor parallelism
  placement, or batch size.
- **High `other %` (>40%)** — element-wise or memory-bound kernels
  dominate; check for small tensor shapes, un-fused ops, or Python-side
  stalls (run with `with_stack=True` to pinpoint).

## `CUDATimer` — manual region timing

[`kempnerforge/profiling/cuda_timer.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/profiling/cuda_timer.py)
provides CUDA-event-based timers for region-scoped measurement without
the overhead of a full `torch.profiler` trace. Not used by `scripts/train.py`;
instantiate yourself for ad-hoc timing:

```python
from kempnerforge.profiling import CUDATimer, CUDATimerCollection

# Single region
t = CUDATimer()
t.start()
# ... GPU work ...
t.stop()
print(f"took {t.elapsed_ms():.2f} ms")

# Multiple regions
timers = CUDATimerCollection(regions=["forward", "backward", "comm"])
timers.start("forward"); loss = model(x, y); timers.stop("forward")
timers.start("backward"); loss.backward(); timers.stop("backward")
print(timers.elapsed_all())   # {"forward": 12.3, "backward": 8.1, "comm": 0.0}
```

Uses `torch.cuda.Event(enable_timing=True)` under the hood, so timing
is GPU-accurate — unlike `time.perf_counter()`, which only sees the CPU
side. `elapsed_ms()` synchronizes the stream once at read time, not per
start/stop call — ~nanosecond overhead in the hot path.

`CUDATimerCollection(..., enabled=False)` makes every method a no-op —
useful for dropping timers in permanently without a per-step `if`.

## See also

- [Metrics tracker](metrics-tracker.md) — per-step metrics are always on;
  the profiler is a deeper, trace-based alternative for a few steps.
- [MFU](mfu.md) — per-step estimate; the profiler computes `achieved_tflops`
  directly from kernel `evt.flops` counts, which is a useful cross-check.
- [Memory monitor](memory-monitor.md) — snapshot-based memory debugging
  is a complement to the profiler's inline `profile_memory=True` events.
- [Configuration § `[profiling]`](../configuration/config-sections.md) —
  `enable`, `start_step`, `end_step`, `trace_dir`.
