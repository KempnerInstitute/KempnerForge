# Debug training regressions

Training failures cluster into five shapes: loss goes NaN, memory
creeps up to OOM, a rank hangs without error, throughput silently
halves, and the loss curve looks plausible but downstream eval is
worse. This page chains the tools the codebase already ships to
diagnose each.

## The diagnostic toolbox

| Tool | What it catches | Config | Enabled by default? |
|------|------------------|:------:|:-------------------:|
| [`NaNDetector`](../resilience/nan-detection.md) | NaN/Inf in loss | hardcoded in `scripts/train.py` | yes (warn mode) |
| [`MetricsTracker`](../metrics-and-profiling/metrics-tracker.md) | loss, grad-norm, tok/s, MFU, memory | `[metrics]` | yes |
| [`DeviceMemoryMonitor`](../metrics-and-profiling/memory-monitor.md) | per-interval peak memory, snapshot export | manual instantiation | no |
| [`torch.profiler` wrapper](../metrics-and-profiling/profiler.md) | kernel traces, efficiency breakdown | `[profiling]` | no |
| [`check_nccl_health`](../resilience/nccl-liveness.md) | dead/deadlocked rank | `train.nccl_health_check_interval` | no |
| [`check_gpu_health`](../resilience/gpu-health.md) | dead GPU, alloc fail | on-demand | no |

Default-on tools give you signal for free. The rest you reach for
when default-on signals point at a specific failure mode.

## Shape 1: loss goes NaN

**Symptom:** one step shows `loss NaN`, the next might too, training
keeps running.

**What the codebase does:** `scripts/train.py` instantiates
`NaNDetector(action="warn", max_consecutive=10)` at startup. Each
step's `avg_loss` flows through
[`check_loss()`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/resilience/health.py),
which all-reduces a NaN flag across ranks and returns `False` on NaN.
The training loop reacts:

- Logs a warning on every NaN/Inf (from `check_loss`)
- Zeros gradients and skips the optimizer step (in the loop, right
  after `check_loss` returns False — preserves last-good weights)
- When `nan_detector.should_rollback` becomes true (consecutive count
  ≥ `max_consecutive`), rank 0 logs `"Too many consecutive NaNs — stopping"`
  and breaks the loop

You still need to act — the detector doesn't roll back, it just keeps
you from optimizing into garbage. The usual flow:

1. Grep the log for the first NaN step → get `N`.
2. Identify the last clean checkpoint at `step_M` where `M < N`.
3. Restart with `--checkpoint.load_path=checkpoints/<run>/step_M` and
   a lower LR or a reduced spike-prone feature (e.g., turn off FP8 if
   you were testing it).

### Configuration caveat

`action` and `max_consecutive` are **hardcoded** in
`scripts/train.py:85` — there's no TOML knob. If you need
`action="raise"` (fail fast) or `action="skip"` without the warning,
edit the source. See
[Resilience § NaN detection § Limitations](../resilience/nan-detection.md#limitations).

### Related signals

Check `train/grad_norm` in your metrics dashboard before the NaN — a
visible spike (orders of magnitude above baseline) usually precedes it.
If you see one, that's the real signal; the NaN is a consequence.

## Shape 2: memory creeps toward OOM

**Symptom:** step 1 uses `X` GB, step 1000 uses `X + Y` GB, step 3000
OOMs.

**What the codebase does:** `MetricsTracker` calls
`get_memory_stats()` every step and logs `gpu/allocated_gb`,
`gpu/peak_gb`, `gpu/reserved_gb`, `gpu/mem_utilization` to WandB /
TensorBoard. Start there — plot `peak_gb` over step count.

```
[step   50] gpu/peak_gb=78.4 | ...
[step  100] gpu/peak_gb=78.4 | ...
[step  500] gpu/peak_gb=78.4 | ...
[step 1000] gpu/peak_gb=78.6 | ...     ← +200 MB
[step 2000] gpu/peak_gb=79.1 | ...     ← +500 MB over another 1k steps
```

A flat curve is healthy; a staircase is a leak, and a steady slope is
fragmentation. For the staircase pattern, a snapshot is the fastest
way to identify the source:

```python
# In your training script, before `trainer.train()` or equivalent:
from kempnerforge.metrics.memory import DeviceMemoryMonitor

monitor = DeviceMemoryMonitor(
    device=torch.cuda.current_device(),
    snapshot_step=2000,
    snapshot_dir="memory_snapshots",
)

# Then after each step:
monitor.report(step)
```

At step 2000 it dumps a pickle you upload to
[pytorch.org/memory_viz](https://pytorch.org/memory_viz) for an
interactive flamegraph — stack traces to allocation sites, so leaks
reveal themselves. See
[Metrics § Memory snapshot export](../metrics-and-profiling/memory-monitor.md#memory-snapshot-export)
for the file layout.

### Fragmentation vs genuine leaks

If `reserved_gb - allocated_gb` grows over time, that's
fragmentation — PyTorch's allocator holds memory even after tensors
are freed. Usually fixable via:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

This is the same flag the 32-GPU MoE benchmark needs — see
[Scaling guide § Mis-sized activation memory](scaling-guide.md#mis-sized-activation-memory).

## Shape 3: a rank hangs without error

**Symptom:** logs stop advancing, no Python traceback, `py-spy
dump --pid <rank0>` shows threads blocked in NCCL.

**What the codebase does:** enable the periodic NCCL liveness probe
in your TOML:

```toml
[train]
nccl_health_check_interval = 100   # all-reduce sanity every 100 steps
```

Every 100 steps, every rank contributes a tensor of ones to a tiny
all-reduce. If the sum ≠ `world_size`, the training loop logs
`"NCCL health check failed at step <N> — stopping"` and breaks.
See
[Resilience § NCCL liveness](../resilience/nccl-liveness.md#periodic-check-in-the-loop).

The check catches **partial-participation bugs**, not true deadlocks —
the all-reduce itself hangs forever if one rank is genuinely dead
(which is why SLURM's `NCCL_TIMEOUT` is load-bearing). Combine both:

```bash
# In the SLURM script
export NCCL_TIMEOUT=1800   # 30 min — kills the job if NCCL hangs
```

### When it fires

Symptoms of a live check failure (sum mismatch): one rank returns a
different result, usually because it went through an unexpected code
branch (e.g., skipped a step due to NaN while others didn't). Hunt
for rank-specific behavior in the training loop.

## Shape 4: throughput silently halves

**Symptom:** steady-state `tok/s` drops, MFU drops, no error, loss
curve looks fine.

**What the codebase does:** start by reading
`train/step_time_sec` from the metrics backend. If it jumps from,
say, 0.5 s/step to 1.0 s/step and stays there, fire up the profiler:

```toml
[profiling]
enable     = true
start_step = 100        # wait for steady state
end_step   = 108        # 8 active steps (warmup + 7 recorded)
trace_dir  = "profiler_traces/"
```

Run for enough steps to cross `end_step`. When the profiler stops,
rank 0 prints a summary and writes `summary.md` into `trace_dir` with
a kernel breakdown:

```
| Category              | Time (s) |    % |
|-----------------------|---------:|-----:|
| MatMul/GEMM           |    1.245 | 62.4 |
| Communication (NCCL)  |    0.312 | 15.6 |
| Memory ops            |    0.186 |  9.3 |
| Other kernels         |    0.253 | 12.7 |
| **Total**             |    1.996 |100.0 |
```

[`print_profiler_summary`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/profiling/profiler.py)
does the classification via substring match on kernel names (see
`_analyze_profiler` for the GEMM / NCCL / memcpy patterns). See
[Metrics § Profiler](../metrics-and-profiling/profiler.md) for what
the table columns mean and how efficiency is computed.

Typical findings:

- **Communication (NCCL) share climbs** → network degraded (check IB
  counters on affected nodes), or you changed parallelism and
  collectives are now inter-node instead of intra-node.
- **Memory ops share climbs** → host-device copies added somewhere (a
  misplaced `.cpu()`, a fresh `numpy` conversion in the data loader).
- **Other kernels share climbs** → framework overhead grew, often due
  to a new Python loop over tensors.

Chrome traces land in `trace_dir/` as JSON — open in
`chrome://tracing` or `ui.perfetto.dev` for a timeline view. Look for
gaps between kernels (exposed latency) and per-step regressions.

## Shape 5: loss looks fine, downstream eval is worse

**Symptom:** training loss curve matches a good run, but
[`scripts/eval_harness.py`](run-evaluation.md#path-3-lm-eval-harness-for-downstream-tasks)
reports lower HellaSwag / MMLU than expected.

This is rarely a tool problem — it's a data or training-config
issue. Quick checks:

1. **Data mixing drift.** If you changed `[[data.datasets]]` weights
   between runs, loss on the mixed distribution can match while
   downstream behavior changes. Re-run the lost-run eval on the
   previous config's checkpoint at the same step — if downstream
   matches, data mix is the cause.
2. **LR / schedule change.** A lower peak LR often gives lower loss
   (training distribution is easier) but worse downstream (less
   exploration). Check `[scheduler]` and `[optimizer.lr]` against the
   reference run.
3. **Tokenizer / vocab mismatch.** If `model.vocab_size` silently
   changed between runs (wrong metadata.yaml, wrong
   `--model.vocab_size`), training loss is still sensible but
   generation and benchmarks rate as nonsense. See
   [Prepare tokenized data](prepare-tokenized-data.md) § "Validate
   with `prepare_data.py`".
4. **Bias in sampling.** If you're comparing `--temperature 0`
   (greedy) on lm-eval-harness to a checkpoint that trained with
   different temperature scaling or repetition penalty assumptions —
   the benchmark number is determined by decode settings you may not
   be matching.

No dedicated tool will catch #1–4; careful bookkeeping will.

## Chaining the tools

A debugging session usually follows this sequence:

```
Symptom
   │
   ├─ Loss NaN      → train/grad_norm + last-good checkpoint → restart lower LR
   ├─ OOM           → peak_gb trace → snapshot → memory_viz
   ├─ Rank hang     → NCCL liveness + NCCL_TIMEOUT → inspect rank-specific code
   ├─ Slowdown      → step_time_sec → torch.profiler → kernel summary
   └─ Bad eval      → re-run eval on baseline → audit data / LR / tokenizer
```

Most of these leave a trace in metrics; start there and narrow down.
Reaching for the profiler or a memory snapshot without a symptom wastes
time — the profiler adds startup overhead, and a snapshot dumps several
hundred MB per rank.

## See also

- [Resilience § NaN detection](../resilience/nan-detection.md) —
  detector internals and the three `action` modes.
- [Resilience § NCCL liveness](../resilience/nccl-liveness.md) —
  what the periodic probe checks and why `NCCL_TIMEOUT` matters.
- [Resilience § GPU health](../resilience/gpu-health.md) — on-demand
  compute/memory probe for a specific device.
- [Metrics § Memory monitor](../metrics-and-profiling/memory-monitor.md)
  — `DeviceMemoryMonitor` and snapshot export.
- [Metrics § Profiler](../metrics-and-profiling/profiler.md) —
  `torch.profiler` schedule and kernel classification.
- [Metrics § Metrics tracker](../metrics-and-profiling/metrics-tracker.md)
  — the per-step metrics that feed the workflows above.
- [Scaling guide § Common pitfalls](scaling-guide.md#common-pitfalls)
  — regression modes specific to parallelism changes.
