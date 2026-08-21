# Benchmarking and Performance Reports

The campaign directories hold finished results. Each one records a specific run, pinned to the
commit, hardware and date behind it, and these are the figures the top-level README and the
documentation quote.

[`micro/`](micro/) is instruments rather than results. `runner.py` and the `bench_*.py` files
are a harness you execute to time individual components against whatever commit you have
checked out, on a single GPU.

## Measured performance

Llama-3 architecture on NVIDIA H200 (141 GB), bf16 with full activation checkpointing, fused
AdamW, cosine LR, `seq_len=4096`. Best measured MFU at each scale:

| GPUs | Nodes | Model | Best config | MFU | tok/s | Source |
|-----:|------:|-------|-------------|----:|------:|--------|
| 1 | 1 | 7B | single GPU | **57.8%** | 10,471 | [`mfu_scaling/`](mfu_scaling/) |
| 2 | 1 | 7B | FSDP=2 | 51.7% | 18,728 | [`mfu_scaling/`](mfu_scaling/) |
| 4 | 1 | 7B | FSDP=4 | **53.8%** | 38,983 | [`mfu_scaling/`](mfu_scaling/) |
| 8 | 2 | 13B | FSDP=8 | **44.4%** | 35,405 | [`mfu_scaling/`](mfu_scaling/) |
| 16 | 4 | 13B | TP=4 + FSDP=4 | 33.7% | 53,814 | [`mfu_scaling/`](mfu_scaling/) |
| 32 | 8 | 13B | TP=4 + FSDP=8 | 33.2% | 110,373 | [`weak_scaling_160gpu/`](weak_scaling_160gpu/) |
| 32 | 8 | 70B | TP=4 + FSDP=8 | 26.5% | 18,688 | [`weak_scaling_160gpu/`](weak_scaling_160gpu/) |
| 160 | 40 | 13B | TP=4 + FSDP=40 | 32.2% | 534,644 | [`weak_scaling_160gpu/`](weak_scaling_160gpu/) |
| 160 | 40 | 70B | TP=4 + FSDP=40 | 24.3% | 85,571 | [`weak_scaling_160gpu/`](weak_scaling_160gpu/) |
| 360 | 90 | 175B | TP=4 + FSDP=90 | **47.7%** | 154,523 | [`175b_360gpu/`](175b_360gpu/) |

MFU is against the H200 dense bf16 peak of 989 TFLOP/s per GPU.

Two things in that table are worth reading twice. **Efficiency improves at 175B**, to 47.7%,
rather than degrading with scale, because larger matmuls amortize communication and
kernel-launch overhead better. And **MFU is nearly flat from 32 to 160 GPUs**, so scaling out is
not what limits these runs.

The 32-GPU rows come from the later campaign, which measured 32 through 160 GPUs in one sweep, so
the curve is internally consistent. [`mfu_scaling/`](mfu_scaling/) measured the same two configs
in April and reported 32.7% and 25.4%; see the caveats below.

## Campaign reports

| Report | Date | Scale | What it establishes |
|--------|------|-------|---------------------|
| [`mfu_scaling/`](mfu_scaling/) | 2026-04-05 | 1-32 GPUs, 7B/13B/70B | 14-config dense sweep. FSDP beats TP by ~18 points when memory allows; the first inter-node hop is the biggest scaling cliff (93% to 53% efficiency); larger models hold MFU better past 8 GPUs. |
| [`config_validation/`](config_validation/) | 2026-04-06 | up to 32 GPUs | The 9 `configs/train/` presets that existed then all train: 9 of 9 PASS, with MFU, loss, memory and step time per config. Records that PP configs report lower MFU. |
| [`profiling/`](profiling/) | 2026-04-07 | 13B on 24 GPUs | Where GPU time actually goes on a multi-node run: 39.3% GEMM, 47.5% NCCL, 13.1% other kernels, at 216.9 achieved TFLOP/s. |
| [`moe_expert_parallel/`](moe_expert_parallel/) | 2026-04-10 | 32 GPUs, MoE | Wrapping `layer.attention` and `layer.mlp` as separate FSDP2 units instead of the whole block: +67% throughput, -13.2 GB. Also argues why single-digit MFU is the correct number for an MoE this size, not a bug. |
| [`moe_packed/`](moe_packed/) | 2026-04-13 | 8 GPUs, MoE E=8/16/64 | Packed expert storage beats unpacked at every expert count tested: +5.1% at E=8, +36.5% at E=16, +22.7% at E=64 with EP=4. Memory is at parity; the win grows with E. |
| [`weak_scaling_160gpu/`](weak_scaling_160gpu/) | 2026-06-30 | 8-160 GPUs, 13B/70B | Weak-scaling efficiency holds at 92-97% out to 160 GPUs and MFU is nearly flat, so scaling out is not the bottleneck. Per-GPU memory falls monotonically as FSDP shards further. 13B beats 70B on MFU at every scale, bounding the "bigger is more efficient" trend. |
| [`175b_360gpu/`](175b_360gpu/) | 2026-07-03 | 360 GPUs, 175B dense | The largest run to date: 47.7% MFU sustained for ~11 h, compute-bound by telemetry (78% SM / 62% tensor / 24% DRAM), async checkpointing at 0.9% of wall-clock. Also documents why `grad_accum > 1` OOMs at this size and why batch, not accumulation, is the knob. |

Each report states its own commit, hardware, dataset, step count, and steady-state window.
Those windows are not uniform across campaigns (some average the last 5 steps, others take a
median over the last 10), so read the header before comparing two reports directly.

Two standing caveats on the numbers:

- **MFU is reported for dense models only.** The MoE and MoT MFU formula underestimates by
  roughly 2x, so MoE campaigns report tok/s as the comparable figure.
- **Different campaigns ran at different commits.** Where two reports cover the same
  configuration, small disagreements are expected and are not errors in either one.

## Micro-benchmarks

Component-level timing via CUDA events. Requires a GPU, no cluster allocation. Results are
printed, not stored, unless you ask for JSON.

```bash
# Everything, with a results table
uv run python benchmarks/micro/runner.py

# One module
uv run python benchmarks/micro/bench_forward.py

# Machine-readable output
uv run python benchmarks/micro/runner.py --output results.json
```

| Module | Covers |
|--------|--------|
| [`micro/bench_forward.py`](micro/bench_forward.py) | Forward pass, forward+backward, attention, MLP (125M model) |
| [`micro/bench_moe.py`](micro/bench_moe.py) | MoE forward+backward, router comparison, grouped GEMM vs Python loop |
| [`micro/bench_optimizer.py`](micro/bench_optimizer.py) | Step time and memory for every registered optimizer |
| [`micro/bench_data.py`](micro/bench_data.py) | Memory-mapped iteration, sequence packing, mixture sampling (CPU only) |
| [`micro/runner.py`](micro/runner.py) | `run_benchmark()`, `BenchmarkResult`, and the CLI the others share |

The four `bench_*.py` modules all measure through `runner.run_benchmark()` on purpose. Do not
copy that function into a module to make it standalone: four copies of CUDA-event timing will
drift, and the moment they do, numbers from two modules stop being comparable.

## Adding a campaign

One directory per campaign, named for the campaign, holding everything specific to it:

```
benchmarks/<campaign>/
  README.md            the report
  <campaign>_bench.sh  the driver
  parse_results.py     optional: logs -> the report's tables
  results/             the driver's output
```

- **`README.md`**: Open with a header block giving date, commit, hardware, dataset, step count, and the steady-state window. Then the configuration under test, the results tables, the analysis, and a reproduction section with the exact commands.
- **`<campaign>_bench.sh`**: The driver that produced the numbers, and it has to run for someone who is not you. Take the dataset path from the environment (`DATA="${DATA:?...}"`), assume an existing allocation instead of hardcoding a partition, QoS or account, and carry no absolute paths. 
- **`parse_results.py`**: Optional, but it is what makes the report checkable. With a parser and a committed `results/`, the tables in the README can be re-derived.
- **`results/`**: The driver's raw output. Keep the logs for the runs that appear in the report. Leave out crash dumps from failed configs and high-frequency telemetry; both run to tens of megabytes and neither is what a reader needs.

Anchor the driver to its own location rather than the caller's, so the folder works from any
working directory and its output stays inside it:

```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${KEMPNERFORGE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"
mkdir -p "$RESULTS_DIR"
cd "$REPO_ROOT"          # so scripts/train.py and configs/ resolve
```

The four existing drivers predate this and write to a directory beside the repo root instead.
New ones should not.

The guiding rule: a campaign may depend on the framework, never on another campaign. A driver
that reads a sibling's results, or a report whose argument only works if you have read the
folder next door, is a defect. Citing a sibling's number for context is fine, as long as you
name the source so the report still reads on its own.

Then add a row to the table above, and update
[`docs/reference/benchmarks.md`](../docs/reference/benchmarks.md) so the documentation site
carries it too.

## See also

- [Parallelism recipes](../docs/reference/parallelism-recipes.md) — the configs behind these numbers.
- [Available configs](../docs/reference/available-configs.md) — every preset in `configs/train/`.
- [Scaling guide](../docs/how-to/scaling-guide.md) — how to pick a parallelism combination.
- [Architecture: parallelism order](../docs/architecture/parallelism-order.md) — why the mesh dimensions compose in a fixed order.
