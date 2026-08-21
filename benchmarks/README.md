# Benchmarking and Performance Reports

This folder holds measurements of two different kinds.

**Campaign reports** are at-scale results: one directory per campaign, each with a standalone
report carrying its numbers alongside the hardware, commit and driver script that produced
them. These are the figures quoted in the top-level README and throughout the documentation.

The **micro-benchmark harness** (`runner.py` and the `bench_*.py` modules) times individual
components on a single GPU. It exists to catch a regression in a forward pass or an optimizer
step, not to characterize the framework at scale.

## Measured performance

Llama-3 architecture on NVIDIA H200 (141 GB), bf16 with full activation checkpointing, fused
AdamW, cosine LR, `seq_len=4096`. Best MFU per GPU count:

| GPUs | Nodes | Model | Best config | MFU | tok/s |
|-----:|------:|-------|-------------|----:|------:|
| 1 | 1 | 7B | single GPU | **57.8%** | 10,471 |
| 2 | 1 | 7B | FSDP=2 | 51.7% | 18,728 |
| 4 | 1 | 7B | FSDP=4 | **53.8%** | 38,983 |
| 8 | 2 | 13B | FSDP=8 | **44.4%** | 35,405 |
| 16 | 4 | 13B | TP=4 + FSDP=4 | 33.7% | 53,814 |
| 32 | 8 | 13B | TP=4 + FSDP=8 | 32.7% | 104,309 |
| 32 | 8 | 70B | TP=4 + FSDP=8 | 25.4% | 17,657 |

From [`mfu_scaling/`](mfu_scaling/), the 14-configuration sweep.
MFU is against the H200 bf16 peak of 989 TFLOP/s per GPU.

## Campaign reports

| Report | Date | Scale | What it establishes |
|--------|------|-------|---------------------|
| [`mfu_scaling/`](mfu_scaling/) | 2026-04-05 | 1-32 GPUs, 7B/13B/70B | 14-config dense sweep. FSDP beats TP by ~18 points when memory allows; the first inter-node hop is the biggest scaling cliff (93% to 53% efficiency); larger models hold MFU better past 8 GPUs. |
| [`config_validation/`](config_validation/) | 2026-04-06 | up to 32 GPUs | The 9 `configs/train/` presets that existed then all train: 9 of 9 PASS, with MFU, loss, memory and step time per config. Records that PP configs report lower MFU. |
| [`profiling/`](profiling/) | 2026-04-07 | 13B on 24 GPUs | Where GPU time actually goes on a multi-node run: 39.3% GEMM, 47.5% NCCL, 13.1% other kernels, at 216.9 achieved TFLOP/s. |
| [`moe_expert_parallel/`](moe_expert_parallel/) | 2026-04-10 | 32 GPUs, 4B MoE | Wrapping `layer.attention` and `layer.mlp` as separate FSDP2 units instead of the whole block: +67% throughput, -13.2 GB. Also explains why 1.5% MFU is the correct number for this model at this scale, not a bug. |
| [`moe_packed/`](moe_packed/) | 2026-04-13 | 8 GPUs, MoE E=8/16/64 | Packed expert storage beats unpacked at every expert count tested: +5.1% at E=8, +36.5% at E=16, +22.7% at E=64 with EP=4. Memory is at parity; the win grows with E. |

Each report states its own commit, hardware, dataset, step count, and steady-state window.
Those windows are not uniform across campaigns (some average the last 5 steps, others take a
median over the last 10), so read the header before comparing two reports directly.

Two standing caveats on the numbers:

- **MFU is reported for dense models only.** The MoE and MoT MFU formula underestimates by
  roughly 2x, so MoE campaigns report tok/s as the comparable figure.
- **Different campaigns ran at different commits.** Where two reports cover the same
  configuration, small disagreements are expected and are not errors in either one.

## Micro-benchmarks

Component-level timing via CUDA events. Requires a GPU; no cluster allocation needed.

```bash
# Everything, with a results table
uv run python benchmarks/runner.py

# One module
uv run python benchmarks/bench_forward.py

# Machine-readable output
uv run python benchmarks/runner.py --output results.json
```

| Module | Covers |
|--------|--------|
| [`bench_forward.py`](bench_forward.py) | Forward pass, forward+backward, attention, MLP (125M model) |
| [`bench_moe.py`](bench_moe.py) | MoE forward+backward, router comparison, grouped GEMM vs Python loop |
| [`bench_optimizer.py`](bench_optimizer.py) | Step time and memory for every registered optimizer |
| [`bench_data.py`](bench_data.py) | Memory-mapped iteration, sequence packing, mixture sampling (CPU only) |
| [`runner.py`](runner.py) | Timing utilities and the CLI the others share |

## Adding a campaign

One directory per campaign, named for the campaign. Inside it:

- **`README.md`** — the report, named `README.md` so GitHub renders it when someone opens the
  directory. Open with a header block giving date, commit, hardware, dataset, step count, and
  the steady-state window. Then the configuration under test, the results tables, the
  analysis, and a reproduction section with the exact commands.
- **`<campaign>_bench.sh`** — the driver that produced the numbers. It must run for someone
  else: take the dataset path from `DATA` (`DATA="${DATA:?...}"`), assume an existing
  allocation rather than hardcoding a partition, QoS or account, and carry no absolute paths.
  Commit `c900c15` removed hardcoded paths from these scripts once already.
- **`parse_results.py`** — optional, where logs need folding into the report's tables.

Then add a row to the table above, and update
[`docs/reference/benchmarks.md`](../docs/reference/benchmarks.md) so the documentation site
carries it too.

## See also

- [Parallelism recipes](../docs/reference/parallelism-recipes.md) — the configs behind these numbers.
- [Available configs](../docs/reference/available-configs.md) — every preset in `configs/train/`.
- [Scaling guide](../docs/how-to/scaling-guide.md) — how to pick a parallelism combination.
- [Architecture: parallelism order](../docs/architecture/parallelism-order.md) — why the mesh dimensions compose in a fixed order.
