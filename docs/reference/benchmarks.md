# Benchmarks

Benchmark campaigns live under
[`benchmarks/`](https://github.com/KempnerInstitute/KempnerForge/tree/main/benchmarks)
in the repo, one directory each. Every campaign is a standalone report with
its numbers, the commit and hardware behind them, reproduction commands,
and the driver script that produced them.

The four dense and MoE campaigns are summarized below. `benchmarks/README.md`
indexes all of them, including config validation and multi-node profiling,
and carries the combined MFU table.

## MFU scaling (dense)

Report:
[`benchmarks/mfu_scaling/`](https://github.com/KempnerInstitute/KempnerForge/tree/main/benchmarks/mfu_scaling).
Driver:
[`benchmarks/mfu_scaling/mfu_bench.sh`](https://github.com/KempnerInstitute/KempnerForge/blob/main/benchmarks/mfu_scaling/mfu_bench.sh).

14 experiments across 7B / 13B / 70B Llama-3 on 1–32 H200s (8 nodes ×
4 H200 141 GB). bf16, full activation checkpointing, fused AdamW,
cosine LR, `seq_len=4096`. Steady-state MFU averaged over the last 5
of 20 steps.

### Best MFU by GPU count

| GPUs | Best config | MFU | tok/s |
|-----:|-------------|----:|------:|
| 1 | 7B, single GPU | 57.8% | 10,471 |
| 2 | 7B, FSDP=2 | 51.7% | 18,728 |
| 4 | 7B, FSDP=4 | 53.8% | 38,983 |
| 8 | 13B, FSDP=8 | 44.4% | 35,405 |
| 16 | 13B, TP=4+FSDP=4 | 33.7% | 53,814 |
| 32 | 13B, TP=4+FSDP=8 | 32.7% | 104,309 |

### Headline observations

- **FSDP dominates when memory allows.** At 4 GPUs on 7B, pure FSDP
  hits 53.8% MFU, vs 34.7% for pure TP=4 and 35.9% for TP=2+FSDP=2.
  TP fires all-gather/reduce-scatter on every matmul; FSDP fires once
  per step.
- **Biggest scaling cliff is the first inter-node hop.** 7B drops from
  93% linear efficiency at 4 GPUs (intra-node NVLink) to 53% at 8 GPUs
  (inter-node IB). Subsequent scaling (8→32) degrades gradually,
  53%→47%.
- **Larger models scale better past 8 GPUs.** At 32 GPUs the 7B model
  uses only 27.6 GB of 140 GB and hits 26.9% MFU; 13B uses 36.3 GB and
  hits 32.7% MFU; 70B (with TP=4+FSDP=8) fits at 93.2 GB and holds
  25.4% MFU. Compute-to-communication ratio is the lever.
- **70B needs TP.** Pure FSDP can't shard 70B enough to fit on H200s
  without activation-checkpoint-aware sharding of attention; TP=4
  across a node cuts the per-GPU attention and MLP state by 4×.

### Reproduction

```bash
salloc -p <partition> --account=<account> \
    --nodes=8 --ntasks-per-node=4 --gpus-per-node=4 \
    --cpus-per-task=16 --mem=1490G -t 00-10:00:00
bash benchmarks/mfu_scaling/mfu_bench.sh
```

The driver runs all 14 experiments sequentially inside a single
interactive allocation. Results go to `mfu_results/*.log` (one per
experiment). The full report prints the per-experiment log layout at
its tail.

## Weak scaling to 160 GPUs (dense)

Report:
[`benchmarks/weak_scaling_160gpu/`](https://github.com/KempnerInstitute/KempnerForge/tree/main/benchmarks/weak_scaling_160gpu).
Driver:
[`benchmarks/weak_scaling_160gpu/weak_scaling_160gpu_bench.sh`](https://github.com/KempnerInstitute/KempnerForge/blob/main/benchmarks/weak_scaling_160gpu/weak_scaling_160gpu_bench.sh).

13B and 70B Llama-3 from 8 to 160 H200s (up to 40 nodes × 4). Parallelism
is held fixed at TP=4 intra-node × FSDP across nodes, and per-GPU batch is
held constant, so the total problem size grows with the machine — weak
scaling, not strong. Steady state is the median over the back half of each
run; the two 160-GPU points are repeated twice and reported mean ± std.

### Weak-scaling efficiency

Anchored at 32 GPUs, since the 8-GPU 70B point is memory-bound at 128.5 GB
of 141 and makes a misleadingly low baseline.

| GPUs | 13B MFU | 13B eff | 70B MFU | 70B eff |
|-----:|--------:|--------:|--------:|--------:|
| 32 | 33.2% | 100% | 26.5% | 100% |
| 64 | 32.6% | 98% | 25.4% | 96% |
| 96 | — | — | 24.7% | 93% |
| 128 | 31.8% | 96% | 24.8% | 94% |
| 160 | 32.2% | 97% | 24.3% | 92% |

### Headline observations

- **Scaling out is not the bottleneck.** Efficiency holds at 92–97% over a
  5× growth in machine size, with no cliff and no accumulating straggler
  penalty.
- **MFU is nearly flat.** 13B loses one point between 32 and 160 GPUs, 70B
  loses 2.2. Whatever limits efficiency is already present at 32 GPUs.
- **FSDP sharding works as intended.** Per-GPU memory falls monotonically:
  13B from 45.0 GB at 8 GPUs to 33.9 GB at 160, 70B from 128.5 to 82.9 GB.
- **13B beats 70B at every scale**, by 6.7 points at 32 GPUs. This bounds
  the "larger models use the hardware better" trend rather than inverting
  it; note 70B runs at `batch_size=2` against 13B's 4.

### Reproduction

Each config is its own SLURM job, so the allocation and the config are the
same node set. 13 jobs, roughly 3.5 hours.

```bash
export DATA=/path/to/tokenized/fineweb-edu
export PARTITION=<gpu-partition> ACCOUNT=<slurm-account>
bash benchmarks/weak_scaling_160gpu/weak_scaling_160gpu_bench.sh preflight
bash benchmarks/weak_scaling_160gpu/weak_scaling_160gpu_bench.sh scaling
bash benchmarks/weak_scaling_160gpu/weak_scaling_160gpu_bench.sh headline
uv run python benchmarks/weak_scaling_160gpu/parse_results.py \
    benchmarks/weak_scaling_160gpu/results
```

`parse_results.py` regenerates every table in the report from the committed
logs, so the numbers can be re-derived rather than trusted.

## 175B dense on 360 GPUs

Report:
[`benchmarks/175b_360gpu/`](https://github.com/KempnerInstitute/KempnerForge/tree/main/benchmarks/175b_360gpu).
Driver:
[`benchmarks/175b_360gpu/175b_360gpu_bench.sh`](https://github.com/KempnerInstitute/KempnerForge/blob/main/benchmarks/175b_360gpu/175b_360gpu_bench.sh).

A 175B-parameter dense decoder-only Transformer on 360 H200s (90 nodes ×
4), TP=4 intra-node × FSDP2 `dp_shard=90` across nodes, for 2,000 steps
and about 11 hours. The largest run performed with KempnerForge to date.
A systems benchmark, not a pre-training run: 5.9B tokens leave the loss
healthy but unconverged.

| Metric | Value |
|--------|-------|
| Sustained MFU | **47.7%** |
| Throughput | 154,523 tokens/s |
| Step time | 19.09 s (global batch 2.95M tokens) |
| Aggregate model FLOPs | ~170 PFLOP/s |
| Peak memory | 53.8 / 140 GB per GPU |
| Loss | 11.75 → 5.47 over 5.90B tokens |
| Async checkpoint cost | +60 s mean, 0.9% of wall-clock |
| Compute activity | 78% SM / 62% tensor / 24% DRAM |

### Headline observations

- **Efficiency improves at this size.** 47.7% exceeds the same framework's
  24.3% at 70B and 32.2% at 13B on 160 GPUs. Larger matmuls amortize
  communication and kernel-launch overhead better.
- **The run is compute-bound, not communication- or memory-bound.** High SM
  and tensor activity against low DRAM activity is the signature, so the
  86 GB of unused memory is not the constraint — recomputation FLOPs from
  full activation checkpointing are.
- **`grad_accum > 1` OOMs at 175B.** With gradient accumulation FSDP2 keeps
  the unsharded fp32 gradient across microbatches, roughly the model size
  in fp32 divided by TP, about 87 GB. The failure is sequence-independent.
  Scale the batch instead: `batch_size=1` reaches only 8.1% MFU because one
  microbatch cannot amortize the FSDP collectives, while `batch_size=8`
  reaches the same global batch in one forward/backward at 47.7%.
- **Async checkpointing is not free but is affordable.** Staging a 175B
  state stalls the step by about a minute; at a 300-step interval that is
  0.9% of wall-clock, and it is the input a Young–Daly interval calculation
  needs.

### Reproduction

One config, one job: 90 nodes for about 11 hours. The driver is a dry run
unless passed `GO`.

```bash
export DATA=/path/to/tokenized/fineweb-edu
export PARTITION=<gpu-partition> ACCOUNT=<slurm-account>
export CKPT_DIR=/path/to/scratch          # room for one 175B checkpoint
bash benchmarks/175b_360gpu/175b_360gpu_bench.sh        # dry run
bash benchmarks/175b_360gpu/175b_360gpu_bench.sh GO     # submit

uv run python benchmarks/175b_360gpu/parse_results.py \
    benchmarks/175b_360gpu/results/175b-360gpu.log
```

Both `parse_results.py` and `make_figures.py` read only the committed
training log, so every number and both figures in the report can be
regenerated.

## MoE Expert Parallelism

Report:
[`benchmarks/moe_expert_parallel/`](https://github.com/KempnerInstitute/KempnerForge/tree/main/benchmarks/moe_expert_parallel).
Driver:
[`benchmarks/moe_expert_parallel/moe_ep_bench.sh`](https://github.com/KempnerInstitute/KempnerForge/blob/main/benchmarks/moe_expert_parallel/moe_ep_bench.sh).

32-GPU MoE: 8 nodes × 4 H200. Mesh `(dp_shard=4, ep=2, tp=4)`.
Architecture: `dim=2048`, 24 layers, 8 experts top-2, `moe_frequency=1`.
~4B total params, ~1.8B active per token.

### Per-sub-module FSDP wrapping fix

The report documents a measured improvement from wrapping
`layer.attention` and `layer.mlp` as separate FSDP2 units (instead of
the entire block or deferring MoE params to the top-level wrap). At
batch_size=12 with full activation checkpointing:

| Metric | Before fix | After fix | Change |
|--------|-----------:|----------:|-------:|
| Memory | 119.9 GB | 106.7 GB | −13.2 GB (−11%) |
| Throughput | 16,200 tok/s | 27,000 tok/s | +67% |
| Step time | 12.0 s | 7.4 s | −38% |

The 67% throughput improvement at batch_size=12 comes from relieving
allocator pressure — at 119.9 GB (86% utilization) before the fix, the
allocator spent substantial time on fragmentation; post-fix at 106.7
GB (76%) it doesn't. Batch_size=12 overtook batch_size=8 as the
optimum after the fix.

### Why MFU is 1.5%

MFU of 1.5% is correct for this model at this scale, not a bug. ~1.8B
active parameters on 32 × H200 means ~56M active params per GPU —
each H200 is designed for billions. Communication overhead (EP
all-to-all, FSDP all-gather/reduce-scatter, TP collectives) dwarfs
compute. For comparison, a dense 13B model hits 32.7% MFU on the same
hardware; an MoE model targeting similar MFU would need ~50B total
parameters (~10B active).

### Reproduction

```bash
salloc -p <partition> --account=<account> \
    --nodes=8 --ntasks-per-node=4 --gpus-per-node=4 \
    --cpus-per-task=16 --mem=1490G -t 00-04:00:00
bash benchmarks/moe_expert_parallel/moe_ep_bench.sh
```

Or run the best config directly:

```bash
srun --nodes=8 --ntasks-per-node=4 --gpus-per-node=4 \
    --kill-on-bad-exit=1 \
    --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run python scripts/train.py configs/train/moe_ep_32gpu.toml
```

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is load-bearing for
this config — fragmentation under bf16 EP without it pushes peak
memory above the limit.

## See also

- [Parallelism recipes](parallelism-recipes.md) — the configs behind
  the 7B/13B/70B and MoE EP numbers.
- [Architecture § Parallelism order](../architecture/parallelism-order.md)
  — why the mesh dimensions compose in a specific order.
- [`benchmarks/README.md`](https://github.com/KempnerInstitute/KempnerForge/tree/main/benchmarks)
  — the index of every campaign, with the combined MFU table across all
  measured scales and the conventions for adding a new one.
