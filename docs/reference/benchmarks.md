# Benchmarks

Two benchmark reports live under
[`benchmarks/`](https://github.com/KempnerInstitute/KempnerForge/tree/main/benchmarks)
in the repo. Each is a standalone markdown report with numbers,
reproduction commands, and the shell script that produced them.

## MFU scaling (dense)

Report:
[`benchmarks/mfu_scaling/mfu_scaling.md`](https://github.com/KempnerInstitute/KempnerForge/blob/main/benchmarks/mfu_scaling/mfu_scaling.md).
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

## MoE Expert Parallelism

Report:
[`benchmarks/moe_expert_parallel/moe_ep_benchmark.md`](https://github.com/KempnerInstitute/KempnerForge/blob/main/benchmarks/moe_expert_parallel/moe_ep_benchmark.md).
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
- [README § Benchmarks](https://github.com/KempnerInstitute/KempnerForge#benchmarks)
  — summary of both reports at the repo root.
