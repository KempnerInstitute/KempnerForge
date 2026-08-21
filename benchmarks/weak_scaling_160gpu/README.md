# Weak Scaling to 160 H200 GPUs (dense 13B / 70B)

**Date**: 2026-06-30
**Git**: `3ad9d9a` (main)
**Hardware**: up to 40 nodes, 4x NVIDIA H200 (141 GB) per node, NVLink intra-node, InfiniBand inter-node
**Dataset**: FineWeb-Edu (Llama-3 tokenized)
**Steps**: 30 per scaling point, 200 per 160-GPU headline; steady state = median over the back half of each run
**Repeats**: 2 on both 160-GPU headlines, reported mean±std
**Architecture**: Llama-3 (decoder-only, GQA, SwiGLU, RMSNorm, RoPE)
**Config**: bf16 mixed precision, activation checkpointing (full), fused AdamW, cosine LR, `seq_len=4096`
**Parallelism**: TP=4 intra-node (NVLink) x FSDP across nodes, held fixed at every GPU count

This extends [`../mfu_scaling/`](../mfu_scaling/), which tops out at 32 GPUs, by a factor of
five. It answers a different question, though. That sweep compares parallelism *strategies* at
small scale; this one fixes the strategy and grows the machine.

Per-GPU batch is held constant, so tokens per step grow linearly with GPU count. That makes
this **weak** scaling: the problem grows with the resources. It is throughput and MFU only, 30
to 200 steps per config, and makes no claims about model quality.

## Models

| Model | Parameters | dim | layers | heads | kv_heads | ffn_hidden | batch/GPU | grad_accum |
|-------|-----------:|----:|-------:|------:|---------:|-----------:|----------:|-----------:|
| 13B | 14.84B | 5120 | 40 | 40 | 8 | 17,920 (1.3x auto) | 4 | 4 |
| 70B | 70.55B | 8192 | 80 | 64 | 8 | 28,672 | 2 | 4 |

## Results

| Model | GPUs | Parallelism | tok/s | MFU % | Mem/GPU GB | Step s | n |
|-------|-----:|-------------|------:|------:|-----------:|-------:|--:|
| 13b | 8 | tp4_fsdp2 | 29,860 | 35.9 | 45.0 | 4.39 | 1 |
| 13b | 32 | tp4_fsdp8 | 110,373 | 33.2 | 36.2 | 4.75 | 1 |
| 13b | 64 | tp4_fsdp16 | 216,507 | 32.6 | 34.8 | 4.84 | 1 |
| 13b | 128 | tp4_fsdp32 | 423,306 | 31.8 | 34.0 | 4.95 | 1 |
| 13b | 160 | tp4_fsdp40 | 534,644±1,138 | 32.2±0.1 | 33.9 | 4.91 | 2 |
| 70b | 8 | tp4_fsdp2 | 3,623 | 20.6 | 128.5 | 18.09 | 1 |
| 70b | 32 | tp4_fsdp8 | 18,688 | 26.5 | 93.2 | 14.03 | 1 |
| 70b | 64 | tp4_fsdp16 | 35,824 | 25.4 | 86.8 | 14.63 | 1 |
| 70b | 96 | tp4_fsdp24 | 52,190 | 24.7 | 84.6 | 15.07 | 1 |
| 70b | 128 | tp4_fsdp32 | 69,985 | 24.8 | 83.6 | 14.98 | 1 |
| 70b | 160 | tp4_fsdp40 | 85,571±43 | 24.3±0.0 | 82.9 | 15.32 | 2 |

### Weak-scaling efficiency

Ideal is linear in GPU count. Anchored at 32 GPUs rather than 8: the 8-GPU 70B point is
memory-bound at 128.5 GB of 141, so it makes a misleadingly low baseline.

**13B** (baseline 32 GPUs = 110,373 tok/s):

| GPUs | tok/s | MFU % | ideal tok/s | scaling eff % |
|-----:|------:|------:|------------:|--------------:|
| 8 | 29,860 | 35.9 | 27,593 | 108 |
| 32 | 110,373 | 33.2 | 110,373 | 100 |
| 64 | 216,507 | 32.6 | 220,746 | 98 |
| 128 | 423,306 | 31.8 | 441,492 | 96 |
| 160 | 534,644 | 32.2 | 551,865 | 97 |

**70B** (baseline 32 GPUs = 18,688 tok/s):

| GPUs | tok/s | MFU % | ideal tok/s | scaling eff % |
|-----:|------:|------:|------------:|--------------:|
| 8 | 3,623 | 20.6 | 4,672 | 78 |
| 32 | 18,688 | 26.5 | 18,688 | 100 |
| 64 | 35,824 | 25.4 | 37,376 | 96 |
| 96 | 52,190 | 24.7 | 56,064 | 93 |
| 128 | 69,985 | 24.8 | 74,752 | 94 |
| 160 | 85,571 | 24.3 | 93,440 | 92 |

### GPU telemetry (busy GPUs, mid-run)

| Config | SM / Tensor / DRAM active |
|--------|---------------------------|
| 13b_032gpu_tp4_fsdp8 | sm=61% tensor=43% dram=26% (n=12) |
| 13b_064gpu_tp4_fsdp16 | sm=59% tensor=37% dram=24% (n=12) |
| 13b_128gpu_tp4_fsdp32 | sm=44% tensor=28% dram=19% (n=12) |
| 13b_160gpu_tp4_fsdp40_r1 | sm=68% tensor=46% dram=28% (n=12) |
| 13b_160gpu_tp4_fsdp40_r2 | sm=59% tensor=39% dram=25% (n=12) |
| 70b_032gpu_tp4_fsdp8 | sm=38% tensor=24% dram=15% (n=11) |
| 70b_064gpu_tp4_fsdp16 | sm=47% tensor=32% dram=18% (n=12) |
| 70b_096gpu_tp4_fsdp24 | sm=35% tensor=23% dram=14% (n=16) |
| 70b_128gpu_tp4_fsdp32 | sm=47% tensor=32% dram=18% (n=12) |
| 70b_160gpu_tp4_fsdp40_r1 | sm=56% tensor=38% dram=21% (n=12) |
| 70b_160gpu_tp4_fsdp40_r2 | sm=45% tensor=30% dram=17% (n=12) |

## Analysis

**Scaling is not the bottleneck.** Weak-scaling efficiency holds at 92-97% out to 160 GPUs for
both models: 13B goes 100 → 98 → 96 → 97 from 32 to 160 GPUs, and 70B goes 100 → 96 → 93 → 94 →
92. There is no cliff and no accumulating straggler penalty over a 5x growth in machine size.

**MFU is nearly flat across scale.** 13B loses one point of MFU between 32 and 160 GPUs (33.2 to
32.2) and 70B loses 2.2 (26.5 to 24.3). Whatever limits efficiency at this model size is already
present at 32 GPUs; adding nodes does not make it materially worse.

**FSDP sharding does what it should.** Per-GPU memory falls monotonically as the shard count
grows: 13B from 45.0 GB at 8 GPUs to 33.9 GB at 160, and 70B from 128.5 GB to 82.9 GB. The 70B
model at 8 GPUs sits at 128.5 of 141 GB, which is why it only reaches 20.6% MFU there and why
that point is excluded from the efficiency baseline.

**13B outperforms 70B on MFU at every scale**, by 6.7 points at 32 GPUs and 7.9 at 160. This
does not invert the earlier sweep's finding that larger models use the hardware better, but it
does bound it: the trend is not monotonic in parameter count, and on this hardware it peaks
somewhere below 70B. Note the two are not a clean size comparison, since 70B runs at
`batch_size=2` against 13B's 4 and so has less arithmetic per collective to hide behind. The
telemetry agrees: 70B sits at 35-56% SM-active versus 44-68% for 13B.

## Overlap with the 1-32 GPU sweep

Two configurations appear in both this campaign and [`../mfu_scaling/`](../mfu_scaling/), and the
numbers differ slightly:

| Config | This campaign (2026-06-30, `3ad9d9a`) | mfu_scaling (2026-04-05, `01b461a`) |
|--------|--------------------------------------:|------------------------------------:|
| 13B, 32 GPUs, TP=4 + FSDP=8 | 33.2% / 110,373 tok/s | 32.7% / 104,309 tok/s |
| 70B, 32 GPUs, TP=4 + FSDP=8 | 26.5% / 18,688 tok/s | 25.4% / 17,657 tok/s |

Different runs, two months and many commits apart, both back-half medians. Neither is wrong;
prefer these when you need the more recent figure.

## Reproduction

Every config is submitted as its own SLURM job, so the allocation and the config are the same
node set. The full sweep is 13 jobs and roughly 3.5 hours of compute, dominated by the four
200-step headline runs.

```bash
export DATA=/path/to/tokenized/fineweb-edu     # tokenized_*.bin shards
export PARTITION=<gpu-partition>               # 4 GPUs per node
export ACCOUNT=<slurm-account>
export QOS=<qos>                               # only if your site requires one

# 1. Validate the path on 2 nodes first. Do not debug at 160 GPUs.
bash weak_scaling_160gpu_bench.sh preflight

# 2. The scaling curve (8 to 128 GPUs, 30 steps each).
bash weak_scaling_160gpu_bench.sh scaling

# 3. The 160-GPU headlines (200 steps, 2 repeats each).
bash weak_scaling_160gpu_bench.sh headline

# 4. Regenerate every table in this report from the logs.
uv run python parse_results.py results
```

`parse_results.py results` reproduces the Results, weak-scaling and telemetry tables above
exactly, so the numbers here can be re-derived rather than trusted. `results/` holds the 13 logs
behind them plus the telemetry CSVs, and the driver's config list covers exactly those 13.
