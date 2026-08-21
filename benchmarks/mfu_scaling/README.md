# KempnerForge MFU Benchmark Report

**Date**: 2026-04-05   
**Git**: `01b461a` (main)   
**Hardware**: 8 nodes, 4x NVIDIA H200 (141 GB) per node, NVLink intra-node, InfiniBand inter-node   
**Dataset**: FineWeb-Edu (Llama-3 tokenized, 499B tokens)   
**Steps**: 20 per experiment, steady-state MFU averaged over last 5 steps   
**Architecture**: Llama-3 (decoder-only, GQA, SwiGLU, RMSNorm, RoPE)   
**Config**: bf16 mixed precision, activation checkpointing (full), fused AdamW, cosine LR schedule   

## Models

| Model | Parameters | dim | layers | heads | kv_heads | ffn_hidden |
|-------|-----------|-----|--------|-------|----------|------------|
| 7B | 8.03B | 4096 | 32 | 32 | 8 | auto (1.3x) |
| 13B | 14.84B | 5120 | 40 | 40 | 8 | auto (1.3x) |
| 70B | 70.55B | 8192 | 80 | 64 | 8 | 28672 |

## Results

### Full Results Table

| # | GPUs | Nodes | Model | Parallelism | tok/s | MFU (%) | Mem/GPU | Step Time |
|---|------|-------|-------|-------------|------:|--------:|--------:|----------:|
| 1 | 1 | 1 | 7B | — | 10,471 | **57.8** | 80.7 GB | 3.13s |
| 2 | 2 | 1 | 7B | FSDP=2 | 18,728 | 51.7 | 62.4 GB | 3.50s |
| 3 | 2 | 1 | 7B | TP=2 | 13,163 | 36.4 | 57.8 GB | 2.49s |
| 4 | 4 | 1 | 7B | FSDP=4 | 38,983 | **53.8** | 61.1 GB | 3.36s |
| 5 | 4 | 1 | 7B | TP=4 | 25,135 | 34.7 | 39.4 GB | 1.30s |
| 6 | 4 | 1 | 7B | TP=2+FSDP=2 | 25,980 | 35.9 | 45.9 GB | 2.52s |
| 7 | 8 | 2 | 7B | TP=4+FSDP=2 | 44,299 | 30.6 | 32.7 GB | 1.48s |
| 8 | 8 | 2 | 13B | FSDP=8 | 35,405 | **44.4** | 88.5 GB | 14.81s |
| 9 | 8 | 2 | 13B | TP=4+FSDP=2 | 27,345 | 34.3 | 45.1 GB | 4.79s |
| 10 | 16 | 4 | 7B | TP=4+FSDP=4 | 82,988 | 28.6 | 29.4 GB | 1.58s |
| 11 | 16 | 4 | 13B | TP=4+FSDP=4 | 53,814 | **33.7** | 39.2 GB | 4.87s |
| 12 | 32 | 8 | 7B | TP=4+FSDP=8 | 155,917 | 26.9 | 27.6 GB | 1.68s |
| 13 | 32 | 8 | 13B | TP=4+FSDP=8 | 104,309 | **32.7** | 36.3 GB | 5.03s |
| 14 | 32 | 8 | 70B | TP=4+FSDP=8 | 17,657 | 25.4 | 93.2 GB | 14.85s |

### Best MFU by GPU Count

| GPUs | Best Config | MFU (%) | tok/s |
|------|-------------|--------:|------:|
| 1 | 7B, single GPU | 57.8 | 10,471 |
| 2 | 7B, FSDP=2 | 51.7 | 18,728 |
| 4 | 7B, FSDP=4 | 53.8 | 38,983 |
| 8 | 13B, FSDP=8 | 44.4 | 35,405 |
| 16 | 13B, TP=4+FSDP=4 | 33.7 | 53,814 |
| 32 | 13B, TP=4+FSDP=8 | 32.7 | 104,309 |

### Throughput Scaling (7B model)

| GPUs | tok/s | Ideal (linear) | Scaling Efficiency |
|------|------:|---------------:|-----------------:|
| 1 | 10,471 | 10,471 | 100% |
| 2 | 18,728 | 20,942 | 89% |
| 4 | 38,983 | 41,884 | 93% |
| 8 | 44,299 | 83,768 | 53% |
| 16 | 82,988 | 167,536 | 50% |
| 32 | 155,917 | 335,072 | 47% |

## Analysis

### Parallelism Strategy Comparison

**FSDP dominates for MFU** when memory allows. At 4 GPUs, pure FSDP (53.8%) beats TP=4 (34.7%) and TP=2+FSDP=2 (35.9%) by ~18 percentage points. FSDP only communicates gradients during backward, while TP requires all-gather/reduce-scatter on every forward and backward matmul.

**TP becomes necessary** when models exceed single-GPU memory or when FSDP alone can't shard enough. The 13B model at 8 GPUs shows this: FSDP=8 (44.4%) beats TP=4+FSDP=2 (34.3%), but FSDP=8 requires 88.5 GB/GPU while TP+FSDP only uses 45.1 GB. For larger models (70B), TP is required to keep per-GPU memory manageable.

**Larger models achieve higher MFU at multi-node scale.** At 32 GPUs, 13B (32.7%) beats 7B (26.9%) by 6 points. The 7B model is too small — compute finishes faster than communication can overlap, leaving GPUs idle during all-reduce. Larger models have a better compute-to-communication ratio.

### Scaling Observations

- **Intra-node (1->4 GPUs)**: Near-linear scaling with FSDP (93% efficiency). NVLink bandwidth is sufficient.
- **Inter-node (4->8 GPUs)**: Significant drop (53% efficiency) as communication crosses to InfiniBand. This is the biggest scaling cliff.
- **Multi-node (8->32 GPUs)**: Gradual degradation (53%->47%) — IB bandwidth becomes the bottleneck, but less dramatic than the initial node boundary crossing.

### Memory Utilization

- **7B on 1 GPU**: 80.7/140 GB (58%) — well-fitted
- **13B on 8 GPUs (FSDP=8)**: 88.5/140 GB (63%) — good utilization
- **70B on 32 GPUs**: 93.2/140 GB (67%) — fits with headroom
- **7B on 32 GPUs**: 27.6/140 GB (20%) — severely underutilized, model too small for this scale

## Batch Configuration

All experiments used `seq_len=4096` with `torch.compile=true` (except experiment 08 which used `compile=false`).

| Model | batch_size | grad_accum | Effective Batch (tokens/step/GPU) |
|-------|-----------|------------|--------------------------------:|
| 7B | 4 | 2 | 32,768 |
| 13B | 4 | 4 | 65,536 |
| 70B | 2 | 4 | 32,768 |

## Reproduction

This report was generated using `benchmarks/mfu_scaling/mfu_bench.sh`, which runs all 14 experiments
sequentially within a SLURM interactive allocation.

```bash
# 1. Get an interactive allocation (8 nodes, 4 H200 GPUs each)
salloc -p <partition-name> --account=<account-name> \
    --nodes=8 --ntasks-per-node=4 --gpus-per-node=4 \
    --cpus-per-task=16 --mem=1490G -t 00-10:00:00

# 2. Run the full benchmark suite (takes ~30-45 min)
bash benchmarks/mfu_scaling/mfu_bench.sh

# 3. Results are saved to mfu_results/*.log (one per experiment)
```

The script handles both single-node experiments (via `torchrun`) and multi-node
experiments (via `srun` with automatic port selection). Each experiment runs 20 steps
and logs per-step metrics including MFU, throughput, memory, and step time.

To run individual experiments, use the training configs directly:

```bash
# Single-node example (7B, FSDP=4, 4 GPUs)
uv run torchrun --nproc_per_node=4 scripts/train.py configs/train/debug.toml \
  --model.dim=4096 --model.n_layers=32 --model.n_heads=32 --model.n_kv_heads=8 \
  --model.vocab_size=128256 --model.ffn_dim_multiplier=1.3 --train.max_steps=20 \
  --distributed.dp_shard=4

# Multi-node example (70B, TP=4+FSDP=8, 32 GPUs)
srun --nodes=8 --ntasks=32 --gpus-per-node=4 --cpus-per-task=4 \
  uv run python scripts/train.py configs/train/70b_32gpu_tp4.toml \
  --train.max_steps=20
```
