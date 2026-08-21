# MoE Expert Parallelism Benchmark Report

**Date**: 2026-04-10
**Git**: `bc775fe` + FSDP per-sub-module wrapping fix
**Hardware**: 8 nodes, 4x NVIDIA H200 (141 GB) per node, NVLink intra-node, InfiniBand inter-node
**Dataset**: FineWeb-Edu (Llama-3 tokenized, 499B tokens)
**Steps**: 20 per experiment, steady-state metrics from last 10 steps
**Architecture**: Llama-3 MoE (decoder-only, GQA, SwiGLU, RMSNorm, RoPE, 8 experts top-2)

## Model

| Field | Value |
|-------|-------|
| Total params | ~4B |
| Active params/token | ~1.8B |
| dim | 2048 |
| layers | 24 |
| heads | 16 |
| kv_heads | 4 |
| experts | 8 |
| top_k | 2 |
| moe_frequency | 1 (all layers MoE) |
| router | softmax_topk |
| ffn_hidden | auto (1.3x) |

## Parallelism

DeviceMesh: `(dp_shard, ep, tp)` = `(4, 2, 4)` for EP experiments, `(8, 1, 4)` for non-EP baselines.

- **TP=4**: Intra-node NVLink, column/row parallel attention + MLP
- **EP=2**: Cross-node InfiniBand, partitions 8 experts into 4 per EP rank, all-to-all dispatch
- **FSDP=4 (EP) / FSDP=8 (no-EP)**: Shards parameters + gradients + optimizer state

## Results

### Before Fix: Deferred FSDP Wrapping (all MoE params in top-level wrap)

| batch_size | AC | Mem/GPU | tok/s | MFU (%) | Step Time |
|:---:|:---:|:---:|---:|:---:|---:|
| 1 | selective | 101.9 GB | 12,200 | 0.7 | 10.7s |
| 2 | selective | OOM (136.6 GB) | - | - | - |
| 2 | full | 57.9 GB | 14,200 | 0.8 | 9.5s |
| **8** | **full** | **87.9 GB** | **21,000** | **1.2** | **6.3s** |
| 12 | full | 119.9 GB | 16,200 | 0.9 | 12.0s |
| 16 | full | OOM | - | - | - |

### After Fix: Per-Sub-Module FSDP Wrapping (attention + MoE wrapped individually)

| batch_size | AC | Mem/GPU | tok/s | MFU (%) | Step Time |
|:---:|:---:|:---:|---:|:---:|---:|
| 2 | selective | 135.2 GB (OOM step 2) | - | - | - |
| 8 | full | **74.7 GB** | **24,200** | **1.4** | **5.4s** |
| **12** | **full** | **106.7 GB** | **27,000** | **1.5** | **7.4s** |
| 16 | full | 134.1 GB (OOM backward) | - | - | - |

### Improvement Summary

| batch_size | Metric | Before | After | Improvement |
|:---:|:---:|:---:|:---:|:---:|
| 8 | Memory | 87.9 GB | 74.7 GB | **-13.2 GB (-15%)** |
| 8 | Throughput | 21,000 tok/s | 24,200 tok/s | **+15%** |
| 12 | Memory | 119.9 GB | 106.7 GB | **-13.2 GB (-11%)** |
| 12 | Throughput | 16,200 tok/s | 27,000 tok/s | **+67%** |

The fix saves ~13 GB consistently and enables batch_size=12 as the new optimum (was batch_size=8). At batch_size=12, throughput increased 67% because the old approach had severe memory pressure at 119.9 GB (86% utilization) that caused allocator overhead, while the fix brings it to 106.7 GB (76% utilization).

### Key Metrics (batch_size=12, Best Config After Fix)

```
step  loss     lr        grad_norm  tok/s   mfu   mem        step_time
1     11.9900  6.00e-05  0.011      12,788  0.7%  102.2 GB  15.4s
5     11.9900  3.00e-04  0.012      17,972  1.0%  106.7 GB  10.9s
9     11.9895  2.55e-04  0.020      25,580  1.5%  106.7 GB   7.7s
12    11.9909  1.79e-04  0.070      25,281  1.5%  106.7 GB   7.8s
```

## Analysis

### The Fix: Per-Sub-Module FSDP Wrapping

**Problem**: EP-MoE blocks could not be individually `fully_shard()`-ed because per-block wrapping causes FSDP2's reduce-scatter to fire between EP's two backward all-to-all calls (NCCL deadlock). The workaround was deferring all MoE params to the top-level `fully_shard(model)`, which caused all 24 blocks' expert params to unshard at once and stay resident for the entire forward pass.

**Fix**: Instead of wrapping the entire block OR deferring everything, wrap `layer.attention` and `layer.mlp` (MoEMLP) as *separate* FSDP units within each block:

```python
# parallel.py — apply_fsdp2()
for layer in model.layers.values():
    if _has_ep_moe(layer):
        fully_shard(layer.attention, ...)  # attention: no EP, always safe
        fully_shard(layer.mlp, ...)        # MoE: reduce-scatter fires AFTER both EP all-to-alls
    else:
        fully_shard(layer, ...)            # dense blocks: wrap entire block as before
```

**Why this is safe**: FSDP2 bucketizes all params in a `fully_shard()` unit. For `fully_shard(layer.mlp)`, the reduce-scatter fires after the *last* param in MoEMLP gets its gradient — which is the router, computed after both EP all-to-all backward calls. All `dp_shard` peers share the same EP rank, so they reach the reduce-scatter at the same phase.

**Why this helps memory**: With per-MoE wrapping, FSDP reshards each block's MoE params after that block's forward, and reduces each block's gradients immediately after backward. Only 1 block's MoE params/grads are live at a time, vs all 24 blocks in the deferred approach.

### Why MFU Is Low

MFU of 1.5% is expected. The model is too small for 32 H200 GPUs:

- **Active params per GPU**: ~1.8B / 32 = ~56M. Each H200 is designed for billions.
- **Theoretical peak**: 32 x 990 TFLOPS (bf16) = 31,680 TFLOPS cluster-wide.
- **Communication overhead dominates**: EP all-to-all, FSDP all-gather/reduce-scatter, and TP collectives dwarf compute at this model size.

For context, the dense 13B model achieves **32.7% MFU** at the same 32-GPU scale (see `benchmarks/mfu_scaling/`). An MoE model targeting similar MFU would need ~50B+ total parameters (10B+ active).

### Selective AC Still Requires Full AC

Even with the fix, selective AC + batch_size=2 uses 135.2 GB (step 1 forward) and OOMs on backward. The bottleneck is **activation memory** from EP dispatch buffers and grouped GEMM intermediates — not parameter memory. FSDP wrapping strategy only helps with param/grad memory. Full AC remains essential for EP workloads because it avoids saving these large intermediate tensors.

### DTensor Redistribution Warning (Benign)

All runs emit a warning about suboptimal redistribution across `dp_shard` and `tp` mesh dimensions:

> While redistributing from (_NormPartial(2.0), _NormPartial(2.0)) to (Replicate(), Replicate()), 2 sequential all_reduce operations will be performed.

The 3D mesh `(dp_shard, ep, tp)` prevents DTensor from fusing the dp_shard and tp reductions into a single all-reduce. Minor impact at this scale.

### Training Correctness

- **Loss trajectory**: 11.990 -> 11.991 (expected for 20 steps with random init)
- **Aux loss**: Starts at 24.0 (uniform routing), decreases as router learns
- **Expert balance**: Near-zero initially, small values (~1e-3) emerge by step 10
- **Grad norms**: Growing from 0.011 to 0.151 — normal warmup behavior
- **Numerical consistency**: Loss values match between before/after fix (same model, same data)

## Reproduction

```bash
# 1. Get an interactive allocation (8 nodes, 4 H200 GPUs each)
salloc -p <partition> --account=<account> \
    --nodes=8 --ntasks-per-node=4 --gpus-per-node=4 \
    --cpus-per-task=16 --mem=1490G -t 00-04:00:00

# 2. Run the benchmark suite
bash benchmarks/moe_expert_parallel/moe_ep_bench.sh

# 3. Results saved to moe_ep_results/*.log

# Or run the best config directly:
srun --nodes=8 --ntasks-per-node=4 --gpus-per-node=4 \
    --kill-on-bad-exit=1 \
    --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run python scripts/train.py configs/train/moe_ep_32gpu.toml
```

## Future Work

- **Reduce EP activation memory**: The main memory bottleneck is EP dispatch buffers and grouped GEMM padding tensors saved for backward. Activation-level optimizations (in-place dispatch, lazy materialization) would reduce this.
- **Benchmark at scale**: Test with 64+ experts and 128+ GPUs where EP becomes necessary (experts don't fit on a single rank).
- **torch.compile for MoE**: Currently disabled due to data-dependent shapes in routing causing graph breaks. CUDAGraph-compatible routing would help.
- **EP + shared experts**: Benchmark DeepSeekMoE's shared expert with EP to measure the overhead of the non-dispatched path.
