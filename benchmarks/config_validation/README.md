# KempnerForge Config Validation Report

**Date**: 2026-04-06   
**Git**: `601aba9` (main)   
**Hardware**: 8 nodes, 4x NVIDIA H200 (141 GB) per node, NVLink intra-node, InfiniBand (NDR 400Gbps) inter-node   
**Dataset**: FineWeb-Edu (Llama-3 tokenized, 499B tokens) for fineweb configs; Wikitext-103 (HF streaming) for hf_wikitext   
**Architecture**: Llama-3 (decoder-only, GQA, SwiGLU, RMSNorm, RoPE)   
**Config**: bf16 mixed precision, activation checkpointing (full), fused AdamW, cosine LR schedule, `cpu:gloo,cuda:nccl` backend   

## Purpose

Validate all training configs in `configs/train/` after:
1. Renaming configs to `{model}_{gpus}gpu_{strategy}.toml` convention
2. Fixing multi-node distributed bugs (gloo backend, IB interface detection, async checkpoint flush)
3. Fixing content issues (data paths, vocab_size, num_workers, PP comment errors)
4. Fixing PP loss logging (schedule.step() API, broadcast from last stage)

## Configs Tested

| Config | Model | GPUs | Nodes | Parallelism | Data Source |
|--------|-------|------|-------|-------------|-------------|
| `debug.toml` | 20M | 4 | 1 | FSDP=4 | Random (no dataset) |
| `hf_wikitext.toml` | 77M | 4 | 1 | FSDP=4 | HF streaming (wikitext-103) |
| `7b.toml` | 8.0B | 4 | 1 | FSDP=4 | FineWeb-Edu |
| `7b_12gpu_tp4.toml` | 8.0B | 12 | 3 | TP=4, FSDP=3 | FineWeb-Edu |
| `7b_32gpu_fsdp.toml` | 8.0B | 32 | 8 | FSDP=32 | FineWeb-Edu |
| `13b_32gpu_tp4_pp2.toml` | 14.8B | 32 | 8 | TP=4, PP=2, FSDP=4 | FineWeb-Edu |
| `29b_32gpu_tp4_pp2.toml` | 28.7B | 32 | 8 | TP=4, PP=2, FSDP=4 | FineWeb-Edu |
| `70b_32gpu_tp4.toml` | 70.6B | 32 | 8 | TP=4, FSDP=8 | FineWeb-Edu |
| `70b_32gpu_tp4_pp4.toml` | 70.6B | 32 | 8 | TP=4, PP=4, FSDP=2 | FineWeb-Edu |

## Results

### Summary

| Config | Status | Steps | MFU (%) | tok/s | Mem/GPU | Step Time | Loss |
|--------|--------|------:|--------:|------:|--------:|----------:|-----:|
| `debug.toml` | **PASS** | 10 | 1.7 | 537,787 | 0.7 GB | 0.02s | 10.44 |
| `hf_wikitext.toml` | **PASS** | 10 | 12.4 | 958,700 | 6.5 GB | 0.07s | 10.06 |
| `7b.toml` | **PASS** | 10 | 53.7 | 38,858 | 81.0 GB | 13.49s | 12.45 |
| `7b_12gpu_tp4.toml` | **PASS** | 10 | 30.9 | 67,045 | 30.4 GB | 5.86s | 11.75 |
| `7b_32gpu_fsdp.toml` | **PASS** | 10 | 51.3 | 297,384 | 51.3 GB | 7.05s | 13.56 |
| `13b_32gpu_tp4_pp2.toml` | **PASS** | 20 | 16.5 | 52,554 | 24.0 GB | 4.99s | 11.75 |
| `29b_32gpu_tp4_pp2.toml` | **PASS** | 20 | 13.6 | 22,688 | 32.8 GB | 5.78s | 11.75 |
| `70b_32gpu_tp4.toml` | **PASS** | 20 | 25.6 | 17,769 | 93.2 GB | 14.76s | 11.75 |
| `70b_32gpu_tp4_pp4.toml` | **PASS** | 20 | 11.0 | 7,641 | 51.0 GB | 4.29s | 11.75 |

Steady-state values (last 5 steps).

### KempnerPulse GPU Metrics (7b_32gpu_fsdp, captured mid-training)

```
gpu_id  model  gpu_util%  mem_mib   real_util%  sm_active%  tensor_active%  dram_active%
0       H200   100.00     73,302    68.75       87.56       69.04           21.69
1       H200   100.00     73,338    70.93       90.64       71.24           22.40
2       H200   100.00     73,302    71.27       90.79       72.14           22.28
3       H200   100.00     73,298    71.42       91.09       72.18           22.47
```

All 4 local GPUs at 100% utilization, ~91% SM active, ~71% tensor core active.

### Parallelism Topology

| Config | DeviceMesh | Mesh Shape | torch.compile |
|--------|-----------|------------|:-------------:|
| `debug.toml` | [dp_shard] | [4] | no |
| `hf_wikitext.toml` | [dp_shard] | [4] | no* |
| `7b.toml` | [dp_shard] | [4] | yes |
| `7b_12gpu_tp4.toml` | [dp_shard, tp] | [3, 4] | yes |
| `7b_32gpu_fsdp.toml` | [dp_shard] | [32] | yes |
| `13b_32gpu_tp4_pp2.toml` | [pp, dp_shard, tp] | [2, 4, 4] | no |
| `29b_32gpu_tp4_pp2.toml` | [pp, dp_shard, tp] | [2, 4, 4] | no |
| `70b_32gpu_tp4.toml` | [dp_shard, tp] | [8, 4] | yes |
| `70b_32gpu_tp4_pp4.toml` | [pp, dp_shard, tp] | [4, 2, 4] | no |

`*` Tested with `--train.compile_model=false` for speed; config default is `true`.
PP configs use `compile_model=false` because torch.compile has limited PP support.

### Memory Utilization

| Config | Params/GPU | Mem/GPU | % of 140 GB | Headroom |
|--------|-----------|---------|:-----------:|---------:|
| `debug.toml` | 5M | 0.7 GB | 0.5% | 139 GB |
| `hf_wikitext.toml` | 19M | 6.5 GB | 4.6% | 134 GB |
| `7b.toml` | 2.0B | 81.0 GB | 57.9% | 59 GB |
| `7b_12gpu_tp4.toml` | 0.7B | 30.4 GB | 21.7% | 110 GB |
| `7b_32gpu_fsdp.toml` | 0.25B | 51.3 GB | 36.6% | 89 GB |
| `13b_32gpu_tp4_pp2.toml` | 0.9B | 24.0 GB | 17.1% | 116 GB |
| `29b_32gpu_tp4_pp2.toml` | 1.8B | 32.8 GB | 23.4% | 107 GB |
| `70b_32gpu_tp4.toml` | 8.8B | 93.2 GB | 66.6% | 47 GB |
| `70b_32gpu_tp4_pp4.toml` | 4.4B | 51.0 GB | 36.4% | 89 GB |

70B with TP=4+FSDP=8 is the tightest fit at 93.2/140 GB (67%). All configs have sufficient headroom for longer training runs with larger batch sizes.

## Known Issues

### MFU lower for PP configs

PP configs show lower MFU (12-17%) compared to pure FSDP (51%) or TP+FSDP (26-31%). This is expected due to:
- **Pipeline bubble**: With small `grad_accum_steps` (4), the pipeline bubble fraction is significant (~25% for PP=2, ~50% for PP=4)
- **No torch.compile**: PP configs disable `compile_model` due to limited torch.compile support with pipeline schedules
- **Small batch size**: PP configs use `batch_size=1` (29b, 70b_pp4) to fit in memory

To improve PP MFU: increase `grad_accum_steps` to reduce bubble fraction, or use `gpipe` schedule for better throughput at the cost of higher memory.

## Changes Validated

### Config Renames

| Old Name | New Name |
|----------|----------|
| `default.toml` | `7b.toml` |
| `llama7b_bench.toml` | *(merged into 7b.toml)* |
| `wikitext_test.toml` | `hf_wikitext.toml` |
| `multinode_7b_fsdp32.toml` | `7b_32gpu_fsdp.toml` |
| `multinode_tp4_fsdp3.toml` | `7b_12gpu_tp4.toml` |
| `multinode_13b_tp4_pp2_fsdp4.toml` | `13b_32gpu_tp4_pp2.toml` |
| `multinode_29b_tp4_pp2_fsdp4.toml` | `29b_32gpu_tp4_pp2.toml` |
| `multinode_70b_tp4_fsdp8.toml` | `70b_32gpu_tp4.toml` |
| `multinode_70b_tp4_pp2_fsdp4.toml` | `70b_32gpu_tp4_pp4.toml` |

### Content Fixes

- **`7b.toml`** (was default.toml): Added fineweb-edu data path, updated vocab 32000→128256, added norm/activation/rope settings
- **`hf_wikitext.toml`**: Converted from nonexistent disk path to HF streaming dataset
- **`70b_32gpu_tp4_pp4.toml`**: Fixed filename (was pp2, actual PP=4), fixed comment ("80 layers → 20 per stage")
- **PP configs** (13b, 29b, 70b_pp4): Fixed `num_workers` 0→4 (removed data loading bottleneck)
- **All configs**: Updated checkpoint dirs to match new filenames

### PP Loss Logging Fix

- **schedule.step() API**: `losses` must be collected via the `losses=` output parameter (list populated by the schedule), not from the return value (which is the model output)
- **Loss broadcast**: Added `dist.broadcast()` with `group_src=pp_size-1` to send avg_loss and grad_norm from the last PP stage to all other stages, so rank 0 can log correct metrics
- **Re-validated** all 3 PP configs (13b, 29b, 70b_pp4) — rank 0 now reports loss=11.75 (was 0.0)

### Distributed Fixes (validated in prior session)

- **Backend**: `"nccl"` → `"cpu:gloo,cuda:nccl"` (required for async DCP)
- **Gloo interface binding**: Added `GLOO_SOCKET_IFNAME` alongside `NCCL_SOCKET_IFNAME` (prevents gloo binding to management Ethernet)
- **Dynamic IB detection**: `ip -br addr` auto-detects first UP InfiniBand interface in both shell scripts and Python
- **Async checkpoint flush**: Added `ckpt_mgr.wait()` before `destroy_distributed()` in train.py
- **singlenode.sh**: Updated to match multinode.sh (dynamic IB, GLOO_SOCKET_IFNAME, removed deprecated TORCH_NCCL_AVOID_RECORD_STREAMS)

## Reproduction

```bash
# 1. Get an interactive allocation (8 nodes, 4 H200 GPUs each)
salloc -p <partition> --account=<account> \
    --nodes=8 --ntasks-per-node=4 --gpus-per-node=4 \
    --cpus-per-task=16 --mem=1490G -t 00-10:00:00

# 2. Set up environment (on head node)
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT=29500
IB_IFNAME=$(ip -br addr | awk '/^ib[0-9]+\s+UP\s+[0-9]/ {print $1; exit}')
export NCCL_SOCKET_IFNAME="${IB_IFNAME:-ib0}"
export GLOO_SOCKET_IFNAME="${IB_IFNAME:-ib0}"
export NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 NCCL_IB_GID_INDEX=3

# 3. Single-node configs (run directly on head node)
uv run torchrun --nproc_per_node=4 scripts/train.py configs/train/debug.toml --train.max_steps=10
uv run torchrun --nproc_per_node=4 scripts/train.py configs/train/hf_wikitext.toml --train.max_steps=10
uv run torchrun --nproc_per_node=4 scripts/train.py configs/train/7b.toml --train.max_steps=10

# 4. Multi-node configs (via srun)
srun --nodes=3 --ntasks-per-node=4 --gpus-per-node=4 --kill-on-bad-exit=1 \
    uv run python scripts/train.py configs/train/7b_12gpu_tp4.toml --train.max_steps=10

srun --nodes=8 --ntasks-per-node=4 --gpus-per-node=4 --kill-on-bad-exit=1 \
    uv run python scripts/train.py configs/train/7b_32gpu_fsdp.toml --train.max_steps=10

srun --nodes=8 --ntasks-per-node=4 --gpus-per-node=4 --kill-on-bad-exit=1 \
    uv run python scripts/train.py configs/train/13b_32gpu_tp4_pp2.toml

# ... and so on for 29b, 70b_tp4, 70b_tp4_pp4

# 5. GPU metrics during training (requires kempnerpulse)
kempnerpulse --once --export --show-all
```
