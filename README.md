# KempnerForge

A researcher-friendly, high-performance LLM training framework for large-scale distributed training on HPC clusters.

## Features

- **Decoder-only Transformer** with RoPE, GQA, SwiGLU MLP, RMSNorm
- **FSDP2** with composable `fully_shard()`, configurable mixed precision (bf16/fp16/fp32)
- **Tensor Parallelism** with SequenceParallel, real-valued RoPE, meta-device initialization
- **Pipeline Parallelism** via `torch.distributed.pipelining` with 1F1B/GPipe schedules
- **Distributed Checkpointing** (DCP) with async save and auto-resume
- **Stateful Data Pipeline** with memory-mapped datasets and exact mid-epoch resumption
- **SLURM Integration** with preemption handling, requeue support, and multi-node launch scripts
- **Resilience** — NaN detection, GPU health checks, NCCL liveness monitoring
- **Metrics** — MFU tracking, WandB/TensorBoard backends, memory monitoring
- **`torch.compile`** support for fused kernels

## Requirements

- Python >= 3.12
- PyTorch >= 2.4 (CUDA)
- [uv](https://docs.astral.sh/uv/) package manager

## Quick Start

```bash
# Install dependencies
uv sync

# Single GPU (debug)
uv run python scripts/train.py configs/train/debug.toml

# Multi-GPU (single node, 4 GPUs)
uv run torchrun --nproc_per_node=4 scripts/train.py configs/train/default.toml

# With CLI overrides
uv run torchrun --nproc_per_node=4 scripts/train.py configs/train/default.toml \
  --train.max_steps=1000 --optimizer.lr=1e-4

# SLURM (single node)
sbatch scripts/slurm/launch.sh configs/train/default.toml

# SLURM (multi-node)
sbatch scripts/slurm/multinode.sh configs/train/default.toml
```

## Configuration

Configs are layered: **defaults → TOML file → CLI overrides**.

```bash
# TOML presets
configs/train/debug.toml       # small model, fast iteration
configs/train/default.toml     # standard training config
configs/train/llama7b_bench.toml  # 7B model benchmark

# CLI overrides use --section.key=value
uv run torchrun --nproc_per_node=4 scripts/train.py configs/train/default.toml \
  --model.dim=2048 --train.batch_size=8 --optimizer.lr=1e-4
```

All config sections (`ModelConfig`, `TrainConfig`, `OptimizerConfig`, `SchedulerConfig`, `DistributedConfig`, `DataConfig`, `CheckpointConfig`, `MetricsConfig`, `ProfilingConfig`) are defined as typed dataclasses in `kempnerforge/config/schema.py`.

Key config options:
- `model.init_std` — weight initialization standard deviation (default 0.02)
- `train.mixed_precision` — `"bf16"`, `"fp16"`, or `"fp32"` (default `"bf16"`)
- `checkpoint.async_mode` — `"disabled"`, `"async"`, or `"async_with_pinned_mem"`

## Data Preparation

Tokenize a dataset into memory-mapped `.npy` shards:

```bash
uv run python scripts/prepare_data.py
```

Then point your config to the output directory:

```toml
[data]
dataset_path = "data/your_dataset"
```

## Project Structure

```
kempnerforge/
  config/      — Typed dataclass configs, TOML loading, CLI overrides, registry
  model/       — Transformer architecture (attention, MLP, norms, RoPE, embeddings)
  distributed/ — DeviceMesh, FSDP2, tensor parallelism, distributed utils
  data/        — MemoryMappedDataset, StatefulDataLoader, DistributedSampler
  training/    — Optimizer (AdamW), LR schedulers (cosine/linear/WSD), gradient utils
  checkpoint/  — DCP-based distributed checkpointing with sync/async save
  resilience/  — Signal handling, NaN detection, GPU/NCCL health checks
  metrics/     — MetricsTracker, MFU computation, WandB/TensorBoard backends
  profiling/   — torch.profiler integration, CUDA timing
configs/       — TOML configs for models, training, and cluster settings
scripts/       — Training entry point, data prep, profiling, SLURM launch scripts
tests/         — Unit (299), integration, distributed, and end-to-end tests
```

## Testing

```bash
# Unit tests (no GPU)
uv run pytest tests/unit/

# Integration tests (1 GPU)
uv run pytest tests/integration/

# Distributed tests (4 GPUs)
uv run torchrun --nproc_per_node=4 -m pytest tests/distributed/ -v

# End-to-end tests (opt-in, 4 GPUs, ~2.5 min)
uv run pytest tests/e2e/ --e2e -v

# End-to-end including 7B model (~3 min)
uv run pytest tests/e2e/ --e2e --slow -v

# Everything (unit + e2e)
uv run pytest tests/ --e2e --slow

# Linting
uv run ruff check kempnerforge/ tests/
```

### End-to-End Tests

E2E tests launch full training runs as subprocesses and verify they complete successfully. They are **opt-in** — skipped by default, activated with `--e2e`. All tests are self-contained (random data or synthetic `.npy` shards in temp directories).

| Test | Parallelism | GPUs | What it verifies |
|------|-------------|------|------------------|
| Single GPU | — | 1 | Basic training loop, config loading |
| FSDP | dp_shard=4 | 4 | `build_parallel_model` non-TP path |
| TP only | tp=4 | 4 | Meta-device init, SequenceParallel |
| TP + FSDP | tp=2, dp_shard=2 | 4 | Combined parallelism |
| Pipeline Parallel | pp=2, dp_shard=2 | 4 | PP schedule, stage splitting |
| fp16 | dp_shard=4, fp16 | 4 | `param_dtype` config path |
| Data pipeline | dp_shard=4, synthetic .npy | 4 | MemoryMappedDataset, sampler, dataloader |
| Checkpoint resume | dp_shard=4, save+load | 4 | DCP save, auto-resume from checkpoint |
| 7B model (`--slow`) | tp=2, dp_shard=2, compile | 4 | Full production path with 7B Llama |

## Profiling

```bash
uv run torchrun --nproc_per_node=4 scripts/profile_run.py configs/train/llama7b_bench.toml
```

Outputs kernel-level GPU time breakdown, FLOPS analysis, MFU estimate, and a Chrome trace viewable at [Perfetto UI](https://ui.perfetto.dev/).

## Design Principles

- **PyTorch-native**: FSDP2, DTensor, DeviceMesh, DCP, SDPA, torch.compile
- **Distributed-first**: multi-GPU is the default, not an afterthought
- **Composition over inheritance**: components composed via config, not class hierarchies
- **Minimal abstraction**: readable code over framework magic
- **Stateful everything**: dataloader, sampler, and training state all support checkpoint/resume
- **Configuration-driven**: all behavior controlled by typed dataclass configs, validated at startup
