# KempnerForge

PyTorch-native framework for fault-tolerant distributed training of foundation models on AI clusters.

## AI / NeuroAI Use Cases

**Scaling law experiments** — Train the same architecture from 125M to 70B parameters by swapping a TOML config file. FSDP, tensor, expert, and pipeline parallelism are handled automatically. One codebase for compute-optimal scaling studies across model sizes.

**Mechanistic interpretability** — Built-in activation extraction hooks capture intermediate representations at any layer with automatic CPU offload. Attention weight capture exposes raw QK^T matrices. Feed extracted activations into your own CKA, SVCCA, RSA, or probing analysis pipelines without patching the training loop.

**Sparse architecture research** — Switch between dense and Mixture-of-Experts models with a single config flag (`num_experts=0` for dense). Two routing strategies (softmax top-k, DeepSeek-V3 sigmoid), shared experts, and configurable MoE frequency let you study expert specialization, capacity allocation, and routing dynamics.

**Optimizer and scheduler comparison** — Four optimizers (AdamW, Lion, Muon, Schedule-Free AdamW) and six LR schedulers (cosine, linear, WSD, constant, REX, none) are composable via config. Data annealing phases support curriculum learning with step-triggered weight shifts and LR scaling. Compare training dynamics without touching code.

**Long-running jobs on shared clusters** — SLURM preemption handling (SIGTERM/SIGUSR1), async distributed checkpointing, and automatic resume from the latest checkpoint. NaN detection with configurable actions (warn, skip, raise) and GPU/NCCL health monitoring keep multi-day runs alive on preemptible partitions.

**Representation analysis for NeuroAI** — Extract batch-level representations across entire datasets with CPU offload via `ActivationStore` and `extract_representations()`. Save to `.npz` for downstream comparison against neural recordings, cross-model similarity studies, or feature emergence analysis.

## Features

**Architecture**
- Decoder-only Transformer with RoPE, GQA, SwiGLU MLP, RMSNorm
- Mixture-of-Experts (MoE) with softmax top-k and DeepSeek-V3 sigmoid routers
- Shared experts, configurable MoE frequency, auxiliary load-balancing loss
- `torch.compile` support for fused kernels

**Parallelism**
- **FSDP2** — composable `fully_shard()`, per-block sharding, mixed precision (bf16/fp16/fp32)
- **Tensor Parallelism** — column/row parallel with SequenceParallel and meta-device init
- **Expert Parallelism** — all-to-all dispatch, multi-node EP+TP+FSDP2 composition
- **Pipeline Parallelism** — 1F1B/GPipe schedules via `torch.distributed.pipelining`
- **FP8 Mixed Precision** — E4M3/E5M2 via torchao with FSDP2 float8 all-gather

**Training**
- Optimizers: AdamW, Muon (Newton-Schulz orthogonalized momentum), Lion (sign-based, half optimizer memory), Schedule-Free AdamW (no LR schedule needed)
- LR schedulers: cosine, linear, WSD (cosine/linear/sqrt cooldown), constant, REX (polynomial decay), none
- Loss functions: cross-entropy, chunked cross-entropy (token-dimension chunking), z-loss regularizer
- Distributed checkpointing (DCP) with async save and auto-resume
- Stateful data pipeline with memory-mapped datasets and exact mid-epoch resumption
- Multi-dataset mixing with weighted sampling and temperature scaling
- Data annealing with step-triggered phase transitions (weight shifts, LR scaling)
- HuggingFace dataset integration (eager and streaming)
- SLURM integration with preemption handling, requeue, and multi-node launch
- Training loop hooks (`TrainingHook`, `HookRunner`) for extensibility without forking `train.py`
- Resilience: NaN detection, GPU health checks, NCCL liveness monitoring
- Metrics: MFU tracking, WandB/TensorBoard backends, memory monitoring

**Interpretability**
- Activation extraction hooks for probing, CKA, SVCCA analysis
- Attention weight capture (explicit QK^T path for mechanistic interpretability)
- Batch extraction over datasets with CPU offload

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
uv run torchrun --nproc_per_node=4 scripts/train.py configs/train/7b.toml

# With CLI overrides
uv run torchrun --nproc_per_node=4 scripts/train.py configs/train/7b.toml \
  --train.max_steps=1000 --optimizer.lr=1e-4

# SLURM (single node)
sbatch scripts/slurm/singlenode.sh configs/train/7b.toml

# SLURM (multi-node)
sbatch --nodes=4 scripts/slurm/multinode.sh configs/train/7b.toml

# SLURM (kempner_requeue — preemption-resilient, auto-resume)
sbatch scripts/slurm/7b_requeue.sh
```

## Training Configurations

### Dense models

```bash
# Debug (small model, fast iteration)
uv run python scripts/train.py configs/train/debug.toml

# Llama-3 7B, 4 GPUs, FSDP
uv run torchrun --nproc_per_node=4 scripts/train.py configs/train/7b.toml

# 7B with TP=4, 32 GPUs (8 nodes)
srun --nodes=8 --ntasks-per-node=4 --gpus-per-node=4 \
  uv run python scripts/train.py configs/train/7b_32gpu_fsdp.toml

# 70B with TP=4, 32 GPUs
srun --nodes=8 --ntasks-per-node=4 --gpus-per-node=4 \
  uv run python scripts/train.py configs/train/70b_32gpu_tp4.toml
```

### MoE models

```bash
# Debug MoE (4 experts, top-2, single GPU)
uv run python scripts/train.py configs/train/debug_moe.toml

# MoE 24 GPUs — 8 experts, TP=4, FSDP=6
srun --nodes=6 --ntasks-per-node=4 --gpus-per-node=4 \
  uv run python scripts/train.py configs/train/moe_24gpu.toml

# MoE + Expert Parallelism, 32 GPUs — 8 experts, TP=4, EP=2, FSDP=4
srun --nodes=8 --ntasks-per-node=4 --gpus-per-node=4 \
  uv run python scripts/train.py configs/train/moe_ep_32gpu.toml
```

### FP8 mixed precision

FP8 uses E4M3 forward / E5M2 backward with bf16 master weights. FSDP2 float8 all-gather halves communication volume.

```bash
# FP8 + FSDP, 4 GPUs
uv run torchrun --nproc_per_node=4 scripts/train.py configs/train/debug.toml \
  --train.mixed_precision=fp8

# FP8 + MoE + FSDP, 16 GPUs (experts/router excluded from Float8)
srun --nodes=4 --ntasks-per-node=4 --gpus-per-node=4 \
  uv run python scripts/train.py configs/train/debug_moe.toml \
  --train.mixed_precision=fp8

# FP8 + MoE + EP + FSDP, 16 GPUs
srun --nodes=4 --ntasks-per-node=4 --gpus-per-node=4 \
  uv run python scripts/train.py configs/train/debug_moe.toml \
  --train.mixed_precision=fp8 --distributed.ep=2 --distributed.dp_shard=8 \
  --train.activation_checkpointing=full
```

> **Note:** FP8 + Tensor Parallelism is not yet supported (torchao DTensor limitation). Use FP8 with FSDP only.

### Available configs

| Config | Model | Params | Parallelism | GPUs |
|--------|-------|--------|-------------|------|
| `debug.toml` | Dense | 20M | FSDP | 1-4 |
| `debug_moe.toml` | MoE (4 experts) | 23M | FSDP | 1-4 |
| `7b.toml` | Dense | 7B | FSDP | 4+ |
| `7b_16gpu_adamw.toml` | Dense | 7B | FSDP | 16 |
| `7b_16gpu_muon.toml` | Dense | 7B | FSDP (Muon) | 16 |
| `7b_16gpu_fp8.toml` | Dense | 7B | FP8, FSDP | 16 |
| `7b_32gpu_fsdp.toml` | Dense | 7B | FSDP | 32 |
| `7b_12gpu_tp4.toml` | Dense | 7B | TP=4, FSDP=3 | 12 |
| `13b_32gpu_tp4_pp2.toml` | Dense | 13B | TP=4, PP=2, FSDP=4 | 32 |
| `29b_32gpu_tp4_pp2.toml` | Dense | 29B | TP=4, PP=2, FSDP=4 | 32 |
| `70b_32gpu_tp4.toml` | Dense | 70B | TP=4, FSDP=8 | 32 |
| `70b_32gpu_tp4_pp4.toml` | Dense | 70B | TP=4, PP=4, FSDP=2 | 32 |
| `moe_24gpu.toml` | MoE (8 experts) | ~7B total | TP=4, FSDP=6 | 24 |
| `moe_ep_32gpu.toml` | MoE (8 experts) | ~4B total | TP=4, EP=2, FSDP=4 | 32 |
| `hf_wikitext.toml` | Dense | 20M | FSDP | 1-4 |

## Configuration

Configs are layered: **defaults -> TOML file -> CLI overrides**.

```bash
# CLI overrides use --section.key=value
uv run torchrun --nproc_per_node=4 scripts/train.py configs/train/7b.toml \
  --model.dim=2048 --train.batch_size=8 --optimizer.lr=1e-4
```

All config sections (`ModelConfig`, `TrainConfig`, `OptimizerConfig`, `SchedulerConfig`, `DistributedConfig`, `DataConfig`, `CheckpointConfig`, `MetricsConfig`, `ProfilingConfig`) are defined as typed dataclasses in `kempnerforge/config/` with one module per domain (e.g., `config/model.py`, `config/training.py`). Backward-compatible re-exports are available via `kempnerforge.config.schema`.

Key config options:
- `train.mixed_precision` — `"bf16"`, `"fp16"`, `"fp32"`, or `"fp8"` (default `"bf16"`)
- `train.activation_checkpointing` — `"none"`, `"full"`, or `"selective"` (default `"none"`)
- `distributed.tp` — tensor parallelism degree (default 1)
- `distributed.ep` — expert parallelism degree (default 1, requires MoE)
- `distributed.pp` — pipeline parallelism degree (default 1)
- `model.num_experts` — number of MoE experts (0 = dense model)
- `model.moe_router` — `"softmax_topk"` or `"sigmoid_topk"` (DeepSeek-V3 style)
- `optimizer.name` — `"adamw"`, `"muon"`, `"lion"`, or `"schedule_free_adamw"` (default `"adamw"`)
- `scheduler.name` — `"cosine"`, `"linear"`, `"wsd"`, `"constant"`, `"rex"`, or `"none"` (default `"cosine"`)
- `scheduler.wsd_decay_type` — WSD cooldown shape: `"cosine"`, `"linear"`, or `"sqrt"` (default `"cosine"`)
- `scheduler.rex_alpha` — REX polynomial exponent (default 1.0)
- `train.z_loss_weight` — z-loss regularization weight (0 = disabled, PaLM uses 1e-4)
- `train.ce_chunk_size` — token chunk size for chunked cross-entropy (0 = auto 4096)
- `checkpoint.async_mode` — `"disabled"`, `"async"`, or `"async_with_pinned_mem"`
- `train.shutdown_timeout_sec` — graceful shutdown timeout before forced exit (default 600)
- `train.nccl_health_check_interval` — check NCCL liveness every N steps (0 = disabled)

## Data

KempnerForge supports two data sources:

**Pre-tokenized (fastest)** — memory-mapped `.npy` shards on disk:
```toml
[data]
dataset_path = "data/your_dataset"
file_pattern = "tokenized_*.bin"
```

**HuggingFace datasets** — eager or streaming, tokenized on-the-fly:
```toml
[data]
hf_dataset_name = "wikitext"
hf_dataset_config = "wikitext-103-raw-v1"
tokenizer_path = "openai-community/gpt2"
hf_streaming = true
```

**Multi-dataset mixing** — weighted sampling with temperature scaling:
```toml
[[data.datasets]]
name = "code"
path = "data/code_shards"
weight = 0.3

[[data.datasets]]
name = "text"
path = "data/text_shards"
weight = 0.7

[data]
mix_temperature = 1.0  # >1 flattens weights, <1 sharpens
```

**Data annealing** — step-triggered phase transitions for curriculum learning:
```toml
[[data.phases]]
start_step = 5000
lr_scale = 0.5
dataset_weights = {code = 0.1, text = 0.9}

[[data.phases]]
start_step = 8000
lr_scale = 0.1
dataset_weights = {code = 0.05, text = 0.95}
```

## Project Structure

```
kempnerforge/
  config/      — Typed dataclass configs, TOML loading, CLI overrides, registry
  model/       — Transformer, attention, MLP, MoE, routers, norms, RoPE, embeddings, activation hooks
  distributed/ — DeviceMesh, FSDP2, tensor/expert/pipeline parallelism, FP8
  data/        — MemoryMappedDataset, MixtureDataset, StatefulDataLoader, MixtureSampler
  training/    — Optimizers (AdamW, Muon, Lion, Schedule-Free), loss functions, LR schedulers, gradient utils
  checkpoint/  — DCP-based distributed checkpointing with sync/async save
  resilience/  — Signal handling, NaN detection, GPU/NCCL health checks
  metrics/     — MetricsTracker, MFU computation, WandB/TensorBoard backends
  profiling/   — torch.profiler integration, CUDA timing
configs/       — TOML configs for training runs and model architecture presets
scripts/       — Training entry point, data validation, checkpoint conversion, SLURM launch
benchmarks/    — Performance benchmarks (forward pass, MoE, data pipeline, optimizer, MFU scaling)
tests/         — Unit (740), integration, distributed, and end-to-end tests
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

# Linting
uv run ruff check kempnerforge/ tests/

# Static type checking (zero errors in CI)
uv run pyright kempnerforge/
```

### End-to-End Tests

E2E tests launch full training runs as subprocesses and verify they complete successfully. They are **opt-in** — skipped by default, activated with `--e2e`. All tests are self-contained (random data or synthetic `.npy` shards in temp directories).

| Test | Parallelism | GPUs | What it verifies |
|------|-------------|------|------------------|
| Single GPU | — | 1 | Basic training loop, config loading |
| FSDP | dp_shard=4 | 4 | `build_parallel_model` non-TP path |
| TP only | tp=4 | 4 | Meta-device init, SequenceParallel |
| TP + FSDP | tp=2, dp_shard=2 | 4 | Combined parallelism |
| Pipeline Parallel | pp=2, dp_shard=2 | 4 | PP schedule, stage splitting, loss broadcast |
| fp16 | dp_shard=4, fp16 | 4 | `param_dtype` config path |
| FP8 | 1 GPU | 1 | Float8 conversion, forward/backward |
| FP8 + FSDP | dp_shard=4 | 4 | Float8 + FSDP2 float8 all-gather |
| FP8 + MoE + FSDP | dp_shard=4, MoE | 4 | Expert/router exclusion from Float8 |
| MoE single GPU | — | 1 | MoE training, aux loss, expert balance |
| MoE + FSDP | dp_shard=4 | 4 | MoE with FSDP |
| MoE + TP + FSDP | tp=2, dp_shard=2 | 4 | MoE with combined parallelism |
| MoE checkpoint | dp_shard=4, save+load | 4 | MoE checkpoint save and resume |
| Data pipeline | dp_shard=4, synthetic .npy | 4 | MemoryMappedDataset, sampler, dataloader |
| HF dataset | — | 1, 4 | HuggingFace eager dataset |
| HF streaming | — | 1, 4 | HuggingFace streaming dataset |
| Checkpoint resume | dp_shard=4, save+load | 4 | DCP save, auto-resume from checkpoint |
| PP checkpoint | pp=2, dp_shard=2 | 4 | Pipeline parallel checkpoint/resume |
| Z-loss | — | 1 | Z-loss regularizer via build_loss_fn |
| Chunked CE | — | 1 | Token-chunked cross-entropy |
| Muon single GPU | — | 1 | Muon optimizer, Newton-Schulz updates |
| Muon + FSDP | dp_shard=4 | 4 | Muon with DTensor momentum buffers |
| Z-loss + chunked CE + FSDP | dp_shard=4 | 4 | Combined loss composition |
| SIGTERM | — | 1 | Graceful shutdown, emergency checkpoint |
| Lion + FSDP | dp_shard=4 | 4 | Loss descent, gradients, checkpoint round-trip |
| Schedule-Free + FSDP | dp_shard=4 | 4 | Loss descent, eval/train toggle, checkpoint |
| Optimizer-scheduler matrix | dp_shard=4 | 4 | 7 (optimizer, scheduler) pairs, finite loss |
| 7B model (`--slow`) | tp=2, dp_shard=2, compile | 4 | Full production path |

## Parallelism Application Order

Parallelisms are applied in a strict order — wrong order causes silent correctness bugs:

1. **Tensor Parallelism** — must see raw `nn.Linear` modules
2. **Expert Parallelism** — partitions MoE experts across EP group
3. **Float8 Training** — converts `nn.Linear` to `Float8Linear` (excludes experts and router)
4. **Activation Checkpointing** — wraps blocks in `CheckpointWrapper`
5. **FSDP2** — shards everything (uses float8 all-gather when FP8 is enabled)

## Profiling

```bash
uv run torchrun --nproc_per_node=4 scripts/train.py configs/train/debug.toml \
    --profiling.enable=true --profiling.start_step=5 --profiling.end_step=8
```

Outputs kernel-level GPU time breakdown, FLOPS analysis, MFU estimate, and TensorBoard traces viewable at [Perfetto UI](https://ui.perfetto.dev/).

## Benchmarks

```bash
# Run all benchmarks
uv run python benchmarks/runner.py

# Run a specific benchmark
uv run python benchmarks/bench_forward.py

# Save results to JSON
uv run python benchmarks/runner.py --output results.json
```

Available benchmarks: forward pass throughput, MoE routing/dispatch, data pipeline I/O, optimizer step timing. Results from scaling experiments and profiling runs are stored in `benchmarks/` subdirectories.

## MoE Engineering Roadmap

Phases 1-9 (core MoE, Expert Parallelism, DeepSeekMoE, grouped GEMM, FSDP2 fix) and Phase 11 (FP8) are complete and validated at multi-node scale. Remaining work toward DeepSeek-V3 production quality:

| Phase | Feature | Status | Impact |
|:-----:|---------|:------:|--------|
| 10 | Router improvements (sequence aux loss, gradient scaling, adaptive bias) | Planned | Training quality |
| 11 | FP8 mixed precision | **Done** | 2x compute throughput |
| 12 | Pipeline parallelism for MoE (PP+EP+TP composition) | Blocked | Required for 100B+ |
| 13 | Communication-computation overlap (async EP dispatch) | Planned | 15-30% throughput |
| 14 | Node-limited expert routing (bounded cross-node traffic) | Planned | Scale to 64+ GPUs |
| 15 | Multi-token prediction (MTP) | Planned | 10-15% sample efficiency |
| 16 | Large-scale EP (hierarchical all-to-all, 256+ experts) | Planned | 1000+ GPU scale |

> **Note:** Dense Pipeline Parallelism (PP) is fully supported. MoE + PP is explicitly rejected at config validation time because MoE data-dependent routing is incompatible with static pipeline stage splitting. Use FSDP, TP, or EP for MoE models.

See `moe_eng_production_plan.md` for detailed implementation plans, steps, and test specifications.

## Design Principles

- **PyTorch-native**: FSDP2, DTensor, DeviceMesh, DCP, SDPA, torch.compile
- **Distributed-first**: multi-GPU is the default, not an afterthought
- **Composition over inheritance**: components composed via config, not class hierarchies
- **Minimal abstraction**: readable code over framework magic
- **Stateful everything**: dataloader, sampler, and training state all support checkpoint/resume
- **Configuration-driven**: all behavior controlled by typed dataclass configs, validated at startup
