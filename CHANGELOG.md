# Changelog

All notable changes to KempnerForge are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `install-and-verify` plugin skill: runs `uv sync`, asserts Python ≥ 3.12, then runs the four CI gate checks (`ruff check`, `ruff format --check`, `pyright`, `pytest tests/unit/`). Canonical first command after cloning.
- `.python-version` pinned to `>=3.12` so uv resolves the interpreter explicitly. Teammates on 3.13 use 3.13 (no download); 3.11-only users get 3.12 auto-fetched.

### Changed
- `docs/getting-started/install.md` Prerequisites: documents `.python-version` and uv's auto-fetch behavior.
- `README.md` and `kempnerforge/README.md` Prerequisites: clarify that uv auto-fetches Python 3.12 via `.python-version`.
- `docs/claude-ready.md` first-run flow: `/kempnerforge:install-and-verify` runs before `/kempnerforge:cluster-config`.
- `README.md` and `kempnerforge/README.md`: list `install-and-verify` in the skill catalog and drop the hardcoded skill count.

## [0.1.0] — 2026-04-16

Initial public release.

### Architecture
- Decoder-only Transformer with RoPE, GQA, SwiGLU MLP, RMSNorm
- Optional QK-Norm (Gemma/DeepSeek-V3 style)
- Mixture-of-Experts (MoE) with softmax top-k and DeepSeek-V3 sigmoid routers
- Shared experts, configurable MoE frequency, auxiliary load-balancing loss
- Grouped GEMM and packed expert storage for MoE throughput
- Sequence-level aux loss, gradient scaling, adaptive bias schedule for sigmoid router

### Parallelism
- **FSDP2** — composable `fully_shard()`, per-block sharding, mixed precision (bf16/fp16/fp32)
- **Tensor Parallelism** — column/row parallel with SequenceParallel and meta-device init
- **Expert Parallelism** — all-to-all dispatch, multi-node EP+TP+FSDP2 composition
- **Pipeline Parallelism** — 1F1B/GPipe/interleaved-1F1B schedules via `torch.distributed.pipelining`
- **FP8 Mixed Precision** — E4M3/E5M2 via torchao with FSDP2 float8 all-gather
- **SDPA backend override** — `model.sdpa_backend` config (`auto`/`flash`/`efficient`/`cudnn`/`math`) for kernel benchmarking and debugging

### Training
- Optimizers: AdamW, Muon (Newton-Schulz orthogonalized momentum), Lion (half optimizer memory), Schedule-Free AdamW
- LR schedulers: cosine, linear, WSD (cosine/linear/sqrt cooldown), constant, REX, none
- Loss functions: cross-entropy, chunked cross-entropy, z-loss regularizer
- Distributed checkpointing (DCP) with async save and auto-resume
- Stateful data pipeline with memory-mapped datasets and exact mid-epoch resumption
- Multi-dataset mixing with weighted sampling and temperature scaling
- Data annealing with step-triggered phase transitions
- HuggingFace dataset integration (eager and streaming)
- Training loop hooks (`TrainingHook`, `HookRunner`) for extensibility

### Resilience
- SLURM preemption handling (SIGTERM/SIGUSR1) with cooperative shutdown
- NaN detection with configurable actions (warn, skip, raise)
- GPU compute/memory health checks
- NCCL liveness monitoring via all-reduce

### Interpretability
- Activation extraction hooks (`ActivationStore`) with CPU offload
- Attention weight capture (explicit QK^T path for mechanistic interpretability)
- Batch extraction over datasets for CKA/SVCCA/probing pipelines

### Metrics & Observability
- Per-step MetricsTracker with EMA smoothing
- MFU computation from architecture params and GPU peak FLOPs
- WandB and TensorBoard backends
- Peak memory monitoring with optional snapshot export

### Configuration
- Typed dataclass configs per domain (`ModelConfig`, `TrainConfig`, etc.)
- Layered loading: defaults → TOML file → CLI overrides
- Fail-fast validation at object construction
- Registry for swappable components (optimizers, schedulers, losses, routers, norms)

### Testing
- 794 unit tests (CPU-only)
- Integration tests (1 GPU)
- Distributed tests (4 GPUs via torchrun)
- End-to-end tests (opt-in full training runs)
- Smoke tests across parallelism configurations

[Unreleased]: https://github.com/KempnerInstitute/KempnerForge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/KempnerInstitute/KempnerForge/releases/tag/v0.1.0
