# Changelog

All notable changes to KempnerForge are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **VLM (Vision-Language Model) foundation + Joint-Decoder arch.** A registry-driven, discriminated-union design so future arches (Cross-Attention, Mixture-of-Transformers) can be added as small additive PRs without changes to existing call sites.
  - `kempnerforge/config/vlm.py`: `VLMConfig` base + `JointDecoderConfig(arch="joint_decoder")` registered via `@registry.register_vlm_config`. `FreezeSpec`, `FreezeStage`, and `DEFAULT_MODULE_PATTERNS` for the freeze plumbing. `_RESERVED_ARCHS = ("cross_attention", "mot")` so TOMLs aimed at future arches get a clear `NotImplementedError`.
  - `kempnerforge/config/registry.py`: `register_vision_encoder` / `register_vlm_config` / `register_modality_strategy` registries.
  - `kempnerforge/config/model.py`: `ModelConfig.vlm` (defaults to `None`, zero behavior change for text-only) + `is_vlm` property + `max_seq_len ≥ residual_image_tokens + max_text_len` cross-check.
  - `kempnerforge/config/loader.py`: TOML `[model.vlm]` table dispatch on `arch`; nested-dataclass / variadic-tuple coercion so `freeze` and `freeze_schedule` round-trip correctly from TOML.
  - `kempnerforge/model/modality.py`: `ModalityContext` (groups `inputs_embeds`, `prefix_embeds`, `output_slice`) with mutual-exclusion invariants. `Transformer.forward` consumes the context.
  - `kempnerforge/model/vision.py`: `VisionEncoder` base + `RandomVisionEncoder` (deterministic test stub) + `_HFVisionEncoder` (SigLIP2 / CLIP wrapper).
  - `kempnerforge/model/vlm.py`: `Adapter` (2-layer MLP, in_dim → out_dim with optional hidden_dim), `VLMWrapper` (vision_encoder + adapter + transformer composed via a `ModalityStrategy`), `JointDecoderStrategy` registered via `@registry.register_modality_strategy("joint_decoder")`, `inner_transformer(model)` unwrap helper.
  - `kempnerforge/model/transformer.py`: routes `Transformer.forward` through `ModalityContext` (`prefix_embeds`/`output_slice`/`inputs_embeds`); cross-arg invariants enforced against `kv_caches` (training-only routes).
  - `kempnerforge/data/vlm_dataset.py`: `HuggingFaceVLMDataset` + `VLMCollator` (fixed-length text padding so DP ranks see identical shapes; reserves `image_positions` for multi-image).
  - `kempnerforge/data/dataloader.py`: `StatefulDataLoader` accepts an optional `collate_fn` for VLM batches.
  - `kempnerforge/training/freeze.py`: `freeze_params`, `apply_freeze_specs`, `canonical_freeze_meta`, and `effective_freeze` (resolves the active spec list at a given step from `base + freeze_schedule`).
  - `kempnerforge/training/eval.py`: `should_build_eval_dataloader` gate that warns and skips eval for VLM configs (eval-for-VLM is a tracked follow-up).
  - `kempnerforge/distributed/parallel.py`: `_build_vlm` (component-wise build + parallelism), `_apply_fsdp_vlm` (per-component FSDP2 wrap; vision encoder stays replicated when fully frozen), `_fsdp_wrap_transformer_blocks` (EP-MoE-aware, shared by text and VLM paths), `default_mp_policy(cast_forward_inputs=True)`. `build_parallel_model` dispatches on `model_config.is_vlm`.
  - `kempnerforge/checkpoint/manager.py`: VLM freeze metadata save/load with cross-arch tolerance (intersection of module keys), `peek_saved_step()` for resume, `flush_pending_save()` for FreezeStage transitions, `ignore_freeze_mismatch` escape hatch.
  - `kempnerforge/config/checkpoint.py`: `CheckpointConfig.ignore_freeze_mismatch`.
  - `kempnerforge/config/data.py`: `hf_dataset_image_field`, `hf_dataset_prompt_field`, `hf_image_size`.
  - `kempnerforge/config/job.py`: VLM seq_len validation + explicit "VLM + PP not supported" guard.
  - `scripts/train.py`: VLM training step (no PP), freeze-schedule transition hook with async-save fence, VLM checkpoint metadata, per-step text-token counter (DP-reduced).
  - `scripts/prep_vlm_coco_smoke.py`: helper to materialize a small COCO Karpathy slice for smoke runs.
  - Tests: `tests/unit/test_vlm.py`, `test_vlm_config.py`, `test_modality_context.py`, `test_vision.py`, `test_vlm_dataset.py`, `test_freeze.py`; `tests/integration/test_vlm_train_step.py`, `test_vlm_checkpoint.py`; `tests/distributed/test_vlm_fsdp.py` (gated on multi-GPU).
  - Configs: `configs/train/vlm_debug.toml` (1-GPU smoke), `vlm_7b.toml` / `vlm_7b_ac.toml` / `vlm_7b_siglip2.toml`, `vlm_7b_freeze_schedule.toml`.

### Changed
- Dropped `pytest-cov` from dev dependencies and the `[tool.coverage]` configuration; CI no longer uploads to Codecov.

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

[0.1.0]: https://github.com/KempnerInstitute/KempnerForge/releases/tag/v0.1.0
