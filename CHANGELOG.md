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
- **Cross-Attention (CA) VLM arch** (`arch = "cross_attention"`, Llama-3-V style). The residual stream stays text-only; image features flow as K/V into separate `CrossAttentionBlock`s inserted at a configurable cadence. CA blocks are zero-initialized so adding the arch on top of a text-only checkpoint is identity at step 0 and learns from there. Composes with MoE in the text `TransformerBlock`s.
  - `kempnerforge/model/cross_attention.py`: `CrossAttention` (text Q × image K/V with optional GQA, no causal mask on the image axis, no RoPE) + `CrossAttentionBlock` (pre-norm wrapper with zero-init `o_proj` and MLP `down_proj`).
  - `kempnerforge/config/vlm.py`: `CrossAttentionConfig(VLMConfig)` registered via `@registry.register_vlm_config("cross_attention")` with `cross_attention_every_n_layers` / `cross_attention_n_heads` / `cross_attention_n_kv_heads` and a `resolved_heads()` helper.
  - `kempnerforge/model/vlm.py`: `CrossAttentionStrategy` registered via `@register_modality_strategy("cross_attention")` fills `image_features` + `image_mask=None`; `num_image_tokens()` returns 0 since the residual stream is text-only.
  - `kempnerforge/model/transformer.py`: when `arch="cross_attention"`, builds `CrossAttentionBlock`s into `transformer.cross_attention_layers` and fires one after text block index `i` iff `(i+1) % cadence == 0`. JD/text-only path is bit-equal to before (empty `cross_attention_layers` ModuleDict).
  - `kempnerforge/model/modality.py`: adds `ModalityContext.image_features` + `image_mask` fields with intra-context invariants (mutually exclusive with other residual routes; `image_mask` requires `image_features`).
  - `kempnerforge/model/init.py`: zero-init for `cross_attention_layers.*.{o_proj,down_proj}.weight` so CA blocks start as identity.
  - `kempnerforge/distributed/parallel.py`: `_fsdp_wrap_transformer_blocks` also wraps `transformer.cross_attention_layers` once per CA block (no-op for non-CA configs).
  - Tests: `tests/unit/test_cross_attention.py`; CA cases in `tests/unit/test_model.py::TestCrossAttentionInterleaving`, `test_vlm.py`, `test_vlm_config.py`, `test_modality_context.py`; `tests/integration/test_vlm_cross_attn.py`; `tests/distributed/test_vlm_cross_attn_fsdp.py` (gated on multi-GPU).
  - Configs: `configs/train/vlm_7b_cross_attn.toml`, `vlm_7b_siglip2_cross_attn.toml`, `vlm_debug_moe.toml` (CA + MoE smoke).
- **Mixture-of-Transformers (MoT) VLM arch** (`arch = "mot"`, Liang et al. 2024 Algorithm 1). Per-modality Q/K/V/O projections plus per-modality FFN at every layer; a single global self-attention mixes all modality streams. Image tokens prepend the text sequence (image-then-text residual layout, same as Joint-Decoder); per-modality residual projections are zero-initialized so a fresh MoT block is identity at construction.
  - `kempnerforge/model/mot.py`: `MoTAttention`, `MoTBlock`, `mot_warm_start_from_text_stack` (translates a JD / text-only state dict into per-modality copies; supports plain Tensors and FSDP2 DTensor targets via `distribute_tensor`).
  - `kempnerforge/config/vlm.py`: `MoTConfig(VLMConfig)` registered via `@registry.register_vlm_config("mot")` with `mot_modalities` / `mot_image_n_heads` / `mot_image_n_kv_heads` / `mot_warm_start_from_text` / `mot_warm_start_path` fields and a `resolved_image_heads()` helper. With both CA and MoT registered, `_RESERVED_ARCHS` is now empty.
  - `kempnerforge/model/vlm.py`: `MoTStrategy` registered via `@register_modality_strategy("mot")` fills `prefix_embeds` + `output_slice` + `modality_ids`.
  - `kempnerforge/model/transformer.py`: when `arch="mot"`, builds `MoTBlock`s into `transformer.layers`, builds per-modality `mot_norms` final-norm dict, and runs a per-modality forward branch (position-based image-then-text split, single global SDPA per layer, per-modality final norm, re-concat). JD/text-only path stays bit-equal.
  - `kempnerforge/model/init.py`: zero-init pass for MoT per-modality residual projections (FQNs do not match `endswith("o_proj.weight")` — handled by a second pass over modules).
  - `kempnerforge/model/modality.py`: adds `ModalityContext.modality_ids` field with intra-context invariant (requires `prefix_embeds` or `inputs_embeds`) and cross-arg checks in `Transformer.forward` (dtype must be `torch.long`; mutually exclusive with `kv_caches`).
  - `scripts/train.py`: training-loop hook that runs `mot_warm_start_from_text_stack` once at step 0 when `mot_warm_start_from_text=True`, between `ckpt_mgr.load(...)` and `apply_freeze_specs(...)`.
  - Tests: `tests/unit/test_mot.py` (Algorithm-1 reference parity, warm-start helper round-trips); `tests/unit/test_model.py::TestMoT` + `TestModalityIdsCrossArgs`; MoT cases in `tests/unit/test_vlm.py`, `test_vlm_config.py`, `test_modality_context.py`; `tests/integration/test_vlm_mot.py`; `tests/distributed/test_vlm_mot_fsdp.py` (gated on multi-GPU).
  - Configs: `configs/train/vlm_debug_mot.toml` (1-GPU smoke) and `configs/train/vlm_7b_mot.toml` (4-GPU 7B).
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
