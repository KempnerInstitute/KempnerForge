# Changelog

All notable changes to KempnerForge are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Fine-grained MoE experts** (`moe_expert_ffn_multiplier`). Decouples each expert's FFN hidden width from the dense FFN: the per-expert hidden dim is `computed_ffn_hidden_dim × moe_expert_ffn_multiplier`, rounded to a multiple of 16. The default `1.0` is a no-op (each expert is a full dense FFN, zero behavior change); set `0.5` for fine-grained experts so top-2 routing matches the dense FFN's activated FLOPs (`2 × F/2 = F`) while adding total capacity — the DeepSeekMoE recipe. Applies to routed and shared experts wherever they are built (`build_moe` and MoMa's `ExpertChoiceMoE`).
  - `kempnerforge/config/model.py`: `moe_expert_ffn_multiplier: float = 1.0` (with a positivity check) and a `computed_expert_ffn_hidden_dim` property; `num_params_estimate` accounts for the smaller experts.
  - `kempnerforge/model/{moe,moma,mot,transformer}.py`: experts are built at `computed_expert_ffn_hidden_dim` instead of the dense FFN width.
  - Tests: `tests/unit/test_config.py` (default-equals-dense, half-size, iso-FLOP top-2, rejects non-positive, param-estimate drop) plus cases in `test_moe.py`, `test_moma.py`, `test_mot.py`.
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
  - `scripts/prep_vlm_coco.py`: prep helper for COCO-Karpathy — writes a bare HF `Dataset` (default, small slice for smoke runs) or a 3-split `DatasetDict` via `--all-splits` for training + eval.
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
- **Dynamic-checkpointing window** (`[checkpoint.dyn_ckpt_window]`). Opt-in dense save phase: inside `[start, stop]` a registered strategy decides which steps to save; outside the window the regular `interval` cadence applies. The default strategy, `"power2"`, saves at `start` and at every `start + 2^k` while `<= stop` — tight near the start of the window, doubling thereafter. Useful for analyzing early-training dynamics, where the loss moves fastest. The default `CheckpointConfig` is unchanged (no `dyn_ckpt_window`, interval-only saves).
  - `kempnerforge/config/checkpoint.py`: new `DynamicCheckpointWindow` dataclass (`start: int = 0`, `stop: int = 512`, `strategy: str = "power2"`), new `CheckpointConfig.dyn_ckpt_window: DynamicCheckpointWindow | None = None`, `should_save(step)`, and `is_dynamic_milestone(step)`. Ships with `"power2"` registered by default; new strategies plug in via the registry without touching `CheckpointConfig`.
  - `kempnerforge/config/registry.py`: `register_dyn_ckpt_strategy(name)` / `get_dyn_ckpt_strategy(name)` / `list_dyn_ckpt_strategies()` — a strategy is any `Callable[[DynamicCheckpointWindow, int], bool]` and registers via `@registry.register_dyn_ckpt_strategy("name")`.
  - Milestone-aware retention: `CheckpointManager._cleanup` never prunes a step where the configured dynamic strategy fired, so `keep_last_n` rotates only the later interval checkpoints. `keep_last_n <= 0` keeps everything (the previous `keep_last_n >= 1` requirement is relaxed).
  - `scripts/train.py`: the save gate now calls `config.checkpoint.should_save(step)`.
  - Tests: `tests/unit/test_config.py` (defaults, power2 firing, offset-based `start > 0`, validation, unknown-strategy rejection, `is_dynamic_milestone`), `tests/unit/test_checkpoint.py::TestCheckpointRetention::test_cleanup_protects_dynamic_milestones`.
- **VLM evaluation pipeline** (`scripts/vlm_eval_harness.py` + `kempnerforge/eval/vlm/`). Evaluates any KempnerForge VLM checkpoint on any standard multimodal benchmarks (MMMU, MMBench, ScienceQA, SEED, AI2D, …) by integrating the [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) harness through a custom model adapter that wraps `VLMWrapper` and loads directly from a DCP checkpoint. Arch-agnostic across Joint-Decoder / Cross-Attention / MoT; MoMa fails fast (its non-causal expert-choice routing cannot autoregressively generate, and eval requires generation). v1 is single-GPU, image-only, and generation-only (`generate_until`). All changes are additive and backward compatible; the only edit to existing code is a behavior-preserving refactor.
  - `kempnerforge/eval/__init__.py`, `kempnerforge/eval/vlm/__init__.py`: new eval-subsystem namespace. Import-isolated — neither is imported on the default `import kempnerforge` path, so the main package keeps working with lmms-eval absent (pinned by `test_import_isolation.py`).
  - `kempnerforge/eval/vlm/adapter.py`: `KempnerForgeVLM(lmms)` chat adapter (`is_simple = False`). Loader (`build_vlm_wrapper` behind a `_build_model` seam → single-process `dcp.load` of model shards only; `resolve_resume_path` with a specific-`step_N` fallback; reads plain-JSON `metadata.json`, never `train_state.pt`); prompt rendering by flattening `ChatMessages` text blocks (no chat template, no `<image>` placeholder — images are conditioned via `pixel_values`); a cache-less greedy/sampled decode loop reusing `kempnerforge.model.generate.sample`; `generate_until` only — `loglikelihood` and `generate_until_multi_round` raise `NotImplementedError`. Guards (clear `NotImplementedError`/`ValueError`): MoMa arch, video/audio, multi-image, multi-turn/few-shot. A file-level `# pyright: reportMissingImports=false` keeps `pyright kempnerforge/` green in CI (where the undeclared lmms-eval is absent) without an inline ignore.
  - `kempnerforge/eval/vlm/registry.py`: `MANIFEST = ModelManifest(model_id="kempnerforge_vlm", chat_class_path="kempnerforge.eval.vlm.adapter.KempnerForgeVLM")` for lmms-eval entry-point discovery.
  - `scripts/vlm_eval_harness.py`: CLI mirroring `scripts/eval_harness.py` (no conversion). `--config`/`--checkpoint`/`--tasks` required (default suite TBD), plus `--limit`/`--output`/`--device`/`--dtype`/`--batch-size`/`--max-new-tokens`; lazy `lmms_eval.evaluator.simple_evaluate` import with a helpful error.
  - `pyproject.toml`: `[project.entry-points."lmms_eval.models"]` for the adapter — metadata only; lmms-eval is NOT added as a dependency (install separately with `uv pip install lmms-eval`, mirroring how lm-eval is handled).
  - `kempnerforge/data/vlm_dataset.py`: behavior-preserving refactor — `_pil_to_tensor` → public `pil_to_tensor`; tokenizer construction and pad-id resolution extracted to public `build_tokenizer` / `resolve_pad_id`, so the eval adapter reuses the exact training-time preprocessing as the single source of truth. (`tests/unit/test_vlm_dataset.py` updated to the renamed helper; all behavior identical.)
  - Tests: `tests/unit/eval/vlm/` (CPU) runs against a faithful in-repo fake `lmms_eval` injected via `conftest.py` (`_fake_lmms_eval.py`) — so the adapter/registry tests **always run in CI and contribute coverage** without the undeclared lmms-eval dependency, instead of skipping. `test_adapter.py` covers rendering / preprocessing / `gen_kwargs` / the decode loop / guards, plus the loader helpers (`_resolve_dtype`, `_load_config`, `_load_weights`, `_log_checkpoint_metadata`, `_first_stop`), the `__init__` guards, and `generate_until` end-to-end; `test_registry.py` covers the manifest. `test_import_isolation.py` still asserts `import kempnerforge` needs no lmms-eval. `tests/integration/test_lmms_eval_contract.py` pins the *real* lmms-eval API and entry-point resolution to the fakes' assumptions (gated on real lmms-eval), and `tests/integration/test_vlm_eval.py` keeps the self-contained DCP round-trip + env-gated real-task path. `tests/conftest.py`: `tiny_vlm_configs` / `tiny_vlm_wrapper` fixtures (random vision encoder, CPU, no checkpoint).
  - Docs: `docs/how-to/run-vlm-evaluation.md`, wired into the how-to `toctree`.
  - Deferred: single image-encode per request (model-side change), data-parallel and sharded multi-GPU inference, a representative default benchmark suite, whether to formalize the lmms-eval dependency, and confirming frozen vision-encoder weights load from a real checkpoint.
- **VLM evaluation: batch size > 1.** `KempnerForgeVLM.generate_until` now decodes requests in batches (the `--batch-size` / `batch_size` model-arg) instead of one at a time. Requests are grouped by `gen_kwargs` and the text is **right-padded** to the batch-max length — the same layout training uses (image prefix at `0..n-1`, text contiguous from `n`, trailing pads causally masked) — so a batched forward gives each row the same real-position logits as decoding it alone (pinned by a batch-equivalence test). Each row's next token is read at its own last real position, and EOS / `until` / `max_new_tokens` are tracked per row. Adapter-only; **no model-code changes**. Multi-image / few-shot / multi-turn remain guarded (deferred until the team convenes — they need model-side changes).
  - `kempnerforge/eval/vlm/adapter.py`: `_generate_one` → batched `_generate_batch`; `generate_until` groups / chunks / reorders via `lmms_eval.utils.Collator` (mirrors `lmms_eval/models/chat/qwen2_5_vl.py`), reusing the now-public `resolve_pad_id` from `kempnerforge/data/vlm_dataset.py` for right-pad ids.
  - Tests: `tests/unit/eval/vlm/test_adapter.py` (`TestGenerateBatchSingle` B=1 regressions + `TestGenerateBatchMulti` batch-equivalence, per-row `max_new_tokens`, per-row EOS); `tests/integration/test_vlm_eval.py` runs `generate_until` at `batch_size=2`. Docs: `docs/how-to/run-vlm-evaluation.md` batch note.

### Changed
- `docs/getting-started/install.md` Prerequisites: documents `.python-version` and uv's auto-fetch behavior.
- `README.md` and `kempnerforge/README.md` Prerequisites: clarify that uv auto-fetches Python 3.12 via `.python-version`.
- `docs/claude-ready.md` first-run flow: `/kempnerforge:install-and-verify` runs before `/kempnerforge:cluster-config`.
- `README.md` and `kempnerforge/README.md`: list `install-and-verify` in the skill catalog and drop the hardcoded skill count.

### Fixed
- **Resume silently reset AdamW optimizer momentum.** `CheckpointManager` round-tripped optimizer state through raw `optimizer.state_dict()` / `optimizer.load_state_dict()`. On resume the optimizer is freshly built, so its `state_dict()` is empty — `dcp.load` then had no `exp_avg` / `exp_avg_sq` tensors to fill, and the moments were silently dropped, resetting Adam momentum to zero at every resume point. Model weights, scheduler, dataloader position, and RNG all restored correctly; only the optimizer moments were lost, so resumed runs were not bit-exact.
  - `kempnerforge/checkpoint/manager.py`: save and load now go through DCP's `get_model_state_dict` / `get_optimizer_state_dict` / `set_model_state_dict` / `set_optimizer_state_dict`. The getters build a load template with the optimizer moments allocated in the correct FSDP/DTensor layout, so `dcp.load` repopulates them; the setters write the loaded values back into the live optimizer.
  - `docs/checkpointing/dcp-model.md`: updated the save/load snippets and the "shape to fill" explanation to the DCP-aware helpers.
  - Tests (fail on the pre-fix code, pass after): `tests/integration/test_checkpoint_roundtrip.py::test_manager_restores_optimizer_moments_single_gpu` and `tests/distributed/test_checkpoint.py::test_resume_restores_optimizer_moments` assert `exp_avg` / `exp_avg_sq` are restored bit-exactly into a *fresh* optimizer (single-GPU + distributed); `tests/e2e/test_training_e2e.py::test_resume_determinism_single_gpu` / `test_resume_determinism_2gpu_fsdp` assert end-to-end bit-exact loss across an interrupt-and-resume on a learnable dataset.
  - **On-disk format note:** optimizer state is now keyed by parameter fully-qualified name rather than positional index. Checkpoints written before this fix will not restore optimizer state on resume (training continues with a fresh optimizer); model state is unaffected.

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
