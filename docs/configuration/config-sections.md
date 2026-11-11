# Config sections

`JobConfig` aggregates ten typed sub-configs. Each one lives in its own
module under
[`kempnerforge/config/`](https://github.com/KempnerInstitute/KempnerForge/tree/main/kempnerforge/config)
and declares its fields with dataclass defaults. TOML sections map
one-to-one onto these dataclass attributes:

```toml
[model]          # → config.model   (ModelConfig)
[train]          # → config.train   (TrainConfig)
[optimizer]      # → config.optimizer
[scheduler]      # → config.scheduler
[data]           # → config.data
[eval]           # → config.eval
[distributed]    # → config.distributed
[checkpoint]     # → config.checkpoint
[metrics]        # → config.metrics
[profiling]      # → config.profiling
```

## JobConfig

Owns the ten sub-configs and the cross-section `validate` method.
[`kempnerforge/config/job.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/job.py).

| Field | Type | Purpose |
|-------|------|---------|
| `model` | `ModelConfig` | architecture + MoE knobs |
| `train` | `TrainConfig` | loop-level hyperparameters |
| `optimizer` | `OptimizerConfig` | registry key + LR + betas + optimizer-specific knobs |
| `scheduler` | `SchedulerConfig` | LR schedule shape and warmup |
| `data` | `DataConfig` | dataset sources, mixing, annealing |
| `eval` | `EvalConfig` | in-loop eval cadence and data source |
| `distributed` | `DistributedConfig` | parallelism dims, NCCL timeout |
| `checkpoint` | `CheckpointConfig` | DCP save cadence, retention, resume path |
| `metrics` | `MetricsConfig` | logging cadence and backends |
| `profiling` | `ProfilingConfig` | torch.profiler window and trace dir |
| `plugins` | `list[str]` | module paths imported before the config is built, for their registry side effects — see [Registry](registry.md) |

`plugins` is a top-level key, not a section: `plugins = ["pkg.module"]`.

## `[model]` — `ModelConfig`

Architecture hyperparameters and MoE knobs.
[`kempnerforge/config/model.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/model.py).

### Dense

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `dim` | `int` | `4096` | hidden size |
| `n_layers` | `int` | `32` | number of transformer blocks |
| `n_heads` | `int` | `32` | attention heads |
| `n_kv_heads` | `int \| None` | `None` | GQA: `None` → MHA (= `n_heads`), `1` → MQA, else GQA |
| `head_dim_override` | `int` | `0` | `0` → `head_dim` is `dim // n_heads`; a positive even value decouples the attention width, so `n_heads * head_dim` need not equal `dim` |
| `vocab_size` | `int` | `32000` | embedding table size |
| `ffn_dim_multiplier` | `float` | `1.0` | scales Llama-style `4·dim·(2/3)` hidden width |
| `ffn_hidden_dim` | `int \| None` | `None` | hard-override the computed FFN width |
| `norm_type` | `"rmsnorm" \| "layernorm"` | `"rmsnorm"` | registry key for norm builder |
| `norm_eps` | `float` | `1e-5` | norm epsilon |
| `activation` | `"silu" \| "gelu" \| "relu"` | `"silu"` | MLP activation (`silu` → SwiGLU) |
| `max_seq_len` | `int` | `2048` | RoPE table length; `train.seq_len` must be ≤ this |
| `rope_theta` | `float` | `10000.0` | RoPE frequency base |
| `tie_embeddings` | `bool` | `False` | share embedding and output-head weight |
| `qk_norm` | `bool` | `False` | RMSNorm over Q/K per head before RoPE |
| `init_std` | `float` | `0.02` | weight-init std (GPT-2 / Llama convention) |
| `model_type` | `str` | `"transformer"` | `model` registry key |
| `sdpa_backend` | `str` | `"auto"` | one of `"auto"`, `"flash"`, `"efficient"`, `"cudnn"`, `"math"` |
| `attention_backend` | `str` | `"sdpa"` | `"sdpa"` or `"flex"`; `"flex"` routes packed batches (`data.pack_sequences`) through FlexAttention's block-sparse mask. Requires `train.seq_len ≥ 128` and `head_dim ≥ 16` |

### MoE (all defaults produce a dense model)

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `num_experts` | `int` | `0` | `0` → dense; `>0` → MoE |
| `moe_top_k` | `int` | `2` | experts selected per token |
| `moe_frequency` | `int` | `1` | MoE every N layers (1=all, 2=alternating) |
| `moe_router` | `str` | `"softmax_topk"` | `router` registry key |
| `moe_shared_experts` | `int` | `0` | shared experts that always process every token |
| `moe_aux_loss_weight` | `float` | `0.01` | coefficient in training loss |
| `moe_capacity_factor` | `float` | `0.0` | `0` → no drop; `>0` → cap tokens/expert (typical `1.25`) |
| `moe_sequence_aux_loss_weight` | `float` | `0.0` | sequence-level balance loss (0 = off) |
| `moe_gradient_scale` | `bool` | `False` | per-expert gradient normalization |
| `moe_bias_schedule` | `str` | `"constant"` | `"constant"`, `"cosine_decay"`, `"linear_warmup"` |
| `moe_packed_experts` | `bool` | `False` | pack expert weights into one tensor per projection |
| `moe_expert_ffn_multiplier` | `float` | `1.0` | expert FFN hidden width relative to dense (`0.5` → fine-grained) |
| `moe_router_z_loss_weight` | `float` | `0.0` | router-logit z-loss coefficient (`0` = off) |

Computed properties: `is_moe`, `head_dim`, `computed_ffn_hidden_dim`,
`computed_expert_ffn_hidden_dim`, `num_params_estimate`.

## `[train]` — `TrainConfig`

Training-loop hyperparameters.
[`kempnerforge/config/training.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/training.py).

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `batch_size` | `int` | `8` | per-device micro-batch size |
| `seq_len` | `int` | `2048` | tokens per sequence |
| `max_steps` | `int` | `100000` | training-loop termination |
| `grad_accum_steps` | `int` | `1` | microbatches per optimizer step |
| `grad_clip_norm` | `float` | `1.0` | `clip_grad_norm_` cap |
| `seed` | `int` | `42` | torch/numpy/python RNG seed |
| `data_seed` | `int \| None` | `None` | RNG seed for data and batch order; `None` → falls back to `seed` |
| `compile_model` | `bool` | `True` | wrap the model with `torch.compile` |
| `mixed_precision` | `"bf16" \| "fp32" \| "fp8"` | `"bf16"` | compute dtype, and the master-weight dtype unless `fp32_master_weights` is set; `"fp8"` computes in fp8, with bf16 masters unless `fp32_master_weights` is set; `"fp16"` is rejected (no loss scaling) |
| `fp32_master_weights` | `bool` | `False` | keep parameters, gradients and optimizer state in fp32 and compute in `param_dtype` through FSDP2's mixed precision; without an FSDP2 wrap (1 GPU, a pp-only mesh) the forward runs under `torch.autocast`; rejected with `mixed_precision = "fp32"`, and for MoE on a single GPU — see [FSDP2 § fp32 master weights](../distributed/fsdp2.md#fp32-master-weights) |
| `tf32` | `bool` | `False` | run fp32 matmuls (a frozen fp32 vision encoder, `mixed_precision = "fp32"`) on TF32 tensor cores via `torch.set_float32_matmul_precision("high")`, applied at startup by training and `scripts/eval.py`; this also covers the matmuls that run in fp32 on purpose in a bf16 run, so Muon's Newton–Schulz step moves to TF32 (still more precise than the bf16 that `torch.optim.Muon` uses); `False` leaves torch's setting as it is |
| `activation_checkpointing` | `"none" \| "full" \| "selective"` | `"none"` | AC policy |
| `loss_fn` | `str` | `"cross_entropy"` | `loss` registry key (or `"chunked_cross_entropy"`) |
| `z_loss_weight` | `float` | `0.0` | logit-magnitude regularizer (PaLM uses `1e-4`) |
| `ce_chunk_size` | `int` | `0` | chunk size for `chunked_cross_entropy` (`0` → auto 4096) |
| `fp32_cross_entropy` | `bool` | `False` | compute `cross_entropy` / `chunked_cross_entropy` and their gradient in fp32 from bf16 logits, without an fp32 copy of the logits; rejected for any other `loss_fn` — see [Losses § fp32 cross-entropy](../training/losses.md#fp32-cross-entropy) |
| `shutdown_timeout_sec` | `float` | `600.0` | graceful shutdown timeout before forced exit |
| `nccl_health_check_interval` | `int` | `0` | NCCL liveness all-reduce every N steps (`0` = disabled) |

Computed properties: `param_dtype` (compute dtype), `master_dtype` (storage dtype: fp32 with
`fp32_master_weights`, otherwise `param_dtype`), `is_fp8`.

## `[optimizer]` — `OptimizerConfig`

Optimizer settings. `name` picks the registry builder; the other fields
are shared (AdamW/Lion) or optimizer-specific.
[`kempnerforge/config/optimizer.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/optimizer.py).

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `name` | `str` | `"adamw"` | one of `adamw`, `lion`, `muon`, `schedule_free_adamw` |
| `lr` | `float` | `3e-4` | peak learning rate |
| `weight_decay` | `float` | `0.1` | decoupled weight decay, applied to the group `weight_decay_scope` selects |
| `betas` | `tuple[float, float]` | `(0.9, 0.95)` | AdamW / Lion momenta |
| `eps` | `float` | `1e-8` | numerical safety (AdamW) |
| `fused` | `bool` | `True` | use fused AdamW when available |
| `muon_momentum` | `float` | `0.95` | Muon momentum coefficient |
| `muon_ns_steps` | `int` | `5` | Newton–Schulz iterations for Muon |
| `muon_adam_lr` | `float \| None` | `None` | LR for the parameters Muon leaves to its AdamW fallback; `None` → same as `lr` |
| `muon_nesterov` | `bool` | `False` | Orthogonalize the Nesterov look-ahead instead of the momentum buffer |
| `muon_adjust_lr` | `"original" \| "match_rms_adamw" \| "spectral_unclamped"` | `"original"` | Factor on each orthogonalized update |
| `muon_orthogonalize` | `"local" \| "exact"` | `"local"` | Newton–Schulz on each rank's shard, or on the full matrix |
| `muon_router` | `bool` | `False` | Orthogonalize MoE router gates with Muon instead of its AdamW fallback |
| `schedule_free_warmup_steps` | `int` | `0` | internal warmup for schedule-free |
| `weight_decay_scope` | `"exclude_norms_and_biases" \| "all"` | `"exclude_norms_and_biases"` | which parameters are decayed; `"all"` decays every parameter, as `torch.optim.AdamW` does by default. A resume that restores optimizer state saved under the other scope is refused |
| `lr_multipliers` | `dict[str, float]` | `{}` | per-module learning rates: each `vlm.module_patterns` module listed trains at `lr` times its multiplier, in its own param groups (see [Optimizers § Per-module learning rates](../training/optimizers.md#per-module-learning-rates)). A resume that restores optimizer or scheduler state saved with other multipliers, or other patterns for their modules, is refused |

## `[scheduler]` — `SchedulerConfig`

LR schedule shape and warmup.
[`kempnerforge/config/scheduler.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/scheduler.py).

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `name` | `"cosine" \| "linear" \| "wsd" \| "constant" \| "rex" \| "none"` | `"cosine"` | `scheduler` registry key |
| `warmup_steps` | `int` | `2000` | linear warmup length |
| `warmup_start_factor` | `float` | `0.0` | LR factor the warmup starts from (torch `LinearLR` `start_factor`) |
| `decay_steps` | `int \| None` | `None` | `None` → decay over remaining steps |
| `min_lr_ratio` | `float` | `0.1` | floor = `lr * min_lr_ratio` |
| `stable_steps` | `int \| None` | `None` | WSD: steps at constant LR between warmup and decay |
| `wsd_decay_type` | `"cosine" \| "linear" \| "sqrt"` | `"cosine"` | WSD cooldown shape |
| `rex_alpha` | `float` | `1.0` | REX exponent: `(1 - t/T)^alpha` |

## `[data]` — `DataConfig`

Single dataset, HuggingFace source, or mixture; optional phase schedule.
[`kempnerforge/config/data.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/data.py).

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `dataset_path` | `str` | `""` | directory of pre-tokenized shards |
| `file_pattern` | `str` | `"*.npy"` | glob inside `dataset_path`; each match is read by its own suffix, `.bin` or `.npy` |
| `token_dtype` | `str` | `""` | dtype of raw `.bin` shards, `"uint16"` or `"uint32"`; `""` takes `tokenized_info.dtype` from the `metadata.yaml` beside them (see [Memory-mapped dataset](../data/memory-mapped.md)) |
| `tokenizer_path` | `str` | `""` | path or HF id for the tokenizer |
| `num_workers` | `int` | `4` | DataLoader workers |
| `pin_memory` | `bool` | `True` | DataLoader pin memory |
| `prefetch_factor` | `int` | `2` | DataLoader prefetch factor |
| `hf_dataset_name` | `str \| None` | `None` | HF dataset id (e.g. `"wikitext"`) |
| `hf_dataset_config` | `str \| None` | `None` | HF dataset config (e.g. `"wikitext-2-raw-v1"`) |
| `hf_dataset_split` | `str` | `"train"` | HF split |
| `hf_dataset_text_field` | `str` | `"text"` | field to tokenize |
| `hf_dataset_image_field` | `str` | `"image"` | VLM only: image column |
| `hf_dataset_prompt_field` | `str` | `""` | VLM only: prompt column (`""` = no prompt mask) |
| `hf_image_size` | `int` | `224` | VLM only: target square image size |
| `hf_streaming` | `bool` | `False` | use `IterableDataset` for large corpora |
| `pack_sequences` | `bool` | `False` | document-aware packing with cross-doc isolation (feeds `doc_ids` to attention); on a Joint-Decoder VLM run, packs `train.batch_size` image-text documents per micro-batch into rows of `train.seq_len`, each document's image tokens before its text, positions restarting per document |
| `datasets` | `list[DatasetSource]` | `[]` | multi-dataset mixture (overrides `dataset_path`/`hf_dataset_name` when non-empty) |
| `mix_temperature` | `float` | `1.0` | weight scaling; `1.0` → as-is, larger → more uniform |
| `phases` | `list[TrainingPhase]` | `[]` | multi-phase schedule with weight/LR transitions |
| `anneal_start_step` | `int` | `0` | syntactic sugar for a common 2-phase annealing pattern (`0` = disabled) |
| `anneal_weights` | `dict[str, float]` | `{}` | per-dataset weights applied at `anneal_start_step` |

### `DatasetSource`

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `path` | `str` | `""` | pre-tokenized directory |
| `weight` | `float` | `1.0` | relative sampling weight (must be `> 0`) |
| `name` | `str` | `""` | name for per-dataset metrics (auto-derived if empty) |
| `hf_name` | `str` | `""` | HF dataset id |
| `hf_config` | `str` | `""` | HF dataset config |
| `token_dtype` | `str` | `""` | dtype of this source's `.bin` shards; `""` uses `data.token_dtype` |

Either `path` or `hf_name` must be set per source.

### `TrainingPhase`

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `start_step` | `int` | `0` | step at which the phase activates |
| `dataset_weights` | `dict[str, float]` | `{}` | per-dataset weights for this phase |
| `lr_scale` | `float` | `1.0` | multiplier applied to scheduler LR |

`phases[*].start_step` must be strictly increasing; `phases` and
`anneal_start_step` are mutually exclusive.

## `[eval]` — `EvalConfig`

In-loop evaluation. Disabled by default.
[`kempnerforge/config/eval.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/eval.py).

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `enabled` | `bool` | `False` | gate in-loop eval |
| `interval` | `int` | `1000` | eval every N training steps |
| `steps` | `int` | `50` | eval batches per evaluation; under pipeline parallelism a multiple of `train.grad_accum_steps` |
| `dataset_path` | `str` | `""` | pre-tokenized eval shards |
| `file_pattern` | `str` | `"*.npy"` | glob inside `dataset_path` |
| `token_dtype` | `str` | `""` | dtype of raw `.bin` eval shards, `"uint16"` or `"uint32"`; `""` takes it from the `metadata.yaml` beside them, never from `data.token_dtype` |
| `hf_dataset_name` | `str \| None` | `None` | HF dataset id |
| `hf_dataset_config` | `str \| None` | `None` | HF dataset config |
| `hf_dataset_split` | `str` | `"validation"` | HF split |

If `enabled=True`, at least one of `dataset_path` / `hf_dataset_name`
must be set; `validate()` rejects the combination otherwise.

## `[distributed]` — `DistributedConfig`

Parallelism dimensions and NCCL settings.
[`kempnerforge/config/distributed.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/distributed.py).

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `dp_shard` | `int` | `-1` | FSDP shard degree; `-1` → auto (use remaining GPUs) |
| `dp_replicate` | `int` | `1` | replicates the `dp_shard` groups (HSDP), and every rank in `dp_replicate × dp_shard` reads its own data; with no `dp_shard` dimension (it resolves to 1, no TP or EP), FSDP2 shards over `dp_replicate` instead |
| `tp` | `int` | `1` | tensor parallel |
| `pp` | `int` | `1` | pipeline parallel |
| `pp_schedule` | `"1f1b" \| "gpipe" \| "interleaved_1f1b"` | `"1f1b"` | pipeline schedule |
| `cp` | `int` | `1` | context parallel; reserved, not implemented, must be `1` (`cp > 1` is rejected at load) |
| `ep` | `int` | `1` | expert parallel (MoE only) |
| `nccl_timeout_sec` | `int` | `1800` | NCCL collective timeout |
| `backend` | `str` | `"cpu:gloo,cuda:nccl"` | `torch.distributed` backend mapping |

The product `dp_replicate × dp_shard × tp × pp × cp × ep` must equal
`world_size`. Methods: `validate_world_size(ws)`, `resolve(ws)`.

## `[checkpoint]` — `CheckpointConfig`

DCP-based checkpointing.
[`kempnerforge/config/checkpoint.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/checkpoint.py).

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `dir` | `str` | `"checkpoints"` | root directory for `step_N/` + `latest` symlink |
| `interval` | `int` | `1000` | save every N steps (outside any `dyn_ckpt_window`) |
| `dyn_ckpt_window` | `DynamicCheckpointWindow \| None` | `None` | optional dense save phase — see [`[checkpoint.dyn_ckpt_window]`](#checkpointdyn_ckpt_window--dynamiccheckpointwindow) |
| `async_mode` | `"disabled" \| "async" \| "async_with_pinned_mem"` | `"disabled"` | DCP async-save mode |
| `keep_last_n` | `int` | `3` | retain the most recent N checkpoints (`<= 0` keeps all); steps saved by `dyn_ckpt_window` are always kept |
| `load_path` | `str \| None` | `None` | checkpoint for training to start from while `dir` holds none (a warm start); once `dir` has a checkpoint, auto-resume loads that instead and rank 0 warns that `load_path` is not used, unless it names that checkpoint. `scripts/eval.py` and `scripts/generate.py` load `load_path` first |
| `export_dtype` | `"float32" \| "bfloat16"` | `"bfloat16"` | dtype for HF exports via `scripts/convert_checkpoint.py dcp-to-hf` when its `--dtype` flag is not given; the flag wins and also takes `float16` (see [HuggingFace conversion](../checkpointing/hf-conversion.md#dtypes)) |
| `exclude_from_loading` | `list[str]` | `[]` | state to skip when warm-starting from `load_path`: `"model"` and/or `"optimizer"`, or exactly `["optimizer", "train_state"]` for a weights-only warm start that loads the model alone and skips `train_state.pt` (step, scheduler, dataloader position, RNG) and the saved freeze comparison, so a new stage starts fresh from the weights; ignored on a resume, which always restores full state |
| `ignore_freeze_mismatch` | `bool` | `False` | load a checkpoint whose VLM freeze metadata disagrees with the configured freeze specs, downgrading the error to a warning |
| `ignore_rank_state_mismatch` | `bool` | `False` | resume when the saved per-rank dataloader/RNG state does not fit this run (changed world size or data-parallel layout, unreadable or incomplete envelope): discard that state on every rank with a warning instead of raising, so data order and random streams restart fresh while step, model, optimizer and scheduler continue (see [Resharding](../checkpointing/resharding.md#scaling-a-run-to-a-different-world-size)) |

### `[checkpoint.dyn_ckpt_window]` — `DynamicCheckpointWindow`

Opt-in dense save phase. Inside `[start, stop]` a registered strategy decides
which steps to save; outside the window the regular `interval` cadence applies.
Steps saved by the strategy are exempt from `keep_last_n` retention, so a
finite `keep_last_n` rotates only the later interval checkpoints — the dense
window checkpoints are never pruned. Useful for inspecting early-training
dynamics at fine granularity.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `start` | `int` | `0` | first step of the dense window; `0` captures the initial weights before any training step |
| `stop` | `int` | `512` | last step of the dense window |
| `strategy` | `str` | `"power2"` | name of a registered strategy (see below) |

Example TOML:

```toml
# Off by default — interval cadence only:
[checkpoint]
interval = 1000
keep_last_n = 3

# Opt in with defaults (start=0, stop=512, strategy="power2"):
[checkpoint.dyn_ckpt_window]

# Customize:
[checkpoint.dyn_ckpt_window]
start = 0
stop  = 1024
strategy = "power2"
```

#### Dyn_ckpt strategies

A strategy is a `Callable[[DynamicCheckpointWindow, int], bool]` registered via
`@registry.register_dyn_ckpt_strategy("name")` (see
[`kempnerforge/config/registry.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/registry.py)).
The strategy receives the window and the current training step and returns
`True` iff the step should be saved. The registry must be populated before
config load — either by `kempnerforge` itself (the default `"power2"` is
registered in `kempnerforge/config/checkpoint.py`) or by an explicit `import`
of the user's module.

| Name | Behavior |
|------|----------|
| `"power2"` (default) | save at `start` and at every `start + 2^k` while `<= stop` (offset-based powers of two; tight near `start`, doubling thereafter) |

## `[metrics]` — `MetricsConfig`

Logging cadence and backend toggles. The backends are independent, so any combination can be
enabled in one run config.
[`kempnerforge/config/metrics.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/metrics.py).

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `log_interval` | `int` | `10` | log every N steps (stdout + enabled backends) |
| `enable_wandb` | `bool` | `False` | turn on WandB backend |
| `enable_tensorboard` | `bool` | `False` | turn on TensorBoard backend |
| `wandb_project` | `str` | `"kempnerforge"` | WandB project name |
| `wandb_run_name` | `str \| None` | `None` | `None` → auto-generated |
| `wandb_run_id` | `str` | `""` | restored from checkpoint on resume; empty = new run |
| `tensorboard_dir` | `str` | `"tb_logs"` | TB log directory |
| `enable_mlflow` | `bool` | `False` | turn on MLflow backend (Databricks or local) |
| `mlflow_tracking_uri` | `str \| None` | `None` | tracking URI; `None` → `$MLFLOW_TRACKING_URI`, else `"databricks"`; `databricks://<profile>` selects a `~/.databrickscfg` profile |
| `mlflow_experiment` | `str \| None` | `None` | absolute workspace path on Databricks; `None` → `$MLFLOW_EXPERIMENT`/auto |
| `mlflow_run_name` | `str \| None` | `None` | `None` → auto-generated |
| `mlflow_description` | `str \| None` | `None` | run description (Markdown, the `mlflow.note.content` tag), set on a new run |
| `mlflow_tags` | `dict[str, str]` | `{}` | extra run tags, set on a new run after the automatic ones; keys and values must be non-empty strings |
| `mlflow_run_id` | `str` | `""` | restored from checkpoint on resume; empty = new run |
| `mlflow_log_system_metrics` | `bool` | `True` | sample CPU/GPU/memory on a background thread |
| `mlflow_system_metrics_interval` | `float` | `10.0` | seconds between system-metric samples |
| `mlflow_register_checkpoints` | `bool` | `False` | register each committed checkpoint as an MLflow external model on the background worker; `metadata.json` records its `model_name` at save and its `model_id` once registered; requires `enable_mlflow` |

Credentials never go in config. The auth routes, the URI and experiment resolution, and the
bounds on tracking calls are on the [MLflow backend](../metrics-and-profiling/mlflow.md) page.

## `[profiling]` — `ProfilingConfig`

`torch.profiler` window.
[`kempnerforge/config/profiling.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/profiling.py).

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `enable` | `bool` | `False` | run the profiler during the loop |
| `start_step` | `int` | `5` | first step recorded |
| `end_step` | `int` | `8` | last step recorded (must be `>` `start_step`) |
| `trace_dir` | `str` | `"profiler_traces"` | output directory for Chrome/Perfetto traces |

## `[adapter]`: `AdapterConfig`

The VLM connector that maps vision-encoder features into the transformer's
embedding space. A run that sets `[vlm]` without `[adapter]` gets these
defaults.
[`kempnerforge/config/adapter.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/adapter.py).

| Field | Type | Default | Used by | Purpose |
|-------|------|---------|---------|---------|
| `type` | `str` | `"mlp_2layer"` | all | adapter registry key: `mlp_2layer` and `linear` keep the token count, `avgpool` and `attentional_pool` pool it |
| `hidden_dim` | `int` | `0` | `mlp_2layer`, `avgpool` | MLP hidden width. `mlp_2layer`: `0` means `out_dim`. `avgpool`: a positive width makes the head after the pool a 2-layer MLP, normed first when `pre_norm` is set; `0` keeps the single `Linear` |
| `activation` | `str` | `"gelu"` | `mlp_2layer`; `avgpool` with `hidden_dim > 0` | `"gelu"`, `"silu"` or `"relu"` between the two MLP projections |
| `pre_norm` | `str` | `""` | `mlp_2layer`; `avgpool` with `hidden_dim > 0` | norm-registry key for a norm ahead of the MLP, over the vision features (`mlp_2layer`) or the pooled features (`avgpool`); `""` builds none |
| `pool_window` | `int` | `2` | `avgpool`, `attentional_pool` | pooling window side, e.g. `2` for 2×2 |
| `pool_heads` | `int` | `16` | `attentional_pool` | attention heads; must divide the vision feature dim |

A field outside the selected adapter's "Used by" entries must keep its
default, or the config fails to load with an error naming the field and the
adapter type. The column follows each adapter's builder parameters, which
`avgpool` narrows by `hidden_dim` with the `accepts` it registers; a plugin
adapter's come from its own builder (see
[Registry](registry.md#adapters-declare-the-fields-they-use)).

An `avgpool` connector with a 2×2 window and an RMSNorm + MLP head as wide as
its input (here a 1152-wide encoder):

```toml
[adapter]
type = "avgpool"
pool_window = 2
hidden_dim = 1152
pre_norm = "rmsnorm"
```

`hidden_dim` picks the `avgpool` head, and the head fixes the adapter's
checkpoint keys. `hidden_dim = 0` builds the single `Linear`, saved as
`adapter.proj.weight` and `adapter.proj.bias`. A positive `hidden_dim` builds
the MLP head, saved as `adapter.proj.proj1.*` and `adapter.proj.proj2.*`, plus
`adapter.proj.ln_q.*` when `pre_norm` is set. A checkpoint loads only into
the head it was saved with: the other head fails the load with `Missing key
in checkpoint state_dict` naming one of its parameters. To resume a
checkpoint saved with the single `Linear`, set `hidden_dim = 0` and leave
`activation` and `pre_norm` at their defaults.

A command-line override sets a value and never restores a default, so
switching `adapter.type` on the command line keeps every other field the TOML
sets. Reset the ones the new type does not use in the same command; an empty
value sets the empty string. For a TOML with `type = "mlp_2layer"`,
`hidden_dim = 2048` and `pre_norm = "rmsnorm"`:

```bash
--adapter.type=attentional_pool --adapter.hidden_dim=0 --adapter.pre_norm=
```

The TOML is checked on its own before the overrides apply, so its fields must
fit its own `type`. A TOML with `type = "attentional_pool"` and
`hidden_dim = 1152` fails to load even with `--adapter.type=avgpool`: set the
type and its fields in the same layer, both in the TOML or both on the
command line.

## Where to read next

- [CLI overrides](cli-overrides.md) — reshape any of these fields from
  the command line.
- [Validation rules](validation-rules.md) — what `__post_init__` and
  `validate(world_size)` enforce.
- [Registry](registry.md) — how the string keys above
  (`moe_router`, `norm_type`, `optimizer.name`, `scheduler.name`,
  `loss_fn`, `model_type`) resolve to builders.
