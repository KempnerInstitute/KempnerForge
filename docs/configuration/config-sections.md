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

Computed properties: `is_moe`, `head_dim`, `computed_ffn_hidden_dim`,
`num_params_estimate`.

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
| `compile_model` | `bool` | `True` | wrap the model with `torch.compile` |
| `mixed_precision` | `"bf16" \| "fp16" \| "fp32" \| "fp8"` | `"bf16"` | master-weight dtype; `"fp8"` uses bf16 masters with fp8 compute |
| `activation_checkpointing` | `"none" \| "full" \| "selective"` | `"none"` | AC policy |
| `loss_fn` | `str` | `"cross_entropy"` | `loss` registry key (or `"chunked_cross_entropy"`) |
| `z_loss_weight` | `float` | `0.0` | logit-magnitude regularizer (PaLM uses `1e-4`) |
| `ce_chunk_size` | `int` | `0` | chunk size for `chunked_cross_entropy` (`0` → auto 4096) |
| `shutdown_timeout_sec` | `float` | `600.0` | graceful shutdown timeout before forced exit |
| `nccl_health_check_interval` | `int` | `0` | NCCL liveness all-reduce every N steps (`0` = disabled) |

Computed properties: `param_dtype`, `is_fp8`.

## `[optimizer]` — `OptimizerConfig`

Optimizer settings. `name` picks the registry builder; the other fields
are shared (AdamW/Lion) or optimizer-specific.
[`kempnerforge/config/optimizer.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/optimizer.py).

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `name` | `str` | `"adamw"` | one of `adamw`, `lion`, `muon`, `schedule_free_adamw` |
| `lr` | `float` | `3e-4` | peak learning rate |
| `weight_decay` | `float` | `0.1` | L2 on 2-D params |
| `betas` | `tuple[float, float]` | `(0.9, 0.95)` | AdamW / Lion momenta |
| `eps` | `float` | `1e-8` | numerical safety (AdamW) |
| `fused` | `bool` | `True` | use fused AdamW when available |
| `muon_momentum` | `float` | `0.95` | Muon momentum coefficient |
| `muon_ns_steps` | `int` | `5` | Newton–Schulz iterations for Muon |
| `muon_adam_lr` | `float \| None` | `None` | LR for 1-D params in Muon's AdamW fallback; `None` → same as `lr` |
| `schedule_free_warmup_steps` | `int` | `0` | internal warmup for schedule-free |

## `[scheduler]` — `SchedulerConfig`

LR schedule shape and warmup.
[`kempnerforge/config/scheduler.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/scheduler.py).

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `name` | `"cosine" \| "linear" \| "wsd" \| "constant" \| "rex" \| "none"` | `"cosine"` | `scheduler` registry key |
| `warmup_steps` | `int` | `2000` | linear warmup length |
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
| `file_pattern` | `str` | `"*.npy"` | glob inside `dataset_path` |
| `tokenizer_path` | `str` | `""` | path or HF id for the tokenizer |
| `num_workers` | `int` | `4` | DataLoader workers |
| `pin_memory` | `bool` | `True` | DataLoader pin memory |
| `prefetch_factor` | `int` | `2` | DataLoader prefetch factor |
| `hf_dataset_name` | `str \| None` | `None` | HF dataset id (e.g. `"wikitext"`) |
| `hf_dataset_config` | `str \| None` | `None` | HF dataset config (e.g. `"wikitext-2-raw-v1"`) |
| `hf_dataset_split` | `str` | `"train"` | HF split |
| `hf_dataset_text_field` | `str` | `"text"` | field to tokenize |
| `hf_streaming` | `bool` | `False` | use `IterableDataset` for large corpora |
| `pack_sequences` | `bool` | `False` | document-aware packing with cross-doc isolation (feeds `doc_ids` to attention) |
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
| `steps` | `int` | `50` | eval batches per evaluation |
| `dataset_path` | `str` | `""` | pre-tokenized eval shards |
| `file_pattern` | `str` | `"*.npy"` | glob inside `dataset_path` |
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
| `dp_replicate` | `int` | `1` | DDP-style replication over FSDP groups |
| `tp` | `int` | `1` | tensor parallel |
| `pp` | `int` | `1` | pipeline parallel |
| `pp_schedule` | `"1f1b" \| "gpipe" \| "interleaved_1f1b"` | `"1f1b"` | pipeline schedule |
| `cp` | `int` | `1` | context parallel (stub; PyTorch 2.11 ring attention) |
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
| `interval` | `int` | `1000` | save every N steps |
| `async_mode` | `"disabled" \| "async" \| "async_with_pinned_mem"` | `"disabled"` | DCP async-save mode |
| `keep_last_n` | `int` | `3` | retain the most recent N checkpoints |
| `load_path` | `str \| None` | `None` | explicit resume path (overrides `latest` symlink) |
| `export_dtype` | `"float32" \| "bfloat16"` | `"bfloat16"` | dtype for HF exports via `scripts/convert_checkpoint.py` |
| `exclude_from_loading` | `list[str]` | `[]` | FQN prefixes to skip on load (e.g. to reinit a head) |

## `[metrics]` — `MetricsConfig`

Logging cadence and backend toggles.
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

## `[profiling]` — `ProfilingConfig`

`torch.profiler` window.
[`kempnerforge/config/profiling.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/profiling.py).

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `enable` | `bool` | `False` | run the profiler during the loop |
| `start_step` | `int` | `5` | first step recorded |
| `end_step` | `int` | `8` | last step recorded (must be `>` `start_step`) |
| `trace_dir` | `str` | `"profiler_traces"` | output directory for Chrome/Perfetto traces |

## Where to read next

- [CLI overrides](cli-overrides.md) — reshape any of these fields from
  the command line.
- [Validation rules](validation-rules.md) — what `__post_init__` and
  `validate(world_size)` enforce.
- [Registry](registry.md) — how the string keys above
  (`moe_router`, `norm_type`, `optimizer.name`, `scheduler.name`,
  `loss_fn`, `model_type`) resolve to builders.
