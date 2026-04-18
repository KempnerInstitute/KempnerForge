# Validation rules

Validation runs in two passes:

1. **Per-dataclass `__post_init__`** — fires the moment the dataclass is
   instantiated (i.e. as soon as the TOML and CLI overrides land).
   Checks fields that can be validated in isolation.
2. **`JobConfig.validate(world_size)`** — called explicitly by the
   launchers
   ([`scripts/train.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/scripts/train.py),
   [`scripts/eval.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/scripts/eval.py))
   once `world_size` is known. Checks rules that cross section
   boundaries.

All invalid cases raise `ValueError` with a message naming the fields
involved, unless noted otherwise.

## Per-dataclass `__post_init__`

### `ModelConfig`

File:
[`kempnerforge/config/model.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/model.py).

- `n_kv_heads` defaults to `n_heads` if left `None` (MHA).
- `dim`, `n_layers`, `n_heads`, `vocab_size`, `n_kv_heads` must all be
  positive.
- `dim % n_heads == 0` (head dim is integral).
- `n_heads % n_kv_heads == 0` (GQA replication factor is integral).
- `sdpa_backend ∈ {"auto", "flash", "efficient", "cudnn", "math"}`.
- When `num_experts > 0` (MoE):
  - `moe_top_k > 0`
  - `moe_top_k ≤ num_experts`
  - `moe_frequency > 0`
  - `moe_sequence_aux_loss_weight ≥ 0`
  - `moe_bias_schedule ∈ {"constant", "cosine_decay", "linear_warmup"}`

### `TrainConfig`

File:
[`kempnerforge/config/training.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/training.py).

- `batch_size`, `seq_len`, `max_steps`, `grad_accum_steps` all positive.
- `grad_clip_norm > 0`.

### `OptimizerConfig`

File:
[`kempnerforge/config/optimizer.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/optimizer.py).

- `lr > 0`.
- `weight_decay ≥ 0`.
- `betas[0], betas[1] ∈ [0, 1)`.
- `muon_momentum ∈ (0, 1)`.
- `muon_ns_steps > 0`.
- `schedule_free_warmup_steps ≥ 0`.

### `SchedulerConfig`

File:
[`kempnerforge/config/scheduler.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/scheduler.py).

- `warmup_steps ≥ 0`.
- `min_lr_ratio ∈ [0, 1]`.
- `wsd_decay_type ∈ {"cosine", "linear", "sqrt"}`.
- `rex_alpha > 0`.

### `DataConfig` (and nested types)

File:
[`kempnerforge/config/data.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/data.py).

`DataConfig`:

- `num_workers ≥ 0`.
- `prefetch_factor ≥ 1`.
- `mix_temperature > 0`.
- Every `DatasetSource` in `datasets` has `path` or `hf_name` set.
- If `phases` is non-empty, `phases[*].start_step` is strictly
  monotonically increasing (and unique).
- `anneal_start_step ≥ 0`.
- `phases` and `anneal_start_step > 0` are mutually exclusive.

`DatasetSource`:

- `weight > 0`.

`TrainingPhase`:

- `start_step ≥ 0`.
- `lr_scale > 0`.
- Every weight in `dataset_weights ≥ 0`.

### `EvalConfig`

File:
[`kempnerforge/config/eval.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/eval.py).

- `interval > 0`.
- `steps > 0`.

(Data-source presence is enforced by `JobConfig.validate`, not here —
a disabled eval is still a valid `EvalConfig`.)

### `DistributedConfig`

File:
[`kempnerforge/config/distributed.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/distributed.py).

- `dp_shard ∈ {-1} ∪ {1, 2, …}` (negative values other than `-1`
  rejected; `-1` is the auto-resolve sentinel).
- `dp_replicate, tp, pp, cp, ep ≥ 1`.

World-size validation happens in `validate_world_size` (called from
`JobConfig.validate`).

### `CheckpointConfig`

File:
[`kempnerforge/config/checkpoint.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/checkpoint.py).

- `interval > 0`.
- `keep_last_n ≥ 1`.

### `MetricsConfig`

File:
[`kempnerforge/config/metrics.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/metrics.py).

- `log_interval > 0`.

### `ProfilingConfig`

File:
[`kempnerforge/config/profiling.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/profiling.py).

- `end_step > start_step`.

## Cross-section rules — `JobConfig.validate(world_size)`

File:
[`kempnerforge/config/job.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/job.py).

### Parallelism arithmetic

`distributed.validate_world_size(world_size)` requires

```
dp_replicate × dp_shard × tp × pp × cp × ep == world_size
```

with `dp_shard=-1` resolving to `world_size / (dp_replicate·tp·pp·cp·ep)`
(must be an integer). Any mismatch raises `ValueError` naming all six
factors and the resolved `dp_shard`.

### Sequence length

```
train.seq_len ≤ model.max_seq_len
```

The RoPE frequency tables are sized by `model.max_seq_len`; exceeding
it would index past the end.

### Tie embeddings vs pipeline parallel

```
model.tie_embeddings && distributed.pp > 1  →  ValueError
```

The embedding and output-head weights must live on different pipeline
stages, so they can't be tied.

### Eval data source

```
eval.enabled && !eval.dataset_path && !eval.hf_dataset_name  →  ValueError
```

### Tensor-parallel head divisibility

When `distributed.tp > 1`:

- `model.n_heads % distributed.tp == 0`
- `model.n_kv_heads % distributed.tp == 0`

### FP8 + TP

```
train.is_fp8 && distributed.tp > 1  →  ValueError
```

Rationale in the error message: `torchao.Float8Linear` does not compose
with DTensor sharding today.

### MoE + PP

```
model.is_moe && distributed.pp > 1  →  ValueError
```

Rationale: routing is data-dependent and resists pipeline-stage
splitting. Use FSDP, TP, or EP instead.

### Expert parallel

When `distributed.ep > 1`:

- `model.is_moe` must be `True` (the error reads
  `"ep > 1 requires an MoE model (num_experts > 0)"`).
- `model.num_experts % distributed.ep == 0`.

### MoE + torch.compile — warning, not error

```
model.is_moe && train.compile_model  →  logger.warning(…)
```

Routing produces data-dependent shapes that break `torch.compile`'s
graph, so the loop logs a warning but still runs. Set
`train.compile_model=false` for MoE runs.

## Example: a failing config and its error

```toml
# configs/train/bad.toml
[model]
n_heads = 17         # prime → TP divisibility will fail
max_seq_len = 1024

[train]
seq_len = 2048       # exceeds max_seq_len

[distributed]
tp = 4
```

Loading works (no `__post_init__` failure since each field is valid in
isolation), but `validate(world_size=4)` raises:

```
ValueError: train.seq_len (2048) exceeds model.max_seq_len (1024)
```

Fix `seq_len`, re-run, and the next error is:

```
ValueError: n_heads (17) must be divisible by tp (4)
```

`validate` is ordered — each call surfaces the first failing invariant.

## See also

- [Config sections](config-sections.md) — every field that these rules
  reference.
- [Architecture § Parallelism order](../architecture/parallelism-order.md)
  — the 5-step order these cross-section rules exist to protect.
