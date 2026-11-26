# Validation rules

Validation runs in two passes:

1. **Per-dataclass `__post_init__`** — fires the moment the dataclass is
   instantiated (i.e. as soon as the TOML and CLI overrides land).
   Checks fields that can be validated in isolation, and in `JobConfig`
   the cross-section rules listed under
   [`JobConfig.__post_init__`](#cross-section-rules-in-jobconfig__post_init__).
2. **`JobConfig.validate(world_size)`** — called explicitly by the
   launchers
   ([`training/runtime.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/training/runtime.py),
   [`scripts/eval.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/scripts/eval.py))
   once `world_size` is known, before the process group starts. Checks
   rules that cross section boundaries.

All invalid cases raise `ValueError` with a message naming the fields
involved, unless noted otherwise.

## Per-dataclass `__post_init__`

### `ModelConfig`

File:
[`kempnerforge/config/model.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/model.py).

- `n_kv_heads` defaults to `n_heads` if left `None` (MHA).
- `dim`, `n_layers`, `n_heads`, `vocab_size`, `n_kv_heads` must all be
  positive.
- `head_dim_override` must be non-negative and even; `0` infers the head
  dim as `dim // n_heads`.
- With `head_dim_override = 0`, `dim % n_heads == 0` (head dim is
  integral) and `dim // n_heads` must be even, since rotary embeddings
  rotate pairs of dimensions. An odd quotient fails when the config is
  built rather than at model build; set an even `head_dim_override` to
  keep that `dim` and `n_heads`.
- `n_heads % n_kv_heads == 0` (GQA replication factor is integral).
- `activation` is not one of core's `mlp` keys (`swiglu`,
  `standard_gelu`, `standard_relu`); the error names the activation
  that selects it (`silu`, `gelu`, `relu`).
- With `num_experts > 0`, `activation` is `silu`, `gelu` or `relu`: a
  plugin's registered MLP is rejected, since the MoE expert kernels
  (grouped GEMM, packed experts) compute core's SwiGLU or standard MLP
  from the expert weights.
- `sdpa_backend ∈ {"auto", "flash", "efficient", "cudnn", "math"}`.
- `attention_backend ∈ {"sdpa", "flex"}`.
- `attention_backend = "flex"` requires `dim // n_heads ≥ 16` (below that the
  Triton template fails to lower), and warns that `sdpa_backend` is ignored when
  set. There is no upper bound — head_dim 256/320/384/512 are all verified
  working. Note the usable set is not an interval: head_dim 192 fails to compile
  on H200 / torch 2.11 while 128 and 256 are fine. That surfaces as a loud
  compile-time error, not silent corruption, so it is documented rather than
  guarded.
- `attention_backend = "flex"` requires `train.seq_len ≥ 128`, checked by
  `JobConfig` since it reads `[train]`. Below that an
  Inductor-compiled model silently leaks attention across document boundaries.
  Localized to Inductor codegen: the same model under `backend="eager"` or
  `backend="aot_eager"` is exact, as is the FlexAttention kernel on its own, and
  a single attention block compiled by Inductor is exact too — it takes a larger
  graph to trigger. Reproduced identically on torch 2.11/2.13/2.14 (cu128 and
  cu130), so it is not waiting on a release. The dense-mask `"sdpa"` path is
  unaffected at every length, so existing packed runs are not at risk. Not
  reduced further; the bound is empirical. It costs nothing, since a sequence
  below the mask block size has no block sparsity to exploit.
- When `num_experts > 0` (MoE):
  - `moe_top_k > 0`
  - `moe_top_k ≤ num_experts`
  - `moe_frequency > 0`
  - `moe_sequence_aux_loss_weight ≥ 0`
  - `moe_bias_schedule ∈ {"constant", "cosine_decay", "linear_warmup"}`
  - `moe_shared_experts ≥ 0`

### `TrainConfig`

File:
[`kempnerforge/config/training.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/training.py).

- `batch_size`, `seq_len`, `max_steps`, `grad_accum_steps`, `nan_max_consecutive` all positive.
- `grad_clip_norm > 0`.
- `nan_action ∈ {"warn", "skip", "raise"}`; the TOML and CLI loader rejects any other value
  first, through the field's `Literal`.
- `mixed_precision = "fp16"` is rejected, since there is no loss scaling; use `"bf16"`,
  or `"fp32"` on GPUs without bf16. A TOML that sets it fails to load everywhere,
  including `scripts/eval.py`, `generate.py` and `convert_checkpoint.py`, and a CLI
  override cannot fix it, so edit the TOML.

### `OptimizerConfig`

File:
[`kempnerforge/config/optimizer.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/optimizer.py).

- `lr > 0`.
- `weight_decay ≥ 0`.
- `weight_decay_scope ∈ {"exclude_norms_and_biases", "all"}`.
- `betas[0], betas[1] ∈ [0, 1)`.
- `muon_momentum ∈ (0, 1)`.
- `muon_ns_steps > 0`.
- `muon_adjust_lr ∈ {"original", "match_rms_adamw", "spectral_unclamped"}`.
- `muon_orthogonalize ∈ {"local", "exact"}`.
- `schedule_free_warmup_steps ≥ 0`.
- `lr_multipliers` maps non-empty module names without `/` to finite numbers `> 0`. A non-empty one
  needs a `[vlm]` section whose `module_patterns` names every listed module; this is checked
  when `JobConfig` is built. When the optimizer is built, a listed module that matches no
  parameter raises `ValueError`, and so do two listed modules that match the same parameters
  or parameters that partly overlap; a module whose parameters another's strictly contain
  keeps them (see [Optimizers](../training/optimizers.md#per-module-learning-rates)). Which
  parameters the modules share depends on the model's parameter names, so only the build
  can tell.

### `SchedulerConfig`

File:
[`kempnerforge/config/scheduler.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/scheduler.py).

- `warmup_steps ≥ 0`.
- `warmup_start_factor ∈ [0, 1]`.
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
- `token_dtype ∈ {"", "uint16", "uint32"}`.
- Every `DatasetSource` in `datasets` has `path` or `hf_name` set.
- If `phases` is non-empty, `phases[*].start_step` is strictly
  monotonically increasing (and unique).
- `anneal_start_step ≥ 0`.
- `phases` and `anneal_start_step > 0` are mutually exclusive.

`DatasetSource`:

- `weight > 0`.
- `token_dtype ∈ {"", "uint16", "uint32"}`.

`TrainingPhase`:

- `start_step ≥ 0`.
- `lr_scale > 0`.
- Every weight in `dataset_weights ≥ 0`.

### `EvalConfig`

File:
[`kempnerforge/config/eval.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/eval.py).

- `interval > 0`.
- `steps > 0`.
- `token_dtype ∈ {"", "uint16", "uint32"}`.

(Data-source presence is enforced by `JobConfig.validate`, not here —
a disabled eval is still a valid `EvalConfig`.)

The `.bin` dtype itself is resolved, checked against `metadata.yaml`
and against each shard's size when the dataset is built, not at config
load; see [Memory-mapped dataset](../data/memory-mapped.md).

### `DistributedConfig`

File:
[`kempnerforge/config/distributed.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/distributed.py).

- `dp_shard ∈ {-1} ∪ {1, 2, …}` (negative values other than `-1`
  rejected; `-1` is the auto-resolve sentinel).
- `dp_replicate, tp, pp, cp, ep ≥ 1`.
- `cp == 1`. Context parallelism is not implemented and the mesh has no
  `cp` dimension, so `cp > 1` would build a mesh that leaves ranks out,
  or train as plain FSDP when `cp` equals the world size. Rejecting it
  here stops the run before `init_distributed`.

World-size validation happens in `validate_world_size` (called from
`JobConfig.validate`).

### `CheckpointConfig`

File:
[`kempnerforge/config/checkpoint.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/checkpoint.py).

- `interval > 0`.
- If `dyn_ckpt_window` is set:
  - `dyn_ckpt_window.start >= 0`.
  - `dyn_ckpt_window.stop >= dyn_ckpt_window.start`.
  - `dyn_ckpt_window.strategy` must name a registered dyn_ckpt strategy
    (`"power2"` ships by default; see
    [`kempnerforge/config/registry.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/registry.py)).

### `MetricsConfig`

File:
[`kempnerforge/config/metrics.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/metrics.py).

- `log_interval > 0`.
- `mlflow_description`, when set, is a non-empty string.
- `mlflow_tags` maps non-empty string names to non-empty string values; the error names
  the offending key.
- With `enable_mlflow` and a Databricks tracking URI, `mlflow_experiment` (when set) is an
  absolute workspace path, starting with `/`. The URI is `mlflow_tracking_uri`, else
  `$MLFLOW_TRACKING_URI`, else `"databricks"`, read when the config loads.
- `mlflow_register_checkpoints` requires `enable_mlflow`.

### `ProfilingConfig`

File:
[`kempnerforge/config/profiling.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/profiling.py).

- `end_step > start_step`.

### `VLMConfig` freeze specs

File:
[`kempnerforge/config/vlm.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/vlm.py).

- `freeze_schedule` start steps are strictly increasing.
- Every `module` in `freeze` and in each `FreezeStage` is a module alias, a key of
  `module_patterns`, or a parameter pattern, which holds a dot or one of `*?[`; a key
  resolves as its alias first, dot or not. A bare word that is neither, such as a typo of
  an alias, fails when the config loads. The check reads the TOML with the CLI overrides
  laid over it, so the alias a spec names may be a `module_patterns` key set on the
  command line.
- The VLM build refuses a pattern in `freeze` or in a `FreezeStage` that matches no
  parameter of the built model, and names it.

### `AdapterConfig`

File:
[`kempnerforge/config/adapter.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/adapter.py).

- `type` must name a registered adapter; the error lists them.
- `hidden_dim >= 0`.
- `activation ∈ {"gelu", "silu", "relu"}`.
- `pre_norm` is `""` or a registered norm; the error lists them.
- `pool_window > 0` and `pool_heads > 0`.
- Every other field must be one the selected adapter uses, or stay at its
  default. The fields an adapter uses come from its builder's parameters, or
  from the `accepts` it registers with (see
  [Registry](registry.md#adapters-declare-the-fields-they-use)), so the check
  needs no list of types; the error names each unused field, its value, the
  adapter type, and the fields that type uses:

  | `type` | Uses |
  |--------|------|
  | `mlp_2layer` | `hidden_dim`, `activation`, `pre_norm` |
  | `linear` | none |
  | `avgpool` | `pool_window`, `hidden_dim`; `activation` and `pre_norm` once `hidden_dim > 0` |
  | `attentional_pool` | `pool_window`, `pool_heads` |

  A plugin adapter whose builder reads a `**kwargs` and registers no
  `accepts` accepts every field.
  The check reads the TOML with the CLI overrides laid over it, so a
  setting the adapter would drop stops the run before `init_distributed`,
  whichever layer sets it. An override never restores a default, so a type
  switched on the command line needs the fields it does not use reset there
  too (for example `--adapter.hidden_dim=0 --adapter.pre_norm=`); see
  [Config sections](config-sections.md#adapter-adapterconfig).

## Cross-section rules in `JobConfig.__post_init__`

File:
[`kempnerforge/config/job.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/job.py).

These run when `JobConfig` is built, with the per-dataclass checks:

- `plugins` must be a list of module path strings.
- `data.hf_max_images != 1` applies to VLM image runs only, not text runs
  or `[video]`, and needs `vlm.arch = "joint_decoder"`. It is refused
  with `train.mixed_precision = "fp8"` and, under
  `attention_backend = "sdpa"`, with `model.sdpa_backend = "flash"`.
- `[video]` needs a `[vlm]` section, and `vlm.tile_grid` must stay 0
  with `[video]`.
- `[vlm]` needs a `[vision_encoder]` section; a missing `[adapter]`
  takes the defaults.
- With `vision_encoder.num_tokens > 0`, `model.max_seq_len` must hold
  the image tokens the residual stream carries plus `vlm.max_text_len`.
  The image tokens count every image a document may hold
  (`data.hf_max_images`, or one clip on a `[video]` run) and every view
  of each (each frame of a clip, or each view of a tiled image) after
  the adapter, and none under Cross-Attention, whose residual stream
  carries text only.
- On a VLM run, an HF-backed encoder (`siglip2`, `clip`) with
  `feature_dim` or `num_tokens` above 0 logs a warning: the values
  override the ones the encoder probes.
- `data.pack_sequences` on a VLM run needs `vlm.arch = "joint_decoder"`
  and takes neither `[video]` nor `train.mixed_precision = "fp8"`.
- A nonempty `optimizer.lr_multipliers` needs a `[vlm]` section whose
  `module_patterns` names every listed module (see
  [`OptimizerConfig`](#optimizerconfig)).
- `attention_backend = "flex"` needs `train.seq_len ≥ 128` (see
  [`ModelConfig`](#modelconfig)).

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

### Eval steps under pipeline parallelism

```
eval.enabled && distributed.pp > 1 && eval.steps % train.grad_accum_steps != 0  →  ValueError
```

Eval runs its batches through the training schedule, `grad_accum_steps` at a
time, so each eval microbatch has the shape the stages were built for.

### VLM architectures under tensor parallelism

Checked first when `distributed.tp > 1`, ahead of head divisibility, so the
error names the arch rather than a head count:

```
isinstance(vlm, MoTConfig) && distributed.tp > 1             →  ValueError
vlm set && distributed.tp > 1 && sequence parallelism on     →  ValueError
```

The tensor-parallel plan
([`apply_tensor_parallel`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/distributed/tensor_parallel.py))
is written for `TransformerBlock` and walks `model.layers` only. The
checks act on the class of the `[vlm]` config, and the errors name that
class's arch.

- `mot`: `MoTBlock` renames the attention modules (`attn_norm`, `attn`) and
  holds `mlp_norm` and `mlp` as per-modality `ModuleDict`s, so the build
  fails with `KeyError: 'parameter name can't contain "."'` or no parameter
  is sharded.
- Every other arch, with sequence parallelism on: the plan shards the
  activations along the sequence for a dense model with untied embeddings
  (`ModelConfig.tp_sequence_parallel`, without pipeline parallelism), and the
  first forward fails on mixed `torch.Tensor` and `DTensor` operands, in
  `joint_decoder` and `moma` where the image prefix joins the token
  embeddings and in `cross_attention` in the first cross-attention block.

With sequence parallelism off, which an MoE model or tied embeddings give
(for `moma` only tied embeddings, since MoMa keeps `model.num_experts = 0`),
`joint_decoder`, `cross_attention` and `moma` train under tensor parallelism
and match an unsharded model with the same weights. The plan shards the
transformer blocks, and the cross-attention blocks and MoMa's expert groups
stay whole on every tensor-parallel rank and compute the same result there,
as MoE experts do in text models. Otherwise set `distributed.tp = 1`.

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

### FP8 geometry

```
train.is_fp8 && model.dim % 16 != 0                                           →  ValueError
train.is_fp8 && vlm is None && (train.batch_size * train.seq_len) % 16 != 0   →  ValueError
```

`torch._scaled_mm` needs both dimensions of an FP8 weight to be
multiples of 16, and every attention and MLP projection and the output
head has `model.dim` on one side, so with any other `dim` no layer
could run in FP8. Other unaligned layers are not errors: `k_proj` and
`v_proj` with a small `n_kv_heads * head_dim`, an output head with a
`vocab_size` such as 50257, or an explicit `ffn_hidden_dim` that is not
a multiple of 16 stay in bf16, and `apply_float8` names them in a
startup warning (see
[FP8 § Exclusion rules](../distributed/fp8.md#exclusion-rules)).

The second rule covers the weight gradient. torchao computes it as
`grad_output.t() @ input`, which multiplies over the micro-batch's
`batch_size * seq_len` tokens, and `torch._scaled_mm` needs that
dimension to be a multiple of 16 as well. Without the check, such a run
passes the forward and fails in its first backward. Under pipeline
parallelism each micro-batch is also `batch_size` rows. VLM runs are
not checked, because their token counts depend on the data (image
tokens plus text) rather than on the config alone; a VLM micro-batch
whose token count is not a multiple of 16 fails in its first backward,
with an error such as `Expected trailing dimension of mat1 to be
divisible by 16 but got mat1 shape: (<out_features>x<tokens>)`. A
packed VLM run (`data.pack_sequences = true`) is refused with FP8 when
`JobConfig` is built (`__post_init__`), because its head runs over the
batch's text tokens, a count that changes from batch to batch. With
`data.hf_streaming`, the last batch of each pass over the stream can be
short, so keep `train.seq_len` itself a multiple of 16 there.

### MoE + PP

```
model.is_moe && distributed.pp > 1  →  ValueError
```

Rationale: routing is data-dependent and resists pipeline-stage
splitting. Use FSDP, TP, or EP instead; for `vlm.arch = "mot"` the error
offers FSDP alone, since MoT runs neither TP nor EP.

### Expert parallel

When `distributed.ep > 1`:

- `vlm.arch` must not be `"mot"` or `"moma"`. Checked first, so these jobs
  get their own reason whatever `model.num_experts` is:
  - `mot`: the FSDP2 wrap for expert-parallel blocks expects
    `layer.attention` and `layer.mlp`, and `MoTBlock` names its attention
    `attn` and keeps its experts per modality, so the model build fails. MoT
    with MoE trains with `ep = 1`.
  - `moma`: `apply_expert_parallel` partitions `MoEMLP` experts only, and
    MoMa's per-modality expert groups need an EP-aware dispatch that is not
    wired yet. MoMa also requires `model.num_experts = 0`, which the next
    check rejects, so this one has to come first to be seen at all.
- `model.is_moe` must be `True` (the error reads
  `"ep > 1 requires an MoE model (num_experts > 0)"`).
- `model.num_experts % distributed.ep == 0`.

### Vision encoder weights

```
vlm set && vision_encoder.type in {"siglip2", "clip"} && vision_encoder.path == ""  →  ValueError
```

The Hugging Face encoders load a pretrained model from `vision_encoder.path`,
a Hub id or a local directory, and the error asks for it. The check runs here,
on the config after the CLI overrides, so a TOML can leave the path to
`--vision_encoder.path`. The model's own config is read only when the model is
built: with `vision_encoder.feature_dim = 0`, a model whose config has no
`hidden_size` fails there, with an error that names `vision_encoder.path` and
asks for `vision_encoder.feature_dim`, the width of the features its vision
tower returns.

### `[video]` and the `[data]` Hugging Face fields

```
video set && a data.hf_* field differs from its default  →  ValueError
```

A video run builds its dataset from `[video]`. From `[data]` it reads
`tokenizer_path` and the loader settings (`num_workers`, `pin_memory`,
`prefetch_factor`), so a Hugging Face field set there would be dropped
silently. The error lists each such field, with the `[video]` field to set
instead where there is one, and the defaults to put back:

| `[data]` field | Set instead |
|----------------|-------------|
| `hf_dataset_name` | `video.data_root` and `video.dataset_name` |
| `hf_dataset_split` | `video.split` |
| `hf_dataset_prompt_field` | `video.prompt`, one instruction for every clip |
| `hf_image_size` | `video.frame_size` |
| `hf_dataset_config`, `hf_dataset_text_field`, `hf_dataset_image_field`, `hf_streaming` | no counterpart |

The fields are compared with their defaults, so `hf_dataset_split = "train"`
or `hf_image_size = 224` passes, and an empty `hf_dataset_name` or
`hf_dataset_config` counts as unset like their default `None`.
`data.hf_max_images` is refused with `[video]` when `JobConfig` is built.

### MoE + torch.compile — warning, not error

```
model.is_moe && train.compile_model  →  logger.warning(…)
```

Routing produces data-dependent shapes that break `torch.compile`'s
graph, so the loop logs a warning but still runs. Set
`train.compile_model=false` for MoE runs.

### Flash attention and masked runs

```
model.sdpa_backend == "flash" && (
    data.pack_sequences && model.attention_backend == "sdpa" && a data source
    || video && vlm.arch in {"joint_decoder", "mot", "moma"}
             && the vlm section's class is not MoT
)  →  ValueError
```

Checked by the training setup on CUDA, after the CLI overrides and
before the process group starts
(`JobConfig.check_flash_attention_masks`). Evaluation and generation
are not checked at setup; on CUDA, a forward there that passes a dense
mask to `Attention`, such as video evaluation's frame mask under
Joint-Decoder or MoMa, raises the `ValueError` from
`Attention.forward`. PyTorch's CUDA FlashAttention
kernel takes no attention mask, and these runs build one: packed rows
read from a dataset mask their documents from each other, and the padded
frames of a video run are masked in self-attention, under
`attention_backend = "flex"` too, which turns only document and group
masks into a block mask. The modality strategy, which `vlm.arch` picks,
masks frames in self-attention under Joint-Decoder, MoT and MoMa, while
Cross-Attention masks them in its cross-attention layers only, and the
attention the model builds, which the section's class picks, runs under
the configured backend except under MoT, so a run counts only when both
apply (`JobConfig.masks_frames_in_backend_attention`), as after a
`--vlm.arch` override. Image runs whose documents can hold several
images (`data.hf_max_images != 1`) are refused under `"sdpa"` when the
config is built, and on CUDA a dataset with text-only rows on the
single-image route (`data.pack_sequences = false` and
`data.hf_max_images = 1`) is refused by the same test when the data
pipeline is built, since only the dataset knows it has such rows. A
forward that passes a dense mask to SDPA some other way on CUDA, such as
attention groups from a collator of your own, raises a `ValueError` that
names the setting instead of PyTorch's `No available kernel`.

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
