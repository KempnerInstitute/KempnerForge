# Optimizers

Four optimizers are registered in
[`kempnerforge/training/optimizer.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/training/optimizer.py):
`adamw`, `lion`, `muon`, `schedule_free_adamw`. All are constructed by
`build_optimizer(model, config)`, which routes by `config.name` through
the registry.

## Shared setup: decay grouping

Before any optimizer is constructed, `build_optimizer` splits the
parameters into two groups:

| Group | Condition | `weight_decay` |
|-------|-----------|----------------|
| decay | `param.ndim > 1` and `"bias"` not in name | `config.weight_decay` |
| no-decay | 1D tensors, biases, norm scales | `0.0` |

The rule is `_should_decay(name, param, scope)` in
[`optimizer.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/training/optimizer.py) —
shared by all four optimizers. No decay on biases / norms is the
Llama / GPT convention; Muon's fallback AdamW inherits the same
grouping. The token embedding and the output head are 2D, so they are
decayed.

`optimizer.weight_decay_scope` selects which rule applies. It decides parameter
group membership, so optimizer state saved under one scope cannot be loaded under
another: depending on the optimizer the load fails, the first step fails, or Muon
silently keeps the coefficients it was saved with. The training entry therefore
records the scope in every checkpoint's `metadata.json`, and a resume that would
restore saved optimizer state, on any rank, stops on every rank, before loading
anything, when the checkpoint was saved under a different scope; the error names both
scopes. A checkpoint that holds no optimizer state, such as one converted from Hugging
Face or one saved before the first step, loads under either scope with a fresh
optimizer. A checkpoint that does not record a scope was written under
`"exclude_norms_and_biases"`, the only grouping that existed before the scope was
recorded. To change the scope, continue from the checkpoint with a fresh
optimizer: set `checkpoint.load_path` to it and
`checkpoint.exclude_from_loading = ["optimizer"]` (add `"train_state"` to also
restart the step, schedule and data position), with a `checkpoint.dir` that holds
no checkpoint. Training auto-resumes from a checkpoint in `checkpoint.dir` in place of
`load_path`, optimizer state included (see
[Auto-resume](../checkpointing/auto-resume.md)). Code that builds its own
`CheckpointManager` gets the same record and check by passing `weight_decay_scope`.

| Value | Effect |
|-------|--------|
| `"exclude_norms_and_biases"` (default) | the grouping above |
| `"all"` | every parameter goes in the decay group, as `torch.optim.AdamW` does by default |

## Per-module learning rates

`optimizer.lr_multipliers` gives modules of a VLM their own learning rate. Its keys are
module names of `vlm.module_patterns` (`transformer`, `vision_encoder`, `adapter`, plus an
arch's own, such as `cross_attention`), and each listed module trains at `optimizer.lr`
times its multiplier. The default `vlm.freeze` keeps the vision encoder frozen, so this
example also trains it:

```toml
[optimizer]
lr = 2e-4

[optimizer.lr_multipliers]
vision_encoder = 0.03     # 6e-6
adapter = 5.0             # 1e-3

[vlm]
freeze = []
```

- Each listed module gets its own decay and no-decay groups, after the groups of the
  parameters no listed module holds, which keep `optimizer.lr`. Groups with no parameters
  are left out, so a module frozen when the optimizer is built gets none, and the build
  warns that its multiplier has no effect.
- The one `[scheduler]` scales every group's own peak by the same factor, so every module
  follows the same warmup and decay shape, each from its own peak (and to its own floor,
  `min_lr_ratio` times that peak).
- Under `muon`, a module's AdamW-fallback parameters also take the multiplier, applied to
  `muon_adam_lr`.
- Listed modules may nest. A parameter that several listed modules match belongs to the one
  matching the fewest parameters, when each of the others matches all of those and more:
  under `vlm.arch = "cross_attention"`, `transformer = 0.1` with `cross_attention = 1.0`
  trains the cross-attention layers at `lr` and the rest of the transformer at a tenth of
  it. Two listed modules that match the same parameters, or parameters that partly overlap
  (such as `*.weight` and `transformer.*`), are refused when the optimizer is built.
  Ownership is decided over every parameter, frozen or not, so it does not change between
  freeze stages, and the build logs each module's parameters and the module it sits in.
- Each listed module's rate is logged as `lr/<module>` on the metrics interval, the rate its
  param groups used for that step's update, read before `optimizer.step()` like `train/lr`.
  Under `muon` that is the rate of its orthogonalized matrices,
  and the rate of its AdamW fallback is logged as `lr_adamw/<module>`. `train/lr` stays the
  rate of the first param group, which holds parameters outside the listed modules when
  there are any. No name nests under another, as `train/lr/<module>` would under `train/lr`,
  which a file-based MLflow store cannot keep.

The multipliers and the patterns of their modules decide the param groups and their rates,
which the saved optimizer state and the scheduler state are laid out for. The training entry
records non-empty multipliers in every checkpoint's `metadata.json`, with those patterns and
the module of each optimizer param group. A load of a checkpoint saved with other multipliers
or patterns stops on every rank when it restores the checkpoint's optimizer state on any
rank, before loading anything, or the scheduler state in its `train_state.pt`, before
restoring anything from that file. A checkpoint that records none was saved without any, so
a run without multipliers loads only checkpoints saved without them. A `metadata.json` that
cannot be read, or holds an invalid record, stops a run with multipliers the same way; a run
without them loads the checkpoint as one saved without them, with a warning. Rank 0 reads
the file once per load and sends what it read to every rank, so the ranks stop or go on
together. From a training checkpoint, that leaves:

- A resume, or a `checkpoint.load_path` warm start with or without the optimizer, under the
  same multipliers and patterns.
- A warm start without the optimizer, `checkpoint.exclude_from_loading = ["optimizer"]`, under
  the same multipliers and patterns whose param groups differ from the saved ones: other
  parameters are trainable, as in a stage that unfreezes the language model or freezes
  another module, or `weight_decay_scope` changed. `train_state.pt` records the module of
  each group the scheduler state belongs to, so each group resumes at its module's saved
  rate, and a group of a module the checkpoint holds no group of, such as the language
  model's in a stage that unfreezes it, at the saved peak times its multiplier. A load that
  also restores the optimizer state into groups laid out otherwise stops on every rank before
  loading anything. When the checkpoint holds another number of groups, lacks a setting of
  one of this run's groups, or has a group of another module at a position (a listed module
  where this run's group belongs to none or the reverse, or another listed module, as when
  one stage trains the adapter and the next the vision encoder in its place), the error
  names the position and the warm start.
- A weights-only warm start, `checkpoint.exclude_from_loading = ["optimizer", "train_state"]`,
  under any multipliers and patterns. It loads the model alone and nothing from
  `train_state.pt`, so it reads no multipliers record, and the run starts at step 0 on its
  own schedule, each module at the new `optimizer.lr` times its new multiplier. A stage that
  changes the multipliers or their patterns, or sets a new `optimizer.lr`, as a finetune
  after an alignment stage often does, starts this way.
- No new peak rate in any other load. A resume, and a warm start that keeps
  `train_state.pt`, restore its scheduler state, and the saved schedule carries on, its peak
  included: the scheduler's param groups run from the `optimizer.lr` of the save times each
  one's multiplier, whatever `optimizer.lr` the new stage sets. When a group names a module,
  the load logs the peak of each module's groups, and warns when they differ from the rates
  the optimizer was built with, as when `optimizer.lr` changed since the save.

Code that builds its own `CheckpointManager` gets the same record and check by passing
`lr_multipliers` and `module_patterns`, and calls `build_optimizer(model, config,
module_patterns=...)` to resolve the module names. Its `load` goes ahead under any
multipliers when it restores neither the optimizer state nor a scheduler state, as with
`exclude_keys=["optimizer", "train_state"]`.

## `adamw`

```toml
[optimizer]
name = "adamw"
lr = 3e-4
betas = [0.9, 0.95]
eps = 1e-8
weight_decay = 0.1
fused = true
```

Direct wrapper of `torch.optim.AdamW`. `fused = true` enables the
fused CUDA kernel on PyTorch 2.x + CUDA; falls back to foreach
implementation otherwise. The `fused` kernel is the default in every
shipped dense config.

Memory: 2 optimizer-state tensors per parameter (exp_avg, exp_avg_sq)
in fp32 — the master-weight budget most estimators assume.

## `lion`

```toml
[optimizer]
name = "lion"
lr = 1e-4
betas = [0.9, 0.99]
weight_decay = 0.01
```

Sign-momentum optimizer. The update rule is
`update = sign(β₁·m + (1-β₁)·g)`, with one momentum buffer per
parameter — roughly half AdamW's optimizer-state memory.

Notes from the code:

- **LR is ~3-10× smaller than AdamW's** — Lion's sign-valued update is
  O(1) per coordinate regardless of gradient scale.
- **Decoupled weight decay**: `p.data.mul_(1 - lr * wd)` before the
  sign-update step (no interaction with momentum).
- **No `eps`** — Lion's update never divides by a state tensor.

## `schedule_free_adamw`

```toml
[optimizer]
name = "schedule_free_adamw"
lr = 0.025
betas = [0.9, 0.999]
eps = 1e-8
schedule_free_warmup_steps = 2000

[scheduler]
name = "none"                           # required
```

Defazio's Schedule-Free AdamW. Replaces the external LR schedule with
an internal Polyak-averaging trick that tracks an iterate `z`, a
running average `x`, and a weight sum. The TOML-visible `scheduler.name
= "none"` is required — adding an external cosine or linear schedule
breaks the internal averaging.

Gotchas:

- **Eval mode switch**: call `optimizer.eval_params()` before
  validation and `optimizer.train_params()` after, to swap in the
  averaged weights `x` and restore the iterate `z`. This is not
  automatic in the current loop.
- **Internal warmup**: `schedule_free_warmup_steps` is the optimizer's
  own linear warmup, independent of any external scheduler. Zero
  disables it.
- **LR is 10-100× larger than AdamW's** — the Polyak averaging absorbs
  aggressive step sizes.

## `muon`

```toml
[optimizer]
name = "muon"
lr = 0.02
muon_momentum = 0.95
muon_ns_steps = 5
muon_nesterov = false
muon_adjust_lr = "original"
muon_orthogonalize = "local"
muon_router = false
# muon_adam_lr omitted -> the internal AdamW uses Muon's LR
weight_decay = 0.1
betas = [0.9, 0.95]                     # used by the internal AdamW
eps = 1e-8
```

Keller Jordan's Muon. Applies Newton-Schulz orthogonalization to the
momentum buffer of each hidden weight matrix before stepping; every
other parameter falls through to an internal AdamW with the same
hyperparameters.

How it decides which path each parameter takes (`_muon_matrices`), by
the module that owns it rather than by its shape:

- Weight of an `nn.Linear` → Muon orthogonalized update, however
  rectangular: attention, MLP, expert, cross-attention and adapter
  projections.
- The output head (`OutputHead`), `nn.Embedding` tables (the token
  embedding, a vision tower's position embedding), conv kernels (a vision
  tower's patch embedding), packed 3-D expert weights, norms and biases →
  internal AdamW fallback.
- MoE router gates (`MoEMLP`'s and MoMa's) → internal AdamW, or Muon with
  `muon_router = true`. Practice is split:
  [Moonlight](https://arxiv.org/abs/2502.16982) trained its MoE router
  with Muon, while [Qwen3.8-Next](https://arxiv.org/abs/2608.30320) keeps
  it on AdamW because Muon destabilized it early in training.

A resume needs the split its checkpoint was saved with. One saved before
Muon routed by module, from a model whose embedding or head Muon then
orthogonalized, or saved with the other `muon_router`, fails to load and
names the parameters that moved. A Muon checkpoint saved before the inner
AdamW was checkpointed is the exception — see
[DCP model § Loading](../checkpointing/dcp-model.md#loading).

DTensor / FSDP2 notes:

- `muon_orthogonalize` sets where Newton-Schulz runs on a sharded weight.
  `"local"` (default) orthogonalizes each rank's shard on its own, after
  `_get_local_tensor(p)` unwraps the DTensor, and takes the update scale
  from the shard's shape. That's cheap, but its direction isn't the
  full-matrix result and changes with the shard count, and so can its RMS
  (see the update scale below). `"exact"` gathers each row-sharded matrix
  onto one rank of its shard group with an all-to-all, orthogonalizes the
  full matrix there and scatters the result back, and takes the scale from
  the full matrix's shape. The update is then bitwise what one process
  would compute, and each matrix's work runs once, spread across the group.
  Other sharded layouts, such as tensor parallel, are all-gathered. The
  cost is two all-to-alls per group of matrices per step. A resume keeps
  the mode the checkpoint was saved with, and a checkpoint saved before
  the option existed resumes with `"local"`.
- Momentum buffers match the parameter dtype (`torch.zeros_like(p.grad)`),
  so DCP serializes them correctly on resume.
- The internal AdamW's parameters are not in Muon's `param_groups`, so
  `CheckpointManager` checkpoints it alongside Muon through
  `Muon.inner_optimizers` and a resume restores both bit-exactly — see
  [DCP model § Inner optimizers](../checkpointing/dcp-model.md#inner-optimizers).
- `muon_adam_lr` unset (default) makes the internal AdamW use the same
  LR as Muon. Pass a smaller value (e.g. `muon_adam_lr = 1e-4`) if you
  want the AdamW parameters on a gentler schedule.

Momentum and update scale:

- `muon_nesterov = true` orthogonalizes the Nesterov look-ahead `g + μ·buf`
  instead of the momentum buffer `buf` -- the default of Keller Jordan's
  reference implementation and `torch.optim.Muon`.
- `muon_adjust_lr` sets the factor on each orthogonalized `rows × cols` update:
  `"original"` (default) `sqrt(max(1, rows/cols))`; `"match_rms_adamw"`
  `0.2·sqrt(max(rows, cols))`, which gives every matrix an AdamW-like update RMS
  so AdamW's `lr` and `weight_decay` carry over; `"spectral_unclamped"`
  `sqrt(rows/cols)`, Bernstein's spectral scale without the clamp at 1. The
  first two are also `torch.optim.Muon`'s `adjust_lr_fn` names; torch accepts
  `"spectral_unclamped"` only from 2.14 on.
- `rows × cols` is the matrix Newton-Schulz ran on: each rank's shard under
  `muon_orthogonalize = "local"`, the full weight under `"exact"`, which gives
  the unsharded update in every mode. Under `"local"` the update's RMS
  therefore depends on the sharding: whether the weight is split by rows or
  by columns, into how many shards, and the momentum's rank. For example,
  tensor parallelism's row-wise layers (`o_proj`, `down_proj`) split the
  weight by columns, and over `N` such shards `"original"` raises a full-rank
  momentum's update RMS by about `sqrt(N)`. Use `"exact"` when the update's
  RMS has to match the unsharded run.
- A resume keeps the `muon_nesterov` and `muon_adjust_lr` the checkpoint was
  saved with, and a checkpoint saved before the options existed resumes with
  `false` and `"original"`.

Newton-Schulz coefficients (hard-coded): `a = 3.4445, b = -4.7750, c =
2.0315`, Frobenius-normalized input — standard values from the Muon
paper. A tall matrix is iterated as its transpose, so the Gram matrix the
iteration squares is `min(rows, cols)` on a side.

The shipped
[`7b_16gpu_muon.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/7b_16gpu_muon.toml)
combines Muon with z-loss + chunked cross-entropy as an integration
test of all three features together.

## Picking one

| Situation | Pick |
|-----------|------|
| Default, known-good | `adamw` with `fused = true` |
| Half the optimizer-state memory, accept LR re-tuning | `lion` |
| Want to skip scheduler tuning | `schedule_free_adamw` (`scheduler.name = "none"`) |
| Research recipe for large dense runs | `muon` |

For any optimizer, the decay-vs-no-decay grouping is handled
automatically — no need to thread `no_decay_params` through your config.

## See also

- [Schedulers](schedulers.md) — the `scheduler.*` side, including
  `"none"` for schedule-free.
- [Configuration § OptimizerConfig](../configuration/config-sections.md) —
  every field and its default.
- [Training loop § Optimizer step](training-loop.md#optimizer-and-scheduler-step)
  — where `optimizer.step()` fires and how phase LR scaling layers on
  after.
