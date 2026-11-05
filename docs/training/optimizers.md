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
Llama / GPT convention; Muon's 1D fallback AdamW inherits the same
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
recorded. To change the scope, start a new run from the checkpoint without its
optimizer state: set `checkpoint.load_path` to it and
`checkpoint.exclude_from_loading = ["optimizer"]`, with a `checkpoint.dir` that holds
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
times its multiplier:

```toml
[optimizer]
lr = 2e-4

[optimizer.lr_multipliers]
vision_encoder = 0.03     # 6e-6
```

- Each listed module gets its own decay and no-decay groups, after the groups of the
  parameters no listed module holds, which keep `optimizer.lr`. Groups with no parameters
  are left out, so a module frozen when the optimizer is built gets none.
- The one `[scheduler]` scales every group's own peak by the same factor, so every module
  follows the same warmup and decay shape, each from its own peak (and to its own floor,
  `min_lr_ratio` times that peak).
- Under `muon`, a module's AdamW-fallback parameters also take the multiplier, applied to
  `muon_adam_lr`.
- A parameter may belong to only one listed module. Modules whose patterns overlap, such as
  `transformer` and `cross_attention`, are refused when the config is built, or else when
  the optimizer is built and two listed modules match one parameter: list one of them, or
  add a module to `vlm.module_patterns` that names just the parameters you mean.
- Each listed module's rate is logged as `train/lr/<module>` on the metrics interval, the
  rate its param groups step at next. Under `muon` that is the rate of its orthogonalized
  matrices, and the rate of its AdamW fallback is logged as `train/lr/<module>/adamw`.
  `train/lr` stays the rate of the first param group, which holds parameters outside the
  listed modules when there are any.

The multipliers and the patterns of their modules decide the param groups and their rates,
which the saved optimizer state and the scheduler state are laid out for. The training entry
records non-empty multipliers in every checkpoint's `metadata.json`, with those patterns,
and a load that restores optimizer state on any rank, or scheduler state from
`train_state.pt`, stops on every rank, before loading anything, when the checkpoint was
saved with other multipliers or patterns. A checkpoint that records none was saved without
any. Code that builds its own `CheckpointManager` gets the same record and check by passing
`lr_multipliers` and `module_patterns`, and calls
`build_optimizer(model, config, module_patterns=...)` to resolve the module names.

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
# muon_adam_lr omitted -> use same LR as Muon for 1D params
weight_decay = 0.1
betas = [0.9, 0.95]                     # used by internal AdamW for 1D params
eps = 1e-8
```

Keller Jordan's Muon. Applies Newton-Schulz orthogonalization to the
momentum buffer of each 2D weight before stepping; 1D parameters
(biases, norms, highly rectangular matrices) fall through to an
internal AdamW with the same hyperparameters.

How it decides which path each parameter takes
(`_is_muon_eligible`):

- 2D weight, aspect ratio ≤ 10 → Muon orthogonalized update.
- 1D, or ratio > 10 → internal AdamW fallback.

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
  want 1D params on a gentler schedule.

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
paper.

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
