# Compare optimizers

KempnerForge ships four optimizers. Each is registered under a name
that goes in `optimizer.name`, and each reads a different subset of
`[optimizer]` fields. This page explains what those subsets are, what
the LR conventions imply, and how to run a fair head-to-head.

| `optimizer.name` | Class | Extra `[optimizer]` fields | Typical LR |
|------------------|-------|----------------------------|------------|
| `adamw` | `torch.optim.AdamW` (fused) | — | 1e-4 – 3e-4 |
| `lion` | [`Lion`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/training/optimizer.py) | — (uses `betas`) | ~3–10× smaller than AdamW |
| `muon` | [`Muon`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/training/optimizer.py) | `muon_momentum`, `muon_ns_steps`, `muon_adam_lr` | ~0.003 (see below) |
| `schedule_free_adamw` | [`ScheduleFreeAdamW`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/training/optimizer.py) | `schedule_free_warmup_steps` | ~0.025 (constant, see below) |

Each is a single-line swap:

```toml
[optimizer]
name = "muon"    # or "adamw", "lion", "schedule_free_adamw"
lr = 0.003
```

## What they share

All four go through
[`build_optimizer`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/training/optimizer.py),
which splits parameters into two groups before handing them to the
chosen builder:

- **Decay group** — 2D+ parameters without `"bias"` in the name get
  `weight_decay = config.weight_decay`.
- **No-decay group** — 1D parameters (norm scales, biases) and any
  parameter whose name contains `"bias"` get `weight_decay = 0.0`.

So `weight_decay` and `lr` mean the same thing across optimizers;
`betas` is shared by AdamW, Lion (interpretation differs), Muon's
internal AdamW fallback, and Schedule-Free AdamW.

## AdamW

```toml
[optimizer]
name = "adamw"
lr = 3e-4
weight_decay = 0.1
betas = [0.9, 0.95]
eps = 1e-8
fused = true
```

Standard PyTorch fused AdamW. Two state tensors per parameter (first
and second moment). The default for KempnerForge —
[`configs/train/7b_16gpu_adamw.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/7b_16gpu_adamw.toml)
is the reference 7B recipe.

## Lion

```toml
[optimizer]
name = "lion"
lr = 3e-5      # ~10× smaller than the AdamW LR you'd use
weight_decay = 0.1
betas = [0.9, 0.99]
```

Sign-based momentum — the update direction is
`sign(beta1 * m + (1 - beta1) * grad)`. Only **one** state tensor per
parameter (vs two for AdamW), so optimizer memory is roughly halved.

The class docstring recommends an LR 3–10× smaller than AdamW's. If you
port a run from AdamW, scale `lr` down and leave `weight_decay`
alone — Lion's decoupled weight-decay term keeps the same meaning.

`eps` is unused by Lion; it's only in the shared config because other
optimizers read it.

## Muon

```toml
[optimizer]
name = "muon"
lr = 0.003            # for 2D matrices (Newton-Schulz orthogonalized)
weight_decay = 0.1
muon_momentum = 0.95
muon_ns_steps = 5
muon_adam_lr = 3e-4   # Optional: LR for 1D params. Default = same as `lr`.
```

Muon orthogonalizes the 2D momentum update via a 5-step Newton-Schulz
iteration, then scales by `sqrt(max(1, m/n))`. This keeps the update
direction independent of parameter scale, so the effective LR is
much larger than an AdamW LR on the same model.
[`configs/train/7b_16gpu_muon.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/7b_16gpu_muon.toml)
uses `lr = 0.003` as a working reference.

**Split under the hood.** `build_optimizer` routes each parameter
based on [`_is_muon_eligible`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/training/optimizer.py):

- **Muon group**: 2D+ parameters with a reasonable aspect ratio
  (`max(m, n) / min(m, n) <= 10`). Transformer blocks' weight matrices
  qualify.
- **AdamW fallback group**: 1D params (biases, norm scales) and
  highly-rectangular matrices (embeddings, output heads — where
  `X @ X^T` would be `vocab_size × vocab_size`).

The builder logs the two counts:

```
Muon: 6,390,525,952 params (NS-orthogonalized), 525,336,576 params (AdamW fallback)
```

**LR coupling.** Muon's scheduler sees only the Muon param groups.
Before every Muon `step()`, the internal AdamW LR is rescaled by
`muon_current_lr / muon_initial_lr` so both groups follow the same
warmup/decay curve. If you set `muon_adam_lr`, it's the *initial* LR
for the 1D group, then tracks the Muon schedule shape.

**FSDP2 interaction.** Newton-Schulz runs on each rank's local shard
of the weight matrix rather than the fully gathered matrix — an
approximation documented in the
[`Muon` docstring](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/training/optimizer.py).
Works in practice at 16+ GPU scale; the reference Muon config has been
validated there.

## Schedule-Free AdamW

```toml
[optimizer]
name = "schedule_free_adamw"
lr = 0.025                      # constant — no scheduler decay
weight_decay = 0.1
betas = [0.9, 0.999]
schedule_free_warmup_steps = 200

[scheduler]
name = "none"                   # required — see below
```

Schedule-Free AdamW (Defazio & Mishchenko, 2024) maintains two iterates
per parameter — `z` (descent point) and `x` (Polyak average) — and
evaluates gradients at an interpolation `y`. The internal averaging
replaces an LR schedule, so **the training-loop scheduler must be
`none`** (the corresponding registered scheduler that leaves LR
constant):

```python
# kempnerforge/config/scheduler.py
none = "none"  # constant LR (for schedule-free optimizers)
```

Internal LR warmup is controlled by `schedule_free_warmup_steps`
(default 0 = no warmup). Use a few hundred to stabilize early training.

**Memory.** Three state tensors per parameter (`z`, `v` = second
moment, `x` = Polyak average) + the `weight_sum` scalar — larger
footprint than AdamW's two.

**Eval quirk.** Best downstream numbers come from evaluating the
Polyak-averaged `x` rather than the training point `y`. The class
exposes `eval_params()` and `train_params()` to swap the parameters
in place, but the training loop does **not** call them automatically.
If you run `scripts/eval.py` on a schedule-free checkpoint, you're
evaluating `y` unless you patch in those calls — usually close, but
not optimal.

## Running a fair comparison

No benchmark utility ships — you run the comparisons yourself. The
minimum protocol:

1. **Fix everything except the optimizer.** Copy
   [`configs/train/7b_16gpu_adamw.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/7b_16gpu_adamw.toml)
   twice and only change `[optimizer]` (and `[scheduler]` for
   schedule-free). Same seed, same data, same `max_steps`, same
   `batch_size × grad_accum_steps × world_size`.
2. **Use one LR per optimizer, not the same LR across optimizers.**
   The conventions are different: a 3e-4 LR for AdamW is roughly
   3e-5 for Lion and 3e-3 for Muon. Using AdamW's LR with Muon makes
   Muon look catastrophic; using Muon's LR with AdamW NaNs out.
3. **Turn on per-optimizer warmup.** AdamW / Lion / Muon use the
   training-loop scheduler's warmup; Schedule-Free has
   `schedule_free_warmup_steps` instead.
4. **Measure steady-state loss after warmup.** Plot loss vs steps from
   at least `max(warmup_steps × 3, 200)`. The first few hundred steps
   are dominated by warmup transients and aren't diagnostic.
5. **Log enough to diagnose.** `train/grad_norm` for instability,
   `gpu/peak_gb` for memory footprint (especially for Schedule-Free vs
   Lion), `tok/s` for throughput. See
   [Metrics § Metrics tracker](../metrics-and-profiling/metrics-tracker.md)
   for the fields.

A realistic short-horizon pilot is ~500–1,000 steps on the
[`configs/train/hf_wikitext.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/hf_wikitext.toml)
recipe — small model, no dataset setup. Rankings there don't
necessarily transfer to a 7B run, but large discrepancies in memory
or divergence behavior will.

## See also

- [Training § Optimizers](../training/optimizers.md) — class-level
  reference for each optimizer's update rule.
- [Configuration § `[optimizer]`](../configuration/config-sections.md) —
  every `OptimizerConfig` field and its validation rule.
- [Training § Schedulers](../training/schedulers.md) — the `"none"`
  scheduler and why Schedule-Free needs it.
- [Scaling guide](scaling-guide.md) — context for the reference
  configs this page points at.
- [`kempnerforge/training/optimizer.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/training/optimizer.py)
  — source for all four optimizers.
