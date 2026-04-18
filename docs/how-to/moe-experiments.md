# MoE experiments

End-to-end workflow for running Mixture-of-Experts models in
KempnerForge: pick a router, stand up a baseline, tune the balance
signals, diagnose hot/cold experts, turn on Expert Parallelism when
memory demands it, and know where the composition rules stop you.

This guide assumes you've read [MoE § overview](../moe/index.md). The
subsystem pages cover the mechanics; this page covers the *when* and
*why*.

## Start from a working config

Two reference configs ship with the repo. Pick whichever matches the
scale you're iterating at:

```bash
# 1-GPU sanity check (~1 min): 4 experts, top-2, alternating MoE layers
uv run python scripts/train.py configs/train/debug_moe.toml

# 32-GPU production profile: 8 experts, top-2, EP=2 + TP=4 + FSDP=4
sbatch --nodes=8 scripts/slurm/multinode.sh configs/train/moe_ep_32gpu.toml
```

`debug_moe.toml` uses `dim=256, n_layers=4, num_experts=4` — small
enough to run on a laptop-scale GPU in under a minute. Use it to
validate config changes before launching a real run.

## Pick a router

```toml
[model]
num_experts    = 8
moe_top_k      = 2
moe_router     = "softmax_topk"   # or "sigmoid_topk"
```

The two routers have different balance strategies — start with
`softmax_topk` if you're just getting a baseline up:

| Goal | Router | Why |
|------|--------|-----|
| Mixtral reproduction | `softmax_topk` | Matches original recipe, aux-loss coefficient well-studied (0.01) |
| First MoE run, any target | `softmax_topk` | Simpler: one knob (`moe_aux_loss_weight`) and balance is visible in the loss |
| DeepSeek-V3 reproduction | `sigmoid_topk` | Bias-based balancer, matching paper |
| Long runs where aux-loss coefficient is brittle | `sigmoid_topk` | Balance signal doesn't perturb the main loss gradient |

See [Routers](../moe/routers.md) for the mechanics.

## Tune the aux loss

For `softmax_topk`, the coefficient `moe_aux_loss_weight` controls how
hard the router is pushed toward uniform:

```toml
[model]
moe_aux_loss_weight = 0.01    # Switch/Mixtral default — keep unless you see imbalance
```

- **0.01 (default)**: the Switch / Mixtral value. Start here.
- **0.1**: over-regularizes — experts stay uniform but can't
  specialize. Use only if you have a severe collapse problem and want
  to force exploration early.
- **0.001**: under-regularizes. Balance drifts. Only use if you
  *also* turn on capacity factor or packed experts.

For `sigmoid_topk`, the coefficient has no effect by default
(`aux_loss` is 0). It only matters if you've enabled
`moe_sequence_aux_loss_weight > 0`; see
[Aux loss § Sequence-level](../moe/aux-loss-and-balancing.md#sequence-level-aux-loss-sigmoid-only).

## Diagnose hot/cold experts

The training loop logs two MoE-specific metrics when `num_experts > 0`:

```
moe/aux_loss        # scalar aux loss this step (dispatches to WandB/TB)
moe/expert_balance  # min/max across (layer × expert) — 1.0 = perfect uniform
```

Interpretation:

- **`expert_balance > 0.5`** — fine. Balance is within 2× across
  experts; normal for softmax router.
- **`expert_balance < 0.2`** — one or more experts are getting ≥ 5×
  more tokens than the quietest. Router is specializing; check if
  intentional.
- **`expert_balance → 0`** — an expert got zero tokens this step.
  If it happens for one step, ignore. If it persists across many
  logging intervals, you have a dead expert.

For per-layer detail, call `Transformer.get_expert_counts()` —
returns `dict[layer_idx → tensor(num_experts,)]`. Useful to spot
*which* layer is collapsing:

```python
# In a custom script or notebook
counts = model.get_expert_counts()
for layer_idx, c in counts.items():
    pct = (c / c.sum() * 100).tolist()
    print(f"layer {layer_idx}: {[f'{p:.1f}%' for p in pct]}")
```

A balanced 8-expert layer prints ~12.5% per expert. A collapsing one
prints something like `[45%, 30%, 10%, 5%, 5%, 3%, 2%, 0%]`.

Recovery options, in order of what to try first:

1. **Raise `moe_aux_loss_weight`** from 0.01 → 0.02 (softmax) or turn
   on `moe_sequence_aux_loss_weight = 0.001` (sigmoid).
2. **Set `moe_capacity_factor = 1.25`** — caps hot experts at 25%
   above average, spilling overflow to residual.
3. **Enable `moe_gradient_scale = true`** — normalizes per-expert
   gradient magnitudes so cold experts aren't starved of learning
   signal.
4. **Restart from a checkpoint** before the collapse with a different
   seed. Balance issues that show up after many thousand steps often
   don't reappear with a different initialization.

## When to turn on EP

Expert Parallelism splits experts across ranks. It's a memory-first
feature: flip it on when experts dominate your parameter budget, not
to go faster.

Rule of thumb — turn on EP when:

1. **Expert weights > 50% of total model memory.** For a 4B-total
   MoE with 8 experts and `moe_frequency=1`, experts are typically
   70-80% of parameters. EP=2 cuts that in half per rank.
2. **You're OOMing with FSDP alone.** FSDP shards by tensor, but
   experts live as `(E, dim, hidden)` parameters (packed) or as
   `ModuleList`. FSDP wraps them but can't split across the expert
   dimension; EP does.
3. **Cross-node bandwidth is adequate.** EP adds two all-to-alls per
   MoE layer (dispatch + combine). The measured 32-GPU EP=2 profile
   assumes InfiniBand; commodity Ethernet interconnects have not been
   benchmarked. See
   [Benchmarks § MoE Expert Parallelism](../reference/benchmarks.md#moe-expert-parallelism)
   for measured numbers on H200 + IB.

Constraints:

- `num_experts` must be divisible by `ep` (validated in
  `JobConfig.__post_init__`).
- `ep > 1` with `num_experts == 0` is rejected — EP requires MoE.
- Typical combinations: `ep=2` for 8 experts, `ep=4` for 8-16 experts,
  `ep=8` for 32+ experts.

Minimal toggle:

```toml
[distributed]
tp       = 4    # unchanged
ep       = 2    # new
dp_shard = 4    # unchanged
```

See [Expert parallelism](../distributed/expert-parallelism.md) for the
all-to-all mechanics and throughput measurements.

## Capacity factor: only when EP is on

`moe_capacity_factor` caps tokens per expert. It's almost always the
wrong knob to turn on without EP:

- **Without EP**: capacity factor drops tokens (residual still flows),
  which is wasted compute with no memory upside. Leave at `0.0`.
- **With EP**: dispatch all-to-all buffers are sized for the
  *worst-case* token distribution. Without a cap, one rank can
  overflow its buffer when many tokens happen to prefer its experts.
  `moe_capacity_factor = 1.25` (Switch default) bounds the buffer
  predictably.

```toml
[model]
moe_capacity_factor = 1.25   # only meaningful with distributed.ep > 1
```

Start at 1.25; raise to 1.5 if you see ~5%+ token drop in metrics,
lower to 1.0 if throughput is buffer-bound. See
[Capacity and dispatch § When to use capacity](../moe/capacity-and-dispatch.md#when-to-use-capacity).

## Packed experts: on if `num_experts ≥ 16`

```toml
[model]
moe_packed_experts = true
```

Replaces the `ModuleList` of experts with three packed
`(num_experts, dim, hidden)` tensors. Measured speedups from
[`benchmarks/moe_packed`](https://github.com/KempnerInstitute/KempnerForge/blob/main/benchmarks/moe_packed/moe_packed_benchmark.md):

| `num_experts` | Unpacked | Packed | Speedup |
|:---:|---:|---:|:---:|
| 8  | 48,521 tok/s | 50,972 tok/s | +5.1% |
| 16 | 26,994 tok/s | 36,860 tok/s | +36.5% |
| 64 | 1,796 tok/s  | 2,204 tok/s  | +22.7% |

At 8 experts the win is marginal; at 16+ it's worth the flag. Default
is off because the EP integration (slicing packed tensors on the
expert axis) is newer than the unpacked path.

## Composition caveats

### MoE + TP: fine, but TP doesn't touch experts

TP applies to attention Q/K/V/O projections. In MoE layers, expert
weights stay **replicated** across the TP group; TP gives you no
memory savings on experts. Use EP for that.

```python
# kempnerforge/distributed/tensor_parallel.py — _apply_block_tp (trimmed)
if isinstance(block.mlp, MoEMLP):
    pass  # experts replicated, TP on attention only
```

Sequence-parallel is also disabled for MoE blocks (boolean indexing in
expert dispatch breaks `Shard(1)` DTensors).

### MoE + FP8: experts stay bf16

FP8 conversion is applied to dense `Linear`s only. Three classes are
excluded: routed experts, shared expert, router gate. See
[MoE + FP8](../moe/moe-fp8.md) for the rationale. Practical consequence:
FP8 throughput lift on an MoE model is smaller than on a dense model
of the same active-parameter count, because expert-weight GEMMs run
at bf16 regardless.

FP8 + EP composes fine — they're orthogonal (EP moves experts across
ranks, FP8 wouldn't touch them anyway).

FP8 + TP does **not** compose — `JobConfig.__post_init__` rejects it
because `Float8Linear`'s DTensor strategy is incomplete. See
[FP8 § TP incompatibility](../distributed/fp8.md#tp-incompatibility).

### MoE + PP: not supported

```
MoE + Pipeline Parallelism is not supported. MoE layers use
all-to-all communication which conflicts with the pipeline schedule.
```

Raised in `JobConfig.__post_init__` when `num_experts > 0` and
`pp > 1`. For very large MoE runs that need PP, you'd need a
schedule that interleaves all-to-all with pipeline microbatches —
not in KempnerForge today.

## Minimal production recipe

A reasonable starting point for a 4B-total / 1.8B-active MoE on
32 H200 GPUs:

```toml
[model]
num_experts             = 8
moe_top_k               = 2
moe_router              = "softmax_topk"    # or "sigmoid_topk" for DeepSeek-V3
moe_frequency           = 1
moe_aux_loss_weight     = 0.01
moe_capacity_factor     = 1.25              # because EP is on
moe_packed_experts      = false             # 8 experts: not worth it
moe_gradient_scale      = false             # baseline only

[distributed]
tp       = 4
ep       = 2
dp_shard = 4

[train]
mixed_precision = "bf16"    # fp8 has limited lift on MoE; stay bf16 for this recipe
```

Copy from [`configs/train/moe_ep_32gpu.toml`](https://github.com/KempnerInstitute/KempnerForge/blob/main/configs/train/moe_ep_32gpu.toml)
if you want to run it directly.

## See also

- [MoE § overview](../moe/index.md) — the subsystem pages this guide
  pulls together.
- [Routers](../moe/routers.md) — full mechanics of the two routers.
- [Aux loss and balancing](../moe/aux-loss-and-balancing.md) —
  coefficient tuning and bias schedules.
- [Capacity and dispatch](../moe/capacity-and-dispatch.md) — drop
  policy and grouped GEMM path.
- [Expert parallelism](../distributed/expert-parallelism.md) — EP
  mechanics and measured throughput.
- [Benchmarks § MoE Expert Parallelism](../reference/benchmarks.md#moe-expert-parallelism)
  — full 32-GPU measurement table.
- [Validation rules](../configuration/validation-rules.md) — the MoE
  cross-section config checks.
