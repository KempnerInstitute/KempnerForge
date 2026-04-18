# Parallelism Order

Parallelisms are applied in a strict order. The wrong order silently
produces wrong gradients or crashes at runtime with an unhelpful
error — it is *not* a case where PyTorch prints a clear message and
refuses. Get the order right.

The canonical source is
[`kempnerforge/distributed/parallel.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/distributed/parallel.py)
(the module docstring and `build_parallel_model`). This page explains
*why* each step sits where it does.

## The five steps

```
1. Tensor Parallelism (TP)        apply_tensor_parallel
2. Expert Parallelism (EP)        apply_expert_parallel
3. Float8 Training                apply_float8         [optional]
4. Activation Checkpointing (AC)  apply_ac             [optional]
5. FSDP2                          apply_fsdp2
```

## Why this order

### 1 · Tensor Parallelism first

`apply_tensor_parallel` walks the model tree, finds `nn.Linear` modules
by name (e.g. `attention.q_proj`, `mlp.gate_proj`), and replaces them
with sharded `ColwiseParallel` / `RowwiseParallel` plans. It must see the
**raw** modules.

If FSDP2, Float8, or activation checkpointing have already run, those
`nn.Linear` modules are no longer where the sharding plan expects them —
they are wrapped in `CheckpointWrapper`, replaced with `Float8Linear`,
or owned by FSDP2's flat-parameter machinery. `parallelize_module`
silently skips entries it can't match.

TP also needs meta-device init: the full, unsharded parameter never has
to be allocated on any one GPU. `build_parallel_model` creates the model
under `with torch.device("meta"):`, applies TP, then materializes just
the local shards via `model.to_empty(device=...)`.

**Violation:** apply FSDP before TP and you get a correct-looking run
with wrong gradients — TP's plan matches zero modules, the model stays
data-parallel only, and the loss curve *looks* fine.

### 2 · Expert Parallelism second

`apply_expert_parallel` walks the MoE layers and partitions experts
across an `ep` sub-mesh. Each rank keeps `num_experts / ep_world_size`
experts; dispatch becomes all-to-all.

It runs **after** TP because TP on the attention projections is
orthogonal to how experts are partitioned, and because EP modifies the
routed `experts` ModuleList in place — TP's plan already skipped the
`mlp.*` names on MoE blocks (see the check in
`_build_block_tp_plan`).

It runs **before** Float8 so the Float8 filter can still see MoE experts
by FQN and exclude them (experts use `torch._grouped_mm`, not the
Linear forward path).

### 3 · Float8 before AC and FSDP

`apply_float8` replaces `nn.Linear` with `Float8Linear` (torchao,
TENSORWISE recipe — E4M3 forward, E5M2 backward, bf16 master weights).
Order matters on both sides:

- **After TP / EP** — Float8 needs to see the post-TP modules so the
  Float8 cast and the TP shard stay composed. A
  `CheckpointWrapper` or FSDP flat param would hide the `nn.Linear`
  entirely.
- **Before FSDP** — with `enable_fsdp_float8_all_gather=True`, FSDP
  all-gathers the params already in float8 rather than bf16, halving
  communication volume. That only works if Float8 ran first.

The filter function excludes any FQN containing `"experts"`,
`"shared_expert"`, or `"router"`. Experts use grouped GEMM (no
`Linear.forward`); routers have small output dims (often not a multiple
of 16, which `torch._scaled_mm` requires) and aren't compute-bound.

**Note:** `enable_fsdp_float8_all_gather=True` is incompatible with TP
because the float8 weight wrapper calls `aten.is_pinned` on `DTensor`s
and that op has no sharding strategy yet.

### 4 · Activation Checkpointing before FSDP

`apply_ac` wraps `TransformerBlock` (mode `full`) or just `Attention`
(mode `selective`, the balanced trade-off) in `CheckpointWrapper`. Mode
`none` is a no-op.

AC must run **before** FSDP2. FSDP2 bucketizes parameters at the
`fully_shard` boundary; wrapping a layer in `CheckpointWrapper` *after*
FSDP has taken ownership of its parameters breaks the bucket layout and
the reduce-scatter that depends on it.

### 5 · FSDP2 last

`apply_fsdp2` calls `fully_shard(layer, mesh=dp_mesh, mp_policy=...)` on
each `TransformerBlock`, then once on the top-level model for the
remaining parameters (embedding, final norm, output head).

- Default mixed-precision policy: `param_dtype=bf16, reduce_dtype=fp32`
  (bf16 compute, fp32 gradient reduction).
- `reshard_after_forward=True` frees the all-gathered parameters after
  each forward pass (default — smallest memory footprint). Set to
  `False` or a bucket size when pipeline parallelism reuses parameters
  across microbatches.

**EP-MoE special case.** When a block contains MoE with `ep_world_size >
1`, FSDP2 wraps `layer.attention` and `layer.mlp` as separate units
instead of wrapping the whole block. Per-block wrapping would fire
`reduce_scatter` between EP's two backward `all_to_all` calls — a
deadlock where every rank waits on a peer that is waiting on it. The
per-sub-module split guarantees each FSDP reduce-scatter fires after
both EP all-to-alls complete.

## `build_parallel_model` in full

The whole sequence, from
[`kempnerforge/distributed/parallel.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/distributed/parallel.py):

**TP path** (mesh contains a `"tp"` dim):
```
with torch.device("meta"): model = registry.get_model(...)(config)
apply_tensor_parallel(model, mesh)
apply_expert_parallel(model, mesh)
if fp8:  apply_float8(model)
apply_ac(model, ac_mode)
apply_fsdp2(model, mesh, mp_policy)
model.to_empty(device=device)
model.init_weights_and_freqs()
model.to(dtype=param_dtype)
```

**Non-TP path:**
```
model = registry.get_model(...)(config).to(device, dtype=param_dtype)
apply_expert_parallel(model, mesh)
if fp8:  apply_float8(model)
apply_ac(model, ac_mode)
apply_fsdp2(model, mesh, mp_policy)
```

`torch.compile` wraps the final model when `compile_model=True`.

## Adding a new parallelism mode

If you are adding a sixth step, follow
[CONTRIBUTING § New parallelism mode](https://github.com/KempnerInstitute/KempnerForge/blob/main/CONTRIBUTING.md#new-parallelism-mode):
decide where it fits in the dependency chain above, extend
`build_parallel_model` with an explicit step, and add a distributed test
that exercises the new mode in combination with the adjacent ones.

## Quick reference: what goes wrong where

| You did | What happens |
|---------|--------------|
| FSDP before TP | TP finds no `nn.Linear` to shard; run "works" with wrong gradients |
| AC after FSDP | FSDP bucket boundaries break; reduce-scatter misfires |
| Float8 after FSDP | FSDP cannot all-gather in float8; comms revert to bf16 |
| FP8 + TP + `enable_fsdp_float8_all_gather=True` | Crash from `aten.is_pinned` on `DTensor` with no sharding strategy |
| MoE+EP with per-block FSDP wrap | Backward deadlock between EP all-to-all and FSDP reduce-scatter |
| MoE + Pipeline Parallelism | Rejected at config validation — data-dependent routing is incompatible with static stage splitting |

## Where to read next

- [Data flow](data-flow.md) — how a batch travels through the
  parallelized model at step time.
- [Reference § Parallelism recipes](../reference/index.md) — proven
  `(model, GPU count, parallelism)` combinations.
