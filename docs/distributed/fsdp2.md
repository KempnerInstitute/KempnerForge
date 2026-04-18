# FSDP2

KempnerForge uses PyTorch's composable `fully_shard()` API (FSDP2),
not the older `FullyShardedDataParallel` module. Entry point:
`apply_fsdp2(model, device_mesh, mp_policy=None, reshard_after_forward=True)`
in
[`kempnerforge/distributed/parallel.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/distributed/parallel.py).

## What it shards

Two levels:

1. **Per-block** — each `TransformerBlock` in `model.layers` is
   wrapped with its own `fully_shard()` call.
2. **Top-level** — the whole model is wrapped once more. This covers
   parameters that live outside the blocks: the token embedding, the
   final norm, the output head, and (for EP-MoE blocks) the layer
   norms that don't get swept up by the sub-module wrap below.

Both wraps share the same `MixedPrecisionPolicy` and
`reshard_after_forward` setting.

## Mixed precision policy

```python
default_mp_policy(param_dtype=torch.bfloat16)
# returns
MixedPrecisionPolicy(param_dtype=bfloat16, reduce_dtype=float32)
```

- **`param_dtype`** — the dtype FSDP2 casts parameters to during
  all-gather. `bfloat16` is the default. If `mixed_precision = "fp8"`,
  the param dtype stays `bfloat16` and FP8 is applied as a separate
  layer on top (see [FP8](fp8.md)); FSDP2 never sees FP8 types
  directly.
- **`reduce_dtype = float32`** — gradient reductions happen in fp32
  regardless of param dtype. This is the standard "bf16 compute, fp32
  reduce" policy and the source of FSDP2's numerical stability
  advantage over bare bf16 training.

Pass a custom `MixedPrecisionPolicy` as the `mp_policy` kwarg if you
need a different combination — most of the time the default is what
you want.

## `reshard_after_forward`

```python
apply_fsdp2(model, device_mesh, reshard_after_forward=True)
```

| Value | Behavior | When to use |
|-------|----------|-------------|
| `True` (default) | All-gather on forward, reshard after, all-gather again on backward | Maximum memory savings; every forward pass reshards |
| `False` | All-gather once, keep gathered across the step | Pipeline parallel: `1F1B` schedule sends many microbatches through each stage; resharding between them would be wasted work |
| `int` | Rate-limit concurrency: keep at most `N` blocks gathered at once | Rarely used; middle-ground for memory tuning |

[`scripts/train.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/scripts/train.py)
calls `apply_fsdp2(model, device_mesh, mp_policy=mp_policy)` on all
paths without overriding `reshard_after_forward`, so both dense and
PP runs default to `True`. The docstring on `pipeline_parallel.py`
recommends `False` for PP (to amortize the all-gather over
microbatches), but that's not currently wired through the training
script — pass it manually if you need it.

## EP-MoE: per-sub-module wrapping

The MoE+EP path needs a wrapping pattern that the dense path does not:

```python
if _has_ep_moe(layer):
    fully_shard(layer.attention, ...)
    fully_shard(layer.mlp, ...)          # MoE MLP wrapped separately
else:
    fully_shard(layer, ...)               # whole block wrapped together
```

Why: under EP, the MoE MLP's backward pass issues **two** all-to-all
calls (unpermute + backward dispatch). If FSDP2's reduce-scatter for
the whole block fires between them, different ranks end up in different
communication phases — deadlock.

Wrapping `attention` and `mlp` separately means attention's reduce-
scatter fires after attention's backward, and MLP's reduce-scatter
fires after both all-to-alls complete. All dp_shard peers share the
same EP rank and so reach each phase together.

This fix is why
[the MoE benchmark](../reference/benchmarks.md#moe-expert-parallelism)
calls out "per-sub-module FSDP wrapping" as the load-bearing
optimization.

## Activation checkpointing

`apply_ac(model, mode)` runs **before** `apply_fsdp2` — the FSDP2
wrap sees the already-checkpointed block as its unit.

Three modes from
[`ActivationCheckpointing`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/training.py):

| `act_ckpt` | What gets rematerialized |
|------------|--------------------------|
| `"none"` | Nothing; fastest; uses most memory |
| `"full"` | Every `TransformerBlock` — entire block is recomputed on backward |
| `"selective"` | Only the `Attention` module within each block — MLP activations stay materialized |

All three use `CheckpointImpl.NO_REENTRANT` — the modern
`torch.utils.checkpoint` implementation that composes cleanly with
FSDP2 and `torch.compile`.

`"selective"` is the right default when memory is tight but you still
want MLP's full forward — attention is both the biggest activation
consumer (O(seq_len^2)) and cheap to recompute (SDPA fused kernel).
`"full"` is the fallback when even selective doesn't fit.

## Application order inside `build_parallel_model`

Non-PP path:

```
model = build_model(...)                   # on device, not meta
apply_expert_parallel(model, mesh)          # MoE only — partitions experts
apply_float8(model)                         # optional — converts nn.Linear
apply_ac(model, mode)                       # optional — wraps blocks / attention
apply_fsdp2(model, mesh, mp_policy)         # shards remaining params
```

TP-enabled path:

```
with torch.device("meta"): model = build_model(...)
apply_tensor_parallel(model, mesh)          # shards Linears with DTensor
apply_expert_parallel(model, mesh)
apply_float8(model)
apply_ac(model, mode)
apply_fsdp2(model, mesh, mp_policy)
model.to_empty(device=device)               # materialize on GPU
model.init_weights_and_freqs()              # init RoPE + weights
model.to(dtype=param_dtype)                 # cast to bf16
```

The meta-device init is required when TP is active: parameters must
already be DTensors before they're materialized, so FSDP2 can wrap
them correctly. For non-TP runs, meta-device init is skipped — the
model is built directly on the target device.

See the full breakdown in [Parallelism order](../architecture/parallelism-order.md).

## Fused optimizer compatibility

When TP or EP are active, `apply_fsdp2` always runs with a `dp_shard`
dimension in the mesh — even if that dimension has size 1. Why: the
fused AdamW kernel requires every parameter to be a DTensor (not a
mix of DTensor and plain tensor). If TP has already wrapped some
Linears as DTensors, FSDP2 has to upgrade the rest — which requires
an (even trivial) `dp_shard` axis to wrap along.

This is handled automatically in `init_distributed` — see
[Device mesh § Special case](device-mesh.md#dimension-order).

## See also

- [Device mesh](device-mesh.md) — how the mesh that `apply_fsdp2`
  consumes is built.
- [Tensor parallelism](tensor-parallelism.md) — what `apply_tensor_parallel`
  does before FSDP2 runs.
- [Expert parallelism § EP-MoE + FSDP2](expert-parallelism.md) — the
  deadlock case that per-sub-module wrapping avoids.
- [Parallelism order](../architecture/parallelism-order.md) — the
  full sequence of passes with reasoning.
- [Gradient utilities § `clip_grad_norm_`](../training/gradient-utilities.md#clip_grad_norm_) —
  the DTensor-aware clipping FSDP2-sharded models need.
