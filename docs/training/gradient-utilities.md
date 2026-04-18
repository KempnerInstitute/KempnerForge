# Gradient utilities

Two primitives govern how gradients move through the training loop:
`maybe_no_sync` controls when they get reduced, `clip_grad_norm_`
controls their magnitude.

## `maybe_no_sync`

```python
from kempnerforge.training.grad import maybe_no_sync

for micro_step in range(grad_accum_steps):
    with maybe_no_sync(model, micro_step, grad_accum_steps):
        logits = model(input_ids, doc_ids=doc_ids)
        loss = loss_fn(logits, labels)
        (loss / grad_accum_steps).backward()
```

Context manager from
[`kempnerforge/training/grad.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/training/grad.py).
On all microbatches except the last:

```python
model.set_requires_gradient_sync(False)
try: yield
finally: model.set_requires_gradient_sync(True)
```

Why:

- FSDP2 fires a reduce-scatter collective at the end of every backward
  pass. Without `maybe_no_sync`, a step with `grad_accum_steps = 8`
  fires **8 collectives per optimizer step**. With it, the first 7
  accumulate locally and the 8th fires one collective that covers all
  accumulated gradients.
- The method is FSDP2-specific (`set_requires_gradient_sync`). On
  non-FSDP models (DDP, single GPU), `hasattr(model,
  "set_requires_gradient_sync")` is `False` and the context is a
  no-op — safe to use unconditionally.
- On the last microbatch (`micro_step + 1 == grad_accum_steps`), the
  context skips straight to `yield` so the final backward triggers a
  normal reduce-scatter.

Not used under pipeline parallel — the PP schedule manages its own
sync internally. See
[Training loop § PP step](training-loop.md#pp-step-pp_enabled-is-true).

## `clip_grad_norm_`

```python
from kempnerforge.distributed import clip_grad_norm_

grad_norm = clip_grad_norm_(model, tc.grad_clip_norm)
```

Lives in
[`kempnerforge/distributed/utils.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/distributed/utils.py)
(re-exported from `kempnerforge.distributed`). Wraps
`torch.nn.utils.clip_grad_norm_` with an extra path for **mixed DTensor
meshes**.

The problem it solves: when a model combines TP and FSDP, some
parameters are DTensors on the `(dp_shard, tp)` mesh (TP-sharded
linears) and others are on `(dp_shard,)` alone (norm scales, biases).
PyTorch's `clip_grad_norm_` doesn't know how to combine gradients
living on different meshes — it would produce an incorrect norm.

Algorithm:

1. Collect all non-`None` `.grad` tensors from the model.
2. Build a `mesh_key` per gradient (DTensor's `_spec.mesh` id, or `0`
   for plain tensors).
3. **If only one mesh** (pure FSDP, single GPU, plain tensors): call
   `torch.nn.utils.clip_grad_norm_` directly — the fast path.
4. **If multiple meshes**: group gradients by mesh, compute per-group
   `sum-of-squares`, call `.full_tensor()` on each DTensor partial sum
   so the underlying all-reduce happens, then combine across groups
   with a plain `sqrt`.
5. Scale every gradient by `clip_coef = max_norm / (total_norm +
   1e-6)`, clamped to ≤ 1.

Returns the **pre-clip** total norm as a scalar tensor — the value you
log as `grad_norm` in metrics.

### `foreach` parameter

`clip_grad_norm_(..., foreach=True)` is the default — faster on CUDA
via the fused foreach implementation. Only touched if your model
contains some exotic parameter type that the foreach path can't
handle.

## Gradient accumulation in practice

The full microbatching pattern in `scripts/train.py` is:

```python
for micro_step in range(tc.grad_accum_steps):
    with maybe_no_sync(model, micro_step, tc.grad_accum_steps):
        logits = model(input_ids, doc_ids=doc_ids)
        loss   = loss_fn(logits, labels)
        scaled_loss = loss / tc.grad_accum_steps
        scaled_loss.backward()
        total_loss += loss.item()

grad_norm = clip_grad_norm_(model, tc.grad_clip_norm)
optimizer.step()
optimizer.zero_grad()
```

Two invariants:

- **Loss scaling**: `scaled_loss = loss / grad_accum_steps` keeps the
  effective LR independent of the accumulation factor. If you change
  `grad_accum_steps=4 → 8`, the optimizer step size stays the same
  per-token.
- **Clip after accumulate**: `clip_grad_norm_` runs **after** the
  microbatch loop, not inside it — the clip sees the final accumulated
  gradient.

## Config fields

```toml
[train]
grad_accum_steps = 8                    # microbatches per optimizer step
grad_clip_norm = 1.0                    # max grad L2 norm
```

Both live in
[`TrainConfig`](../configuration/config-sections.md).
`TrainConfig.__post_init__` requires `grad_clip_norm > 0` — there is no
"disable clipping" escape hatch; `1.0` is a near-free safety margin
and what every shipped config uses.

## See also

- [Training loop](training-loop.md) — where these utilities are called
  from.
- [Resilience § NaN detection](../resilience/index.md) — what happens
  when `clip_grad_norm_` returns a NaN or Inf.
- [Distributed § DeviceMesh](../distributed/index.md) — why gradients
  on different meshes happen and what the mesh structure looks like.
