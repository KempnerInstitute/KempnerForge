# Pipeline parallelism

Pipeline parallelism (PP) splits the Transformer across stages so each
rank in the `pp` mesh dimension holds a contiguous chunk of layers —
plus the embedding on stage 0 and the final norm + output head on the
last stage. Activations flow forward through the pipeline in
microbatches; `torch.distributed.pipelining` handles the schedule.

Entry points, all in
[`kempnerforge/distributed/pipeline_parallel.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/distributed/pipeline_parallel.py):

- `compute_layer_assignment(n_layers, pp_size)` — list of
  `(start, end)` tuples, one per stage.
- `build_stage_module(config, pp_rank, pp_size)` — instantiates a
  `PipelineStageModule` with only the params for this rank.
- `build_pipeline_stage(stage_module, ...)` — wraps it in
  `torch.distributed.pipelining.PipelineStage`.
- `build_pipeline_schedule(stage, n_microbatches, loss_fn, schedule)`
  — creates `Schedule1F1B`, `ScheduleGPipe`, or
  `ScheduleInterleaved1F1B`.

## Stage layout

Given `n_layers` and `pp_size`,
`compute_layer_assignment` distributes layers evenly. Remainder layers
go to the **earlier** stages:

```python
base = n_layers // pp_size
remainder = n_layers % pp_size
# stage i gets base + 1 layers if i < remainder else base
```

For Llama-3 70B (80 layers) with `pp=4`: stages 0-3 each get 20
layers. For Llama-3 8B (32 layers) with `pp=3`: stage 0 and 1 get
11 each, stage 2 gets 10.

### What each stage holds

| Stage | Token embed | Transformer blocks | Final norm | Output head |
|-------|:-----------:|:------------------:|:----------:|:-----------:|
| 0 (first) | yes | `layers[0:k]` | no | no |
| middle | no | `layers[k:m]` | no | no |
| N-1 (last) | no | `layers[m:n_layers]` | yes | yes |

The layer keys in `PipelineStageModule.layers` match the full
`Transformer` — stage 1 has keys `"11", "12", ..."21"`, not
`"0", "1", ...`. This keeps the DCP checkpoint layout
consistent across PP sizes, so you can checkpoint from a `pp=4` run
and resume with `pp=2`.

## Application order with PP

The non-PP path goes through `build_parallel_model`. PP has its own
path in `scripts/train.py`:

```python
# PP + TP
with torch.device("meta"):
    stage_mod = build_stage_module(config.model, pp_rank, pp_size)
model = stage_mod
apply_tensor_parallel(model, device_mesh)
if tc.is_fp8:
    apply_float8(model)
apply_ac(model, tc.activation_checkpointing)
apply_fsdp2(model, device_mesh, mp_policy=mp_policy)
model.to_empty(device=device)
model.init_weights_and_freqs()

# PP without TP
stage_mod = build_stage_module(config.model, pp_rank, pp_size)
model = stage_mod.to(device=device, dtype=tc.param_dtype)
if tc.is_fp8:
    apply_float8(model)
apply_ac(model, tc.activation_checkpointing)
apply_fsdp2(model, device_mesh, mp_policy=mp_policy)
```

After that, `build_pipeline_stage` wraps the model in a
`PipelineStage` (using a zero-filled example input for shape
inference — token IDs on stage 0, hidden states elsewhere), and
`build_pipeline_schedule` picks the 1F1B / GPipe / interleaved
schedule.

## Schedules

`pp_schedule` is a `DistributedConfig` enum with three values:

| Value | Class | Behavior |
|-------|-------|----------|
| `"1f1b"` (default) | `Schedule1F1B` | Warmup fills the pipe with `pp_size` forwards, then alternates `1F, 1B, 1F, 1B, ...` — steady-state memory is `pp_size` activations, same as GPipe but lower peak |
| `"gpipe"` | `ScheduleGPipe` | All forwards first, then all backwards — simpler, higher peak activation memory |
| `"interleaved_1f1b"` | `ScheduleInterleaved1F1B` | Virtual pipeline stages: each rank holds **multiple** non-contiguous chunks. Reduces pipeline bubble at the cost of extra communication |

Interleaved 1F1B requires passing a list of stages to
`build_pipeline_schedule` — the builder currently raises if you pass
a single stage:

```python
if schedule == "interleaved_1f1b":
    if not isinstance(stage, (list, tuple)):
        raise ValueError(
            "interleaved_1f1b schedule requires a list of PipelineStage objects"
        )
```

The virtual-stage construction isn't wired through the rest of
`train.py` yet — if you want interleaved, you have to build stage
list manually. 1F1B is the default for shipped configs.

## `n_microbatches`

`build_pipeline_schedule` takes `n_microbatches=tc.grad_accum_steps`
— gradient accumulation and PP microbatches are the same number.
Every value in a step's `grad_accum_steps` corresponds to one
microbatch entering the pipeline.

**Constraint**: `n_microbatches >= pp_size` for 1F1B to fill the
pipeline. With `n_microbatches = pp_size`, there is no steady-state
phase (only warmup and drain) and the bubble overhead is maximal;
`2x` or `4x` is the practical range for amortizing the bubble.

## Training step under PP

The PP branch in
[`scripts/train.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/scripts/train.py)
looks very different from the non-PP step:

```python
full_input  = torch.cat(input_ids_list, dim=0)    # concat microbatches
full_labels = torch.cat(labels_list, dim=0)

if is_first:
    pp_schedule.step(full_input, target=full_labels, losses=pp_losses)
elif is_last:
    pp_schedule.step(target=full_labels, losses=pp_losses)
else:
    pp_schedule.step()

# Loss lives only on the last stage
avg_loss = sum(l.item() for l in pp_losses) / len(pp_losses) if is_last else 0.0

# Broadcast loss + grad_norm to every PP stage for consistent logging
dist.broadcast(loss_tensor, group_src=pp_size - 1, group=pp_group)
```

`schedule.step()` handles forward, backward, and gradient accumulation
across all microbatches internally. The training loop skips
`maybe_no_sync`, `loss_fn`, and the manual microbatch loop — the
schedule owns all of that. See
[Training loop § PP step](../training/training-loop.md).

## FSDP2 interaction

When PP is on, the training loop calls `apply_fsdp2(model, device_mesh,
mp_policy=mp_policy)` without overriding `reshard_after_forward`, so
FSDP2 uses its default (`True`).

That's not ideal: 1F1B sends many microbatches through each stage, and
resharding between them triggers a fresh all-gather every
microbatch. The docstring on `pipeline_parallel.py` recommends
`reshard_after_forward=False` for PP to amortize the all-gather over
`n_microbatches`, but the current `scripts/train.py` doesn't thread
the flag through. If you're running PP at scale and can't fit the
extra all-gathers, pass it manually. See [FSDP2](fsdp2.md) — the
**`reshard_after_forward`** section.

## Checkpointing

DCP needs a process group scoped to ranks that share the **same
parameters** — which is every non-PP mesh axis. `train.py` builds
this with:

```python
non_pp_dims = [d for d in device_mesh.mesh_dim_names if d != "pp"]
ckpt_pg = device_mesh[non_pp_dims[0]].get_group()  # 1D case
# or a flat subgroup for multi-dim DP/TP/EP meshes
```

Each stage then saves its own DCP shard subdirectory
(`checkpoints/step_N/stage_0/`, `stage_1/`, …) so different stages'
files don't collide. See [Checkpointing](../checkpointing/index.md).

## Config

```toml
[distributed]
pp = 4                                    # pipeline stages
pp_schedule = "1f1b"                      # "1f1b", "gpipe", "interleaved_1f1b"

[train]
grad_accum_steps = 16                     # = n_microbatches; must be >= pp
```

Validation checks (from
[`DistributedConfig`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/distributed.py)
and [JobConfig](../configuration/validation-rules.md)):

- `pp_size <= n_layers` — enforced in `compute_layer_assignment`.
- `pp > 1` is incompatible with `tie_embeddings=True` — the tied
  weights live on both stage 0 and the last stage, which PP can't
  reconcile. See [Tie embeddings vs pipeline parallel](../configuration/validation-rules.md#tie-embeddings-vs-pipeline-parallel).
- `pp > 1` + MoE requires the MoE layer to fit on one stage — see
  [Validation rules](../configuration/validation-rules.md) (the
  **MoE + PP** section).

## Example: `70b_32gpu_tp4_pp4.toml`

```
dp_shard=2, tp=4, pp=4
n_layers=80, grad_accum_steps=4

mesh: ("pp", "dp_shard", "tp") → (4, 2, 4)
```

- `pp=4` splits 80 layers into `[0:20]`, `[20:40]`, `[40:60]`,
  `[60:80]`.
- `tp=4` shards each stage's Linears within a single node.
- `dp_shard=2` doubles the throughput by splitting the batch
  across two pipeline replicas.
- `grad_accum_steps=4` → 4 microbatches per step. With `pp=4`, that
  exactly fills the 1F1B pipeline (no steady-state phase) — the
  minimum allowed. Bumping to `8` or `16` would amortize the bubble
  further at the cost of extra activation memory.

See [Parallelism recipes § 70B](../reference/parallelism-recipes.md).

## See also

- [Device mesh](device-mesh.md) — see the 70B TP+PP+FSDP example for
  how the `pp` sub-mesh is extracted.
- [FSDP2](fsdp2.md) — the **`reshard_after_forward`** section
  covers why PP sets it to `False`.
- [Training loop § PP step](../training/training-loop.md) — the
  `schedule.step()` path.
- [Checkpointing](../checkpointing/index.md) — per-stage DCP shards.
- [Validation rules](../configuration/validation-rules.md) (see
  **MoE + PP**) — MoE / PP compatibility constraint.
- [Parallelism recipes](../reference/parallelism-recipes.md) —
  which model sizes need PP.
