# Resharding

The reason KempnerForge uses DCP instead of plain `torch.save` is
**automatic resharding**: a checkpoint written at 32 GPUs can be
loaded at 64 GPUs (or 16, or 8, or 1) without any conversion step.

No code in `CheckpointManager` does this — DCP handles it entirely.
What KempnerForge contributes is a checkpoint layout that survives
the trip.

## How it works

Each `.distcp` shard contains **local tensor data** plus coordinates
— which slice of which full parameter this shard represents. The
`.metadata` file is a global index mapping parameter name → list of
shards. On load:

1. The loader's state dict gives DCP the target shape: e.g.
   `layers.0.attention.q_proj.weight` is a `DTensor(Shard(0))` on a
   `dp_shard=16` mesh.
2. DCP reads `.metadata` to find every shard contributing to that
   parameter.
3. DCP assembles the right slice for each loader rank, doing
   all-to-alls / point-to-points internally. Ranks that owned
   non-overlapping slices exchange data; ranks with overlapping
   slices pick the right piece.
4. The loader's tensor is mutated in place.

The catch: **parameter names must match**. The load is strict: every
tensor the model holds must be in the checkpoint under the same name.
Rename a layer or add one between save and load, and DCP finds no
saved shard for it and the load fails with `Missing key in checkpoint
state_dict`, naming the tensor.

## What it lets you change

| Across save/load | Works | Caveat |
|------------------|:-----:|--------|
| `dp_shard` | yes | Any factor of the parameter shape |
| `dp_replicate` | yes | With a `dp_shard` dimension it changes how many ranks hold each shard; without one, FSDP2 shards over it, so it reshards like `dp_shard` |
| `tp` | yes | Heads must still divide evenly |
| `ep` | yes | `num_experts` still must divide evenly |
| World size | yes | Pure rebalance across fewer/more ranks |
| **`pp` stage count** | **no** | Per-stage `pp{N}/` subdirs; changing `pp` reassigns layers and moves the embedding/output to different stages |
| Precision (bf16 → fp32) | yes | DCP upcasts on load |
| Add a new parameter | no | Not in `.metadata` → `Missing key in checkpoint state_dict`; see Missing parameter below |
| Rename a parameter | no | Old name not in `.metadata` → error (or you patch the state dict keys before `load_state_dict`) |

## PP — the one special case

PP stores shards per-stage under `pp0/`, `pp1/`, …. A checkpoint saved
at `pp=4` is **not loadable at `pp=2`** directly — DCP has no way to
reassemble the stage boundaries. You have one of two paths:

1. **Convert first.** Use
   [`scripts/convert_checkpoint.py dcp-to-hf`](hf-conversion.md) to
   flatten the per-stage shards into a single HF state dict, then
   `hf-to-dcp` to re-emit a single-stage DCP checkpoint. This works
   because the HF conversion reads every `pp{i}/` subdirectory and
   merges them. The round trip keeps the weights only: the result has
   no optimizer state and no `train_state.pt`, so a run that loads it
   starts at step 0 with a fresh optimizer, LR schedule and data
   position.
2. **Keep the same `pp` on resume.** Auto-resume never changes the
   parallelism shape, so a straight restart always works.

The conversion path is what shipped configs use in practice: PP is
chosen for the largest runs where memory forces it, and those runs
typically don't need to rebalance mid-training.

## What to check if resharding fails

- **Tensor shape mismatch**: the most common cause is that the
  model config changed between save and load (e.g. `n_heads`
  changed, `dim` changed, `vocab_size` changed). DCP errors with
  the target vs saved shape. Diff the configs.
- **Missing parameter** (`Missing key in checkpoint state_dict:
  model.layers.42.mlp.some_new_module.weight`): the saver never wrote
  that tensor because the module wasn't in the model yet. Neither a
  resume nor a `checkpoint.load_path` warm start can add it:
  `exclude_from_loading` skips the whole model or the whole optimizer,
  not single tensors. For a change `ModelConfig` can express (more
  layers, for example), the weights-only round trip in
  [HF conversion](hf-conversion.md) carries the old weights over:
  `dcp-to-hf` with the old config, then `hf-to-dcp` with the new one,
  which leaves the new parameters at their initial values with a
  warning. The export casts the weights to the export dtype, bfloat16
  unless you choose another (see [Dtypes](hf-conversion.md#dtypes)),
  and the result has no optimizer state or `train_state.pt`, so the
  run starts at step 0.
- **Extra parameter**: the saver wrote a tensor the loader doesn't
  want (you removed a module). DCP ignores the extra shard —
  silently, so watch the checkpoint size rather than relying on
  warnings.
- **`.metadata` disagreement between `pp{i}/` subdirs**: always a
  bug — different PP stages should save independently. File an
  issue.

## Example: scale-up from 16 to 64 GPUs

```bash
# Train at 16 GPUs
uv run torchrun --nproc_per_node=16 scripts/train.py configs/train/7b.toml \
    --train.max_steps=5000 --checkpoint.dir=/checkpoints/7b_run

# Bump to 64 GPUs and resume
uv run torchrun --nproc_per_node=64 scripts/train.py configs/train/7b.toml \
    --train.max_steps=20000 --checkpoint.dir=/checkpoints/7b_run
```

Auto-resume follows the `latest` symlink, `CheckpointManager.load`
calls `dcp.load`, DCP reshards the 16-way sharded state into 64-way,
training continues from step 5000. No manual intervention.

The one thing that changes: effective batch size — 4× the DP ranks
means 4× the batch per step. Adjust `grad_accum_steps` downward if
you need to keep the global batch size constant.

## See also

- [DCP model + optimizer](dcp-model.md) — what's in each shard and
  why `state_dict()` has to describe the target shape on load.
- [Auto-resume](auto-resume.md) — how the training script chooses
  which checkpoint to reshard.
- [HF conversion](hf-conversion.md) — the escape hatch for
  re-emitting a checkpoint with different PP.
- [Device mesh](../distributed/device-mesh.md) — the mesh construction
  that determines the target shard layout.
