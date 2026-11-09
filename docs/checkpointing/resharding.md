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

The catch: **parameter names must match**. If you rename a layer
between save and load, DCP won't find the saved shard and will
either error or silently leave the tensor uninitialized (it warns).

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
| Add a new parameter | yes | Missing-from-shard → warned, default-initialized |
| Rename a parameter | no | Old name not in `.metadata` → error (or you patch the state dict keys before `load_state_dict`) |

The table is about model and optimizer state, which is all DCP reshards. The
dataloader cursor and the RNG streams are rank-local and live outside DCP, so
any row above that moves a rank onto a different data partition (a changed
world size, `dp_shard`, `dp_replicate`, `tp` or `ep`) needs one extra config
line on resume: see
[Scaling a run to a different world size](#scaling-a-run-to-a-different-world-size).

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
- **Missing parameter**: the saver never wrote a tensor for
  `layers.42.mlp.some_new_module` because that module wasn't in the
  model yet. DCP warns; the tensor stays default-initialized. Fine
  when you've added a new head or adapter; bad when you expected
  transfer.
- **Extra parameter**: the saver wrote a tensor the loader doesn't
  want (you removed a module). DCP ignores the extra shard —
  silently, so watch the checkpoint size rather than relying on
  warnings.
- **`.metadata` disagreement between `pp{i}/` subdirs**: always a
  bug — different PP stages should save independently. File an
  issue.
- **"checkpoint per-rank state was saved with world_size=..."**: not
  a DCP failure at all. The model and optimizer reshard fine; the
  dataloader and RNG state cannot follow the topology change. See
  [Scaling a run to a different world size](#scaling-a-run-to-a-different-world-size).

## Scaling a run to a different world size

DCP reshards model and optimizer state freely, but the dataloader
cursor and the RNG streams are not DCP state. They are rank-local:
`train_state.pt` holds one payload per rank
([Train state](train-state.md#per-rank-state-schema-v1)), and a
payload is only meaningful on the rank and data partition that saved
it. An auto-resume is an *exact* resume, so it refuses to start when
the topology moved:

- a different world size, or
- a different data-parallel layout at the same world size, which is
  what changing `dp_shard`, `dp_replicate`, `tp` or `ep` does: with
  `dp_shard=4` rank 2 reads data partition 2 of 4, and with
  `dp_shard=2`, `tp=2` it reads partition 1 of 2.

Both raise on every rank, with an error naming the saved topology,
the current one, and the setting below. The refusal is deliberate:
restoring rank 0's single copy on every rank instead would move every
other rank's data position silently, with a loss curve that looks
entirely normal and nothing in the logs to say that data was re-read
or skipped.

To scale anyway, add one line to the config:

```toml
[checkpoint]
ignore_rank_state_mismatch = true
```

or pass `--checkpoint.ignore_rank_state_mismatch=true` on the command
line. Every rank then discards the saved per-rank dataloader and RNG
state (rank 0 logs a warning saying so) and resumes from the rest of
the checkpoint:

| State | With the flag set |
|-------|-------------------|
| `step`, `tokens_seen` | continue |
| Model, optimizer, scheduler | continue, resharded by DCP onto the new topology |
| Dataloader position | restarts from the first sample |
| RNG streams | restart from the run's seeds |

Resharding model and optimizer state never needs the flag; it is
untouched by the per-rank state. Nor do the loads that are not an
exact resume: a `checkpoint.load_path` warm start and
`scripts/eval.py` ignore the per-rank state and never check the
topology, on any world size or layout.

Restarting the data is not the same as continuing it: a run that
scales mid-flight will re-read the samples it already consumed.
Exact data continuity across a change in the data-parallel size is
tracked separately in
[issue #314](https://github.com/KempnerInstitute/KempnerForge/issues/314).

### Example: scale-up from 16 to 64 GPUs

```bash
# Train at 16 GPUs
uv run torchrun --nproc_per_node=16 scripts/train.py configs/train/7b.toml \
    --train.max_steps=5000 --checkpoint.dir=/checkpoints/7b_run

# Bump to 64 GPUs and resume
uv run torchrun --nproc_per_node=64 scripts/train.py configs/train/7b.toml \
    --train.max_steps=20000 --checkpoint.dir=/checkpoints/7b_run \
    --checkpoint.ignore_rank_state_mismatch=true
```

Auto-resume follows the `latest` symlink, `CheckpointManager.load`
calls `dcp.load`, and DCP reshards the 16-way sharded state into
64-way. Training continues from step 5000 with the model, optimizer
and scheduler intact; the data order and the RNG streams restart,
because the 64-rank job's data partitions are not the ones the
16-rank job saved cursors for. Without that last flag the resume
stops instead, naming the saved and current world sizes.

The other thing that changes is the effective batch size: 4× the DP
ranks means 4× the batch per step. Adjust `grad_accum_steps` downward
if you need to keep the global batch size constant.

## See also

- [DCP model + optimizer](dcp-model.md) — what's in each shard and
  why `state_dict()` has to describe the target shape on load.
- [Auto-resume](auto-resume.md) — how the training script chooses
  which checkpoint to reshard.
- [HF conversion](hf-conversion.md) — the escape hatch for
  re-emitting a checkpoint with different PP.
- [Device mesh](../distributed/device-mesh.md) — the mesh construction
  that determines the target shard layout.
