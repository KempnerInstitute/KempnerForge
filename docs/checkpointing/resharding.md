# Resharding

The reason KempnerForge uses DCP instead of plain `torch.save` is
**automatic resharding**: a checkpoint written at 32 GPUs can be
loaded at 64 GPUs (or 16, or 8, or 1) without any conversion step.

No code in `CheckpointManager` does this — DCP handles it entirely.
What KempnerForge contributes is a checkpoint layout that survives
the trip. Under expert parallelism that takes one step before DCP:
see [EP](#ep-every-expert-under-its-global-index).

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
| `ep` | yes | `num_experts` must still divide evenly; see [EP](#ep-every-expert-under-its-global-index) |
| World size | yes | Pure rebalance across fewer/more ranks |
| **`pp` stage count** | **no** | Per-stage `pp{N}/` subdirs; changing `pp` reassigns layers and moves the embedding/output to different stages. The weights move by conversion, and a PP run loads a checkpoint saved without PP; see [PP](#pp-the-one-special-case) |
| Precision (bf16 → fp32) | yes | DCP upcasts on load |
| Add a new parameter | no | Not in `.metadata` → `Missing key in checkpoint state_dict`; see Missing parameter below |
| Rename a parameter | no | Old name not in `.metadata` → error (or you patch the state dict keys before `load_state_dict`) |

The table is about model and optimizer state, which is all DCP reshards. The
dataloader position and the RNG streams live outside DCP and are saved per
rank, so any row above that moves a rank onto a different data partition (a
changed world size, `dp_shard`, `dp_replicate`, `tp` or `ep`) needs one extra
config line on resume: see
[Scaling a run to a different world size](#scaling-a-run-to-a-different-world-size).

## PP: the one special case

PP stores shards per-stage under `pp0/`, `pp1/`, …, and records the
degree as `pp` in `metadata.json`. Each stage loads only its own
`pp{N}/` subdirectory, and a different `pp` gives each stage different
layers, so a stage checkpoint loads only at the `pp` it was saved
with. Auto-resume never changes the parallelism shape, so a straight
restart always works. Every rank loads the checkpoint as rank 0 reads
its layout, and before reading any shard, every rank refuses with a
`ValueError` naming both layouts:

- a checkpoint saved at another `pp`, for example `pp=4` loaded at
  `pp=2`. One saved before the degree was recorded loads when it holds
  as many stage directories as the run has stages, and is refused
  otherwise;
- a stage checkpoint loaded without PP, by a run or by
  `scripts/eval.py`.

A PP run loads a checkpoint saved without PP, such as the output of
`hf-to-dcp`: each stage loads its own layers from the one DCP index,
through `checkpoint.load_path` or auto-resume. Of its DCP state a
stage loads the model weights alone: the saved optimizer param groups
are the whole model's, not one stage's, so a load that would restore
the optimizer state such a checkpoint holds is refused, also on
auto-resume. To continue from its
weights with a fresh optimizer, set `checkpoint.load_path` to it with
`checkpoint.exclude_from_loading = ["optimizer"]` (add
`"train_state"` to also restart the step, schedule and data
position), in a `checkpoint.dir` that holds no checkpoint yet.

To change `pp`, or to load a PP checkpoint without PP, carry the
weights over: `dcp-to-hf` exports the stages as one model (see
[HF conversion § With PP](hf-conversion.md#with-pp)), `hf-to-dcp`
imports the export as a checkpoint saved without PP, and
`checkpoint.load_path` points the new run at it:

```bash
uv run python scripts/convert_checkpoint.py dcp-to-hf \
    --dcp-dir checkpoints/run/step_1000 --hf-dir exports/run \
    --config <the run's TOML> --dtype float32
uv run python scripts/convert_checkpoint.py hf-to-dcp \
    --hf-dir exports/run --dcp-dir checkpoints/run_flat \
    --config <the run's TOML>
```

The new run starts at step 0 from those weights, with a fresh
optimizer, as after any `hf-to-dcp` import.

## EP: every expert under its global index

With `ep > 1` each rank holds `num_experts / ep` experts of every MoE
layer, numbered from 0 in its own state dict, so every EP rank would
give DCP the same names and offsets, and DCP keeps one copy of each.
`CheckpointManager` therefore places each rank's experts among all of
them before DCP sees them: an unpacked expert is saved under its global
index, and each packed weight as one tensor of `num_experts` rows. The
checkpoint is the one an `ep=1` run writes, so DCP reshards it to any
`ep` that divides `num_experts`, including `ep=1`, and an `ep=1`
checkpoint loads into an EP run. Packed experts under FSDP need torch
2.10 or newer for this. See
[Expert parallelism § Checkpoints](../distributed/expert-parallelism.md#checkpoints).

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
  not single tensors. For a change `ModelConfig` can express that adds
  tensors without resizing existing ones (more layers, for example),
  with or without PP (see [PP](#pp-the-one-special-case)), the
  weights-only round trip in [HF conversion](hf-conversion.md) carries
  the old weights over:
  `dcp-to-hf` with the old config, then `hf-to-dcp` with the new one,
  which leaves the new parameters at their initial values with a
  warning. The export casts the weights to the export dtype, bfloat16
  unless you choose another (see [Dtypes](hf-conversion.md#dtypes)),
  and the result has no optimizer state or `train_state.pt`, so the
  run starts at step 0. A change that also resizes an existing tensor,
  such as more experts (the router grows) or a larger vocabulary, fails
  in `hf-to-dcp` with a size mismatch. A model that Llama or Mixtral
  cannot hold needs `--no-architecture-check` on `dcp-to-hf` (see
  [HF conversion](hf-conversion.md#what-transformers-can-load)).
- **Extra parameter**: the saver wrote a tensor the loader doesn't
  want (you removed a module). DCP ignores the extra shard —
  silently, so watch the checkpoint size rather than relying on
  warnings.
- **A `pp{i}/` subdir without its own `.metadata`**: each stage saves
  over its own process group and indexes only its own shards, so the
  stages' indexes differ by design. A save coordinated over every rank
  writes one `.metadata` into `pp0/` instead, and the other stages
  cannot load (see
  [DCP model § With pipeline parallelism](dcp-model.md#with-pipeline-parallelism)).
  That is a bug; file an issue.
- **An EP checkpoint saved before experts had a global index**
  ([#202](https://github.com/KempnerInstitute/KempnerForge/issues/202)):
  it holds `num_experts / ep` experts per layer, one rank's, and the
  others were never written. The load refuses it and says how many
  experts it holds; no `ep` can restore the missing ones.
- **"checkpoint per-rank state was saved with world_size=..."**: not
  a DCP failure at all. The model and optimizer reshard fine; it is
  each rank's saved RNG and loader state that belongs to the saved
  topology. See
  [Scaling a run to a different world size](#scaling-a-run-to-a-different-world-size).

## Scaling a run to a different world size

DCP reshards model and optimizer state freely, but the dataloader
position and the RNG streams are not DCP state. `train_state.pt` holds
one payload of them per rank
([Train state](train-state.md#per-rank-state-schema-v1)), and an
auto-resume is an *exact* resume: each rank restores the payload its
own rank saved, its RNG streams and its loader's state. Those payloads
fit only the topology that saved them, so an auto-resume refuses to
start when the topology moved:

- a different world size, or
- a different data-parallel layout at the same world size, which is
  what changing `dp_shard`, `dp_replicate`, `tp` or `ep` does: with
  `dp_shard=4` rank 2 reads data partition 2 of 4, and with
  `dp_shard=2`, `tp=2` it reads partition 1 of 2.

Both raise on every rank, before any rank restores loader or RNG
state, with an error naming the saved topology, the current one, and
the setting below. The refusal is deliberate: an exact resume
continues every rank's random streams as they were, and on another
topology no rank has saved streams of its own to continue.

To scale anyway, add one line to the config:

```toml
[checkpoint]
ignore_rank_state_mismatch = true
```

or pass `--checkpoint.ignore_rank_state_mismatch=true` on the command
line. Every rank then resumes from the rest of the checkpoint, keeps
the data position and drops its saved RNG state, and a warning names
the mismatch, what was kept and what was dropped:

| State | With the flag set |
|-------|-------------------|
| `step`, `tokens_seen` | continue |
| Model, optimizer, scheduler | continue, resharded by DCP onto the new topology |
| Dataloader position | continues where it carries over (below), from the saved rank 0's position, or for a stream on the saved data-parallel size from each data-parallel rank's own; otherwise restarts from the first sample on every rank, with a warning |
| RNG streams | restart from the run's seeds |

The data position kept is the one the saved rank 0 reached, except a
stream's (below), and every rank's loader applies it by the loader's
own rule
([Stateful dataloader § Resuming on another data-parallel size](../data/stateful-dataloader.md#resuming-on-another-data-parallel-size)).
Every rank checks the position before any rank applies it, and the
ranks agree: they all keep it, or all restart the data, also where one
rank's view of the data, such as a data file's size, differs.
A single map-style dataset's position (`data.dataset_path`, an
in-memory Hugging Face dataset or a VLM dataset) counts the samples of
the epoch's order that all data-parallel ranks consumed, so it carries
over to any data-parallel size and batch size: each rank resumes at
its first sample past that count, and no sample is read twice. To keep
the ranks in step, up to one sample per rank at the end of that epoch
goes unread, beside the partial batch `drop_last` drops (see
[Stateful dataloader § Resuming on another data-parallel size](../data/stateful-dataloader.md#resuming-on-another-data-parallel-size)).
Three positions carry over only on the saved data-parallel
size: a weighted mixture (`data.datasets`), a streamed dataset
(`data.hf_streaming`), and a checkpoint saved before the loader
recorded its batch size. On another data-parallel size their data
restarts from the first sample, with a warning that names the reason,
and the resume goes on, since the flag opted in. A stream resumed this
way on the saved data-parallel size continues each data-parallel rank
from the count that rank saved, so it resumes exactly in every pass
([Stateful dataloader § Several ranks](../data/stateful-dataloader.md#several-ranks)).
The data order does not depend on the RNG streams: the samplers and
the stream shuffle with the data seed and the epoch.

Two cases keep no data position. A loader other than
`StatefulDataLoader` and `StreamingDataLoader`, passed to
`CheckpointManager` directly, does not check that rank 0's state fits
its rank, so that state is dropped with a warning. An envelope that the
flag lets through for another reason, unreadable or incomplete, keeps
neither the data position nor the RNG state.

A checkpoint saved before per-rank state was recorded holds no envelope
to compare, so it cannot show whether the topology changed. Without the
flag every rank applies rank 0's data position and RNG streams, and a
position that cannot carry over stops the resume with a
`DataPositionError`; with the flag the resume goes on as if the topology
changed, even on the topology that saved it, with rank 0's data
position by the rule above and fresh RNG streams.

Resharding model and optimizer state never needs the flag; it is
untouched by the per-rank state. Nor do the loads that are not an
exact resume: a `checkpoint.load_path` warm start and
`scripts/eval.py` ignore the per-rank state and never check the
topology, on any world size or layout. A warm start applies rank 0's
data position by the same rule and restores rank 0's RNG streams on
every rank.

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
and scheduler intact, and the data continues from the same sample:
the saved position counts the samples of the epoch's order that the
16 ranks consumed, and each of the 64 ranks resumes at its first
sample past that count. The RNG streams restart from the run's seeds,
since each of the 16 ranks saved streams of its own. With a weighted
mixture, a streamed dataset or a checkpoint saved before the loader
recorded its batch size, the data restarts from its first sample, with
a warning. Without that last flag the resume stops instead, naming the
saved and current world sizes.

The other thing that changes is the effective batch size: 4× the DP
ranks means 4× the batch per step. Adjust `grad_accum_steps` downward
if you need to keep the global batch size constant.

## See also

- [DCP model + optimizer](dcp-model.md): what's in each shard and
  why `state_dict()` has to describe the target shape on load.
- [Auto-resume](auto-resume.md): how the training script chooses
  which checkpoint to reshard.
- [HF conversion](hf-conversion.md): the weights-only round trip
  that adds parameters or changes `pp`.
- [Device mesh](../distributed/device-mesh.md): the mesh construction
  that determines the target shard layout.
