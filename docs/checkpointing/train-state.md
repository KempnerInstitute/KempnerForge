# Train state

DCP covers model and optimizer; everything else — scheduler, RNG,
training metadata, user extras — goes in a separate
`train_state.pt` file written by rank 0.

Entry points, both in
[`kempnerforge/checkpoint/state.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/checkpoint/state.py):

- `build_train_state(step, tokens_seen, scheduler, dataloader, extra)`
  — builds the save dict.
- `restore_train_state(state, scheduler, dataloader)` — reloads into
  the scheduler / dataloader, returns `(step, tokens_seen, extra)`.

## What's inside `train_state.pt`

```python
{
    "step":        int,                   # current training step
    "tokens_seen": int,                   # cumulative token count across DP ranks
    "rng": {                              # rank 0's RNG capture (see below)
        "python":     random.getstate(),
        "numpy":      np.random.get_state(),
        "torch_cpu":  torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(),   # if CUDA available
    },
    "scheduler":   scheduler.state_dict(),            # present if scheduler passed
    "scheduler_group_modules": [None, "adapter", ...],  # if a param group names a module
    "dataloader":  dataloader.state_dict(),           # rank 0's; present if dataloader passed
    "per_rank": {                         # schema v1: every rank's own state
        "version": 1,
        "world_size": W,
        "data_parallel": {0: (dp_rank, dp_size), ...},  # or None if unrecorded
        "ranks": {
            0: {"rng": {...}, "dataloader": {...}},   # global rank -> payload
            # ... one entry per global rank 0..W-1
        },
    },
    # ... plus whatever was in the `extra` dict
}
```

## Who writes it

```python
# manager.py, inside save() (abridged)
per_rank, gather_error = self._gather_rank_states(dataloader)  # every rank
if error is None:
    error = gather_error

if self._rank == 0 and error is None:
    train_state = build_train_state(
        step=step, tokens_seen=tokens_seen,
        scheduler=scheduler, dataloader=dataloader, extra=extra,
        per_rank=per_rank,
    )
    torch.save(train_state, ckpt_dir / "train_state.pt")
```

Only global rank 0 writes; the file is not sharded. Before the
write, every rank's own dataloader/RNG payload is gathered
collectively (on the manager's dedicated gloo group) into the
versioned `per_rank` envelope, so the single file still carries every
rank's state. Each rank captures and pickles its payload before the
gather. A rank whose capture fails (its loader's `state_dict()`
raises, or the payload cannot be pickled) still enters the gather,
without a payload, so the save fails on every rank instead of leaving
the others waiting: that rank raises its own error, every other rank
raises a `RuntimeError` naming it, rank 0 writes no `train_state.pt`,
and `latest` stays on the previous checkpoint.

On load, rank 0 reads the file and broadcasts it to every rank
via `dist.broadcast_object_list`; all ranks agree on `step` and
scheduler state. On an exact resume (the training entry's auto-resume
of the run's own checkpoint, `load(..., exact_resume=True)`) each rank
then restores **its own** entry from `per_rank["ranks"]`, keyed by
global rank. Any other load (a `load_path` warm start,
`scripts/eval.py`) restores the single-copy `rng`/`dataloader` keys
(rank 0's) on every rank with no topology check, exactly as before.
A weights-only warm start
(`exclude_from_loading = ["optimizer", "train_state"]`) never reads
the file, so no rank restores either copy or enters the broadcast or
the layout gather.

## Per-rank state (schema v1)

Dataloader cursors and RNG streams are rank-local: pipeline stages
seed `seed + pp_rank`, CUDA streams diverge during training, and a
stateful loader's position is its own rank's. The gather spans every
rank of every pipeline stage, so each stage resumes its own streams
rather than stage 0's. The `per_rank` envelope records each rank's
payload:

- `version`: schema version, currently `1`. A newer version than
  the running code understands refuses to load.
- `world_size`: the topology that saved the checkpoint. Resuming
  under a **different world size raises** with an actionable error:
  exact data/RNG continuity only exists on the saved topology, while
  model/optimizer resharding via DCP is unaffected.
- `data_parallel`: every global rank's `(dp_rank, dp_size)`, the data
  partition its loader read. The same world size can split into a
  different partition per rank (for example DP=4 vs DP=2 x TP=2), so
  resuming under a **different layout also raises**: a loader cursor
  is only valid on the partition it was saved on. `None` when the saving
  `CheckpointManager` was built without `dp_info` (the training entry
  always passes it); the layout check is then skipped and only the
  world size is checked.
- `ranks`: one payload per global rank, holding that rank's `rng`
  capture plus, when its loader is stateful, its `dataloader` state
  (on every rank or none).

The whole envelope is checked before any rank selects its entry: a
missing, extra or incomplete entry (no full `rng` capture, an RNG
stream such as `torch_cuda` or `dataloader` state on only some ranks)
refuses the resume. The check reads only what every rank shares (the
broadcast envelope, the world size, the gathered current layout), so
all ranks restore, refuse, or discard together. To proceed past a
refusal, set
`checkpoint.ignore_rank_state_mismatch = true`: the saved loader/RNG
state is discarded on every rank with a warning and data order /
random streams restart fresh (model, optimizer, scheduler and step
still resume). See
[Scaling a run to a different world size](resharding.md#scaling-a-run-to-a-different-world-size)
for the scale-up recipe and what it costs.

Compatibility:

- **Old checkpoint, new reader**: no `per_rank` key, so rank 0's
  single-copy `rng`/`dataloader` keys restore on every rank, exactly
  as before.
- **New checkpoint, old reader**: the single-copy keys (rank 0's)
  are still written beside the envelope, so pre-envelope readers
  keep loading. Their documented limitation is unchanged: they
  restore rank 0's copy on every rank and ignore `per_rank`.
- **Not a resume**: a `load_path` warm start or an evaluation load
  ignores `per_rank` and restores the single-copy keys (rank 0's) on
  every rank, exactly as before, on any world size or layout. A
  weights-only warm start skips the whole file, `per_rank` included:
  every rank keeps the RNG streams seeded at startup and starts its
  loader from the first sample, on any world size or layout.

## RNG capture

`get_rng_state()` snapshots **four** generators:

| Key | Generator | What it controls |
|-----|-----------|------------------|
| `python` | `random.getstate()` | Any use of the `random` module |
| `numpy` | `np.random.get_state()` | Data loader sampling seeds, augmentations |
| `torch_cpu` | `torch.random.get_rng_state()` | Dropout, random init on CPU |
| `torch_cuda` | `torch.cuda.get_rng_state()` | Any CUDA-side randomness (dropout on GPU) |

On load, `set_rng_state()` restores them all. Result: a resumed run
produces the **exact same loss trajectory** as an uninterrupted run,
step for step.

The one thing the RNG snapshot doesn't cover: multi-process
dataloader workers spawned by `torch.utils.data.DataLoader` have
their own RNG state. Their seeds are set deterministically at worker
init from the parent process's state, so they stay reproducible as
long as `num_workers` doesn't change.

## Scheduler state

Passed through verbatim: whatever `scheduler.state_dict()` returns.
For `LambdaLR` (the base of every registered scheduler — cosine,
linear, wsd, rex, constant) the dict contains the step count and
base LR, which together determine the next LR value. Restoring it
gets you back the same LR schedule from that step.

The base LR is kept per param group, in the order of the optimizer's
groups. When a group names a module under `"module"`, as
`build_optimizer` names the groups of
[`optimizer.lr_multipliers`](../training/optimizers.md#per-module-learning-rates),
`scheduler_group_modules` records the module of every group, None for
one outside the listed modules. A load into groups laid out otherwise,
because other parameters are trainable or the weight-decay scope
changed, carries the saved schedule on in every group, as a load into
the same groups does: each group gets the saved entries of its
module's groups, and a group of a module the state holds no group of
gets those of the first saved group, its rates scaled from that
group's multiplier to its own, so it runs at the saved peak times its
multiplier. Without a recorded module on either side, the state loads
as saved, unless the number of groups changed: a `LambdaLR` state is
then laid out the same way, its groups all of no module. Either way
the saved base LRs win over the ones the optimizer was built with, so
a changed `optimizer.lr` takes no effect; a load whose groups name a
module logs the peak of each module's groups, and warns when they
differ from the ones the optimizer was built with.

Pure `phase.lr_scale` overlays live in the training loop and are
recomputed from the current phase, so they don't need to be saved.

See [Training § Schedulers](../training/schedulers.md).

## Dataloader state

`StatefulDataLoader` exposes `state_dict()` / `load_state_dict()`, and the
training loop passes `dataloader=` at every save site, so `build_train_state`
records the loader position under the `dataloader` key:

```python
# training/loop.py::run_training_loop
ckpt_mgr.save(
    step=step,
    tokens_seen=tokens_seen,
    scheduler=scheduler,
    dataloader=data.dataloader,
    extra=ckpt_extra,
)
```

A checkpoint taken mid-epoch carries e.g.
`{'epoch': 7, 'batches_yielded': 14, 'sampler': {...}}`. On resume
`StatefulDataLoader.__iter__` re-applies the skip, so training continues at the
same sample boundary rather than restarting the epoch. Plain
`torch.utils.data.DataLoader` (the streaming and eval paths) has no
`state_dict`, so it is skipped and those loaders do restart.

## The `extra` dict

Anything that isn't step / tokens / RNG / scheduler / dataloader
can be threaded through `extra`:

```python
# training/loop.py::checkpoint_extra
extra = {"phase_idx": phases.next_idx} if phases.phases else {}
if config.metrics.wandb_run_id:
    extra["wandb_run_id"] = config.metrics.wandb_run_id
if config.metrics.mlflow_run_id:
    extra["mlflow_run_id"] = config.metrics.mlflow_run_id
...
return extra

# training/loop.py::run_training_loop
ckpt_extra = checkpoint_extra(config, step, phases)
ckpt_mgr.save(step=step, tokens_seen=tokens_seen, scheduler=scheduler,
              dataloader=data.dataloader, extra=ckpt_extra)
```

On load, `restore_train_state` strips out the "standard" keys
(`step`, `tokens_seen`, `rng`, `scheduler`, `scheduler_group_modules`,
`dataloader`, `per_rank`) and returns the rest as the third tuple element:

```python
step, tokens_seen, extra = ckpt_mgr.load(path=..., scheduler=scheduler)
if extra.get("wandb_run_id"):
    config.metrics.wandb_run_id = extra["wandb_run_id"]
if extra.get("mlflow_run_id"):
    config.metrics.mlflow_run_id = extra["mlflow_run_id"]
```

KempnerForge uses this for:

- `phase_idx` — current curriculum phase (so training resumes the
  correct data mix without skipping forward).
- `wandb_run_id` — so resumed runs append to the same W&B run rather
  than starting a new one.
- `mlflow_run_id`: the same for MLflow; on resume, steps up to the last
  `train/loss` step the run already holds are not logged again (best
  effort: if reading the run's history fails, nothing is skipped). The
  checkpoint's `metadata.json` also records the live run as plain JSON,
  for tools that read it without loading this file.

Custom training workflows can use it for anything
`torch.save`-serializable.

## Save failures

`torch.save` uses `pickle` under the hood. If any object in the
train state isn't picklable (an `extra` value, or any rank's
dataloader state), the save raises on every rank: the DCP shard is
already on disk, but the non-distributed state file is missing and
`latest` does not move to it.

Practical implication: keep `extra=` values and a custom loader's
`state_dict()` to primitives, lists, dicts, and tensors. Don't stuff
a live logger, process group, or generator in there.

## When `train_state.pt` is missing

`save()` writes `train_state.pt` with the DCP shards (right after them in
a synchronous save, while an async save's shards are still flushing), and
moves `latest` only once both are written, so every checkpoint it
completes has one. A checkpoint whose shards hold
optimizer state but that has no `train_state.pt` was cut short between
the two writes, or lost the file afterwards. Resuming it would continue
its weights and optimizer moments from step 0, with the LR schedule
restarting its warmup and the data stream starting over. So:

- `CheckpointManager.load` raises `FileNotFoundError`, naming the
  directory, before it loads anything, whenever the load would restore
  that optimizer state: on auto-resume through `latest`, and from
  `checkpoint.load_path`. Rank 0 checks for the file and broadcasts its
  answer, so every rank raises together.
- The `step_N` scan of [auto-resume](auto-resume.md) skips such a
  directory with a warning and takes the newest complete one. When none
  is complete, training starts from step 0 (or from
  `checkpoint.load_path`, if set), which it reports at ERROR level.
  `scripts/eval.py` loads the weights alone, so its scan keeps such a
  directory (`require_train_state=False`).
- To start from its weights alone, warm-start a new run from it: a new
  `checkpoint.dir`, `checkpoint.load_path` pointing at it, and
  `checkpoint.exclude_from_loading = ["optimizer"]`.

A checkpoint whose shards hold no optimizer state needs no
`train_state.pt`: an `hf-to-dcp` import, for example. It loads at step
0 with a fresh optimizer. A load that excludes the optimizer is not
refused either (`scripts/eval.py` passes `exclude_keys=["optimizer"]`):
without `train_state.pt` it loads the weights alone at step 0. Either
way an INFO line says that the step counter, token count, LR schedule,
data position and RNG state start fresh.

## See also

- [DCP model + optimizer](dcp-model.md) — what DCP handles that
  train state does not.
- [Auto-resume](auto-resume.md) — how the training loop reads
  `train_state.pt` on startup.
- [Training § Schedulers](../training/schedulers.md) — what's in
  `scheduler.state_dict()`.
- [Data § Stateful dataloader](../data/stateful-dataloader.md): the
  loader state the training loop saves at every checkpoint and restores
  on resume.
- [Metrics § WandB resume](../metrics-and-profiling/index.md) — how
  `wandb_run_id` in `extra` enables resumed W&B runs.
- [MLflow backend](../metrics-and-profiling/mlflow.md): how
  `mlflow_run_id` resumes the same MLflow run.
