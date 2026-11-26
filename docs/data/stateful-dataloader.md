# Stateful dataloader

[`StatefulDataLoader`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/data/dataloader.py)
wraps `torch.utils.data.DataLoader` with three additions:

1. Tracks the epoch and how far into it training has read, across
   iterator boundaries.
2. Exposes `state_dict()` / `load_state_dict()` that capture that
   position with the sampler state, in a form that survives a change of
   data-parallel size or batch size.
3. On resume, tells the sampler where to start, so training continues
   at the sample it reached within the current epoch.

Streaming datasets use `StreamingDataLoader` from the same module, which
resumes by replaying its stream: see [Streaming loader](#streaming-loader).

## Construction

```python
# training/data_pipeline.py
dataloader = StatefulDataLoader(
    dataset,
    batch_size=tc.batch_size,
    sampler=sampler,             # DistributedSampler or MixtureSampler
    config=config.data,
)
```

Internally it builds a plain `DataLoader` with these options wired
from `DataConfig`:

```python
DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=self.sampler,
    num_workers=config.num_workers,
    pin_memory=config.pin_memory,
    prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
    persistent_workers=config.num_workers > 0,
    drop_last=True,
)
```

Two choices worth calling out:

- **`drop_last=True`** — ensures all DP ranks see the same number of
  batches per epoch. The `DistributedSampler` already trims the
  per-rank index list for this case, so `drop_last` is effectively
  double-insurance.
- **`persistent_workers=num_workers > 0`** — workers are kept alive
  across epochs. This avoids re-importing the tokenizer and
  re-mapping shards every epoch, but means any mutable state in a
  worker process persists too.

## Iteration

```python
def __iter__(self):
    self.sampler.set_epoch(self._epoch)
    consumed = self._consumed()
    if consumed > 0 and isinstance(self.sampler, DistributedSampler):
        self.sampler.set_position(consumed)
    elif consumed > 0:
        self.sampler.set_skip(consumed)
    self._iterator = iter(self._dataloader)
    return self

def __next__(self):
    batch = next(self._iterator)
    self._batches_yielded += 1
    return batch
```

`_consumed()` is how far into the epoch training has read: all ranks'
samples with a `DistributedSampler`, this rank's samples otherwise. On
`StopIteration`, `_epoch` increments, the position resets, and the next
`__iter__` call rebuilds the underlying iterator (and reseeds the
sampler via `set_epoch`).

## State capture

`state_dict()` returns the epoch, the batch size, the sampler state,
and the position:

- With a `DistributedSampler`, `global_position`: the samples of the
  epoch's global order that all data-parallel ranks have consumed
  together, `batches × batch_size × num_replicas` in a run that never
  changed size.
- With any other sampler (a `MixtureSampler`), `samples_consumed`: this
  rank's samples.

It also keeps the dataset length and `batches_yielded`, the position in
whole batches of this loader's size, for versions that predate the
position: this rank's batches in the epoch in a run that never changed
size, and the position divided by the global batch, rounded down, after a
resize. And it keeps `base_seed_rng_state`, the CPU generator state the
loader's iterator drew its workers' base seed from (see
[Worker RNG](#worker-rng)).

`load_state_dict` first closes the loader, so workers started before the
load stop and the next `__iter__` starts new ones from the loaded state.
It restores the epoch and the sampler state and records
the position, which the next `__iter__` applies. It logs a warning when
the shuffle seed or the dataset length differs from the checkpoint's,
since the position counts samples of the saved order. A checkpoint
written before the batch size was recorded holds only `batches_yielded`;
on the saved data-parallel size it resumes as before, `batches_yielded ×
batch_size` samples into this rank's list, which is right only at the
batch size it was saved with. The loader cannot check that, so it logs
the assumption.

## Resuming on another data-parallel size

`DistributedSampler` shuffles the whole dataset with `seed + epoch` and
gives rank `r` the positions `r`, `r + num_replicas`, ... of that order.
The order does not depend on `num_replicas`, so `global_position` means
the same thing on any number of ranks. `set_position(G)` starts rank `r`
at its first position at or after `G`, and every rank yields the same
number of samples, `samples_after(G)`, so the ranks stay in step to the
end of the epoch. Together they read the order from `G` on, with no
sample skipped or read twice, whatever data-parallel size and batch size
saved the checkpoint. Keeping the ranks in step can drop up to one more
sample per rank at the end of that epoch, beside the partial batch
`drop_last` already drops. A position with no full batch left at the new
size starts the next epoch, and a resume onto another data-parallel size
or batch size logs both.

Two positions cannot carry over to another data-parallel size, and
`load_state_dict` raises `DataPositionError`, naming both sizes:

- A `MixtureSampler` draws and shuffles each rank's samples separately,
  so a position on one data-parallel size has no counterpart on another.
  It still resumes exactly on the saved size, at any batch size.
- A checkpoint written before the batch size was recorded cannot say how
  many samples of the global order were consumed. Resume it once on its
  saved data-parallel size and batch size; the checkpoints saved after
  that carry over to any size.

`CheckpointManager.apply_dataloader_state` checks the position on every
rank with `check_state`, which changes nothing, and the ranks share their
verdicts before any rank applies it, so every rank raises together and none
is left waiting on the others, even where one rank's view of the data
differs. A `checkpoint.load_path` warm start does not raise: it logs the
reason and every rank starts the data at its first sample (see
[Auto-resume](../checkpointing/auto-resume.md#where-its-called)).
An auto-resume onto another world size or data-parallel layout stops
earlier, before any rank applies a position, unless
`checkpoint.ignore_rank_state_mismatch` is set. A checkpoint saved before
per-rank state was recorded cannot show the change, so without the flag its
position meets the rules above and raises where it cannot carry over. With
the flag, every rank applies the position the saved rank 0 reached by the
rules above and, where it cannot carry over, starts the data at its first
sample with a warning, as a warm start does (see
[Resharding § Scaling a run to a different world size](../checkpointing/resharding.md#scaling-a-run-to-a-different-world-size)).

## Save and restore

The training loop passes `dataloader=` at every save site, so the loader
position travels with the checkpoint:

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

An emergency checkpoint taken mid-epoch on 64 data-parallel ranks records
e.g. `{'epoch': 7, 'batch_size': 4, 'dataset_length': 1000000, 'global_position': 3584, 'batches_yielded': 14, 'sampler': {...}, 'base_seed_rng_state': b'...'}`,
and `__iter__` re-applies the position on resume, so training picks up at
the same sample rather than at batch 0. See
[Checkpointing § Train state](../checkpointing/train-state.md#dataloader-state).

## Streaming loader

`StreamingDataLoader` wraps the `torch.utils.data.DataLoader` over a
`StreamingHuggingFaceDataset`. A stream has no index to seek to, so its
position is the number of batches training took in the current pass. On
resume it rebuilds the pass and discards that many batches before
yielding, which lands on the exact next sample because the shuffle, the
split across ranks and DataLoader workers, and the packing are
deterministic. A pass restarted midway gets fresh workers, since
persistent workers keep their own copy of the dataset.

The position is saved with what decides the stream's order, and
`load_state_dict` raises `DataPositionError` when any of it changed,
since the same count would then land on different samples:

- the source: for a local directory, the data files `datasets` resolves
  for the split, by name relative to the directory and by size, so the
  same files moved to another directory still match; for a Hub dataset,
  its name and config as given; for both, the split, the text field and
  the shard count;
- the data-parallel size, the seed and shuffle buffer, `seq_len`, the
  tokenizer as named and `pack_sequences`;
- the batch size and `num_workers`.

The check does not see a file rewritten at the same size, a new revision
of a Hub dataset with the same shard count, or a new version of
`datasets` or of the tokenizer files, and any of those can shift the
replay. A map-style position meeting a stream, or the reverse, raises too.
Each rank checks the stream as it sees it, the sizes of its data files
included, and the ranks share their verdicts before any rank replays, so a
rank whose copy of a file differs stops every rank, not only itself, or under
the flag onto another layout restarts every rank's stream.
A `checkpoint.load_path` warm start does not raise: it logs the reason and
every rank starts its stream at the first document (see
[Auto-resume](../checkpointing/auto-resume.md#where-its-called)). A
replay that runs off the end of the stream starts the next pass with a
warning instead of failing on one rank.

### Replay cost

The replay reads and tokenizes the consumed part of the pass again
before the first step, and logs its progress with a rate and an estimate
of the time left. A resume at batch `k` costs about
`k × batch_size × seq_len / rate`. Measured on local JSONL with the
Llama 3.1 tokenizer, batch 4 and `seq_len` 2048 on a busy node, the rate
was about 0.88M tokens/s per rank with `num_workers = 0` and 3.2M
tokens/s with `num_workers = 4`, so a resume 10,000 batches (82M tokens)
into a pass waits about 26 s with 4 workers and 93 s without. From the
Hub, the replay downloads the consumed part again, and when the shard
count does not divide by the data-parallel size every rank downloads the
whole consumed prefix. Ranks replay at their own pace, and a difference
longer than `distributed.nccl_timeout_sec` (1800 s by default) fails the
first collective of the resumed run; a requeue then repeats the same
replay. Pre-tokenized data resumes without a replay.

### Several ranks

Every rank saves its own pass and count, with the data-parallel rank that
saved them, in the checkpoint's per-rank state
([Train state](../checkpointing/train-state.md#per-rank-state-schema-v1)).
An auto-resume restores each rank's own, so every rank replays its own
stream to where it stopped, in any pass. The ranks' parts of the stream
can differ by many batches: a rank reads whole shards when the shard
count divides by the data-parallel size, and shards differ in size. So
once the first rank finishes a pass, the ranks are in different passes
or at different counts, which their own saved positions record.

An auto-resume that `checkpoint.ignore_rank_state_mismatch` lets onto
another layout of the same data-parallel size gives each rank the position
its own data-parallel rank saved, found by the `data_parallel_rank` saved
with it, so it is exact in every pass too. On another data-parallel size
the saved positions have no counterpart, and every rank starts its stream
at the first document, with a warning.

A `checkpoint.load_path` warm start gives every rank rank 0's position
instead, which is exact only while every rank started its current pass on
the same step, which in practice means until the first rank finishes a
pass. Past that, a rank that had fallen behind rank 0
skips data and a rank that was ahead reads data again, up to the
difference between their parts of the stream. Such a rank logs a
warning: a rank whose part of the pass is shorter than rank 0's count
runs off its end during the replay, says that the stream ended and
starts the next pass at its first batch, and in a later pass every rank
that takes another rank's position warns when it loads it. Both
warnings show in every rank's log, past the training log's rank-0
filter.

## Worker RNG

PyTorch's `DataLoader` seeds worker `k` with `base_seed + k`, and draws
`base_seed` from the parent process's CPU generator each time it starts
an iterator: at every pass with `num_workers = 0`, and with persistent
workers at the first pass and the first after `close()`. Any other pass
resets the running iterator, which keeps its base seed. Both loaders
record the CPU generator state each new iterator drew its base seed
from, and save it as `base_seed_rng_state`. After `load_state_dict`,
the next iterator draws its base seed from the saved state through a
generator of its own, so a resume neither draws from the CPU generator
where the uninterrupted run did not nor gives its workers other seeds.
With the CPU generator restored too
([Checkpointing § Train state](../checkpointing/train-state.md#rng-capture)),
the resumed run continues it as the uninterrupted run does. The
exception is a position that starts the next pass of a loader without
persistent workers: the uninterrupted run starts a new iterator there,
drawing a new base seed, and the resume does the same.

`close()` keeps the saved state, so the training loop's final
checkpoint, written after it closes the loader, resumes as a run that
never closed. The next `iter()` in the same process starts new workers
with a base seed drawn from the CPU generator, as any start of workers
does.

When a data phase changes a mixture's weights, the training loop resets
its stream (`BatchStream.reset()`), and the next step starts the pass
again at the same position. The reset calls the loader's `restart()`,
which drops the loader's iterator, so a checkpoint saved before that step
records what the step's new iterator draws: a new base seed without
persistent workers, or the running workers' seeds with them.

What a resume does not reproduce:

- A worker's position in its own random streams. A resumed worker
  starts from its seed, while a persistent worker of the uninterrupted
  run has already drawn for the samples it loaded, and the resumed
  iterator hands batches to its workers in turn from its first batch
  on. A dataset that draws random numbers in its workers, such as a
  random augmentation, draws differently after a resume.
- A run that changes `num_workers`: the count decides when a pass draws
  a new base seed and how many workers take one, so neither the CPU
  generator nor the workers' seeds follow the uninterrupted run. Keep
  the count stable through a run.
- A checkpoint saved before `base_seed_rng_state` was recorded. Its
  resume draws a new base seed from the CPU generator, and the load
  warns where the uninterrupted run would have drawn none.

## Shutting the workers down

Both loaders start their workers with a `worker_init_fn` that ignores
`SIGTERM` and `SIGUSR1`. SLURM preemption signals the whole process
group, and a worker that died on the signal would fail the step in
progress before the emergency checkpoint is written. The same shield
keeps `multiprocessing` from stopping the workers at exit: its exit hook
sends leftover workers `SIGTERM` and then waits for them.

`close()` shuts the workers down through the loader's own sentinel path
instead, and kills a worker still alive after it. The loader keeps its
position: `state_dict()` does not change, and the next `iter()` starts new
workers where it stopped. The training loop calls `close()` on every exit
path. A process also calls it at exit, before `multiprocessing` stops its
leftover workers, for every loader whose workers are still alive, so a
script, or a `multiprocessing` child process, that ends or raises with an
iterator alive still exits. `shutdown_workers(loader)` from the same module
does the same for a plain `DataLoader` whose workers use
`ignore_shutdown_signals_in_worker`.

## Eval dataloader

The eval path uses a plain `torch.utils.data.DataLoader` (not
`StatefulDataLoader`), because eval is stateless — the full eval set
is replayed from the top at every evaluation interval:

```python
eval_dataloader = TorchDataLoader(
    eval_dataset,
    batch_size=tc.batch_size,
    sampler=eval_sampler,          # DistributedSampler with shuffle=False
)
```

## See also

- [Sampler](sampler.md) — `set_epoch`, `set_skip`, and how the
  sampler actually carries out the resume-aware iteration.
- [Checkpointing § Train state](../checkpointing/train-state.md#dataloader-state):
  the position the training loop saves with every checkpoint, through
  `dataloader=` at each save site, for both loaders.
- [Checkpointing § Auto-resume](../checkpointing/auto-resume.md) —
  the symlink-based restart that triggers this load path.
