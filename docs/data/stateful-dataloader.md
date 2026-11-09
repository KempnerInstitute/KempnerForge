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

It also keeps `batches_yielded`, this rank's batches in the epoch, for
versions that predate the position.

`load_state_dict` restores the epoch and the sampler state and records
the position, which the next `__iter__` applies. A checkpoint written
before the batch size was recorded holds only `batches_yielded`; on the
saved data-parallel size it resumes as before, `batches_yielded ×
batch_size` samples into this rank's list.

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
  many samples of the global order were consumed. Resume it on the saved
  data-parallel size once; the checkpoints saved after that carry over to
  any size.

The saved state and the loader's settings are the same on every rank, so
every rank raises together and none is left waiting on the others.

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
e.g. `{'epoch': 7, 'batch_size': 4, 'global_position': 3584, 'batches_yielded': 14, 'sampler': {...}}`,
and `__iter__` re-applies the position on resume, so training picks up at
the same sample rather than at batch 0. See
[Checkpointing § Train state](../checkpointing/train-state.md#dataloader-state).

## Streaming loader

`StreamingDataLoader` wraps the `torch.utils.data.DataLoader` over a
`StreamingHuggingFaceDataset`. A stream has no index to seek to, so its
position is the number of batches training took in the current pass,
saved with the settings that decide the stream's order: the source, the
data-parallel size, the seed and shuffle buffer, `seq_len`, the tokenizer,
`pack_sequences`, the batch size and `num_workers`. On resume it rebuilds
the pass and discards that many batches before yielding, which lands on
the exact next sample because the shuffle, the split across ranks and
DataLoader workers, and the packing are deterministic. The replay reads
and tokenizes the consumed part of the pass again, so a resume deep into
a large pass takes time; the log says how many batches it replays.

`load_state_dict` raises `DataPositionError` when one of those settings
changed, since the same count would then land on different samples, and
when a map-style position meets a stream or the reverse. A replay that
runs off the end of the stream starts the next pass with a warning
instead of failing on one rank.

The checkpoint holds rank 0's count, and every rank replays its own
stream by it. Every rank takes the same number of batches per step, so
the count is right on every rank while the ranks are in the same pass
over the stream, which holds throughout a run's first pass. Ranks whose
streams differ in length can finish a pass a few steps apart; a
checkpoint saved in that window resumes those ranks at rank 0's place.

## Worker RNG

PyTorch's `DataLoader` seeds worker RNGs deterministically from the
parent process RNG at worker init. Because
[Checkpointing § Train state](../checkpointing/train-state.md#rng-capture)
snapshots all four parent-process generators, workers see the same
seeds after resume — as long as `num_workers` doesn't change.

Changing `num_workers` across a resume means different worker seeds
and different downstream shuffling, even though the sampler state is
restored. Keep the count stable through a run.

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
- [Checkpointing § Train state](../checkpointing/train-state.md) —
  why `dataloader=` isn't wired by default and what you gain by
  opting in.
- [Checkpointing § Auto-resume](../checkpointing/auto-resume.md) —
  the symlink-based restart that triggers this load path.
