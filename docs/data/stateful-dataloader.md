# Stateful dataloader

[`StatefulDataLoader`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/data/dataloader.py)
wraps `torch.utils.data.DataLoader` with three additions:

1. Tracks `(epoch, batches_yielded)` across iterator boundaries.
2. Exposes `state_dict()` / `load_state_dict()` that capture sampler
   state alongside position.
3. On resume, calls `sampler.set_skip(...)` to fast-forward past
   already-seen batches within the current epoch.

## Construction

```python
# scripts/train.py
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
    self._iterator = iter(self._dataloader)
    self._batches_yielded = 0
    return self

def __next__(self):
    batch = next(self._iterator)
    self._batches_yielded += 1
    return batch
```

On `StopIteration`, `_epoch` increments, `_batches_yielded` resets,
and the next `__iter__` call rebuilds the underlying iterator (and
reseeds the sampler via `set_epoch`).

## State capture

```python
def state_dict(self) -> dict:
    return {
        "epoch": self._epoch,
        "batches_yielded": self._batches_yielded,
        "sampler": self.sampler.state_dict(),
    }
```

`load_state_dict` reads these back, restores sampler state via its
own `load_state_dict`, then — if `batches_yielded > 0` — tells the
sampler to skip:

```python
self.sampler.set_skip(batches_yielded * self.batch_size)
```

Effect: on the next `__iter__()`, the sampler yields its index list
minus the first `batches_yielded * batch_size` elements. Training
picks up at the exact same sample boundary.

## The wiring gap

The infrastructure works, but `scripts/train.py` **does not currently
pass the dataloader to `ckpt_mgr.save()`**:

```python
# scripts/train.py
ckpt_mgr.save(
    step=step,
    tokens_seen=tokens_seen,
    scheduler=scheduler,
    extra=ckpt_extra,
    # no dataloader=...
)
```

Consequence: on resume, the dataloader restarts from batch 0 of the
current epoch. With deterministic seeding and a shuffled sampler, this
means a few batches get replayed but the loss trajectory is otherwise
indistinguishable from an uninterrupted run.

For exact-batch-level reproducibility, pass `dataloader=dataloader`
into `ckpt_mgr.save` yourself — `build_train_state` picks it up
automatically. See
[Checkpointing § Train state](../checkpointing/train-state.md#dataloader-state).

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
