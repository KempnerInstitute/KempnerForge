# Sampler

Two sampler classes in
[`kempnerforge/data/sampler.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/data/sampler.py):

- **`DistributedSampler`** — rank-partitioned, deterministic shuffle,
  skip-ahead for resume. The default.
- **`MixtureSampler`** — weighted sampling over a `MixtureDataset`,
  partitioned per rank.

Both expose `set_epoch(epoch)`, `set_skip(samples)`, and `state_dict()`
/ `load_state_dict()` — the contract that `StatefulDataLoader`
assumes.

## `DistributedSampler`

Indices are partitioned across DP ranks with a stride pattern, not a
contiguous block. Rank `r` of `N` ranks gets indices
`perm[r], perm[r+N], perm[r+2N], ...`:

```python
# kempnerforge/data/sampler.py — DistributedSampler.__iter__
if self.shuffle:
    g = torch.Generator()
    g.manual_seed(self.seed + self._epoch)
    indices = torch.randperm(len(self.dataset), generator=g).tolist()
else:
    indices = list(range(len(self.dataset)))

if self.drop_last:
    indices = indices[: self.total_size]   # total = num_samples * num_replicas
else:
    padding = self.total_size - len(indices)
    indices += indices[:padding]

# Stride partition
indices = indices[self.rank : self.total_size : self.num_replicas]
```

Why stride, not contiguous? With stride, changing `num_replicas`
between runs *does* change which rank sees which samples — but every
sample is still seen exactly once per epoch and the seed-derived
shuffle is identical. Contiguous block partitions are more sensitive
to dataset-size / rank-count interactions.

### Shuffle seeding

The shuffle is keyed on `seed + epoch`, where `seed` is
`TrainConfig.seed` (default 42) and `epoch` is set via
`set_epoch(self._epoch)` at the top of every `StatefulDataLoader.__iter__`.
Two runs with the same seed and the same number of ranks produce
identical shuffle orders every epoch.

### Drop-last vs pad-wrap

```python
if drop_last:
    self.num_samples = total // num_replicas
    self.total_size  = self.num_samples * num_replicas      # shorter than total
else:
    self.num_samples = math.ceil(total / num_replicas)
    self.total_size  = self.num_samples * num_replicas      # padded by wrap-around
```

KempnerForge defaults to `drop_last=True` in every call site. Losing
a handful of samples per epoch at world sizes of 16-64 is a
rounding-error effect at pretraining scale; the upside is every rank
does exactly the same amount of work, every step.

### Skip-ahead

```python
# At the end of __iter__
if self._skip > 0:
    indices = indices[self._skip :]
    self._skip = 0   # one-shot — resets after being applied
```

`set_skip(k)` tells the sampler "drop the first `k` samples from
this epoch's index list". `StatefulDataLoader` computes
`k = batches_yielded * batch_size` on resume and passes it in. The
reset after application is important: on the *next* epoch, no skip
is applied, and iteration starts from index 0.

## `MixtureSampler`

Used when `data.datasets` contains multiple sources. The pipeline
splits between construction and per-epoch iteration.

In `__init__` (and re-run by `update_weights`):

1. **Probabilities** — `weights` normalized, with optional temperature
   scaling — see [Mixing and annealing](mixing-and-annealing.md).
2. **Per-rank budget** — sum of `size // num_replicas` across all
   sub-datasets (or `ceil` if `drop_last=False`).
3. **Target counts** — `round(p * total_per_rank)` per dataset, with
   a rounding-fix pass that adds or subtracts 1 from datasets in
   probability order until the total matches exactly.

Each `__iter__()` then does:

4. **Shuffled indexes** — per sub-dataset, `torch.randperm(size)`
   with the same `seed + epoch` trick, then the rank's stride slice.
5. **Oversample if needed** — if the target exceeds per-rank available
   indices, the index list is repeated until it's long enough.
6. **Global offset** — each sub-dataset's local index is added to its
   offset so the final list indexes into the parent `MixtureDataset`.
7. **Final shuffle** — the assembled index list is shuffled once more
   so samples from different sources are interleaved, not grouped.

### `update_weights(weights, temperature)`

Called by the training loop at phase boundaries. Recomputes
probabilities and `_target_counts`; takes effect on the next
`__iter__()` call. The sampler's internal `seed + epoch` behavior is
unchanged — weight changes don't re-seed.

### Skip-ahead behavior

Same as `DistributedSampler`: `set_skip(k)` drops the first `k`
entries of the assembled (final, shuffled) index list. Resumes are
robust if the weights and `num_replicas` match between save and load;
a phase change between save and load would shift the index list.

## What's in `state_dict`

Both samplers return the same keys:

```python
{
    "epoch": self._epoch,
    "seed": self.seed,
    "num_replicas": self.num_replicas,
    "rank": self.rank,
}
```

`load_state_dict` restores **only** `epoch`. The other fields are
recorded for diagnostic purposes — `num_replicas` and `rank` are
local to this process and must be re-derived from the current world
size, not transplanted from the save. `seed` is already in
`TrainConfig` and doesn't change across resume.

## Eval sampler

Evaluation uses `DistributedSampler(eval_dataset, shuffle=False)`:

```python
# scripts/train.py
eval_sampler = DistributedSampler(
    eval_dataset, num_replicas=dp_size, rank=dp_rank,
    shuffle=False, seed=tc.seed,
)
```

`shuffle=False` makes the order deterministic (no permutation),
which is what you want for reproducible eval loss numbers.

## See also

- [Stateful dataloader](stateful-dataloader.md) — the caller that
  drives `set_epoch`, `set_skip`, and the state dict.
- [Mixing and annealing](mixing-and-annealing.md) — `update_weights`
  and how phase transitions interact with the sampler.
- [Checkpointing § Train state](../checkpointing/train-state.md) —
  the sampler state is captured inside the dataloader state dict.
