# Mixing and annealing

Two orthogonal features that work together:

- **Mixing** — sample from several datasets in one epoch, each with
  its own weight. Built on `MixtureDataset` + `MixtureSampler`.
- **Annealing / phase scheduling** — at user-specified steps, switch
  the mixing weights and optionally scale the learning rate. Used
  for data curriculum (e.g. "70% web + 30% code until step 100k,
  then 20% web + 80% books").

Both are opt-in. Ignore this page unless you want multiple data
sources in the same run.

## Enabling mixing

Set one or more `[[data.datasets]]` tables in your TOML:

```toml
[data]
tokenizer_path  = "meta-llama/Llama-3-8B"
mix_temperature = 1.0          # 1.0 = use weights as-is, >1 = flatten

[[data.datasets]]
path   = "/data/web_tokens"    # MemoryMappedDataset
weight = 1.0
name   = "web"                 # shows up in per-dataset metrics

[[data.datasets]]
hf_name = "bigcode/the-stack-v2"
weight  = 0.3
name    = "code"

[[data.datasets]]
path   = "/data/books_tokens"
weight = 0.5
name   = "books"
```

Each source can be either a mmap directory (`path`) or a HuggingFace
dataset (`hf_name`). `DataConfig.__post_init__` rejects a
`DatasetSource` that has neither.

When `data.datasets` is non-empty, it overrides `data.dataset_path`
and `data.hf_dataset_name` — the single-source paths are ignored.

## What happens at construction

In `scripts/train.py`:

1. Each `DatasetSource` builds a sub-dataset — `MemoryMappedDataset`
   or `HuggingFaceDataset` — with the global `tokenizer_path` and
   `pack_sequences` setting.
2. `MixtureDataset(sub_datasets, names)` concatenates them:

   ```python
   # kempnerforge/data/dataset.py — MixtureDataset
   self._cumulative: list[int] = [0]
   for ds in datasets:
       self._cumulative.append(self._cumulative[-1] + len(ds))
   ```

   Global index `idx` maps to `(ds_idx, local_idx)` via
   `bisect.bisect_right(self._cumulative, idx) - 1`. Each sample gets
   a `dataset_idx` field added for per-source metric attribution.

3. `MixtureSampler(cumulative_sizes, weights, ...)` computes:
   - Per-dataset per-rank available samples.
   - A target draw count per dataset per epoch, proportional to the
     normalized weights.
   - Rounds to integers, then fixes the total to match the full
     per-rank budget.

## Weights, temperature, and oversampling

The sampler normalizes raw `weights` into probabilities. Temperature
`> 1` flattens the distribution:

```python
# kempnerforge/data/sampler.py — MixtureSampler.__init__
if temperature != 1.0:
    log_w = [math.log(max(w, 1e-12)) / temperature for w in weights]
    max_lw = max(log_w)
    scaled = [math.exp(lw - max_lw) for lw in log_w]
    self._probs = [s / sum(scaled) for s in scaled]
else:
    self._probs = [w / sum(weights) for w in weights]
```

After normalization, each dataset contributes `round(p * total) `
samples per epoch. If a dataset has fewer samples than its target
(high weight on a tiny source), the sampler wraps around and
oversamples:

```python
if target <= len(rank_indices):
    drawn = rank_indices[:target]
else:
    reps = target // len(rank_indices) + 1
    drawn = (rank_indices * reps)[:target]
```

This is standard practice for upsampling small curated sources
alongside a large crawl; just know that the same physical sample
will reappear within one epoch.

## Phase scheduling (annealing)

Two equivalent ways to express step-triggered weight changes:

### Multi-phase

```toml
[[data.phases]]
start_step       = 100_000
dataset_weights  = { web = 0.4, code = 0.3, books = 0.3 }
lr_scale         = 1.0

[[data.phases]]
start_step       = 180_000
dataset_weights  = { web = 0.1, code = 0.2, books = 0.7 }
lr_scale         = 0.3                 # anneal LR alongside the data shift
```

Constraints (enforced in `DataConfig.__post_init__`):

- `start_step` is strictly monotonically increasing across phases.
- `lr_scale` > 0.
- Any dataset named in a phase but *not* listed in the phase's
  `dataset_weights` falls back to its original weight from
  `[[data.datasets]]`.

### Annealing shortcut

For the common "run normally, then switch for the last N steps"
pattern:

```toml
[data]
anneal_start_step = 180_000
anneal_weights    = { books = 1.0 }    # everything else drops to its default
```

This is just syntactic sugar that the training loop compiles into a
single-phase list at startup. Using both `data.phases` and
`data.anneal_start_step` is rejected.

## How phases execute

Inside `scripts/train.py`:

```python
# --- Phase activation in the training loop ---
while (current_phase_idx < len(active_phases)
       and step >= active_phases[current_phase_idx].start_step):
    phase = active_phases[current_phase_idx]
    new_weights = [
        phase.dataset_weights.get(name, original_weights_dict[name])
        for name in dataset_names
    ]
    sampler.update_weights(new_weights, temperature=config.data.mix_temperature)
    phase_lr_scale = phase.lr_scale
    current_phase_idx += 1
```

Two effects:

- **Sampler re-weighting** — `MixtureSampler.update_weights` recomputes
  probabilities and target counts. Takes effect on the *next*
  `__iter__()` call (so within the current epoch, the previous weights
  are still in play; phase changes are clean at epoch boundaries).
- **LR scale** — `phase_lr_scale` is applied to every optimizer param
  group on each step: `pg["lr"] *= phase_lr_scale`. The base LR comes
  from the scheduler; the phase scale is an overlay.

### Resume into the right phase

On auto-resume, `scripts/train.py` replays the phase activations
against the current `step` so the sampler and LR scale are correct
before the training loop starts:

```python
# On load — re-derive phase state
for i, phase in enumerate(active_phases):
    if step >= phase.start_step:
        sampler.update_weights([...], temperature=...)
        phase_lr_scale = phase.lr_scale
        current_phase_idx = i + 1
```

`current_phase_idx` itself is saved in the checkpoint's `extra` dict
so the resume logic matches what was live at save time — see
[Checkpointing § Train state](../checkpointing/train-state.md).

## Per-source metrics

Each `MixtureDataset` sample has a `dataset_idx` field. The training
loop breaks the loss down per source so you can see which dataset is
contributing what:

```
loss/total  = 2.41
loss/web    = 2.18    (0.5 of samples)
loss/code   = 3.02    (0.3 of samples)
loss/books  = 2.22    (0.2 of samples)
```

Exposed names come from `DatasetSource.name` (auto-derived from
`path` / `hf_name` if left blank).

## Limitations

- Sampler weight changes are per-epoch: `update_weights` takes effect
  on the next iteration, not mid-epoch.
- `MixtureSampler` doesn't expose a true `load_state_dict` for
  skip-ahead — `StatefulDataLoader` calls `set_skip()` on it, which
  the sampler honors, but mid-epoch exact reproducibility across
  weight changes is not guaranteed.
- All sub-datasets share the same global `tokenizer_path` and
  `pack_sequences` setting. Multiple tokenizers in one run is not
  supported.

## See also

- [Memory-mapped](memory-mapped.md) and [HuggingFace](huggingface.md) —
  the sub-dataset types you can mix.
- [Sampler](sampler.md) — `MixtureSampler` rank partitioning details.
- [Checkpointing § Train state](../checkpointing/train-state.md) —
  `phase_idx` in the `extra` dict and why it's saved.
- [Training § Schedulers](../training/schedulers.md) — the base LR
  that `phase_lr_scale` multiplies.
