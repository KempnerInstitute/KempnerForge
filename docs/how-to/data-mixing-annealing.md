# Mix datasets and anneal data weights

Training on more than one corpus at once — and shifting the mix mid-run
— is common practice. KempnerForge supports both through three
orthogonal config controls: multi-dataset sources in `[data]`, a
temperature knob on the mixture weights, and optional `[data.phases]`
that retarget the mix (and the LR) at fixed steps. This page explains
each and shows how they compose.

## The three controls

```toml
[data]
mix_temperature = 1.0            # knob on the weights; 1.0 = as-declared

[[data.datasets]]                # multi-dataset mixture
name   = "web"
path   = "/data/tokenized/web"
weight = 0.7

[[data.datasets]]
name    = "code"
hf_name = "bigcode/the-stack-dedup"
weight  = 0.3

[[data.phases]]                  # phase transitions
start_step      = 10_000
dataset_weights = { web = 0.3, code = 0.7 }
lr_scale        = 0.5
```

- **`[[data.datasets]]`** turns mixing on. Without it, `[data]` falls
  back to the single-dataset fields (`dataset_path` or
  `hf_dataset_name`).
- **`mix_temperature`** rescales declared weights before sampling.
- **`[[data.phases]]`** swaps weights (and scales LR) at specific
  steps during training.

A fourth control, `anneal_start_step` / `anneal_weights`, is sugar for
the common "change the mix once, late in training" pattern — see
[Annealing shortcut](#annealing-shortcut) below.

## Multi-dataset mixture

Each `[[data.datasets]]` block is a
[`DatasetSource`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/data.py):

| Field | Meaning |
|-------|---------|
| `name`   | Label for per-dataset metrics and phase overrides. Auto-filled from `path`/`hf_name` if empty. |
| `path`   | Pre-tokenized directory (use with `file_pattern`). |
| `hf_name` | HuggingFace dataset ID (alternative to `path`). |
| `hf_config` | HF dataset config (e.g. `"wikitext-103-raw-v1"`). |
| `weight` | Relative sampling weight (must be positive). Normalized internally. |

At least one of `path` / `hf_name` must be set per source, enforced by
`DataConfig.__post_init__`.

When any `[[data.datasets]]` is present, `scripts/train.py` builds a
[`MixtureDataset`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/data/dataset.py)
over the sub-datasets and drives it with a
[`MixtureSampler`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/data/sampler.py).
The sampler:

- Partitions each sub-dataset's indices across data-parallel ranks
  (stride-based, like `DistributedSampler`).
- Allocates `target_counts[i] = round(prob[i] × total_per_rank)` indices
  per epoch from each source and over- or undersamples to match
  the target.
- Interleaves the drawn indices with one final shuffle so the model
  sees a randomly mixed order, not source-blocked batches.

Every sample returned by `MixtureDataset.__getitem__` includes a
`dataset_idx` key so the training loop can slot the per-batch loss into
per-dataset metrics.

### Per-dataset metrics

When the mixture is active and the metrics interval fires,
`scripts/train.py` emits two series per dataset name:

- `loss/{name}` — mean loss of samples from that dataset in the
  accumulation window.
- `data/{name}/tokens` — running token count consumed from that
  dataset.

Plot these in WandB / TensorBoard to see whether a dataset is
contributing normally or drifting. A rising `loss/code` while
`loss/web` stays flat is a signal.

## Temperature

`mix_temperature` rescales the declared `weight` values before
normalization. The math in
[`MixtureSampler.__init__`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/data/sampler.py)
is:

```python
log_w  = [log(max(w, 1e-12)) / temperature for w in weights]
probs  = softmax(log_w)           # after subtracting the max for stability
```

So the three interesting regimes are:

- **`temperature = 1.0`** (default) — probabilities are just the
  declared weights, normalized.
- **`temperature > 1.0`** — weights are *flattened* toward uniform. At
  `temperature → ∞`, every source has probability `1/N` regardless of
  its declared weight.
- **`temperature < 1.0`** — weights are *sharpened*. At
  `temperature → 0`, the heaviest source takes everything.

The common setting is `temperature > 1` when the declared weights
reflect corpus size but you want to undersample the largest corpus so
a small, high-quality source isn't drowned out. A typical value is in
the 1.3–2.0 range; the right number depends on the relative size of
your corpora.

## Phase transitions

A `[[data.phases]]` entry is a
[`TrainingPhase`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/config/data.py):

| Field | Meaning |
|-------|---------|
| `start_step`      | Step at which this phase activates (must be non-negative). |
| `dataset_weights` | `{name: weight}` — any source not listed keeps its original weight. |
| `lr_scale`        | Multiplier applied to *every* param group's LR once this phase fires. |

Constraints (validated in `DataConfig.__post_init__`):

- Phase `start_step` values must be strictly monotonically increasing.
- You can use either `[[data.phases]]` **or** the annealing shortcut
  below — not both.
- All `dataset_weights` values must be non-negative; `lr_scale > 0`.

### What happens on a phase transition

On the first training step where `step >= phase.start_step`, the loop
in
[`scripts/train.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/scripts/train.py)
does two things:

1. Builds a new `weights` list by overriding the original declared
   weights with `phase.dataset_weights` entries where specified, then
   calls `sampler.update_weights(new_weights, temperature=config.data.mix_temperature)`
   — the next `__iter__()` call on the sampler uses the new mix.
2. Sets `phase_lr_scale = phase.lr_scale`. From this step forward,
   the training loop multiplies every parameter group's LR by
   `phase_lr_scale` after the scheduler has computed the base LR. The
   scheduler still runs; `phase_lr_scale` is an additional factor on
   top.

So if your scheduler is driving `lr` from `3e-4` down to `3e-5` via
cosine decay, a phase with `lr_scale = 0.5` means the optimizer sees
`1.5e-4 → 1.5e-5` while that phase is active. Later phases overwrite
`phase_lr_scale` with their own value; there's no compounding across
phases.

### Resume behavior

At startup, the training loop walks the phase list and applies every
phase whose `start_step <= current_step`, logging
`"Resumed into phase K, lr_scale=S"`. The checkpoint itself tracks
the current phase index (stored in the `ckpt_extra` field), so a
mid-phase resume lands on the correct weights without re-firing older
phases.

`mixture_dataset.dataset_names` is the key into `phase.dataset_weights`
— keep `name` stable across checkpoint / resume or you'll silently
revert to the original declared weights for any name that no longer
matches.

## Annealing shortcut

For the very common "one change, late" pattern, you can skip the
verbose `[[data.phases]]` block:

```toml
[data]
anneal_start_step = 40_000
anneal_weights    = { web = 0.1, code = 0.9 }
```

`scripts/train.py` converts this into a one-element `TrainingPhase`
list internally:

```python
TrainingPhase(
    start_step=config.data.anneal_start_step,
    dataset_weights=dict(config.data.anneal_weights),
    # lr_scale defaults to 1.0 — no LR change
)
```

Default `lr_scale` is **1.0** — the LR curve doesn't change unless you
say so. If you want both weight annealing **and** an LR drop on
transition, use `[[data.phases]]` explicitly.

## Sanity checks

Before committing to a long run, verify the mix is what you think:

```bash
uv run python scripts/train.py configs/train/your_mix.toml \
    --train.max_steps=100 --metrics.log_interval=1
```

Inspect the `data/{name}/tokens` counters after 100 steps. The ratios
should roughly match your target probabilities.

```
data/web/tokens        ≈ (web_prob)  × total_tokens
data/code/tokens       ≈ (code_prob) × total_tokens
```

If the numbers are off by much more than one batch's worth of tokens,
double-check:

- `name` values in `[[data.datasets]]` vs any `phase.dataset_weights`
  overrides (case-sensitive, exact match).
- That phase `start_step` values haven't already fired before step
  100 (phase transitions change what you're measuring).
- That `mix_temperature` is what you intended — a temperature other
  than 1.0 changes sampling probabilities away from the declared
  weights.

For a phase transition specifically, watch the log for the
`"Phase transition at step N: phase=K, lr_scale=S"` line — it fires
exactly once per transition, and the `data/{name}/tokens` slopes
should visibly change immediately after.

## See also

- [Data § Mixing and annealing](../data/mixing-and-annealing.md) —
  `MixtureDataset` internals and the non-mixing dataset classes.
- [Data § Sampler](../data/sampler.md) — `MixtureSampler` and
  `update_weights` internals.
- [Configuration § `[data]`](../configuration/config-sections.md) —
  every `DataConfig` field and its validation rule.
- [Training § Training loop](../training/training-loop.md) — where
  `phase_lr_scale` is applied and how phase checkpoint state is
  persisted.
- [Compare optimizers](compare-optimizers.md) — LR conventions matter
  when you combine `lr_scale` with non-AdamW optimizers.
- [Prepare tokenized data](prepare-tokenized-data.md) — how to produce
  the per-source `path` directories this page consumes.
