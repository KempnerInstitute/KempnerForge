# Training

The training loop itself and the knobs around it: optimizers, LR
schedulers, loss functions, gradient utilities, in-loop evaluation,
sampling, and the `TrainingHook` extension point.

The [Data flow](../architecture/data-flow.md) page is the one-slide
overview of the step. This section zooms into each collaborator.

```{toctree}
:maxdepth: 1

training-loop
optimizers
schedulers
losses
gradient-utilities
evaluation
generation
hooks
```

- **[Training loop](training-loop.md)** — a reader's walkthrough of
  [`scripts/train.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/scripts/train.py):
  setup, the two step bodies (PP vs non-PP), phase transitions,
  periodic work.
- **[Optimizers](optimizers.md)** — `adamw`, `lion`, `muon`,
  `schedule_free_adamw`: the four registered optimizers, when to pick
  each, decay grouping, DTensor / FSDP2 notes.
- **[Schedulers](schedulers.md)** — `cosine`, `linear`, `wsd`,
  `constant`, `rex`, `none`: warmup and decay math, required fields.
- **[Losses](losses.md)** — `cross_entropy`, `chunked_cross_entropy`,
  and `z_loss` as a train-config regularizer (`train.z_loss_weight`).
- **[Gradient utilities](gradient-utilities.md)** — `maybe_no_sync` for
  accumulation, `clip_grad_norm_` for DTensor-aware clipping.
- **[Evaluation](evaluation.md)** — `run_eval()`, `EvalConfig`, the PP
  eval path, standalone
  [`scripts/eval.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/scripts/eval.py).
- **[Generation](generation.md)** — `generate()` from
  [`kempnerforge/model/generate.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/model/generate.py),
  top-k / top-p / temperature, KV cache, standalone
  [`scripts/generate.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/scripts/generate.py).
- **[Hooks](hooks.md)** — `TrainingHook`, `HookRunner`, lifecycle
  events, when to fork `train.py` vs write a hook.

## See also

- [Data flow](../architecture/data-flow.md) — the one-page training
  loop overview.
- [Configuration § TrainConfig](../configuration/config-sections.md) —
  every field this subsystem reads.
