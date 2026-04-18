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
    "rng": {                              # full RNG capture (see below)
        "python":     random.getstate(),
        "numpy":      np.random.get_state(),
        "torch_cpu":  torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(),   # if CUDA available
    },
    "scheduler":   scheduler.state_dict(),            # present if scheduler passed
    "dataloader":  dataloader.state_dict(),           # present if dataloader passed
    # ... plus whatever was in the `extra` dict
}
```

## Who writes it

```python
# manager.py, inside save()
if self._rank == 0:
    train_state = build_train_state(
        step=step, tokens_seen=tokens_seen,
        scheduler=scheduler, dataloader=dataloader, extra=extra,
    )
    torch.save(train_state, ckpt_dir / "train_state.pt")
```

Only global rank 0 writes — the file is not sharded. On load, rank
0 reads and broadcasts it to every rank via
`dist.broadcast_object_list` so all ranks agree on `step`, scheduler
LR, RNG seeds, etc.

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

Pure `phase.lr_scale` overlays live in the training loop and are
recomputed from the current phase, so they don't need to be saved.

See [Training § Schedulers](../training/schedulers.md).

## Dataloader state

The infrastructure supports dataloader state (via
`dataloader.state_dict()` / `load_state_dict()` on KempnerForge's
`StatefulDataLoader`), but the shipped training loop in
`scripts/train.py` **does not currently pass the dataloader** to
`ckpt_mgr.save()`:

```python
ckpt_mgr.save(
    step=step,
    tokens_seen=tokens_seen,
    scheduler=scheduler,
    extra=ckpt_extra,          # note: no dataloader=...
)
```

On resume, the dataloader restarts at the beginning of its epoch.
Combined with sampler resume logic (see
[Data § Stateful dataloader](../data/index.md)) and
deterministic RNG, this is usually fine for most pretraining
workflows — you may replay a few batches on resume but loss is not
affected long-term.

If exact-batch-level reproducibility matters, pass `dataloader=` into
`ckpt_mgr.save()` yourself; `build_train_state` will pick it up
automatically.

## The `extra` dict

Anything that isn't step / tokens / RNG / scheduler / dataloader
can be threaded through `extra`:

```python
# scripts/train.py — around the checkpoint call
ckpt_extra = {"phase_idx": current_phase_idx} if active_phases else {}
if config.metrics.wandb_run_id:
    ckpt_extra["wandb_run_id"] = config.metrics.wandb_run_id

ckpt_mgr.save(step=step, tokens_seen=tokens_seen, scheduler=scheduler,
              extra=ckpt_extra)
```

On load, `restore_train_state` strips out the "standard" keys
(`step`, `tokens_seen`, `rng`, `scheduler`, `dataloader`) and returns
the rest as the third tuple element:

```python
step, tokens_seen, extra = ckpt_mgr.load(path=..., scheduler=scheduler)
if extra.get("wandb_run_id"):
    config.metrics.wandb_run_id = extra["wandb_run_id"]
```

KempnerForge uses this for:

- `phase_idx` — current curriculum phase (so training resumes the
  correct data mix without skipping forward).
- `wandb_run_id` — so resumed runs append to the same W&B run rather
  than starting a new one.

Custom training workflows can use it for anything
`torch.save`-serializable.

## Save failures

`torch.save` uses `pickle` under the hood. If any object in the
train state isn't picklable, the save raises — the DCP shard is
already on disk, but the non-distributed state file is missing.

Practical implication: keep `extra=` values to primitives, lists,
dicts, and tensors. Don't stuff a live logger, process group, or
generator in there.

## See also

- [DCP model + optimizer](dcp-model.md) — what DCP handles that
  train state does not.
- [Auto-resume](auto-resume.md) — how the training loop reads
  `train_state.pt` on startup.
- [Training § Schedulers](../training/schedulers.md) — what's in
  `scheduler.state_dict()`.
- [Data § Stateful dataloader](../data/index.md) —
  why the current train.py doesn't pass the dataloader.
- [Metrics § WandB resume](../metrics-and-profiling/index.md) — how
  `wandb_run_id` in `extra` enables resumed W&B runs.
