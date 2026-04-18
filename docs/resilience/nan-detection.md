# NaN detection

[`NaNDetector`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/resilience/health.py)
watches for NaN / Inf in the loss every step. When it fires, the
training loop zeros gradients, skips the optimizer step, and increments
a consecutive-NaN counter. If too many consecutive NaNs pile up, the
loop stops so a human can roll back to an earlier checkpoint.

## Wiring

```python
# scripts/train.py
nan_detector = NaNDetector(action="warn", max_consecutive=10)
...
# Inside the training loop, after backward:
if not nan_detector.check_loss(avg_loss, step):
    optimizer.zero_grad()
    if nan_detector.should_rollback:
        logger.error("Too many consecutive NaNs — stopping")
        break
    step += 1
    continue
```

Two things to note:

- **`action="warn"` and `max_consecutive=10` are hardcoded in
  `scripts/train.py`** — not exposed as TOML config. If you want
  `"skip"` or `"raise"` behavior, edit the script or construct the
  detector yourself.
- **`check_gradients` is *not* called by the training loop.** Gradient
  NaNs typically manifest as loss NaNs on the next step anyway; skip it
  unless you're specifically debugging a gradient-explosion case.

## Three actions

```python
# kempnerforge/resilience/health.py — NaNDetector.__init__
if action not in ("warn", "skip", "raise"):
    raise ValueError(f"Invalid NaN action: {action!r} (expected warn/skip/raise)")
```

| Action | Behavior | When to pick it |
|--------|----------|-----------------|
| `"warn"` | Log a warning, return `False`. Loop zeros grads and continues. | Default. NaN-tolerant training where a single bad step shouldn't kill the run. |
| `"skip"` | Same as `"warn"` but logs `"— skipping optimizer step"` explicitly. | Same as warn; the two are nearly equivalent since the caller already skips the optimizer step on `False`. |
| `"raise"` | Raise `RuntimeError` immediately. | Early development when you want the run to die loudly on first NaN. |

In the shipped training loop, `"warn"` and `"skip"` produce the same
outcome (the caller already zeros grads and advances the step). The
distinction exists for callers that only call `check_loss` and let the
return value drive their own logic.

## Cross-rank sync

The critical detail. On a distributed run, a NaN on one rank must stop
*every* rank — otherwise one rank zeros grads while the others keep
optimizing and FSDP gets an inconsistent view of the parameter sharding
on the next step.

```python
# kempnerforge/resilience/health.py — check_loss
local_nan = not _is_finite(loss)
if dist.is_initialized():
    nan_flag = torch.tensor([1.0 if local_nan else 0.0], device="cuda")
    dist.all_reduce(nan_flag)
    any_nan = nan_flag.item() > 0
else:
    any_nan = local_nan
```

Four bytes per step — dwarfed by the gradient all-reduce. The
`all_reduce` is a `SUM` (the default) — any rank with NaN lifts the
flag above zero on every rank.

If `any_nan` is true but `local_nan` is false, the log line mentions
"detected on another rank" so you can correlate which rank blew up from
per-rank logs.

## State tracking

```python
@dataclass
class NaNState:
    consecutive_nans: int = 0       # reset on a good step
    total_nans: int = 0             # monotonic across the run
    last_good_loss: float = inf     # last finite loss value
    last_good_step: int = 0
    nan_steps: list[int] = []       # capped at max_history (default 100)
```

`consecutive_nans` resets on any finite step. It's the one that drives
rollback: when it reaches `max_consecutive` (default 5 in the class, 10
in the shipped config), `should_rollback` flips to `True`.

`nan_steps` is a diagnostic — a post-hoc "which steps actually failed"
list. Capped at 100 entries to bound memory on pathological runs.

## Rollback recommendation

```python
@property
def should_rollback(self) -> bool:
    return self.state.consecutive_nans >= self.max_consecutive
```

When this trips, the training loop *stops* — it doesn't roll back
automatically:

```python
# scripts/train.py
if nan_detector.should_rollback:
    logger.error("Too many consecutive NaNs — stopping")
    break
```

Rolling back is manual: resubmit with `checkpoint.load_path` pointing at
an earlier `step_N` directory. The reason it's not automatic is that
the *source* of the NaN determines what's safe:

- **LR spike** — reduce `optimizer.lr` or `scheduler.warmup_steps`,
  restart from an earlier checkpoint.
- **Bad data** — skip the offending shard, restart from the same
  checkpoint.
- **FP8 overflow** — reduce `distributed.fp8_interval` or disable FP8
  for sensitive layers, restart.

A rule-of-thumb: if you hit `should_rollback`, don't resume from the
most recent checkpoint. It was written just before the NaN storm, so
whatever state caused the explosion is baked in.

## Manual use

Call the detector outside the training loop for ad-hoc checks:

```python
from kempnerforge.resilience import NaNDetector

det = NaNDetector(action="raise", max_consecutive=1)  # fail fast
for step, batch in enumerate(loader):
    loss = model(batch).item()
    det.check_loss(loss, step)    # raises on first NaN
```

`check_gradients(model, step)` does the same but walks
`model.named_parameters()` and returns `False` on the first NaN grad.
The `action="raise"` mode raises `RuntimeError` instead of returning
`False` (the warning case still returns `False`).

## Limitations

- **NaN action isn't in config.** Fixed at `"warn"` / `max_consecutive=10`
  in `scripts/train.py:85`. Change the source if you need something
  different.
- **No gradient scan in the hot path.** `check_gradients` exists but
  isn't wired in. Add it if you're hunting a specific gradient
  pathology; expect a small per-step cost (one `isfinite` + `.all()`
  per parameter).
- **Loss is already a CPU scalar.** `check_loss` gets a Python float, so
  the distributed sync creates a new tensor on CUDA and all-reduces it
  — a negligible one-off each step but not free. If you optimize this
  path, aggregate the NaN flag into the existing grad-norm all-reduce.

## See also

- [SLURM preemption](slurm-preemption.md) — the other "stop cleanly"
  mechanism; both rely on the training loop polling a flag between
  steps.
- [GPU health](gpu-health.md) — coarser health check; run it at
  startup and after any NCCL failure.
- [Checkpointing § Auto-resume](../checkpointing/auto-resume.md) —
  where to point `checkpoint.load_path` when rolling back.
