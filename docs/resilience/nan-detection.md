# NaN detection

[`NaNDetector`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/resilience/health.py)
watches for NaN / Inf in the loss and the gradient norm every step. When it fires
with the default `nan_action`, the training loop zeros gradients, skips the
optimizer step, and increments a consecutive-NaN counter; `"raise"` stops the
run instead (see [Three actions](#three-actions)). The skipped step still
counts as a step (see
[What a skipped step still does](#what-a-skipped-step-still-does)). If
too many consecutive NaNs pile up, the loop stops so a human can roll
back to an earlier checkpoint.

## Wiring

```python
# training/entry.py::run_training
nan_detector = NaNDetector(
    action=config.train.nan_action, max_consecutive=config.train.nan_max_consecutive
)
...
# Inside the training loop, after backward:
if not nan_detector.check_loss(result.loss, step, grad_norm=result.grad_norm):
    optimizer.zero_grad()
    if nan_detector.rollback_recommended:    # stops without a checkpoint; no rollback
        logger.error(f"Too many consecutive NaNs at step {step}: stopping ...")
        break
    step += 1
    _apply_step_transitions(session, step)   # FreezeStages and data phases due at step
    _advance_lr_schedule(session)            # scheduler.step(), then the phase lr_scale
    if _shutdown_if_requested(               # saves only finite weights
        session, step, tokens_seen, saved_this_step=False, nan_skipped=True
    ):
        break
    continue
```

Three things to note:

- **The action and the threshold come from `[train]`.** `nan_action`
  (default `"warn"`, see [Three actions](#three-actions)) and
  `nan_max_consecutive` (default `10`, see
  [Rollback recommendation](#rollback-recommendation)):

  ```toml
  [train]
  nan_action = "raise"         # fail fast on the first NaN / Inf step
  # or keep "warn" and tolerate a longer run of them:
  # nan_max_consecutive = 20
  ```
- **The gradient norm is checked with the loss.** A finite loss can come with
  non-finite gradients: a micro-batch whose labels are all `-100` has an exact
  zero cross-entropy loss whatever its activations, and without the norm check
  its NaN gradients would reach `optimizer.step()`. The norm goes into the same
  all-reduced flag as the loss, so every rank skips the step together even when
  a non-finite gradient reaches only some ranks' norms.
- **`check_gradients` is *not* called by the training loop.** The gradient
  norm check above covers non-finite gradients. Like `check_loss`, it is a
  collective that gives every rank the same verdict (see
  [Manual use](#manual-use)).

## What a skipped step still does

A skipped step has consumed its batch, so it counts as a step: the step
counter advances, and the LR schedule and the step-keyed transitions
move with it. The update, the per-step reporting and every periodic
tick except the shutdown check are dropped:

| Work | On a skipped step |
|------|-------------------|
| Optimizer update | Skipped. Gradients are zeroed and the sigmoid routers' expert counts dropped; weights, optimizer state and the routers' balancing state stay as they were. |
| `tokens_seen`, metrics, `on_step_end` hooks | Skipped. |
| `FreezeStage` and data phases due at the new step | Applied. A `FreezeStage` waits behind the same async-save fence as on a finite step. |
| LR schedule | Advanced: `scheduler.step()`, then the `lr_scale` of the phase active at the new step, so warmup and decay stay on the step counter. |
| Shutdown check | Run. A pending SIGTERM / SIGUSR1 writes the emergency checkpoint at the new step, unless the weights are not finite (see [A shutdown after a bad update](#a-shutdown-after-a-bad-update)), then the loop stops. |
| Eval, NCCL health check, scheduled checkpoint | Skipped until their next interval. |
| Profiler step | Skipped; the profiler steps on the next finite step. |

The checkpoint holds the weights as the skipped step left them, and the
schedule, freeze state, data phase and data position are all those of
the new step, so resuming from that emergency checkpoint continues as
the uninterrupted run would have, with one exception:

- **Partway through a run of NaN steps.** The consecutive-NaN count is
  not checkpointed, so the resumed run counts from zero and allows a
  full `max_consecutive` more before it stops. Tracked in
  [#480](https://github.com/KempnerInstitute/KempnerForge/issues/480).

A checkpoint written before the run's first optimizer update holds no
optimizer state. The resumed optimizer starts fresh, as the
uninterrupted run's still is at that step, and at the LR the restored
schedule sets for the step, times the active phase's `lr_scale`.

Two edge cases:

- **The NaN that trips the stop.** `rollback_recommended` is checked first,
  so that step ends the run without a checkpoint, even with a shutdown
  pending (see [Rollback recommendation](#rollback-recommendation)).
- **A fresh run whose first step is skipped.** PyTorch warns
  "Detected call of `lr_scheduler.step()` before `optimizer.step()`".
  That is expected: the skipped step uses up its LR value like any
  other step.

### A shutdown after a bad update

A run of NaN steps can start with a bad update: an optimizer step that
writes NaN / Inf into the weights although its loss and gradient norm
were finite, so the NaN check let it through, and every loss after it
is NaN. Non-finite gradients never get that far: they make the gradient
norm non-finite, and the check skips that step. The steps that follow
are skipped, but the weights they leave are not finite, and an
emergency checkpoint of one of them would become `latest` and count
toward `checkpoint.keep_last_n`. With `keep_last_n = 1` it would prune
the last good checkpoint, and the requeued run would resume from NaN
weights.

So before it saves a skipped step, the shutdown check calls
`params_are_finite` (`kempnerforge/resilience/health.py`): each rank
checks the parameters it holds, and one all-reduce gives every rank the
same answer. The check runs only when a shutdown is pending on a
skipped step. If any parameter is not finite, the loop logs an ERROR,
writes no checkpoint, finishes the shutdown and stops: `latest` still
names the last checkpoint saved, and no older one is pruned to make
room. A skipped step whose weights are finite, such as one whose NaN
came from the data, writes its emergency checkpoint as before. Like any
save, that checkpoint becomes `latest` and counts toward
`checkpoint.keep_last_n`, so with `keep_last_n = 1` it replaces the
checkpoint saved before the NaN run (see
[Rollback recommendation](#rollback-recommendation)).

The bad update's own step passed the NaN check, so a shutdown or a
scheduled save on that step is not checked and writes the weights it
produced.

## Three actions

```python
# kempnerforge/resilience/health.py — NaNDetector.__init__
if action not in ("warn", "skip", "raise"):
    raise ValueError(f"Invalid NaN action: {action!r} (expected warn/skip/raise)")
```

| Action | Behavior | When to pick it |
|--------|----------|-----------------|
| `"warn"` | Log a warning, return `False`. Loop zeros grads and continues. | Default. NaN-tolerant training where a single bad step shouldn't kill the run. |
| `"skip"` | Same as `"warn"` but logs `"; skipping optimizer step"` explicitly. | Same as warn; the two are nearly equivalent since the caller already skips the optimizer step on `False`. |
| `"raise"` | Raise `RuntimeError` immediately, on every rank (the raise comes after the cross-rank reduction). | Early development when you want the run to die loudly on first NaN. |

The training loop takes its action from `train.nan_action`, in the TOML
`[train]` table or as `--train.nan_action=raise` on the command line.
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
bad_loss = not _is_finite(loss)
bad_grad_norm = grad_norm is not None and not _is_finite(grad_norm)
local_nan = bad_loss or bad_grad_norm
any_nan = _any_rank_flagged(local_nan)

# _any_rank_flagged(flag)
if not dist.is_initialized():
    return flag
device = "cuda" if torch.cuda.is_available() else "cpu"
reduced = torch.tensor([1.0 if flag else 0.0], device=device)
dist.all_reduce(reduced)
return reduced.item() > 0
```

Four bytes per step — dwarfed by the gradient all-reduce. The
`all_reduce` is a `SUM` (the default) — any rank with NaN lifts the
flag above zero on every rank. The flag goes on the current CUDA device
when CUDA is available, so NCCL reduces it, and on the CPU otherwise,
so a gloo process group without CUDA (a CPU-only test) reduces it too.

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
rollback: when it reaches `max_consecutive` (default 5 in the class;
the training loop's detector takes `train.nan_max_consecutive`, default
10), `rollback_recommended` flips to `True`.

`nan_steps` is a diagnostic — a post-hoc "which steps actually failed"
list. Capped at 100 entries to bound memory on pathological runs.

## Rollback recommendation

```python
@property
def rollback_recommended(self) -> bool:
    return self.state.consecutive_nans >= self.max_consecutive
```

The property recommends a rollback and does none. It used to be named
`should_rollback`, which reads as if the detector rolled back;
`should_rollback` still works as a deprecated alias that emits a
`DeprecationWarning`.

When this trips, the training loop *stops* — it doesn't roll back
automatically:

```python
# training/loop.py::run_training_loop
if nan_detector.rollback_recommended:
    logger.error(
        f"Too many consecutive NaNs at step {step}: stopping without a "
        "checkpoint. Nothing rolls back on its own; to roll back, resubmit "
        "with a new, empty checkpoint.dir and checkpoint.load_path set to "
        "an earlier step_N"
    )
    break
```

Rolling back is manual: resubmit with a new, empty `checkpoint.dir` and
`checkpoint.load_path` pointing at an earlier `step_N` directory; with
`exclude_from_loading` empty this restores its full state. Its
`train_state.pt` restores the scheduler's base LR whatever
`exclude_from_loading` says, so a lower `optimizer.lr` in the resubmitted
config does not take effect; schedule settings such as
`scheduler.warmup_steps` do, from the restored step on. Keep the old
`checkpoint.dir` and the run does not roll back: a checkpoint there takes
precedence over `load_path`, so it resumes from the newest one, and rank 0
warns that `load_path` is not used. The reason it's not automatic is that
the *source* of the NaN determines what's safe:

- **LR spike** — lower the LR and restart from an earlier checkpoint. A
  rollback through `load_path` brings back the saved base LR, so a lower
  `optimizer.lr` does not take effect there. A longer
  `scheduler.warmup_steps` does, but it can raise the LR as well as
  lower it. One that ends before the restored step starts the decay
  later, which never lowers the LR there. One that extends past it
  scales the saved base LR by `step / warmup_steps` (with the default
  `warmup_start_factor = 0`), so right after the restored step it lowers
  the LR only when that ratio falls below the old schedule's factor
  there (its LR at that step over the base LR). Restoring a 1000-step
  cosine run (`optimizer.lr = 3e-4`, `min_lr_ratio = 0.1`, 10 warmup
  steps) at step 500 with a warmup of 100 or 800 raises the LR at step
  501 from 1.67e-4 to 1.88e-4; only a warmup longer than about 900 steps
  lowers it.
- **Bad data** — skip the offending shard, restart from the same
  checkpoint.
- **FP8 overflow** — reduce `distributed.fp8_interval` or disable FP8
  for sensitive layers, restart.

A rule-of-thumb: if you hit `rollback_recommended`, don't resume from the
most recent checkpoint. It holds the weights from just before the NaN
storm, so whatever state caused the explosion is baked in. That also
holds for an emergency checkpoint a shutdown wrote during the storm:
skipped steps leave the weights as they were, so it carries the
pre-storm weights under a later step. A storm that a bad update started
leaves no such checkpoint (see
[A shutdown after a bad update](#a-shutdown-after-a-bad-update)).

## Manual use

Call the detector outside the training loop for ad-hoc checks:

```python
from kempnerforge.resilience import NaNDetector

det = NaNDetector(action="raise", max_consecutive=1)  # fail fast
for step, batch in enumerate(loader):
    loss = model(batch).item()
    det.check_loss(loss, step)    # raises on first NaN
```

`check_gradients(model, step)` does the same for the gradients. Each
rank walks `model.named_parameters()` and checks the gradients it holds
(the local shard of a `DTensor`); when running distributed, one
all-reduce of a flag gives every rank the same verdict, so every rank
must call it. It returns `False` on every rank if any rank holds a
NaN / Inf gradient, and the `action="raise"` mode raises `RuntimeError`
on every rank instead. Unlike `check_loss`, it leaves the consecutive-NaN
count as it is.

`params_are_finite(model, device)` from
`kempnerforge.resilience.health` checks the parameters themselves and
returns `False` if any rank holds a NaN / Inf one. When running
distributed it all-reduces a flag on `device` (CUDA under NCCL), so
every rank must call it.

## Limitations

- **No gradient scan in the hot path.** `check_gradients` exists but
  isn't wired in. Add it if you're hunting a specific gradient
  pathology, on every rank; expect a small per-step cost (one
  `isfinite` + `.all()` per parameter, plus one all-reduce).
- **Loss is already a CPU scalar.** `check_loss` gets a Python float, so
  the distributed sync creates a new tensor on CUDA (on the CPU when CUDA
  is unavailable) and all-reduces it
  — a negligible one-off each step but not free. If you optimize this
  path, aggregate the NaN flag into the existing grad-norm all-reduce.

## See also

- [SLURM preemption](slurm-preemption.md) — the other "stop cleanly"
  mechanism; both rely on the training loop polling a flag between
  steps.
- [GPU health](gpu-health.md) — coarser health check; run it at
  startup and after any NCCL failure.
- [Checkpointing § Auto-resume](../checkpointing/auto-resume.md#loading-a-specific-step) —
  how to roll back to an earlier checkpoint.
