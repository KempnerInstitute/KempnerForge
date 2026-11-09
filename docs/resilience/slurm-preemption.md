# SLURM preemption

Long training runs on shared clusters get preempted. When SLURM sends a
termination signal, KempnerForge catches it, saves a checkpoint, and
exits cleanly so the next job can auto-resume from that checkpoint.

Two pieces:

- **`ShutdownHandler`** in
  [`kempnerforge/resilience/signal_handler.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/resilience/signal_handler.py)
  — registers signal handlers, flips a flag, and enforces a timeout.
- **`scripts/slurm/7b_requeue.sh`** — the SLURM batch script that asks
  for the early signal and requests auto-requeue.

## `ShutdownHandler`

Registered in `run_training`, and checked by `_shutdown_if_requested`
at the end of every step, finite or skipped:

```python
# training/entry.py::run_training
shutdown_handler = ShutdownHandler(timeout_sec=config.train.shutdown_timeout_sec)
try:
    shutdown_handler.register()
    ...
    run_training_loop(session, step=step, tokens_seen=tokens_seen)
    finished = True
finally:
    ...
    if finished:
        destroy_distributed()
        shutdown_handler.finish()

# training/loop.py::run_training_loop, on a step skipped for a NaN / Inf loss
if _shutdown_if_requested(
    session, step, tokens_seen, saved_this_step=False, nan_skipped=True
):
    break
...
# training/loop.py::run_training_loop, after a finite step's scheduled save
if _shutdown_if_requested(session, step, tokens_seen, saved_this_step=saved_this_step):
    break
...
# training/loop.py::run_training_loop, after the loop
ckpt_mgr.wait()

# training/loop.py::_shutdown_if_requested
session.shutdown_handler.reinstall()
requested_here = session.shutdown_handler.should_shutdown()
if not any_rank(requested_here, session.control_group):
    return False
if saved_this_step:
    ckpt_mgr.wait()
elif nan_skipped and not params_are_finite(session.model, session.runtime.device):
    logger.error(...)
else:
    ckpt_mgr.save(step=step, tokens_seen=tokens_seen, scheduler=session.scheduler,
                  dataloader=session.data.dataloader, extra=...)
return True
```

`register()` installs handlers for `SIGTERM` and `SIGUSR1`, and
`reinstall()` installs them again at every step (see `SIGUSR1` below).
The loop never calls `finish()`: `run_training` calls it once the loop
has flushed the emergency checkpoint with `ckpt_mgr.wait()` and the
process group is destroyed. Until then a second signal only sets the
flag again, and the forced-exit timer still bounds a stalled write or
teardown. `finish()` restores the previous signal handlers and cancels
the timer; `unregister()` and `cancel_watchdog()` do one each. A
shutdown on a step the checkpoint schedule already saved reuses that
save as the emergency checkpoint instead of writing the step twice.

A step skipped for a NaN / Inf loss runs the same check, so a signal
that arrives during a run of NaN steps is acted on at the next one
rather than after the run. The schedule never saves a skipped step, so
its shutdown writes the emergency checkpoint itself, and the checkpoint
resumes on the same LR schedule and data position, with two exceptions
listed in [NaN detection](nan-detection.md#what-a-skipped-step-still-does).
It writes one only if every rank's weights are finite: when a bad update
(an optimizer step that passed the NaN check) has already put NaN into
the weights, the loop logs an ERROR, saves nothing and stops, so
`latest` still names the last checkpoint saved. That checkpoint holds
NaN weights only if the bad update's own step saved one: that step
passed the NaN check, so its save was not checked (see
[NaN detection § A shutdown after a bad update](nan-detection.md#a-shutdown-after-a-bad-update)).
The NaN that reaches `max_consecutive` is different: the loop stops on
it before the check and saves nothing.

### Why cooperative, not `sys.exit`

When the signal arrives, the handler does **not** call `sys.exit` or
`os._exit`. It just sets a flag:

```python
# kempnerforge/resilience/signal_handler.py — _handle_signal
self._shutdown_requested = True
self._signal_received = sig
```

Reason: exiting mid-collective leaves NCCL buffers and open file
descriptors in an inconsistent state. On the next requeue, the new
process can inherit stale state (shared memory, port bindings) or hit a
torch.distributed deadlock because peers think the group is still
active.

The training loop polls `should_shutdown()` after every step, a
NaN-skipped one included, at which point no collective is in flight.
The flag is per rank, though, and a signal can reach the ranks at
different times, or only some of them. A rank that acted on its own
flag would enter the checkpoint's collectives while the others went on
to the next step's, and the job would hang without a checkpoint. So the
ranks vote first: `any_rank` all-reduces their flags over
`session.control_group`, and every rank stops if any rank was signaled.
A rank that learns of the shutdown this way logs `Shutdown requested on
another rank at step N`, so rank 0's log shows the shutdown even when
only another rank was signaled. Then a checkpoint can be written
safely, `ckpt_mgr.wait()` can flush the async writer, and the process
group can be destroyed cleanly.

`session.control_group` is a gloo group that `run_training_loop` makes
when it starts, so every rank must call the loop. It is not the default
process group: DCP's async save issues gloo collectives on that group
from a background thread, and a main-thread all-reduce there could run
in a different order on different ranks. On a gloo group of its own,
the vote also completes when NCCL is wedged. The NCCL liveness check
votes the same way (see [NCCL liveness](nccl-liveness.md)).

### Forced-exit timer

Cooperative shutdown assumes the loop reaches the polling point. If a
collective hangs (network drops, peer dies), it won't. A daemon thread
enforces a hard limit:

```python
# kempnerforge/resilience/signal_handler.py — _handle_signal
if self._timeout_sec > 0 and self._timer is None:
    self._timer = threading.Timer(self._timeout_sec, self._force_exit)
    self._timer.daemon = True
    self._timer.start()

# _force_exit
os._exit(1)
```

`os._exit(1)` is the blunt-force option — skips Python atexit handlers,
skips C++ destructors, kills the process immediately. That's intentional:
at this point graceful didn't work, and leaving a zombie holding GPU
memory would block the requeue.

The timer stays armed until `run_training` calls `finish()`, after the
emergency checkpoint is flushed and the process group is destroyed, so
it bounds a hung flush or teardown too. Only a rank that received a
signal arms it. If the shutdown hangs, a rank that stopped because
another rank was signaled goes down with the job when that rank's timer
fires: the SLURM scripts launch with `srun --kill-on-bad-exit=1`, and
`torchrun` stops the other workers when one fails.

`timeout_sec` comes from `train.shutdown_timeout_sec` (default 600s = 10
min). Set to 0 to disable the timer entirely — not recommended under
SLURM because you then rely on SLURM's own kill-timeout, which may
leave GPU state half-freed.

## SLURM side

```bash
# scripts/slurm/7b_requeue.sh
#SBATCH --signal=B:SIGTERM@120
#SBATCH --requeue
```

Two directives:

- **`--signal=B:SIGTERM@120`** — SLURM sends `SIGTERM` to the batch
  script's bash shell 120 seconds before the hard kill. The `B:` prefix
  means "signal the batch process"; without it the signal goes only to
  the allocation's `srun` child and bash (and therefore our Python)
  might miss it.

  `srun` then forwards the signal to each task's Python process. The
  120-second lead time must exceed `shutdown_timeout_sec` plus the time
  for the in-flight step + checkpoint to finish — 600s default is safe
  for most configs; reduce to ~90s if your checkpoints save quickly and
  you want to maximize step time.

- **`--requeue`** — if the job is preempted, SLURM re-submits it. The
  new run reads
  [`resolve_resume_path`](../checkpointing/auto-resume.md) and picks up
  from the emergency checkpoint written in the previous run. SLURM
  increments `SLURM_RESTART_COUNT` — visible in `log_job_info()` output.

## The full preemption timeline

```
  T-120s: SLURM sends SIGTERM to batch shell → srun → all Python ranks
  T-120s: ShutdownHandler sets _shutdown_requested = True
          ShutdownHandler starts 600s forced-exit timer (T+480s)
  T-120s: Training loop in mid-step — completes the step
   T-??s: Ranks vote on should_shutdown() → all save the emergency
          checkpoint at the same step
          ckpt_mgr.wait() flushes async writer to disk
   T-??s: Process group destroyed; shutdown_handler.finish() restores
          the handlers and cancels the forced-exit timer
   T-??s: Process exits cleanly, SLURM epilogue runs
     T-0: SLURM's SIGTERM deadline — if still alive, hard SIGKILL
          (in practice we exit well before this because async save is fast)
   T+??: SLURM sees job failed; `--requeue` re-submits
          new job starts, resolve_resume_path finds the checkpoint
          training resumes from step N+1
```

## `SIGUSR1`

Intercepted identically to `SIGTERM`. Useful for a "nicely save and
exit" command:

```bash
scancel --signal=USR1 <JOBID>
```

Forces a clean shutdown + checkpoint without relying on preemption
timing. This is the recommended way to stop a run manually — don't
`scancel` without a signal, because default `scancel` sends `SIGTERM`
with a short grace period and may skip the emergency checkpoint if the
step is long.

Native code can replace the handlers below Python, where
`signal.getsignal` still reports KempnerForge's. The LLVM inside Triton
does on its first in-process compile (with `train.compile_model` or
FlexAttention), after which a `SIGUSR1` does nothing; a `SIGTERM` still
gets through, because LLVM's handler hands it back. The loop therefore
reinstalls the handlers at every step, in `_shutdown_if_requested`, so
only a `SIGUSR1` that arrives between that first compile and the end of
its step is lost.

## Troubleshooting

**"Received SIGTERM" logged but no emergency checkpoint saved.** Usually
means `shutdown_timeout_sec` was too short: the forced-exit timer fired
while the async checkpoint was still writing. Increase the timeout. If
an ERROR says the parameters are not finite, the signal arrived during a
run of NaN steps whose weights a bad update had already ruined, and the
loop skipped the save on purpose; `latest` still names the last
checkpoint saved.

**`SIGUSR1` sent but nothing logged.** If it arrived during the first
step of a compiled run, Triton's LLVM handler may have swallowed it
before the loop reinstalled its own (see `SIGUSR1` above). Send it
again.

**Process hangs after "Shutdown requested".** Check for an in-flight
NCCL collective that can't complete (peer already killed).
`check_nccl_health()` from [NCCL liveness](nccl-liveness.md) is useful
for detecting this proactively.

**Checkpoint exists but resume starts from step 0.** Look for
`Auto-resume: skipping` warnings in the log. Without a `latest`
symlink, the resolver skips every `step_N` whose save did not complete
(no DCP `.metadata`, or optimizer state without `train_state.pt`), and
when none completed it starts from step 0, or from
`checkpoint.load_path` if one is set, with an ERROR. See
[Auto-resume](../checkpointing/auto-resume.md) for the resolution order.

## See also

- [NaN detection](nan-detection.md) — in-training failure mode; also
  uses a "stop cleanly" pattern.
- [NCCL liveness](nccl-liveness.md) — detect hung collectives before
  they make preemption-handling fail.
- [Auto-resume](../checkpointing/auto-resume.md) — what the *next* run
  does after requeue.
- [`scripts/slurm/7b_requeue.sh`](https://github.com/KempnerInstitute/KempnerForge/blob/main/scripts/slurm/7b_requeue.sh)
  — preemption-resilient launch script.
