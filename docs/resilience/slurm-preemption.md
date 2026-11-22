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
if not requested_here:
    session.shutdown_handler.request()
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
A rank that learns of the shutdown only from the vote calls
`request()`, which sets its flag and starts its forced-exit timer as a
signal would. The loop never calls `finish()`: `run_training` calls it
once the loop has flushed the emergency checkpoint with
`ckpt_mgr.wait()` and the process group is destroyed. Until then a
second signal only sets the flag again, and the forced-exit timer stays
armed (see Forced-exit timer below). `finish()` restores the previous
signal handlers and cancels the timer; `unregister()` and
`cancel_watchdog()` do one each. A shutdown on a step the checkpoint
schedule already saved reuses that save as the emergency checkpoint
instead of writing the step twice.

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
when it starts, unless the caller passed one in, and destroys when it
ends, so every rank must call the loop. It is not the default
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

Every rank that stops arms its timer: a signaled rank when the signal
arrives, and a rank that learned of the shutdown from the vote when it
calls `request()`. The timer stays armed until `run_training` calls
`finish()`, after the emergency checkpoint is flushed and the process
group is destroyed, so, when it is shorter than the time SLURM allows,
it bounds a hung flush or teardown too.

`timeout_sec` comes from `train.shutdown_timeout_sec` (default 600s = 10
min). Set to 0 to disable the timer entirely — not recommended under
SLURM because you then rely on SLURM's own kill-timeout, which may
leave GPU state half-freed. Under the shipped SLURM scripts the 600 s
default fires only after SLURM's SIGKILL; set it below the signal lead
if you want the timer to bound a hung save.

## SLURM side

```bash
# scripts/slurm/7b_requeue.sh
#SBATCH --signal=SIGTERM@120
#SBATCH --requeue
```

Two directives:

- **`--signal=SIGTERM@120`**: when the job is within 120 seconds of its
  time limit, SLURM sends `SIGTERM` to every job step, which here means
  every `srun`-launched rank. The batch shell is not signaled. SLURM's
  event handling can send it up to 60 seconds early, so the ranks get 120
  to 180 seconds. Each task runs `uv run python scripts/train.py`, and
  `uv run` forwards the signal to the rank; where SLURM signals every
  process of a step (`SignalChildrenProcesses=yes` in `cgroup.conf`, as
  on the Kempner cluster), the rank gets it directly as well. A second
  `SIGTERM` only sets the shutdown flag again.

  With the `B:` prefix (`--signal=B:SIGTERM@120`) SLURM signals the batch
  shell only and none of its children (`man sbatch`). The ranks under
  `srun` never see that signal, and a batch shell without a `trap` dies of
  it, which ends the job early. Forwarding the signal to `srun` does not
  help either: `srun` answers `SIGTERM` by sending `SIGKILL` to its tasks
  (`man srun`, under `--ignore-signals`).

  `scripts/slurm/singlenode.sh` is the exception. It runs `torchrun` in
  the batch shell, so the job has no step for SLURM to signal, and it
  keeps `--signal=B:SIGTERM@120`. It starts `uv run torchrun` in the
  background and traps `SIGTERM` and `SIGUSR1`. On the first of them it
  sends a `SIGTERM` to that `uv` process, which forwards it to `torchrun`.
  It does not forward later ones: `torchrun` restarts its shutdown on a
  second `SIGTERM` and can exit on a third without waiting for its
  workers. `torchrun` signals its workers and kills any still running 30
  seconds later, so a single-node run has 30 seconds for its emergency
  checkpoint, whatever the lead time. `torchrun` then exits with status 1,
  so `sacct` reports the job as failed even when the checkpoint was
  written; look in the log for `Checkpoint committed` for the step that
  `Shutdown requested at step N` names.

  Both launch shapes rely on `uv run` passing `SIGTERM` and `SIGUSR1` on to
  its child and exiting with the child's status (143 when the child dies
  of `SIGTERM`), which was verified with uv 0.9.16.

  The lead time has to cover the step in flight plus the emergency
  checkpoint, including the flush of an async save. Measure the save on
  your cluster before trusting 120 seconds for a large model, and raise
  the lead if it does not fit. Plan for the emergency save to finish
  before the time limit: at the limit SLURM sends `SIGTERM` again, and the
  ranks may be killed at once rather than after `KillWait` (30 seconds on
  the Kempner cluster), because `srun` itself can receive that `SIGTERM`
  and turn it into a `SIGKILL` of its tasks.

  The forced-exit timer only helps if it fires before SLURM's `SIGKILL`,
  so `shutdown_timeout_sec` must be shorter than the lead. The 600 s
  default is not, so under these scripts SLURM's kill always comes
  first. Deriving the timeout from the lead is not done yet.

- **`--requeue`**: if the job is preempted, SLURM puts it back in the
  queue. The new run reads
  [`resolve_resume_path`](../checkpointing/auto-resume.md) and resumes
  from the last checkpoint saved. SLURM increments
  `SLURM_RESTART_COUNT`, which `log_job_info()` prints. A job that reaches
  its time limit is not requeued; resubmit it, and it resumes the same
  way.

## The full shutdown timeline

At the time limit, for a script that launches the ranks with `srun`:

```
  T-180..T-120 s  SLURM sends SIGTERM to every job step; each rank gets it
                  through `uv run`, and directly where SLURM signals every
                  process of a step; a second SIGTERM only sets the flag
                  again. ShutdownHandler sets _shutdown_requested = True
                  and starts its forced-exit timer, due
                  shutdown_timeout_sec later (with the 600 s default that
                  is after SLURM's SIGKILL, so it never fires)
  next step end   the ranks vote on should_shutdown() and all write the
                  emergency checkpoint at the same step (or reuse that
                  step's scheduled save), then stop; a rank that learns of
                  the shutdown only from the vote arms its own timer with
                  request(); ckpt_mgr.wait() flushes an async save
  then            the process group is destroyed; shutdown_handler.finish()
                  restores the handlers and cancels the timer; srun returns
                  0 and the job completes before its limit
  T-0             a rank still running gets SIGTERM again and may be killed
                  at once; plan for the save to finish before this point
```

On preemption there is no lead. A partition without a preemption grace
time (`GraceTime=0`, as on every Kempner partition) terminates the job at
once with SLURM's kill sequence, `SIGTERM` and then `SIGKILL` after
`KillWait` at the latest, and `--requeue` puts it back in the queue. What
a requeued job loses is bounded by `checkpoint.interval`, not by the
`--signal` lead.

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

`scancel` sends `USR1` to the job steps only. `scripts/slurm/singlenode.sh`
has none, so stop it with `scancel --batch --signal=USR1 <JOBID>`: its
trap turns that into one `SIGTERM` for `uv run torchrun`, with the
30-second limit described above.

Native code can replace the handlers below Python, where
`signal.getsignal` still reports KempnerForge's. The LLVM inside Triton
does when it compiles a kernel in the training process that Triton's
cache does not already hold (with `train.compile_model` or
FlexAttention), after which a `SIGUSR1` does nothing; a `SIGTERM` still
gets through, because LLVM's handler hands it back. The loop therefore
reinstalls the handlers at every step, in `_shutdown_if_requested`, so
only a `SIGUSR1` that arrives between such a compile and the end of its
step is lost. Only the main thread can install a handler, so a loop run
on another thread skips the reinstall.

## Troubleshooting

**"Received SIGTERM" logged but no emergency checkpoint saved.** Usually
means SLURM killed the ranks before the save and its flush finished:
they took longer than the `--signal` lead, plus `KillWait` where SLURM
waits that long after the limit. Raise the `--signal` lead or make the
save faster. For `singlenode.sh` only a faster save helps, because
`torchrun` allows its workers 30 seconds whatever the lead. With
`shutdown_timeout_sec` below that window, the forced exit can also end
a slow flush; the log then shows `Graceful shutdown timed out`. If an
ERROR says the parameters are not finite, the signal arrived during a
run of NaN steps whose weights a bad update had already ruined, and the
loop skipped the save on purpose; `latest` still names the last
checkpoint saved.

**`SIGUSR1` sent but nothing logged.** If it arrived during a step in
which Triton compiled a kernel in the training process, usually the
first step of a compiled run, Triton's LLVM handler may have swallowed
it before the loop reinstalled its own (see `SIGUSR1` above). Send it
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
