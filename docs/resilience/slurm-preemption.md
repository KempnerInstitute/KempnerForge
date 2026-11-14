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
shutdown_handler.register()
...
# training/loop.py::run_training_loop, on a step skipped for a NaN / Inf loss
if _shutdown_if_requested(
    session, step, tokens_seen, saved_this_step=False, nan_skipped=True
):
    break
...
# training/loop.py::run_training_loop, after a finite step's scheduled save
if _shutdown_if_requested(session, step, tokens_seen, saved_this_step=saved_this_step):
    break

# training/loop.py::_shutdown_if_requested
if not session.shutdown_handler.should_shutdown():
    return False
if saved_this_step:
    ckpt_mgr.wait()
elif nan_skipped and not params_are_finite(session.model, session.runtime.device):
    logger.error(...)
else:
    ckpt_mgr.save(step=step, tokens_seen=tokens_seen, scheduler=session.scheduler,
                  dataloader=session.data.dataloader, extra=...)
session.shutdown_handler.finish()
return True
```

`register()` installs handlers for `SIGTERM` and `SIGUSR1`. `finish()`
cancels the forced-exit timer and restores the previous signal handlers.
A shutdown on a step the checkpoint schedule already saved reuses that
save as the emergency checkpoint instead of writing the step twice, and
waits for it before `finish()`, so the forced-exit timer still bounds a
stalled write.

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
A checkpoint can be written safely, `ckpt_mgr.wait()` can flush the
async writer, and the process group can be destroyed cleanly.

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

`timeout_sec` comes from `train.shutdown_timeout_sec` (default 600s = 10
min). Set to 0 to disable the timer entirely — not recommended under
SLURM because you then rely on SLURM's own kill-timeout, which may
leave GPU state half-freed.

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
                  and starts its forced-exit timer (shutdown_timeout_sec)
  next step end   the ranks vote on should_shutdown() and all write the
                  emergency checkpoint at the same step (or reuse that
                  step's scheduled save), then stop; ckpt_mgr.wait()
                  flushes an async save
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

## Troubleshooting

**"Received SIGTERM" logged but no emergency checkpoint saved.** Usually
means SLURM killed the ranks before the save finished: the lead time was
shorter than the step in flight plus the save. Raise the `--signal` lead
or make the save faster. For `singlenode.sh` only a faster save helps,
because `torchrun` allows its workers 30 seconds whatever the lead. If
an ERROR says the parameters are not finite, the signal arrived during a
run of NaN steps whose weights a bad update had already ruined, and the
loop skipped the save on purpose; `latest` still names the last
checkpoint saved.

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
