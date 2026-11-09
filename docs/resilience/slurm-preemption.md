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
   T-??s: Loop polls should_shutdown() → saves emergency checkpoint
          ckpt_mgr.wait() flushes async writer to disk
   T-??s: shutdown_handler.finish() cancels the forced-exit timer
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

## Troubleshooting

**"Received SIGTERM" logged but no emergency checkpoint saved.** Usually
means `shutdown_timeout_sec` was too short: the forced-exit timer fired
while the async checkpoint was still writing. Increase the timeout. If
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
