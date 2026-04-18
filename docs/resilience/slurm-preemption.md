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

Wired into the training loop at two points:

```python
# scripts/train.py
shutdown_handler = ShutdownHandler(timeout_sec=config.train.shutdown_timeout_sec)
shutdown_handler.register()
...
while step < tc.max_steps:
    # ... train step, log, checkpoint ...
    if shutdown_handler.should_shutdown():
        logger.warning(f"Shutdown requested at step {step} — saving emergency checkpoint")
        ckpt_mgr.save(step=step, tokens_seen=tokens_seen,
                      scheduler=scheduler, extra=ckpt_extra)
        shutdown_handler.finish()
        break
```

`register()` installs handlers for `SIGTERM` and `SIGUSR1`. `finish()`
cancels the forced-exit timer and restores the previous signal handlers.

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

The training loop polls `should_shutdown()` after every step, at which
point no collective is in flight. A checkpoint can be written safely,
`ckpt_mgr.wait()` can flush the async writer, and the process group can
be destroyed cleanly.

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
while the async checkpoint was still writing. Increase the timeout.

**Process hangs after "Shutdown requested".** Check for an in-flight
NCCL collective that can't complete (peer already killed).
`check_nccl_health()` from [NCCL liveness](nccl-liveness.md) is useful
for detecting this proactively.

**Checkpoint exists but resume starts from step 0.** Inspect
`checkpoint/latest` — the symlink may be stale if a previous save
crashed mid-write. See [Auto-resume](../checkpointing/auto-resume.md)
for the resolution order.

## See also

- [NaN detection](nan-detection.md) — in-training failure mode; also
  uses a "stop cleanly" pattern.
- [NCCL liveness](nccl-liveness.md) — detect hung collectives before
  they make preemption-handling fail.
- [Auto-resume](../checkpointing/auto-resume.md) — what the *next* run
  does after requeue.
- [`scripts/slurm/7b_requeue.sh`](https://github.com/KempnerInstitute/KempnerForge/blob/main/scripts/slurm/7b_requeue.sh)
  — preemption-resilient launch script.
