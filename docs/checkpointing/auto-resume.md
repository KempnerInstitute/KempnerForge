# Auto-resume

KempnerForge resumes a training run automatically on restart — no
flag to pass, no manual path to point at. Every SLURM requeue, every
preemption-and-restart, every "I killed the job and started it
again" just works, as long as a checkpoint directory exists.

## The resolution order

[`resolve_resume_path`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/resilience/elastic.py):

```python
# kempnerforge/resilience/elastic.py (abridged — logging calls elided)
def resolve_resume_path(checkpoint_dir: str) -> Path | None:
    base = Path(checkpoint_dir)
    if not base.exists():
        return None

    # 1. latest symlink
    latest = base / "latest"
    if latest.exists():
        resolved = latest.resolve()
        if resolved.exists():
            return resolved

    # 2. highest-numbered step_N directory
    step_dirs = sorted(
        (d for d in base.iterdir()
         if d.is_dir() and d.name.startswith("step_") and d.name.split("_")[1].isdigit()),
        key=lambda d: int(d.name.split("_")[1]),
    )
    if step_dirs:
        return step_dirs[-1]

    return None
```

Two fallbacks:

1. **`latest` symlink** — the canonical pointer, updated atomically
   after every successful save (see [Symlink updates](#symlink-updates)
   below). Used in all healthy cases.
2. **Highest `step_N` directory** — a safety net. If the symlink
   is missing (disk corruption, manual `rm`, never created because
   no save has completed), the resolver scans for the highest-numbered
   `step_N` directory and uses that.

If neither path finds anything, the function returns `None` and
training starts from step 0.

## Where it's called

`training/entry.py::restore_checkpoint` calls this **once**, right after creating the
`CheckpointManager`:

```python
resume_path = resolve_resume_path(config.checkpoint.dir)
step, tokens_seen = 0, 0
if resume_path or config.checkpoint.load_path:
    step, tokens_seen, ckpt_extra_loaded = ckpt_mgr.load(
        path=str(resume_path) if resume_path else None,
        scheduler=scheduler,
    )
    if ckpt_extra_loaded.get("wandb_run_id"):
        config.metrics.wandb_run_id = ckpt_extra_loaded["wandb_run_id"]
    if ckpt_extra_loaded.get("mlflow_run_id"):
        config.metrics.mlflow_run_id = ckpt_extra_loaded["mlflow_run_id"]
```

So two things can trigger a resume:

- **`resume_path` found** — auto-resume; user passed nothing.
- **`config.checkpoint.load_path` set** — explicit override; skip
  the symlink lookup and load from that path. Useful for loading
  pretrained weights, fine-tuning from a specific checkpoint, or
  debugging.

If both are set, `resume_path` wins (auto-resume takes precedence
over the static config). This is deliberate: SLURM requeues should
always pick up where they left off, not re-load the initial
`load_path` every time.
`exclude_from_loading` follows the same rule: it applies only to the
`load_path` warm start, so when a resume wins it is dropped with
`load_path` and a weights-only warm start silently becomes a full
resume. Use an empty `dir` to guarantee the warm start.

## Symlink updates

Inside `CheckpointManager.save` (rank 0 only):

```python
# manager.py
latest = self._latest_link()                # <dir>/latest
tmp_link = latest.with_suffix(".tmp")       # <dir>/latest.tmp
tmp_link.unlink(missing_ok=True)
tmp_link.symlink_to(ckpt_dir.name)          # relative link: "step_1000"
tmp_link.rename(latest)                     # atomic rename
```

Two details that matter:

- **Relative target** — the symlink points at `step_1000`, not
  `/abs/path/checkpoints/step_1000`. Checkpoint directories stay
  portable when moved or bind-mounted.
- **Atomic rename** — `tmp_link.rename(latest)` is a single atomic
  syscall (`rename(2)` on POSIX). Either the old symlink is still
  there or the new one is — never a half-written state. Safe against
  crashes mid-save.

The symlink is updated **after** the DCP save completes (async save
futures are resolved before the next save starts), so `latest` only
ever points at a fully-flushed checkpoint. Under pipeline parallelism
each stage saves through its own process group, so every rank first
waits at a barrier until every stage's save has completed; only then
does rank 0 move `latest` and prune.

## Retention cleanup

After updating the symlink, `CheckpointManager._cleanup()` keeps the
`config.checkpoint.keep_last_n` highest-numbered `step_*` directories
and removes the older ones, except two kinds it never removes:

- the checkpoint `latest` points at;
- every `dyn_ckpt_window` milestone, so the dense early checkpoints
  survive a finite `keep_last_n`.

Nothing else is protected. An async save's directory survives while its
flush runs only because it is normally the newest. When newer `step_*`
directories are present, as after the manual rewind below, retention
can delete it mid-save: the save fails, or the checkpoint is left
without its shards.

Default `keep_last_n = 3`. Set `keep_last_n = 0`, or any value `<= 0`,
to keep every checkpoint: cleanup then removes nothing.

A checkpoint it cannot remove is logged as a warning and left for the
next cleanup to retry. On NFS a file closed a moment earlier can leave
a `.nfs*` placeholder that makes the removal fail, and rank 0 must not
leave `save()` while the other ranks wait in its barrier.

## Edge cases

- **Empty checkpoint directory** — `resolve_resume_path` returns
  `None`, training starts from step 0. No error.
- **`latest` symlink points at a removed directory** — the resolver
  checks `resolved.exists()` after following the link; if it
  doesn't, it falls through to the highest-`step_N` scan. This
  catches the case where someone rm-rf'd a checkpoint but left the
  symlink.
- **Corrupted `step_N` directory** — `resolve_resume_path` doesn't
  verify the contents. If DCP can't load from the resolved path,
  `ckpt_mgr.load` raises and training aborts. In practice, async
  saves either complete or leave a truncated directory that DCP
  detects and errors on clearly.
- **Fresh checkpoint dir, `load_path` set** — `load_path` is used
  (the else branch). Training starts from the step in that
  checkpoint.
- **Multi-node NFS vs local disk** — the `latest` symlink lives on
  disk along with the shards. For a shared filesystem (Lustre, NFS)
  this just works. For local scratch, every node needs its own copy
  of the checkpoint directory, which KempnerForge does not manage
  automatically — use a shared filesystem.

## Loading a specific step

To rewind to an earlier checkpoint manually:

```bash
# Delete the symlink and let resolve_resume_path pick up step_5000
cd checkpoints
rm latest
ln -s step_5000 latest
```

Move the abandoned `step_*` directories newer than step 5000 out of
`checkpoints/` as well. Retention counts them among the newest, so they
take the `keep_last_n` places of the rerun's own checkpoints, and with
an async `async_mode` it can delete a directory a new save is still
writing.

Or override explicitly:

```toml
[checkpoint]
load_path = "checkpoints/step_5000"
```

The explicit override skips the symlink; SLURM requeues will still
try auto-resume first (finding nothing newer than step 5000, they'll
fall through to `load_path`).

## Resilience interaction

Auto-resume pairs with the SLURM preemption handler in
[`kempnerforge/resilience/`](../resilience/index.md):

1. SLURM sends `SIGTERM` on preemption.
2. `ShutdownHandler` flags shutdown; the training loop finishes the
   current step and saves an emergency checkpoint.
3. `ckpt_mgr.wait()` flushes any async save before `destroy_distributed()`.
4. SLURM requeues the job.
5. New job starts, `resolve_resume_path` finds `latest`, training
   resumes from the emergency checkpoint.

See [Resilience § Signal handling](../resilience/index.md).

## See also

- [DCP model + optimizer](dcp-model.md) — the checkpoint format
  that `ckpt_mgr.load` consumes.
- [Train state](train-state.md) — what gets restored alongside the
  model weights.
- [Resharding](resharding.md) — what auto-resume does when the
  GPU count changes between save and load.
- [Configuration § CheckpointConfig](../configuration/config-sections.md) —
  `load_path`, `keep_last_n`, `dir`.
- [Resilience](../resilience/index.md) — the preemption handler
  that drives the emergency save.
