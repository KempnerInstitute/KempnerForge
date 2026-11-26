# Training loop

Companion to [Data flow](../architecture/data-flow.md): where that page
maps the whole step onto one diagram, this one zooms into the step
bodies (PP vs non-PP), the conditional paths, and the periodic work.

The loop lives in
[`kempnerforge/training/loop.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/training/loop.py);
`scripts/train.py` is a thin CLI wrapper that loads the config and calls
`kempnerforge.training.entry.run_training`.

## Step bodies

`run_training_loop` has a single outer loop and calls one step body per
step — `text_step`, `vlm_step`, or `pipeline_step`, picked once by
`select_step_fn(config)`. The PP and non-PP paths diverge on how
microbatching interacts with the communication pattern.

### Non-PP step (`text_step`)

```python
for micro_step in range(tc.grad_accum_steps):
    batch = next(data_iter)
    with maybe_no_sync(model, micro_step, tc.grad_accum_steps):
        if mc.is_moe:
            model.set_moe_step(step, tc.max_steps)
        logits = model(input_ids, doc_ids=doc_ids)
        loss   = loss_fn(logits, labels)
        if mc.is_moe:
            loss = loss + mc.moe_aux_loss_weight * model.get_moe_aux_loss()
        (loss / tc.grad_accum_steps).backward()
```

Key mechanics:

- **`maybe_no_sync`** (see [Gradient utilities](gradient-utilities.md))
  disables FSDP2's reduce-scatter on all but the last microbatch. One
  collective fires per optimizer step instead of `grad_accum_steps`.
- **Per-dataset metrics** — if the dataloader is a `MixtureDataset`,
  a step that will be logged counts the positions drawn from each source
  and sums each source's cross-entropy over its labeled tokens inside
  the `no_sync` block while logits are still alive, in fp32, at most
  1024 tokens at a time (`ce_chunk_size` when it is smaller). Both
  return on the `StepResult` by source index and are summed over the
  data-parallel ranks when logged.
- **Loss scaling** — `loss / grad_accum_steps` keeps the effective
  learning rate invariant to the accumulation factor.

### PP step (`pipeline_step`)

```python
input_ids_list, labels_list = [], []
for _ in range(tc.grad_accum_steps):
    batch = next(data_iter)
    input_ids_list.append(batch["input_ids"].to(device))
    labels_list.append(batch["labels"].to(device))

full_input  = torch.cat(input_ids_list, dim=0)
full_labels = torch.cat(labels_list, dim=0)

if is_first: pp_schedule.step(full_input, target=full_labels, losses=pp_losses)
elif is_last: pp_schedule.step(target=full_labels, losses=pp_losses)
else:        pp_schedule.step()
```

Under PP, microbatches are collected up front and handed as one tensor
to the schedule (`1f1b` / `gpipe`, built by
[`build_pipeline_schedule`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/distributed/pipeline_parallel.py)).
The schedule splits along dim 0 internally; the Python loop only sees
one `step()` call. Loss is meaningful only on the last stage and is
broadcast across the PP dimension for logging. The gradient norm is not
broadcast: every stage clips with the whole model's norm, so each one
already has it.

## Gradient clipping and NaN check

After the step body (either branch):

```python
grad_norm = clip_grad_norm_(model, tc.grad_clip_norm)
if not nan_detector.check_loss(avg_loss, step, grad_norm=grad_norm):
    _route_balance(session, step_ran=False)
    optimizer.zero_grad()
    if nan_detector.rollback_recommended:
        break
    step += 1
    _apply_step_transitions(session, step)
    _advance_lr_schedule(session)
    if _shutdown_if_requested(
        session, step, tokens_seen, saved_this_step=False, nan_skipped=True
    ):
        break
    continue
```

`clip_grad_norm_` is the DTensor-aware wrapper from
[`kempnerforge.distributed.utils`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/distributed/utils.py) —
see [Gradient utilities](gradient-utilities.md).
Under PP, `pipeline_step` passes `pp_group=`, so the norm is summed over
the stages and every stage scales by the same coefficient (see
[Gradient utilities § Under pipeline parallelism](gradient-utilities.md#under-pipeline-parallelism)).
Under EP it reads the EP group from the model, so the norm covers every
rank's experts (see
[Gradient utilities § Under expert parallelism](gradient-utilities.md#under-expert-parallelism)).
`NaNDetector.check_loss` returns `False` when the loss or the gradient norm is
NaN / Inf on any rank; the loop then zeroes the gradients and skips the step.
The detector escalates to `rollback_recommended` after `max_consecutive` bad steps
in a row (`train.nan_max_consecutive`, default 10; see
[NaN detection](../resilience/nan-detection.md)).
The gradient norm is checked because a finite loss can come with
non-finite gradients: a micro-batch whose labels are all `-100` has an exact
zero cross-entropy loss whatever its activations.

A skipped step still counts as a step. It drops the optimizer update,
the token count, metrics, hooks and every periodic tick below except the
shutdown check, but the LR schedule advances, the freeze stages and data
phases due at the new step are applied, and a pending shutdown writes
the emergency checkpoint, if every rank's weights are finite, and stops
the loop. The step that trips `rollback_recommended` stops before any of
that. See
[NaN detection § What a skipped step still does](../resilience/nan-detection.md#what-a-skipped-step-still-does).

## Optimizer and scheduler step

```python
optimizer.step()
_route_balance(session, step_ran=True)
optimizer.zero_grad()
step += 1
_apply_step_transitions(session, step)   # freeze stages and data phases due at step
scheduler.step()
if phase_lr_scale != 1.0:
    for pg in optimizer.param_groups: pg["lr"] *= phase_lr_scale
```

With MoE, `_route_balance` applies the sigmoid routers' balancing right
after the update: one all-reduce sums their expert counts over the
step's micro-batches and all ranks, so every rank runs it (see
[Aux loss and balancing](../moe/aux-loss-and-balancing.md#sigmoid-router--bias-based-balancing)).
A NaN-skipped step drops the counts instead.

Phase LR scaling runs *after* the scheduler — it multiplies the base
LR that `scheduler.step()` just computed. This lets a curriculum phase
(see [Data](../data/index.md)) halve the LR for a cooldown segment
without rewriting the scheduler. Both lines live in
`_advance_lr_schedule`, which a NaN-skipped step also runs, so the
schedule stays on the step counter. It runs after the transitions of
the new step, so the step that starts at a phase's `start_step` runs at
that phase's `lr_scale` as well as on its weights.

Before the first step, an optimizer whose state no checkpoint restored
(`CheckpointManager.restored_optimizer` is False), as in a fresh run,
after a checkpoint saved before the first update, or after a
`load_path` warm start that leaves the optimizer out, gets the LR its
scheduler last set, restored with the checkpoint's schedule, times the
active phase's `lr_scale`. An optimizer restored with its state keeps
the LR it was saved at.

## Phase transitions

```python
while current_phase_idx < len(active_phases) \
        and step >= active_phases[current_phase_idx].start_step:
    phase = active_phases[current_phase_idx]
    phase_lr_scale = phase.lr_scale
    if mixture_dataset is not None:
        new_weights = [phase.dataset_weights.get(name, original_weights_dict[name])
                       for name in mixture_dataset.dataset_names]
        sampler.update_weights(new_weights, temperature=config.data.mix_temperature)
        data_iter = None   # force refresh so new weights take effect
    current_phase_idx += 1
```

`data_iter = None` forces a fresh iterator on the next microbatch —
without it, the already-materialized iterator would keep emitting
batches from the old weights for one more step. A run without a
mixture, on one source or a VLM run, keeps its iterator: its phases
set only `phase_lr_scale`.

## Metrics and hooks

Metrics fire first, hooks second:

```python
applied_lr = optimizer.param_groups[0]["lr"]   # read before optimizer.step()
...
loss = dp_mean(result.loss, session.dp_groups, runtime.device)
step_metrics = tracker.end_step(step=step, loss=loss,
                                grad_norm=result.grad_norm, lr=applied_lr,
                                tokens_in_step=tokens_in_step,
                                seq_len=result.seq_len)
hook_runner.on_step_end(StepContext(
    step=step, loss=loss, grad_norm=result.grad_norm, lr=applied_lr,
    tokens_seen=tokens_seen, model=model, optimizer=optimizer,
))
```

`loss` is the mean of the step's loss over the data-parallel ranks, the
loss whose gradient the update applied, on every rank. `lr` is the LR
the update used (for `schedule_free_adamw`, the rate before the warmup
the optimizer applies inside `step()`). The per-module `lr/<module>` and
`lr_adamw/<module>` metrics are read at the same point, so they describe
the same update. `tokens_in_step` sums the positions each
data-parallel rank ran, `result.tokens` (the configured batch for a step
body that reports none), in one reduction with the step's documents and
encoder images; a VLM step counts the image positions in the residual
stream as well as the text.

`StepContext` freezes the full step state for hooks that need to read
gradients or parameter values before the next iteration. See
[Hooks](hooks.md).

MoE-specific metrics (`moe/aux_loss`, per-expert token counts) are
logged immediately after, only when `step_metrics is not None` — that
is, only on `metrics.log_interval` boundaries.

## Periodic work

After the step body, before advancing:

| Tick | Trigger | What it does |
|------|---------|--------------|
| NCCL health | `step % tc.nccl_health_check_interval == 0` | Small all-reduce; if it failed on any rank, every rank stops without saving and the loop raises `RuntimeError` |
| Eval | `step % eval_config.interval == 0` | [`run_eval`](evaluation.md), `on_eval_end` hook |
| Profiler | every step | `prof.step()` advances the schedule |
| Checkpoint | `checkpoint.should_save(step)`: `dyn_ckpt_window` milestones inside the window, every `checkpoint.interval` steps outside it | `ckpt_mgr.save(step)`, logged as `checkpoint/save_sec`, then the `on_checkpoint_save` hook |
| Shutdown | SIGTERM / SIGUSR1 pending on any rank | `ckpt_mgr.save(step)` (`checkpoint/save_sec`), or `ckpt_mgr.wait()` when the checkpoint tick just saved this step (`checkpoint/wait_sec` if that save is still pending); then break |

The two stop rows are votes. Each row all-reduces the ranks' own
verdicts, a failed probe or a pending signal, over
`session.control_group`, a gloo group the loop makes when it starts, so
every rank stops at the same step or none does (see
[SLURM preemption](../resilience/slurm-preemption.md)). A rank the
shutdown vote stops arms its own forced-exit timer, and the loop leaves
the shutdown handler armed; `run_training` calls `finish()` only after
the final flush and the teardown. An NCCL stop saves nothing and fails
the run: once the last async save is drained, the loop raises
`RuntimeError` on every rank.

A step skipped for a NaN / Inf loss runs only the shutdown row, through
the same `_shutdown_if_requested` helper. The checkpoint tick does not
save a skipped step, so its shutdown writes its own emergency checkpoint,
but only after an all-reduced check that every rank's parameters are
finite: if a bad update has left NaN in them, it logs an ERROR and saves
nothing (see
[NaN detection § A shutdown after a bad update](../resilience/nan-detection.md#a-shutdown-after-a-bad-update)).

`nccl_health_check_interval = 0` disables the all-reduce probe — it is
off in every shipped config but worth enabling for long multi-node
runs. See [Resilience § NCCL health](../resilience/index.md).

None of this periodic work is inside the step's `step_time`, so `tok/s`
and MFU leave it out. Each checkpoint save (startup, scheduled,
emergency or final) logs the time the loop waited for it as
`checkpoint/save_sec`, and each drain that finds an async save pending
(the final `wait()`, the wait when a shutdown finds its step already
saved, a freeze stage's flush) logs `checkpoint/wait_sec`; a drain with
nothing pending logs nothing. A freeze stage's flush runs inside its
step, so it is also in that step's `step_time`.

## Entry-point setup

`kempnerforge.training.entry.run_training(config)` runs the build phases
in order, then hands a `TrainingSession` to `run_training_loop`:

1. `load_config(path, cli_args)` — TOML + CLI overrides into a
   `JobConfig` dataclass (done by `scripts/train.py`).
2. `setup_distributed(config)` — `JobConfig.validate(world_size)`, then
   `dist.init_process_group`, `DeviceMesh` and seeded RNG; returns a
   `RuntimeContext`.
3. `build_loss_fn(tc)` — loss registry lookup with optional z-loss wrap
   (see [Losses](losses.md)).
4. `build_model(config, runtime, loss_fn)` — architecture + full
   parallelism stack (see
   [Parallelism order](../architecture/parallelism-order.md)), plus the
   `PipelineBundle` when `distributed.pp > 1`.
5. `build_optimizer(model, config.optimizer)` — decay grouping +
   registry lookup (see [Optimizers](optimizers.md)).
6. `build_scheduler(optimizer, config.scheduler, max_steps=tc.max_steps)` —
   warmup + decay LambdaLR (see [Schedulers](schedulers.md)).
7. `build_checkpoint_manager(...)`, then `ckpt_mgr.claim_dir()`, which
   makes the run the owner of `checkpoint.dir` (see
   [Auto-resume § One run per checkpoint directory](../checkpointing/auto-resume.md#one-run-per-checkpoint-directory)),
   then `restore_checkpoint(...)`, which resumes from the `latest` symlink.
8. `MetricsTracker`, profiler, `build_data_pipeline`,
   `build_eval_dataloader`, `build_phase_state`, `HookRunner`.

The full list with links lives in
[Data flow § Startup, once](../architecture/data-flow.md#startup-once).

## Shutdown

At the end of `run_training_loop`, then in `run_training`:

```python
prof.stop()
# Clean off-schedule finish: persist the fully-trained final step
if completed_normally and not config.checkpoint.should_save(step):
    ckpt_mgr.save(step, ...)           # final checkpoint; `latest` committed after wait
ckpt_mgr.wait()                        # drain last async save
if nccl_failed_at is not None:
    raise RuntimeError(...)            # an NCCL stop fails the run
hook_runner.on_train_end(step, tokens_seen)
tracker.close()
ckpt_mgr.release_dir()
destroy_distributed()
shutdown_handler.finish()              # restore the handlers, cancel the timer
```

On a clean finish, an unconditional checkpoint is written at `max_steps`
when that step is not already on the save schedule — so a completed run's
fully-trained model (including the WSD decay tail) is always recoverable
and `latest` points at it. This mirrors the emergency checkpoint the
preemption path writes on shutdown. The `should_save` guard avoids a
duplicate when `max_steps` already coincided with the schedule. A run
whose last step is skipped for a NaN / Inf loss is not a clean finish:
it ends without this checkpoint, unless a pending shutdown saves that
step as its emergency checkpoint.

`ckpt_mgr.wait()` is load-bearing — without it, a rank can exit before
its async DCP write completes, corrupting the checkpoint for
everyone else on the same save. See
[Checkpointing § Async save](../checkpointing/index.md).

## See also

- [Data flow](../architecture/data-flow.md) — the same loop, as a
  single diagram.
- [Optimizers](optimizers.md), [Schedulers](schedulers.md),
  [Losses](losses.md) — the collaborators this loop composes.
- [Gradient utilities](gradient-utilities.md) — `maybe_no_sync`,
  `clip_grad_norm_`.
- [Hooks](hooks.md) — the extension points this loop fires.
