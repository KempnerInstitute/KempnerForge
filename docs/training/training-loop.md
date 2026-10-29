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
  per-dataset loss is computed inside the `no_sync` block while logits
  are still alive, and returned on the `StepResult`.
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
if not nan_detector.check_loss(avg_loss, step):
    optimizer.zero_grad()
    if nan_detector.should_rollback:
        break
    _advance_lr_schedule(session)
    step += 1
    _apply_step_transitions(session, step)
    if _shutdown_if_requested(session, step, tokens_seen, saved_this_step=False):
        break
    continue
```

`clip_grad_norm_` is the DTensor-aware wrapper from
[`kempnerforge.distributed.utils`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/distributed/utils.py) —
see [Gradient utilities](gradient-utilities.md).
Under PP, `pipeline_step` passes `pp_group=`, so the norm is summed over
the stages and every stage scales by the same coefficient (see
[Gradient utilities § Under pipeline parallelism](gradient-utilities.md#under-pipeline-parallelism)).
`NaNDetector.check_loss` returns `False` on NaN / Inf, and the loop
zeroes the grads. The detector escalates to `should_rollback` after
`max_consecutive` bad steps in a row (`NaNDetector(max_consecutive=10)`
in `kempnerforge/training/entry.py`; see
[NaN detection](../resilience/nan-detection.md)).

A skipped step still counts as a step. It drops the optimizer update,
the token count, metrics, hooks and every periodic tick below except the
shutdown check, but the LR schedule advances, the freeze stages and data
phases due at the new step are applied, and a pending shutdown writes
the emergency checkpoint and stops the loop. The step that trips
`should_rollback` stops before any of that. See
[NaN detection § What a skipped step still does](../resilience/nan-detection.md#what-a-skipped-step-still-does).

## Optimizer and scheduler step

```python
optimizer.step()
scheduler.step()
if phase_lr_scale != 1.0:
    for pg in optimizer.param_groups: pg["lr"] *= phase_lr_scale
optimizer.zero_grad()
```

Phase LR scaling runs *after* the scheduler — it multiplies the base
LR that `scheduler.step()` just computed. This lets a curriculum phase
(see [Data](../data/index.md)) halve the LR for a cooldown segment
without rewriting the scheduler. Both lines live in
`_advance_lr_schedule`, which a NaN-skipped step also runs, so the
schedule stays on the step counter.

## Phase transitions

```python
while current_phase_idx < len(active_phases) \
        and step >= active_phases[current_phase_idx].start_step:
    phase = active_phases[current_phase_idx]
    new_weights = [phase.dataset_weights.get(name, original_weights_dict[name])
                   for name in mixture_dataset.dataset_names]
    sampler.update_weights(new_weights, temperature=config.data.mix_temperature)
    phase_lr_scale = phase.lr_scale
    current_phase_idx += 1
    data_iter = None   # force refresh so new weights take effect
```

`data_iter = None` forces a fresh iterator on the next microbatch —
without it, the already-materialized iterator would keep emitting
batches from the old weights for one more step.

## Metrics and hooks

Metrics fire first, hooks second:

```python
step_metrics = tracker.end_step(step=step, loss=avg_loss,
                                grad_norm=grad_norm_val, lr=current_lr,
                                tokens_in_step=tokens_in_step)
hook_runner.on_step_end(StepContext(
    step=step, loss=avg_loss, grad_norm=grad_norm_val, lr=current_lr,
    tokens_seen=tokens_seen, model=model, optimizer=optimizer,
))
```

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
| NCCL health | `step % tc.nccl_health_check_interval == 0` | Small all-reduce; break on failure |
| Eval | `step % eval_config.interval == 0` | [`run_eval`](evaluation.md), `on_eval_end` hook |
| Profiler | every step | `prof.step()` advances the schedule |
| Checkpoint | `checkpoint.should_save(step)`: `dyn_ckpt_window` milestones inside the window, every `checkpoint.interval` steps outside it | `ckpt_mgr.save(step)`, `on_checkpoint_save` hook |
| Shutdown | SIGTERM / SIGUSR1 pending | `ckpt_mgr.save(step)`, or `ckpt_mgr.wait()` when the checkpoint tick just saved this step; then `finish()`, break |

A step skipped for a NaN / Inf loss runs only the shutdown row, through
the same `_shutdown_if_requested` helper. The checkpoint tick does not
save a skipped step, so its shutdown always writes its own emergency
checkpoint.

`nccl_health_check_interval = 0` disables the all-reduce probe — it is
off in every shipped config but worth enabling for long multi-node
runs. See [Resilience § NCCL health](../resilience/index.md).

## Entry-point setup

`kempnerforge.training.entry.run_training(config)` runs the build phases
in order, then hands a `TrainingSession` to `run_training_loop`:

1. `load_config(path, cli_args)` — TOML + CLI overrides into a
   `JobConfig` dataclass (done by `scripts/train.py`).
2. `setup_distributed(config)` — `dist.init_process_group`, `DeviceMesh`,
   seeded RNG, world-size validation; returns a `RuntimeContext`.
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
7. `build_checkpoint_manager(...)` + `restore_checkpoint(...)` —
   auto-resume from the `latest` symlink.
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
hook_runner.on_train_end(step, tokens_seen)
tracker.close()
destroy_distributed()
```

On a clean finish, an unconditional checkpoint is written at `max_steps`
when that step is not already on the save schedule — so a completed run's
fully-trained model (including the WSD decay tail) is always recoverable
and `latest` points at it. This mirrors the emergency checkpoint the
preemption path writes on shutdown. The `should_save` guard avoids a
duplicate when `max_steps` already coincided with the schedule.

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
