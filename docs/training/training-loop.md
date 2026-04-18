# Training loop

Companion to [Data flow](../architecture/data-flow.md): where that page
maps the whole step onto one diagram,
[`scripts/train.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/scripts/train.py)
zooms into the two step bodies (PP vs non-PP), the conditional paths,
and the periodic work.

## Two step bodies

`scripts/train.py` has a single outer loop but two internal step bodies
selected on `pp_enabled = config.distributed.pp > 1`. The paths diverge
on how microbatching interacts with the communication pattern.

### Non-PP step (`pp_enabled is False`)

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
  are still alive (`scripts/train.py` lines 606-622).
- **Loss scaling** — `loss / grad_accum_steps` keeps the effective
  learning rate invariant to the accumulation factor.

### PP step (`pp_enabled is True`)

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
broadcast across the PP dimension for logging
(`scripts/train.py` lines 556-571).

## Gradient clipping and NaN check

After the step body (either branch):

```python
grad_norm = clip_grad_norm_(model, tc.grad_clip_norm)
if not nan_detector.check_loss(avg_loss, step):
    optimizer.zero_grad()
    if nan_detector.should_rollback:
        break
    step += 1; continue
```

`clip_grad_norm_` is the DTensor-aware wrapper from
[`kempnerforge.distributed.utils`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/distributed/utils.py) —
see [Gradient utilities](gradient-utilities.md).
`NaNDetector.check_loss` returns `False` on NaN / Inf, zeroes grads,
and escalates to `should_rollback` after
`nan_consecutive_max` bad steps (see
[Resilience](../resilience/index.md)).

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
without rewriting the scheduler.

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
| Checkpoint | `step % checkpoint.interval == 0` | `ckpt_mgr.save(step)`, `on_checkpoint_save` hook |
| Shutdown | SIGTERM / SIGUSR1 pending | `ckpt_mgr.save(emergency=True)`, break |

`nccl_health_check_interval = 0` disables the all-reduce probe — it is
off in every shipped config but worth enabling for long multi-node
runs. See [Resilience § NCCL health](../resilience/index.md).

## Entry-point setup

Before the loop:

1. `load_config(path, cli_args)` — TOML + CLI overrides into a
   `JobConfig` dataclass.
2. `init_distributed(config.distributed, seed=...)` — `dist.init_process_group`,
   `DeviceMesh`, seeded RNG.
3. `build_loss_fn(tc)` — loss registry lookup with optional z-loss wrap
   (see [Losses](losses.md)).
4. `build_parallel_model(...)` — architecture + full parallelism stack
   (see [Parallelism order](../architecture/parallelism-order.md)).
5. `build_optimizer(model, config.optimizer)` — decay grouping +
   registry lookup (see [Optimizers](optimizers.md)).
6. `build_scheduler(optimizer, config.scheduler, max_steps=tc.max_steps)` —
   warmup + decay LambdaLR (see [Schedulers](schedulers.md)).
7. `CheckpointManager(...)`, `resolve_resume_path(...)` — auto-resume
   from the `latest` symlink.
8. `MetricsTracker`, `HookRunner`, data pipeline, optional eval
   dataloader, optional profiler.

The full list with links lives in
[Data flow § Startup, once](../architecture/data-flow.md#startup-once).

## Shutdown

After the loop:

```python
prof.stop()
ckpt_mgr.wait()                        # drain last async save
hook_runner.on_train_end(step, tokens_seen)
tracker.close()
destroy_distributed()
```

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
