# Data Flow

The path a batch of tokens takes from the dataloader to a committed
gradient update. Follow this page with
[`scripts/train.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/scripts/train.py)
open in another tab — line numbers move, but the structure below maps
one-to-one onto the training loop body.

## One-slide view

```
   ┌─ MemoryMappedDataset or MixtureDataset
   │
   ├─ DistributedSampler / MixtureSampler (rank-partitioned indices)
   │
   ├─ StatefulDataLoader ──► batch = {input_ids, labels, doc_ids?}
   │
   ▼
for micro_step in range(grad_accum_steps):       ── maybe_no_sync sets
    ├─ model.set_moe_step(step, max_steps)          DDP/FSDP grad-sync off
    ├─ logits = model(input_ids, doc_ids=doc_ids)   on all but last micro
    ├─ loss   = loss_fn(logits, labels)
    ├─ [+ moe_aux_loss_weight * model.get_moe_aux_loss()]
    └─ (loss / grad_accum_steps).backward()       ── gradient accumulation

grad_norm = clip_grad_norm_(model, grad_clip_norm)
if NaN: zero_grad, skip, maybe stop
optimizer.step() ; scheduler.step() ; optimizer.zero_grad()

tracker.end_step(step, loss, grad_norm, lr, tokens_in_step)
hook_runner.on_step_end(StepContext(...))
[every N steps] eval, NCCL health, profiler.step, ckpt_mgr.save
if shutdown_handler.should_shutdown(): save + break
```

## Startup, once

Before the loop starts,
[`scripts/train.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/scripts/train.py)
initializes the collaborators:

- **`init_distributed(config.distributed, seed=...)`** from
  [`kempnerforge.distributed.setup`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/distributed/setup.py)
  — reads `RANK` / `LOCAL_RANK` / `WORLD_SIZE` (torchrun) or
  `SLURM_PROCID` / `SLURM_NTASKS` (multi-node srun), calls
  `dist.init_process_group`, builds the `DeviceMesh`, seeds torch.
- **`ShutdownHandler`** from
  [`kempnerforge.resilience.signal_handler`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/resilience/signal_handler.py)
  — installs SIGTERM / SIGUSR1 handlers for cooperative SLURM
  preemption.
- **`NaNDetector`** from
  [`kempnerforge.resilience.health`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/resilience/health.py)
  — tracks consecutive NaN steps; `action="warn"` by default, escalates
  to rollback after `max_consecutive=10`.
- **Loss function** from the registry (`"cross_entropy"` or
  `"chunked_cross_entropy"`).
- **Model** via `build_parallel_model` — applies the full parallelism
  stack (see [Parallelism order](parallelism-order.md)).
- **Optimizer and scheduler** from their registries.
- **`CheckpointManager`** from
  [`kempnerforge.checkpoint.manager`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/checkpoint/manager.py)
  — owns DCP async save/load, latest-symlink, step_N directories.
- **`resolve_resume_path(config.checkpoint.dir)`** from
  [`kempnerforge.resilience.elastic`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/resilience/elastic.py)
  — follows the `latest` symlink or picks the highest `step_N`. If
  non-`None`, `ckpt_mgr.load(resume_path)` restores model, optimizer,
  scheduler, dataloader position, and RNG state before the loop
  starts.
- **`MetricsTracker`** from
  [`kempnerforge.metrics.tracker`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/metrics/tracker.py)
  — per-step metrics + EMA smoothing + WandB / TensorBoard backends.
- **Data pipeline** — `MemoryMappedDataset` or `MixtureDataset` or an
  HF streaming / eager dataset (config dispatches), wrapped by
  `DistributedSampler` or `MixtureSampler` and then by
  `StatefulDataLoader` from
  [`kempnerforge.data`](https://github.com/KempnerInstitute/KempnerForge/tree/main/kempnerforge/data).
- **Eval dataloader and `torch.profiler`** (optional).
- **Phase schedule** (optional) — for curriculum training, rebalances
  the mixture at `phase.start_step`.

## Inside the loop: one step

The step body is where the interesting routing happens.

### 1 · Microbatch fetch

```python
for micro_step in range(grad_accum_steps):
    batch = next(data_iter)
    input_ids = batch["input_ids"].to(device)
    labels    = batch["labels"].to(device)
    doc_ids   = batch.get("doc_ids").to(device) if "doc_ids" in batch else None
```

When `dataloader is None`, the loop generates random integer tokens —
useful for smoke-testing the parallelism stack without any corpus.

`doc_ids` is optional: non-`None` only when the dataset packs multiple
documents into one sequence. It triggers the block-diagonal attention
mask path (see [Model § Three attention paths](model.md#three-attention-paths)).

### 2 · `maybe_no_sync`

```python
with maybe_no_sync(model, micro_step, grad_accum_steps):
    ...
```

Utility from
[`kempnerforge.training.grad`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/training/grad.py),
re-exported from `kempnerforge.training`. On all microbatches except
the last, it disables gradient synchronization so the backward pass
accumulates locally and does not fire reduce-scatter N times per
optimizer step.

### 3 · MoE step propagation

```python
if mc.is_moe:
    model.set_moe_step(step, tc.max_steps)
```

Forwards `(step, max_steps)` to every `SigmoidTopKRouter`. Used by the
adaptive bias schedule (see [docs/moe/](../moe/index.md)). Dense models
skip this.

### 4 · Forward

```python
logits = model(input_ids, doc_ids=doc_ids)
loss   = loss_fn(logits, labels)
```

The forward pass follows [the model page](model.md): token embedding →
N transformer blocks (RoPE + GQA + SwiGLU or MoE) → final RMSNorm →
output head → `(batch, seq_len, vocab_size)` logits.

Per-dataset metrics, if the dataloader is a mixture, are collected here
before backward so the logits are still alive.

### 5 · MoE auxiliary loss

```python
if mc.is_moe:
    aux_loss = model.get_moe_aux_loss()
    loss = loss + mc.moe_aux_loss_weight * aux_loss
```

`get_moe_aux_loss()` sums the per-layer `MoEMLP.aux_loss` attributes.
For dense runs it returns `0.0` and the line is a no-op.

### 6 · Backward with gradient accumulation

```python
scaled_loss = loss / tc.grad_accum_steps
scaled_loss.backward()
total_loss += loss.item()
```

Scaling by `grad_accum_steps` keeps the effective learning rate
invariant to the accumulation factor.

### 7 · Clip, NaN check, optimizer step

After the microbatch loop:

```python
grad_norm = clip_grad_norm_(model, tc.grad_clip_norm)

if not nan_detector.check_loss(avg_loss, step):
    optimizer.zero_grad(); step += 1; continue   # skip this step

optimizer.step()
scheduler.step()
if phase_lr_scale != 1.0:
    for pg in optimizer.param_groups: pg["lr"] *= phase_lr_scale
optimizer.zero_grad()
```

`clip_grad_norm_` wraps PyTorch's utility so it works with FSDP2 sharded
parameters. The NaN detector returns `False` on NaN/Inf loss, zeroes
the grads, and (after `max_consecutive`) signals a rollback to the
previous checkpoint.

Phase LR scaling applies *after* the scheduler — it multiplies the base
LR that `scheduler.step()` just computed.

### 8 · Step accounting

```python
step += 1
tokens_in_step = tc.batch_size * tc.seq_len * tc.grad_accum_steps * dp_size
tokens_seen   += tokens_in_step
```

`tokens_in_step` times all ranks in the data-parallel dimension, not the
full `world_size` — TP, PP, and EP don't multiply unique tokens.

### 9 · Phase transitions

If a mixture phase is due, `sampler.update_weights(...)` rebalances the
MixtureSampler and `data_iter = None` forces a refresh next microbatch
so the new weights take effect immediately.

### 10 · Metrics and hooks

```python
step_metrics = tracker.end_step(step=step, loss=avg_loss,
                                grad_norm=grad_norm_val, lr=current_lr,
                                tokens_in_step=tokens_in_step)
hook_runner.on_step_end(StepContext(...))
```

`tracker.end_step` dispatches to WandB / TensorBoard at
`metrics.log_interval`. `HookRunner` (from
[`kempnerforge.training.hooks`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/training/hooks.py))
runs user-defined `TrainingHook` callbacks each step.

MoE runs additionally log `moe/aux_loss` and `moe/expert_balance`
(`min/max` of per-expert token counts) at the same cadence.

### 11 · Periodic work

In order:

- **NCCL health** (every `tc.nccl_health_check_interval` steps) — fires
  a small all-reduce; on failure, break the loop.
- **Eval** (every `eval_config.interval` steps) — runs `run_eval` on the
  eval dataloader, logs metrics, fires `on_eval_end` hooks.
- **Profiler** — `prof.step()` advances the `torch.profiler` schedule.
- **Checkpoint** (every `checkpoint.interval` steps) — `ckpt_mgr.save`
  writes a DCP checkpoint asynchronously and updates the `latest`
  symlink.
- **Graceful shutdown** — if SIGTERM/SIGUSR1 fired,
  `ckpt_mgr.save(emergency)` and exit.

## Shutdown

After the loop exits:

- `prof.stop()` flushes traces.
- `ckpt_mgr.wait()` drains the last async save.
- `hook_runner.on_train_end(step, tokens_seen)`.
- `tracker.close()` flushes WandB / TB.
- `destroy_distributed()` tears down the process group.

## Where to read next

- [Training subsystem](../training/index.md) — loss functions,
  optimizers, schedulers in detail.
- [Checkpointing](../checkpointing/index.md) — DCP internals and the
  resume protocol.
- [Resilience](../resilience/index.md) — SIGTERM, NaN, NCCL health
  mechanics.
- [Metrics and profiling](../metrics-and-profiling/index.md) —
  `MetricsTracker`, MFU, profiler.
