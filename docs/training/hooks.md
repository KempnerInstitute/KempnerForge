# Hooks

The `TrainingHook` interface in
[`kempnerforge/training/hooks.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/training/hooks.py)
is the supported way to run custom logic during training without
forking [`scripts/train.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/scripts/train.py).

## Interface

```python
from kempnerforge.training.hooks import TrainingHook, StepContext, HookRunner

class MyHook(TrainingHook):
    def on_train_begin(self, config): ...
    def on_step_end(self, ctx: StepContext): ...
    def on_eval_end(self, metrics: dict[str, float], step: int): ...
    def on_checkpoint_save(self, step: int, path: str): ...
    def on_train_end(self, step: int, tokens_seen: int): ...
```

`TrainingHook` is a concrete base class with empty method bodies —
subclass and override only what you need. The `HookRunner` dispatches
calls to a list of hooks; when the list is empty, each entry point
short-circuits on an `if not self.hooks: return` check, so an unused
runner costs one branch per call site.

## `StepContext`

Passed to `on_step_end`, every step:

```python
@dataclass
class StepContext:
    step: int
    loss: float
    grad_norm: float
    lr: float
    tokens_seen: int
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
```

`model` and `optimizer` are the live references — you can read
parameter values and optimizer state. **Gradients have already been
zeroed by the time `on_step_end` fires.** The loop order is:

```python
optimizer.step()
scheduler.step()
optimizer.zero_grad()                   # grads cleared here
...
tracker.end_step(...)
hook_runner.on_step_end(StepContext(...))
```

If your hook needs to inspect gradients, it can't — you need to
modify `train.py` and capture them before `optimizer.zero_grad()`. The
`wandb.log` of `p.grad.norm()` in the module's docstring example
requires capturing the norms during the backward pass (via a
pre-`zero_grad` hook), not reading `p.grad` in `on_step_end`.

## Lifecycle events

| Event | Fires at | What you get |
|-------|----------|--------------|
| `on_train_begin(config)` | Once, after setup, before the loop | Full `JobConfig` |
| `on_step_end(ctx)` | Every training step, after metrics | `StepContext` |
| `on_eval_end(metrics, step)` | After every eval round | `{"eval/loss": ..., "eval/perplexity": ...}` |
| `on_checkpoint_save(step, path)` | After a DCP checkpoint is written | `path` is `config.checkpoint.dir` |
| `on_train_end(step, tokens_seen)` | Once, after the loop exits (normal or shutdown) | Final counters |

Ordering at step boundaries:

```
… step body …
tracker.end_step(...)                   # metrics dispatched to WandB/TB
hook_runner.on_step_end(StepContext(...))
# MoE-specific metrics logged
# NCCL health / eval / profiler / checkpoint / shutdown
```

`on_step_end` fires **after** the metrics tracker — logging from a
hook into WandB happens on the same step number, not a step off. At
the end of a checkpoint-save step, the order is `on_step_end` first,
then `on_checkpoint_save`.

## Registering hooks

Hooks are created in-process, not from TOML:

```python
# In a custom launcher that imports scripts/train.py as a library,
# or in a fork that monkey-patches the hook_runner after construction:

from kempnerforge.training.hooks import HookRunner
hook_runner = HookRunner([MyHook1(), MyHook2()])
```

`scripts/train.py` instantiates `HookRunner()` with no hooks at line
502. To register hooks today, you either fork `train.py` to pass a
populated list, or import-and-patch from a custom entry point.
A config-driven hook registry is not yet wired up.

## When to write a hook vs fork `train.py`

| Task | Hook or fork |
|------|--------------|
| Custom WandB logging (histograms, attention maps from eval) | **Hook** (`on_step_end`, `on_eval_end`) |
| External heartbeat / health ping | **Hook** (`on_step_end`) |
| Upload checkpoint to S3 after save | **Hook** (`on_checkpoint_save`) |
| Inspect gradients | **Fork** (grads are zeroed before `on_step_end`) |
| Inject a custom optimizer step | **Fork** (no `on_pre_optimizer_step` event) |
| Change the loss function mid-run | **Fork** (`loss_fn` is closed over once at setup) |
| Custom learning-rate schedule not in the registry | Register a new scheduler, not a hook |

The guiding principle: hooks observe, they don't intervene. Anything
that needs to modify the step (not just log about it) belongs in a
fork or a new registry entry.

## Example: per-step gradient-norm histogram

```python
class GradHistogramHook(TrainingHook):
    """Stash pre-clip grad norms during backward, log them in on_step_end.

    This requires adding a forward hook during on_train_begin because
    gradients are zeroed by the time on_step_end fires.
    """
    def __init__(self) -> None:
        self._norms: dict[str, float] = {}

    def on_train_begin(self, config) -> None:
        # Subclass or patch training loop to populate _norms during backward
        pass

    def on_step_end(self, ctx: StepContext) -> None:
        if self._norms:
            wandb.log(
                {f"grad_norm/{n}": v for n, v in self._norms.items()},
                step=ctx.step,
            )
            self._norms.clear()
```

The pragmatic version: for a one-off gradient-norm histogram, just
edit `scripts/train.py` to log it directly after the microbatch loop —
the hook machinery adds complexity when you only ever need one
logger.

## See also

- [Training loop § Metrics and hooks](training-loop.md#metrics-and-hooks)
  — exact ordering of `tracker.end_step` vs `hook_runner.on_step_end`.
- [Metrics and profiling](../metrics-and-profiling/index.md) —
  `MetricsTracker`, the out-of-the-box logger your hook complements.
- [Evaluation § In-loop](evaluation.md#in-loop-evaluation) — where
  `on_eval_end` fires.
- [Checkpointing](../checkpointing/index.md) — where
  `on_checkpoint_save` fires.
