# Schedulers

Six schedulers are registered in
[`kempnerforge/training/scheduler.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/training/scheduler.py):
`cosine`, `linear`, `wsd`, `constant`, `rex`, `none`. All return a
`torch.optim.lr_scheduler.LambdaLR` — a single callable that multiplies
the optimizer's base LR by a schedule-dependent factor at each
`scheduler.step()`.

## Shared warmup

Every schedule (except `"none"`) warms up linearly from `0` to `1.0`
over `warmup_steps`. The helper is

```python
def _warmup_factor(step, warmup_steps):
    return min(1.0, step / warmup_steps)
```

So all the fields below describe the **decay** phase, after warmup
completes.

## `cosine`

```toml
[scheduler]
name = "cosine"
warmup_steps = 2000
# decay_steps omitted -> max_steps - warmup_steps
min_lr_ratio = 0.1
```

Classic cosine decay from `1.0` to `min_lr_ratio` over `decay_steps`:

```
factor = min_ratio + 0.5 * (1 - min_ratio) * (1 + cos(π · progress))
```

where `progress = (step - warmup_steps) / decay_steps`, clamped to `[0,
1]`. After `decay_steps`, LR stays flat at `base_lr * min_lr_ratio`.

## `linear`

```toml
[scheduler]
name = "linear"
warmup_steps = 2000
# decay_steps omitted -> max_steps - warmup_steps
min_lr_ratio = 0.0
```

Linear decay from `1.0` to `min_lr_ratio`:

```
factor = 1 - (1 - min_ratio) · progress
```

Straight line in LR space. Most commonly used with `min_lr_ratio =
0.0` for strict linear-to-zero cooldowns.

## `wsd`

```toml
[scheduler]
name = "wsd"
warmup_steps = 2000
stable_steps = 80000
decay_steps = 18000
min_lr_ratio = 0.0
wsd_decay_type = "cosine"               # "cosine", "linear", or "sqrt"
```

Warmup-Stable-Decay: three segments.

| Phase | Steps | LR factor |
|-------|-------|-----------|
| warmup | `[0, warmup_steps)` | linear `0 → 1` |
| stable | `[warmup_steps, warmup_steps + stable_steps)` | `1.0` |
| decay | `[warmup + stable, warmup + stable + decay_steps)` | selected shape down to `min_lr_ratio` |

`wsd_decay_type` picks the cooldown shape:

- `"cosine"` — `min_ratio + 0.5 · (1 - min_ratio) · (1 + cos(π · p))`
- `"linear"` — `1 - (1 - min_ratio) · p`
- `"sqrt"` — `min_ratio + (1 - min_ratio) · sqrt(1 - p)`

WSD pairs well with curriculum-style data schedules — the flat stable
phase is a natural time to anneal data mixtures, then the decay phase
hardens the model on the final mixture.

## `constant`

```toml
[scheduler]
name = "constant"
warmup_steps = 2000
```

Warmup to `1.0`, then hold. No decay. Useful for short experiments and
for debugging the interaction of loss / optimizer with the rest of the
stack without a moving LR.

## `rex`

```toml
[scheduler]
name = "rex"
warmup_steps = 2000
# decay_steps omitted -> max_steps - warmup_steps
min_lr_ratio = 0.1
rex_alpha = 1.0
```

Polynomial decay (REX):

```
factor = max(min_ratio, (1 - progress) ** alpha)
```

`rex_alpha = 1.0` is a linear decay; `< 1` is concave (fast early,
slow late); `> 1` is convex (slow early, fast late). Reasonable values
are in `[0.5, 2.0]`.

## `none`

```toml
[scheduler]
name = "none"
```

Returns a constant `1.0`. No warmup, no decay. Pair with
`schedule_free_adamw`, which manages its own warmup internally —
adding any external schedule on top will interfere with the internal
Polyak averaging.

## Choosing a scheduler

| Situation | Pick |
|-----------|------|
| Default dense pretraining | `cosine` to `min_lr_ratio=0.1` |
| Need to continue training beyond `decay_steps` | `constant` or `wsd` (keeps LR schedulable) |
| Anneal data mix then cool down | `wsd` with `wsd_decay_type="linear"` |
| Schedule-free optimizer | `none` |
| Linear cooldown to zero | `linear` with `min_lr_ratio=0.0` |

## How phase LR scaling layers on top

The training loop applies `phase_lr_scale` *after* `scheduler.step()`
each step (see
[Training loop § Optimizer step](training-loop.md#optimizer-and-scheduler-step)).
The scheduler computes the base LR; the phase scales it. So a cosine
schedule with `phase.lr_scale = 0.5` in a curriculum phase halves the
cosine LR for that phase without touching the scheduler's own state.

## See also

- [Optimizers](optimizers.md) — in particular `schedule_free_adamw`,
  which requires `scheduler.name = "none"`.
- [Configuration § SchedulerConfig](../configuration/config-sections.md) —
  the dataclass with defaults.
- [Data § Phase schedule](../data/index.md) — curriculum phases that
  can overlay `lr_scale` on whatever the scheduler produces.
