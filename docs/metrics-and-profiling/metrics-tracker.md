# Metrics tracker

[`MetricsTracker`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/metrics/tracker.py)
is the central per-step metric collector. One instance is created once in
`scripts/train.py`; the training loop bookends every step with
`start_step()` / `end_step()` and the tracker handles timing, smoothing,
stdout formatting, and backend dispatch.

## Wiring

```python
# scripts/train.py
tracker = MetricsTracker(config, num_gpus=world_size)
tracker.init_backends(config)     # lazy rank-0 init
...
while step < tc.max_steps:
    tracker.start_step()
    # ... forward, backward, optimizer step ...
    tracker.end_step(step, loss=loss, grad_norm=gn, lr=lr,
                     tokens_in_step=tokens_per_step)
    ...
tracker.close()
```

`init_backends` is separate from `__init__` because distributed init and
config rewrites (e.g. restored `wandb_run_id`) happen between tracker
construction and the first step.

## Per-step flow

`end_step` runs five things in order:

1. **Timing** — `step_time = perf_counter() - _step_start`; derives
   `tokens_per_sec = tokens_in_step / step_time`.
2. **MFU** — calls `compute_mfu(model_config, tokens_per_sec, num_gpus,
   gpu_peak_tflops, seq_len)`. See [MFU](mfu.md).
3. **Memory stats** — `get_memory_stats()` + `get_memory_utilization()`,
   captured every step from device 0.
4. **EMA smoothing** — updates four smoothed keys (`loss`,
   `tokens_per_sec`, `mfu`, `step_time`) with `alpha=0.1`.
5. **Log if due** — if `step % log_interval == 0` or `step == 1`, dispatch
   to stdout + all backends and return a `StepMetrics`. Otherwise return
   `None`.

## `StepMetrics`

```python
@dataclass
class StepMetrics:
    loss: float           # raw, not smoothed
    grad_norm: float
    lr: float
    tokens_per_sec: float # global (across all GPUs)
    mfu: float
    step_time_sec: float
    allocated_gb: float
    peak_gb: float
    reserved_gb: float
    total_gb: float
    mem_utilization: float
```

Returned from `end_step()` only on logging steps. The raw values are
written to backends; the EMA-smoothed copies go under a `smoothed/` prefix
(see below).

## EMA smoothing

```python
# kempnerforge/metrics/tracker.py — _update_smoothed
if key not in self._smoothed:
    self._smoothed[key] = value      # bootstrap with first sample
else:
    self._smoothed[key] = alpha * value + (1 - alpha) * self._smoothed[key]
```

`alpha = 0.1` is hardcoded. Four metrics get smoothed: `loss`,
`tokens_per_sec`, `mfu`, `step_time`. Smoothed values are reported to
backends as `smoothed/loss`, `smoothed/tokens_per_sec`, etc. Stdout
reports the raw values.

Smoothing runs every step regardless of `log_interval` — the EMA is built
from every data point, even ones that aren't logged.

## Log interval

```python
if step % self.metrics_config.log_interval == 0 or step == 1:
    self._log_step(step, metrics)
```

Default `log_interval = 10`. Step 1 is always logged (useful sanity check
that the pipeline is alive). Step 0 is never reached — the training loop
starts at `step = 1`.

## Stdout format

```
[step 1000] loss=2.3400 | lr=3.00e-04 | grad_norm=1.250 | tok/s=125,000 | mfu=52.3% | mem=71.2/80GB | step_time=1.25s
```

Produced by `format_metrics(step, {...})` from
[`kempnerforge/metrics/logger.py`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/metrics/logger.py).
Numbers use `_format_number`: int gets k/M/B suffix, float gets scientific
notation below 0.01 or above 1e6, otherwise 4 decimals.

The same logger writes every other `logger.info` in the framework — with
a `[rank N] LEVEL   ` prefix and ANSI color if `stdout.isatty()` and
`NO_COLOR` isn't set (see [`_RankFormatter`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/metrics/logger.py)).

## Backend dispatch

`end_step` emits a numeric dict to every configured backend:

```python
backend_dict = {
    "train/loss": ..., "train/grad_norm": ..., "train/lr": ...,
    "train/tokens_per_sec": ..., "train/mfu": ..., "train/step_time_sec": ...,
    "gpu/allocated_gb": ..., "gpu/peak_gb": ..., "gpu/reserved_gb": ...,
    "gpu/mem_utilization": ...,
    # plus smoothed/*
}
for backend in self._backends:
    backend.log(backend_dict, step=step)
```

Backends are initialized lazily and only on rank 0:

```python
# kempnerforge/metrics/tracker.py — _init_backends
if dist.is_initialized() and dist.get_rank() != 0:
    return
if mc.enable_wandb:
    self._backends.append(WandBBackend(mc))
if mc.enable_tensorboard:
    self._backends.append(TensorBoardBackend(mc))
```

Non-rank-0 workers keep `_backends` empty; their `_log_step` call iterates
over an empty list and only the stdout line is emitted (and even that is
filtered by the `_RankFilter` — see
[Logger](#logger-get_logger-format_metrics)).

## Eval path

```python
def log_eval(self, metrics: dict[str, float], step: int) -> None:
    logger.info(format_metrics(step, metrics))
    for backend in self._backends:
        backend.log(metrics, step=step)
```

Eval metrics don't touch the smoothing table — they're reported raw. The
training loop calls `log_eval` after `run_eval`, and separately for MoE
and per-dataset breakdowns:

```python
# scripts/train.py
tracker.log_eval(moe_metrics, step)     # moe/aux_loss, moe/expert_balance
tracker.log_eval(ds_metrics, step)      # per-source loss
tracker.log_eval(eval_metrics, step)    # eval loss
```

Each call logs with the same `step`, so a single training step can
produce several side-by-side rows in WandB / TensorBoard.

## Close

```python
def close(self) -> None:
    for backend in self._backends:
        backend.close()
```

Called at the end of `scripts/train.py`. WandB closes the run; TensorBoard
flushes the event file.

## Logger: `get_logger`, `format_metrics`

Anywhere in the framework, call:

```python
from kempnerforge.metrics import get_logger
logger = get_logger(__name__)
logger.info("starting distributed init")
```

Returns a `logging.Logger` named `kempnerforge.<name>`. On first call,
configures the root `kempnerforge` logger with a `_RankFormatter` (adds
`[rank N]` + colored level) and a `_RankFilter` (drops records from
non-zero ranks by default).

Override rank filtering with `get_logger(__name__, rank_zero_only=False)`
when you genuinely want every rank to log — e.g. for distributed-init
debugging. Once the root logger is configured it's cached, so this flag
only takes effect on the very first call.

## See also

- [WandB](wandb.md) — how `WandBBackend` initializes and what it logs.
- [TensorBoard](tensorboard.md) — `TensorBoardBackend` specifics and
  where event files land.
- [MFU](mfu.md) — what the `mfu` field is computed from.
- [Memory monitor](memory-monitor.md) — the module behind `allocated_gb`,
  `peak_gb`, `reserved_gb`, `total_gb`.
