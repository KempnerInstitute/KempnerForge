# TensorBoard backend

[`TensorBoardBackend`](https://github.com/KempnerInstitute/KempnerForge/blob/main/kempnerforge/metrics/tracker.py)
is the sibling to `WandBBackend` — same metric dict, local event files
instead of a cloud run.

```toml
[metrics]
enable_tensorboard = true
tensorboard_dir    = "tb_logs"    # default; relative to cwd
```

Both backends can be enabled simultaneously; they don't interact.

## Init is lazy

```python
# kempnerforge/metrics/tracker.py — TensorBoardBackend
def _ensure_init(self) -> None:
    if self._writer is not None:
        return
    try:
        from torch.utils.tensorboard import SummaryWriter
        self._writer = SummaryWriter(log_dir=self._config.tensorboard_dir)
    except ImportError:
        logger.warning("tensorboard not installed — disabling TensorBoard backend")
        self._writer = False
```

`SummaryWriter(log_dir=...)` creates the directory on first use and opens
an event file. Import happens on first log call, not at tracker
construction — same lazy pattern as WandB.

## What gets logged

Every key in the backend dict becomes a scalar:

```python
def log(self, metrics: dict[str, float], step: int) -> None:
    for key, val in metrics.items():
        self._writer.add_scalar(key, val, global_step=step)
```

The namespaces `train/...`, `gpu/...`, `smoothed/...` become tabs in the
TensorBoard UI.

## Rank 0 only

`_init_backends` gates construction on `dist.get_rank() == 0`, so only
rank 0's training loop writes events. This avoids event-file corruption
when multiple ranks write to the same log directory.

If you want per-rank event files (e.g. for NCCL diagnostics), instantiate
a `SummaryWriter` directly and skip the tracker — the backend plumbing
is single-writer on purpose.

## Output layout

```
tb_logs/
├── events.out.tfevents.1714838401.node01.12345.0
└── ...
```

View with:

```bash
uv run tensorboard --logdir tb_logs
```

## Co-location with profiler traces

`[profiling].trace_dir` defaults to `profiler_traces/`. If you set both
`tensorboard_dir` and `trace_dir` to the same path, TensorBoard will show
both the scalar metrics and the profiler's PyTorch Profiler tab:

```toml
[metrics]
enable_tensorboard = true
tensorboard_dir    = "runs/7b"

[profiling]
enable    = true
trace_dir = "runs/7b"     # same directory
```

But keeping them separate is fine — `tensorboard --logdir runs/` picks
up both.

## `close()`

```python
def close(self) -> None:
    if self._writer and self._writer is not False:
        self._writer.close()
```

Flushes the event file. Called from `tracker.close()` at training exit.

## See also

- [Metrics tracker](metrics-tracker.md) — what metrics end up in the
  event file.
- [Profiler](profiler.md) — traces also land in a TensorBoard-readable
  format via `tensorboard_trace_handler`.
- [Configuration § `[metrics]`](../configuration/config-sections.md) —
  `enable_tensorboard` and `tensorboard_dir`.
