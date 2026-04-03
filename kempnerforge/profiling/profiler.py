"""torch.profiler integration for KempnerForge.

Provides a step-aware profiler wrapper that activates only within a
configured step range, exports Chrome traces, and integrates with
the training loop via a simple .step() interface.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

from kempnerforge.config.schema import ProfilingConfig

logger = logging.getLogger(__name__)


def build_profiler(
    config: ProfilingConfig,
    rank: int = 0,
) -> torch.profiler.profile | None:
    """Build a torch.profiler instance from config.

    Returns None if profiling is disabled.

    Args:
        config: Profiling configuration.
        rank: Current rank (for output directory naming).

    Returns:
        A torch.profiler.profile context manager, or None.
    """
    if not config.enable:
        return None

    trace_dir = Path(config.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)

    # Profile schedule: wait → warmup → active → repeat
    # wait: skip steps before start_step
    # warmup: 1 step to stabilize profiler
    # active: profile for (end_step - start_step) steps
    wait_steps = max(0, config.start_step - 1)
    active_steps = config.end_step - config.start_step

    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(
            wait=wait_steps,
            warmup=1,
            active=active_steps,
            repeat=1,
        ),
        on_trace_ready=tensorboard_trace_handler(str(trace_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    )

    logger.info(
        f"Profiler configured: steps {config.start_step}–{config.end_step}, "
        f"traces → {trace_dir}"
    )

    return prof


class CUDATimer:
    """CUDA event-based timer for accurate GPU timing.

    Uses CUDA events to measure elapsed time without CPU synchronization
    overhead (synchronizes only when reading the result).

    Usage:
        timer = CUDATimer()
        timer.start()
        # ... GPU work ...
        timer.stop()
        elapsed_ms = timer.elapsed_ms()
    """

    def __init__(self) -> None:
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)

    def start(self) -> None:
        """Record the start event on the current CUDA stream."""
        self._start.record()

    def stop(self) -> None:
        """Record the end event on the current CUDA stream."""
        self._end.record()

    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds (synchronizes CUDA)."""
        self._end.synchronize()
        return self._start.elapsed_time(self._end)
