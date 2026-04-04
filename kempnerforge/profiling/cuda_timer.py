"""CUDA event-based timing utilities.

Provides lightweight, GPU-accurate timers for profiling specific regions
of the training loop (forward pass, backward pass, communication, etc.).
Uses CUDA events to avoid CPU synchronization overhead during measurement.

Usage:
    timer = CUDATimer()
    timer.start()
    # ... GPU work ...
    timer.stop()
    elapsed_ms = timer.elapsed_ms()

    # Or use the multi-region tracker:
    timers = CUDATimerCollection(regions=["forward", "backward", "comm"])
    timers.start("forward")
    # ... forward pass ...
    timers.stop("forward")
    report = timers.elapsed_all()  # {"forward": 12.3, "backward": 0.0, ...}
"""

from __future__ import annotations

import torch


class CUDATimer:
    """CUDA event-based timer for accurate GPU timing.

    Uses CUDA events to measure elapsed time without CPU synchronization
    overhead (synchronizes only when reading the result).
    """

    def __init__(self) -> None:
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        self._running = False
        self._recorded = False

    def start(self) -> None:
        """Record the start event on the current CUDA stream."""
        self._start.record()
        self._running = True

    def stop(self) -> None:
        """Record the end event on the current CUDA stream."""
        self._end.record()
        self._running = False
        self._recorded = True

    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds (synchronizes CUDA)."""
        self._end.synchronize()
        return self._start.elapsed_time(self._end)


class CUDATimerCollection:
    """Collection of named CUDA timers for profiling multiple regions.

    Manages timers for distinct training phases (forward, backward, comm, etc.)
    and reports all elapsed times as a dictionary.

    When ``enabled=False``, all operations are no-ops with zero overhead —
    start/stop calls return immediately without recording CUDA events.

    Args:
        regions: List of region names to track.
        enabled: Whether timing is active. When False, all calls are no-ops.
    """

    def __init__(self, regions: list[str], enabled: bool = True) -> None:
        self._enabled = enabled
        self._regions = regions
        self._timers: dict[str, CUDATimer] = {}
        if enabled:
            for name in regions:
                self._timers[name] = CUDATimer()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start(self, region: str) -> None:
        """Start timing a named region."""
        if not self._enabled:
            return
        self._timers[region].start()

    def stop(self, region: str) -> None:
        """Stop timing a named region."""
        if not self._enabled:
            return
        self._timers[region].stop()

    def elapsed_ms(self, region: str) -> float:
        """Get elapsed time for a specific region in milliseconds."""
        if not self._enabled:
            return 0.0
        return self._timers[region].elapsed_ms()

    def elapsed_all(self) -> dict[str, float]:
        """Get elapsed times for all regions in milliseconds.

        Returns a dict mapping region name → elapsed_ms.
        Regions that were never started/stopped return 0.0.
        """
        if not self._enabled:
            return {name: 0.0 for name in self._regions}
        result = {}
        for name, timer in self._timers.items():
            if timer._recorded:
                result[name] = timer.elapsed_ms()
            else:
                result[name] = 0.0  # Never completed or still running
        return result
