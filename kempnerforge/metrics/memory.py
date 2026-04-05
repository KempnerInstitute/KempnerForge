"""GPU memory tracking and reporting.

Provides utilities for monitoring GPU memory usage during training:
  - Current / peak / reserved memory
  - Memory utilization as a percentage of total
  - Human-readable formatting
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def get_memory_stats(device: int = 0) -> dict[str, float]:
    """Get current GPU memory statistics in GB.

    Args:
        device: CUDA device index.

    Returns:
        Dict with allocated, peak, reserved, and total memory in GB.
    """
    if not torch.cuda.is_available():
        return {"allocated_gb": 0, "peak_gb": 0, "reserved_gb": 0, "total_gb": 0}

    gb = 1024**3
    return {
        "allocated_gb": torch.cuda.memory_allocated(device) / gb,
        "peak_gb": torch.cuda.max_memory_allocated(device) / gb,
        "reserved_gb": torch.cuda.memory_reserved(device) / gb,
        "total_gb": torch.cuda.get_device_properties(device).total_memory / gb,
    }


def get_memory_utilization(device: int = 0) -> float:
    """Get peak memory utilization as a fraction of total GPU memory.

    Returns:
        Utilization between 0.0 and 1.0.
    """
    stats = get_memory_stats(device)
    if stats["total_gb"] == 0:
        return 0.0
    return stats["peak_gb"] / stats["total_gb"]


def format_memory_stats(device: int = 0) -> str:
    """Format memory stats as a human-readable string."""
    stats = get_memory_stats(device)
    util = get_memory_utilization(device)
    return (
        f"GPU mem: {stats['allocated_gb']:.1f}GB allocated, "
        f"{stats['peak_gb']:.1f}GB peak, "
        f"{stats['reserved_gb']:.1f}GB reserved / "
        f"{stats['total_gb']:.1f}GB total ({util:.0%})"
    )


def reset_peak_memory(device: int = 0) -> None:
    """Reset peak memory tracking counter."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def log_memory_stats(device: int = 0) -> dict[str, float]:
    """Log and return current memory statistics."""
    stats = get_memory_stats(device)
    logger.info(format_memory_stats(device))
    return stats


# ---------------------------------------------------------------------------
# DeviceMemoryMonitor — per-step tracking with snapshot support
# ---------------------------------------------------------------------------


class DeviceMemoryMonitor:
    """Tracks GPU memory usage across training steps.

    Resets peak memory stats at each reporting interval so that the
    peak reflects per-interval usage rather than all-time peak.

    Supports memory snapshot capture at a configurable step for
    debugging OOM and memory fragmentation with pytorch.org/memory_viz.

    Args:
        device: CUDA device index.
        snapshot_step: Step at which to capture a memory snapshot. None to disable.
        snapshot_dir: Directory to save snapshots.
    """

    def __init__(
        self,
        device: int = 0,
        snapshot_step: int | None = None,
        snapshot_dir: str = "memory_snapshots",
    ) -> None:
        self.device = device
        self.snapshot_step = snapshot_step
        self.snapshot_dir = snapshot_dir
        self._snapshot_taken = False

    def report(self, step: int) -> dict[str, float]:
        """Report memory stats for the current interval and reset peak.

        Args:
            step: Current training step.

        Returns:
            Dict with memory stats.
        """
        stats = get_memory_stats(self.device)
        util = get_memory_utilization(self.device)
        stats["mem_utilization"] = util

        logger.info(f"[step {step}] {format_memory_stats(self.device)}")

        # Check if we should capture a snapshot
        if (
            self.snapshot_step is not None
            and step == self.snapshot_step
            and not self._snapshot_taken
        ):
            self.capture_snapshot(step)

        # Reset peak for next interval
        reset_peak_memory(self.device)

        return stats

    def capture_snapshot(self, step: int) -> str | None:
        """Capture a CUDA memory snapshot and save as pickle.

        The snapshot can be visualized at https://pytorch.org/memory_viz

        Args:
            step: Current step (used in filename).

        Returns:
            Path to the saved snapshot, or None if capture failed.
        """
        if not torch.cuda.is_available():
            return None

        try:
            from pathlib import Path

            snapshot_dir = Path(self.snapshot_dir)
            snapshot_dir.mkdir(parents=True, exist_ok=True)

            # Record and export memory snapshot
            torch.cuda.memory._record_memory_history()
            # Give it a moment to record current state
            torch.cuda.synchronize(self.device)
            snapshot = torch.cuda.memory._snapshot()
            torch.cuda.memory._record_memory_history(enabled=None)

            import pickle

            path = snapshot_dir / f"snapshot_step_{step}_device_{self.device}.pickle"
            with open(path, "wb") as f:
                pickle.dump(snapshot, f)

            self._snapshot_taken = True
            logger.info(f"Memory snapshot saved: {path}")
            return str(path)

        except Exception as e:
            logger.warning(f"Memory snapshot failed: {e}")
            return None
