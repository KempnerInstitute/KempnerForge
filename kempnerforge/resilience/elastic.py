"""Elastic training and SLURM integration helpers.

Provides utilities for training jobs that may be preempted, requeued,
or restarted with a different number of nodes:
  - SLURM job info detection
  - Requeue detection
  - Auto-resume path resolution
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SLURMInfo:
    """Information about the current SLURM job."""

    job_id: str
    job_name: str
    node_list: str
    num_nodes: int
    ntasks_per_node: int
    restart_count: int
    partition: str
    array_task_id: str | None  # None if not an array job

    @property
    def is_requeued(self) -> bool:
        """Whether this job has been requeued (restart_count > 0)."""
        return self.restart_count > 0


def get_slurm_info() -> SLURMInfo | None:
    """Read SLURM job information from environment variables.

    Returns:
        SLURMInfo if running under SLURM, None otherwise.
    """
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is None:
        return None

    return SLURMInfo(
        job_id=job_id,
        job_name=os.environ.get("SLURM_JOB_NAME", ""),
        node_list=os.environ.get("SLURM_JOB_NODELIST", ""),
        num_nodes=int(os.environ.get("SLURM_NNODES", "1")),
        ntasks_per_node=int(os.environ.get("SLURM_NTASKS_PER_NODE", "1")),
        restart_count=int(os.environ.get("SLURM_RESTART_COUNT", "0")),
        partition=os.environ.get("SLURM_JOB_PARTITION", ""),
        array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
    )


def is_slurm_job() -> bool:
    """Check if we are running under SLURM."""
    return "SLURM_JOB_ID" in os.environ


def is_slurm_requeue() -> bool:
    """Check if this is a requeued SLURM job.

    Uses ``SLURM_RESTART_COUNT`` (set by SLURM on requeue).
    """
    return int(os.environ.get("SLURM_RESTART_COUNT", "0")) > 0


def resolve_resume_path(checkpoint_dir: str) -> Path | None:
    """Find the latest checkpoint for auto-resume.

    Checks:
      1. ``{checkpoint_dir}/latest`` symlink
      2. Most recent ``step_N`` directory by step number

    Args:
        checkpoint_dir: Base checkpoint directory.

    Returns:
        Path to the latest checkpoint, or None if none found.
    """
    base = Path(checkpoint_dir)
    if not base.exists():
        return None

    # Check "latest" symlink first
    latest = base / "latest"
    if latest.exists():
        resolved = latest.resolve()
        if resolved.exists():
            logger.info(f"Auto-resume: found latest checkpoint at {resolved}")
            return resolved

    # Fall back to most recent step_N directory
    step_dirs = sorted(
        (
            d
            for d in base.iterdir()
            if d.is_dir() and d.name.startswith("step_") and d.name.split("_")[1].isdigit()
        ),
        key=lambda d: int(d.name.split("_")[1]),
    )

    if step_dirs:
        path = step_dirs[-1]
        logger.info(f"Auto-resume: found checkpoint at {path}")
        return path

    return None


def log_job_info() -> None:
    """Log SLURM job information (if running under SLURM)."""
    info = get_slurm_info()
    if info is None:
        logger.info("Not running under SLURM")
        return

    logger.info(
        f"SLURM job: id={info.job_id}, name={info.job_name}, "
        f"nodes={info.num_nodes}, tasks/node={info.ntasks_per_node}, "
        f"partition={info.partition}, restart_count={info.restart_count}"
    )

    if info.is_requeued:
        logger.info(f"Job was requeued (restart #{info.restart_count}) — will auto-resume")
