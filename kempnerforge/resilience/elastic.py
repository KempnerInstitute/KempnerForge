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

_DCP_METADATA_FILE = ".metadata"


def _dcp_durable(ckpt_dir: Path, num_stages: int | None = None) -> bool:
    """Whether ``ckpt_dir`` holds a complete set of DCP shards.

    Accepts both layouts: a flat directory, and the per-stage ``pp{k}/``
    subdirectories written under pipeline parallelism.

    Under pipeline parallelism *every* stage must be durable. Each stage saves
    through its own process group, so an interrupted save can leave the earlier
    stages complete and the later ones empty; resuming into that mix fails in
    ``dcp.load`` on the stages that have no ``.metadata``, or resumes them from
    different steps. An empty stage list is not durable either -- ``all()`` is
    vacuously true on it, and a directory with no stage at all holds nothing.

    A stage that died before its save started leaves no ``pp{k}/`` directory at
    all, so only ``num_stages``, the run's pipeline degree, can tell that it is
    missing: with it, ``pp0`` through ``pp{num_stages - 1}`` must also be there.
    Without it, only the stage directories that exist are checked. The stage
    layout records no degree of its own, so a checkpoint saved with fewer stages
    than ``num_stages`` looks the same as one whose later stages never saved.
    """
    if (ckpt_dir / _DCP_METADATA_FILE).exists():
        return True
    if not ckpt_dir.is_dir():
        return False
    stages = [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("pp")]
    if not stages or not all((d / _DCP_METADATA_FILE).exists() for d in stages):
        return False
    if num_stages is not None and num_stages > 1:
        return all((ckpt_dir / f"pp{k}" / _DCP_METADATA_FILE).exists() for k in range(num_stages))
    return True


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
    array_task_id: str | None

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


def resolve_resume_path(checkpoint_dir: str, num_stages: int | None = None) -> Path | None:
    """Find the latest checkpoint for auto-resume.

    Checks:
      1. ``{checkpoint_dir}/latest`` symlink
      2. Most recent ``step_N`` directory *whose DCP shards are durable*

    ``CheckpointManager`` only ever points ``latest`` at a durable checkpoint,
    so (1) needs no further test. The ``step_N`` fallback has no such guarantee:
    an interrupted save leaves a directory that exists but holds no ``.metadata``,
    and resuming into it fails in ``dcp.load`` with "metadata is None". Skipping
    incomplete directories turns an unrecoverable run into one that resumes from
    the last durable checkpoint, whatever left the partial directory behind.

    Args:
        checkpoint_dir: Base checkpoint directory.
        num_stages: The run's pipeline degree. With it, a ``step_N`` in the
            per-stage layout counts only when ``pp0`` through ``pp{num_stages - 1}``
            all hold ``.metadata``, including stages that never created their
            directory; see ``_dcp_durable``. When no step has every stage, the
            newest one whose stage directories are complete is returned as
            before, after an error log, rather than the run starting over: a
            checkpoint saved with another pipeline degree looks the same, and
            resume does not support changing the degree.

    Returns:
        Path to the latest usable checkpoint, or None if none found.
    """
    base = Path(checkpoint_dir)
    if not base.exists():
        return None

    latest = base / "latest"
    if latest.exists():
        resolved = latest.resolve()
        if resolved.exists():
            logger.info(f"Auto-resume: found latest checkpoint at {resolved}")
            return resolved

    step_dirs = sorted(
        (
            d
            for d in base.iterdir()
            if d.is_dir() and d.name.startswith("step_") and d.name.split("_")[1].isdigit()
        ),
        key=lambda d: int(d.name.split("_")[1]),
        reverse=True,
    )

    short_of_stages: Path | None = None
    for path in step_dirs:
        if _dcp_durable(path, num_stages):
            logger.info(f"Auto-resume: found checkpoint at {path}")
            return path
        if _dcp_durable(path):
            short_of_stages = short_of_stages or path
            logger.warning(
                f"Auto-resume: skipping {path}: its stage directories are complete, but "
                f"this run has {num_stages} pipeline stages and not all of them are there"
            )
            continue
        logger.warning(
            f"Auto-resume: skipping {path} — no DCP .metadata, so the save that "
            f"produced it did not complete"
        )

    if short_of_stages is not None:
        logger.error(
            f"Auto-resume: no checkpoint under {base} has all {num_stages} pipeline stages; "
            f"resuming {short_of_stages}, the newest whose stage directories are complete. "
            f"Either a stage stopped before saving, or the checkpoint was saved with another "
            f"pipeline degree, which resume does not support"
        )
        return short_of_stages

    if step_dirs:
        logger.error(
            f"Auto-resume: {len(step_dirs)} checkpoint directories under "
            f"{base} and none has durable DCP state; starting from step 0. "
            f"Every saved step would have to be re-trained."
        )
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
