"""Shutdown-signal handling in the SLURM launch scripts under ``scripts/slurm``."""

from __future__ import annotations

from pathlib import Path

import pytest

SLURM = Path(__file__).resolve().parents[2] / "scripts" / "slurm"
SRUN_SCRIPTS = ("7b_requeue.sh", "multinode.sh", "13b_validation.sh")


@pytest.mark.parametrize("name", SRUN_SCRIPTS)
def test_srun_scripts_signal_the_job_steps(name):
    """``--signal=B:`` reaches only the batch shell, never the srun-launched ranks."""
    text = (SLURM / name).read_text()
    assert "#SBATCH --signal=SIGTERM@120" in text
    assert "--signal=B:" not in text
