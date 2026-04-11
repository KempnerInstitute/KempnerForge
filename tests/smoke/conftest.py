"""Smoke test configuration and fixtures.

Two launch modes:
  Default (torchrun): Works on any machine. Auto-detects GPUs.
      uv run pytest tests/smoke/ --smoke -v

  SLURM (srun): Uses an existing SLURM allocation for multi-node.
      uv run pytest tests/smoke/ --smoke --slurm --jobid=<ID> -v
      uv run pytest tests/smoke/ --smoke --slurm -v  # auto-detect from SLURM_JOB_ID
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest


# ---------------------------------------------------------------------------
# CLI options
# ---------------------------------------------------------------------------
def pytest_addoption(parser):
    group = parser.getgroup("smoke", "Smoke test options")
    # --smoke is registered in tests/conftest.py (root) to work from any directory
    group.addoption("--slurm", action="store_true", default=False, help="Use srun launch mode")
    group.addoption(
        "--jobid", type=str, default=None,
        help="SLURM job ID (auto-detected if omitted)",
    )
    group.addoption(
        "--tokenizer", type=str, default="gpt2",
        help="HF tokenizer for eval tests (default: gpt2, no auth)",
    )
    group.addoption(
        "--vocab-size", type=int, default=50257,
        help="Vocab size matching the tokenizer (default: 50257 for gpt2)",
    )


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------
def _detect_gpu_count() -> int:
    """Detect local GPU count without importing torch."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
            capture_output=True, text=True, timeout=30,
        )
        return int(result.stdout.strip()) if result.returncode == 0 else 0
    except Exception:
        return 0


def _detect_slurm_env(jobid: str | None) -> dict[str, str | int] | None:
    """Detect SLURM allocation details. Returns None if not available."""
    jobid = jobid or os.environ.get("SLURM_JOB_ID")
    if not jobid:
        return None

    try:
        result = subprocess.run(
            ["scontrol", "show", "job", str(jobid)],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None

        info = result.stdout
        nodes_match = re.search(r"NumNodes=(\d+)", info)
        tasks_match = re.search(r"NumTasks=(\d+)", info)
        nodelist_match = re.search(r"NodeList=(\S+)", info)
        state_match = re.search(r"JobState=(\w+)", info)

        if not all([nodes_match, nodelist_match, state_match]):
            return None
        if state_match.group(1) != "RUNNING":
            return None

        nodes = int(nodes_match.group(1))
        total_tasks = int(tasks_match.group(1)) if tasks_match else nodes
        gpus_per_node = total_tasks // nodes

        # Resolve master address
        hostnames = subprocess.run(
            ["scontrol", "show", "hostnames", nodelist_match.group(1)],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().split("\n")

        return {
            "jobid": str(jobid),
            "nodes": nodes,
            "gpus_per_node": gpus_per_node,
            "total_gpus": nodes * gpus_per_node,
            "nodelist": nodelist_match.group(1),
            "master_addr": hostnames[0],
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
class HardwareInfo:
    """Detected hardware capabilities."""

    def __init__(self, gpu_count: int, slurm: dict | None, use_slurm: bool):
        self.local_gpus = gpu_count
        self.slurm = slurm
        self.use_slurm = use_slurm and slurm is not None

        if self.use_slurm:
            self.total_gpus = slurm["total_gpus"]
            self.nodes = slurm["nodes"]
            self.gpus_per_node = slurm["gpus_per_node"]
        else:
            self.total_gpus = gpu_count
            self.nodes = 1
            self.gpus_per_node = gpu_count

        self.has_gpu = self.total_gpus > 0
        self.mode = "slurm" if self.use_slurm else ("gpu" if self.has_gpu else "cpu")

    def __repr__(self) -> str:
        if self.use_slurm:
            return f"SLURM({self.nodes}N×{self.gpus_per_node}G={self.total_gpus}GPU)"
        elif self.has_gpu:
            return f"Local({self.total_gpus}GPU)"
        else:
            return "CPU"


@pytest.fixture(scope="session")
def hw(request) -> HardwareInfo:
    """Session-scoped hardware info."""
    gpu_count = _detect_gpu_count()
    use_slurm = request.config.getoption("--slurm")
    jobid = request.config.getoption("--jobid")
    slurm = _detect_slurm_env(jobid) if use_slurm else None

    if use_slurm and slurm is None:
        pytest.fail(
            "--slurm requested but no valid SLURM allocation found. "
            "Pass --jobid or set SLURM_JOB_ID."
        )

    return HardwareInfo(gpu_count, slurm, use_slurm)


@pytest.fixture(scope="session")
def tokenizer_name(request) -> str:
    return request.config.getoption("--tokenizer")


@pytest.fixture(scope="session")
def vocab_size(request) -> int:
    return request.config.getoption("--vocab-size")
