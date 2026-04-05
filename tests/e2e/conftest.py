"""E2E test fixtures.

CLI flags (--e2e, --slow) and collection hooks are registered in
tests/conftest.py so they work regardless of which test directory is targeted.

Usage:
    uv run pytest tests/e2e/ --e2e          # run all e2e tests
    uv run pytest tests/e2e/ --e2e -k fsdp  # run specific e2e test
    uv run pytest tests/e2e/ --e2e --slow   # include slow tests (7B model)
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest


def _gpu_count() -> int:
    """Detect available CUDA GPUs without importing torch (fast)."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return int(result.stdout.strip()) if result.returncode == 0 else 0
    except Exception:
        return 0


GPU_COUNT = _gpu_count()


def requires_gpus(n: int):
    """Skip test if fewer than n GPUs are available."""
    return pytest.mark.skipif(n > GPU_COUNT, reason=f"Requires {n} GPUs, found {GPU_COUNT}")


@pytest.fixture
def synthetic_data_dir(tmp_path):
    """Create a self-contained synthetic dataset (small .npy token shards).

    Generates 2 shards of 65536 uint16 tokens each — enough for ~250 samples
    at seq_len=512 across 4 GPUs. No external data dependency.
    """
    for i in range(2):
        tokens = np.random.default_rng(seed=42 + i).integers(0, 32000, size=65536, dtype=np.uint16)
        np.save(tmp_path / f"shard_{i:03d}.npy", tokens)
    return str(tmp_path)
