"""Helpers for tests that start DataLoader worker processes."""

from __future__ import annotations

import multiprocessing.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

import kempnerforge

EXIT_TIMEOUT_S = 50


def worker_sockets_fit() -> bool:
    """Whether DataLoader workers can hand batches to this process here.

    Workers send tensors over AF_UNIX sockets created in the multiprocessing temp
    directory, and a socket path may not exceed 107 bytes.
    """
    return len(multiprocessing.util.get_temp_dir()) + len("/listener-12345678") <= 107


def run_to_exit(script: Path, *args: str) -> subprocess.CompletedProcess:
    """Run ``script`` with ``args`` in a new interpreter that imports this ``kempnerforge``.

    Fails the calling test when the process is still running ``EXIT_TIMEOUT_S``
    seconds after it started, inside CI's 60 s limit per test.
    """
    root = str(Path(kempnerforge.__file__).resolve().parents[1])
    path = os.pathsep.join([root, *filter(None, [os.environ.get("PYTHONPATH")])])
    try:
        return subprocess.run(
            [sys.executable, str(script), *args],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": path},
            timeout=EXIT_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as hung:
        pytest.fail(
            f"the process was still running {EXIT_TIMEOUT_S} s after it started; "
            f"its output so far: {hung.stdout!r}"
        )
