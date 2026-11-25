"""Helpers for tests that start DataLoader worker processes."""

from __future__ import annotations

import contextlib
import multiprocessing.util
import os
import signal
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

    The script runs in a session of its own. When it is still running
    ``EXIT_TIMEOUT_S`` seconds after it started, inside CI's 60 s limit per test,
    every process in that session is killed and the calling test fails.
    """
    root = str(Path(kempnerforge.__file__).resolve().parents[1])
    path = os.pathsep.join([root, *filter(None, [os.environ.get("PYTHONPATH")])])
    command = [sys.executable, str(script), *args]
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONPATH": path},
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=EXIT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            stdout, _ = process.communicate()
            pytest.fail(
                f"the process was still running {EXIT_TIMEOUT_S} s after it started; "
                f"its output: {stdout!r}"
            )
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
