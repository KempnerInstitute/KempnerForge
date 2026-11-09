"""Shutdown-signal handling in the SLURM launch scripts under ``scripts/slurm``."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

SLURM = Path(__file__).resolve().parents[2] / "scripts" / "slurm"
SRUN_SCRIPTS = ("7b_requeue.sh", "multinode.sh", "13b_validation.sh")

CHILD = """
import os, signal, sys, time
from pathlib import Path

out = Path(sys.argv[1])
got = []
signal.signal(signal.SIGTERM, lambda s, f: got.append(s))
(out / "pid").write_text(str(os.getpid()))
(out / "pid").rename(out / "ready")
deadline = time.monotonic() + 60
while not got and time.monotonic() < deadline:
    time.sleep(0.02)
if not got:
    sys.exit(3)
(out / "got").write_text(signal.Signals(got[0]).name)
time.sleep(0.5)
(out / "done").write_text("done")
sys.exit(7)
"""


@pytest.mark.parametrize("name", SRUN_SCRIPTS)
def test_srun_scripts_signal_the_job_steps(name):
    """``--signal=B:`` reaches only the batch shell, never the srun-launched ranks."""
    text = (SLURM / name).read_text()
    assert "#SBATCH --signal=SIGTERM@120" in text
    assert "--signal=B:" not in text


def test_singlenode_signals_the_batch_shell_and_traps_it():
    """singlenode.sh has no job step, so it keeps ``B:`` and forwards the signal."""
    text = (SLURM / "singlenode.sh").read_text()
    assert "#SBATCH --signal=B:SIGTERM@120" in text
    assert 'trap \'kill -TERM "${launcher}"' in text


@pytest.mark.parametrize("path", sorted(SLURM.glob("*.sh")), ids=lambda p: p.name)
def test_no_script_exports_nccl_timeout(path):
    """Neither NCCL nor PyTorch reads NCCL_TIMEOUT; nccl_timeout_sec is the bound."""
    assert "NCCL_TIMEOUT" not in path.read_text()


@pytest.mark.skipif(sys.platform == "win32", reason="needs bash and POSIX signals")
@pytest.mark.parametrize("signals", [["SIGTERM"], ["SIGUSR1"], ["SIGTERM", "SIGTERM"]])
def test_singlenode_forwards_the_batch_signal_and_waits(tmp_path, signals):
    """The batch shell survives the signal, sends SIGTERM to the launcher, exits with its code."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (tmp_path / "child.py").write_text(CHILD)
    (bin_dir / "uv").write_text(
        f'#!/bin/sh\nexec "{sys.executable}" "{tmp_path}/child.py" "{tmp_path}"\n'
    )
    (bin_dir / "ip").write_text("#!/bin/sh\nexit 0\n")
    for tool in ("uv", "ip"):
        (bin_dir / tool).chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", SLURM_JOB_ID="1")
    proc = subprocess.Popen(
        ["bash", str(SLURM / "singlenode.sh"), "configs/train/debug.toml"],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 30
    while not (tmp_path / "ready").exists():
        assert proc.poll() is None and time.monotonic() < deadline
        time.sleep(0.02)
    child = int((tmp_path / "ready").read_text())
    try:
        for name in signals:
            proc.send_signal(getattr(signal, name))
            time.sleep(0.1)
        assert proc.wait(timeout=30) == 7
        assert (tmp_path / "got").read_text() == "SIGTERM"
        assert (tmp_path / "done").exists()
    finally:
        proc.kill()
        with contextlib.suppress(OSError):
            if b"child.py" in Path(f"/proc/{child}/cmdline").read_bytes():
                os.kill(child, signal.SIGKILL)
