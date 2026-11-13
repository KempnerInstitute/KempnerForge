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
SCRIPTS = sorted(SLURM.glob("*.sh"))
SRUN_SCRIPTS = ("7b_requeue.sh", "multinode.sh", "13b_validation.sh")
FORWARD_ONCE = (
    'trap \'[ "$forwarded" = 1 ] || '
    '{ forwarded=1; kill -TERM "${launcher}" 2>/dev/null || true; }\' TERM USR1'
)

CHILD = """
import os, signal, sys, time
from pathlib import Path

out, code = Path(sys.argv[1]), int(sys.argv[2])
got = []
signal.signal(signal.SIGTERM, lambda s, f: got.append(s))
(out / "pid").write_text(str(os.getpid()))
(out / "pid").rename(out / "ready")
deadline = time.monotonic() + 60
while not got and time.monotonic() < deadline:
    time.sleep(0.02)
if not got:
    sys.exit(3)
time.sleep(1.0)
(out / "got").write_text(",".join(signal.Signals(s).name for s in got))
sys.exit(code)
"""


def sbatch_directives(path: Path) -> list[str]:
    """Return the ``#SBATCH`` lines sbatch reads: those before the first command."""
    directives = []
    for line in path.read_text().splitlines()[1:]:
        if line.startswith("#SBATCH"):
            directives.append(line.strip())
        elif line.strip() and not line.startswith("#"):
            break
    return directives


def test_the_launch_scripts_are_found():
    """The parametrized checks below run over these scripts."""
    names = {path.name for path in SCRIPTS}
    assert {
        *SRUN_SCRIPTS,
        "singlenode.sh",
        "_run_training.sh",
        "_run_distributed_tests.sh",
    } <= names


@pytest.mark.parametrize("name", SRUN_SCRIPTS)
def test_srun_scripts_signal_the_job_steps(name):
    """``--signal=B:`` reaches only the batch shell, never the srun-launched ranks."""
    directives = sbatch_directives(SLURM / name)
    assert "#SBATCH --signal=SIGTERM@120" in directives
    assert not any("--signal=B:" in line for line in directives)


def test_singlenode_signals_the_batch_shell_and_traps_it():
    """singlenode.sh has no job step, so it keeps ``B:`` and forwards the signal once."""
    script = SLURM / "singlenode.sh"
    assert "#SBATCH --signal=B:SIGTERM@120" in sbatch_directives(script)
    assert FORWARD_ONCE in [line.strip() for line in script.read_text().splitlines()]


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_no_script_exports_nccl_timeout(path):
    """Neither NCCL nor PyTorch reads NCCL_TIMEOUT."""
    assert "NCCL_TIMEOUT" not in path.read_text()


@pytest.mark.skipif(sys.platform == "win32", reason="needs bash and POSIX signals")
@pytest.mark.parametrize("code", [7, 0])
@pytest.mark.parametrize(
    "signals",
    [["SIGTERM"], ["SIGUSR1"], ["SIGTERM", "SIGTERM"]],
    ids=["TERM", "USR1", "TERM-TERM"],
)
def test_singlenode_forwards_the_batch_signal_once_and_waits(tmp_path, signals, code):
    """The shell survives, sends the launcher one SIGTERM and exits with its status."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (tmp_path / "child.py").write_text(CHILD)
    (bin_dir / "uv").write_text(
        f'#!/bin/sh\nexec "{sys.executable}" "{tmp_path}/child.py" "{tmp_path}" {code}\n'
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
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 30
        while not (tmp_path / "ready").exists():
            assert proc.poll() is None and time.monotonic() < deadline
            time.sleep(0.02)
        for name in signals:
            proc.send_signal(getattr(signal, name))
            time.sleep(0.1)
        assert proc.wait(timeout=30) == code
        assert (tmp_path / "got").read_text() == "SIGTERM"
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
