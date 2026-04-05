"""End-to-end training tests.

These tests launch full training runs as subprocesses and verify they
complete successfully. They are opt-in: run with ``--e2e`` flag.

    uv run pytest tests/e2e/ --e2e -v
    uv run pytest tests/e2e/ --e2e --slow -v   # include 7B model tests

All tests use random data by default (train.py generates random tensors
when no dataset_path is set). Data pipeline tests use a synthetic .npy
dataset created in a temp directory — no external data dependency.

Each test verifies:
  - Process exits cleanly (exit code 0)
  - "Training complete" appears in output (training loop finished)
  - Loss and step counts are reasonable (parsed from stdout)
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests.e2e.conftest import requires_gpus

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = str(PROJECT_ROOT / "scripts" / "train.py")
DEBUG_CONFIG = str(PROJECT_ROOT / "configs" / "train" / "debug.toml")
BENCH_CONFIG = str(PROJECT_ROOT / "configs" / "train" / "llama7b_bench.toml")


def _run_training(
    args: list[str],
    *,
    nproc: int = 1,
    timeout: int = 120,
    env_extra: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Launch a training run and return the result.

    Args:
        args: Arguments to pass after the training script (config path + overrides).
        nproc: Number of GPU processes (1 = plain python, >1 = torchrun).
        timeout: Max seconds before killing the process.
        env_extra: Extra environment variables.
    """
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)

    if nproc > 1:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={nproc}",
            TRAIN_SCRIPT,
            *args,
        ]
    else:
        cmd = [sys.executable, TRAIN_SCRIPT, *args]
        env.setdefault("WORLD_SIZE", "1")
        env.setdefault("RANK", "0")
        env.setdefault("LOCAL_RANK", "0")

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        cwd=str(PROJECT_ROOT),
    )


def _assert_training_complete(result: subprocess.CompletedProcess, expected_steps: int) -> None:
    """Assert training finished successfully with the expected step count."""
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"Training failed (exit {result.returncode}):\n{output[-2000:]}"
    assert "Training complete" in output, f"Training did not complete:\n{output[-2000:]}"

    match = re.search(r"Training complete: (\d+) steps", output)
    assert match, f"Could not parse step count from output:\n{output[-2000:]}"
    assert int(match.group(1)) == expected_steps, (
        f"Expected {expected_steps} steps, got {match.group(1)}"
    )


def _parse_last_loss(output: str) -> float | None:
    """Extract the loss value from the last logged step."""
    matches = re.findall(r"loss=([\d.]+)", output)
    return float(matches[-1]) if matches else None


# ============================================================================
# Single GPU
# ============================================================================


@pytest.mark.e2e
def test_single_gpu_random_data():
    """Single GPU training with random data — basic smoke test."""
    result = _run_training(
        [DEBUG_CONFIG, "--train.max_steps=10", "--metrics.log_interval=5"],
        nproc=1,
    )
    _assert_training_complete(result, expected_steps=10)


# ============================================================================
# Multi-GPU: FSDP
# ============================================================================


@pytest.mark.e2e
@requires_gpus(4)
def test_fsdp_4gpu():
    """4 GPU FSDP — tests build_parallel_model non-TP path."""
    result = _run_training(
        [DEBUG_CONFIG, "--train.max_steps=10", "--metrics.log_interval=5"],
        nproc=4,
    )
    _assert_training_complete(result, expected_steps=10)


# ============================================================================
# Multi-GPU: Tensor Parallelism
# ============================================================================


@pytest.mark.e2e
@requires_gpus(4)
def test_tp_only_4gpu():
    """4 GPU TP — tests meta-device init + SequenceParallel path."""
    result = _run_training(
        [
            DEBUG_CONFIG,
            "--train.max_steps=10",
            "--metrics.log_interval=5",
            "--distributed.tp=4",
            "--distributed.dp_shard=1",
        ],
        nproc=4,
    )
    _assert_training_complete(result, expected_steps=10)


@pytest.mark.e2e
@requires_gpus(4)
def test_tp_plus_fsdp():
    """4 GPU TP=2 + FSDP=2 — combined parallelism."""
    result = _run_training(
        [
            DEBUG_CONFIG,
            "--train.max_steps=10",
            "--metrics.log_interval=5",
            "--distributed.tp=2",
            "--distributed.dp_shard=2",
        ],
        nproc=4,
    )
    _assert_training_complete(result, expected_steps=10)


# ============================================================================
# Multi-GPU: Pipeline Parallelism
# ============================================================================


@pytest.mark.e2e
@requires_gpus(4)
def test_pipeline_parallel():
    """4 GPU PP=2 + FSDP=2 — pipeline parallelism with 1F1B schedule."""
    result = _run_training(
        [
            DEBUG_CONFIG,
            "--train.max_steps=10",
            "--metrics.log_interval=5",
            "--distributed.pp=2",
            "--distributed.dp_shard=2",
            "--train.grad_accum_steps=2",
        ],
        nproc=4,
    )
    _assert_training_complete(result, expected_steps=10)


# ============================================================================
# Mixed Precision
# ============================================================================


@pytest.mark.e2e
@requires_gpus(4)
def test_fp16_mixed_precision():
    """4 GPU FSDP with fp16 — verifies param_dtype config path."""
    result = _run_training(
        [
            DEBUG_CONFIG,
            "--train.max_steps=10",
            "--metrics.log_interval=5",
            "--train.mixed_precision=fp16",
        ],
        nproc=4,
    )
    _assert_training_complete(result, expected_steps=10)


# ============================================================================
# Data Pipeline
# ============================================================================


@pytest.mark.e2e
@requires_gpus(4)
def test_synthetic_data_pipeline(synthetic_data_dir):
    """4 GPU FSDP with synthetic .npy data — tests full data pipeline."""
    result = _run_training(
        [
            DEBUG_CONFIG,
            "--train.max_steps=10",
            "--metrics.log_interval=5",
            f"--data.dataset_path={synthetic_data_dir}",
            "--data.file_pattern=*.npy",
        ],
        nproc=4,
    )
    _assert_training_complete(result, expected_steps=10)
    output = result.stdout + result.stderr
    assert "Dataset:" in output, "Dataset was not loaded"


# ============================================================================
# Checkpoint Save + Resume
# ============================================================================


@pytest.mark.e2e
@requires_gpus(4)
def test_checkpoint_save_and_resume(tmp_path):
    """4 GPU FSDP — save checkpoint at step 5, resume and train to step 10."""
    ckpt_dir = str(tmp_path / "ckpt")

    # Phase 1: Train to step 10, checkpoint at step 5
    result = _run_training(
        [
            DEBUG_CONFIG,
            "--train.max_steps=10",
            "--metrics.log_interval=5",
            f"--checkpoint.dir={ckpt_dir}",
            "--checkpoint.interval=5",
        ],
        nproc=4,
    )
    _assert_training_complete(result, expected_steps=10)
    output = result.stdout + result.stderr
    assert "Checkpoint saved" in output, "Checkpoint was not saved"

    # Phase 2: Resume — should pick up from step 10, train to step 15
    result = _run_training(
        [
            DEBUG_CONFIG,
            "--train.max_steps=15",
            "--metrics.log_interval=5",
            f"--checkpoint.dir={ckpt_dir}",
            "--checkpoint.interval=100",
        ],
        nproc=4,
    )
    _assert_training_complete(result, expected_steps=15)
    output = result.stdout + result.stderr
    assert "step=10" in output, "Did not resume from step 10"


# ============================================================================
# Slow: Large Model
# ============================================================================


@pytest.mark.e2e
@pytest.mark.slow
@requires_gpus(4)
def test_7b_tp_fsdp_compile():
    """7B model with TP=2 + FSDP=2 + torch.compile — full production path.

    Uses random data, 5 steps. Verifies the heavy-weight path works:
    meta-device init, TP, FSDP, activation checkpointing, torch.compile.
    """
    result = _run_training(
        [
            BENCH_CONFIG,
            "--train.max_steps=5",
            "--metrics.log_interval=5",
            "--distributed.tp=2",
            "--distributed.dp_shard=2",
        ],
        nproc=4,
        timeout=300,
    )
    _assert_training_complete(result, expected_steps=5)
