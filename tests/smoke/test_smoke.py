"""Smoke tests: train → checkpoint → eval across parallelism configurations.

Auto-detects hardware and selects applicable tests:
  - CPU only: config validation, single-step CPU training
  - 1 GPU: dense + MoE single-GPU training and eval
  - 4+ GPUs: FSDP, TP, PP, FP8, compile, activation checkpointing
  - Multi-node (SLURM): all of the above at full scale

Usage:
    # Default (torchrun, auto-detect GPUs)
    uv run pytest tests/smoke/ --smoke -v

    # SLURM multi-node
    uv run pytest tests/smoke/ --smoke --slurm --jobid=<ID> -v

    # Filter specific tests
    uv run pytest tests/smoke/ --smoke -v -k "moe"

    # Custom tokenizer for eval
    uv run pytest tests/smoke/ --smoke -v --tokenizer meta-llama/Llama-2-7b-hf --vocab-size 32000
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = str(PROJECT_ROOT / "scripts" / "train.py")
EVAL_SCRIPT = str(PROJECT_ROOT / "scripts" / "eval.py")
DEBUG_CONFIG = str(PROJECT_ROOT / "configs" / "train" / "debug.toml")
DEBUG_MOE_CONFIG = str(PROJECT_ROOT / "configs" / "train" / "debug_moe.toml")
FP8_7B_CONFIG = str(PROJECT_ROOT / "configs" / "train" / "7b_16gpu_fp8.toml")
CKPT_ROOT = PROJECT_ROOT / "checkpoints" / "_smoke_test"

# Port counter to avoid collisions between tests in the same session
_port_counter = 15600


def _next_port() -> int:
    global _port_counter
    _port_counter += 2
    return _port_counter


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Launch helpers
# ---------------------------------------------------------------------------
def _run_torchrun(
    script: str,
    args: list[str],
    nproc: int,
    timeout: int = 180,
) -> subprocess.CompletedProcess:
    """Launch via torchrun (single node)."""
    env = os.environ.copy()

    if nproc > 1:
        port = _find_free_port()
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={nproc}",
            f"--master_port={port}",
            script,
            *args,
        ]
    else:
        cmd = [sys.executable, script, *args]
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


def _run_srun(
    script: str,
    args: list[str],
    slurm: dict,
    timeout: int = 300,
) -> subprocess.CompletedProcess:
    """Launch via srun on a SLURM allocation."""
    port = _next_port()
    env = os.environ.copy()
    env["MASTER_ADDR"] = slurm["master_addr"]
    env["MASTER_PORT"] = str(port)
    env["NCCL_SOCKET_IFNAME"] = os.environ.get("NCCL_SOCKET_IFNAME", "ib0")
    env["GLOO_SOCKET_IFNAME"] = os.environ.get("GLOO_SOCKET_IFNAME", "ib0")
    env["NCCL_IB_DISABLE"] = "0"

    cmd = [
        "srun",
        f"--jobid={slurm['jobid']}",
        f"--nodes={slurm['nodes']}",
        f"--ntasks={slurm['total_gpus']}",
        f"--gpus-per-node={slurm['gpus_per_node']}",
        "--kill-on-bad-exit=1",
        "uv",
        "run",
        "python",
        script,
        *args,
    ]

    # Use file-based capture to avoid pipe deadlocks with srun
    log_dir = CKPT_ROOT / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    tag = f"srun_{port}"
    stdout_f = log_dir / f"{tag}.out"
    stderr_f = log_dir / f"{tag}.err"

    with open(stdout_f, "w") as fout, open(stderr_f, "w") as ferr:
        proc = subprocess.Popen(cmd, stdout=fout, stderr=ferr, env=env, cwd=str(PROJECT_ROOT))
        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            stdout = stdout_f.read_text()
            stderr = stderr_f.read_text()
            pytest.fail(
                f"srun timed out after {timeout}s.\n"
                f"stdout:\n{stdout[-2000:]}\nstderr:\n{stderr[-2000:]}"
            )

    return subprocess.CompletedProcess(
        cmd,
        returncode=returncode,
        stdout=stdout_f.read_text(),
        stderr=stderr_f.read_text(),
    )


def _run(hw, script, args, timeout=180):
    """Dispatch to torchrun or srun based on hardware config."""
    if hw.use_slurm:
        return _run_srun(script, args, hw.slurm, timeout=timeout)
    else:
        return _run_torchrun(script, args, nproc=hw.total_gpus or 1, timeout=timeout)


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------
def _assert_train_ok(result, expected_steps=15):
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"Training failed (exit {result.returncode}):\n{output[-3000:]}"
    assert "Training complete" in output, f"Training did not finish:\n{output[-3000:]}"
    match = re.search(r"Training complete: (\d+) steps", output)
    assert match, f"Cannot parse step count:\n{output[-3000:]}"
    assert int(match.group(1)) == expected_steps


def _parse_train_loss(output: str) -> float | None:
    matches = re.findall(r"\[step \d+\] loss=([\d.]+)", output)
    return float(matches[-1]) if matches else None


def _parse_eval_json(output: str) -> dict | None:
    matches = re.findall(r'\{[^{}]*"eval/loss"[^{}]*\}', output, re.DOTALL)
    if matches:
        try:
            return json.loads(matches[-1])
        except json.JSONDecodeError:
            pass
    return None


def _assert_eval_ok(result):
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"Eval failed (exit {result.returncode}):\n{output[-3000:]}"
    data = _parse_eval_json(output)
    assert data is not None, f"No eval JSON in output:\n{output[-3000:]}"
    assert data["eval/loss"] > 0, f"Eval loss should be > 0: {data}"
    assert data["eval/perplexity"] > 1.0, f"Perplexity should be > 1: {data}"
    return data


# ---------------------------------------------------------------------------
# Common args builders
# ---------------------------------------------------------------------------
def _train_args(config, ckpt_name, extra=None, vocab_size=None):
    """Build standard train args."""
    ckpt_dir = CKPT_ROOT / ckpt_name
    args = [
        config,
        "--train.max_steps=15",
        "--metrics.log_interval=5",
        "--metrics.enable_wandb=false",
        "--metrics.enable_tensorboard=false",
        f"--checkpoint.dir={ckpt_dir}",
        "--checkpoint.interval=10",
        "--checkpoint.keep_last_n=1",
    ]
    if vocab_size is not None:
        args.append(f"--model.vocab_size={vocab_size}")
    if extra:
        args.extend(extra)
    return args


def _eval_args(config, ckpt_name, tokenizer, vocab_size, extra=None):
    """Build standard eval args."""
    ckpt_dir = CKPT_ROOT / ckpt_name
    args = [
        config,
        f"--model.vocab_size={vocab_size}",
        f"--checkpoint.load_path={ckpt_dir}/step_10",
        "--eval.hf_dataset_name=wikitext",
        "--eval.hf_dataset_config=wikitext-103-raw-v1",
        "--eval.steps=5",
        f"--data.tokenizer_path={tokenizer}",
    ]
    if extra:
        args.extend(extra)
    return args


# ---------------------------------------------------------------------------
# Skip helpers
# ---------------------------------------------------------------------------
def skip_unless_gpus(hw, n, reason=None):
    if hw.total_gpus < n:
        pytest.skip(reason or f"Requires {n}+ GPUs, have {hw.total_gpus}")


# ============================================================================
# Tests: CPU / No GPU
# ============================================================================


@pytest.mark.smoke
class TestCPU:
    """Tests that run on any machine, including CPU-only."""

    def test_config_loads(self):
        """Verify debug configs parse without error."""
        cmd = [
            sys.executable,
            "-c",
            "from kempnerforge.config.loader import load_config; "
            "c = load_config('configs/train/debug.toml', cli_args=[]); "
            "print(f'OK: {c.model.dim}d, {c.model.n_layers}L')",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        assert result.returncode == 0, result.stderr
        assert "OK:" in result.stdout

    def test_moe_config_loads(self):
        """Verify MoE debug config parses without error."""
        cmd = [
            sys.executable,
            "-c",
            "from kempnerforge.config.loader import load_config; "
            "c = load_config('configs/train/debug_moe.toml', cli_args=[]); "
            "print(f'OK: {c.model.num_experts} experts')",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        assert result.returncode == 0, result.stderr
        assert "OK: 4 experts" in result.stdout


# ============================================================================
# Tests: Single GPU
# ============================================================================


@pytest.mark.smoke
class TestSingleGPU:
    """Single-GPU tests. Skipped on CPU-only machines."""

    def test_dense_train(self, hw):
        skip_unless_gpus(hw, 1)
        result = _run_torchrun(
            TRAIN_SCRIPT,
            _train_args(DEBUG_CONFIG, "single_dense"),
            nproc=1,
        )
        _assert_train_ok(result)

    def test_moe_train(self, hw):
        skip_unless_gpus(hw, 1)
        result = _run_torchrun(
            TRAIN_SCRIPT,
            _train_args(DEBUG_MOE_CONFIG, "single_moe"),
            nproc=1,
        )
        _assert_train_ok(result)


# ============================================================================
# Tests: Full pipeline — train → checkpoint → eval
# ============================================================================


@pytest.mark.smoke
class TestFullPipeline:
    """Train → checkpoint → standalone eval. Core parallelism combos."""

    def test_dense_fsdp(self, hw, tokenizer_name, vocab_size):
        skip_unless_gpus(hw, 2)
        name = "pipe_dense_fsdp"
        result = _run(hw, TRAIN_SCRIPT, _train_args(DEBUG_CONFIG, name, vocab_size=vocab_size))
        _assert_train_ok(result)
        result = _run(hw, EVAL_SCRIPT, _eval_args(DEBUG_CONFIG, name, tokenizer_name, vocab_size))
        _assert_eval_ok(result)

    def test_dense_tp2_fsdp(self, hw, tokenizer_name, vocab_size):
        skip_unless_gpus(hw, 4)
        name = "pipe_dense_tp2"
        extra = ["--distributed.tp=2"]
        result = _run(
            hw,
            TRAIN_SCRIPT,
            _train_args(
                DEBUG_CONFIG,
                name,
                extra,
                vocab_size,
            ),
        )
        _assert_train_ok(result)
        result = _run(
            hw,
            EVAL_SCRIPT,
            _eval_args(
                DEBUG_CONFIG,
                name,
                tokenizer_name,
                vocab_size,
                extra,
            ),
        )
        _assert_eval_ok(result)

    def test_moe_fsdp(self, hw, tokenizer_name, vocab_size):
        skip_unless_gpus(hw, 2)
        name = "pipe_moe_fsdp"
        result = _run(
            hw,
            TRAIN_SCRIPT,
            _train_args(
                DEBUG_MOE_CONFIG,
                name,
                vocab_size=vocab_size,
            ),
        )
        _assert_train_ok(result)
        result = _run(
            hw,
            EVAL_SCRIPT,
            _eval_args(
                DEBUG_MOE_CONFIG,
                name,
                tokenizer_name,
                vocab_size,
            ),
        )
        _assert_eval_ok(result)

    def test_moe_tp2_fsdp(self, hw, tokenizer_name, vocab_size):
        skip_unless_gpus(hw, 4)
        name = "pipe_moe_tp2"
        extra = ["--distributed.tp=2"]
        result = _run(
            hw,
            TRAIN_SCRIPT,
            _train_args(
                DEBUG_MOE_CONFIG,
                name,
                extra,
                vocab_size,
            ),
        )
        _assert_train_ok(result)
        result = _run(
            hw,
            EVAL_SCRIPT,
            _eval_args(
                DEBUG_MOE_CONFIG,
                name,
                tokenizer_name,
                vocab_size,
                extra,
            ),
        )
        _assert_eval_ok(result)


# ============================================================================
# Tests: Train-only — verify specific features don't crash
# ============================================================================


@pytest.mark.smoke
class TestTrainFeatures:
    """Train-only tests for specific features. No eval — just verify no crash."""

    def test_dense_compile(self, hw):
        skip_unless_gpus(hw, 2)
        result = _run(
            hw,
            TRAIN_SCRIPT,
            _train_args(DEBUG_CONFIG, "feat_compile", ["--train.compile_model=true"]),
            timeout=300,  # compile is slow first time
        )
        _assert_train_ok(result)

    def test_dense_fp8(self, hw):
        skip_unless_gpus(hw, 1)
        result = _run(
            hw,
            TRAIN_SCRIPT,
            _train_args(
                DEBUG_CONFIG,
                "feat_fp8",
                ["--train.mixed_precision=fp8"],
            ),
        )
        _assert_train_ok(result)

    def test_dense_grad_accum(self, hw):
        skip_unless_gpus(hw, 2)
        result = _run(
            hw,
            TRAIN_SCRIPT,
            _train_args(
                DEBUG_CONFIG,
                "feat_grad_accum",
                ["--train.grad_accum_steps=4"],
            ),
        )
        _assert_train_ok(result)

    def test_dense_ac_full(self, hw):
        skip_unless_gpus(hw, 2)
        result = _run(
            hw,
            TRAIN_SCRIPT,
            _train_args(
                DEBUG_CONFIG,
                "feat_ac",
                ["--train.activation_checkpointing=full"],
            ),
        )
        _assert_train_ok(result)

    def test_moe_fp8(self, hw):
        skip_unless_gpus(hw, 1)
        result = _run(
            hw,
            TRAIN_SCRIPT,
            _train_args(
                DEBUG_MOE_CONFIG,
                "feat_moe_fp8",
                ["--train.mixed_precision=fp8"],
            ),
        )
        _assert_train_ok(result)

    def test_moe_sigmoid_router(self, hw):
        """MoE with sigmoid router + sequence aux loss + gradient scaling + bias schedule."""
        skip_unless_gpus(hw, 1)
        result = _run(
            hw,
            TRAIN_SCRIPT,
            _train_args(
                DEBUG_MOE_CONFIG,
                "feat_moe_sigmoid",
                [
                    "--model.moe_router=sigmoid_topk",
                    "--model.moe_sequence_aux_loss_weight=0.01",
                    "--model.moe_gradient_scale=true",
                    "--model.moe_bias_schedule=cosine_decay",
                    "--model.moe_aux_loss_weight=0.01",
                ],
            ),
        )
        _assert_train_ok(result)

    def test_dense_pp2(self, hw):
        skip_unless_gpus(hw, 4)
        result = _run(
            hw,
            TRAIN_SCRIPT,
            _train_args(
                DEBUG_CONFIG,
                "feat_pp2",
                ["--distributed.pp=2", "--train.grad_accum_steps=2"],
            ),
        )
        _assert_train_ok(result)

    def test_dense_inline_eval(self, hw, tokenizer_name, vocab_size):
        skip_unless_gpus(hw, 2)
        result = _run(
            hw,
            TRAIN_SCRIPT,
            _train_args(
                DEBUG_CONFIG,
                "feat_inline_eval",
                [
                    "--eval.enabled=true",
                    "--eval.interval=5",
                    "--eval.steps=3",
                    "--eval.hf_dataset_name=wikitext",
                    "--eval.hf_dataset_config=wikitext-103-raw-v1",
                    f"--data.tokenizer_path={tokenizer_name}",
                ],
                vocab_size=vocab_size,
            ),
        )
        _assert_train_ok(result)


# ============================================================================
# Tests: Checkpoint auto-resume (post-release fix regression coverage)
# ============================================================================


def _assert_resume_markers(output: str, expected_resume_step: int) -> None:
    """Assert the warm-resume log markers fire and carry the right step count."""
    assert "Auto-resume: found latest checkpoint" in output, (
        f"missing auto-resume marker:\n{output[-3000:]}"
    )
    assert "Restored RNG states" in output, f"missing RNG restore marker:\n{output[-3000:]}"
    assert f"Resumed from step {expected_resume_step}" in output, (
        f"did not resume at step {expected_resume_step}:\n{output[-3000:]}"
    )
    assert f"skip_batches={expected_resume_step}" in output, (
        f"dataloader skip_batches not restored to {expected_resume_step}:\n{output[-3000:]}"
    )
    assert "Applied stashed dataloader state" in output, (
        f"stashed dataloader state not applied:\n{output[-3000:]}"
    )


@pytest.mark.smoke
class TestAutoResume:
    """Train → save → relaunch → verify full state restoration.

    Exercises the checkpoint save / auto-resume path end-to-end: init-path
    barrier timeout, RNG restoration, StatefulDataLoader replay with monotonic
    batches_yielded, train_state.pt ownership gate, and scheduler continuity.

    Requires a pre-tokenized dataset (``--data-path``) because ``scripts/train.py``
    falls back to synthetic ``torch.randint`` batches when no data source is
    configured, which bypasses StatefulDataLoader entirely.
    """

    def _common_args(self, config, ckpt_name, data_path, file_pattern, data_vocab_size):
        ckpt_dir = CKPT_ROOT / ckpt_name
        return [
            config,
            f"--model.vocab_size={data_vocab_size}",
            f"--data.dataset_path={data_path}",
            f"--data.file_pattern={file_pattern}",
            "--data.pack_sequences=false",
            f"--checkpoint.dir={ckpt_dir}",
            "--checkpoint.interval=10",
            "--checkpoint.keep_last_n=2",
            "--checkpoint.async_mode=disabled",
            "--metrics.log_interval=5",
            "--metrics.enable_wandb=false",
            "--metrics.enable_tensorboard=false",
        ]

    def test_dense_fsdp_auto_resume(self, hw, data_path, file_pattern, data_vocab_size):
        if data_path is None:
            pytest.skip("Requires --data-path to exercise StatefulDataLoader")
        skip_unless_gpus(hw, 2)

        args = self._common_args(
            DEBUG_CONFIG, "resume_dense_fsdp", data_path, file_pattern, data_vocab_size
        )
        r1 = _run(hw, TRAIN_SCRIPT, args + ["--train.max_steps=20"])
        _assert_train_ok(r1, expected_steps=20)

        r2 = _run(hw, TRAIN_SCRIPT, args + ["--train.max_steps=30"])
        output2 = r2.stdout + r2.stderr
        assert r2.returncode == 0, f"Resume failed:\n{output2[-3000:]}"
        _assert_resume_markers(output2, expected_resume_step=20)
        match = re.search(r"Training complete: (\d+) steps", output2)
        assert match and int(match.group(1)) == 30, (
            f"Expected 30 steps, got output:\n{output2[-1000:]}"
        )

    def test_moe_fsdp_auto_resume(self, hw, data_path, file_pattern, data_vocab_size):
        if data_path is None:
            pytest.skip("Requires --data-path to exercise StatefulDataLoader")
        skip_unless_gpus(hw, 2)

        args = self._common_args(
            DEBUG_MOE_CONFIG, "resume_moe_fsdp", data_path, file_pattern, data_vocab_size
        )
        r1 = _run(hw, TRAIN_SCRIPT, args + ["--train.max_steps=20"])
        _assert_train_ok(r1, expected_steps=20)

        r2 = _run(hw, TRAIN_SCRIPT, args + ["--train.max_steps=30"])
        output2 = r2.stdout + r2.stderr
        assert r2.returncode == 0, f"MoE resume failed:\n{output2[-3000:]}"
        _assert_resume_markers(output2, expected_resume_step=20)
        assert "moe/aux_loss" in output2, (
            f"MoE aux_loss metric missing post-resume:\n{output2[-1000:]}"
        )


# ============================================================================
# Tests: Real-config sanity (post-release fix regression coverage)
# ============================================================================


@pytest.mark.smoke
class TestRealConfigs:
    """Run the published production configs at reduced scope to catch drift.

    Complements TestTrainFeatures which exercises individual features via
    ``debug.toml`` overrides. These tests execute the actual config file so
    any silent rot (renamed field, stale comment, broken default) surfaces.
    """

    def test_fp8_7b_config(self, hw, data_path, file_pattern):
        """7B FP8 training starts cleanly and applies Float8 + FSDP2 float8 all-gather."""
        if data_path is None:
            pytest.skip("Requires --data-path; 7B config uses dataset_path not synthetic")
        skip_unless_gpus(hw, 4)
        ckpt_dir = CKPT_ROOT / "fp8_7b"
        result = _run(
            hw,
            TRAIN_SCRIPT,
            [
                FP8_7B_CONFIG,
                f"--data.dataset_path={data_path}",
                f"--data.file_pattern={file_pattern}",
                "--data.pack_sequences=false",
                "--train.max_steps=3",
                "--train.batch_size=4",
                "--train.grad_accum_steps=1",
                "--train.compile_model=false",
                f"--checkpoint.dir={ckpt_dir}",
                "--checkpoint.interval=100",
                "--checkpoint.async_mode=disabled",
                "--metrics.enable_wandb=false",
                "--metrics.enable_tensorboard=false",
                "--metrics.log_interval=1",
            ],
            timeout=600,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0, f"FP8 7B failed:\n{output[-3000:]}"
        assert "Applied Float8 training" in output, f"Float8 was not applied:\n{output[-2000:]}"
        assert "fsdp_float8_all_gather=True" in output, (
            f"FSDP2 float8 all-gather was not enabled:\n{output[-2000:]}"
        )
        _assert_train_ok(result, expected_steps=3)


# ============================================================================
# Session cleanup
# ============================================================================


@pytest.fixture(autouse=True, scope="session")
def _cleanup_checkpoints(request):
    """Clean up smoke test checkpoints after all tests."""
    yield
    if CKPT_ROOT.exists():
        shutil.rmtree(CKPT_ROOT, ignore_errors=True)
