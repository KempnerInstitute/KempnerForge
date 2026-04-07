"""Unit tests for distributed utilities (no GPU/multi-process required)."""

from __future__ import annotations

import os
from unittest.mock import patch

import torch

from kempnerforge.config.schema import ModelConfig
from kempnerforge.distributed.setup import _detect_ib_interface, _set_nccl_env, get_world_info
from kempnerforge.distributed.tensor_parallel import _build_block_tp_plan
from kempnerforge.distributed.utils import clip_grad_norm_
from kempnerforge.model.transformer import TransformerBlock


def _make_block(activation: str = "silu") -> TransformerBlock:
    """Create a small TransformerBlock on CPU for plan builder testing."""
    config = ModelConfig(dim=64, n_layers=2, n_heads=4, vocab_size=128, activation=activation)
    return TransformerBlock(config, layer_idx=0)


class TestClipGradNorm:
    """Tests for clip_grad_norm_ on plain (non-FSDP) tensors."""

    def _make_model_with_grad(self, grad_value: float) -> torch.nn.Module:
        """Create a small model with uniform gradients for predictable norms."""
        model = torch.nn.Linear(4, 4, bias=False)
        model.weight.grad = torch.full_like(model.weight, grad_value)
        return model

    def test_returns_total_norm(self):
        model = self._make_model_with_grad(1.0)
        # 4x4 matrix of 1s → norm = sqrt(16) = 4.0
        norm = clip_grad_norm_(model, max_norm=100.0)
        assert abs(norm.item() - 4.0) < 1e-5

    def test_clips_when_above_max(self):
        model = self._make_model_with_grad(1.0)
        # norm=4.0, clip to 2.0 → gradients scaled by 2.0/4.0 = 0.5
        clip_grad_norm_(model, max_norm=2.0)
        assert torch.allclose(model.weight.grad, torch.full((4, 4), 0.5), atol=1e-5)

    def test_no_clip_when_below_max(self):
        model = self._make_model_with_grad(1.0)
        original_grad = model.weight.grad.clone()
        # norm=4.0, max_norm=10.0 → no clipping
        clip_grad_norm_(model, max_norm=10.0)
        assert torch.allclose(model.weight.grad, original_grad)

    def test_no_grads_returns_zero(self):
        model = torch.nn.Linear(4, 4, bias=False)
        # No .grad set
        norm = clip_grad_norm_(model, max_norm=1.0)
        assert norm.item() == 0.0


# ---------------------------------------------------------------------------
# Tensor parallel plan builder
# ---------------------------------------------------------------------------


class TestBuildBlockTPPlan:
    def test_plan_with_sequence_parallel(self):
        """Plan should include SequenceParallel for norms and Shard(1) layouts."""
        block = _make_block()
        plan = _build_block_tp_plan(block, sequence_parallel=True)
        assert "attention_norm" in plan
        assert "mlp_norm" in plan
        assert "attention.q_proj" in plan
        assert "attention.o_proj" in plan
        assert "mlp.down_proj" in plan

    def test_plan_without_sequence_parallel(self):
        """Plan should NOT include SequenceParallel norms in basic TP mode."""
        block = _make_block()
        plan = _build_block_tp_plan(block, sequence_parallel=False)
        assert "attention_norm" not in plan
        assert "mlp_norm" not in plan
        # Projections should still be parallelized
        assert "attention.q_proj" in plan
        assert "attention.o_proj" in plan

    def test_swiglu_includes_gate_proj(self):
        """SwiGLU blocks should shard both gate_proj and up_proj."""
        block = _make_block(activation="silu")  # SwiGLU
        plan = _build_block_tp_plan(block, sequence_parallel=False)
        assert "mlp.gate_proj" in plan
        assert "mlp.up_proj" in plan
        assert "mlp.down_proj" in plan

    def test_standard_mlp_excludes_gate_proj(self):
        """Standard MLP blocks should only shard up_proj, not gate_proj."""
        block = _make_block(activation="relu")  # StandardMLP
        plan = _build_block_tp_plan(block, sequence_parallel=False)
        assert "mlp.gate_proj" not in plan
        assert "mlp.up_proj" in plan
        assert "mlp.down_proj" in plan

    def test_all_attention_projections_present(self):
        """All four attention projections should be in the plan."""
        block = _make_block()
        plan = _build_block_tp_plan(block, sequence_parallel=False)
        for proj in [
            "attention.q_proj",
            "attention.k_proj",
            "attention.v_proj",
            "attention.o_proj",
        ]:
            assert proj in plan, f"{proj} missing from plan"

    def test_plan_key_count_with_sp(self):
        """With SP + SwiGLU: 2 norms + 4 attn + 3 mlp = 9 entries."""
        block = _make_block(activation="silu")
        plan = _build_block_tp_plan(block, sequence_parallel=True)
        assert len(plan) == 9

    def test_plan_key_count_without_sp(self):
        """Without SP + SwiGLU: 4 attn + 3 mlp = 7 entries (no norms)."""
        block = _make_block(activation="silu")
        plan = _build_block_tp_plan(block, sequence_parallel=False)
        assert len(plan) == 7


# ---------------------------------------------------------------------------
# InfiniBand interface detection
# ---------------------------------------------------------------------------


class TestDetectIBInterface:
    def _mock_ip_output(self, stdout: str):
        """Create a mock for subprocess.run returning the given stdout."""
        from unittest.mock import MagicMock

        result = MagicMock()
        result.stdout = stdout
        return result

    def test_detects_ib0(self):
        """Should return 'ib0' when it's UP with an IP address."""
        output = (
            "lo               UNKNOWN        127.0.0.1/8\n"
            "eth0             UP             10.0.0.1/24\n"
            "ib0              UP             172.16.0.1/16\n"
        )
        with patch("subprocess.run", return_value=self._mock_ip_output(output)):
            assert _detect_ib_interface() == "ib0"

    def test_detects_first_ib_interface(self):
        """Should return the first UP IB interface when multiple exist."""
        output = (
            "ib0              DOWN           \n"
            "ib1              UP             172.16.1.1/16\n"
            "ib2              UP             172.16.2.1/16\n"
        )
        with patch("subprocess.run", return_value=self._mock_ip_output(output)):
            assert _detect_ib_interface() == "ib1"

    def test_returns_none_when_no_ib(self):
        """Should return None when no IB interfaces exist."""
        output = (
            "lo               UNKNOWN        127.0.0.1/8\n"
            "eth0             UP             10.0.0.1/24\n"
        )
        with patch("subprocess.run", return_value=self._mock_ip_output(output)):
            assert _detect_ib_interface() is None

    def test_returns_none_when_ib_down(self):
        """Should return None when IB interface exists but is DOWN."""
        output = (
            "ib0              DOWN           \n"
            "eth0             UP             10.0.0.1/24\n"
        )
        with patch("subprocess.run", return_value=self._mock_ip_output(output)):
            assert _detect_ib_interface() is None

    def test_returns_none_on_subprocess_error(self):
        """Should return None if ip command fails."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert _detect_ib_interface() is None


# ---------------------------------------------------------------------------
# NCCL environment setup
# ---------------------------------------------------------------------------


class TestSetNcclEnv:
    def test_sets_ib_env_vars(self):
        """Should set NCCL and GLOO socket interface when IB is detected."""
        env = {}
        with patch.dict(os.environ, env, clear=True), patch(
            "kempnerforge.distributed.setup._detect_ib_interface", return_value="ib0"
        ):
            _set_nccl_env()
            assert os.environ["NCCL_SOCKET_IFNAME"] == "ib0"
            assert os.environ["GLOO_SOCKET_IFNAME"] == "ib0"
            assert os.environ["NCCL_IB_DISABLE"] == "0"

    def test_respects_existing_nccl_socket_ifname(self):
        """Should not override NCCL_SOCKET_IFNAME if already set."""
        env = {"NCCL_SOCKET_IFNAME": "eth0"}
        with patch.dict(os.environ, env, clear=True):
            # _detect_ib_interface should NOT be called when env var is already set
            _set_nccl_env()
            assert os.environ["NCCL_SOCKET_IFNAME"] == "eth0"
            # GLOO should default to the same value from env
            assert os.environ["GLOO_SOCKET_IFNAME"] == "eth0"

    def test_no_ib_detected(self):
        """Should not set socket ifname when no IB interface is found."""
        env = {}
        with patch.dict(os.environ, env, clear=True), patch(
            "kempnerforge.distributed.setup._detect_ib_interface", return_value=None
        ):
            _set_nccl_env()
            assert "NCCL_SOCKET_IFNAME" not in os.environ
            assert "GLOO_SOCKET_IFNAME" not in os.environ
            # IB defaults should still be set
            assert os.environ["NCCL_IB_DISABLE"] == "0"


# ---------------------------------------------------------------------------
# get_world_info
# ---------------------------------------------------------------------------


class TestGetWorldInfo:
    def test_defaults_to_single_process(self):
        """Should return (0, 0, 1) when no env vars are set."""
        env = {}
        with patch.dict(os.environ, env, clear=True):
            rank, local_rank, world_size = get_world_info()
            assert (rank, local_rank, world_size) == (0, 0, 1)

    def test_reads_torchrun_env(self):
        """Should read RANK/LOCAL_RANK/WORLD_SIZE from torchrun."""
        env = {"RANK": "3", "LOCAL_RANK": "1", "WORLD_SIZE": "8"}
        with patch.dict(os.environ, env, clear=True):
            rank, local_rank, world_size = get_world_info()
            assert (rank, local_rank, world_size) == (3, 1, 8)

    def test_falls_back_to_slurm_env(self):
        """Should fall back to SLURM_PROCID/SLURM_LOCALID/SLURM_NTASKS."""
        env = {"SLURM_PROCID": "5", "SLURM_LOCALID": "2", "SLURM_NTASKS": "32"}
        with patch.dict(os.environ, env, clear=True):
            rank, local_rank, world_size = get_world_info()
            assert (rank, local_rank, world_size) == (5, 2, 32)

    def test_torchrun_takes_precedence_over_slurm(self):
        """torchrun env vars should take precedence over SLURM vars."""
        env = {
            "RANK": "1",
            "LOCAL_RANK": "1",
            "WORLD_SIZE": "4",
            "SLURM_PROCID": "99",
            "SLURM_LOCALID": "99",
            "SLURM_NTASKS": "99",
        }
        with patch.dict(os.environ, env, clear=True):
            rank, local_rank, world_size = get_world_info()
            assert (rank, local_rank, world_size) == (1, 1, 4)

    def test_sets_standard_env_vars(self):
        """Should set RANK/LOCAL_RANK/WORLD_SIZE for PyTorch env:// rendezvous."""
        env = {"SLURM_PROCID": "5", "SLURM_LOCALID": "2", "SLURM_NTASKS": "32"}
        with patch.dict(os.environ, env, clear=True):
            get_world_info()
            assert os.environ["RANK"] == "5"
            assert os.environ["LOCAL_RANK"] == "2"
            assert os.environ["WORLD_SIZE"] == "32"
