"""Unit tests for distributed seed coverage.

Cold-start seeding must cover the same four RNGs captured on checkpoint
(Python random, NumPy, torch CPU, torch CUDA) so that cold-start runs
have the same reproducibility guarantees as warm-resumed runs.
"""

from __future__ import annotations

import inspect
import random

import numpy as np
import pytest
import torch

from kempnerforge.distributed.setup import _set_seed


def test_set_seed_seeds_all_four_generators():
    """_set_seed must make Python random, NumPy, and torch CPU deterministic."""
    _set_seed(42)
    py_a = [random.random() for _ in range(4)]
    np_a = np.random.rand(4).tolist()
    torch_cpu_a = torch.randn(4).tolist()

    _set_seed(42)
    py_b = [random.random() for _ in range(4)]
    np_b = np.random.rand(4).tolist()
    torch_cpu_b = torch.randn(4).tolist()

    assert py_a == py_b, "Python random.random() is not deterministic — random.seed not called"
    assert np_a == np_b, "np.random.rand() is not deterministic — np.random.seed not called"
    assert torch_cpu_a == torch_cpu_b, (
        "torch.randn() is not deterministic — torch.manual_seed broken"
    )


def test_set_seed_varies_with_pp_rank():
    """Different PP stages must get different seeds (for per-stage dropout variation)."""
    _set_seed(42, pp_rank=0)
    py_stage0 = [random.random() for _ in range(4)]

    _set_seed(42, pp_rank=1)
    py_stage1 = [random.random() for _ in range(4)]

    assert py_stage0 != py_stage1, "PP rank offset not applied to seed"


class _Mesh:
    """The parts of a DeviceMesh ``init_distributed`` reads, for one pipeline stage."""

    mesh_dim_names = ("pp", "dp_shard")

    def __init__(self, pp_rank: int) -> None:
        self.pp_rank = pp_rank

    def __getitem__(self, name: str) -> _Mesh:
        return self

    def get_local_rank(self) -> int:
        return self.pp_rank


def _draws_after_init(monkeypatch: pytest.MonkeyPatch, rank: int, pp_rank: int) -> list[float]:
    """Python ``random``'s first draws after ``init_distributed`` as ``rank`` of 4 on a stage."""
    from kempnerforge.config.schema import DistributedConfig
    from kempnerforge.distributed import setup

    monkeypatch.setattr(setup, "get_world_info", lambda: (rank, rank % 2, 4))
    monkeypatch.setattr(setup, "_set_nccl_env", lambda: None)
    monkeypatch.setattr(setup.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(setup.dist, "get_backend", lambda group=None: "gloo")
    monkeypatch.setattr(setup.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(setup.torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(setup, "init_device_mesh", lambda **kwargs: _Mesh(pp_rank))
    monkeypatch.setattr(setup, "_barrier_with_timeout", lambda seconds, reason: None)
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "29500")
    setup.init_distributed(DistributedConfig(pp=2), seed=42)
    return [random.random() for _ in range(4)]


def test_set_seed_same_across_dp_ranks(monkeypatch):
    """DP replicas must share the seed so dropout masks agree (within a PP stage).

    ``init_distributed`` seeds global ranks 0 and 1, the two data-parallel replicas
    of stage 0, alike, and rank 2 on stage 1 otherwise: the seed takes the stage
    and never the global rank.
    """
    assert "rank" not in inspect.signature(_set_seed).parameters
    stage0_rank0 = _draws_after_init(monkeypatch, rank=0, pp_rank=0)
    stage0_rank1 = _draws_after_init(monkeypatch, rank=1, pp_rank=0)
    stage1_rank2 = _draws_after_init(monkeypatch, rank=2, pp_rank=1)

    assert stage0_rank0 == stage0_rank1, (
        "Same PP stage across DP replicas must produce the same Python random sequence"
    )
    assert stage0_rank0 != stage1_rank2


def test_set_seed_matches_checkpoint_coverage():
    """Cold-start seeding must cover the same four RNGs the checkpoint path captures.

    The checkpoint.state module captures {python, numpy, torch_cpu, torch_cuda}.
    _set_seed should seed at least the three of those that are always present
    (torch_cuda is conditional on availability).
    """
    from kempnerforge.checkpoint.state import get_rng_state

    _set_seed(7)
    state_a = get_rng_state()

    _set_seed(7)
    state_b = get_rng_state()

    assert state_a["python"] == state_b["python"]
    np_a = state_a["numpy"]
    np_b = state_b["numpy"]
    assert np_a[0] == np_b[0]
    assert (np_a[1] == np_b[1]).all()
    assert torch.equal(state_a["torch_cpu"], state_b["torch_cpu"])
