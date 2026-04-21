"""Unit tests for distributed seed coverage.

Cold-start seeding must cover the same four RNGs captured on checkpoint
(Python random, NumPy, torch CPU, torch CUDA) so that cold-start runs
have the same reproducibility guarantees as warm-resumed runs.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from kempnerforge.distributed.setup import _set_seed


def test_set_seed_seeds_all_four_generators():
    """_set_seed must make Python random, NumPy, and torch CPU deterministic."""
    _set_seed(42, rank=0)
    py_a = [random.random() for _ in range(4)]
    np_a = np.random.rand(4).tolist()
    torch_cpu_a = torch.randn(4).tolist()

    _set_seed(42, rank=0)
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
    _set_seed(42, rank=0, pp_rank=0)
    py_stage0 = [random.random() for _ in range(4)]

    _set_seed(42, rank=0, pp_rank=1)
    py_stage1 = [random.random() for _ in range(4)]

    assert py_stage0 != py_stage1, "PP rank offset not applied to seed"


def test_set_seed_same_across_dp_ranks():
    """DP replicas must share the seed so dropout masks agree (within a PP stage)."""
    _set_seed(42, rank=0, pp_rank=0)
    py_dp0 = [random.random() for _ in range(4)]

    _set_seed(42, rank=4, pp_rank=0)  # different DP rank, same PP stage
    py_dp4 = [random.random() for _ in range(4)]

    assert py_dp0 == py_dp4, (
        "Same PP stage across DP replicas must produce the same Python random sequence"
    )


def test_set_seed_matches_checkpoint_coverage():
    """Cold-start seeding must cover the same four RNGs the checkpoint path captures.

    The checkpoint.state module captures {python, numpy, torch_cpu, torch_cuda}.
    _set_seed should seed at least the three of those that are always present
    (torch_cuda is conditional on availability).
    """
    from kempnerforge.checkpoint.state import get_rng_state

    _set_seed(7, rank=0)
    state_a = get_rng_state()

    _set_seed(7, rank=0)
    state_b = get_rng_state()

    # Python random state must be identical after reseeding.
    assert state_a["python"] == state_b["python"]
    # NumPy state must be identical (tuple compare handles array contents).
    np_a = state_a["numpy"]
    np_b = state_b["numpy"]
    assert np_a[0] == np_b[0]
    assert (np_a[1] == np_b[1]).all()
    # Torch CPU state must be identical.
    assert torch.equal(state_a["torch_cpu"], state_b["torch_cpu"])
