"""Shared distributed test fixtures.

A single session-scoped process group is created once and shared across
all distributed test modules. This avoids NCCL reinitialization issues
that occur when destroy_process_group / init_process_group are called
multiple times in the same process.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch.distributed as dist

from kempnerforge.config.schema import DistributedConfig
from kempnerforge.distributed.setup import destroy_distributed, init_distributed

# Tests that use shared_tmp_dir write under <repo>/.test_tmp/ instead of /tmp
# so multi-node srun runs see the same directory on every rank. /tmp is
# node-local and breaks DCP save/load across nodes when each rank writes its
# shard to its own /tmp.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TEST_TMP = _PROJECT_ROOT / ".test_tmp"


@pytest.fixture(autouse=True, scope="session")
def _distributed_init():
    """Initialize distributed once for the entire test session."""
    if "RANK" not in os.environ:
        yield None
        return

    config = DistributedConfig()
    mesh = init_distributed(config, seed=42)
    yield mesh
    destroy_distributed()


@pytest.fixture
def distributed_env(_distributed_init):
    """Per-test accessor for the shared DeviceMesh."""
    return _distributed_init


@pytest.fixture
def shared_tmp_dir():
    """Create a tmp directory on the shared filesystem visible to all ranks.

    Rank 0 creates the directory under ``<repo>/.test_tmp/`` and broadcasts
    the path; all other ranks read the same path. The directory is removed
    after the test. Use this instead of ``tmp_path_factory`` for any
    distributed test that writes files one rank can read on another (DCP
    save/load, torch.save+torch.load, etc.); ``/tmp`` is node-local and
    such tests will fail under multi-node srun even though they pass under
    single-node torchrun.
    """
    rank = dist.get_rank()
    if rank == 0:
        _TEST_TMP.mkdir(exist_ok=True)
        d = tempfile.mkdtemp(dir=_TEST_TMP)
    else:
        d = ""
    obj_list: list[object] = [d]
    dist.broadcast_object_list(obj_list, src=0)
    d = obj_list[0]  # type: ignore[assignment]
    yield d
    dist.barrier()
    if rank == 0:
        shutil.rmtree(d, ignore_errors=True)
