"""Shared distributed test fixtures.

A single session-scoped process group is created once and shared across
all distributed test modules. This avoids NCCL reinitialization issues
that occur when destroy_process_group / init_process_group are called
multiple times in the same process.
"""

from __future__ import annotations

import os

import pytest

from kempnerforge.config.schema import DistributedConfig
from kempnerforge.distributed.setup import destroy_distributed, init_distributed


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
