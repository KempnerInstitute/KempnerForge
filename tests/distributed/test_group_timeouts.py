"""Every process group a run creates times out a collective after nccl_timeout_sec.

Run with: torchrun --nproc_per_node=4 -m pytest tests/distributed/test_group_timeouts.py -v
"""

from __future__ import annotations

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from kempnerforge.checkpoint.manager import CheckpointManager
from kempnerforge.config.schema import CheckpointConfig, DistributedConfig
from kempnerforge.distributed.pipeline_parallel import build_stage_group
from kempnerforge.distributed.setup import init_distributed

pytestmark = pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 4,
    reason="Requires torchrun with 4 ranks",
)

TIMEOUT = timedelta(seconds=123)


def _timeout(group: dist.ProcessGroup, device: str) -> timedelta:
    return group._get_backend(torch.device(device)).options._timeout


class TestGroupTimeouts:
    @pytest.mark.parametrize(
        "dims",
        [{"dp_shard": 2, "tp": 2}, {"pp": 2, "dp_shard": 2}, {"dp_shard": 2, "ep": 2}],
    )
    def test_every_mesh_group_gets_nccl_timeout_sec(self, distributed_env, dims) -> None:
        world = {d: _timeout(dist.group.WORLD, d) for d in ("cpu", "cuda")}
        mesh = init_distributed(DistributedConfig(nccl_timeout_sec=123, **dims), seed=42)
        assert mesh is not None
        for name in mesh.mesh_dim_names or ():
            group = mesh.get_group(name)
            assert _timeout(group, "cuda") == TIMEOUT, name
            assert _timeout(group, "cpu") == TIMEOUT, name
        assert {d: _timeout(dist.group.WORLD, d) for d in ("cpu", "cuda")} == world

    def test_a_dimension_as_wide_as_the_world_keeps_the_default_group_timeout(
        self, distributed_env
    ) -> None:
        """The session's default group was created by its caller, so its timeout stays."""
        world = {d: _timeout(dist.group.WORLD, d) for d in ("cpu", "cuda")}
        assert world["cuda"] != TIMEOUT
        mesh = init_distributed(DistributedConfig(tp=4, nccl_timeout_sec=123), seed=42)
        assert mesh is not None and mesh.mesh_dim_names == ("dp_shard", "tp")
        assert mesh.get_group("tp") is dist.group.WORLD
        assert {d: _timeout(dist.group.WORLD, d) for d in ("cpu", "cuda")} == world
        assert _timeout(mesh.get_group("dp_shard"), "cuda") == TIMEOUT

    def test_the_pipeline_stage_group_gets_nccl_timeout_sec(self, distributed_env) -> None:
        mesh = init_distributed(DistributedConfig(pp=2, dp_shard=2), seed=42)
        assert mesh is not None
        group = build_stage_group(mesh, 123)
        assert _timeout(group, "cuda") == TIMEOUT
        assert _timeout(group, "cpu") == TIMEOUT

    def test_the_checkpoint_sync_group_gets_nccl_timeout_sec(
        self, distributed_env, shared_tmp_dir
    ) -> None:
        model = torch.nn.Linear(4, 4).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        manager = CheckpointManager(
            CheckpointConfig(dir=str(shared_tmp_dir)), model, optimizer, group_timeout_sec=123
        )
        assert manager._sync_group is not None
        assert _timeout(manager._sync_group, "cpu") == TIMEOUT
        flag = torch.ones(1)
        dist.all_reduce(flag, group=manager._sync_group)
        assert flag.item() == 4
