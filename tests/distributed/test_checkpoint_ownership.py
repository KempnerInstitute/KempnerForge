"""A checkpoint directory claim succeeds or fails on every rank together.

Run with: torchrun --nproc_per_node=4 -m pytest tests/distributed/test_checkpoint_ownership.py -v
"""

from __future__ import annotations

import json
import os
import socket
import uuid
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from kempnerforge.checkpoint.manager import CheckpointManager
from kempnerforge.checkpoint.ownership import OWNER_FILE, CheckpointDirInUseError
from kempnerforge.config.schema import CheckpointConfig

pytestmark = pytest.mark.skipif(
    "RANK" not in os.environ,
    reason="Requires torchrun launcher (RANK not set)",
)


def _manager(directory) -> CheckpointManager:
    model = torch.nn.Linear(4, 4).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return CheckpointManager(CheckpointConfig(dir=str(directory)), model, optimizer)


class TestClaimOnEveryRank:
    def test_rank_zero_claims_a_free_directory_and_releases_it(
        self, distributed_env, shared_tmp_dir, monkeypatch
    ) -> None:
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        directory = Path(shared_tmp_dir) / "ckpt"
        manager = _manager(directory)
        manager.claim_dir()
        dist.barrier()
        record = json.loads((directory / OWNER_FILE).read_text())
        assert record["host"] == socket.gethostname()
        if dist.get_rank() == 0:
            assert record["pid"] == os.getpid()
        dist.barrier()
        manager.release_dir()
        dist.barrier()
        assert json.loads((directory / OWNER_FILE).read_text())["released"] is True

    def test_a_live_owner_refuses_the_claim_on_every_rank(
        self, distributed_env, shared_tmp_dir, monkeypatch
    ) -> None:
        """Rank 0 raises the refusal, and every other rank raises naming rank 0."""
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        directory = Path(shared_tmp_dir) / "taken"
        if dist.get_rank() == 0:
            directory.mkdir()
            owner = {"host": socket.gethostname(), "pid": os.getppid(), "since": "earlier"}
            record = {"claim": uuid.uuid4().hex, "slurm_job": None, **owner}
            (directory / OWNER_FILE).write_text(json.dumps(record))
        dist.barrier()
        manager = _manager(directory)
        if dist.get_rank() == 0:
            with pytest.raises(CheckpointDirInUseError, match="still running"):
                manager.claim_dir()
        else:
            failed_on_rank_zero = r"Checkpoint directory claim failed on ranks \[0\]"
            with pytest.raises(RuntimeError, match=failed_on_rank_zero):
                manager.claim_dir()
        dist.barrier()
