"""Checkpoints under pipeline parallelism, on CPU over gloo."""

from __future__ import annotations

import math
import socket
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor

from kempnerforge.config.checkpoint import AsyncCheckpointMode, CheckpointConfig
from kempnerforge.config.job import JobConfig
from kempnerforge.config.model import ModelConfig
from kempnerforge.config.training import TrainConfig
from kempnerforge.distributed.parallel import apply_fsdp2
from kempnerforge.distributed.pipeline_parallel import build_stage_module
from kempnerforge.distributed.tensor_parallel import apply_tensor_parallel
from kempnerforge.training.entry import build_checkpoint_manager, restore_checkpoint
from kempnerforge.training.runtime import PipelineBundle, RuntimeContext

MODEL = ModelConfig(dim=32, n_layers=4, n_heads=4, vocab_size=64, max_seq_len=16)
STEP = 3
TOKENS = 96


def _free_port() -> int:
    """An unused local TCP port for the gloo rendezvous."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _stage_members(rank: int, shape: tuple[int, ...], names: tuple[str, ...]) -> list[int]:
    """The ranks that share ``rank``'s ``pp`` coordinate in a row-major mesh of ``shape``."""
    pp_stride = math.prod(shape[names.index("pp") + 1 :])
    pp_size = shape[names.index("pp")]

    def stage(r: int) -> int:
        return r // pp_stride % pp_size

    return [r for r in range(math.prod(shape)) if stage(r) == stage(rank)]


def _stage(mesh: DeviceMesh, seed: int) -> torch.nn.Module:
    """This rank's pipeline stage, tensor parallel and sharded the way ``build_model`` does it."""
    torch.manual_seed(seed)
    stage = build_stage_module(MODEL, mesh.get_local_rank("pp"), mesh["pp"].size())
    if "tp" in (mesh.mesh_dim_names or ()):
        apply_tensor_parallel(stage, mesh)
    fp32 = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)
    apply_fsdp2(stage, mesh, mp_policy=fp32)
    return stage


def _stepped_optimizer(stage: torch.nn.Module) -> torch.optim.Optimizer:
    """AdamW after one step, so it holds moments to save."""
    optimizer = torch.optim.AdamW(stage.parameters(), lr=0.1)
    for p in stage.parameters():
        p.grad = torch.ones_like(p)
    optimizer.step()
    return optimizer


def _local(value: Any) -> Any:
    return value.to_local() if isinstance(value, DTensor) else value


def _assert_same(actual: Any, expected: Any) -> None:
    """Equal state, compared shard by shard, since both sides share one layout."""
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            _assert_same(actual[key], value)
    elif isinstance(expected, torch.Tensor):
        assert type(actual) is type(expected)
        assert torch.equal(_local(actual), _local(expected))
    else:
        assert actual == expected


def _check_save_and_resume(
    rank: int,
    world: int,
    port: int,
    shape: tuple[int, ...],
    names: tuple[str, ...],
    ckpt_dir: str,
    async_mode: AsyncCheckpointMode,
) -> None:
    """Save every stage through ``build_checkpoint_manager``, then resume a fresh model from it."""
    dist.init_process_group(
        "gloo", rank=rank, world_size=world, init_method=f"tcp://127.0.0.1:{port}"
    )
    try:
        mesh = init_device_mesh("cpu", shape, mesh_dim_names=names)
        pp_size = mesh["pp"].size()
        config = JobConfig(
            model=MODEL,
            train=TrainConfig(batch_size=1, seq_len=16),
            checkpoint=CheckpointConfig(dir=ckpt_dir, async_mode=async_mode),
        )
        runtime = RuntimeContext(
            rank=rank,
            local_rank=rank,
            world_size=world,
            device=torch.device("cpu"),
            device_mesh=mesh,
        )
        pipeline = PipelineBundle(rank=mesh.get_local_rank("pp"), size=pp_size, schedule=None)

        saved = _stage(mesh, seed=0)
        saved_optimizer = _stepped_optimizer(saved)
        manager = build_checkpoint_manager(config, runtime, saved, saved_optimizer, pipeline)
        manager.save(STEP, tokens_seen=TOKENS)
        manager.wait()

        step_dir = Path(ckpt_dir) / f"step_{STEP}"
        stage_dirs = sorted(d.name for d in step_dir.iterdir() if d.is_dir())
        assert stage_dirs == [f"pp{k}" for k in range(pp_size)]
        missing = [name for name in stage_dirs if not (step_dir / name / ".metadata").is_file()]
        assert not missing, f"no .metadata in {missing}"
        assert dist.get_process_group_ranks(manager._process_group) == _stage_members(
            rank, shape, names
        )

        resumed = _stage(mesh, seed=1)
        resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.1)
        assert any(
            not torch.equal(_local(a), _local(b))
            for a, b in zip(resumed.state_dict().values(), saved.state_dict().values(), strict=True)
        )
        resume_manager = build_checkpoint_manager(
            config, runtime, resumed, resumed_optimizer, pipeline
        )
        assert restore_checkpoint(config, resumed, None, resume_manager) == (STEP, TOKENS)
        _assert_same(resumed.state_dict(), saved.state_dict())
        _assert_same(resumed_optimizer.state_dict()["state"], saved_optimizer.state_dict()["state"])
    finally:
        dist.destroy_process_group()


class TestPipelineCheckpointSaveAndResume:
    @pytest.mark.parametrize(
        ("shape", "names", "async_mode"),
        [
            ((4,), ("pp",), AsyncCheckpointMode.disabled),
            ((4,), ("pp",), AsyncCheckpointMode.async_),
            ((2, 2), ("pp", "dp_shard"), AsyncCheckpointMode.disabled),
            ((2, 1, 2), ("pp", "dp_shard", "tp"), AsyncCheckpointMode.disabled),
            ((2, 2, 2), ("pp", "dp_shard", "tp"), AsyncCheckpointMode.disabled),
            ((2, 2, 2), ("pp", "dp_replicate", "dp_shard"), AsyncCheckpointMode.disabled),
        ],
        ids=["pp4", "pp4-async", "pp2-fsdp2", "pp2-tp2", "pp2-fsdp2-tp2", "pp2-hsdp"],
    )
    def test_every_stage_writes_a_checkpoint_that_resumes(self, tmp_path, shape, names, async_mode):
        """Each ``pp{k}`` directory gets its own ``.metadata`` and restores its stage exactly."""
        world = math.prod(shape)
        mp.spawn(
            _check_save_and_resume,
            args=(world, _free_port(), shape, names, str(tmp_path / "ckpt"), async_mode),
            nprocs=world,
        )
