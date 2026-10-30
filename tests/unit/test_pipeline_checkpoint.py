"""Checkpoint saves across gloo ranks on CPU: under pipeline parallelism, and when a rank fails."""

from __future__ import annotations

import math
import socket
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from torch.distributed.checkpoint.filesystem import FileSystemWriter
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor

from kempnerforge.checkpoint import manager as manager_module
from kempnerforge.config.checkpoint import AsyncCheckpointMode, CheckpointConfig
from kempnerforge.config.job import JobConfig
from kempnerforge.config.model import ModelConfig
from kempnerforge.config.training import TrainConfig
from kempnerforge.distributed.parallel import apply_fsdp2
from kempnerforge.distributed.pipeline_parallel import build_stage_module
from kempnerforge.distributed.tensor_parallel import apply_tensor_parallel
from kempnerforge.model.transformer import Transformer
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
    """This rank's pipeline stage (the model, without ``pp``), parallelized like ``build_model``."""
    torch.manual_seed(seed)
    names = mesh.mesh_dim_names or ()
    if "pp" in names:
        stage = build_stage_module(MODEL, mesh.get_local_rank("pp"), mesh["pp"].size())
    else:
        stage = Transformer(MODEL)
    if "tp" in names:
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


def _job(
    rank: int, world: int, mesh: DeviceMesh, ckpt_dir: str, async_mode: AsyncCheckpointMode
) -> tuple[JobConfig, RuntimeContext, PipelineBundle | None]:
    """The config, runtime and pipeline ``build_checkpoint_manager`` takes on this rank."""
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
    if "pp" not in (mesh.mesh_dim_names or ()):
        return config, runtime, None
    pipeline = PipelineBundle(rank=mesh.get_local_rank("pp"), size=mesh["pp"].size(), schedule=None)
    return config, runtime, pipeline


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
        config, runtime, pipeline = _job(rank, world, mesh, ckpt_dir, async_mode)

        saved = _stage(mesh, seed=0)
        saved_optimizer = _stepped_optimizer(saved)
        manager = build_checkpoint_manager(config, runtime, saved, saved_optimizer, pipeline)
        manager.save(STEP, tokens_seen=TOKENS)
        manager.wait()

        step_dir = Path(ckpt_dir) / f"step_{STEP}"
        stage_dirs = sorted(d.name for d in step_dir.iterdir() if d.is_dir())
        assert stage_dirs == [f"pp{k}" for k in range(pp_size)], f"stage dirs: {stage_dirs}"
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


class _LateWrite:
    """A blocking checkpoint write that starts ``delay`` seconds late, as on a slow stage."""

    def __init__(self, delay: float) -> None:
        self.delay = delay

    def save(self, state_dict: dict, checkpoint_id: str, process_group: Any = None) -> None:
        time.sleep(self.delay)
        dcp.save(state_dict, checkpoint_id=checkpoint_id, process_group=process_group)

    def wait(self) -> None:
        return None


class _LateFlush:
    """An async checkpoint write whose background flush starts ``delay`` seconds late."""

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future: Future | None = None

    def save(self, state_dict: dict, checkpoint_id: str, process_group: Any = None) -> None:
        self.wait()
        self.future = self.pool.submit(self._write, state_dict, checkpoint_id, process_group)

    def _write(self, state_dict: dict, checkpoint_id: str, process_group: Any) -> None:
        time.sleep(self.delay)
        dcp.save(state_dict, checkpoint_id=checkpoint_id, process_group=process_group)

    def wait(self) -> None:
        if self.future is not None:
            self.future.result()
            self.future = None


def _check_latest_waits_for_every_stage(
    rank: int, world: int, port: int, ckpt_dir: str, async_mode: AsyncCheckpointMode
) -> None:
    """The last stage writes late; rank 0 commits ``latest`` only once every stage has saved."""
    dist.init_process_group(
        "gloo", rank=rank, world_size=world, init_method=f"tcp://127.0.0.1:{port}"
    )
    try:
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("pp",))
        config, runtime, pipeline = _job(rank, world, mesh, ckpt_dir, async_mode)
        stage = _stage(mesh, seed=0)
        manager = build_checkpoint_manager(
            config, runtime, stage, _stepped_optimizer(stage), pipeline
        )
        if rank == world - 1:
            sync = async_mode == AsyncCheckpointMode.disabled
            manager._async_ckpt = _LateWrite(2.0) if sync else _LateFlush(2.0)
        complete_at_commit: dict[int, list[str]] = {}
        commit = manager._commit_latest

        def recording_commit(step: int, step_dir: Path) -> None:
            complete_at_commit[step] = sorted(
                d.name for d in step_dir.iterdir() if (d / ".metadata").is_file()
            )
            commit(step, step_dir)

        manager._commit_latest = recording_commit
        manager.save(STEP, tokens_seen=TOKENS)
        manager.wait()
        if rank == 0:
            expected = {STEP: [f"pp{k}" for k in range(world)]}
            assert complete_at_commit == expected, f"stages saved at commit: {complete_at_commit}"
            assert (Path(ckpt_dir) / "latest").resolve().name == f"step_{STEP}"
    finally:
        dist.destroy_process_group()


def _no_space_left(*args: Any, **kwargs: Any) -> Any:
    raise OSError(28, "No space left on device")


def _fail_part(rank: int, world: int, failing: str) -> None:
    """Make one rank's part of the save fail, as named by ``failing``.

    ``"shards"`` is the DCP write of the last rank, ``"metadata"`` the ``metadata.json`` that
    rank 0 alone writes.
    """
    if failing == "shards" and rank == world - 1:
        FileSystemWriter.write_data = _no_space_left
    if failing == "metadata" and rank == 0:
        replace_file = manager_module._replace_file

        def replace_all_but_metadata(path: Path, text: str) -> None:
            if path.name == "metadata.json":
                _no_space_left()
            replace_file(path, text)

        manager_module._replace_file = replace_all_but_metadata


def _check_a_failed_save(
    rank: int,
    world: int,
    port: int,
    shape: tuple[int, ...],
    names: tuple[str, ...],
    ckpt_dir: str,
    async_mode: AsyncCheckpointMode,
    failing: str,
    out_dir: str,
) -> None:
    """One rank's part of the save fails; write how ``save`` and ``wait`` end on this rank."""
    dist.init_process_group(
        "gloo", rank=rank, world_size=world, init_method=f"tcp://127.0.0.1:{port}"
    )
    try:
        mesh = init_device_mesh("cpu", shape, mesh_dim_names=names)
        config, runtime, pipeline = _job(rank, world, mesh, ckpt_dir, async_mode)
        stage = _stage(mesh, seed=0)
        manager = build_checkpoint_manager(
            config, runtime, stage, _stepped_optimizer(stage), pipeline
        )
        _fail_part(rank, world, failing)
        try:
            manager.save(STEP, tokens_seen=TOKENS)
            manager.wait()
            outcome = "returned"
        except BaseException as e:
            outcome = f"{type(e).__name__}: {e}"
        Path(out_dir, f"rank{rank}").write_text(outcome)
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


class TestLatestWaitsForEveryStage:
    @pytest.mark.parametrize(
        "async_mode", [AsyncCheckpointMode.disabled, AsyncCheckpointMode.async_], ids=str
    )
    def test_latest_names_a_step_only_after_every_stage_saved_it(self, tmp_path, async_mode):
        """Committing (and pruning) on rank 0 waits for the slowest stage's ``.metadata``."""
        mp.spawn(
            _check_latest_waits_for_every_stage,
            args=(2, _free_port(), str(tmp_path / "ckpt"), async_mode),
            nprocs=2,
        )


class TestAFailedSaveFailsEveryRank:
    @pytest.mark.parametrize(
        ("shape", "names", "async_mode", "failing", "failed"),
        [
            ((2,), ("pp",), AsyncCheckpointMode.disabled, "shards", [1]),
            ((2,), ("pp",), AsyncCheckpointMode.async_, "shards", [1]),
            ((2, 2), ("pp", "dp_shard"), AsyncCheckpointMode.disabled, "shards", [2, 3]),
            ((2, 2), ("pp", "dp_shard"), AsyncCheckpointMode.async_, "shards", [2, 3]),
            ((2,), ("pp",), AsyncCheckpointMode.disabled, "metadata", [0]),
            ((2,), ("dp_shard",), AsyncCheckpointMode.disabled, "metadata", [0]),
            ((2,), ("dp_shard",), AsyncCheckpointMode.async_, "metadata", [0]),
            ((2,), ("dp_shard",), AsyncCheckpointMode.async_, "shards", [0, 1]),
        ],
        ids=[
            "pp2-shards",
            "pp2-async-shards",
            "pp2-fsdp2-shards",
            "pp2-fsdp2-async-shards",
            "pp2-metadata",
            "fsdp2-metadata",
            "fsdp2-async-metadata",
            "fsdp2-async-shards",
        ],
    )
    def test_every_rank_raises_and_latest_stays_unset(
        self, tmp_path, shape, names, async_mode, failing, failed
    ):
        """No rank is left waiting for one whose part failed, and none moves ``latest``.

        The ranks in ``failed`` raise the failure itself: the rank whose part failed, and the
        other ranks DCP saves with. Every other rank raises a ``RuntimeError`` naming them.
        """
        world = math.prod(shape)
        out_dir = tmp_path / "outcomes"
        out_dir.mkdir()
        context = mp.spawn(
            _check_a_failed_save,
            args=(
                world,
                _free_port(),
                shape,
                names,
                str(tmp_path / "ckpt"),
                async_mode,
                failing,
                str(out_dir),
            ),
            nprocs=world,
            join=False,
        )
        deadline = time.monotonic() + 45
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                for process in context.processes:
                    process.terminate()
                waiting = sorted(set(range(world)) - {int(p.name[4:]) for p in out_dir.iterdir()})
                pytest.fail(f"ranks {waiting} still in save() or wait(); ranks {failed} failed")
        outcomes = {int(p.name[4:]): p.read_text() for p in out_dir.iterdir()}
        assert sorted(outcomes) == list(range(world))
        for rank, outcome in outcomes.items():
            if rank in failed:
                assert "No space left on device" in outcome, f"rank {rank}: {outcome}"
            else:
                expected = f"RuntimeError: Checkpoint save failed on ranks {failed}"
                assert outcome == expected, f"rank {rank}: {outcome}"
        assert not (tmp_path / "ckpt" / "latest").exists()
