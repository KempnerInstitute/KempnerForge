"""Gradient clipping across pipeline stages, on CPU over gloo."""

from __future__ import annotations

import math
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from kempnerforge.distributed.utils import clip_grad_norm_

N_LAYERS = 8
DIM = 16
MAX_NORM = 1.0


def _free_port() -> int:
    """An unused local TCP port for the gloo rendezvous."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _model() -> nn.Sequential:
    """Linear layers whose gradients grow with depth, so every stage has its own norm."""
    torch.manual_seed(0)
    model = nn.Sequential(*(nn.Linear(DIM, DIM) for _ in range(N_LAYERS)))
    generator = torch.Generator().manual_seed(1)
    for depth, layer in enumerate(model):
        for p in layer.parameters():
            p.grad = torch.randn(p.shape, generator=generator) * (depth + 1) ** 2
    return model


def _stage(model: nn.Sequential, stage: int, n_stages: int) -> nn.Sequential:
    """The layers ``stage`` holds when ``n_stages`` split the model evenly."""
    per_stage = N_LAYERS // n_stages
    return model[stage * per_stage : (stage + 1) * per_stage]


def _reference(n_stages: int, no_grads: int | None) -> tuple[torch.Tensor, nn.Sequential]:
    """One process clipping the whole model, without the gradients of stage ``no_grads``."""
    model = _model()
    if no_grads is not None:
        for p in _stage(model, no_grads, n_stages).parameters():
            p.grad = None
    return clip_grad_norm_(model, MAX_NORM), model


def _model_norm(stage: nn.Module) -> float:
    """The whole model's gradient norm, summed over every stage's gradients."""
    sq = torch.tensor(0.0)
    for p in stage.parameters():
        if p.grad is not None:
            sq = sq + p.grad.pow(2).sum()
    dist.all_reduce(sq)
    return sq.sqrt().item()


def _check_stage(rank: int, world: int, port: int, with_group: bool, no_grads: int | None) -> None:
    """Clip this rank's stage and compare with one process clipping the whole model."""
    dist.init_process_group(
        "gloo", rank=rank, world_size=world, init_method=f"tcp://127.0.0.1:{port}"
    )
    try:
        ref_norm, reference = _reference(world, no_grads)
        stage = _stage(_model(), rank, world)
        if rank == no_grads:
            for p in stage.parameters():
                p.grad = None
        norm = clip_grad_norm_(stage, MAX_NORM, pp_group=dist.group.WORLD if with_group else None)
        clipped_norm = _model_norm(stage)
        if not with_group:
            assert norm < ref_norm
            assert math.isclose(clipped_norm, math.sqrt(world) * MAX_NORM, rel_tol=1e-4)
            return
        torch.testing.assert_close(norm, ref_norm, rtol=1e-5, atol=0.0)
        assert math.isclose(clipped_norm, MAX_NORM, rel_tol=1e-4)
        expected = _stage(reference, rank, world).parameters()
        for p, q in zip(stage.parameters(), expected, strict=True):
            if q.grad is None:
                assert p.grad is None
            else:
                torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-7)
    finally:
        dist.destroy_process_group()


def _check_sharded_stage(rank: int, world: int, port: int) -> None:
    """pp=2 x dp_shard=2: each stage's parameters are row-sharded DTensors, as under FSDP2."""
    dist.init_process_group(
        "gloo", rank=rank, world_size=world, init_method=f"tcp://127.0.0.1:{port}"
    )
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("pp", "dp_shard"))
        pp_rank = mesh.get_local_rank("pp")
        ref_norm, reference = _reference(2, None)
        stage = _stage(_model(), pp_rank, 2)
        for layer in stage:
            for name, p in list(layer.named_parameters()):
                sharded = nn.Parameter(
                    distribute_tensor(p.detach().clone(), mesh["dp_shard"], [Shard(0)])
                )
                sharded.grad = distribute_tensor(p.grad.clone(), mesh["dp_shard"], [Shard(0)])
                setattr(layer, name, sharded)
        norm = clip_grad_norm_(stage, MAX_NORM, pp_group=mesh["pp"].get_group())
        torch.testing.assert_close(norm, ref_norm, rtol=1e-5, atol=0.0)
        expected = _stage(reference, pp_rank, 2).parameters()
        for p, q in zip(stage.parameters(), expected, strict=True):
            torch.testing.assert_close(p.grad.full_tensor(), q.grad, rtol=1e-5, atol=1e-7)
    finally:
        dist.destroy_process_group()


class TestClipAcrossPipelineStages:
    @pytest.mark.parametrize("world", [2, 4])
    def test_every_stage_clips_with_the_whole_model_norm(self, world):
        """Each stage returns the model's norm, and the clipped model norm is ``max_norm``."""
        mp.spawn(_check_stage, args=(world, _free_port(), True, None), nprocs=world)

    def test_without_the_group_each_stage_clips_alone(self):
        """Alone, each stage reaches ``max_norm``, so the model ends at sqrt(pp) times it."""
        mp.spawn(_check_stage, args=(2, _free_port(), False, None), nprocs=2)

    def test_a_stage_without_gradients_still_joins(self):
        """A stage with no gradients takes part in the sum instead of hanging the others."""
        mp.spawn(_check_stage, args=(2, _free_port(), True, 1), nprocs=2)

    def test_one_stage_matches_the_plain_path(self):
        mp.spawn(_check_stage, args=(1, _free_port(), True, None), nprocs=1)

    def test_row_sharded_stages_match_one_process(self):
        """With pp=2 x dp_shard=2 the norm is reduced over the shards and over the stages."""
        mp.spawn(_check_sharded_stage, args=(4, _free_port()), nprocs=4)
