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
from torch.distributed.tensor import DTensor, Shard, distribute_tensor
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

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


def _reference(
    n_stages: int, no_grads: int | None, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, nn.Sequential]:
    """One process clipping the whole model, without the gradients of stage ``no_grads``.

    The gradients are rounded to ``dtype`` and clipped in fp32.
    """
    model = _model().to(dtype).float()
    if no_grads is not None:
        for p in _stage(model, no_grads, n_stages).parameters():
            p.grad = None
    return clip_grad_norm_(model, MAX_NORM), model


def _model_norm(stage: nn.Module) -> float:
    """The whole model's gradient norm, summed over every stage's gradients."""
    sq = torch.tensor(0.0)
    for p in stage.parameters():
        if p.grad is not None:
            sq = sq + p.grad.float().pow(2).sum()
    dist.all_reduce(sq)
    return sq.sqrt().item()


def _assert_grads_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """A clipped gradient against the fp32 reference, to the precision of its own dtype."""
    if actual.dtype == torch.float32:
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
    else:
        torch.testing.assert_close(actual, expected.to(actual.dtype))


def _check_stage(
    rank: int,
    world: int,
    port: int,
    with_group: bool,
    no_grads: int | None,
    dtype: torch.dtype = torch.float32,
) -> None:
    """Clip this rank's stage and compare with one process clipping the whole model."""
    dist.init_process_group(
        "gloo", rank=rank, world_size=world, init_method=f"tcp://127.0.0.1:{port}"
    )
    try:
        ref_norm, reference = _reference(world, no_grads, dtype)
        stage = _stage(_model().to(dtype), rank, world)
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
        assert math.isclose(clipped_norm, MAX_NORM, rel_tol=max(1e-4, torch.finfo(dtype).eps))
        expected = _stage(reference, rank, world).parameters()
        for p, q in zip(stage.parameters(), expected, strict=True):
            if q.grad is None:
                assert p.grad is None
            else:
                _assert_grads_close(p.grad, q.grad)
    finally:
        dist.destroy_process_group()


def _check_sharded_stage(
    rank: int, world: int, port: int, dtype: torch.dtype = torch.float32
) -> None:
    """pp=2 x dp_shard=2: each stage's parameters are row-sharded DTensors, as under FSDP2."""
    dist.init_process_group(
        "gloo", rank=rank, world_size=world, init_method=f"tcp://127.0.0.1:{port}"
    )
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("pp", "dp_shard"))
        pp_rank = mesh.get_local_rank("pp")
        ref_norm, reference = _reference(2, None, dtype)
        stage = _stage(_model().to(dtype), pp_rank, 2)
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
            _assert_grads_close(p.grad.full_tensor(), q.grad)
    finally:
        dist.destroy_process_group()


def _check_tp_stage(rank: int, world: int, port: int, pp: int, dtype: torch.dtype) -> None:
    """pp x tp=2: each stage's last two layers are Colwise and Rowwise on the tp mesh.

    The stage's other layers stay plain tensors, as norms and embeddings do under
    TP, so its gradients live on two meshes. With ``pp=1`` no group is passed.
    """
    dist.init_process_group(
        "gloo", rank=rank, world_size=world, init_method=f"tcp://127.0.0.1:{port}"
    )
    try:
        mesh = init_device_mesh("cpu", (pp, 2), mesh_dim_names=("pp", "tp"))
        pp_rank = mesh.get_local_rank("pp")
        ref_norm, reference = _reference(pp, None, dtype)
        stage = _stage(_model().to(dtype), pp_rank, pp)
        grads = [p.grad for p in stage.parameters()]
        *_, colwise, rowwise = (name for name, _ in stage.named_children())
        parallelize_module(
            stage, mesh["tp"], {colwise: ColwiseParallel(), rowwise: RowwiseParallel()}
        )
        for p, g in zip(stage.parameters(), grads, strict=True):
            if isinstance(p, DTensor):
                p.grad = distribute_tensor(g, p.device_mesh, p.placements)
        group = mesh["pp"].get_group() if pp > 1 else None
        norm = clip_grad_norm_(stage, MAX_NORM, pp_group=group)
        torch.testing.assert_close(norm, ref_norm, rtol=1e-5, atol=0.0)
        expected = _stage(reference, pp_rank, pp).parameters()
        for p, q in zip(stage.parameters(), expected, strict=True):
            grad = p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
            _assert_grads_close(grad, q.grad)
    finally:
        dist.destroy_process_group()


def _check_fp64_stage(rank: int, world: int, port: int) -> None:
    """fp64 gradients keep an fp64 norm, since ``vector_norm`` cannot narrow them to fp32."""
    dist.init_process_group(
        "gloo", rank=rank, world_size=world, init_method=f"tcp://127.0.0.1:{port}"
    )
    try:
        stage = _stage(_model().double(), rank, world)
        norm_sq = torch.stack([p.grad.pow(2).sum() for p in stage.parameters()]).sum()
        dist.all_reduce(norm_sq)
        norm = clip_grad_norm_(stage, MAX_NORM, pp_group=dist.group.WORLD)
        assert norm.dtype == torch.float64
        torch.testing.assert_close(norm, norm_sq.sqrt())
    finally:
        dist.destroy_process_group()


def _check_pipeline_step(rank: int, world: int, port: int) -> None:
    """A real 1F1B step with FSDP2 on pp=2 x dp_shard=2, clipped, against one process."""
    import torch.nn.functional as F
    from torch.distributed.fsdp import MixedPrecisionPolicy

    from kempnerforge.config.schema import ModelConfig
    from kempnerforge.distributed.parallel import apply_fsdp2
    from kempnerforge.distributed.pipeline_parallel import (
        build_pipeline_schedule,
        build_pipeline_stage,
        build_stage_module,
    )
    from kempnerforge.model.transformer import Transformer

    def loss_fn(out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(out.flatten(0, 1).float(), target.flatten())

    n_micro, per_dp_batch, seq = 2, 4, 32
    dist.init_process_group(
        "gloo", rank=rank, world_size=world, init_method=f"tcp://127.0.0.1:{port}"
    )
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("pp", "dp_shard"))
        pp_rank, dp_rank = mesh.get_local_rank("pp"), mesh.get_local_rank("dp_shard")
        config = ModelConfig(dim=64, n_layers=4, n_heads=4, vocab_size=128, max_seq_len=seq)
        torch.manual_seed(0)
        reference = Transformer(config)
        stage_module = build_stage_module(config, pp_rank=pp_rank, pp_size=2)
        own = stage_module.state_dict()
        subset = {k: v for k, v in reference.state_dict().items() if k in own}
        stage_module.load_state_dict(subset, strict=True)
        fp32 = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)
        apply_fsdp2(stage_module, mesh, mp_policy=fp32)
        stage = build_pipeline_stage(
            stage_module,
            mesh,
            torch.device("cpu"),
            batch_size=per_dp_batch // n_micro,
            seq_len=seq,
            param_dtype=torch.float32,
        )
        schedule = build_pipeline_schedule(
            stage, n_microbatches=n_micro, loss_fn=loss_fn, schedule="1f1b"
        )

        generator = torch.Generator().manual_seed(7)
        tokens = torch.randint(0, config.vocab_size, (2 * per_dp_batch, seq), generator=generator)
        labels = torch.randint(0, config.vocab_size, (2 * per_dp_batch, seq), generator=generator)
        rows = slice(dp_rank * per_dp_batch, (dp_rank + 1) * per_dp_batch)
        if pp_rank == 0:
            schedule.step(tokens[rows], target=labels[rows], losses=[])
        else:
            schedule.step(target=labels[rows], losses=[])
        norm = clip_grad_norm_(stage_module, 0.05, pp_group=mesh["pp"].get_group())

        loss_fn(reference(tokens), labels).backward()
        ref_norm = clip_grad_norm_(reference, 0.05)
        torch.testing.assert_close(norm, ref_norm, rtol=2e-4, atol=0.0)
        for name, p in stage_module.named_parameters():
            torch.testing.assert_close(
                p.grad.full_tensor(), reference.get_parameter(name).grad, rtol=2e-4, atol=2e-5
            )
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

    def test_pipeline_step_with_fsdp2_matches_one_process(self):
        """Gradients from a real pipeline step with FSDP2 clip exactly as one process's do."""
        mp.spawn(_check_pipeline_step, args=(4, _free_port()), nprocs=4)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    def test_half_precision_stages_clip_with_the_fp32_norm(self, dtype):
        """Stage norms far above 256 neither overflow in fp16 nor round in bf16."""
        mp.spawn(_check_stage, args=(2, _free_port(), True, None, dtype), nprocs=2)

    def test_half_precision_row_sharded_stages_clip_with_the_fp32_norm(self):
        """fp16 DTensor gradients reduce their partial norms in fp32 as well."""
        mp.spawn(_check_sharded_stage, args=(4, _free_port(), torch.float16), nprocs=4)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
    @pytest.mark.parametrize("pp", [1, 2])
    def test_tp_sharded_stages_match_one_process(self, pp, dtype):
        """Colwise, Rowwise and plain gradients are reduced over tp, then over the stages.

        With ``pp=1`` there is no group, and the two meshes take the mixed-mesh path.
        """
        mp.spawn(_check_tp_stage, args=(2 * pp, _free_port(), pp, dtype), nprocs=2 * pp)

    def test_fp64_stages_keep_an_fp64_norm(self):
        """fp64 gradients clip with an fp64 norm instead of failing to narrow to fp32."""
        mp.spawn(_check_fp64_stage, args=(2, _free_port()), nprocs=2)
