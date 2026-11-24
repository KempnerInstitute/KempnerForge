"""Unit tests for exact Muon orthogonalization."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from kempnerforge.training.optimizer import Muon, _newton_schulz
from kempnerforge.training.orthogonalize import _row_chunks, orthogonalize_exact
from tests.gloo_ranks import gloo_group, spawn


def _ns(x: torch.Tensor) -> torch.Tensor:
    """Newton-Schulz as Muon runs it: fp32, returned in the input's dtype."""
    return _newton_schulz(x.float(), steps=5).to(x.dtype)


def _matrices() -> list[torch.Tensor]:
    """Matrices with even, uneven and fewer-rows-than-ranks row counts."""
    generator = torch.Generator().manual_seed(0)
    shapes = ((10, 6), (7, 12), (16, 4), (3, 5))
    return [torch.randn(rows, cols, generator=generator) for rows, cols in shapes]


def _check_rank(rank: int, world: int, port: int, mesh_shape: tuple, placements: tuple) -> None:
    """Orthogonalize sharded matrices on one rank and compare with one process's result."""
    with gloo_group(rank, world, port):
        mesh = init_device_mesh("cpu", mesh_shape)
        fulls = _matrices()
        params = [distribute_tensor(full, mesh, list(placements)) for full in fulls]
        results = orthogonalize_exact(params, [p.to_local() for p in params], _ns)
        for full, result in zip(fulls, results, strict=True):
            expected = distribute_tensor(_ns(full), mesh, list(placements), src_data_rank=None)
            assert torch.equal(result, expected.to_local())


def _own_rows(full: torch.Tensor, mesh: DeviceMesh) -> torch.Tensor:
    """This rank's rows of ``full`` under ``Shard(0)``."""
    return distribute_tensor(full, mesh, [Shard(0)], src_data_rank=None).to_local().clone()


def _check_rowless_rank(rank: int, world: int, port: int, orthogonalize: str) -> None:
    """Step Muon on a weight with fewer rows than ranks beside a larger one and compare with
    unsharded Muon. The ranks past its rows hold none of it, and step with the others."""
    with gloo_group(rank, world, port):
        mesh = init_device_mesh("cpu", (world,))
        generator = torch.Generator().manual_seed(0)
        fulls = [torch.randn(rows, 8, generator=generator) for rows in (2 * world + 1, 1)]
        steps = [[torch.randn(full.shape, generator=generator) for full in fulls] for _ in range(2)]
        exact = orthogonalize == "exact"
        sharded = [
            nn.Parameter(distribute_tensor(full.clone(), mesh, [Shard(0)], src_data_rank=None))
            for full in fulls
        ]
        references = [
            nn.Parameter(full.clone() if exact else _own_rows(full, mesh)) for full in fulls
        ]
        optimizer = Muon(
            [{"params": sharded}], [{"params": []}], lr=0.1, orthogonalize=orthogonalize
        )
        ref_optimizer = Muon([{"params": references}], [{"params": []}], lr=0.1)
        assert sharded[1].to_local().shape == (1 if rank == 0 else 0, 8)
        for grads in steps:
            for param, reference, grad in zip(sharded, references, grads, strict=True):
                param.grad = distribute_tensor(grad.clone(), mesh, [Shard(0)], src_data_rank=None)
                reference.grad = grad.clone() if exact else _own_rows(grad, mesh)
            optimizer.step()
            ref_optimizer.step()
        for param, reference in zip(sharded, references, strict=True):
            expected = _own_rows(reference.data, mesh) if exact else reference.data
            assert torch.equal(param.to_local(), expected)


def _check_muon_rank(rank: int, world: int, port: int, orthogonalize: str, options: dict) -> None:
    """Step Muon on a row-sharded weight and compare with unsharded Muon."""
    with gloo_group(rank, world, port):
        mesh = init_device_mesh("cpu", (world,))
        generator = torch.Generator().manual_seed(0)
        full = torch.randn(4 * world + 3, 8, generator=generator)
        grads = [torch.randn(full.shape, generator=generator) for _ in range(2)]
        exact = orthogonalize == "exact"
        sharded = nn.Parameter(
            distribute_tensor(full.clone(), mesh, [Shard(0)], src_data_rank=None)
        )
        reference = nn.Parameter(full.clone() if exact else _own_rows(full, mesh))
        optimizer = Muon(
            [{"params": [sharded]}],
            [{"params": []}],
            lr=0.1,
            orthogonalize=orthogonalize,
            **options,
        )
        ref_optimizer = Muon([{"params": [reference]}], [{"params": []}], lr=0.1, **options)
        for grad in grads:
            sharded.grad = distribute_tensor(grad.clone(), mesh, [Shard(0)], src_data_rank=None)
            reference.grad = grad.clone() if exact else _own_rows(grad, mesh)
            optimizer.step()
            ref_optimizer.step()
        expected = _own_rows(reference.data, mesh) if exact else reference.data
        assert torch.equal(sharded.to_local(), expected)


class TestRowChunks:
    @pytest.mark.parametrize(("rows", "world"), [(10, 4), (3, 4), (8, 2), (7, 3)])
    def test_matches_torch_chunk(self, rows, world):
        sizes = [chunk.shape[0] for chunk in torch.chunk(torch.empty(rows), world)]
        assert _row_chunks(rows, world) == sizes + [0] * (world - len(sizes))


class TestOrthogonalizeExact:
    def test_unsharded_matrices_are_orthogonalized_whole(self):
        fulls = _matrices()
        for full, result in zip(fulls, orthogonalize_exact(fulls, fulls, _ns), strict=True):
            assert torch.equal(result, _ns(full))

    @pytest.mark.parametrize(
        ("world", "mesh_shape", "placements"),
        [
            (2, (2,), (Shard(0),)),
            (4, (4,), (Shard(0),)),
            (4, (2, 2), (Replicate(), Shard(0))),
            (2, (2,), (Shard(1),)),
        ],
    )
    def test_sharded_matrices_match_one_process(self, world, mesh_shape, placements):
        spawn(_check_rank, world, mesh_shape, placements)


class TestMuonOnRowShards:
    @pytest.mark.parametrize("orthogonalize", ["local", "exact"])
    def test_a_rank_without_rows_of_a_weight_steps_with_the_others(self, orthogonalize):
        """Every rank finishes the step, and exact still steps the rows of the full matrix."""
        spawn(_check_rowless_rank, 2, orthogonalize)

    @pytest.mark.parametrize("options", [{}, {"nesterov": True, "adjust_lr": "match_rms_adamw"}])
    @pytest.mark.parametrize("orthogonalize", ["local", "exact"])
    def test_step_matches_unsharded_muon(self, orthogonalize, options):
        """Exact steps a shard as rows of the full matrix; local steps it as its own matrix."""
        spawn(_check_muon_rank, 2, orthogonalize, options)
