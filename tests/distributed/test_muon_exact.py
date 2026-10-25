"""Distributed tests for exact Muon orthogonalization."""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from kempnerforge.training.optimizer import Muon

pytestmark = pytest.mark.skipif(
    "RANK" not in os.environ,
    reason="Requires torchrun launcher (RANK not set)",
)


class TestMuonExactDistributed:
    def test_exact_step_matches_one_process(self, distributed_env):
        """A row-sharded exact-mode Muon step equals the unsharded step."""
        world = dist.get_world_size()
        mesh = init_device_mesh("cuda", (world,))
        torch.manual_seed(0)
        full = torch.randn(4 * world + 3, 8, device="cuda")
        grads = [torch.randn_like(full) for _ in range(2)]
        reference = nn.Parameter(full.clone())
        sharded = nn.Parameter(distribute_tensor(full.clone(), mesh, [Shard(0)]))
        ref_opt = Muon([{"params": [reference]}], [{"params": []}], lr=0.1)
        opt = Muon([{"params": [sharded]}], [{"params": []}], lr=0.1, orthogonalize="exact")
        for grad in grads:
            reference.grad = grad.clone()
            sharded.grad = distribute_tensor(grad.clone(), mesh, [Shard(0)])
            ref_opt.step()
            opt.step()
        assert torch.equal(sharded.full_tensor(), reference.data)

    def test_local_step_differs_from_one_process(self, distributed_env):
        """Without exact mode, each shard is orthogonalized on its own."""
        world = dist.get_world_size()
        if world < 2:
            pytest.skip("needs at least two ranks to shard rows")
        mesh = init_device_mesh("cuda", (world,))
        torch.manual_seed(0)
        full = torch.randn(4 * world, 8, device="cuda")
        grad = torch.randn_like(full)
        reference = nn.Parameter(full.clone())
        sharded = nn.Parameter(distribute_tensor(full.clone(), mesh, [Shard(0)]))
        ref_opt = Muon([{"params": [reference]}], [{"params": []}], lr=0.1)
        opt = Muon([{"params": [sharded]}], [{"params": []}], lr=0.1)
        reference.grad = grad.clone()
        sharded.grad = distribute_tensor(grad.clone(), mesh, [Shard(0)])
        ref_opt.step()
        opt.step()
        assert not torch.allclose(sharded.full_tensor(), reference.data)
