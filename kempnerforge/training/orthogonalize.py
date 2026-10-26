"""Exact Newton-Schulz orthogonalization of sharded matrices for Muon."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

_GATHER = -1


def orthogonalize_exact(
    params: list[torch.Tensor],
    updates: list[torch.Tensor],
    newton_schulz: Callable[[torch.Tensor], torch.Tensor],
) -> list[torch.Tensor]:
    """Orthogonalize each update as its full matrix; a collective over every rank."""
    results: list[torch.Tensor | None] = [None] * len(params)
    row_sharded: dict[tuple[DeviceMesh, int, torch.dtype], list[int]] = {}
    for i, (param, update) in enumerate(zip(params, updates, strict=True)):
        dim = _row_shard_dim(param)
        if dim is None:
            results[i] = newton_schulz(update)
        elif dim == _GATHER:
            results[i] = _orthogonalize_gathered(param, update, newton_schulz)
        else:
            assert isinstance(param, DTensor)
            row_sharded.setdefault((param.device_mesh, dim, update.dtype), []).append(i)
    for (mesh, dim, _), indices in row_sharded.items():
        _orthogonalize_row_sharded(mesh, dim, indices, params, updates, results, newton_schulz)
    missing = [i for i, result in enumerate(results) if result is None]
    if missing:
        raise RuntimeError(f"no orthogonalized update for matrices {missing}")
    return cast("list[torch.Tensor]", results)


def _matrix_shape(param: torch.Tensor) -> tuple[int, int]:
    """Rows and columns of ``param`` viewed as a 2D matrix."""
    return param.shape[0], param.shape[1:].numel()


def _row_shard_dim(param: torch.Tensor) -> int | None:
    """Mesh dim splitting ``param``'s rows, ``None`` if unsplit, ``_GATHER`` for other layouts."""
    if not isinstance(param, DTensor):
        return None
    mesh = param.device_mesh
    split = [
        dim
        for dim, placement in enumerate(param.placements)
        if not isinstance(placement, Replicate) and mesh.size(dim) > 1
    ]
    if not split:
        return None
    placement = param.placements[split[0]]
    if len(split) == 1 and type(placement) is Shard and placement.dim == 0:
        return split[0]
    return _GATHER


def _row_chunks(rows: int, world: int) -> list[int]:
    """Rows each of ``world`` ranks holds under ``Shard(0)``."""
    chunk = -(-rows // world)
    sizes = []
    for _ in range(world):
        sizes.append(min(chunk, rows))
        rows -= sizes[-1]
    return sizes


def _orthogonalize_gathered(
    param: torch.Tensor,
    update: torch.Tensor,
    newton_schulz: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """All-gather a matrix, orthogonalize it on every rank, and keep this rank's part."""
    assert isinstance(param, DTensor)
    local_shape = param.to_local().shape
    full = DTensor.from_local(
        update.reshape(local_shape),
        param.device_mesh,
        param.placements,
        run_check=False,
        shape=param.shape,
        stride=param.stride(),
    ).full_tensor()
    ortho = newton_schulz(full.reshape(_matrix_shape(param))).reshape(param.shape)
    local = distribute_tensor(ortho, param.device_mesh, param.placements, src_data_rank=None)
    return local.to_local().reshape(update.shape)


def _orthogonalize_row_sharded(
    mesh: DeviceMesh,
    dim: int,
    indices: list[int],
    params: list[torch.Tensor],
    updates: list[torch.Tensor],
    results: list[torch.Tensor | None],
    newton_schulz: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    """Gather each row-sharded matrix onto one rank of its group, orthogonalize, scatter back."""
    group = mesh.get_group(dim)
    world = mesh.size(dim)
    rank = mesh.get_local_rank(dim)
    order = sorted(indices, key=lambda i: params[i].numel(), reverse=True)
    for start in range(0, len(order), world):
        batch = order[start : start + world]
        shards = [updates[i] for i in batch]
        for i, shard in zip(batch, shards, strict=True):
            expected = _row_chunks(params[i].shape[0], world)[rank]
            if shard.shape[0] != expected:
                raise RuntimeError(
                    f"rank {rank} of {world} holds {shard.shape[0]} rows of a "
                    f"{tuple(params[i].shape)} matrix, expected {expected}"
                )
        send_splits = [shard.numel() for shard in shards] + [0] * (world - len(batch))
        send = torch.cat([shard.reshape(-1) for shard in shards])
        owned = batch[rank] if rank < len(batch) else None
        if owned is None:
            recv_splits = [0] * world
        else:
            rows, cols = _matrix_shape(params[owned])
            recv_splits = [chunk * cols for chunk in _row_chunks(rows, world)]
        recv = send.new_empty(sum(recv_splits))
        dist.all_to_all_single(recv, send, recv_splits, send_splits, group=group)
        if owned is None:
            ortho = send.new_empty(0)
        else:
            ortho = newton_schulz(recv.view(rows, cols)).reshape(-1)
        back = send.new_empty(sum(send_splits))
        dist.all_to_all_single(back, ortho, send_splits, recv_splits, group=group)
        pieces = back.split(send_splits[: len(batch)])
        for i, shard, piece in zip(batch, shards, pieces, strict=True):
            results[i] = piece.view(shard.shape)
