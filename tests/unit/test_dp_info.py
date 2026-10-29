"""Data-parallel rank and size on real DeviceMeshes, on CPU over gloo.

Each mesh has the dimension names and order ``init_distributed`` builds: ``pp``,
``dp_replicate``, ``dp_shard``, ``ep``, ``tp``, with a size-1 ``dp_shard`` inserted
when ``tp`` or ``ep`` is active.
"""

from __future__ import annotations

import socket

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh

from kempnerforge.distributed.utils import get_dp_info

HSDP_LAYOUTS = {
    "dp_replicate2_dp_shard2": (
        ("dp_replicate", "dp_shard"),
        (2, 2),
        [(0, 4), (1, 4), (2, 4), (3, 4)],
    ),
    "dp_replicate2_tp2": (
        ("dp_replicate", "dp_shard", "tp"),
        (2, 1, 2),
        [(0, 2), (0, 2), (1, 2), (1, 2)],
    ),
    "dp_replicate2_ep2": (
        ("dp_replicate", "dp_shard", "ep"),
        (2, 1, 2),
        [(0, 2), (0, 2), (1, 2), (1, 2)],
    ),
    "pp2_dp_replicate2_dp_shard2": (
        ("pp", "dp_replicate", "dp_shard"),
        (2, 2, 2),
        [(0, 4), (1, 4), (2, 4), (3, 4)] * 2,
    ),
}

OTHER_LAYOUTS = [
    (("dp_shard",), (4,), [(0, 4), (1, 4), (2, 4), (3, 4)]),
    (("dp_replicate",), (4,), [(0, 4), (1, 4), (2, 4), (3, 4)]),
    (("dp_shard", "tp"), (2, 2), [(0, 2), (0, 2), (1, 2), (1, 2)]),
    (("dp_shard", "ep", "tp"), (1, 2, 2), [(0, 1)] * 4),
    (("pp", "dp_shard"), (2, 2), [(0, 2), (1, 2), (0, 2), (1, 2)]),
    (("pp", "dp_replicate"), (2, 2), [(0, 2), (1, 2), (0, 2), (1, 2)]),
    (("pp",), (4,), [(0, 1)] * 4),
]


def _free_port() -> int:
    """An unused local TCP port for the gloo rendezvous."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _init(rank: int, world: int, port: int) -> None:
    """Join a ``world``-rank gloo group."""
    dist.init_process_group(
        "gloo", rank=rank, world_size=world, init_method=f"tcp://127.0.0.1:{port}"
    )


def _check_layouts(rank: int, world: int, port: int, layouts: list) -> None:
    """Build each ``(names, sizes, expected)`` mesh and compare this rank's DP info."""
    _init(rank, world, port)
    try:
        for names, sizes, expected in layouts:
            mesh = init_device_mesh("cpu", sizes, mesh_dim_names=names)
            got = get_dp_info(mesh)
            assert got == expected[rank], f"mesh {names} of {sizes}, rank {rank}: {got}"
    finally:
        dist.destroy_process_group()


class TestGetDpInfoOnRealMeshes:
    @pytest.mark.parametrize("layout", list(HSDP_LAYOUTS))
    def test_hsdp_meshes_give_one_dp_rank_per_data_shard(self, layout):
        """Under HSDP every (dp_replicate, dp_shard) pair reads its own data.

        The DP rank is the row-major index of the pair, so on a mesh of DP dims
        alone it equals the global rank. Ranks that differ only in ``pp``,
        ``tp`` or ``ep`` share it, and the DP size is dp_replicate x dp_shard.
        """
        world = len(HSDP_LAYOUTS[layout][2])
        mp.spawn(_check_layouts, args=(world, _free_port(), [HSDP_LAYOUTS[layout]]), nprocs=world)

    def test_meshes_without_hsdp_keep_their_dp_ranks(self):
        """One DP dim, or none under pure PP, gives the same DP info as before HSDP."""
        mp.spawn(_check_layouts, args=(4, _free_port(), OTHER_LAYOUTS), nprocs=4)
