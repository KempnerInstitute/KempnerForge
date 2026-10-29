"""Data-parallel rank and size on real DeviceMeshes, and what reads them, on CPU over gloo.

Each mesh has the dimension names and order ``init_distributed`` builds: ``pp``,
``dp_replicate``, ``dp_shard``, ``ep``, ``tp``, with a size-1 ``dp_shard`` inserted
when ``tp`` or ``ep`` is active.
"""

from __future__ import annotations

import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh

from kempnerforge.config.data import DataConfig
from kempnerforge.config.eval import EvalConfig
from kempnerforge.config.job import JobConfig
from kempnerforge.config.model import ModelConfig
from kempnerforge.config.training import TrainConfig
from kempnerforge.distributed.utils import get_dp_info
from kempnerforge.training.data_pipeline import build_data_pipeline, build_eval_dataloader
from kempnerforge.training.runtime import RuntimeContext

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


def _check_data_partition(rank: int, world: int, port: int, data_dir: str) -> None:
    """Build the training and eval loaders on a 2 x 2 HSDP mesh; each rank reads its own part."""
    _init(rank, world, port)
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp_replicate", "dp_shard"))
        runtime = RuntimeContext(
            rank=rank,
            local_rank=rank,
            world_size=world,
            device=torch.device("cpu"),
            device_mesh=mesh,
        )
        config = JobConfig(
            model=ModelConfig(dim=8, n_layers=1, n_heads=1, vocab_size=16, max_seq_len=64),
            train=TrainConfig(batch_size=2, seq_len=15, compile_model=False),
            data=DataConfig(dataset_path=data_dir, num_workers=0),
            eval=EvalConfig(enabled=True, dataset_path=data_dir),
        )
        data = build_data_pipeline(config, runtime)
        eval_loader = build_eval_dataloader(config, runtime)
        assert (data.dp_rank, data.dp_size) == (rank, world)
        mine = [list(data.dataloader.sampler), list(eval_loader.sampler)]
        everyone: list = [None] * world
        dist.all_gather_object(everyone, mine)
        for loader, samples in enumerate(mine):
            seen = [i for theirs in everyone for i in theirs[loader]]
            assert len(samples) > 0
            assert len(seen) == len(set(seen)) == world * len(samples)
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


class TestDataPipelineOnHsdp:
    def test_training_and_eval_loaders_split_the_data_over_both_dp_dims(self, mmap_data_dir):
        """With dp_replicate=2 x dp_shard=2 the four ranks read four disjoint parts.

        The loaders are built where training builds them, from ``get_dp_info``,
        so a rank that shared another's part would repeat its samples.
        """
        mp.spawn(_check_data_partition, args=(4, _free_port(), mmap_data_dir), nprocs=4)
