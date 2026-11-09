"""Data-parallel rank and size on real DeviceMeshes, on CPU over gloo.

Also the data loaders and an HSDP training step that use them, and the reduction of
the logged losses and token counts over the data-parallel ranks. Each mesh has the
dimension names and order ``init_distributed`` builds: ``pp``, ``dp_replicate``,
``dp_shard``, ``ep``, ``tp``, with a size-1 ``dp_shard`` inserted when ``tp`` or
``ep`` is active.
"""

from __future__ import annotations

import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor, Replicate, Shard

from kempnerforge.config.data import DataConfig
from kempnerforge.config.eval import EvalConfig
from kempnerforge.config.job import JobConfig
from kempnerforge.config.model import ModelConfig
from kempnerforge.config.training import TrainConfig
from kempnerforge.distributed.parallel import apply_fsdp2
from kempnerforge.distributed.utils import (
    clip_grad_norm_,
    data_parallel_groups,
    dp_all_reduce_,
    dp_mean,
    get_dp_info,
)
from kempnerforge.model.transformer import Transformer
from kempnerforge.training.data_pipeline import (
    DataPipeline,
    build_data_pipeline,
    build_eval_dataloader,
)
from kempnerforge.training.runtime import RuntimeContext

HSDP_LAYOUTS = {
    "dp_replicate2_dp_shard2": (
        ("dp_replicate", "dp_shard"),
        (2, 2),
        [(0, 4), (1, 4), (2, 4), (3, 4)],
    ),
    "dp_replicate2_dp_shard4": (
        ("dp_replicate", "dp_shard"),
        (2, 4),
        [(rank, 8) for rank in range(8)],
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

OTHER_LAYOUTS = {
    "dp_shard4": (("dp_shard",), (4,), [(0, 4), (1, 4), (2, 4), (3, 4)]),
    "dp_replicate4": (("dp_replicate",), (4,), [(0, 4), (1, 4), (2, 4), (3, 4)]),
    "dp_shard2_tp2": (("dp_shard", "tp"), (2, 2), [(0, 2), (0, 2), (1, 2), (1, 2)]),
    "dp_shard1_ep2_tp2": (("dp_shard", "ep", "tp"), (1, 2, 2), [(0, 1)] * 4),
    "pp2_dp_shard2": (("pp", "dp_shard"), (2, 2), [(0, 2), (1, 2), (0, 2), (1, 2)]),
    "pp2_dp_replicate2": (("pp", "dp_replicate"), (2, 2), [(0, 2), (1, 2), (0, 2), (1, 2)]),
    "pp4": (("pp",), (4,), [(0, 1)] * 4),
}


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
    """Build each ``(names, sizes, expected)`` mesh and compare this rank's DP info.

    Every mesh is checked even after one fails, and the failure names each mesh
    whose DP info was wrong or whose ``get_dp_info`` raised.
    """
    _init(rank, world, port)
    try:
        wrong = []
        for names, sizes, expected in layouts:
            mesh = init_device_mesh("cpu", sizes, mesh_dim_names=names)
            try:
                got = get_dp_info(mesh)
            except Exception as e:
                got = f"{type(e).__name__}: {e}"
            if got != expected[rank]:
                wrong.append(f"mesh {names} of {sizes}: {got}, expected {expected[rank]}")
        assert not wrong, f"rank {rank}:\n" + "\n".join(wrong)
    finally:
        dist.destroy_process_group()


def _by_world(layouts: dict) -> dict[int, list]:
    """The ``(names, sizes, expected)`` layouts by world size, for one spawn per size."""
    groups: dict[int, list] = {}
    for layout in layouts.values():
        groups.setdefault(len(layout[2]), []).append(layout)
    return groups


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
        mine = [list(data.dataloader.sampler), list(eval_loader.sampler)]
        everyone: list = [None] * world
        dist.all_gather_object(everyone, ((data.dp_rank, data.dp_size), mine))
        dp_infos = [info for info, _ in everyone]
        assert dp_infos == [(r, world) for r in range(world)], f"DP info by rank: {dp_infos}"
        for loader, samples in enumerate(mine):
            seen = [i for _, theirs in everyone for i in theirs[loader]]
            assert len(samples) > 0, f"rank {rank}, loader {loader}: no samples"
            assert len(seen) == len(set(seen)) == world * len(samples), (
                f"loader {loader}: {len(seen)} samples, {len(set(seen))} distinct, "
                f"expected {world * len(samples)} disjoint"
            )
    finally:
        dist.destroy_process_group()


def _loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean token cross-entropy, as the default training loss computes it."""
    return F.cross_entropy(logits.flatten(0, 1).float(), labels.flatten())


def _check_hsdp_step(rank: int, world: int, port: int) -> None:
    """One FSDP2 step on a 2 x 2 HSDP mesh, each rank on its DP rank's rows, against one process.

    The batch has two rows per rank, sized from the world rather than from
    ``get_dp_info``, so a DP rank shared by two ranks repeats rows and drops others.
    """
    seq = 16
    _init(rank, world, port)
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp_replicate", "dp_shard"))
        config = ModelConfig(dim=64, n_layers=2, n_heads=4, vocab_size=128, max_seq_len=seq)
        torch.manual_seed(0)
        reference = Transformer(config)
        model = Transformer(config)
        model.load_state_dict(reference.state_dict())
        fp32 = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)
        apply_fsdp2(model, mesh, mp_policy=fp32)

        generator = torch.Generator().manual_seed(7)
        tokens = torch.randint(0, config.vocab_size, (2 * world, seq), generator=generator)
        labels = torch.randint(0, config.vocab_size, (2 * world, seq), generator=generator)
        dp_rank, _ = get_dp_info(mesh)
        rows = slice(2 * dp_rank, 2 * dp_rank + 2)
        _loss(model(tokens[rows]), labels[rows]).backward()
        norm = clip_grad_norm_(model, 0.05)

        _loss(reference(tokens), labels).backward()
        ref_norm = clip_grad_norm_(reference, 0.05)
        norm = norm.full_tensor() if isinstance(norm, DTensor) else norm
        placements = {name: p.placements for name, p in model.named_parameters()}
        grads = {name: p.grad.full_tensor() for name, p in model.named_parameters()}
        torch.testing.assert_close(norm, ref_norm, rtol=1e-5, atol=0.0)
        for name, grad in grads.items():
            assert placements[name] == (Replicate(), Shard(0)), name
            torch.testing.assert_close(
                grad, reference.get_parameter(name).grad, rtol=1e-5, atol=1e-7
            )
    finally:
        dist.destroy_process_group()


DP_PEERS = [
    (("dp_replicate", "dp_shard"), (2, 2), [[0, 1, 2, 3]] * 4),
    (("dp_shard", "tp"), (2, 2), [[0, 2], [1, 3], [0, 2], [1, 3]]),
    (("pp", "dp_shard"), (2, 2), [[0, 1], [0, 1], [2, 3], [2, 3]]),
    (("pp", "dp_replicate", "dp_shard"), (2, 1, 2), [[0, 1], [0, 1], [2, 3], [2, 3]]),
    (("dp_replicate", "dp_shard", "tp"), (2, 1, 2), [[0, 2], [1, 3], [0, 2], [1, 3]]),
    (("dp_shard", "ep", "tp"), (1, 2, 2), [[0], [1], [2], [3]]),
]


def _check_dp_reduction(rank: int, world: int, port: int) -> None:
    """Sum and average over each mesh's DP groups; only the rank's DP peers take part."""
    _init(rank, world, port)
    try:
        wrong = []
        for names, sizes, peers in DP_PEERS:
            mesh = init_device_mesh("cpu", sizes, mesh_dim_names=names)
            groups = data_parallel_groups(mesh)
            total = dp_all_reduce_(torch.tensor([10.0**rank]), groups).item()
            mean = dp_mean(float(rank), groups, torch.device("cpu"))
            mine = peers[rank]
            expected = (sum(10.0**r for r in mine), sum(mine) / len(mine))
            if (total, mean) != expected:
                wrong.append(f"mesh {names} of {sizes}: {(total, mean)}, expected {expected}")
        assert not wrong, f"rank {rank}:\n" + "\n".join(wrong)
    finally:
        dist.destroy_process_group()


class _AllFinite:
    """Stands in for NaNDetector, whose agreement all-reduce allocates on CUDA."""

    should_rollback = False

    def check_loss(self, loss: float, step: int, grad_norm: float | None = None) -> bool:
        return True


def _check_loop_reduces_losses(rank: int, world: int, port: int) -> None:
    """Each rank trains and evaluates on its own data; every rank logs the DP mean."""
    from kempnerforge.training.hooks import HookRunner, TrainingHook
    from kempnerforge.training.loop import run_training_loop, text_step
    from tests.unit.test_train_entry import VOCAB, make_config, make_session

    _init(rank, world, port)
    try:
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp_shard",))
        generator = torch.Generator().manual_seed(10 + rank)

        def batch(rows: int) -> dict[str, torch.Tensor]:
            return {
                "input_ids": torch.randint(0, VOCAB, (rows, 4), generator=generator),
                "labels": torch.randint(0, VOCAB, (rows, 4), generator=generator),
            }

        train_batch, eval_batches = batch(4), [batch(2), batch(2)]
        local_losses: list[float] = []

        def step_fn(session, step):
            result = text_step(session, step)
            local_losses.append(result.loss)
            return result

        class Record(TrainingHook):
            def __init__(self) -> None:
                self.losses: list[float] = []
                self.local_evals: list[float] = []
                self.evals: list[float] = []

            def on_step_end(self, ctx) -> None:
                self.losses.append(ctx.loss)
                with torch.no_grad():
                    losses = [
                        session.loss_fn(ctx.model(b["input_ids"]), b["labels"]).item()
                        for b in eval_batches
                    ]
                self.local_evals.append(sum(losses) / len(losses))

            def on_eval_end(self, metrics, step) -> None:
                self.evals.append(metrics["eval/loss"])

        config = make_config(max_steps=2, batch_size=4)
        config.eval = EvalConfig(enabled=True, interval=1, steps=2, dataset_path="/unused")
        hook = Record()
        torch.manual_seed(0)
        session = make_session(
            config,
            runtime=RuntimeContext(
                rank=rank,
                local_rank=rank,
                world_size=world,
                device=torch.device("cpu"),
                device_mesh=mesh,
            ),
            data=DataPipeline(dataloader=[train_batch], dp_rank=rank, dp_size=world),
            step_fn=step_fn,
            hooks=HookRunner([hook]),
            nan_detector=_AllFinite(),
            eval_dataloader=eval_batches,
        )
        session.tracker.metrics_config.log_interval = 1
        published: list[float] = []
        log_step = session.tracker._log_step
        session.tracker._log_step = lambda s, m: (published.append(m.loss), log_step(s, m))[1]
        run_training_loop(session)

        everyone: list = [None] * world
        dist.all_gather_object(everyone, (local_losses, hook.local_evals))
        train_means = [sum(e[0][i] for e in everyone) / world for i in range(2)]
        eval_means = [sum(e[1][i] for e in everyone) / world for i in range(2)]
        assert published == pytest.approx(train_means, rel=1e-12), (rank, published, everyone)
        assert hook.losses == pytest.approx(train_means, rel=1e-12), (rank, hook.losses)
        assert hook.evals == pytest.approx(eval_means, rel=1e-6), (rank, hook.evals, everyone)
    finally:
        dist.destroy_process_group()


def _check_source_metrics(rank: int, world: int, port: int) -> None:
    """Each rank draws a different mix of two sources; every rank logs the global split."""
    from kempnerforge.training.data_pipeline import PhaseState
    from kempnerforge.training.loop import run_training_loop
    from tests.unit.test_train_entry import VOCAB, _mixture_pipeline, make_config, make_session

    _init(rank, world, port)
    try:
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp_shard",))
        generator = torch.Generator().manual_seed(20 + rank)
        sources = [torch.tensor([0, 1, 1, 1]), torch.tensor([0, 0, 0, 1])][rank]
        batch = {
            "input_ids": torch.randint(0, VOCAB, (4, 4), generator=generator),
            "labels": torch.randint(0, VOCAB, (4, 4), generator=generator),
            "dataset_idx": sources,
        }
        data = _mixture_pipeline()
        data.dataloader = [batch]
        torch.manual_seed(0)
        session = make_session(
            make_config(max_steps=1, batch_size=4),
            runtime=RuntimeContext(
                rank=rank,
                local_rank=rank,
                world_size=world,
                device=torch.device("cpu"),
                device_mesh=mesh,
            ),
            data=data,
            phases=PhaseState(),
            nan_detector=_AllFinite(),
        )
        with torch.no_grad():
            logits = session.model(batch["input_ids"])
        rows = sources == 0
        own = F.cross_entropy(
            logits[rows].reshape(-1, VOCAB), batch["labels"][rows].reshape(-1), reduction="sum"
        ).item()
        logged: dict = {}
        session.tracker.log_eval = lambda m, s: logged.update(m)
        run_training_loop(session)

        everyone: list = [None] * world
        dist.all_gather_object(everyone, (own, int(rows.sum()) * 4))
        expected = sum(e[0] for e in everyone) / sum(e[1] for e in everyone)
        assert logged["loss/a"] == pytest.approx(expected, rel=1e-6), (rank, logged, everyone)
        assert (logged["data/a/tokens"], logged["data/b/tokens"]) == (16.0, 16.0), (rank, logged)
    finally:
        dist.destroy_process_group()


def _check_text_tokens(rank: int, world: int, port: int) -> None:
    """Two ranks that share one DP rank, as TP peers do, count their batch once."""
    from kempnerforge.config.schema import AdapterConfig, VisionEncoderConfig, VLMConfig
    from kempnerforge.training.loop import StepResult, _log_periodic_metrics
    from tests.unit.test_train_entry import make_config, make_session

    _init(rank, world, port)
    try:
        mesh = init_device_mesh("cpu", (1, world), mesh_dim_names=("dp_shard", "tp"))
        config = JobConfig(
            model=ModelConfig(dim=64, n_layers=2, n_heads=4, vocab_size=256, max_seq_len=64),
            train=TrainConfig(batch_size=1, seq_len=64, max_steps=1),
            vision_encoder=VisionEncoderConfig(type="random", feature_dim=96, num_tokens=8),
            adapter=AdapterConfig(),
            vlm=VLMConfig(max_text_len=32),
        )
        runtime = RuntimeContext(
            rank=rank,
            local_rank=rank,
            world_size=world,
            device=torch.device("cpu"),
            device_mesh=mesh,
        )
        session = make_session(make_config(), runtime=runtime)
        session.config = config
        logged: dict = {}
        session.tracker.log_eval = lambda m, s: logged.update(m)
        _log_periodic_metrics(session, StepResult(loss=1.0, grad_norm=0.1, text_tokens=10), 1)
        assert logged["data/text_tokens_trained"] == 10.0, (rank, logged)
    finally:
        dist.destroy_process_group()


class TestGetDpInfoOnRealMeshes:
    @pytest.mark.parametrize("world", sorted(_by_world(HSDP_LAYOUTS)))
    def test_hsdp_meshes_give_one_dp_rank_per_data_shard(self, world):
        """Under HSDP every (dp_replicate, dp_shard) pair reads its own data.

        The DP rank is the row-major index of the pair, so on a mesh of DP dims
        alone it equals the global rank. Ranks that differ only in ``pp``,
        ``tp`` or ``ep`` share it, and the DP size is dp_replicate x dp_shard.
        """
        layouts = _by_world(HSDP_LAYOUTS)[world]
        mp.spawn(_check_layouts, args=(world, _free_port(), layouts), nprocs=world)

    @pytest.mark.parametrize("world", sorted(_by_world(OTHER_LAYOUTS)))
    def test_meshes_without_hsdp_keep_their_dp_ranks(self, world):
        """One DP dim, or none under pure PP, gives the same DP info as before HSDP."""
        layouts = _by_world(OTHER_LAYOUTS)[world]
        mp.spawn(_check_layouts, args=(world, _free_port(), layouts), nprocs=world)


class TestDataParallelReduction:
    def test_sums_and_averages_over_the_dp_peers_only(self):
        """The sum runs over ``dp_shard``, then ``dp_replicate``, on the mesh's own groups.

        Ranks that differ only in ``pp``, ``ep`` or ``tp`` read the same data, so they
        are not summed, and with one DP rank nothing is.
        """
        mp.spawn(_check_dp_reduction, args=(4, _free_port()), nprocs=4)


class TestLoggedMetricsOverDataParallel:
    def test_train_and_eval_loss_are_dp_means_on_every_rank(self):
        """The tracker, the hooks and eval see the mean over the DP ranks, not each
        rank's own loss, so rank 0's backends log the whole job's loss."""
        mp.spawn(_check_loop_reduces_losses, args=(2, _free_port()), nprocs=2)

    def test_per_source_metrics_cover_every_dp_rank(self):
        """``loss/<name>`` is token-weighted over every rank's rows of that source and
        ``data/<name>/tokens`` sums every rank's, on every rank."""
        mp.spawn(_check_source_metrics, args=(2, _free_port()), nprocs=2)

    def test_vlm_text_tokens_count_each_dp_rank_once(self):
        """The text-token count is summed over the DP ranks, not every rank of the world."""
        mp.spawn(_check_text_tokens, args=(2, _free_port()), nprocs=2)


class TestDataPipelineOnHsdp:
    def test_training_and_eval_loaders_split_the_data_over_both_dp_dims(self, mmap_data_dir):
        """With dp_replicate=2 x dp_shard=2 the four ranks read four disjoint parts.

        The loaders are built where training builds them, from ``get_dp_info``,
        so a rank that shared another's part would repeat its samples.
        """
        mp.spawn(_check_data_partition, args=(4, _free_port(), mmap_data_dir), nprocs=4)


class TestHsdpTrainingStep:
    def test_step_matches_one_process_on_the_global_batch(self):
        """FSDP2 on (dp_replicate, dp_shard) shards each parameter as (Replicate, Shard(0)).

        Each rank trains on the rows its DP rank selects, and the clipped gradients,
        averaged over all four ranks, equal one process's on every row.
        """
        mp.spawn(_check_hsdp_step, args=(4, _free_port()), nprocs=4)
