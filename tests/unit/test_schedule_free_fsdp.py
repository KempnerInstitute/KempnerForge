"""Schedule-free AdamW under FSDP2 on two gloo ranks.

Checkpoints hold x and resume exactly, and an eval at x leaves the training run as it was,
also when FSDP2 keeps the units gathered after a forward.
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from kempnerforge.checkpoint.manager import CheckpointManager
from kempnerforge.config.schema import CheckpointConfig, ModelConfig, OptimizerConfig
from kempnerforge.distributed.parallel import apply_fsdp2, default_mp_policy
from kempnerforge.model.transformer import Transformer
from kempnerforge.training.optimizer import build_optimizer, evaluation_point

CONFIG = ModelConfig(dim=64, n_layers=2, n_heads=2, vocab_size=256, max_seq_len=64)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _local(t: torch.Tensor) -> torch.Tensor:
    return t.to_local() if isinstance(t, DTensor) else t


def _same(a: list[torch.Tensor], b: list[torch.Tensor]) -> bool:
    return all(torch.equal(_local(x), _local(y)) for x, y in zip(a, b, strict=True))


def _resume_is_exact(rank: int, world: int, port: int, ckpt_dir: str) -> None:
    dist.init_process_group(
        "gloo", rank=rank, world_size=world, init_method=f"tcp://127.0.0.1:{port}"
    )
    try:
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp_shard",))

        def build(seed):
            torch.manual_seed(seed)
            model = Transformer(CONFIG)
            apply_fsdp2(model, mesh, mp_policy=default_mp_policy(torch.float32))
            config = OptimizerConfig(name="schedule_free_adamw", lr=1e-2, fused=False)
            return model, build_optimizer(model, config)

        def train(model, opt, batches):
            for tokens in batches:
                logits = model(tokens)
                F.cross_entropy(logits.flatten(0, 1), tokens.flatten()).backward()
                opt.step()
                opt.zero_grad()

        g = torch.Generator().manual_seed(7 + rank)
        batches = [torch.randint(0, 256, (2, 32), generator=g) for _ in range(5)]
        reference, ref_opt = build(0)
        train(reference, ref_opt, batches)

        model, opt = build(0)
        train(model, opt, batches[:3])
        y = [p.detach().clone() for p in model.parameters()]
        x = [opt.state[p]["x"].clone() for p in model.parameters()]
        CheckpointManager(CheckpointConfig(dir=ckpt_dir), model, opt).save(step=3)
        assert _same(y, [p.detach() for p in model.parameters()]), "the save left the run off y"

        fresh, _ = build(1)
        dummy = torch.optim.SGD(fresh.parameters(), lr=0.0)
        CheckpointManager(CheckpointConfig(dir=ckpt_dir), fresh, dummy).load(
            exclude_keys=["optimizer"]
        )
        assert _same(x, [p.detach() for p in fresh.parameters()]), "the model entry is not x"

        model2, opt2 = build(1)
        CheckpointManager(CheckpointConfig(dir=ckpt_dir), model2, opt2).load()
        assert _same(y, [p.detach() for p in model2.parameters()]), "the resume is not at y"
        train(model2, opt2, batches[3:])
        assert _same(
            [p.detach() for p in reference.parameters()], [p.detach() for p in model2.parameters()]
        ), "the resumed run diverged from the uninterrupted one"
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _eval_leaves_the_run(rank: int, world: int, port: int, reshard_after_forward: bool) -> None:
    dist.init_process_group(
        "gloo", rank=rank, world_size=world, init_method=f"tcp://127.0.0.1:{port}"
    )
    try:
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp_shard",))
        g = torch.Generator().manual_seed(7 + rank)
        batches = [torch.randint(0, 256, (2, 32), generator=g) for _ in range(8)]
        probe = torch.randint(0, 256, (2, 32), generator=torch.Generator().manual_seed(99))

        def run(eval_after: int | None) -> list[torch.Tensor]:
            torch.manual_seed(0)
            model = Transformer(CONFIG)
            apply_fsdp2(
                model,
                mesh,
                mp_policy=default_mp_policy(torch.float32),
                reshard_after_forward=reshard_after_forward,
            )
            config = OptimizerConfig(name="schedule_free_adamw", lr=1e-2, fused=False)
            opt = build_optimizer(model, config)
            for i, tokens in enumerate(batches):
                logits = model(tokens)
                F.cross_entropy(logits.flatten(0, 1), tokens.flatten()).backward()
                opt.step()
                opt.zero_grad()
                if i == eval_after:
                    with torch.no_grad(), evaluation_point(opt, model):
                        model(probe)
            return [p.detach() for p in model.parameters()]

        assert _same(run(eval_after=None), run(eval_after=3)), "the eval changed the run"
        dist.barrier()
    finally:
        dist.destroy_process_group()


class TestScheduleFreeUnderFsdp2:
    def test_a_checkpoint_holds_x_and_resumes_exactly(self, tmp_path: Path) -> None:
        mp.spawn(_resume_is_exact, args=(2, _free_port(), str(tmp_path / "ckpt")), nprocs=2)

    @pytest.mark.parametrize("reshard_after_forward", [True, False])
    def test_an_eval_at_x_leaves_the_training_run_unchanged(
        self, reshard_after_forward: bool
    ) -> None:
        """Units kept gathered after the eval forward are resharded, so training sees y."""
        mp.spawn(_eval_leaves_the_run, args=(2, _free_port(), reshard_after_forward), nprocs=2)
