"""fp32 master weights: fp32 parameters, gradients and optimizer state, bf16 compute."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from kempnerforge.checkpoint.manager import CheckpointManager
from kempnerforge.config.registry import registry
from kempnerforge.config.schema import CheckpointConfig, ModelConfig, OptimizerConfig
from kempnerforge.distributed.parallel import (
    apply_fsdp2,
    autocast_dtype,
    default_mp_policy,
    is_fsdp_wrapped,
)
from kempnerforge.model.transformer import Transformer
from kempnerforge.training.optimizer import build_optimizer
from tests.gloo_ranks import gloo_group, spawn

CONFIG = ModelConfig(dim=64, n_layers=2, n_heads=4, vocab_size=128, max_seq_len=32)
OPTIMIZERS = sorted(registry.list("optimizer"))
RESUMABLE_OPTIMIZERS = [
    pytest.param(
        name,
        marks=pytest.mark.xfail(
            raises=ZeroDivisionError, strict=True, reason="#236: schedule_free_adamw cannot resume"
        ),
    )
    if name == "schedule_free_adamw"
    else name
    for name in OPTIMIZERS
]


def _tokens(seed: int) -> torch.Tensor:
    return torch.randint(
        0, CONFIG.vocab_size, (2, 16), generator=torch.Generator().manual_seed(seed)
    )


def _sharded(dtype: torch.dtype, world: int) -> Transformer:
    """A seeded Transformer stored in ``dtype``, FSDP2-wrapped with the bf16 compute policy."""
    torch.manual_seed(0)
    model = Transformer(CONFIG).to(dtype)
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp_shard",))
    apply_fsdp2(model, mesh, mp_policy=default_mp_policy(torch.bfloat16))
    return model


def _forward_matches_a_bf16_copy(rank: int, world: int) -> None:
    torch.manual_seed(0)
    reference = Transformer(CONFIG).to(torch.bfloat16)
    model = _sharded(torch.float32, world)
    assert is_fsdp_wrapped(model)
    assert autocast_dtype(model, torch.float32, torch.bfloat16) is None
    logits = model(_tokens(rank))
    assert logits.dtype == torch.bfloat16
    assert torch.equal(logits, reference(_tokens(rank)))
    logits.float().sum().backward()
    for p in model.parameters():
        assert isinstance(p, DTensor) and p.dtype == torch.float32
        assert p.grad is not None and p.grad.dtype == torch.float32


def _small_updates_survive_only_in_fp32(rank: int, world: int) -> None:
    for dtype, should_move in ((torch.float32, True), (torch.bfloat16, False)):
        model = _sharded(dtype, world)
        before = model.norm.weight.full_tensor().clone()
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        torch.optim.SGD(model.parameters(), lr=1e-3).step()
        moved = not torch.equal(model.norm.weight.full_tensor(), before)
        assert moved == should_move, (dtype, moved)


def _optimizer_state_is_fp32(rank: int, world: int) -> None:
    for name in OPTIMIZERS:
        model = _sharded(torch.float32, world)
        optimizer = build_optimizer(model, OptimizerConfig(name=name, lr=1e-3, fused=False))
        model(_tokens(rank)).float().sum().backward()
        optimizer.step()
        for opt in (optimizer, *getattr(optimizer, "inner_optimizers", ())):
            for state in opt.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor) and value.is_floating_point():
                        assert value.dtype == torch.float32, (name, key, value.dtype)


CHECKS: dict[str, Callable[[int, int], None]] = {
    "forward": _forward_matches_a_bf16_copy,
    "small_updates": _small_updates_survive_only_in_fp32,
    "optimizer_state": _optimizer_state_is_fp32,
}


def _run(rank: int, world: int, port: int, check: str) -> None:
    with gloo_group(rank, world, port):
        CHECKS[check](rank, world)


class TestShardedFp32MasterWeights:
    """FSDP2 keeps fp32 shards and casts them to bf16 for compute (2 gloo ranks)."""

    @pytest.mark.parametrize("check", sorted(CHECKS))
    def test_check(self, check: str) -> None:
        spawn(_run, 2, check)


class TestAutocastDtype:
    def test_none_when_master_and_compute_match(self) -> None:
        assert autocast_dtype(torch.nn.Linear(2, 2), torch.bfloat16, torch.bfloat16) is None

    def test_compute_dtype_without_an_fsdp_wrap(self) -> None:
        model = torch.nn.Linear(2, 2)
        assert not is_fsdp_wrapped(model)
        assert autocast_dtype(model, torch.float32, torch.bfloat16) == torch.bfloat16


class TestCheckpointAcrossMasterDtypes:
    """A checkpoint saved with one master dtype loads into the other, cast by DCP."""

    @pytest.mark.parametrize("name", RESUMABLE_OPTIMIZERS)
    @pytest.mark.parametrize(
        ("saved", "loaded"),
        [(torch.bfloat16, torch.float32), (torch.float32, torch.bfloat16)],
        ids=["bf16-to-fp32", "fp32-to-bf16"],
    )
    def test_model_and_optimizer_load(
        self, tmp_path: Path, name: str, saved: torch.dtype, loaded: torch.dtype
    ) -> None:
        def build(dtype: torch.dtype) -> tuple[Transformer, torch.optim.Optimizer]:
            torch.manual_seed(0)
            model = Transformer(CONFIG).to(dtype)
            return model, build_optimizer(model, OptimizerConfig(name=name, lr=1e-3, fused=False))

        model, opt = build(saved)
        model(_tokens(0)).float().sum().backward()
        opt.step()
        CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model, opt).save(step=1)

        model2, opt2 = build(loaded)
        CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model2, opt2).load()

        for src, dst in zip(model.parameters(), model2.parameters(), strict=True):
            assert dst.dtype == loaded
            assert torch.equal(dst, src.detach().to(loaded))
        for src_opt, dst_opt in zip(
            (opt, *getattr(opt, "inner_optimizers", ())),
            (opt2, *getattr(opt2, "inner_optimizers", ())),
            strict=True,
        ):
            src_state, dst_state = src_opt.state_dict()["state"], dst_opt.state_dict()["state"]
            assert src_state.keys() == dst_state.keys()
            for idx, state in src_state.items():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        restored = dst_state[idx][key]
                        assert torch.equal(restored, value.to(restored.dtype)), (idx, key)
