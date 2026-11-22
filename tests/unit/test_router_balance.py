"""The sigmoid router's bias is balanced once per optimizer step, never trained (#571)."""

from __future__ import annotations

import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor

from kempnerforge.checkpoint.manager import (
    CheckpointManager,
    derived_buffer_owners,
    drop_derived_model_keys,
)
from kempnerforge.config.model import ModelConfig
from kempnerforge.config.schema import CheckpointConfig, OptimizerConfig
from kempnerforge.distributed.parallel import apply_fsdp2
from kempnerforge.model.router import SigmoidTopKRouter
from kempnerforge.model.transformer import Transformer
from kempnerforge.training.freeze import freeze_params
from kempnerforge.training.optimizer import build_optimizer


def _config(**overrides) -> ModelConfig:
    base = dict(
        dim=32,
        n_layers=2,
        n_heads=4,
        vocab_size=64,
        max_seq_len=16,
        num_experts=4,
        moe_top_k=2,
        moe_router="sigmoid_topk",
    )
    return ModelConfig(**{**base, **overrides})


EMA = torch.tensor([0.125, 0.25, 0.25, 0.375])


def _routers(model: torch.nn.Module) -> list[SigmoidTopKRouter]:
    return [m for m in model.modules() if isinstance(m, SigmoidTopKRouter)]


def _expected(
    ema: torch.Tensor, bias: torch.Tensor, counts: torch.Tensor, rate: float, decay: float
):
    ema = ema.double().lerp(counts.double() / counts.sum(), 1.0 - decay)
    return ema, bias.double() + rate * (1.0 / counts.numel() - ema)


class TestFrozenBias:
    def test_the_optimizer_never_holds_or_moves_the_bias(self) -> None:
        torch.manual_seed(0)
        model = Transformer(_config())
        optimizer = build_optimizer(model, OptimizerConfig())
        held = {p for group in optimizer.param_groups for p in group["params"]}
        biases = [router.expert_bias for router in _routers(model)]
        assert biases and all(not b.requires_grad and b not in held for b in biases)
        before = [b.detach().clone() for b in biases]
        tokens = torch.randint(0, 64, (2, 8))
        F.cross_entropy(model(tokens).flatten(0, 1), tokens.flatten()).backward()
        optimizer.step()
        assert all(b.grad is None for b in biases)
        assert all(torch.equal(b, x) for b, x in zip(biases, before, strict=True))

    def test_unfreezing_every_parameter_leaves_the_bias_frozen(self) -> None:
        model = Transformer(_config())
        freeze_params(model, ["*"], frozen=True)
        freeze_params(model, ["*"], frozen=False)
        assert all(not router.expert_bias.requires_grad for router in _routers(model))
        assert all(router.gate.weight.requires_grad for router in _routers(model))


class TestBalance:
    def test_balance_moves_the_ema_and_the_bias_and_clears_the_counts(self) -> None:
        router = SigmoidTopKRouter(dim=8, num_experts=4, top_k=2, bias_update_rate=0.01)
        start_ema, start_bias = router._masters()
        start_ema.copy_(torch.tensor([0.4, 0.3, 0.2, 0.1]))
        start_bias.copy_(torch.tensor([0.1, -0.2, 0.0, 0.05]))
        counts = torch.tensor([10, 30, 20, 40])
        router.pending_counts.copy_(counts)
        ema, bias = _expected(start_ema.clone(), start_bias.clone(), counts, 0.01, 0.99)
        router.balance(counts)
        assert torch.allclose(router.expert_ema.double(), ema, atol=1e-7)
        assert torch.allclose(router.expert_bias.double(), bias, atol=1e-7)
        assert not router.pending_counts.any()

    def test_a_step_without_tokens_changes_nothing(self) -> None:
        router = SigmoidTopKRouter(dim=8, num_experts=4, top_k=2)
        router.pending_counts.fill_(3)
        router.balance(torch.zeros(4, dtype=torch.long))
        assert torch.equal(router.expert_ema, torch.full((4,), 0.25))
        assert not router.expert_bias.any() and not router.pending_counts.any()

    def test_bf16_storage_keeps_the_small_moves(self) -> None:
        """A bf16 EMA lerped in place would stay at 0.25; the FP32 copy keeps moving."""
        router = SigmoidTopKRouter(dim=8, num_experts=4, top_k=2).to(torch.bfloat16)
        reference = SigmoidTopKRouter(dim=8, num_experts=4, top_k=2)
        counts = torch.tensor([27, 25, 25, 23])
        for _ in range(20):
            router.balance(counts)
            reference.balance(counts)
        assert router.expert_ema.dtype == torch.bfloat16
        assert not torch.equal(router.expert_ema.float(), torch.full((4,), 0.25))
        assert torch.allclose(router.expert_ema.float(), reference.expert_ema, atol=4e-3)
        assert torch.allclose(router.expert_bias.float(), reference.expert_bias, atol=1e-5)

    def test_a_bf16_run_resumes_exactly_where_it_stopped(self) -> None:
        """The FP32 masters are checkpointed, so a resume keeps the moves bf16 rounds away."""
        counts = torch.tensor([27, 25, 25, 23])
        straight = SigmoidTopKRouter(dim=8, num_experts=4, top_k=2).to(torch.bfloat16)
        first = SigmoidTopKRouter(dim=8, num_experts=4, top_k=2).to(torch.bfloat16)
        for router in (straight, first):
            for _ in range(5):
                router.balance(counts)
        resumed = SigmoidTopKRouter(dim=8, num_experts=4, top_k=2).to(torch.bfloat16)
        resumed.load_state_dict(first.state_dict())
        for router in (straight, resumed):
            for _ in range(5):
                router.balance(counts)
        for name in ("expert_ema", "expert_bias", *SigmoidTopKRouter.derived_buffers):
            assert torch.equal(resumed.state_dict()[name], straight.state_dict()[name]), name

    def test_a_checkpoint_without_masters_starts_them_from_its_ema_and_bias(self) -> None:
        loaded_ema = torch.tensor([0.1, 0.2, 0.3, 0.4])
        loaded_bias = torch.tensor([0.5, 0.0, -0.5, 0.25])
        router = SigmoidTopKRouter(dim=8, num_experts=4, top_k=2, bias_update_rate=0.01)
        old = {k: v for k, v in router.state_dict().items() if not k.startswith("fp32_")}
        router.load_state_dict({**old, "expert_ema": loaded_ema, "expert_bias": loaded_bias})
        counts = torch.tensor([10, 30, 20, 40])
        ema, bias = _expected(loaded_ema, loaded_bias, counts, 0.01, 0.99)
        router.balance(counts)
        assert torch.allclose(router.expert_ema.double(), ema, atol=1e-7)
        assert torch.allclose(router.expert_bias.double(), bias, atol=1e-7)


class TestBalanceRouters:
    def test_every_router_balances_from_the_counts_of_its_training_forwards(self) -> None:
        torch.manual_seed(0)
        model = Transformer(_config())
        model.train()
        routers = _routers(model)
        emas = [router.expert_ema.clone() for router in routers]
        model(torch.randint(0, 64, (2, 8)))
        model(torch.randint(0, 64, (2, 8)))
        assert all(int(router.pending_counts.sum()) == 2 * 2 * 8 * 2 for router in routers)
        model.balance_routers()
        assert all(not router.pending_counts.any() for router in routers)
        assert all(not torch.equal(r.expert_ema, e) for r, e in zip(routers, emas, strict=True))

    def test_discarding_drops_the_counts_without_balancing(self) -> None:
        model = Transformer(_config())
        model.train()
        model(torch.randint(0, 64, (2, 8)))
        model.discard_router_counts()
        for router in _routers(model):
            assert not router.pending_counts.any()
            assert torch.equal(router.expert_ema, torch.full((4,), 0.25))

    def test_an_eval_forward_counts_nothing(self) -> None:
        model = Transformer(_config())
        model.eval()
        model(torch.randint(0, 64, (2, 8)))
        assert all(not router.pending_counts.any() for router in _routers(model))


def _manager(
    directory: Path, *, trained_bias: bool, lr_multipliers=None, patterns=None, compiled=False
):
    """A sigmoid MoE model, its optimizer and manager; ``trained_bias`` as on main.

    With ``compiled``, the manager holds the model wrapped by ``torch.compile``.
    """
    torch.manual_seed(0)
    model = Transformer(_config())
    for router in _routers(model):
        router.expert_bias.requires_grad_(trained_bias)
    config = OptimizerConfig(lr_multipliers=lr_multipliers) if lr_multipliers else OptimizerConfig()
    optimizer = build_optimizer(model, config, module_patterns=patterns)
    manager = CheckpointManager(
        CheckpointConfig(dir=str(directory)),
        torch.compile(model) if compiled else model,
        optimizer,
        lr_multipliers=lr_multipliers,
        module_patterns=patterns,
    )
    return model, optimizer, manager


def _save_as_main(directory: Path, **groups) -> torch.Tensor:
    """Save a checkpoint as main did: a trained bias and no FP32 masters; return a bias.

    Every router's EMA is saved as ``EMA``.
    """
    model, optimizer, manager = _manager(directory, trained_bias=True, **groups)
    for router in _routers(model):
        for name in SigmoidTopKRouter.derived_buffers:
            router._buffers.pop(name)
    tokens = torch.randint(0, 64, (2, 8))
    F.cross_entropy(model(tokens).flatten(0, 1), tokens.flatten()).backward()
    optimizer.step()
    with torch.no_grad():
        for router in _routers(model):
            router.expert_bias.add_(0.125)
            router.expert_ema.copy_(EMA)
    manager.save(step=1)
    manager.wait()
    return _routers(model)[0].expert_bias.detach().clone()


class TestCheckpointsThatTrainedTheBias:
    @pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
    def test_a_default_optimizer_checkpoint_resumes_without_the_bias_moments(
        self, tmp_path: Path, compiled: bool
    ) -> None:
        """The masters start from the loaded EMA and bias, under torch.compile's key prefix too."""
        saved = _save_as_main(tmp_path)
        model, optimizer, manager = _manager(tmp_path, trained_bias=False, compiled=compiled)
        manager.load()
        router = _routers(model)[0]
        assert torch.equal(router.expert_bias.detach(), saved)
        assert torch.equal(router.fp32_bias_bits.view(torch.float32), saved.float())
        assert torch.equal(router.expert_ema, EMA)
        assert torch.equal(router.fp32_ema_bits.view(torch.float32), EMA)
        assert all(state for state in optimizer.state.values())
        assert all(not router.expert_bias.requires_grad for router in _routers(model))

    def test_a_router_only_lr_group_is_refused_with_the_way_to_resume(self, tmp_path: Path) -> None:
        """The frozen biases empty that group, which KF's group check refuses."""
        groups = {"lr_multipliers": {"router": 2.0}, "patterns": {"router": ["*.router.*"]}}
        saved = _save_as_main(tmp_path, **groups)
        model, _, manager = _manager(tmp_path, trained_bias=False, **groups)
        with pytest.raises(RuntimeError, match=r"expert_bias.*exclude_from_loading"):
            manager.load()
        model, _, manager = _manager(tmp_path, trained_bias=False, **groups)
        manager.load(exclude_keys=["optimizer"])
        assert torch.equal(_routers(model)[0].expert_bias.detach(), saved)


def _bf16_managers(directory: Path, count: int = 2):
    """``count`` bf16 sigmoid MoE models built alike, each with its optimizer's manager."""
    pairs = []
    for _ in range(count):
        torch.manual_seed(0)
        model = Transformer(_config()).to(torch.bfloat16)
        optimizer = build_optimizer(model, OptimizerConfig())
        pairs.append(
            (model, CheckpointManager(CheckpointConfig(dir=str(directory)), model, optimizer))
        )
    return pairs


class TestManagerResume:
    @pytest.mark.parametrize(
        "exclude",
        [None, ["optimizer"], ["optimizer", "train_state"]],
        ids=["all", "no_optim", "weights"],
    )
    def test_a_checkpoint_with_masters_resumes_them_as_saved(
        self, tmp_path: Path, exclude: list[str] | None
    ) -> None:
        """The manager loads the saved masters, not ones rounded from the bf16 storage."""
        (saved, manager), (resumed, resumer) = _bf16_managers(tmp_path)
        for router in _routers(saved):
            for _ in range(5):
                router.balance(torch.tensor([27, 25, 25, 23]))
        manager.save(step=1)
        manager.wait()
        resumer.load(exclude_keys=exclude)
        for want, got in zip(_routers(saved), _routers(resumed), strict=True):
            ema, bias = got._masters()
            assert not torch.equal(ema, got.expert_ema.float())
            assert not torch.equal(bias, got.expert_bias.float())
            for a, b in zip(want._masters(), (ema, bias), strict=True):
                assert torch.equal(a, b)

    def test_a_checkpoint_missing_one_master_derives_that_one_alone(self, tmp_path: Path) -> None:
        (saved, manager), (resumed, resumer) = _bf16_managers(tmp_path)
        for router in _routers(saved):
            for _ in range(5):
                router.balance(torch.tensor([27, 25, 25, 23]))
        kept = [tuple(m.clone() for m in router._masters()) for router in _routers(saved)]
        _routers(saved)[0]._buffers.pop("fp32_ema_bits")
        manager.save(step=1)
        manager.wait()
        resumer.load()
        routers = _routers(resumed)
        derived = routers[0]._masters()[0]
        assert torch.equal(derived, routers[0].expert_ema.float())
        assert not torch.equal(derived, kept[0][0])
        assert torch.equal(routers[0]._masters()[1], kept[0][1])
        for router, masters in zip(routers[1:], kept[1:], strict=True):
            assert all(torch.equal(a, b) for a, b in zip(router._masters(), masters, strict=True))


class TestDerivedBufferOwners:
    def test_a_buffer_of_the_same_name_elsewhere_is_not_derived(self) -> None:
        holder = torch.nn.Module()
        holder.router = SigmoidTopKRouter(dim=8, num_experts=4, top_k=2)
        holder.register_buffer("fp32_bias_bits", torch.zeros(4, dtype=torch.int32))
        owners = derived_buffer_owners(holder)
        assert owners.keys() == {"router.fp32_ema_bits", "router.fp32_bias_bits"}
        assert derived_buffer_owners(torch.compile(holder)).keys() == owners.keys()
        state = holder.state_dict()
        assert drop_derived_model_keys(state, holder, set()).keys() == owners.keys()
        assert "fp32_bias_bits" in state


def _balance_on_ranks(rank: int, world: int, store: str) -> None:
    """Shard a sigmoid MoE model with FSDP2, route rank-specific tokens, balance once."""
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world,
        init_method=f"file://{store}",
        timeout=timedelta(seconds=60),
    )
    try:
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp_shard",))
        torch.manual_seed(0)
        model = Transformer(_config())
        fp32 = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)
        apply_fsdp2(model, mesh, mp_policy=fp32)
        model.train()
        torch.manual_seed(1 + rank)
        model(torch.randint(0, 64, (1, 4 + 4 * rank)))
        routers = _routers(model)
        local = torch.cat([router.pending_counts for router in routers])
        summed = local.clone()
        dist.all_reduce(summed)
        before = [(r.expert_ema.clone(), r.expert_bias.full_tensor().clone()) for r in routers]
        placements = [r.expert_bias.placements for r in routers]
        model.balance_routers()
        for router, counts, (ema, bias), kept in zip(
            routers, summed.split(4), before, placements, strict=True
        ):
            assert isinstance(router.expert_bias, DTensor) and router.expert_bias.placements == kept
            want_ema, want_bias = _expected(ema, bias, counts, 0.001, 0.99)
            full = router.expert_bias.full_tensor()
            gathered = [torch.empty_like(full) for _ in range(world)]
            dist.all_gather(gathered, full)
            assert all(torch.equal(g, full) for g in gathered)
            assert torch.allclose(full.double(), want_bias, atol=1e-7)
            assert torch.allclose(router.expert_ema.double(), want_ema, atol=1e-7)
            assert full.abs().sum() > 0
    finally:
        dist.destroy_process_group()


class TestFSDP2:
    def test_ranks_with_different_tokens_balance_the_shards_alike(self, tmp_path: Path) -> None:
        """The update comes from the counts summed over ranks and lands on each rank's shard."""
        ranks = mp.spawn(_balance_on_ranks, args=(2, str(tmp_path / "store")), nprocs=2, join=False)
        deadline = time.monotonic() + 120
        try:
            while not ranks.join(timeout=5):
                assert time.monotonic() < deadline, "the ranks did not finish"
        finally:
            for process in ranks.processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()
                    process.join()
            for error_file in ranks.error_files:
                Path(error_file).unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)
