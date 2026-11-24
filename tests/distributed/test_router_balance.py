"""The sigmoid routers balance alike on every rank from the counts of all ranks (#571).

Run with: torchrun --nproc_per_node=4 -m pytest tests/distributed/test_router_balance.py -v
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, distribute_tensor

from kempnerforge.checkpoint.manager import CheckpointManager
from kempnerforge.config.model import ModelConfig
from kempnerforge.config.schema import CheckpointConfig
from kempnerforge.distributed.parallel import build_parallel_model
from kempnerforge.model.router import SigmoidTopKRouter

pytestmark = pytest.mark.skipif(
    "RANK" not in os.environ,
    reason="Requires torchrun launcher (RANK not set)",
)

_LAYOUTS = {
    "fsdp2": ((4,), ("dp_shard",)),
    "hsdp": ((2, 2), ("dp_replicate", "dp_shard")),
    "tp": ((2, 2), ("dp_shard", "tp")),
    "ep": ((2, 2), ("dp_shard", "ep")),
}


def _full(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor.full_tensor() if isinstance(tensor, DTensor) else tensor).detach()


def _config(num_experts: int) -> ModelConfig:
    return ModelConfig(
        dim=64,
        n_layers=2,
        n_heads=4,
        vocab_size=128,
        max_seq_len=32,
        num_experts=num_experts,
        moe_top_k=2,
        moe_router="sigmoid_topk",
    )


@pytest.mark.parametrize(
    ("layout", "dtype", "num_experts"),
    [
        ("fsdp2", torch.float32, 4),
        ("fsdp2", torch.bfloat16, 4),
        ("fsdp2", torch.float32, 6),
        ("hsdp", torch.bfloat16, 4),
        ("tp", torch.float32, 4),
        ("ep", torch.float32, 4),
    ],
    ids=["fsdp2-fp32", "fsdp2-bf16", "fsdp2-uneven", "hsdp-bf16", "tp-fp32", "ep-fp32"],
)
def test_training_steps_balance_every_rank_alike(
    distributed_env, layout, dtype, num_experts
) -> None:
    """One all-reduce per step moves each router by the reference rule from all ranks' counts.

    Every rank ends with the same bias, which the next forward routes with.
    """
    if dist.get_world_size() != 4:
        pytest.skip("needs 4 ranks")
    config = _config(num_experts)
    shape, names = _LAYOUTS[layout]
    mesh = init_device_mesh("cuda", shape, mesh_dim_names=names)
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(0)
    model = build_parallel_model(config, device, mesh, param_dtype=dtype)
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    routers = [m for m in model.modules() if isinstance(m, SigmoidTopKRouter)]
    assert routers and all(not r.expert_bias.requires_grad for r in routers)
    assert all(isinstance(r.expert_bias, DTensor) for r in routers)
    seen: dict[SigmoidTopKRouter, torch.Tensor] = {}
    for router in routers:
        router.register_forward_pre_hook(
            lambda module, _: seen.__setitem__(module, _full(module.expert_bias).clone())
        )
    prev = [_full(router.expert_bias).clone() for router in routers]
    seed_rank = dist.get_rank() // 2 if layout == "tp" else dist.get_rank()
    all_reduce, calls = dist.all_reduce, []

    def counted(*args, **kwargs):
        calls.append(1)
        return all_reduce(*args, **kwargs)

    for step in range(3):
        torch.manual_seed(100 * step + seed_rank)
        tokens = torch.randint(0, config.vocab_size, (2, 16), device=device)
        logits = model(tokens)
        for router, bias in zip(routers, prev, strict=True):
            assert torch.equal(seen[router], bias.to(seen[router].dtype))
        F.cross_entropy(logits.float().flatten(0, 1), tokens.flatten()).backward()
        optimizer.step()
        optimizer.zero_grad()
        if step == 1 and dist.get_rank() == 3:
            for router in routers:
                router.pending_counts.zero_()
        local = torch.cat([router.pending_counts for router in routers])
        gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, local)
        summed = torch.stack(gathered).sum(dim=0)
        assert summed.sum() > 0
        before = [tuple(v.clone() for v in router._masters()) for router in routers]
        calls.clear()
        dist.all_reduce = counted
        try:
            model.balance_routers()
        finally:
            dist.all_reduce = all_reduce
        assert len(calls) == 1
        for i, (router, counts, (ema, master)) in enumerate(
            zip(routers, summed.split(num_experts), before, strict=True)
        ):
            want_ema = ema.double().lerp(counts.double() / counts.sum(), 1.0 - router.ema_decay)
            want_bias = master.double() + router.bias_update_rate * (1 / num_experts - want_ema)
            new_ema, new_master = router._masters()
            assert torch.allclose(new_ema.double(), want_ema, atol=1e-6)
            assert torch.allclose(new_master.double(), want_bias, atol=1e-6)
            bias = _full(router.expert_bias)
            assert torch.equal(bias, new_master.to(bias.dtype))
            copies = [torch.empty_like(bias) for _ in range(dist.get_world_size())]
            dist.all_gather(copies, bias.contiguous())
            assert all(torch.equal(copy, bias) for copy in copies)
            assert not router.pending_counts.any()
            prev[i] = bias.clone()


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_a_checkpoint_without_masters_starts_them_from_the_sharded_bias(
    distributed_env, shared_tmp_dir, compiled
) -> None:
    """A checkpoint saved before the masters existed resumes them from its EMA and bias.

    Every rank derives the same masters from the bias it gathers, uneven shards included.
    """
    if dist.get_world_size() != 4:
        pytest.skip("needs 4 ranks")
    mesh = init_device_mesh("cuda", (4,), mesh_dim_names=("dp_shard",))
    device = torch.device("cuda", torch.cuda.current_device())
    config = _config(6)
    bias = torch.linspace(-0.5, 0.5, 6, device=device)
    ema = torch.linspace(0.1, 0.2, 6, device=device)

    def build() -> tuple[torch.nn.Module, list[SigmoidTopKRouter], torch.optim.Optimizer]:
        torch.manual_seed(0)
        model = build_parallel_model(config, device, mesh, param_dtype=torch.bfloat16)
        trainable = [p for p in model.parameters() if p.requires_grad]
        routers = [m for m in model.modules() if isinstance(m, SigmoidTopKRouter)]
        return model, routers, torch.optim.AdamW(trainable, lr=1e-3)

    old, routers, optimizer = build()
    for router in routers:
        for name in SigmoidTopKRouter.derived_buffers:
            router._buffers.pop(name)
        param = router.expert_bias
        with torch.no_grad():
            param.copy_(
                distribute_tensor(bias.to(param.dtype), param.device_mesh, param.placements)
            )
            router.expert_ema.copy_(ema)
    config_dir = CheckpointConfig(dir=shared_tmp_dir)
    CheckpointManager(config_dir, old, optimizer).save(step=1)

    model, routers, optimizer = build()
    CheckpointManager(config_dir, torch.compile(model) if compiled else model, optimizer).load()
    for router in routers:
        loaded = _full(router.expert_bias)
        assert torch.equal(loaded, bias.to(loaded.dtype))
        ema_master, bias_master = router._masters()
        assert torch.equal(bias_master, loaded.float())
        assert torch.equal(ema_master, router.expert_ema.float())
        assert torch.equal(router.expert_ema, ema.to(router.expert_ema.dtype))
        copies = [torch.empty_like(bias_master) for _ in range(dist.get_world_size())]
        dist.all_gather(copies, bias_master)
        assert all(torch.equal(copy, bias_master) for copy in copies)


def test_saved_masters_resume_on_another_mesh(distributed_env, shared_tmp_dir) -> None:
    """A checkpoint saved under FSDP2 resumes its masters exactly under HSDP, optimizer left out."""
    if dist.get_world_size() != 4:
        pytest.skip("needs 4 ranks")
    device = torch.device("cuda", torch.cuda.current_device())
    config = _config(6)
    counts = torch.tensor([27, 25, 25, 23, 30, 20], device=device)

    def build(shape, names):
        torch.manual_seed(0)
        mesh = init_device_mesh("cuda", shape, mesh_dim_names=names)
        model = build_parallel_model(config, device, mesh, param_dtype=torch.bfloat16)
        trainable = [p for p in model.parameters() if p.requires_grad]
        routers = [m for m in model.modules() if isinstance(m, SigmoidTopKRouter)]
        return model, routers, torch.optim.AdamW(trainable, lr=1e-3)

    old, routers, optimizer = build((4,), ("dp_shard",))
    for router in routers:
        for _ in range(5):
            router.balance(counts)
    kept = [(*(m.clone() for m in r._masters()), _full(r.expert_bias).clone()) for r in routers]
    checkpoint = CheckpointConfig(dir=shared_tmp_dir)
    CheckpointManager(checkpoint, old, optimizer).save(step=1)

    new, routers, optimizer = build((2, 2), ("dp_replicate", "dp_shard"))
    CheckpointManager(checkpoint, new, optimizer).load(exclude_keys=["optimizer"])
    for router, (ema, master, bias) in zip(routers, kept, strict=True):
        assert torch.equal(router._masters()[0], ema)
        assert torch.equal(router._masters()[1], master)
        assert not torch.equal(master, bias.float())
        assert torch.equal(_full(router.expert_bias), bias)
