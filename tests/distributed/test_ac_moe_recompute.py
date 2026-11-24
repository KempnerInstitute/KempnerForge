"""Full activation checkpointing of MoE blocks on 4 GPUs leaves the routers alone (#563).

Run with: torchrun --nproc_per_node=4 -m pytest tests/distributed/test_ac_moe_recompute.py -v
"""

from __future__ import annotations

import os
from dataclasses import replace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from kempnerforge.checkpoint.expert_layout import state_dict_name
from kempnerforge.config.adapter import AdapterConfig
from kempnerforge.config.model import ModelConfig
from kempnerforge.config.schema import ActivationCheckpointing
from kempnerforge.config.vision import VisionEncoderConfig
from kempnerforge.config.vlm import MoTConfig
from kempnerforge.distributed.parallel import build_parallel_model
from kempnerforge.model.router import SigmoidTopKRouter, SoftmaxTopKRouter
from kempnerforge.model.vlm import inner_transformer
from kempnerforge.training.loop import add_moe_losses

pytestmark = pytest.mark.skipif(
    "RANK" not in os.environ,
    reason="Requires torchrun launcher (RANK not set)",
)

_CONFIG = ModelConfig(
    dim=64,
    n_layers=2,
    n_heads=4,
    vocab_size=128,
    max_seq_len=64,
    ffn_hidden_dim=128,
    num_experts=4,
    moe_top_k=2,
    moe_aux_loss_weight=0.01,
    moe_router_z_loss_weight=0.001,
)
_LAYOUTS = {
    "fsdp2": ((4,), ("dp_shard",)),
    "hsdp": ((2, 2), ("dp_replicate", "dp_shard")),
    "tp": ((2, 2), ("dp_shard", "tp")),
    "ep": ((2, 2), ("dp_shard", "ep")),
}
_TOLERANCE = {torch.float32: 1e-4, torch.bfloat16: 3e-2}


def _full(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor.full_tensor() if isinstance(tensor, DTensor) else tensor).detach().float()


def _run(arch, layout, router, dtype, mode) -> tuple[list, list, dict]:
    """Two training steps of a fresh model; its losses, counts and parameters."""
    shape, names = _LAYOUTS[layout]
    mesh = init_device_mesh("cuda", shape, mesh_dim_names=names)
    device = torch.device("cuda", torch.cuda.current_device())
    config = replace(_CONFIG, moe_router=router)
    torch.manual_seed(0)
    if arch == "text":
        model = build_parallel_model(config, device, mesh, ac_mode=mode, param_dtype=dtype)
    else:
        model = build_parallel_model(
            config,
            device,
            mesh,
            vision_config=VisionEncoderConfig(type="random", feature_dim=96, num_tokens=8),
            adapter_config=AdapterConfig(),
            vlm_config=MoTConfig(max_text_len=32),
            ac_mode=mode,
            param_dtype=dtype,
        )
    model.train()
    routers = [m for m in model.modules() if isinstance(m, (SoftmaxTopKRouter, SigmoidTopKRouter))]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    seed_rank = dist.get_rank() // 2 if layout == "tp" else dist.get_rank()
    losses, counts = [], []
    for step in range(2):
        torch.manual_seed(100 * step + seed_rank)
        ids = torch.randint(0, config.vocab_size, (2, 32), device=device)
        if arch == "text":
            out = model(ids)
            loss = F.cross_entropy(out.float().flatten(0, 1), ids.flatten())
        else:
            out = model(torch.randn(2, 3, 32, 32, device=device), ids, ids)[0]
            loss = out.float().pow(2).mean()
        loss = add_moe_losses(loss, model, config)
        kept = [(r.aux_loss, r.z_loss, r.expert_counts) for r in routers]
        assert all(r.aux_loss.grad_fn is None and r.z_loss.grad_fn is None for r in routers)
        loss.backward()
        for router_, attrs in zip(routers, kept, strict=True):
            now = (router_.aux_loss, router_.z_loss, router_.expert_counts)
            assert all(a is b for a, b in zip(now, attrs, strict=True))
        counts.append(
            [r.pending_counts.clone() for r in routers if isinstance(r, SigmoidTopKRouter)]
        )
        optimizer.step()
        optimizer.zero_grad()
        inner_transformer(model).balance_routers()
        losses.append(loss.detach().float())
    params = {state_dict_name(n): _full(p) for n, p in model.named_parameters()}
    return losses, counts, params


@pytest.mark.parametrize(
    ("arch", "layout", "router", "dtype"),
    [
        ("text", "fsdp2", "softmax_topk", torch.float32),
        ("text", "fsdp2", "sigmoid_topk", torch.bfloat16),
        ("text", "hsdp", "softmax_topk", torch.bfloat16),
        ("text", "hsdp", "sigmoid_topk", torch.float32),
        ("text", "tp", "softmax_topk", torch.float32),
        ("text", "tp", "sigmoid_topk", torch.bfloat16),
        ("text", "ep", "softmax_topk", torch.bfloat16),
        ("text", "ep", "sigmoid_topk", torch.float32),
        ("mot", "fsdp2", "softmax_topk", torch.bfloat16),
        ("mot", "fsdp2", "sigmoid_topk", torch.float32),
    ],
    ids=lambda v: str(v).replace("torch.", ""),
)
def test_full_checkpointing_trains_like_none(distributed_env, arch, layout, router, dtype):
    """Losses, router counts and weights match ``none``; the recompute changes no router state.

    Every step checks that the router losses kept no graph past the loss and that the
    backward's recompute replaced no router attribute.
    """
    if dist.get_world_size() != 4:
        pytest.skip("needs 4 ranks")
    none = _run(arch, layout, router, dtype, ActivationCheckpointing.none)
    full = _run(arch, layout, router, dtype, ActivationCheckpointing.full)
    tol = _TOLERANCE[dtype]
    torch.testing.assert_close(full[0], none[0], rtol=tol, atol=tol)
    for a, b in zip(full[1], none[1], strict=True):
        assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
    if router == "sigmoid_topk":
        assert all(c.sum() > 0 for step in full[1] for c in step)
    assert full[2].keys() == none[2].keys()
    for name, param in full[2].items():
        torch.testing.assert_close(param, none[2][name], rtol=tol, atol=tol, msg=name)
