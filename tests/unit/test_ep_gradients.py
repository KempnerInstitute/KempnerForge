"""Expert-parallel gradients, norms and clipping against one process, on CPU over gloo.

Ranks that differ only in ``ep`` or ``tp`` share a DP rank and read the same rows. Each
case builds the model with ``build_parallel_model``, runs a step on its DP rank's rows,
and compares it with one process that runs every DP rank's rows through the unsharded
model, loaded with the weights gathered from the parallel one.
"""

from __future__ import annotations

import re
import socket
from typing import NamedTuple

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor

from kempnerforge.config.model import ModelConfig
from kempnerforge.distributed.parallel import build_parallel_model
from kempnerforge.model.transformer import Transformer
from kempnerforge.training.grad import maybe_no_sync
from kempnerforge.training.loop import add_moe_losses

SEQ = 16
ROWS = 2
FP32 = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)
UNPACKED_EXPERT = re.compile(r"(.*\.mlp\.experts)\.(\d+)\.(.*)")
PACKED_EXPERT = (".mlp.up_w", ".mlp.down_w", ".mlp.gate_w")

LAYOUTS = {
    "ep2": (("dp_shard", "ep"), (1, 2), {}),
    "dp_shard2_ep2_packed": (("dp_shard", "ep"), (2, 2), {"moe_packed_experts": True}),
    "dp_replicate2_ep2": (("dp_replicate", "dp_shard", "ep"), (2, 1, 2), {}),
    "ep2_tp2": (("dp_shard", "ep", "tp"), (1, 2, 2), {}),
}


class _Step(NamedTuple):
    """A step's parallel model, its one-process reference, and the mesh."""

    model: torch.nn.Module
    reference: Transformer
    mesh: DeviceMesh


def _free_port() -> int:
    """An unused local TCP port for the gloo rendezvous."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _init(rank: int, world: int, port: int) -> None:
    """Join a ``world``-rank gloo group, with one intra-op thread per rank.

    The ranks share the CPU, and one thread each keeps the step fast and its sums in
    a fixed order.
    """
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", rank=rank, world_size=world, init_method=f"tcp://127.0.0.1:{port}"
    )


def _spawn(fn, names: tuple[str, ...], sizes: tuple[int, ...], *args) -> None:
    """Run ``fn(rank, world, port, names, sizes, *args)`` on one gloo rank per mesh slot."""
    world = 1
    for size in sizes:
        world *= size
    mp.spawn(fn, args=(world, _free_port(), names, sizes, *args), nprocs=world)


def _is_expert(name: str) -> bool:
    return UNPACKED_EXPERT.fullmatch(name) is not None or name.endswith(PACKED_EXPERT)


def _dp(mesh: DeviceMesh) -> tuple[int, int]:
    """This rank's DP rank and the DP size: the row-major index over the DP dimensions."""
    rank, size = 0, 1
    for name in ("dp_replicate", "dp_shard"):
        if name in mesh.mesh_dim_names:
            rank = rank * mesh[name].size() + mesh.get_local_rank(name)
            size *= mesh[name].size()
    return rank, size


def _ep_size(mesh: DeviceMesh) -> int:
    return mesh["ep"].size() if "ep" in mesh.mesh_dim_names else 1


def _unsharded(
    model: torch.nn.Module, mesh: DeviceMesh, num_experts: int, grads: bool
) -> dict[str, torch.Tensor]:
    """Full weights or gradients under the unsharded model's names.

    DTensors are gathered over their mesh, and the routed experts over ``ep``, each
    rank's slice placed at its global expert index.
    """
    ep = _ep_size(mesh)
    out = {}
    for name, p in model.named_parameters():
        t = p.grad if grads else p.detach()
        assert t is not None, name
        t = t.full_tensor() if isinstance(t, DTensor) else t
        if ep == 1 or not _is_expert(name):
            out[name] = t
            continue
        parts = [torch.empty_like(t) for _ in range(ep)]
        dist.all_gather(parts, t.contiguous(), group=mesh["ep"].get_group())
        unpacked = UNPACKED_EXPERT.fullmatch(name)
        if unpacked is None:
            out[name] = torch.cat(parts)
            continue
        prefix, local, rest = unpacked.groups()
        for ep_rank, part in enumerate(parts):
            out[f"{prefix}.{ep_rank * (num_experts // ep) + int(local)}.{rest}"] = part
    return out


def _loss(model: torch.nn.Module, rows: torch.Tensor, config: ModelConfig) -> torch.Tensor:
    """Token cross-entropy plus the MoE auxiliary and router z-losses, as ``text_step`` adds."""
    logits = model(rows[:, :-1])
    loss = F.cross_entropy(logits.flatten(0, 1).float(), rows[:, 1:].flatten())
    return add_moe_losses(loss, model, config)


def _step(
    names: tuple[str, ...],
    sizes: tuple[int, ...],
    overrides: dict,
    micro_steps: int = 1,
) -> _Step:
    """One step on this DP rank's rows, and one process's step on every DP rank's rows.

    The reference averages per-batch losses, as DP does, since the auxiliary loss of a
    concatenated batch is not the mean of its parts'.
    """
    mesh = init_device_mesh("cpu", sizes, mesh_dim_names=names)
    config = ModelConfig(
        dim=64,
        n_layers=2,
        n_heads=4,
        vocab_size=128,
        max_seq_len=SEQ,
        num_experts=4,
        moe_top_k=2,
        moe_router_z_loss_weight=1e-3,
        **overrides,
    )
    torch.manual_seed(0)
    model = build_parallel_model(
        config, torch.device("cpu"), mesh, mp_policy=FP32, param_dtype=torch.float32
    )
    reference = Transformer(config)
    reference.load_state_dict(_unsharded(model, mesh, config.num_experts, grads=False))

    dp_rank, dp_size = _dp(mesh)
    generator = torch.Generator().manual_seed(7)
    batches = torch.randint(
        0, config.vocab_size, (dp_size, micro_steps, ROWS, SEQ + 1), generator=generator
    )
    for micro, rows in enumerate(batches[dp_rank]):
        with maybe_no_sync(model, micro, micro_steps):
            (_loss(model, rows, config) / micro_steps).backward()
    for rows in batches.flatten(0, 1):
        (_loss(reference, rows, config) / (dp_size * micro_steps)).backward()
    return _Step(model, reference, mesh)


def _assert_grads_match(step: _Step) -> None:
    """Every gradient against one process's, to fp32 rounding of its largest entry."""
    num_experts = step.reference.config.num_experts
    got = _unsharded(step.model, step.mesh, num_experts, grads=True)
    for name, p in step.reference.named_parameters():
        assert p.grad is not None, name
        scale = float(p.grad.abs().max())
        torch.testing.assert_close(got[name], p.grad, rtol=1e-4, atol=1e-5 * scale, msg=name)


def _check_gradients(rank, world, port, names, sizes, overrides, micro_steps) -> None:
    """The step's gradients, unclipped, against one process."""
    _init(rank, world, port)
    try:
        _assert_grads_match(_step(names, sizes, overrides, micro_steps))
    finally:
        dist.destroy_process_group()


class TestExpertParallelGradients:
    @pytest.mark.parametrize("layout", list(LAYOUTS))
    def test_expert_gradients_match_one_process(self, layout):
        """Routed-expert gradients equal one process's, not ``ep`` times them."""
        _spawn(_check_gradients, *LAYOUTS[layout], 1)
