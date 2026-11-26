"""Gradient accumulation turns gradient sync off for every FSDP2 unit until the last micro-step.

A text model is switched through its FSDP2 root. A ``VLMWrapper`` is not an FSDP2 module:
its blocks, transformer, adapter and trainable encoder are sharded as separate units, and
each of them must skip its reduce-scatter, and under HSDP its all-reduce, on the
intermediate micro-steps all the same.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from kempnerforge.config.registry import registry
from kempnerforge.model.vision import VisionEncoder
from tests.gloo_ranks import gloo_group, spawn

MICRO_STEPS = 4
KINDS = ("text", "vlm", "vlm_trainable_encoder")
LAYOUTS = {
    "dp_shard2": ((2,), ("dp_shard",)),
    "dp_replicate2_dp_shard2": ((2, 2), ("dp_replicate", "dp_shard")),
}
_ENCODER = "trainable_for_grad_sync_tests"


class _TrainableEncoder(VisionEncoder):
    """Pools the image to a square grid and projects each cell's channels with one Linear."""

    def __init__(self, num_tokens: int, feature_dim: int) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.feature_dim = feature_dim
        self._side = int(num_tokens**0.5)
        self.proj = nn.Linear(3, feature_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = F.adaptive_avg_pool2d(pixel_values.to(self.proj.weight.dtype), self._side)
        return self.proj(x.flatten(2).transpose(1, 2))


def _build_encoder(
    path: str = "", num_tokens: int | None = None, feature_dim: int | None = None, **_: Any
) -> VisionEncoder:
    return _TrainableEncoder(num_tokens or 4, feature_dim or 16)


def _model(kind: str, mesh: Any) -> nn.Module:
    """A tiny fp32 text model, or a VLM whose encoder is frozen or trainable, built on ``mesh``."""
    from torch.distributed.fsdp import MixedPrecisionPolicy

    from kempnerforge.config.adapter import AdapterConfig
    from kempnerforge.config.schema import ModelConfig
    from kempnerforge.config.vision import VisionEncoderConfig
    from kempnerforge.config.vlm import FreezeSpec, VLMConfig
    from kempnerforge.distributed.parallel import build_parallel_model

    fp32 = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)
    config = ModelConfig(dim=32, n_layers=2, n_heads=2, n_kv_heads=2, vocab_size=64, max_seq_len=16)
    cpu = torch.device("cpu")
    torch.manual_seed(0)
    if kind == "text":
        return build_parallel_model(config, cpu, mesh, mp_policy=fp32, param_dtype=torch.float32)
    if kind == "vlm":
        vision = VisionEncoderConfig(type="random", feature_dim=16, num_tokens=4)
        freeze = [FreezeSpec("vision_encoder", True)]
    else:
        if _ENCODER not in registry.list_vision_encoders():
            registry.register_vision_encoder(_ENCODER)(_build_encoder)
        vision = VisionEncoderConfig(type=_ENCODER, feature_dim=16, num_tokens=4)
        freeze = []
    return build_parallel_model(
        config,
        cpu,
        mesh,
        vision_config=vision,
        adapter_config=AdapterConfig(),
        vlm_config=VLMConfig(max_text_len=8, freeze=freeze),
        mp_policy=fp32,
        param_dtype=torch.float32,
    )


def _loss(model: nn.Module, kind: str, rank: int, micro_step: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(100 * rank + micro_step)
    tokens = torch.randint(0, 64, (2, 8), generator=generator)
    if kind == "text":
        logits = model(tokens)
    else:
        logits, _ = model(torch.rand(2, 3, 8, 8, generator=generator), tokens, tokens)
    return logits.float().square().mean() / MICRO_STEPS


def _sync_flags(model: nn.Module) -> list[tuple[bool, bool]]:
    """``(reduce_grads, all_reduce_grads)`` of every FSDP2 parameter group in ``model``."""
    from torch.distributed.fsdp import FSDPModule

    return [
        (group.reduce_grads, group.all_reduce_grads)
        for module in model.modules()
        if isinstance(module, FSDPModule)
        for group in module._get_fsdp_state()._fsdp_param_groups
    ]


def _check_sync_flags(rank: int, world: int, port: int, kind: str) -> None:
    """Every parameter group skips sync on an intermediate micro-step, also behind
    ``torch.compile``, syncs on the last one, and syncs again once a block exits or raises."""
    from torch.distributed.device_mesh import init_device_mesh

    from kempnerforge.training.grad import maybe_no_sync

    with gloo_group(rank, world, port):
        model = _model(kind, init_device_mesh("cpu", (world,), mesh_dim_names=("dp_shard",)))
        groups = len(_sync_flags(model))
        on, off = [(True, True)] * groups, [(False, False)] * groups
        assert groups >= 3
        for wrapped in (model, torch.compile(model)):
            with maybe_no_sync(wrapped, 0, MICRO_STEPS):
                assert _sync_flags(model) == off, _sync_flags(model)
            assert _sync_flags(model) == on, _sync_flags(model)
            with maybe_no_sync(wrapped, MICRO_STEPS - 1, MICRO_STEPS):
                assert _sync_flags(model) == on, _sync_flags(model)
        with pytest.raises(RuntimeError, match="interrupted"), maybe_no_sync(model, 1, MICRO_STEPS):
            raise RuntimeError("interrupted")
        assert _sync_flags(model) == on, _sync_flags(model)


def _check_reductions(
    rank: int, world: int, port: int, sizes: tuple[int, ...], names: tuple[str, ...], kind: str
) -> None:
    """Each FSDP2 unit reduces once per accumulation, on the last micro-step, and the
    gradients equal those of a model that reduces on every micro-step."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import FSDPModule
    from torch.distributed.tensor.debug import CommDebugMode

    from kempnerforge.training.grad import maybe_no_sync

    with gloo_group(rank, world, port):
        mesh = init_device_mesh("cpu", sizes, mesh_dim_names=names)
        model = _model(kind, mesh)
        units = sum(isinstance(module, FSDPModule) for module in model.modules())
        reduce_scatters, all_reduces = [], []
        for micro_step in range(MICRO_STEPS):
            with CommDebugMode() as comm, maybe_no_sync(model, micro_step, MICRO_STEPS):
                _loss(model, kind, rank, micro_step).backward()
            counts = {str(op): n for op, n in comm.get_comm_counts().items()}
            reduce_scatters.append(sum(n for op, n in counts.items() if "reduce_scatter" in op))
            all_reduces.append(sum(n for op, n in counts.items() if "allreduce" in op))

        reference = _model(kind, mesh)
        for micro_step in range(MICRO_STEPS):
            _loss(reference, kind, rank, micro_step).backward()

        before_last = [0] * (MICRO_STEPS - 1)
        replicated = units if "dp_replicate" in names else 0
        assert units >= 3
        assert reduce_scatters == [*before_last, units], (reduce_scatters, units)
        assert all_reduces == [*before_last, replicated], (all_reduces, replicated)
        pairs = zip(model.named_parameters(), reference.parameters(), strict=True)
        for (name, param), expected in pairs:
            got, want = param.grad.full_tensor(), expected.grad.full_tensor()
            gap = (got - want).abs().max().item()
            assert gap <= 1e-5 * want.abs().max().item(), (name, gap)


class TestGradientSyncUnderAccumulation:
    @pytest.mark.parametrize("kind", KINDS)
    def test_every_unit_skips_sync_until_the_last_micro_step(self, kind):
        spawn(_check_sync_flags, 2, kind)

    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("layout", list(LAYOUTS))
    def test_each_unit_reduces_once_per_accumulation(self, layout, kind):
        sizes, names = LAYOUTS[layout]
        spawn(_check_reductions, math.prod(sizes), sizes, names, kind)
