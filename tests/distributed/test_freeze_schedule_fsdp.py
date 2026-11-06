"""A vision encoder the build freezes and a freeze stage unfreezes, under FSDP2 on GPUs.

Run with:
    torchrun --nproc_per_node=4 -m pytest tests/distributed/test_freeze_schedule_fsdp.py -v
"""

from __future__ import annotations

import os
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor

from kempnerforge.config.adapter import AdapterConfig
from kempnerforge.config.model import ModelConfig
from kempnerforge.config.optimizer import OptimizerConfig
from kempnerforge.config.registry import registry
from kempnerforge.config.vision import VisionEncoderConfig
from kempnerforge.config.vlm import FreezeSpec, FreezeStage, VLMConfig
from kempnerforge.distributed.parallel import build_parallel_model
from kempnerforge.model.vision import VisionEncoder
from kempnerforge.training.freeze import trainable_for_the_run
from kempnerforge.training.loop import _apply_stage_specs
from kempnerforge.training.optimizer import build_optimizer

pytestmark = pytest.mark.skipif(
    "RANK" not in os.environ,
    reason="Requires torchrun launcher (RANK not set)",
)

_ENCODER = "trainable_for_distributed_freeze_tests"


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
    return _TrainableEncoder(num_tokens or 4, feature_dim or 32)


def _vlm(mesh: Any, vlm_config: VLMConfig) -> nn.Module:
    if _ENCODER not in registry.list_vision_encoders():
        registry.register_vision_encoder(_ENCODER)(_build_encoder)
    torch.manual_seed(0)
    return build_parallel_model(
        ModelConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=4, vocab_size=128, max_seq_len=32),
        torch.device("cuda"),
        mesh,
        vision_config=VisionEncoderConfig(type=_ENCODER, feature_dim=32, num_tokens=4),
        adapter_config=AdapterConfig(),
        vlm_config=vlm_config,
        mp_policy=(
            MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
            if mesh is not None
            else None
        ),
        param_dtype=torch.float32,
    )


def _batch(rank: int, step: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(100 * rank + step)
    pixels = torch.rand(2, 3, 8, 8, generator=generator) * 4 - 2 + 3 * rank
    tokens = torch.randint(0, 128, (2, 8), generator=generator)
    return pixels.cuda(), tokens.cuda()


def _loss(model: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    pixels, tokens = batch
    logits, _ = model(pixels, tokens, tokens)
    return logits.float().square().mean()


def _full(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


class TestFreezeScheduleUnderFSDP2:
    def test_an_encoder_frozen_for_the_whole_run_stays_an_unwrapped_replica(self, distributed_env):
        if dist.get_world_size() < 2:
            pytest.skip("needs a data-parallel mesh")
        model = _vlm(
            distributed_env, VLMConfig(max_text_len=8, freeze=[FreezeSpec("vision_encoder")])
        )
        assert not any(isinstance(p, DTensor) for p in model.vision_encoder.parameters())
        assert not model.vision_encoder.training

    @pytest.mark.parametrize("compiled", [False, True])
    def test_an_encoder_a_stage_unfreezes_trains_with_averaged_gradients(
        self, distributed_env, compiled
    ):
        """bf16 compute over fp32 storage. The encoder is FSDP2-wrapped while frozen and a
        step leaves it as it is. After the stage, applied as the loop applies it (with the
        dtype refresh and, for a compiled model, a recompile), its fp32 gradients match an
        unsharded reference fed every rank's batch, which one rank's own batch does not, and
        a step trains it."""
        world, rank = dist.get_world_size(), dist.get_rank()
        if world < 2:
            pytest.skip("needs a data-parallel mesh")
        stage = FreezeStage(2, (FreezeSpec("vision_encoder", False),))
        vlm = VLMConfig(
            max_text_len=8, freeze=[FreezeSpec("vision_encoder")], freeze_schedule=[stage]
        )
        model = _vlm(distributed_env, vlm)
        encoder = list(model.vision_encoder.parameters())
        assert all(isinstance(p, DTensor) for p in encoder)
        with trainable_for_the_run(model, vlm):
            optimizer = build_optimizer(model, OptimizerConfig(lr=1e-2))
        runner = torch.compile(model) if compiled else model
        before = [p.full_tensor().clone() for p in encoder]

        _loss(runner, _batch(rank, 0)).backward()
        optimizer.step()
        optimizer.zero_grad()
        assert all(torch.equal(p.full_tensor(), b) for p, b in zip(encoder, before, strict=True))

        _apply_stage_specs(runner, stage.specs, vlm.module_patterns)
        weights = {name: _full(p).detach().clone() for name, p in model.named_parameters()}
        _loss(runner, _batch(rank, 1)).backward()
        grads = [p.grad.full_tensor() for p in encoder]

        reference = _vlm(None, vlm)
        reference.load_state_dict(weights, strict=False)
        reference.requires_grad_(True)
        (sum(_loss(reference, _batch(r, 1)) for r in range(world)) / world).backward()
        expected = [p.grad.clone() for p in reference.vision_encoder.parameters()]
        reference.zero_grad()
        _loss(reference, _batch(rank, 1)).backward()
        local = [p.grad.clone() for p in reference.vision_encoder.parameters()]
        for grad, mean, own in zip(grads, expected, local, strict=True):
            assert grad.dtype == torch.float32
            assert (grad - mean).norm() <= 0.05 * mean.norm()
            assert (own - mean).norm() > 0.2 * mean.norm()

        optimizer.step()
        assert not any(
            torch.equal(p.full_tensor(), b) for p, b in zip(encoder, before, strict=True)
        )
        if compiled:
            torch.compiler.reset()
