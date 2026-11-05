"""How the build treats a vision encoder by what the freeze specs and stages let it train."""

from __future__ import annotations

import socket
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from kempnerforge.config.registry import registry
from kempnerforge.model.vision import VisionEncoder

_ENCODER = "trainable_for_unfreeze_tests"


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


def _register_encoder() -> None:
    if _ENCODER not in registry.list_vision_encoders():
        registry.register_vision_encoder(_ENCODER)(_build_encoder)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _check_unfreeze(rank: int, world: int, port: int, unfreeze: bool) -> None:
    """Build a VLM whose encoder the build freezes, with bf16 compute over fp32 storage.

    Without a stage that unfreezes it, the encoder stays an unwrapped replica in eval mode.
    With one, it is FSDP2-wrapped: a step leaves it as it is, and after the stage and
    ``refresh_fsdp_dtypes`` a step trains it, keeps its fp32 storage, and leaves the same
    weights on every rank although the ranks see different data.
    """
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy
    from torch.distributed.tensor import DTensor

    from kempnerforge.config.adapter import AdapterConfig
    from kempnerforge.config.optimizer import OptimizerConfig
    from kempnerforge.config.schema import ModelConfig
    from kempnerforge.config.vision import VisionEncoderConfig
    from kempnerforge.config.vlm import FreezeSpec, FreezeStage, VLMConfig
    from kempnerforge.distributed.parallel import build_parallel_model, refresh_fsdp_dtypes
    from kempnerforge.training.freeze import apply_freeze_specs, trainable_for_the_run
    from kempnerforge.training.optimizer import build_optimizer

    _register_encoder()
    dist.init_process_group(
        "gloo", rank=rank, world_size=world, init_method=f"tcp://127.0.0.1:{port}"
    )
    try:
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp_shard",))
        stage = FreezeStage(2, (FreezeSpec("vision_encoder", False),))
        vlm = VLMConfig(
            max_text_len=8,
            freeze=[FreezeSpec("vision_encoder")],
            freeze_schedule=[stage] if unfreeze else [],
        )
        torch.manual_seed(0)
        model = build_parallel_model(
            ModelConfig(dim=32, n_layers=1, n_heads=2, n_kv_heads=2, vocab_size=64, max_seq_len=16),
            torch.device("cpu"),
            mesh,
            vision_config=VisionEncoderConfig(type=_ENCODER, feature_dim=16, num_tokens=4),
            adapter_config=AdapterConfig(),
            vlm_config=vlm,
            mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
            param_dtype=torch.float32,
        )
        encoder = list(model.vision_encoder.parameters())
        if not unfreeze:
            assert not any(isinstance(p, DTensor) for p in encoder)
            assert not model.vision_encoder.training
            return
        assert all(isinstance(p, DTensor) for p in encoder)
        with trainable_for_the_run(model, vlm):
            optimizer = build_optimizer(model, OptimizerConfig(lr=1e-2, fused=False))
        before = [p.full_tensor().clone() for p in encoder]
        generator = torch.Generator().manual_seed(rank)

        def step() -> None:
            pixels = torch.rand(1, 3, 8, 8, generator=generator)
            tokens = torch.randint(0, 64, (1, 8), generator=generator)
            logits, _ = model(pixels, tokens, tokens)
            logits.float().square().mean().backward()
            optimizer.step()
            optimizer.zero_grad()

        step()
        assert all(torch.equal(p.full_tensor(), b) for p, b in zip(encoder, before, strict=True))
        apply_freeze_specs(model, stage.specs, vlm.module_patterns)
        refresh_fsdp_dtypes(model, encoder)
        step()
        for param, initial in zip(encoder, before, strict=True):
            full = param.full_tensor()
            assert full.dtype == torch.float32
            assert not torch.equal(full, initial)
            gathered = [torch.empty_like(full) for _ in range(world)]
            dist.all_gather(gathered, full)
            assert all(torch.equal(g, gathered[0]) for g in gathered)
    finally:
        dist.destroy_process_group()


class TestUnfreezeUnderFSDP2:
    @pytest.mark.parametrize("unfreeze", [True, False])
    def test_an_encoder_a_stage_unfreezes_is_sharded_and_trains_in_step(self, unfreeze):
        import torch.multiprocessing as mp

        mp.spawn(_check_unfreeze, args=(2, _free_port(), unfreeze), nprocs=2)


def _build(freeze: list, schedule: list | None = None) -> torch.nn.Module:
    from kempnerforge.config.adapter import AdapterConfig
    from kempnerforge.config.schema import ModelConfig
    from kempnerforge.config.vision import VisionEncoderConfig
    from kempnerforge.config.vlm import VLMConfig
    from kempnerforge.distributed.parallel import build_parallel_model

    _register_encoder()
    return build_parallel_model(
        ModelConfig(dim=32, n_layers=1, n_heads=2, n_kv_heads=2, vocab_size=64, max_seq_len=16),
        torch.device("cpu"),
        None,
        vision_config=VisionEncoderConfig(type=_ENCODER, feature_dim=16, num_tokens=4),
        adapter_config=AdapterConfig(),
        vlm_config=VLMConfig(max_text_len=8, freeze=freeze, freeze_schedule=schedule or []),
        param_dtype=torch.float32,
    )


class TestEncoderMode:
    def test_an_encoder_frozen_for_the_whole_run_is_put_in_eval_mode(self):
        from kempnerforge.config.vlm import FreezeSpec

        model = _build([FreezeSpec("vision_encoder")])
        assert not model.vision_encoder.training

    def test_a_partly_unfrozen_encoder_stays_in_train_mode(self):
        from kempnerforge.config.vlm import FreezeSpec

        freeze = [FreezeSpec("vision_encoder"), FreezeSpec("vision_encoder.proj.bias", False)]
        model = _build(freeze)
        assert model.vision_encoder.training
        assert [p.requires_grad for p in model.vision_encoder.parameters()] == [False, True]

    def test_an_encoder_a_stage_unfreezes_stays_in_train_mode(self):
        from kempnerforge.config.vlm import FreezeSpec, FreezeStage

        stage = FreezeStage(5, (FreezeSpec("vision_encoder", False),))
        model = _build([FreezeSpec("vision_encoder")], [stage])
        assert model.vision_encoder.training
        assert not any(p.requires_grad for p in model.vision_encoder.parameters())
