"""``vision_encoder.mixed_precision`` in the VLM build, for image and video-shaped runs.

A tiny SigLIP vision tower saved to disk stands in for a pretrained encoder, so the
``siglip2`` builder loads real Hugging Face weights offline (in fp32). Image runs feed
``(B, 3, H, W)`` pixels; video-shaped runs build with ``frames_per_clip = 2`` and feed
``(B, 2, 3, H, W)``. FSDP2 cases shard over two gloo ranks on CPU.
"""

from __future__ import annotations

import socket
from typing import Literal

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed._composable.fsdp import FSDPModule
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from kempnerforge.config.adapter import AdapterConfig
from kempnerforge.config.model import ModelConfig
from kempnerforge.config.vision import VisionEncoderConfig
from kempnerforge.config.vlm import FreezeSpec, VLMConfig
from kempnerforge.distributed.parallel import build_parallel_model, default_mp_policy
from kempnerforge.model.vlm import VLMWrapper

IMAGE_SIZE = 32
TEXT_LEN = 16
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}

Precision = Literal["bf16", "fp32"]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def siglip_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    from transformers import SiglipVisionConfig, SiglipVisionModel

    path = tmp_path_factory.mktemp("tiny_siglip")
    torch.manual_seed(0)
    SiglipVisionModel(
        SiglipVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            image_size=IMAGE_SIZE,
            patch_size=8,
            vision_use_head=False,
        )
    ).save_pretrained(path)
    return str(path)


def _build(
    siglip_dir: str,
    mixed_precision: Precision | None,
    lm_dtype: torch.dtype,
    frames: int,
    mesh=None,
    frozen: bool = False,
) -> VLMWrapper:
    torch.manual_seed(0)
    model = build_parallel_model(
        ModelConfig(dim=64, n_layers=2, n_heads=4, vocab_size=256, max_seq_len=64),
        torch.device("cpu"),
        mesh,
        vision_config=VisionEncoderConfig(
            type="siglip2",
            path=siglip_dir,
            mixed_precision=mixed_precision,
        ),
        adapter_config=AdapterConfig(),
        vlm_config=VLMConfig(
            max_text_len=TEXT_LEN,
            freeze=[FreezeSpec("vision_encoder", True)] if frozen else [],
        ),
        mp_policy=default_mp_policy(lm_dtype),
        param_dtype=lm_dtype,
        frames_per_clip=frames,
    )
    assert isinstance(model, VLMWrapper)
    return model


def _batch(frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(1)
    frame_dims = (frames,) if frames > 1 else ()
    pixels = torch.randn(2, *frame_dims, 3, IMAGE_SIZE, IMAGE_SIZE, generator=gen)
    ids = torch.randint(0, 256, (2, TEXT_LEN), generator=gen)
    return pixels, ids


def _dtype_recorder(seen: dict[str, tuple[torch.dtype, torch.dtype]], name: str):
    def hook(module: nn.Module, args: tuple, output: torch.Tensor) -> None:
        seen.setdefault(name, (module.weight.dtype, args[0].dtype))

    return hook


def _forward_backward(
    wrapper: VLMWrapper, frames: int
) -> tuple[torch.Tensor, dict[str, tuple[torch.dtype, torch.dtype]]]:
    """Run one step; return the logits and each tower's first-Linear (weight, input) dtypes."""
    seen: dict[str, tuple[torch.dtype, torch.dtype]] = {}
    handles = []
    for name, tower in (("encoder", wrapper.vision_encoder), ("transformer", wrapper.transformer)):
        linear = next(m for m in tower.modules() if isinstance(m, nn.Linear))
        handles.append(linear.register_forward_hook(_dtype_recorder(seen, name)))
    pixels, ids = _batch(frames)
    logits, _ = wrapper(pixels, ids, ids)
    logits.float().mean().backward()
    for handle in handles:
        handle.remove()
    return logits.detach(), seen


def _local(t: torch.Tensor) -> torch.Tensor:
    return t.to_local() if isinstance(t, DTensor) else t


def _encoder_grads(wrapper: VLMWrapper) -> list[torch.Tensor]:
    grads = []
    for p in wrapper.vision_encoder.parameters():
        assert p.grad is not None
        grads.append(_local(p.grad))
    return grads


FRAMES = pytest.mark.parametrize("frames", [1, 2], ids=["image", "video"])


@FRAMES
@pytest.mark.parametrize(
    ("precision", "lm_dtype"), [("fp32", torch.bfloat16), ("bf16", torch.float32)]
)
def test_unwrapped_encoder_is_cast_to_its_precision(
    siglip_dir: str, precision: Precision, lm_dtype: torch.dtype, frames: int
) -> None:
    """Without FSDP2 the encoder is cast to its own dtype; the language model keeps its own."""
    enc_dtype = DTYPES[precision]
    wrapper = _build(siglip_dir, precision, lm_dtype, frames)
    assert {p.dtype for p in wrapper.vision_encoder.parameters()} == {enc_dtype}
    assert {p.dtype for p in wrapper.transformer.parameters()} == {lm_dtype}
    assert {p.dtype for p in wrapper.adapter.parameters()} == {lm_dtype}

    logits, seen = _forward_backward(wrapper, frames)
    assert seen == {"encoder": (enc_dtype, enc_dtype), "transformer": (lm_dtype, lm_dtype)}
    assert logits.dtype == lm_dtype
    for grad in _encoder_grads(wrapper):
        assert grad.dtype == enc_dtype
        assert torch.isfinite(grad).all()


@FRAMES
def test_frozen_encoder_is_cast_to_its_precision(siglip_dir: str, frames: int) -> None:
    wrapper = _build(siglip_dir, "bf16", torch.float32, frames, frozen=True)
    assert not wrapper.vision_encoder.training
    assert {p.dtype for p in wrapper.vision_encoder.parameters()} == {torch.bfloat16}
    _, seen = _forward_backward(wrapper, frames)
    assert seen["encoder"] == (torch.bfloat16, torch.bfloat16)


@FRAMES
def test_unset_precision_keeps_the_unwrapped_encoder_as_loaded(
    siglip_dir: str, frames: int
) -> None:
    """Unset is the default build: the encoder stays fp32 as loaded, under a bf16 language model.

    That equals pinning ``"fp32"`` for this fp32 checkpoint, so the two runs match bit for bit.
    """
    default = _build(siglip_dir, None, torch.bfloat16, frames)
    assert {p.dtype for p in default.vision_encoder.parameters()} == {torch.float32}
    default_logits, default_seen = _forward_backward(default, frames)
    assert default_seen["encoder"] == (torch.float32, torch.float32)

    pinned = _build(siglip_dir, "fp32", torch.bfloat16, frames)
    pinned_logits, _ = _forward_backward(pinned, frames)
    assert torch.equal(default_logits, pinned_logits)
    for a, b in zip(_encoder_grads(default), _encoder_grads(pinned), strict=True):
        assert torch.equal(a, b)


def _check_fsdp_precision(rank: int, world: int, port: int, siglip_dir: str) -> None:
    dist.init_process_group(
        "gloo", rank=rank, world_size=world, init_method=f"tcp://127.0.0.1:{port}"
    )
    try:
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp_shard",))
        for precision, lm_dtype in (("fp32", torch.bfloat16), ("bf16", torch.float32)):
            for frames in (1, 2):
                wrapper = _build(siglip_dir, precision, lm_dtype, frames, mesh=mesh)
                assert isinstance(wrapper.vision_encoder, FSDPModule)
                assert {p.dtype for p in wrapper.vision_encoder.parameters()} == {torch.float32}
                assert {p.dtype for p in wrapper.transformer.parameters()} == {lm_dtype}
                enc_dtype = DTYPES[precision]
                logits, seen = _forward_backward(wrapper, frames)
                assert seen == {
                    "encoder": (enc_dtype, enc_dtype),
                    "transformer": (lm_dtype, lm_dtype),
                }
                assert logits.dtype == lm_dtype
                for grad in _encoder_grads(wrapper):
                    assert grad.dtype == torch.float32
                    assert torch.isfinite(grad).all()

        frozen = _build(siglip_dir, "bf16", torch.float32, 2, mesh=mesh, frozen=True)
        assert not isinstance(frozen.vision_encoder, FSDPModule)
        assert {p.dtype for p in frozen.vision_encoder.parameters()} == {torch.bfloat16}
        _, seen = _forward_backward(frozen, 2)
        assert seen["encoder"] == (torch.bfloat16, torch.bfloat16)
    finally:
        dist.destroy_process_group()


def _check_fsdp_unset_follows_the_language_model(
    rank: int, world: int, port: int, siglip_dir: str
) -> None:
    dist.init_process_group(
        "gloo", rank=rank, world_size=world, init_method=f"tcp://127.0.0.1:{port}"
    )
    try:
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp_shard",))
        for lm_dtype, same_precision in ((torch.bfloat16, "bf16"), (torch.float32, "fp32")):
            default = _build(siglip_dir, None, lm_dtype, 1, mesh=mesh)
            default_logits, default_seen = _forward_backward(default, 1)
            assert default_seen["encoder"] == (lm_dtype, lm_dtype)

            pinned = _build(siglip_dir, same_precision, lm_dtype, 1, mesh=mesh)
            pinned_logits, _ = _forward_backward(pinned, 1)
            assert torch.equal(default_logits, pinned_logits)
            for a, b in zip(_encoder_grads(default), _encoder_grads(pinned), strict=True):
                assert torch.equal(a, b)
    finally:
        dist.destroy_process_group()


def test_fsdp_encoder_computes_in_its_precision(siglip_dir: str) -> None:
    """Under FSDP2 the encoder's own policy sets its compute dtype for image and video runs.

    The shards and their gradients stay in the loaded fp32, and the language model computes
    in ``param_dtype``. A frozen encoder is not wrapped, so it is cast instead.
    """
    mp.spawn(_check_fsdp_precision, args=(2, _free_port(), siglip_dir), nprocs=2)


def test_fsdp_unset_precision_follows_the_language_model(siglip_dir: str) -> None:
    """Unset keeps the default wrap: the encoder computes in the language model's dtype.

    Pinning that same dtype gives the same policy, so logits and gradients match bit for bit.
    """
    mp.spawn(
        _check_fsdp_unset_follows_the_language_model,
        args=(2, _free_port(), siglip_dir),
        nprocs=2,
    )
