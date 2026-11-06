"""``vision_encoder.mixed_precision`` in the VLM build, for image and video-shaped runs.

A tiny SigLIP vision tower saved to disk stands in for a pretrained encoder, so the
``siglip2`` builder loads real Hugging Face weights offline, in fp32 or, saved in bf16,
in bf16. The parameter-free ``random`` encoder and a plain Conv2d encoder with a
floating-point buffer cover encoders that neither hold parameters nor cast their own
inputs. Image runs feed ``(B, 3, H, W)`` pixels; video-shaped runs build with
``frames_per_clip = 2`` and feed ``(B, 2, 3, H, W)``. FSDP2 cases shard over two gloo
ranks on CPU.
"""

from __future__ import annotations

import json
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

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
from kempnerforge.config.registry import registry
from kempnerforge.config.vision import VisionEncoderConfig
from kempnerforge.config.vlm import FreezeSpec, VLMConfig
from kempnerforge.distributed.parallel import build_parallel_model, default_mp_policy
from kempnerforge.model.vision import VisionEncoder
from kempnerforge.model.vlm import VLMWrapper

IMAGE_SIZE = 32
PATCH = 8
TEXT_LEN = 16
CONV = "test_precision_conv"
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
LR = 5e-5

Precision = Literal["bf16", "fp32"]
Seen = dict[str, tuple[torch.dtype, torch.dtype]]


class _ConvEncoder(VisionEncoder):
    """A Conv2d patch encoder that casts nothing itself.

    It has a bias-free Linear, and one parameter and one buffer under two names each.
    """

    def __init__(self) -> None:
        super().__init__()
        self.num_tokens = (IMAGE_SIZE // PATCH) ** 2
        self.feature_dim = 24
        self.patch = nn.Conv2d(3, self.feature_dim, kernel_size=PATCH, stride=PATCH)
        self.mix = nn.Linear(self.feature_dim, self.feature_dim, bias=False)
        self.gain = nn.Parameter(torch.ones(1))
        self.register_parameter("gain_alias", self.gain)
        scale = torch.ones(1)
        self.register_buffer("scale", scale)
        self.register_buffer("scale_alias", scale)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        tokens = self.mix(self.patch(pixel_values).flatten(2).transpose(1, 2))
        scale = self.get_buffer("scale") * self.get_buffer("scale_alias")
        return tokens * self.gain * self.gain_alias * scale


def _build_conv(path: str = "", **_: Any) -> VisionEncoder:
    return _ConvEncoder()


@pytest.fixture
def conv_encoder() -> Iterator[str]:
    registry.register("vision_encoder", CONV, _build_conv)
    yield CONV
    registry._get_store("vision_encoder").pop(CONV)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _save_tiny_siglip(path: Path, dtype: torch.dtype) -> str:
    from transformers import SiglipVisionConfig, SiglipVisionModel

    torch.manual_seed(0)
    SiglipVisionModel(
        SiglipVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            image_size=IMAGE_SIZE,
            patch_size=PATCH,
            vision_use_head=False,
        )
    ).to(dtype).save_pretrained(path)
    return str(path)


@pytest.fixture(scope="module")
def siglip_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    return _save_tiny_siglip(tmp_path_factory.mktemp("tiny_siglip"), torch.float32)


@pytest.fixture(scope="module")
def siglip_bf16_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    return _save_tiny_siglip(tmp_path_factory.mktemp("tiny_siglip_bf16"), torch.bfloat16)


def _siglip(siglip_dir: str, precision: Precision | None) -> VisionEncoderConfig:
    return VisionEncoderConfig(type="siglip2", path=siglip_dir, mixed_precision=precision)


def _random(precision: Precision | None) -> VisionEncoderConfig:
    return VisionEncoderConfig(
        type="random",
        feature_dim=24,
        num_tokens=(IMAGE_SIZE // PATCH) ** 2,
        mixed_precision=precision,
    )


def _build(
    vision_config: VisionEncoderConfig,
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
        vision_config=vision_config,
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


def _recorder(seen: Seen, name: str, *, weight: bool):
    def hook(module: nn.Module, args: tuple, output: torch.Tensor) -> None:
        first = module.weight.dtype if weight else args[0].dtype
        seen.setdefault(name, (first, args[0].dtype if weight else output.dtype))

    return hook


def _forward_backward(wrapper: VLMWrapper, frames: int) -> tuple[torch.Tensor, Seen]:
    """Run one step and record dtypes.

    ``encoder`` is the encoder's (input, output) dtypes; ``encoder_linear`` and
    ``transformer`` are their first Linear's (weight, input) dtypes in the forward.
    """
    seen: Seen = {}
    handles = [
        wrapper.vision_encoder.register_forward_hook(_recorder(seen, "encoder", weight=False))
    ]
    for name, tower in (
        ("encoder_linear", wrapper.vision_encoder),
        ("transformer", wrapper.transformer),
    ):
        linear = next((m for m in tower.modules() if isinstance(m, nn.Linear)), None)
        if linear is not None:
            handles.append(linear.register_forward_hook(_recorder(seen, name, weight=True)))
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


def _float_buffer_dtypes(module: nn.Module) -> set[torch.dtype]:
    return {
        b.dtype for _, b in module.named_buffers(remove_duplicate=False) if b.is_floating_point()
    }


def _holds(module: nn.Module, stored: dict[str, nn.Parameter]) -> bool:
    """Whether ``module`` holds exactly the ``stored`` parameter objects under their names."""
    current = dict(module.named_parameters())
    return current.keys() == stored.keys() and all(current[n] is p for n, p in stored.items())


FRAMES = pytest.mark.parametrize("frames", [1, 2], ids=["image", "video"])
PRECISION_VS_LM = pytest.mark.parametrize(
    ("precision", "lm_dtype"), [("fp32", torch.bfloat16), ("bf16", torch.float32)]
)


@FRAMES
@PRECISION_VS_LM
def test_unwrapped_encoder_computes_in_its_precision(
    siglip_dir: str, precision: Precision, lm_dtype: torch.dtype, frames: int
) -> None:
    """Without FSDP2 the encoder computes in its own dtype; the language model keeps its own.

    The encoder's parameters and gradients stay fp32, the dtype this checkpoint loads in,
    and after the forward it holds the same parameter objects as before.
    """
    enc_dtype = DTYPES[precision]
    wrapper = _build(_siglip(siglip_dir, precision), lm_dtype, frames)
    stored = dict(wrapper.vision_encoder.named_parameters())
    assert {p.dtype for p in stored.values()} == {torch.float32}
    assert {p.dtype for p in wrapper.transformer.parameters()} == {lm_dtype}
    assert {p.dtype for p in wrapper.adapter.parameters()} == {lm_dtype}

    logits, seen = _forward_backward(wrapper, frames)
    assert seen == {
        "encoder": (enc_dtype, enc_dtype),
        "encoder_linear": (enc_dtype, enc_dtype),
        "transformer": (lm_dtype, lm_dtype),
    }
    assert logits.dtype == lm_dtype
    assert _holds(wrapper.vision_encoder, stored)
    for grad in _encoder_grads(wrapper):
        assert grad.dtype == torch.float32
        assert torch.isfinite(grad).all()


@FRAMES
@PRECISION_VS_LM
def test_unwrapped_encoder_gets_inputs_and_buffers_in_its_precision(
    conv_encoder: str, precision: Precision, lm_dtype: torch.dtype, frames: int
) -> None:
    """An encoder that casts nothing itself still runs: its inputs and buffers take its dtype.

    In the forward a parameter with two names is one tensor in that dtype and a missing bias
    stays missing; afterwards the parameter is back under both names, still fp32.
    """
    enc_dtype = DTYPES[precision]
    vision = VisionEncoderConfig(type=conv_encoder, mixed_precision=precision)
    wrapper = _build(vision, lm_dtype, frames)
    encoder = wrapper.vision_encoder
    assert {p.dtype for p in encoder.parameters()} == {torch.float32}
    assert _float_buffer_dtypes(encoder) == {enc_dtype}

    inside: list[tuple[bool, torch.dtype, object]] = []
    handle = encoder.register_forward_pre_hook(
        lambda module, args: inside.append(
            (
                module.gain is module.gain_alias,
                module.gain.dtype,
                module.mix.bias,
            )
        )
    )
    logits, seen = _forward_backward(wrapper, frames)
    handle.remove()
    assert inside == [(True, enc_dtype, None)]
    assert seen["encoder"] == (enc_dtype, enc_dtype)
    assert seen["encoder_linear"] == (enc_dtype, enc_dtype)
    assert logits.dtype == lm_dtype
    assert encoder.gain is encoder.gain_alias
    for grad in _encoder_grads(wrapper):
        assert grad.dtype == torch.float32


@FRAMES
def test_frozen_encoder_computes_in_its_precision(siglip_dir: str, frames: int) -> None:
    """A frozen encoder, which FSDP2 never wraps, keeps its loaded parameters too."""
    wrapper = _build(_siglip(siglip_dir, "bf16"), torch.float32, frames, frozen=True)
    assert not wrapper.vision_encoder.training
    assert {p.dtype for p in wrapper.vision_encoder.parameters()} == {torch.float32}
    _, seen = _forward_backward(wrapper, frames)
    assert seen["encoder_linear"] == (torch.bfloat16, torch.bfloat16)
    assert all(p.grad is None for p in wrapper.vision_encoder.parameters())


def test_unwrapped_encoder_gets_its_parameters_back_when_the_forward_raises(
    siglip_dir: str,
) -> None:
    wrapper = _build(_siglip(siglip_dir, "bf16"), torch.float32, 1)
    encoder = wrapper.vision_encoder
    stored = dict(encoder.named_parameters())
    with pytest.raises(RuntimeError):
        encoder(torch.randn(2, 5, IMAGE_SIZE, IMAGE_SIZE))
    assert _holds(encoder, stored)
    _, seen = _forward_backward(wrapper, 1)
    assert seen["encoder_linear"] == (torch.bfloat16, torch.bfloat16)
    assert _holds(encoder, stored)


@FRAMES
def test_unset_precision_keeps_the_unwrapped_encoder_as_loaded(
    siglip_dir: str, frames: int
) -> None:
    """Unset is the default build: the encoder stays fp32 as loaded, under a bf16 language model.

    That equals pinning ``"fp32"`` for this fp32 checkpoint, so the two runs match bit for bit.
    """
    default = _build(_siglip(siglip_dir, None), torch.bfloat16, frames)
    assert {p.dtype for p in default.vision_encoder.parameters()} == {torch.float32}
    default_logits, default_seen = _forward_backward(default, frames)
    assert default_seen["encoder_linear"] == (torch.float32, torch.float32)

    pinned = _build(_siglip(siglip_dir, "fp32"), torch.bfloat16, frames)
    pinned_logits, _ = _forward_backward(pinned, frames)
    assert torch.equal(default_logits, pinned_logits)
    for a, b in zip(_encoder_grads(default), _encoder_grads(pinned), strict=True):
        assert torch.equal(a, b)


def test_input_cast_hook_touches_only_floating_tensors() -> None:
    from kempnerforge.distributed.parallel import _cast_floating_inputs

    ids = torch.ones(2, dtype=torch.long)
    args, kwargs = _cast_floating_inputs(
        torch.bfloat16,
        nn.Identity(),
        (torch.ones(2), ids, "frames"),
        {"mask": torch.ones(2), "n": 3},
    )
    assert args[0].dtype == torch.bfloat16
    assert args[1] is ids
    assert args[2] == "frames"
    assert kwargs["mask"].dtype == torch.bfloat16
    assert kwargs["n"] == 3


def test_unset_precision_leaves_an_encoder_that_casts_nothing_alone(conv_encoder: str) -> None:
    wrapper = _build(VisionEncoderConfig(type=conv_encoder), torch.bfloat16, 1)
    assert {p.dtype for p in wrapper.vision_encoder.parameters()} == {torch.float32}
    assert _float_buffer_dtypes(wrapper.vision_encoder) == {torch.float32}
    _, seen = _forward_backward(wrapper, 1)
    assert seen["encoder"] == (torch.float32, torch.float32)


def _check_fsdp_precision(rank: int, world: int, port: int, siglip_dir: str) -> None:
    dist.init_process_group(
        "gloo", rank=rank, world_size=world, init_method=f"tcp://127.0.0.1:{port}"
    )
    try:
        registry.register("vision_encoder", CONV, _build_conv)
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp_shard",))
        for precision, lm_dtype in (("fp32", torch.bfloat16), ("bf16", torch.float32)):
            enc_dtype = DTYPES[precision]
            for frames in (1, 2):
                wrapper = _build(_siglip(siglip_dir, precision), lm_dtype, frames, mesh=mesh)
                assert isinstance(wrapper.vision_encoder, FSDPModule)
                assert {p.dtype for p in wrapper.vision_encoder.parameters()} == {torch.float32}
                assert {p.dtype for p in wrapper.transformer.parameters()} == {lm_dtype}
                logits, seen = _forward_backward(wrapper, frames)
                assert seen == {
                    "encoder": (enc_dtype, enc_dtype),
                    "encoder_linear": (enc_dtype, enc_dtype),
                    "transformer": (lm_dtype, lm_dtype),
                }
                assert logits.dtype == lm_dtype
                for grad in _encoder_grads(wrapper):
                    assert grad.dtype == torch.float32
                    assert torch.isfinite(grad).all()

            for vision in (
                _random(precision),
                VisionEncoderConfig(type=CONV, mixed_precision=precision),
            ):
                wrapper = _build(vision, lm_dtype, 2, mesh=mesh)
                assert isinstance(wrapper.vision_encoder, FSDPModule)
                assert _float_buffer_dtypes(wrapper.vision_encoder) == {enc_dtype}
                assert {p.dtype for p in wrapper.vision_encoder.parameters()} <= {torch.float32}
                _, seen = _forward_backward(wrapper, 2)
                assert seen["encoder"] == (enc_dtype, enc_dtype)

        frozen = _build(_siglip(siglip_dir, "bf16"), torch.float32, 2, mesh=mesh, frozen=True)
        assert not isinstance(frozen.vision_encoder, FSDPModule)
        assert {p.dtype for p in frozen.vision_encoder.parameters()} == {torch.float32}
        _, seen = _forward_backward(frozen, 2)
        assert seen["encoder_linear"] == (torch.bfloat16, torch.bfloat16)
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
            default = _build(_siglip(siglip_dir, None), lm_dtype, 1, mesh=mesh)
            default_logits, default_seen = _forward_backward(default, 1)
            assert default_seen["encoder_linear"] == (lm_dtype, lm_dtype)

            pinned = _build(_siglip(siglip_dir, same_precision), lm_dtype, 1, mesh=mesh)
            pinned_logits, _ = _forward_backward(pinned, 1)
            assert torch.equal(default_logits, pinned_logits)
            for a, b in zip(_encoder_grads(default), _encoder_grads(pinned), strict=True):
                assert torch.equal(a, b)

        default_random = _build(_random(None), torch.bfloat16, 1, mesh=mesh)
        assert _float_buffer_dtypes(default_random.vision_encoder) == {torch.float32}
        _, seen = _forward_backward(default_random, 1)
        assert seen["encoder"] == (torch.bfloat16, torch.float32)
    finally:
        dist.destroy_process_group()


def test_fsdp_encoder_computes_in_its_precision(siglip_dir: str) -> None:
    """Under FSDP2 the encoder's own policy sets its compute dtype for image and video runs.

    Its parameter shards and their gradients stay in the loaded fp32 while its floating-point
    buffers take the set dtype, so encoders without parameters follow it too; the language
    model computes in ``param_dtype``. A frozen encoder is not wrapped, so it is cast instead.
    """
    mp.spawn(_check_fsdp_precision, args=(2, _free_port(), siglip_dir), nprocs=2)


def test_fsdp_unset_precision_follows_the_language_model(siglip_dir: str) -> None:
    """Unset keeps the default wrap: the encoder computes in the language model's dtype.

    Pinning that same dtype gives the same policy, so logits and gradients match bit for bit,
    and an unset encoder's buffers keep their loaded dtype.
    """
    mp.spawn(
        _check_fsdp_unset_follows_the_language_model,
        args=(2, _free_port(), siglip_dir),
        nprocs=2,
    )


def _train_step_stats(wrapper: VLMWrapper) -> dict[str, Any]:
    """One AdamW step at ``LR`` on a fixed batch, summarized for the encoder.

    Gives the dtypes of its parameters, gradients and AdamW state, the (weight, input)
    dtypes its first Linear computes with, and the share of its parameter elements the step
    changed, counted over every rank so a sharded encoder reports the whole of it.
    """
    params = list(wrapper.vision_encoder.parameters())
    before = [_local(p).detach().clone() for p in params]
    _, seen = _forward_backward(wrapper, 1)
    trainable = [p for p in wrapper.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.0)
    optimizer.step()
    counts = torch.tensor(
        [
            sum(int((_local(p) != b).sum()) for p, b in zip(params, before, strict=True)),
            sum(b.numel() for b in before),
        ]
    )
    if dist.is_initialized():
        dist.all_reduce(counts)
    states = [optimizer.state[p][key] for p in params for key in ("exp_avg", "exp_avg_sq")]
    return {
        "param": sorted({str(p.dtype) for p in params}),
        "grad": sorted({str(g.dtype) for g in _encoder_grads(wrapper)}),
        "state": sorted({str(s.dtype) for s in states}),
        "compute": [str(d) for d in seen["encoder_linear"]],
        "moved": counts[0].item() / counts[1].item(),
    }


def _check_fsdp_train_step(
    rank: int, world: int, port: int, dirs: dict[str, str], out: str
) -> None:
    dist.init_process_group(
        "gloo", rank=rank, world_size=world, init_method=f"tcp://127.0.0.1:{port}"
    )
    try:
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp_shard",))
        stats = {}
        for loaded, path in dirs.items():
            for precision in DTYPES:
                wrapper = _build(_siglip(path, precision), torch.bfloat16, 1, mesh=mesh)
                assert isinstance(wrapper.vision_encoder, FSDPModule)
                stats[f"{loaded}/{precision}"] = _train_step_stats(wrapper)
        if rank == 0:
            Path(out).write_text(json.dumps(stats))
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def fsdp_train_step(
    siglip_dir: str, siglip_bf16_dir: str, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, dict[str, Any]]:
    out = tmp_path_factory.mktemp("fsdp_train_step") / "stats.json"
    dirs = {"fp32": siglip_dir, "bf16": siglip_bf16_dir}
    mp.spawn(_check_fsdp_train_step, args=(2, _free_port(), dirs, str(out)), nprocs=2)
    return json.loads(out.read_text())


@pytest.mark.parametrize("loaded", ["fp32", "bf16"])
@pytest.mark.parametrize("precision", ["bf16", "fp32"])
def test_one_process_trains_the_encoder_as_fsdp2_does(
    fsdp_train_step: dict[str, dict[str, Any]],
    siglip_dir: str,
    siglip_bf16_dir: str,
    loaded: str,
    precision: Precision,
) -> None:
    """The setting decides the compute dtype only, on one process as under FSDP2.

    Built without a mesh and on two FSDP2 ranks, an encoder loaded in fp32 or in bf16 keeps
    that dtype for its parameters, gradients and AdamW state, computes in the set dtype, and
    one AdamW step at a learning rate of 5e-5 changes the same share of its weights.
    """
    path = {"fp32": siglip_dir, "bf16": siglip_bf16_dir}[loaded]
    single = _train_step_stats(_build(_siglip(path, precision), torch.bfloat16, 1))
    sharded = fsdp_train_step[f"{loaded}/{precision}"]
    stored = [str(DTYPES[loaded])]
    assert single["param"] == sharded["param"] == stored
    assert single["grad"] == sharded["grad"] == stored
    assert single["state"] == sharded["state"] == stored
    assert single["compute"] == sharded["compute"] == [str(DTYPES[precision])] * 2
    assert single["moved"] == pytest.approx(sharded["moved"], abs=0.01)
