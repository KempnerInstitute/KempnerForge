"""``vision_encoder.mixed_precision`` in the VLM build, for image and video-shaped runs.

A tiny SigLIP vision tower saved to disk stands in for a pretrained encoder, so the
``siglip2`` builder loads real Hugging Face weights offline, in fp32. A plain Conv2d
encoder, loaded in fp32 or in bf16 and with or without a floating-point buffer, and the
parameter-free ``random`` encoder cover encoders that cast nothing themselves. Image runs
feed ``(B, 3, H, W)`` pixels; video-shaped runs build with ``frames_per_clip = 2`` and feed
``(B, 2, 3, H, W)``. The FSDP2 cases shard over two gloo ranks on CPU, all in one spawn.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch._dynamo.utils import counters
from torch.distributed._composable.fsdp import FSDPModule
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from kempnerforge.config.adapter import AdapterConfig
from kempnerforge.config.model import ModelConfig
from kempnerforge.config.registry import registry
from kempnerforge.config.vision import VisionEncoderConfig
from kempnerforge.config.vlm import FreezeSpec, VLMConfig
from kempnerforge.distributed.parallel import (
    _cast_floating,
    build_parallel_model,
    default_mp_policy,
)
from kempnerforge.model.vision import VisionEncoder
from kempnerforge.model.vlm import VLMWrapper
from tests.gloo_ranks import gloo_group, spawn

IMAGE_SIZE = 32
PATCH = 8
TEXT_LEN = 16
CONV = "test_precision_conv"
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
LR = 5e-5

Precision = Literal["bf16", "fp32"]
Seen = dict[str, tuple[torch.dtype, ...]]


class _ConvEncoder(VisionEncoder):
    """A Conv2d patch encoder that casts nothing itself.

    It has a bias-free Linear and one parameter under two names, and with ``buffers`` one
    floating-point buffer under two names.
    """

    def __init__(self, buffers: bool) -> None:
        super().__init__()
        self.num_tokens = (IMAGE_SIZE // PATCH) ** 2
        self.feature_dim = 24
        self.patch = nn.Conv2d(3, self.feature_dim, kernel_size=PATCH, stride=PATCH)
        self.mix = nn.Linear(self.feature_dim, self.feature_dim, bias=False)
        self.gain = nn.Parameter(torch.ones(1))
        self.register_parameter("gain_alias", self.gain)
        if buffers:
            scale = torch.ones(1)
            self.register_buffer("scale", scale)
            self.register_buffer("scale_alias", scale)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        tokens = self.mix(self.patch(pixel_values).flatten(2).transpose(1, 2))
        tokens = tokens * self.gain * self.gain_alias
        for name in ("scale", "scale_alias"):
            if name in self._buffers:
                tokens = tokens * self.get_buffer(name)
        return tokens


def _build_conv(path: str = "", **_: Any) -> VisionEncoder:
    """``path`` picks the variant: ``"bf16"`` loads it in bf16, ``"plain"`` drops the buffer."""
    encoder = _ConvEncoder(buffers=path != "plain")
    return encoder.to(torch.bfloat16) if path == "bf16" else encoder


@pytest.fixture
def conv_encoder() -> Iterator[str]:
    registry.register("vision_encoder", CONV, _build_conv)
    yield CONV
    registry._get_store("vision_encoder").pop(CONV)


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
            patch_size=PATCH,
            vision_use_head=False,
        )
    ).save_pretrained(path)
    return str(path)


def _siglip(siglip_dir: str, precision: Precision | None) -> VisionEncoderConfig:
    return VisionEncoderConfig(type="siglip2", path=siglip_dir, mixed_precision=precision)


def _conv(precision: Precision | None, variant: str = "") -> VisionEncoderConfig:
    return VisionEncoderConfig(type=CONV, path=variant, mixed_precision=precision)


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


def _recorder(seen: Seen, name: str):
    def hook(module: nn.Module, args: tuple, output: torch.Tensor) -> None:
        seen.setdefault(name, (module.weight.dtype, args[0].dtype, output.dtype))

    return hook


def _first(tower: nn.Module, kind: type[nn.Module]) -> nn.Module | None:
    return next((m for m in tower.modules() if isinstance(m, kind)), None)


def _forward_backward(
    wrapper: nn.Module, frames: int, autocast: torch.dtype | None = None
) -> tuple[torch.Tensor, Seen]:
    """Run one forward, under ``torch.autocast(autocast)`` if given, and a backward outside it.

    ``encoder_conv``, ``encoder_linear`` and ``transformer`` are the (weight, input, output)
    dtypes of the first Conv2d and Linear of the encoder and the first Linear of the
    language model in the forward; ``encoder_out`` is the encoder's output dtype.
    """
    seen: Seen = {}
    encoder = wrapper.vision_encoder

    def record_output(module: nn.Module, args: tuple, output: torch.Tensor) -> None:
        seen.setdefault("encoder_out", (output.dtype,))

    handles = [encoder.register_forward_hook(record_output)]
    for name, tower, kind in (
        ("encoder_conv", encoder, nn.Conv2d),
        ("encoder_linear", encoder, nn.Linear),
        ("transformer", wrapper.transformer, nn.Linear),
    ):
        module = _first(tower, kind)
        if module is not None:
            handles.append(module.register_forward_hook(_recorder(seen, name)))
    pixels, ids = _batch(frames)
    with torch.autocast("cpu", dtype=autocast or torch.bfloat16, enabled=autocast is not None):
        logits, _ = wrapper(pixels, ids, ids)
    logits.float().mean().backward()
    for handle in handles:
        handle.remove()
    return logits.detach(), seen


def _local(t: torch.Tensor) -> torch.Tensor:
    return t.to_local() if isinstance(t, DTensor) else t


def _encoder_grads(wrapper: nn.Module) -> list[torch.Tensor]:
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


def _names(dtypes) -> list[str]:
    return [str(d) for d in dtypes]


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
        "encoder_out": (enc_dtype,),
        "encoder_conv": (enc_dtype,) * 3,
        "encoder_linear": (enc_dtype,) * 3,
        "transformer": (lm_dtype,) * 3,
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
    stays missing; afterwards the parameter is back under both names, still fp32. The
    buffer is stored in that dtype, so a checkpoint saves it in it.
    """
    enc_dtype = DTYPES[precision]
    wrapper = _build(_conv(precision), lm_dtype, frames)
    encoder = wrapper.vision_encoder
    assert {p.dtype for p in encoder.parameters()} == {torch.float32}
    assert _float_buffer_dtypes(encoder) == {enc_dtype}
    assert wrapper.state_dict()["vision_encoder.scale"].dtype == enc_dtype

    inside: list[tuple[bool, torch.dtype, object]] = []
    handle = encoder.mix.register_forward_pre_hook(
        lambda module, args: inside.append(
            (encoder.gain is encoder.gain_alias, encoder.gain.dtype, module.bias)
        )
    )
    logits, seen = _forward_backward(wrapper, frames)
    handle.remove()
    assert inside == [(True, enc_dtype, None)]
    assert seen["encoder_conv"] == (enc_dtype,) * 3
    assert seen["encoder_linear"] == (enc_dtype,) * 3
    assert seen["encoder_out"] == (enc_dtype,)
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
    assert seen["encoder_linear"] == (torch.bfloat16,) * 3
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
    assert seen["encoder_linear"] == (torch.bfloat16,) * 3
    assert _holds(encoder, stored)


def test_nested_call_leaves_the_outer_forward_in_its_precision(conv_encoder: str) -> None:
    """A call of the encoder inside its own forward puts back none of the outer call's copies.

    The nested call finds the parameters already cast, so the rest of the outer forward still
    computes in the set dtype, and the parameters return once the outer call ends.
    """
    wrapper = _build(_conv("bf16"), torch.float32, 1)
    encoder = wrapper.vision_encoder
    stored = dict(encoder.named_parameters())
    pixels, _ = _batch(1)
    nested: list[torch.dtype] = []
    mix_calls: list[tuple[torch.dtype, ...]] = []

    def call_again(module: nn.Module, args: tuple, output: torch.Tensor) -> None:
        if not nested:
            nested.append(torch.float64)
            nested.append(encoder(pixels).dtype)

    handles = [
        encoder.patch.register_forward_hook(call_again),
        encoder.mix.register_forward_hook(
            lambda module, args, output: mix_calls.append(
                (module.weight.dtype, args[0].dtype, output.dtype)
            )
        ),
    ]
    out = encoder(pixels)
    for handle in handles:
        handle.remove()
    assert nested == [torch.float64, torch.bfloat16]
    assert mix_calls == [(torch.bfloat16,) * 3] * 2
    assert out.dtype == torch.bfloat16
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
    assert default_seen["encoder_linear"] == (torch.float32,) * 3

    pinned = _build(_siglip(siglip_dir, "fp32"), torch.bfloat16, frames)
    pinned_logits, _ = _forward_backward(pinned, frames)
    assert torch.equal(default_logits, pinned_logits)
    for a, b in zip(_encoder_grads(default), _encoder_grads(pinned), strict=True):
        assert torch.equal(a, b)


def test_cast_floating_casts_only_floating_tensors() -> None:
    ids = torch.ones(2, dtype=torch.long)
    cast = _cast_floating(torch.ones(2), torch.bfloat16)
    assert isinstance(cast, torch.Tensor) and cast.dtype == torch.bfloat16
    assert _cast_floating(ids, torch.bfloat16) is ids
    assert _cast_floating("frames", torch.bfloat16) == "frames"


def test_unwrapped_encoder_casts_keyword_inputs(conv_encoder: str) -> None:
    encoder = _build(_conv("bf16"), torch.float32, 1).vision_encoder
    seen: Seen = {}
    conv = _first(encoder, nn.Conv2d)
    assert conv is not None
    handle = conv.register_forward_hook(_recorder(seen, "conv"))
    out = encoder(pixel_values=_batch(1)[0])
    handle.remove()
    assert seen["conv"] == (torch.bfloat16,) * 3
    assert out.dtype == torch.bfloat16


def test_unset_precision_leaves_an_encoder_that_casts_nothing_alone(conv_encoder: str) -> None:
    wrapper = _build(_conv(None), torch.bfloat16, 1)
    assert {p.dtype for p in wrapper.vision_encoder.parameters()} == {torch.float32}
    assert _float_buffer_dtypes(wrapper.vision_encoder) == {torch.float32}
    _, seen = _forward_backward(wrapper, 1)
    assert seen["encoder_conv"] == (torch.float32,) * 3
    assert seen["encoder_out"] == (torch.float32,)


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_set_precision_keeps_the_encoder_out_of_the_language_models_autocast(
    siglip_dir: str, precision: Precision
) -> None:
    """With fp32 master weights and no FSDP2 wrap the forward runs under ``torch.autocast``.

    The language model's matmuls then compute in bf16 from its fp32 weights. The encoder
    computes in its set dtype, so an ``"fp32"`` encoder runs no matmul in bf16, and its
    output and gradients match those it gives without the autocast bit for bit.
    """
    enc_dtype = DTYPES[precision]
    wrapper = _build(_siglip(siglip_dir, precision), torch.float32, 1)
    _, seen = _forward_backward(wrapper, 1, autocast=torch.bfloat16)
    assert seen["encoder_conv"] == (enc_dtype,) * 3
    assert seen["encoder_linear"] == (enc_dtype,) * 3
    assert seen["encoder_out"] == (enc_dtype,)
    assert seen["transformer"] == (torch.float32, torch.float32, torch.bfloat16)

    encoder = wrapper.vision_encoder
    pixels, _ = _batch(1)
    runs = []
    for autocast in (False, True):
        encoder.zero_grad()
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            out = encoder(pixels)
        out.float().square().mean().backward()
        runs.append((out.detach(), [g.clone() for g in _encoder_grads(wrapper)]))
    assert torch.equal(runs[0][0], runs[1][0])
    for a, b in zip(runs[0][1], runs[1][1], strict=True):
        assert a.dtype == torch.float32
        assert torch.equal(a, b)


def test_unset_precision_leaves_the_encoder_under_the_language_models_autocast(
    siglip_dir: str,
) -> None:
    """Unset, the encoder follows the language model: the autocast reaches its matmuls."""
    wrapper = _build(_siglip(siglip_dir, None), torch.float32, 1)
    _, seen = _forward_backward(wrapper, 1, autocast=torch.bfloat16)
    assert seen["encoder_linear"][0] == torch.float32
    assert seen["encoder_linear"][2] == torch.bfloat16


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_compiled_wrapper_matches_eager_under_autocast_without_recompiling(
    siglip_dir: str, precision: Precision
) -> None:
    """Under ``torch.compile`` the set precision holds in the forward and in the backward.

    The encoder's cast runs at a graph break, the only one, and its forward compiles as its
    own graph with autocast off, so over three training steps under the language model's
    autocast the compiled model's logits and encoder gradients match eager bit for bit, and
    no step after the first compiles a graph.
    """
    eager = _build(_siglip(siglip_dir, precision), torch.float32, 1)
    model = _build(_siglip(siglip_dir, precision), torch.float32, 1)
    stored = dict(model.vision_encoder.named_parameters())
    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(model, backend="aot_eager")
    optimizers = [torch.optim.SGD(m.parameters(), lr=0.1) for m in (eager, model)]
    pixels, ids = _batch(1)
    graphs = []
    for _ in range(3):
        outputs = []
        for run in (eager, compiled):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                logits, _ = run(pixels, ids, ids)
            logits.float().mean().backward()
            outputs.append(logits.detach())
        assert torch.equal(outputs[0], outputs[1])
        for a, b in zip(_encoder_grads(eager), _encoder_grads(model), strict=True):
            assert a.dtype == b.dtype == torch.float32
            assert torch.equal(a, b)
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad()
        graphs.append(counters["stats"]["unique_graphs"])
    breaks = list(counters["graph_break"])
    torch._dynamo.reset()
    assert graphs[0] > 0
    assert graphs == [graphs[0]] * 3
    assert breaks
    assert all("_ComputeInDtype._run" in reason for reason in breaks)
    assert _holds(model.vision_encoder, stored)


def _step_summary(wrapper: VLMWrapper, frames: int) -> dict[str, Any]:
    """One forward and backward, then one AdamW step at ``LR``, summarized for the encoder.

    Gives whether FSDP2 wrapped it, the dtypes of its parameters, gradients, AdamW state and
    floating-point buffers, the dtypes ``_forward_backward`` records, the logits' dtype,
    whether its gradients are finite, and the share of its parameter elements the step
    changed, counted over every rank so a sharded encoder reports the whole of it.
    """
    params = list(wrapper.vision_encoder.parameters())
    before = [_local(p).detach().clone() for p in params]
    logits, seen = _forward_backward(wrapper, frames)
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
    grads = _encoder_grads(wrapper)
    states = [optimizer.state[p][key] for p in params for key in ("exp_avg", "exp_avg_sq")]
    return {
        "wrapped": isinstance(wrapper.vision_encoder, FSDPModule),
        "param": sorted({str(p.dtype) for p in params}),
        "grad": sorted({str(g.dtype) for g in grads}),
        "state": sorted({str(s.dtype) for s in states}),
        "buffers": sorted({str(d) for d in _float_buffer_dtypes(wrapper.vision_encoder)}),
        "transformer_param": sorted({str(p.dtype) for p in wrapper.transformer.parameters()}),
        "seen": {name: _names(dtypes) for name, dtypes in seen.items()},
        "logits": str(logits.dtype),
        "finite": all(bool(torch.isfinite(g).all()) for g in grads),
        "moved": counts[0].item() / counts[1].item(),
    }


TRAINED = (("fp32", "bf16", torch.float32), ("bf16", "fp32", torch.bfloat16))


def _parameter_free_summary(wrapper: VLMWrapper) -> dict[str, Any]:
    """Whether FSDP2 wrapped the encoder, its floating-point buffer dtypes, and its I/O dtypes."""
    inputs: list[torch.dtype] = []
    handle = wrapper.vision_encoder.register_forward_hook(
        lambda module, args, output: inputs.append(args[0].dtype)
    )
    _, seen = _forward_backward(wrapper, 1)
    handle.remove()
    return {
        "wrapped": isinstance(wrapper.vision_encoder, FSDPModule),
        "buffers": sorted(str(d) for d in _float_buffer_dtypes(wrapper.vision_encoder)),
        "input": _names(inputs),
        "output": _names(seen["encoder_out"]),
    }


def _check_fsdp(rank: int, world: int, port: int, out: str) -> None:
    """Build every FSDP2 case on two gloo ranks and write what rank 0 sees to ``out``."""
    with gloo_group(rank, world, port):
        registry.register("vision_encoder", CONV, _build_conv)
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp_shard",))
        results: dict[str, Any] = {}
        for loaded, precision, lm_dtype in TRAINED:
            variant = "bf16" if loaded == "bf16" else ""
            wrapper = _build(_conv(precision, variant), lm_dtype, 2, mesh=mesh)
            results[f"trained/{loaded}/{precision}"] = _step_summary(wrapper, 2)

        default = _build(_conv(None, "plain"), torch.bfloat16, 1, mesh=mesh)
        default_logits, default_seen = _forward_backward(default, 1)
        pinned = _build(_conv("bf16", "plain"), torch.bfloat16, 1, mesh=mesh)
        pinned_logits, _ = _forward_backward(pinned, 1)
        results["unset"] = {
            "wrapped": isinstance(default.vision_encoder, FSDPModule),
            "seen": {name: _names(dtypes) for name, dtypes in default_seen.items()},
            "logits_equal": torch.equal(default_logits, pinned_logits),
            "grads_equal": all(
                torch.equal(a, b)
                for a, b in zip(_encoder_grads(default), _encoder_grads(pinned), strict=True)
            ),
        }

        for precision, lm_dtype in ((None, torch.bfloat16), ("bf16", torch.float32)):
            random = _build(_random(precision), lm_dtype, 1, mesh=mesh)
            results[f"random/{precision}"] = _parameter_free_summary(random)

        frozen = _build(_conv("bf16"), torch.float32, 2, mesh=mesh, frozen=True)
        _, seen = _forward_backward(frozen, 2)
        results["frozen"] = {
            "wrapped": isinstance(frozen.vision_encoder, FSDPModule),
            "param": sorted({str(p.dtype) for p in frozen.vision_encoder.parameters()}),
            "seen": {name: _names(dtypes) for name, dtypes in seen.items()},
            "no_grads": all(p.grad is None for p in frozen.vision_encoder.parameters()),
        }
        if rank == 0:
            Path(out).write_text(json.dumps(results))


@pytest.fixture(scope="module")
def fsdp_results(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    out = tmp_path_factory.mktemp("fsdp_precision") / "results.json"
    spawn(_check_fsdp, 2, str(out))
    return json.loads(out.read_text())


def test_fsdp_encoder_computes_in_its_precision(fsdp_results: dict[str, Any]) -> None:
    """Under FSDP2 the encoder's own policy sets its compute dtype on video-shaped runs.

    Its parameter shards and their gradients stay in the dtype it was loaded in while its
    floating-point buffers take the set dtype, so encoders without parameters follow it
    too; the language model computes in ``param_dtype``. A frozen encoder is not wrapped,
    so it is cast instead.
    """
    for loaded, precision, lm_dtype in TRAINED:
        enc, lm = str(DTYPES[precision]), str(lm_dtype)
        result = fsdp_results[f"trained/{loaded}/{precision}"]
        assert result["wrapped"]
        assert result["param"] == result["grad"] == [str(DTYPES[loaded])]
        assert result["buffers"] == [enc]
        assert result["transformer_param"] == [lm]
        assert result["seen"] == {
            "encoder_out": [enc],
            "encoder_conv": [enc] * 3,
            "encoder_linear": [enc] * 3,
            "transformer": [lm] * 3,
        }
        assert result["logits"] == lm
        assert result["finite"]

    random = fsdp_results["random/bf16"]
    assert random["wrapped"]
    assert random["buffers"] == random["output"] == ["torch.bfloat16"]

    frozen = fsdp_results["frozen"]
    assert not frozen["wrapped"]
    assert frozen["param"] == ["torch.float32"]
    assert frozen["seen"]["encoder_linear"] == ["torch.bfloat16"] * 3
    assert frozen["no_grads"]


def test_fsdp_unset_precision_follows_the_language_model(fsdp_results: dict[str, Any]) -> None:
    """Unset keeps the default wrap: the encoder computes in the language model's dtype.

    Pinning that same dtype on an encoder without floating-point buffers gives the same
    policy, so logits and gradients match bit for bit, and an unset encoder's buffers keep
    their loaded dtype while its inputs take the language model's.
    """
    unset = fsdp_results["unset"]
    assert unset["wrapped"]
    assert unset["seen"]["encoder_linear"] == ["torch.bfloat16"] * 3
    assert unset["logits_equal"]
    assert unset["grads_equal"]

    random = fsdp_results["random/None"]
    assert random["wrapped"]
    assert random["buffers"] == ["torch.float32"]
    assert random["input"] == ["torch.bfloat16"]
    assert random["output"] == ["torch.float32"]


@pytest.mark.parametrize(("loaded", "precision", "lm_dtype"), TRAINED)
def test_one_process_trains_the_encoder_as_fsdp2_does(
    fsdp_results: dict[str, Any],
    conv_encoder: str,
    loaded: str,
    precision: Precision,
    lm_dtype: torch.dtype,
) -> None:
    """The setting decides the compute dtype only, on one process as under FSDP2.

    Built without a mesh and on two FSDP2 ranks, an encoder loaded in fp32 or in bf16 keeps
    that dtype for its parameters, gradients and AdamW state, computes in the set dtype, and
    one AdamW step at a learning rate of 5e-5 changes the same share of its weights.
    """
    variant = "bf16" if loaded == "bf16" else ""
    single = _step_summary(_build(_conv(precision, variant), lm_dtype, 2), 2)
    sharded = fsdp_results[f"trained/{loaded}/{precision}"]
    stored = [str(DTYPES[loaded])]
    assert not single["wrapped"]
    assert single["param"] == sharded["param"] == stored
    assert single["grad"] == sharded["grad"] == stored
    assert single["state"] == sharded["state"] == stored
    assert single["buffers"] == sharded["buffers"]
    assert single["seen"] == sharded["seen"]
    assert single["moved"] == pytest.approx(sharded["moved"], abs=0.01)
