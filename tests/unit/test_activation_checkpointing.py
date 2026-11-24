"""Activation checkpointing (``apply_ac``) across every block class.

``full`` wraps each child of ``layers`` and ``cross_attention_layers`` by
position, so the MoT, MoMa and cross-attention blocks are checkpointed like
``TransformerBlock``; ``selective`` wraps the ``Attention``, ``MoTAttention``
and ``CrossAttention`` modules inside them. Most models are tiny ones built
on CPU through ``build_parallel_model``, the path training takes.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import replace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from kempnerforge.config.adapter import AdapterConfig
from kempnerforge.config.model import ModelConfig
from kempnerforge.config.schema import ActivationCheckpointing
from kempnerforge.config.vision import VisionEncoderConfig
from kempnerforge.config.vlm import (
    CrossAttentionConfig,
    JointDecoderConfig,
    MoMaConfig,
    MoTConfig,
    VLMConfig,
)
from kempnerforge.distributed.parallel import (
    _RoutingCheckpointWrapper,
    apply_ac,
    build_parallel_model,
    default_mp_policy,
)
from kempnerforge.distributed.pipeline_parallel import build_stage_module
from kempnerforge.metrics.logger import get_logger
from kempnerforge.model import router as router_module
from kempnerforge.model.mot import MoTBlock, mot_warm_start_from_text_stack
from kempnerforge.model.router import Recomputing, SigmoidTopKRouter, SoftmaxTopKRouter
from kempnerforge.model.transformer import Transformer
from kempnerforge.model.vlm import inner_transformer
from kempnerforge.training.loop import add_moe_losses

NONE = ActivationCheckpointing.none
FULL = ActivationCheckpointing.full
SELECTIVE = ActivationCheckpointing.selective

VLM_ARCHS: dict[str, type[VLMConfig]] = {
    "joint_decoder": JointDecoderConfig,
    "cross_attention": CrossAttentionConfig,
    "mot": MoTConfig,
    "moma": MoMaConfig,
}
ARCHS = ["text", *VLM_ARCHS]

WRAPPED = {
    ("text", FULL): {"TransformerBlock": 4},
    ("text", SELECTIVE): {"Attention": 4},
    ("joint_decoder", FULL): {"TransformerBlock": 4},
    ("joint_decoder", SELECTIVE): {"Attention": 4},
    ("cross_attention", FULL): {"TransformerBlock": 4, "CrossAttentionBlock": 2},
    ("cross_attention", SELECTIVE): {"Attention": 4, "CrossAttention": 2},
    ("mot", FULL): {"MoTBlock": 4},
    ("mot", SELECTIVE): {"MoTAttention": 4},
    ("moma", FULL): {"MoMaBlock": 4},
    ("moma", SELECTIVE): {"Attention": 4},
}
"""The classes each arch and mode wraps: 4 layers, and 2 cross-attention blocks."""

MODES = [pytest.param(FULL, id="full"), pytest.param(SELECTIVE, id="selective")]


def _model_config() -> ModelConfig:
    return ModelConfig(
        dim=64, n_layers=4, n_heads=4, vocab_size=256, max_seq_len=128, ffn_hidden_dim=128
    )


def _vlm_config(arch: str) -> VLMConfig:
    extra = {"cross_attention_every_n_layers": 2} if arch == "cross_attention" else {}
    return VLM_ARCHS[arch](max_text_len=32, **extra)


def _build(arch: str, mode: ActivationCheckpointing, mesh=None, seed: int = 0) -> nn.Module:
    """A 4-layer fp32 model of ``arch`` on CPU, checkpointed in ``mode``."""
    torch.manual_seed(seed)
    if arch == "text":
        return build_parallel_model(
            _model_config(), torch.device("cpu"), mesh, ac_mode=mode, param_dtype=torch.float32
        )
    return build_parallel_model(
        _model_config(),
        torch.device("cpu"),
        mesh,
        vision_config=VisionEncoderConfig(type="random", feature_dim=96, num_tokens=8),
        adapter_config=AdapterConfig(),
        vlm_config=_vlm_config(arch),
        ac_mode=mode,
        param_dtype=torch.float32,
        mp_policy=default_mp_policy(torch.float32) if mesh is not None else None,
    )


def _loss(arch: str, model: nn.Module, rows: slice = slice(None)) -> torch.Tensor:
    """Mean squared output over a fixed batch of 4 rows, or the ``rows`` of it."""
    gen = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 256, (4, 16), generator=gen)[rows]
    pixels = torch.randn(4, 3, 32, 32, generator=gen)[rows]
    out = model(ids) if arch == "text" else model(pixels, ids, ids)[0]
    return out.float().pow(2).mean()


def _blocks(model: nn.Module) -> list[nn.Module]:
    transformer = inner_transformer(model)
    return [*transformer.layers.values(), *transformer.cross_attention_layers.values()]


def _wrapped(model: nn.Module) -> dict[str, int]:
    """The class of each module wrapped in a ``CheckpointWrapper``, counted."""
    return dict(
        Counter(
            type(m._checkpoint_wrapped_module).__name__
            for m in model.modules()
            if isinstance(m, CheckpointWrapper)
        )
    )


@pytest.fixture
def parallel_logs(caplog, monkeypatch):
    """Messages ``kempnerforge.distributed.parallel`` logged at one level, INFO by default.

    Configuring KempnerForge logging, which the first ``get_logger`` call does once,
    stops the ``kempnerforge`` logger from propagating to the root logger where caplog
    listens. The fixture configures it first, so a module imported later in the test,
    such as the freeze helpers a VLM build loads, cannot turn propagation back off.
    """
    get_logger(__name__)
    monkeypatch.setattr(logging.getLogger("kempnerforge"), "propagate", True)
    caplog.set_level(logging.INFO, logger="kempnerforge.distributed.parallel")

    def _records(level: int = logging.INFO) -> list[str]:
        return [r.getMessage() for r in caplog.records if r.levelno == level]

    return _records


class TestWhatGetsWrapped:
    @pytest.mark.parametrize("arch", ARCHS)
    def test_full_wraps_every_block(self, arch):
        model = _build(arch, FULL)
        assert all(isinstance(block, CheckpointWrapper) for block in _blocks(model))
        assert _wrapped(model) == WRAPPED[(arch, FULL)]

    @pytest.mark.parametrize("arch", ARCHS)
    def test_selective_wraps_every_attention_module(self, arch):
        model = _build(arch, SELECTIVE)
        assert not any(isinstance(block, CheckpointWrapper) for block in _blocks(model))
        assert _wrapped(model) == WRAPPED[(arch, SELECTIVE)]

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("arch", ARCHS)
    def test_returns_the_number_of_modules_it_wrapped(self, arch, mode):
        model = _build(arch, NONE)
        assert apply_ac(inner_transformer(model), mode) == sum(WRAPPED[(arch, mode)].values())

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("arch", ARCHS)
    def test_apply_ac_is_idempotent(self, arch, mode):
        """A second call finds every block checkpointed and wraps nothing new."""
        model = _build(arch, mode)
        assert apply_ac(inner_transformer(model), mode) == 0
        assert _wrapped(model) == WRAPPED[(arch, mode)]

    @pytest.mark.parametrize("mode", MODES)
    def test_accepts_the_mode_as_its_string_value(self, mode):
        model = _build("text", NONE)
        apply_ac(model, mode.value)
        assert _wrapped(model) == WRAPPED[("text", mode)]

    def test_rejects_an_unknown_mode(self):
        model = _build("text", NONE)
        with pytest.raises(ValueError, match="not a valid ActivationCheckpointing"):
            apply_ac(model, "partial")
        assert _wrapped(model) == {}

    def test_full_wraps_a_pipeline_stage(self):
        """A pipeline stage has ``layers`` and no ``cross_attention_layers``."""
        stage = build_stage_module(_model_config(), pp_rank=1, pp_size=2)
        assert apply_ac(stage, FULL) == len(stage.layers) == 2
        assert all(isinstance(block, CheckpointWrapper) for block in stage.layers.values())


class TestCheckpointingTakesEffect:
    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("arch", ARCHS)
    def test_backward_recomputes_every_checkpointed_module(self, arch, mode):
        """Each wrapped module runs its forward once more, inside backward."""
        model = _build(arch, mode)
        wrappers = [m for m in model.modules() if isinstance(m, CheckpointWrapper)]
        assert len(wrappers) == sum(WRAPPED[(arch, mode)].values())
        calls = {"forward": 0, "backward": 0}
        phase = {"now": "forward"}

        def count(module, args):
            calls[phase["now"]] += 1

        for wrapper in wrappers:
            wrapper._checkpoint_wrapped_module.register_forward_pre_hook(count)
        loss = _loss(arch, model)
        phase["now"] = "backward"
        loss.backward()
        assert calls == {"forward": len(wrappers), "backward": len(wrappers)}

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("arch", ARCHS)
    def test_checkpointing_saves_fewer_activations(self, arch, mode):
        """Fewer bytes are saved for backward outside the checkpointed regions."""

        def saved_bytes(model: nn.Module) -> int:
            params = {p.untyped_storage().data_ptr() for p in model.parameters()}
            saved: dict[int, int] = {}

            def pack(t: torch.Tensor) -> torch.Tensor:
                ptr = t.untyped_storage().data_ptr()
                if ptr not in params:
                    saved[ptr] = t.untyped_storage().nbytes()
                return t

            with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
                _loss(arch, model)
            return sum(saved.values())

        assert saved_bytes(_build(arch, mode)) < saved_bytes(_build(arch, NONE))

    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("arch", ARCHS)
    def test_checkpointing_keeps_loss_and_grads(self, arch, mode):
        """The loss is bit-equal to ``none`` and the gradients match.

        MoMa routes with Gumbel noise, so this also checks that the replay in
        backward restores the RNG state and routes the tokens alike.
        """
        reference, checkpointed = _build(arch, NONE), _build(arch, mode)
        torch.manual_seed(2)
        expected = _loss(arch, reference)
        expected.backward()
        torch.manual_seed(2)
        loss = _loss(arch, checkpointed)
        loss.backward()
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)
        ref_grads = {n: p.grad for n, p in reference.named_parameters() if p.grad is not None}
        grads = {
            n.replace("_checkpoint_wrapped_module.", ""): p.grad
            for n, p in checkpointed.named_parameters()
            if p.grad is not None
        }
        assert grads.keys() == ref_grads.keys()
        for name, grad in grads.items():
            torch.testing.assert_close(grad, ref_grads[name], msg=name)


class _NoAttentionBlock(nn.Module):
    """A block with no attention module ``selective`` knows."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(64, 64)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x + self.proj(x)


class TestLogging:
    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("arch", ARCHS)
    def test_log_reports_what_was_wrapped(self, arch, mode, parallel_logs):
        _build(arch, mode)
        counts = ", ".join(f"{n} {name}" for name, n in WRAPPED[(arch, mode)].items())
        assert f"Applied {mode.value} activation checkpointing: wrapped {counts}" in (
            parallel_logs()
        )
        assert parallel_logs(logging.WARNING) == []

    def test_warns_when_a_block_has_no_attention_module(self, parallel_logs):
        model = Transformer(_model_config())
        model.layers["1"] = _NoAttentionBlock()
        assert apply_ac(model, SELECTIVE) == 3
        (warning,) = parallel_logs(logging.WARNING)
        assert "leaves 1 of 4 blocks unwrapped (_NoAttentionBlock)" in warning
        assert "Use 'full'" in warning
        assert "wrapped 3 Attention" in parallel_logs()[-1]

    def test_full_covers_a_block_selective_cannot(self, parallel_logs):
        model = Transformer(_model_config())
        model.layers["1"] = _NoAttentionBlock()
        assert apply_ac(model, FULL) == 4
        assert parallel_logs(logging.WARNING) == []

    def test_warns_when_the_model_has_no_blocks(self, parallel_logs):
        empty = nn.Module()
        empty.layers = nn.ModuleDict()
        assert apply_ac(empty, FULL) == 0
        (warning,) = parallel_logs(logging.WARNING)
        assert "checkpoints nothing" in warning

    def test_none_wraps_nothing_and_logs_nothing(self, parallel_logs):
        model = Transformer(_model_config())
        assert apply_ac(model, NONE) == 0
        assert _wrapped(model) == {}
        assert parallel_logs() == parallel_logs(logging.WARNING) == []


def _text_state_dict() -> dict[str, torch.Tensor]:
    """A joint-decoder transformer's state dict, the MoT warm start's usual source."""
    model = _build("joint_decoder", NONE, seed=1)
    return inner_transformer(model).state_dict()


def _assert_warm_started(transformer: nn.Module, source: dict[str, torch.Tensor]) -> None:
    for idx, layer in transformer.layers.items():
        block = getattr(layer, "_checkpoint_wrapped_module", layer)
        assert isinstance(block, MoTBlock)
        for modality in block.modalities:
            torch.testing.assert_close(
                block.attn.q_proj[modality].weight,
                source[f"layers.{idx}.attention.q_proj.weight"],
                rtol=0,
                atol=0,
            )


class TestMoTWarmStart:
    @pytest.mark.parametrize("mode", MODES)
    def test_mot_warm_start_reaches_checkpointed_blocks(self, mode):
        """The warm start runs after the build, so after ``apply_ac`` wrapped the blocks."""
        source = _text_state_dict()
        transformer = inner_transformer(_build("mot", mode))
        mot_warm_start_from_text_stack(transformer, source)
        _assert_warm_started(transformer, source)

    def test_warm_start_unwraps_a_checkpoint_wrapper(self):
        source = _text_state_dict()
        transformer = inner_transformer(_build("mot", NONE))
        for key in transformer.layers:
            transformer.layers[key] = checkpoint_wrapper(transformer.layers[key])
        mot_warm_start_from_text_stack(transformer, source)
        _assert_warm_started(transformer, source)


def _full(grad: torch.Tensor) -> torch.Tensor:
    """The whole gradient, gathered when FSDP2 holds it as a sharded ``DTensor``."""
    return grad.full_tensor() if isinstance(grad, DTensor) else grad


def _check_fsdp_parity(rank: int, world: int, store: str) -> None:
    """Under FSDP2, checkpointed models match their unchecked twins on each arch and mode."""
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world)
    try:
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp_shard",))
        rows = slice(rank * 2, rank * 2 + 2)
        for arch in VLM_ARCHS:
            reference = _build(arch, NONE, mesh)
            torch.manual_seed(2)
            expected = _loss(arch, reference, rows)
            expected.backward()
            for mode in (FULL, SELECTIVE):
                model = _build(arch, mode, mesh)
                assert _wrapped(model) == WRAPPED[(arch, mode)], (arch, mode)
                torch.manual_seed(2)
                loss = _loss(arch, model, rows)
                loss.backward()
                torch.testing.assert_close(loss, expected, rtol=0, atol=0)
                compared = 0
                for (name, p), (_, q) in zip(
                    reference.named_parameters(), model.named_parameters(), strict=True
                ):
                    if p.grad is None:
                        assert q.grad is None, name
                        continue
                    torch.testing.assert_close(_full(q.grad), _full(p.grad), msg=name)
                    compared += 1
                assert compared > 0, (arch, mode)
    finally:
        dist.destroy_process_group()


class TestUnderFsdp2:
    def test_checkpointed_vlm_blocks_match_under_fsdp2(self, tmp_path):
        """Two gloo ranks shard each VLM with FSDP2 after ``apply_ac`` wrapped its blocks."""
        mp.spawn(_check_fsdp_parity, args=(2, str(tmp_path / "store")), nprocs=2)


ROUTERS = ["softmax_topk", "sigmoid_topk"]


def _moe_model(arch: str, router: str, mode: ActivationCheckpointing, **overrides) -> nn.Module:
    """A 4-layer fp32 MoE model on CPU: text, or MoT with one router per modality."""
    torch.manual_seed(0)
    config = replace(_model_config(), num_experts=4, moe_top_k=2, moe_router=router, **overrides)
    if arch == "text":
        return build_parallel_model(
            config, torch.device("cpu"), None, ac_mode=mode, param_dtype=torch.float32
        )
    return build_parallel_model(
        config,
        torch.device("cpu"),
        None,
        vision_config=VisionEncoderConfig(type="random", feature_dim=96, num_tokens=8),
        adapter_config=AdapterConfig(),
        vlm_config=_vlm_config(arch),
        ac_mode=mode,
        param_dtype=torch.float32,
    )


def _routers(model: nn.Module) -> list[SoftmaxTopKRouter | SigmoidTopKRouter]:
    return [m for m in model.modules() if isinstance(m, (SoftmaxTopKRouter, SigmoidTopKRouter))]


def _counts(model: nn.Module) -> list[torch.Tensor]:
    return [r.pending_counts.clone() for r in _routers(model) if isinstance(r, SigmoidTopKRouter)]


def _moe_loss(arch: str, model: nn.Module, step: int) -> torch.Tensor:
    """The fixed batch's loss at ``step`` plus the MoE terms, the way the loop adds them."""
    config = replace(_model_config(), moe_aux_loss_weight=0.01, moe_router_z_loss_weight=0.001)
    gen = torch.Generator().manual_seed(step)
    ids = torch.randint(0, 256, (4, 16), generator=gen)
    pixels = torch.randn(4, 3, 32, 32, generator=gen)
    out = model(ids) if arch == "text" else model(pixels, ids, ids)[0]
    return add_moe_losses(out.float().pow(2).mean(), model, config)


class TestMoERecompute:
    """Under ``full``, the recompute leaves the routers as the forward set them (#563)."""

    @pytest.mark.parametrize("router", ROUTERS)
    @pytest.mark.parametrize("arch", ["text", "mot"])
    def test_full_trains_like_none(self, arch, router):
        """Losses, parameters and router state match over steps with the per-step balance."""
        runs = []
        for mode in (NONE, FULL):
            model = _moe_model(arch, router, mode)
            optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
            losses, counts = [], []
            for step in range(3):
                loss = _moe_loss(arch, model, step)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                counts.append(_counts(model))
                inner_transformer(model).balance_routers()
                losses.append(loss.detach())
            runs.append((model, losses, counts))
        (none, none_losses, none_counts), (full, full_losses, full_counts) = runs
        torch.testing.assert_close(full_losses, none_losses)
        for a, b in zip(full_counts, none_counts, strict=True):
            assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
        torch.testing.assert_close(full.state_dict(), none.state_dict())

    @pytest.mark.parametrize("router", ROUTERS)
    @pytest.mark.parametrize("arch", ["text", "mot"])
    def test_the_recompute_leaves_the_routers_as_the_forward_set_them(self, arch, router):
        """No attribute is replaced and no token counted twice; nothing keeps a graph."""
        model = _moe_model(arch, router, FULL)
        loss = _moe_loss(arch, model, 0)
        routers = _routers(model)
        kept = [(r.aux_loss, r.z_loss, r.expert_counts) for r in routers]
        counts = _counts(model)
        assert all(r.aux_loss.grad_fn is None and r.z_loss.grad_fn is None for r in routers)
        loss.backward()
        for router_, attrs in zip(routers, kept, strict=True):
            now = (router_.aux_loss, router_.z_loss, router_.expert_counts)
            assert all(a is b for a, b in zip(now, attrs, strict=True))
        assert all(torch.equal(a, b) for a, b in zip(_counts(model), counts, strict=True))
        assert all(c.sum() > 0 for c in counts) and len(counts) == (
            len(routers) if router == "sigmoid_topk" else 0
        )

    @pytest.mark.parametrize("router", ROUTERS)
    def test_the_recompute_routes_as_the_forward_did(self, router):
        """Gate weights moved between the forward and the backward change no token's experts.

        The move stands in for a GPU computing the recompute's scores a little
        differently: the recompute takes the forward's top-k.
        """
        model = _moe_model("text", router, FULL)
        routed: dict[str, dict[nn.Module, torch.Tensor]] = {"forward": {}, "recompute": {}}

        def record(module: nn.Module, args: tuple, out: tuple) -> None:
            phase = "forward" if torch._C._current_graph_task_id() == -1 else "recompute"
            routed[phase][module] = out[1].clone()

        for router_ in _routers(model):
            router_.register_forward_hook(record)
        loss = _moe_loss("text", model, 0)
        with torch.no_grad():
            for router_ in _routers(model):
                weight = router_.gate.weight
                weight.add_(10.0 * torch.sin(1.7 * torch.arange(weight.numel())).view_as(weight))
        loss.backward()
        assert (
            len(routed["forward"]) == 4 and routed["recompute"].keys() == routed["forward"].keys()
        )
        for module, indices in routed["forward"].items():
            assert torch.equal(routed["recompute"][module], indices)

    def test_only_blocks_with_an_moe_layer_get_the_routing_wrapper(self):
        model = _moe_model("text", "softmax_topk", FULL, moe_frequency=2)
        wrappers = [m for m in model.modules() if isinstance(m, CheckpointWrapper)]
        routing = [isinstance(m, _RoutingCheckpointWrapper) for m in wrappers]
        assert routing == [False, True, False, True]
        selective = _moe_model("text", "softmax_topk", SELECTIVE)
        assert not any(isinstance(m, _RoutingCheckpointWrapper) for m in selective.modules())

    def test_the_routing_wrapper_runs_eagerly_under_compile(self):
        assert getattr(_RoutingCheckpointWrapper.forward, "_torchdynamo_disable", False)


class _Context:
    def __init__(self) -> None:
        self.events: list[str] = []

    def __enter__(self) -> None:
        self.events.append("enter")

    def __exit__(self, *exc) -> None:
        self.events.append("exit")


class TestRecomputing:
    def test_sets_the_flag_inside_and_enters_the_inner_context(self):
        inner = _Context()
        context = Recomputing(inner)
        assert not router_module._RECOMPUTING.get()
        for _ in range(2):
            with context:
                assert router_module._RECOMPUTING.get()
            assert not router_module._RECOMPUTING.get()
        assert inner.events == ["enter", "exit"] * 2

    def test_an_inner_context_that_fails_to_enter_leaves_the_flag_unset(self):
        class Failing(_Context):
            def __enter__(self) -> None:
                raise RuntimeError("inner")

        with pytest.raises(RuntimeError, match="inner"), Recomputing(Failing()):
            pass
        assert not router_module._RECOMPUTING.get()
