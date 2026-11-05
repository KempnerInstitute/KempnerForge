"""Tests for selecting a plugin-registered scheduler, norm or MLP from config.

``scheduler.name``, ``model.norm_type`` and ``model.activation`` are typed as enums but
select registry builders, so the loader also accepts a name registered in their
category. The other config enums stay closed.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from kempnerforge.config.checkpoint import AsyncCheckpointMode
from kempnerforge.config.distributed import PipelineSchedule
from kempnerforge.config.loader import load_config
from kempnerforge.config.model import CORE_MLP_KEY_ACTIVATIONS, Activation, ModelConfig, NormType
from kempnerforge.config.registry import registry
from kempnerforge.config.scheduler import SchedulerType
from kempnerforge.config.training import ActivationCheckpointing
from kempnerforge.model.mlp import _ACTIVATION_TO_MLP
from kempnerforge.model.moma import ExpertChoiceMoE
from kempnerforge.model.transformer import Transformer
from kempnerforge.training.scheduler import build_scheduler

REPO_ROOT = Path(__file__).resolve().parents[2]

PLUGIN = """\
import torch
from torch import nn

from kempnerforge.config.registry import registry


class KfNorm(nn.LayerNorm):
    pass


class KfMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(torch.tanh(self.up_proj(x)))


@registry.register_scheduler("{module}_sched")
def build_scheduler(optimizer, config, max_steps):
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5)


registry.register("norm", "{module}_norm", lambda dim, eps=1e-5: KfNorm(dim, eps=eps))
registry.register("mlp", "{module}_mlp", KfMLP)
"""

TINY_MODEL = {
    "dim": 64,
    "n_layers": 1,
    "n_heads": 4,
    "n_kv_heads": 4,
    "vocab_size": 64,
    "max_seq_len": 16,
}

SELECTORS = [
    pytest.param("scheduler", "name", SchedulerType, "scheduler", "_sched", id="scheduler"),
    pytest.param("model", "norm_type", NormType, "norm", "_norm", id="norm"),
    pytest.param("model", "activation", Activation, "mlp", "_mlp", id="activation"),
]


@pytest.fixture
def write_plugin(tmp_path, monkeypatch, isolated_registry):
    """Write a plugin registering ``<module>_sched``, ``<module>_norm`` and ``<module>_mlp``;
    its ``sys.modules`` entry, and the registry entries it adds, go afterward."""
    written: list[str] = []
    monkeypatch.syspath_prepend(str(tmp_path))

    def write(module: str) -> str:
        (tmp_path / f"{module}.py").write_text(PLUGIN.replace("{module}", module))
        importlib.invalidate_caches()
        written.append(module)
        return module

    yield write
    for module in written:
        sys.modules.pop(module, None)


def _load(tmp_path, route: str, module: str, sections: dict[str, dict[str, Any]]):
    """Load a config that imports ``module`` and sets ``sections``, from a TOML or the CLI."""
    if route == "cli":
        args = [f'--plugins=["{module}"]']
        args += [
            f"--{s}.{key}={value}"
            for s, values in sections.items()
            for key, value in values.items()
        ]
        return load_config(cli_args=args)
    lines = [f'plugins = ["{module}"]']
    for section, values in sections.items():
        lines.append(f"[{section}]")
        lines += [f"{key} = {value!r}" for key, value in values.items()]
    path = tmp_path / "job.toml"
    path.write_text("\n".join(lines) + "\n")
    return load_config(str(path), cli_args=[])


@pytest.mark.parametrize("route", ["toml", "cli"])
class TestPluginSelection:
    def test_a_plugin_scheduler_is_built(self, tmp_path, write_plugin, route):
        module = write_plugin(f"kf250_sched_{route}")
        config = _load(tmp_path, route, module, {"scheduler": {"name": f"{module}_sched"}})

        assert config.scheduler.name == f"{module}_sched"
        assert type(config.scheduler.name) is str
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
        build_scheduler(optimizer, config.scheduler, max_steps=10)
        assert optimizer.param_groups[0]["lr"] == 0.5

    def test_a_plugin_norm_and_mlp_build_the_model(self, tmp_path, write_plugin, route):
        module = write_plugin(f"kf250_model_{route}")
        model_section = {**TINY_MODEL, "norm_type": f"{module}_norm", "activation": f"{module}_mlp"}
        config = _load(tmp_path, route, module, {"model": model_section})

        assert (config.model.norm_type, config.model.activation) == (
            f"{module}_norm",
            f"{module}_mlp",
        )
        model = Transformer(config.model)
        block = model.layers["0"]
        assert {type(m).__name__ for m in (model.norm, block.attention_norm, block.mlp_norm)} == {
            "KfNorm"
        }
        assert type(block.mlp).__name__ == "KfMLP"
        logits = model(torch.randint(0, TINY_MODEL["vocab_size"], (2, 8)))
        assert logits.shape == (2, 8, TINY_MODEL["vocab_size"])


@pytest.mark.parametrize(("section", "field", "enum_type", "category", "suffix"), SELECTORS)
def test_an_unregistered_name_is_rejected_with_the_valid_names(
    write_plugin, section, field, enum_type, category, suffix
):
    module = write_plugin("kf250_listed")
    with pytest.raises(ValueError) as excinfo:
        load_config(cli_args=[f'--plugins=["{module}"]', f"--{section}.{field}=kf250_missing"])

    message = str(excinfo.value)
    assert message.startswith(
        f"'kf250_missing' is neither in {enum_type.__name__} nor registered under '{category}'."
    )
    assert f"{enum_type.__name__} members: {[m.value for m in enum_type]}." in message
    assert f"'{module}{suffix}'" in message.split("Registered")[1]


@pytest.mark.parametrize(("section", "field", "enum_type", "category", "suffix"), SELECTORS)
def test_a_name_registered_in_another_category_is_rejected(
    write_plugin, section, field, enum_type, category, suffix
):
    """Only the field's own category counts, so the plugin's two other names are rejected."""
    module = write_plugin("kf250_other")
    for other in sorted({"_sched", "_norm", "_mlp"} - {suffix}):
        name = f"{module}{other}"
        with pytest.raises(
            ValueError,
            match=f"^'{name}' is neither in {enum_type.__name__} nor registered under '{category}'",
        ):
            load_config(cli_args=[f'--plugins=["{module}"]', f"--{section}.{field}={name}"])


@pytest.mark.parametrize(
    ("section", "field", "enum_type"),
    [
        ("scheduler", "name", SchedulerType),
        ("model", "norm_type", NormType),
        ("model", "activation", Activation),
    ],
)
def test_a_member_name_still_becomes_the_member(section, field, enum_type):
    for member in enum_type:
        config = load_config(cli_args=[f"--{section}.{field}={member.value}"])
        assert getattr(getattr(config, section), field) is member


@pytest.mark.parametrize(
    ("key", "enum_type"),
    [
        ("train.activation_checkpointing", ActivationCheckpointing),
        ("distributed.pp_schedule", PipelineSchedule),
        ("checkpoint.async_mode", AsyncCheckpointMode),
    ],
)
def test_a_closed_enum_rejects_names_registered_for_the_open_ones(write_plugin, key, enum_type):
    module = write_plugin("kf250_closed")
    for name in (f"{module}_sched", f"{module}_norm", f"{module}_mlp"):
        with pytest.raises(ValueError, match=f"^'{name}' is not a valid {enum_type.__name__}$"):
            load_config(cli_args=[f'--plugins=["{module}"]', f"--{key}={name}"])


CORE_MLP_KEYS = [("swiglu", "silu"), ("standard_gelu", "gelu"), ("standard_relu", "relu")]


def _core_key_message(key: str, activation: str) -> str:
    return (
        f"activation={key!r} is the mlp registry key that activation={activation!r} selects; "
        f"set activation={activation!r} instead"
    )


class TestCoreMlpKeys:
    """Core's own ``mlp`` keys are rejected as ``model.activation``, with the activation that
    selects each, whatever was imported before the config loads."""

    def test_they_are_the_keys_the_activations_select(self):
        selected = {key: Activation(activation) for activation, key in _ACTIVATION_TO_MLP.items()}
        assert selected == CORE_MLP_KEY_ACTIVATIONS

    @pytest.mark.parametrize(("key", "activation"), CORE_MLP_KEYS)
    def test_rejected_at_load_with_the_model_imported(self, key, activation):
        assert key in registry.list("mlp")
        with pytest.raises(ValueError) as excinfo:
            load_config(cli_args=[f"--model.activation={key}"])
        assert str(excinfo.value) == _core_key_message(key, activation)

    def test_rejected_the_same_way_with_only_the_loader_imported(self):
        """A fresh interpreter, where no core module has registered the keys yet."""
        code = (
            "import sys\n"
            "from kempnerforge.config.loader import load_config\n"
            "from kempnerforge.config.registry import registry\n"
            "assert registry.list('mlp') == []\n"
            "for key in sys.argv[1:]:\n"
            "    try:\n"
            "        load_config(cli_args=[f'--model.activation={key}'])\n"
            "    except ValueError as e:\n"
            "        print(e)\n"
        )
        keys = [key for key, _ in CORE_MLP_KEYS]
        result = subprocess.run(
            [sys.executable, "-c", code, *keys],
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
            capture_output=True,
            text=True,
            timeout=150,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines() == [_core_key_message(*pair) for pair in CORE_MLP_KEYS]

    @pytest.mark.parametrize(("key", "activation"), CORE_MLP_KEYS)
    def test_rejected_when_the_config_is_built_directly(self, key, activation):
        with pytest.raises(ValueError) as excinfo:
            ModelConfig(activation=key)
        assert str(excinfo.value) == _core_key_message(key, activation)

    def test_left_out_of_the_names_an_unknown_activation_lists(self):
        with pytest.raises(ValueError) as excinfo:
            load_config(cli_args=["--model.activation=kf250_unknown"])
        listed = str(excinfo.value).split("Registered 'mlp' names: ")[1]
        assert not any(key in listed for key, _ in CORE_MLP_KEYS)


class TestPluginMlpInMoE:
    """MoE experts compute core's SwiGLU or standard MLP from their weights, so a plugin's MLP
    is rejected with ``num_experts > 0``. MoMa's experts run each expert module as it is."""

    @pytest.mark.parametrize("route", ["toml", "cli"])
    def test_rejected_with_moe(self, tmp_path, write_plugin, route):
        module = write_plugin(f"kf250_moe_{route}")
        model = {**TINY_MODEL, "activation": f"{module}_mlp", "num_experts": 2, "moe_top_k": 1}
        with pytest.raises(ValueError) as excinfo:
            _load(tmp_path, route, module, {"model": model})
        assert str(excinfo.value).startswith(
            f"activation='{module}_mlp' selects a plugin-registered MLP, which MoE experts "
            "do not support (num_experts=2)"
        )

    def test_rejected_with_moe_when_the_config_is_built_directly(self):
        with pytest.raises(ValueError, match="^activation='kf250_direct' selects a plugin"):
            ModelConfig(**TINY_MODEL, activation="kf250_direct", num_experts=2, moe_top_k=1)

    def test_rejected_for_mot_with_moe(self, write_plugin):
        """MoT builds an MoE layer per modality when ``num_experts > 0``."""
        module = write_plugin("kf250_mot")
        args = [f'--plugins=["{module}"]', f"--model.activation={module}_mlp"]
        args += ["--model.num_experts=2", "--vlm.arch=mot", "--vision_encoder.type=random"]
        with pytest.raises(ValueError, match="which MoE experts do not support"):
            load_config(cli_args=args)

    def test_moma_experts_run_a_plugin_mlp_as_it_is(self, write_plugin):
        """MoMa keeps ``num_experts = 0``, and each of its expert-choice experts calls its own
        module, so a plugin MLP loads and computes its own forward, in fp32 and bf16."""
        module = write_plugin("kf250_moma")
        args = [f'--plugins=["{module}"]', f"--model.activation={module}_mlp"]
        config = load_config(cli_args=[*args, "--vlm.arch=moma", "--vision_encoder.type=random"])
        assert (config.model.activation, type(config.vlm).__name__) == (
            f"{module}_mlp",
            "MoMaConfig",
        )
        torch.manual_seed(0)
        for dtype in (torch.float32, torch.bfloat16):
            moe = ExpertChoiceMoE(
                dim=64,
                hidden_dim=128,
                num_experts=2,
                capacity_factor=0.5,
                activation=f"{module}_mlp",
                gumbel_noise=False,
            )
            moe = moe.to(dtype).eval()
            x = torch.randn(16, 64, dtype=dtype)
            with torch.no_grad():
                got = moe(x)
                scores, indices = moe.router(x)
                expected = torch.zeros_like(x)
                for e, expert in enumerate(moe.experts):
                    out = expert(x.index_select(0, indices[e]))
                    expected = expected.index_add(0, indices[e], scores[e].unsqueeze(-1) * out)
            assert type(moe.experts[0]).__name__ == "KfMLP"
            torch.testing.assert_close(got, expected)
