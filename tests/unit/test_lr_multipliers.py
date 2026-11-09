"""Per-module learning rates: ``optimizer.lr_multipliers``.

Each module listed there trains at ``optimizer.lr`` times its multiplier, in param groups of
its own, under the one learning-rate schedule. With nothing listed, nothing changes.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import re
from pathlib import Path

import pytest
import torch
from torch import nn

from kempnerforge.checkpoint.manager import CheckpointManager
from kempnerforge.checkpoint.state import build_train_state, restore_train_state
from kempnerforge.config.checkpoint import CheckpointConfig
from kempnerforge.config.job import JobConfig
from kempnerforge.config.loader import load_config
from kempnerforge.config.model import ModelConfig
from kempnerforge.config.optimizer import OptimizerConfig, checked_lr_multipliers
from kempnerforge.config.registry import registry
from kempnerforge.config.scheduler import SchedulerConfig
from kempnerforge.config.vlm import VLMConfig
from kempnerforge.model.transformer import Transformer
from kempnerforge.training.optimizer import Muon, build_optimizer, module_learning_rates
from kempnerforge.training.scheduler import build_scheduler

REGISTERED_OPTIMIZERS = sorted(registry.list("optimizer"))
RESUMABLE_OPTIMIZERS = [name for name in REGISTERED_OPTIMIZERS if name != "schedule_free_adamw"]

VLM_PATTERNS = {
    "transformer": ["transformer", "transformer.*"],
    "vision_encoder": ["vision_encoder", "vision_encoder.*"],
    "adapter": ["adapter", "adapter.*"],
}


def _vlm_like() -> nn.Module:
    """Three modules named as a VLM's are, each with a matrix and 1-D parameters."""
    torch.manual_seed(0)
    model = nn.Module()
    model.vision_encoder = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8))
    model.adapter = nn.Linear(8, 8)
    model.transformer = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8))
    return model


def _forward(model: nn.Module) -> torch.Tensor:
    x = torch.randn(4, 8, generator=torch.Generator().manual_seed(1))
    return model.transformer(model.adapter(model.vision_encoder(x))).square().sum()


def _rates(model: nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, float]:
    """The learning rate each parameter steps at, across Muon's inner optimizer too."""
    names = {id(p): n for n, p in model.named_parameters()}
    rates = {}
    for opt in (optimizer, *getattr(optimizer, "inner_optimizers", ())):
        for group in opt.param_groups:
            for p in group["params"]:
                rates[names[id(p)]] = group["lr"]
    return rates


def _restore_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The warnings the train-state restore logged."""
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING and r.name == "kempnerforge.checkpoint.state"
    ]


def _layout(model: nn.Module, optimizer: torch.optim.Optimizer) -> list[tuple]:
    names = {id(p): n for n, p in model.named_parameters()}
    return [
        (
            group.get("module"),
            group["lr"],
            group["weight_decay"],
            sorted(names[id(p)] for p in group["params"]),
        )
        for group in optimizer.param_groups
    ]


class TestConfig:
    def test_the_default_lists_no_module(self):
        assert OptimizerConfig().lr_multipliers == {}

    def test_numbers_become_floats(self):
        config = OptimizerConfig(lr_multipliers={"adapter": 2, "vision_encoder": 0.5})
        assert config.lr_multipliers == {"adapter": 2.0, "vision_encoder": 0.5}
        assert all(type(v) is float for v in config.lr_multipliers.values())

    @pytest.mark.parametrize(
        ("value", "match"),
        [
            (0.5, "must map module names to numbers"),
            ({1: 0.5}, "keys must be module names"),
            ({"": 0.5}, "keys must be module names"),
            ({"adapter": True}, r"lr_multipliers\['adapter'\] must be a positive finite"),
            ({"adapter": "2"}, r"lr_multipliers\['adapter'\] must be a positive finite"),
            ({"adapter": 0}, r"lr_multipliers\['adapter'\] must be a positive finite"),
            ({"adapter": -1.0}, r"lr_multipliers\['adapter'\] must be a positive finite"),
            ({"adapter": math.nan}, r"lr_multipliers\['adapter'\] must be a positive finite"),
            ({"adapter": math.inf}, r"lr_multipliers\['adapter'\] must be a positive finite"),
        ],
        ids=[
            "not-a-dict",
            "int-key",
            "empty-key",
            "bool",
            "str",
            "zero",
            "negative",
            "nan",
            "inf",
        ],
    )
    def test_invalid_multipliers_are_refused(self, value, match):
        with pytest.raises(ValueError, match=match):
            OptimizerConfig(lr_multipliers=value)
        with pytest.raises(ValueError, match=match):
            checked_lr_multipliers(value)

    def test_a_module_name_with_a_slash_is_refused_by_the_config(self):
        """Its ``lr/<module>`` would nest under another module's; a saved record keeps any name."""
        with pytest.raises(ValueError, match=r"names modules \['a/b'\] with '/'"):
            OptimizerConfig(lr_multipliers={"a": 1.0, "a/b": 2.0})
        assert checked_lr_multipliers({"a/b": 2.0}) == {"a/b": 2.0}

    def test_a_run_without_vlm_cannot_list_modules(self):
        with pytest.raises(ValueError, match="needs a \\[vlm\\] section"):
            JobConfig(optimizer=OptimizerConfig(lr_multipliers={"adapter": 2.0}))

    def test_an_unknown_module_is_refused_with_the_known_ones(self, tiny_vlm_configs):
        mc, vision, adapter, vlm = tiny_vlm_configs
        with pytest.raises(ValueError, match=r"unknown modules \['adaptor'\].*'adapter'"):
            JobConfig(
                model=mc,
                vision_encoder=vision,
                adapter=adapter,
                vlm=vlm,
                optimizer=OptimizerConfig(lr_multipliers={"adaptor": 2.0}),
            )

    @pytest.mark.parametrize("arch", ["cross_attention", "mot", "moma"])
    def test_the_config_leaves_aliases_that_share_parameters_to_the_build(
        self, tiny_vlm_configs, arch
    ):
        """Which parameters two modules share depends on the model's parameter names, which
        only the optimizer build has; the pattern strings alone cannot tell."""
        mc, vision, adapter, _ = tiny_vlm_configs
        multipliers = {"transformer": 0.5, arch: 2.0}
        config = JobConfig(
            model=mc,
            vision_encoder=vision,
            adapter=adapter,
            vlm=VLMConfig.for_arch(arch, max_text_len=32),
            optimizer=OptimizerConfig(lr_multipliers=multipliers),
        )
        assert config.optimizer.lr_multipliers == multipliers

    def test_disjoint_aliases_pass(self, tiny_vlm_configs):
        mc, vision, adapter, vlm = tiny_vlm_configs
        config = JobConfig(
            model=mc,
            vision_encoder=vision,
            adapter=adapter,
            vlm=vlm,
            optimizer=OptimizerConfig(lr_multipliers={"adapter": 2.0, "vision_encoder": 0.1}),
        )
        assert config.optimizer.lr_multipliers == {"adapter": 2.0, "vision_encoder": 0.1}

    def test_the_toml_table_and_a_cli_override_load(self, tmp_path):
        toml = tmp_path / "vlm.toml"
        toml.write_text(
            """
[model]
dim = 64
n_layers = 2
n_heads = 4
n_kv_heads = 4
vocab_size = 256
max_seq_len = 96

[train]
seq_len = 96

[vision_encoder]
type = "random"
num_tokens = 16

[vlm]
max_text_len = 64

[optimizer.lr_multipliers]
vision_encoder = 0.05
adapter = 1
"""
        )
        config = load_config(str(toml), cli_args=[])
        assert config.optimizer.lr_multipliers == {"vision_encoder": 0.05, "adapter": 1.0}
        override = load_config(str(toml), cli_args=["--optimizer.lr_multipliers={'adapter':3}"])
        assert override.optimizer.lr_multipliers == {"adapter": 3.0}
        with pytest.raises(ValueError, match="unknown modules"):
            load_config(str(toml), cli_args=["--optimizer.lr_multipliers={'mlp':3}"])


class TestGroups:
    def test_without_multipliers_the_groups_are_unchanged(self):
        """The two decay groups, with no module keys, whether or not patterns are passed."""
        model = _vlm_like()
        config = OptimizerConfig(lr=0.01, weight_decay=0.1)
        for optimizer in (
            build_optimizer(model, config),
            build_optimizer(model, config, module_patterns=VLM_PATTERNS),
        ):
            assert _layout(model, optimizer) == [
                (
                    None,
                    0.01,
                    0.1,
                    ["adapter.weight", "transformer.0.weight", "vision_encoder.0.weight"],
                ),
                (
                    None,
                    0.01,
                    0.0,
                    [
                        "adapter.bias",
                        "transformer.0.bias",
                        "transformer.1.bias",
                        "transformer.1.weight",
                        "vision_encoder.0.bias",
                        "vision_encoder.1.bias",
                        "vision_encoder.1.weight",
                    ],
                ),
            ]
            assert module_learning_rates(optimizer) == {}

    def test_each_listed_module_gets_its_own_groups(self):
        """Unlisted parameters first, then each module by name, decay before no-decay."""
        model = _vlm_like()
        config = OptimizerConfig(
            lr=0.01, weight_decay=0.1, lr_multipliers={"vision_encoder": 0.5, "adapter": 2.0}
        )
        optimizer = build_optimizer(model, config, module_patterns=VLM_PATTERNS)
        assert _layout(model, optimizer) == [
            (None, 0.01, 0.1, ["transformer.0.weight"]),
            (None, 0.01, 0.0, ["transformer.0.bias", "transformer.1.bias", "transformer.1.weight"]),
            ("adapter", 0.02, 0.1, ["adapter.weight"]),
            ("adapter", 0.02, 0.0, ["adapter.bias"]),
            ("vision_encoder", 0.005, 0.1, ["vision_encoder.0.weight"]),
            (
                "vision_encoder",
                0.005,
                0.0,
                ["vision_encoder.0.bias", "vision_encoder.1.bias", "vision_encoder.1.weight"],
            ),
        ]
        assert module_learning_rates(optimizer) == {
            "lr/adapter": 0.02,
            "lr/vision_encoder": 0.005,
        }

    @pytest.mark.parametrize("name", REGISTERED_OPTIMIZERS)
    def test_every_optimizer_steps_each_module_at_its_rate(self, name):
        """Muon's AdamW fallback scales by the multiplier from ``muon_adam_lr`` instead."""
        model = _vlm_like()
        config = OptimizerConfig(
            name=name,
            lr=0.01,
            fused=False,
            muon_adam_lr=0.001,
            lr_multipliers={"vision_encoder": 0.5, "adapter": 2.0},
        )
        optimizer = build_optimizer(model, config, module_patterns=VLM_PATTERNS)
        multiplier = {"vision_encoder": 0.5, "adapter": 2.0, "transformer": 1.0}
        muon_matrices = {"vision_encoder.0.weight", "adapter.weight", "transformer.0.weight"}
        for param_name, rate in _rates(model, optimizer).items():
            base = 0.001 if name == "muon" and param_name not in muon_matrices else 0.01
            assert rate == pytest.approx(base * multiplier[param_name.split(".")[0]], rel=1e-12)
        _forward(model).backward()
        optimizer.step()
        assert all(torch.isfinite(p).all() for p in model.parameters())

    def test_empty_groups_are_left_out(self):
        """With every trained parameter in a listed module, no unlisted group is built."""
        model = _vlm_like()
        model.transformer.requires_grad_(False)
        config = OptimizerConfig(
            lr=0.01,
            weight_decay_scope="all",
            lr_multipliers={"vision_encoder": 0.5, "adapter": 2.0},
        )
        optimizer = build_optimizer(model, config, module_patterns=VLM_PATTERNS)
        assert [(g["module"], len(g["params"])) for g in optimizer.param_groups] == [
            ("adapter", 2),
            ("vision_encoder", 4),
        ]

    def test_a_frozen_module_gets_no_group(self):
        model = _vlm_like()
        model.vision_encoder.requires_grad_(False)
        config = OptimizerConfig(lr=0.01, lr_multipliers={"vision_encoder": 0.5, "adapter": 2.0})
        optimizer = build_optimizer(model, config, module_patterns=VLM_PATTERNS)
        assert module_learning_rates(optimizer) == {"lr/adapter": 0.02}
        held = {id(p) for g in optimizer.param_groups for p in g["params"]}
        assert not any(id(p) in held for p in model.vision_encoder.parameters())

    def test_a_listed_module_with_nothing_to_train_warns(self, caplog, monkeypatch):
        """Its multiplier changes no rate, as with the vision encoder the default freeze keeps."""
        monkeypatch.setattr(logging.getLogger("kempnerforge"), "propagate", True)
        model = _vlm_like()
        model.vision_encoder.requires_grad_(False)
        config = OptimizerConfig(lr=0.01, lr_multipliers={"vision_encoder": 0.5, "adapter": 2.0})
        with caplog.at_level(logging.INFO, logger="kempnerforge.training.optimizer"):
            build_optimizer(model, config, module_patterns=VLM_PATTERNS)
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings == [
            "optimizer.lr_multipliers lists 'vision_encoder', which has no trainable parameters "
            "when the optimizer is built, so its multiplier has no effect"
        ]

    def test_a_compiled_model_matches_the_names_it_wraps(self):
        model = _vlm_like()
        config = OptimizerConfig(lr=0.01, lr_multipliers={"adapter": 2.0})
        plain = build_optimizer(model, config, module_patterns=VLM_PATTERNS)
        compiled = build_optimizer(torch.compile(model), config, module_patterns=VLM_PATTERNS)
        assert _layout(model, compiled) == _layout(model, plain)

    def test_module_patterns_are_required(self):
        config = OptimizerConfig(lr_multipliers={"adapter": 2.0})
        with pytest.raises(ValueError, match="build_optimizer needs module_patterns"):
            build_optimizer(_vlm_like(), config)

    def test_an_unknown_module_is_refused(self):
        config = OptimizerConfig(lr_multipliers={"mlp": 2.0})
        with pytest.raises(ValueError, match=r"unknown modules \['mlp'\].*'adapter'"):
            build_optimizer(_vlm_like(), config, module_patterns=VLM_PATTERNS)

    def test_a_module_that_matches_nothing_is_refused(self):
        config = OptimizerConfig(lr_multipliers={"head": 2.0})
        patterns = {**VLM_PATTERNS, "head": ["output.*"]}
        with pytest.raises(ValueError, match=r"modules \['head'\] whose module_patterns match no"):
            build_optimizer(_vlm_like(), config, module_patterns=patterns)

    NESTED = {
        **VLM_PATTERNS,
        "outer": ["transformer.*"],
        "middle": ["transformer.0.*", "transformer.1.weight"],
        "inner": ["transformer.0.weight"],
    }

    def _owners(self, model, optimizer):
        """The module each parameter's group names, None for one of no listed module."""
        names = {id(p): n for n, p in model.named_parameters()}
        return {names[id(p)]: g.get("module") for g in optimizer.param_groups for p in g["params"]}

    def test_a_module_nested_in_another_keeps_its_own_parameters(self, caplog, monkeypatch):
        """A parameter two listed modules match goes to the one whose parameters the other's
        strictly contain, and the build log says that module sits inside the other."""
        monkeypatch.setattr(logging.getLogger("kempnerforge"), "propagate", True)
        model = _vlm_like()
        config = OptimizerConfig(lr=0.01, lr_multipliers={"transformer": 0.5, "inner": 2.0})
        with caplog.at_level(logging.INFO, logger="kempnerforge.training.optimizer"):
            optimizer = build_optimizer(model, config, module_patterns=self.NESTED)
        rates = _rates(model, optimizer)
        assert rates["transformer.0.weight"] == pytest.approx(0.02)
        assert {
            n: r
            for n, r in rates.items()
            if n.startswith("transformer.") and n != ("transformer.0.weight")
        } == {
            "transformer.0.bias": pytest.approx(0.005),
            "transformer.1.weight": pytest.approx(0.005),
            "transformer.1.bias": pytest.approx(0.005),
        }
        assert rates["adapter.weight"] == 0.01
        assert [m for m in caplog.messages if m.startswith("Optimizer module")] == [
            "Optimizer module inner: owns 64 params, 64 trainable, at lr 0.02, inside transformer",
            "Optimizer module transformer: owns 24 params, 24 trainable, at lr 0.005",
        ]

    def test_a_three_level_nest_resolves_to_the_innermost(self, caplog, monkeypatch):
        monkeypatch.setattr(logging.getLogger("kempnerforge"), "propagate", True)
        model = _vlm_like()
        multipliers = {"outer": 0.5, "middle": 2.0, "inner": 4.0}
        config = OptimizerConfig(lr=0.01, lr_multipliers=multipliers)
        with caplog.at_level(logging.INFO, logger="kempnerforge.training.optimizer"):
            optimizer = build_optimizer(model, config, module_patterns=self.NESTED)
        owners = self._owners(model, optimizer)
        assert {n: m for n, m in owners.items() if n.startswith("transformer.")} == {
            "transformer.0.weight": "inner",
            "transformer.0.bias": "middle",
            "transformer.1.weight": "middle",
            "transformer.1.bias": "outer",
        }
        inner = [m for m in caplog.messages if m.startswith("Optimizer module inner")]
        assert inner == [
            "Optimizer module inner: owns 64 params, 64 trainable, at lr 0.04, inside middle, outer"
        ]

    def test_ownership_does_not_depend_on_what_is_frozen(self):
        """With the rest of ``transformer`` frozen, both modules match the same trainable
        parameters, but ownership is decided over every parameter, so it stands."""
        model = _vlm_like()
        model.transformer[1].requires_grad_(False)
        config = OptimizerConfig(lr=0.01, lr_multipliers={"transformer": 0.5, "inner": 2.0})
        patterns = {**self.NESTED, "inner": ["transformer.0.*"]}
        optimizer = build_optimizer(model, config, module_patterns=patterns)
        owners = self._owners(model, optimizer)
        assert {n: m for n, m in owners.items() if n.startswith("transformer.")} == {
            "transformer.0.weight": "inner",
            "transformer.0.bias": "inner",
        }
        assert module_learning_rates(optimizer) == {"lr/inner": pytest.approx(0.02)}

    @pytest.mark.parametrize(
        ("patterns", "listed", "match"),
        [
            (
                {**VLM_PATTERNS, "weights": ["*.weight"]},
                {"transformer": 0.5, "weights": 2.0},
                r"lists 'transformer' and 'weights', whose module_patterns match parameters that "
                r"partly overlap, such as \['transformer.0.weight', 'transformer.1.weight'\]",
            ),
            (
                {**VLM_PATTERNS, "connector": ["adapter.*"]},
                {"adapter": 2.0, "connector": 3.0},
                r"lists 'adapter' and 'connector', whose module_patterns match the same "
                r"parameters, such as \['adapter.bias', 'adapter.weight'\]",
            ),
        ],
        ids=["partial-overlap", "identical"],
    )
    def test_modules_that_neither_nest_nor_part_are_refused(self, patterns, listed, match):
        """No module's parameters contain the other's, so neither owns the shared ones."""
        config = OptimizerConfig(lr_multipliers=listed)
        with pytest.raises(ValueError, match=match):
            build_optimizer(_vlm_like(), config, module_patterns=patterns)

    def test_a_vlm_wrapper_splits_by_its_module_patterns(self, tiny_vlm_configs):
        """The aliases of ``VLMConfig.module_patterns`` select the wrapper's own parameters."""
        from kempnerforge.model.vlm import build_vlm_wrapper

        wrapper = build_vlm_wrapper(*tiny_vlm_configs)
        vlm: VLMConfig = tiny_vlm_configs[3]
        config = OptimizerConfig(lr=1e-3, lr_multipliers={"adapter": 10.0, "transformer": 0.5})
        optimizer = build_optimizer(wrapper, config, module_patterns=vlm.module_patterns)
        rates = {
            name.split(".")[0]: rate
            for name, rate in _rates(wrapper, optimizer).items()
            if not name.startswith("vision_encoder")
        }
        assert rates == {"adapter": pytest.approx(1e-2), "transformer": pytest.approx(5e-4)}
        assert optimizer.param_groups[0]["module"] == "adapter"

    def test_an_arch_alias_inside_transformer_takes_its_own_rate(self, tmp_path):
        """Through the config loader and the VLM wrapper: under ``vlm.arch = "cross_attention"``
        the cross-attention layers train at their own rate, and the rest of ``transformer`` at
        its."""
        from kempnerforge.model.vlm import build_vlm_wrapper

        toml = tmp_path / "ca.toml"
        toml.write_text(
            """
[model]
dim = 64
n_layers = 2
n_heads = 4
n_kv_heads = 4
vocab_size = 256
max_seq_len = 96

[train]
seq_len = 96

[vision_encoder]
type = "random"
num_tokens = 16

[vlm]
arch = "cross_attention"
max_text_len = 64
cross_attention_every_n_layers = 1

[optimizer]
lr = 0.01

[optimizer.lr_multipliers]
transformer = 0.1
cross_attention = 1.0
"""
        )
        config = load_config(str(toml), cli_args=[])
        assert config.vlm is not None and config.vision_encoder is not None
        assert config.adapter is not None
        wrapper = build_vlm_wrapper(config.model, config.vision_encoder, config.adapter, config.vlm)
        optimizer = build_optimizer(
            wrapper, config.optimizer, module_patterns=config.vlm.module_patterns
        )
        rates = _rates(wrapper, optimizer)
        cross = {n for n in rates if n.startswith("transformer.cross_attention_layers.")}
        rest = {n for n in rates if n.startswith("transformer.")} - cross
        assert cross and rest
        assert all(rates[n] == pytest.approx(0.01) for n in cross)
        assert all(rates[n] == pytest.approx(0.001) for n in rest)
        assert module_learning_rates(optimizer) == {
            "lr/cross_attention": pytest.approx(0.01),
            "lr/transformer": pytest.approx(0.001),
        }


class TestSchedule:
    @pytest.mark.parametrize("name", REGISTERED_OPTIMIZERS)
    def test_one_schedule_scales_every_group_from_its_own_peak(self, name):
        model = _vlm_like()
        config = OptimizerConfig(
            name=name, lr=0.01, fused=False, lr_multipliers={"vision_encoder": 0.5, "adapter": 2.0}
        )
        optimizer = build_optimizer(model, config, module_patterns=VLM_PATTERNS)
        peaks = [group["lr"] for group in optimizer.param_groups]
        scheduler = build_scheduler(
            optimizer,
            SchedulerConfig(warmup_steps=3, warmup_start_factor=0.1, min_lr_ratio=0.1),
            max_steps=10,
        )
        for step in range(10):
            factor = scheduler.lr_lambdas[0](scheduler.last_epoch)
            for group, peak in zip(optimizer.param_groups, peaks, strict=True):
                assert group["lr"] == peak * factor, (step, group.get("module"))
            for p in model.parameters():
                p.grad = torch.zeros_like(p)
            optimizer.step()
            scheduler.step()


class TestMuonFallback:
    @staticmethod
    def _muon(model, **overrides):
        config = OptimizerConfig(name="muon", lr=0.02, muon_adam_lr=3e-4, **overrides)
        return build_optimizer(model, config, module_patterns=VLM_PATTERNS)

    def test_the_default_path_scales_the_fallback_as_before(self):
        """No multipliers: the fallback rate is ``muon_adam_lr`` times Muon's own ratio, exactly."""
        model = _vlm_like()
        optimizer = self._muon(model)
        optimizer.param_groups[0]["lr"] = 0.0123
        _forward(model).backward()
        optimizer.step()
        assert [g["lr"] for g in optimizer._adam.param_groups] == [3e-4 * (0.0123 / 0.02)]

    def test_each_fallback_group_keeps_its_own_rate(self):
        model = _vlm_like()
        optimizer = self._muon(model, lr_multipliers={"vision_encoder": 0.5, "adapter": 2.0})
        scale = 0.25
        for group in optimizer.param_groups:
            group["lr"] *= scale
        _forward(model).backward()
        optimizer.step()
        fallback = {g.get("module"): g["lr"] for g in optimizer._adam.param_groups}
        assert fallback == {
            None: 3e-4 * scale,
            "adapter": pytest.approx(6e-4 * scale, rel=1e-12),
            "vision_encoder": pytest.approx(1.5e-4 * scale, rel=1e-12),
        }

    def test_a_listed_module_can_lead_the_schedule(self):
        """When Muon's first group belongs to a module, its own peak sets the factor."""
        model = _vlm_like()
        model.transformer.requires_grad_(False)
        optimizer = self._muon(model, lr_multipliers={"vision_encoder": 0.5, "adapter": 2.0})
        assert optimizer.param_groups[0]["module"] == "adapter"
        for group in optimizer.param_groups:
            group["lr"] *= 0.5
        _forward(model).backward()
        optimizer.step()
        rates = {g["module"]: g["lr"] for g in optimizer._adam.param_groups}
        assert rates == {
            "adapter": pytest.approx(3e-4, rel=1e-12),
            "vision_encoder": pytest.approx(7.5e-5, rel=1e-12),
        }

    def test_a_zero_lead_rate_leaves_the_fallback_alone(self):
        weight = nn.Parameter(torch.randn(4, 4))
        bias = nn.Parameter(torch.zeros(4))
        optimizer = Muon([{"params": [weight]}], [{"params": [bias]}], lr=0.0, adam_lr=0.1)
        weight.grad = torch.ones_like(weight)
        bias.grad = torch.ones_like(bias)
        assert module_learning_rates(optimizer) == {}
        optimizer.step()
        assert optimizer._adam is not None
        assert optimizer._adam.param_groups[0]["lr"] == 0.1

    def test_each_module_logs_the_rate_its_fallback_steps_at_next(self):
        """A module's AdamW fallback groups are logged too, at the rate of the coming step."""
        model = _vlm_like()
        optimizer = self._muon(model, lr_multipliers={"vision_encoder": 0.5, "adapter": 2.0})
        for group in optimizer.param_groups:
            group["lr"] *= 0.25
        assert module_learning_rates(optimizer) == {
            "lr/adapter": pytest.approx(0.01),
            "lr_adamw/adapter": pytest.approx(1.5e-4),
            "lr/vision_encoder": pytest.approx(0.0025),
            "lr_adamw/vision_encoder": pytest.approx(3.75e-5),
        }
        _forward(model).backward()
        optimizer.step()
        fallback = {g["module"]: g["lr"] for g in optimizer._adam.param_groups if "module" in g}
        assert fallback == {
            "adapter": pytest.approx(1.5e-4),
            "vision_encoder": pytest.approx(3.75e-5),
        }

    def test_a_state_dict_round_trip_keeps_the_fallback_base_rates(self):
        """A Muon built under other rates takes the saved fallback rates, as it does ``lr``."""
        multipliers = {"vision_encoder": 0.5, "adapter": 2.0}
        model = _vlm_like()
        optimizer = self._muon(model, lr_multipliers=multipliers)
        _forward(model).backward()
        optimizer.step()
        state = optimizer.state_dict()
        assert state["_adam_group_lrs"] == [None, pytest.approx(6e-4), pytest.approx(1.5e-4)]
        other = build_optimizer(
            _vlm_like(),
            OptimizerConfig(name="muon", lr=0.04, muon_adam_lr=1e-3, lr_multipliers=multipliers),
            module_patterns=VLM_PATTERNS,
        )
        other.load_state_dict(copy.deepcopy(state))
        assert isinstance(other, Muon)
        assert other._fallback_lrs() == optimizer._fallback_lrs()

    def test_a_state_dict_without_the_fallback_rates_keeps_its_own(self):
        """The default path saves no new key, and a state saved without them still loads."""
        multipliers = {"vision_encoder": 0.5, "adapter": 2.0}
        default = self._muon(_vlm_like()).state_dict()
        assert not {"_lead_group_lr", "_adam_group_lrs"} & set(default)
        optimizer = self._muon(_vlm_like(), lr_multipliers=multipliers)
        rates = optimizer._fallback_lrs()
        state = optimizer.state_dict()
        del state["_lead_group_lr"], state["_adam_group_lrs"]
        optimizer.load_state_dict(state)
        assert optimizer._fallback_lrs() == rates

    @pytest.mark.parametrize(
        "rates",
        [[None, 6e-4], [None, 6e-4, 1.5e-4, 1e-3], 6e-4],
        ids=["fewer", "more", "not-a-list"],
    )
    def test_fallback_rates_that_do_not_fit_its_groups_are_not_taken(
        self, caplog, monkeypatch, rates
    ):
        """A state from a Muon whose fallback had other groups keeps this one's rates, the
        rate of Muon's lead group included, so its next step still runs."""
        monkeypatch.setattr(logging.getLogger("kempnerforge"), "propagate", True)
        model = _vlm_like()
        optimizer = self._muon(model, lr_multipliers={"vision_encoder": 0.5, "adapter": 2.0})
        own = optimizer._fallback_lrs()
        state = optimizer.state_dict()
        state["_adam_group_lrs"] = rates
        state["_lead_group_lr"] = 1.0
        with caplog.at_level(logging.WARNING, logger="kempnerforge.training.optimizer"):
            optimizer.load_state_dict(state)
        assert optimizer._fallback_lrs() == own
        assert [m for m in caplog.messages if "fallback the base rates" in m] == [
            f"The Muon state gives the AdamW fallback the base rates {rates!r}, not one per "
            f"fallback param group (3); keeping this optimizer's own, {optimizer._adam_group_lrs}"
        ]
        _forward(model).backward()
        optimizer.step()
        assert optimizer._adam is not None
        assert [g["lr"] for g in optimizer._adam.param_groups] == own

    def test_a_module_on_the_fallback_alone_is_logged(self):
        """A module with no matrix for Muon still reports its rate, through its fallback."""
        model = _vlm_like()
        patterns = {**VLM_PATTERNS, "norms": ["transformer.1.*"]}
        config = OptimizerConfig(
            name="muon", lr=0.02, muon_adam_lr=3e-4, lr_multipliers={"norms": 0.5}
        )
        optimizer = build_optimizer(model, config, module_patterns=patterns)
        assert not any(g.get("module") == "norms" for g in optimizer.param_groups)
        assert module_learning_rates(optimizer) == {"lr_adamw/norms": pytest.approx(1.5e-4)}


MODEL_CONFIG = ModelConfig(dim=64, n_layers=2, n_heads=2, vocab_size=256, max_seq_len=64)
PATTERNS = {"layer0": ["layers.0.*"], "layer1": ["layers.1.*"], "head": ["output_head.*"]}


class _ReachedTheModel(Exception):
    """Raised in place of the model load, to observe that a load got past every check."""


class TestResume:
    """A load that restores optimizer or scheduler state refuses other multipliers."""

    @staticmethod
    def _build(name, multipliers, seed, patterns=PATTERNS):
        torch.manual_seed(seed)
        model = Transformer(MODEL_CONFIG)
        config = OptimizerConfig(name=name, lr=1e-3, fused=False, lr_multipliers=multipliers)
        optimizer = build_optimizer(model, config, module_patterns=patterns)
        scheduler = build_scheduler(optimizer, SchedulerConfig(warmup_steps=2), max_steps=20)
        return model, optimizer, scheduler

    @staticmethod
    def _train(model, optimizer, scheduler, steps):
        g = torch.Generator().manual_seed(1)
        for _ in range(steps):
            tokens = torch.randint(0, 256, (2, 16), generator=g)
            logits = model(tokens)
            torch.nn.functional.cross_entropy(logits.flatten(0, 1), tokens.flatten()).backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    @staticmethod
    def _manager(tmp_path, model, optimizer, multipliers, patterns=PATTERNS, **kwargs):
        return CheckpointManager(
            CheckpointConfig(dir=str(tmp_path)),
            model,
            optimizer,
            lr_multipliers=multipliers,
            module_patterns=patterns,
            **kwargs,
        )

    def _save(self, tmp_path, multipliers, name="adamw", steps=3):
        model, optimizer, scheduler = self._build(name, multipliers, seed=0)
        self._train(model, optimizer, scheduler, steps)
        self._manager(tmp_path, model, optimizer, multipliers).save(step=steps, scheduler=scheduler)
        return tmp_path / f"step_{steps}"

    def _resume(self, tmp_path, multipliers, name="adamw", patterns=PATTERNS, **kwargs):
        model, optimizer, scheduler = self._build(name, multipliers, seed=1, patterns=patterns)
        manager = self._manager(tmp_path, model, optimizer, multipliers, patterns, **kwargs)
        return model, scheduler, manager

    def test_the_save_records_the_multipliers_by_name(self, tmp_path):
        step_dir = self._save(tmp_path, {"layer0": 0.5, "head": 2.0})
        meta = json.loads((step_dir / "metadata.json").read_text())
        assert list(meta["lr_multipliers"].items()) == [("head", 2.0), ("layer0", 0.5)]

    def test_no_multipliers_record_nothing(self, tmp_path):
        step_dir = self._save(tmp_path, {})
        assert "lr_multipliers" not in json.loads((step_dir / "metadata.json").read_text())

    @pytest.mark.parametrize("name", RESUMABLE_OPTIMIZERS)
    def test_the_same_multipliers_resume_exactly(self, tmp_path, name):
        multipliers = {"layer0": 0.5, "head": 2.0}
        model, optimizer, scheduler = self._build(name, multipliers, seed=0)
        self._train(model, optimizer, scheduler, 3)
        self._manager(tmp_path, model, optimizer, multipliers).save(step=3, scheduler=scheduler)
        rates = [group["lr"] for group in optimizer.param_groups]
        self._train(model, optimizer, scheduler, 2)

        model2, optimizer2, scheduler2 = self._build(name, multipliers, seed=1)
        step, _, _ = self._manager(tmp_path, model2, optimizer2, multipliers).load(
            scheduler=scheduler2
        )
        assert step == 3
        assert [group["lr"] for group in optimizer2.param_groups] == rates
        self._train(model2, optimizer2, scheduler2, 2)
        for a, b in zip(model.parameters(), model2.parameters(), strict=True):
            assert torch.equal(a, b), "the resumed run diverged from the uninterrupted one"

    def test_other_patterns_for_the_same_multipliers_are_refused(self, tmp_path):
        """Swapping two modules' patterns regroups the parameters under the same multipliers."""
        multipliers = {"head": 2.0, "layer0": 0.5}
        self._save(tmp_path, multipliers)
        swapped = {**PATTERNS, "head": PATTERNS["layer0"], "layer0": PATTERNS["head"]}
        model, scheduler, manager = self._resume(tmp_path, multipliers, patterns=swapped)
        before = [p.detach().clone() for p in model.parameters()]
        with pytest.raises(ValueError, match="saved with the module patterns"):
            manager.load(scheduler=scheduler)
        for b, p in zip(before, model.parameters(), strict=True):
            assert torch.equal(b, p), "a refused load changed the model"
        assert scheduler.last_epoch == 0

    def test_the_save_records_the_patterns_of_the_listed_modules(self, tmp_path):
        step_dir = self._save(tmp_path, {"head": 2.0})
        meta = json.loads((step_dir / "metadata.json").read_text())
        assert meta["lr_module_patterns"] == {"head": ["output_head.*"]}

    def test_listed_modules_need_their_patterns(self, tmp_path):
        model, optimizer, _ = self._build("adamw", {}, seed=0)
        with pytest.raises(ValueError, match=r"module_patterns for \['head'\]"):
            CheckpointManager(
                CheckpointConfig(dir=str(tmp_path)), model, optimizer, lr_multipliers={"head": 2.0}
            )

    @pytest.mark.parametrize(
        ("saved", "resumed"),
        [
            ({}, {"head": 2.0}),
            ({"head": 2.0}, {}),
            ({"head": 2.0}, {"head": 3.0}),
            ({"head": 2.0}, {"layer1": 2.0}),
        ],
        ids=["added", "removed", "changed", "moved"],
    )
    def test_other_multipliers_are_refused_before_anything_loads(self, tmp_path, saved, resumed):
        self._save(tmp_path, saved)
        model, scheduler, manager = self._resume(tmp_path, resumed)
        before = [p.detach().clone() for p in model.parameters()]
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"saved with optimizer.lr_multipliers={saved}, but this run uses {resumed}"
            ),
        ):
            manager.load(scheduler=scheduler)
        for b, p in zip(before, model.parameters(), strict=True):
            assert torch.equal(b, p), "a refused load changed the model"
        assert not any(o.state for o in manager.checkpointed_optimizers)
        assert scheduler.last_epoch == 0

    def test_leaving_out_the_optimizer_still_guards_the_scheduler(self, tmp_path):
        """The scheduler's per-group base rates belong to the saved groups too, so the load
        stops before restoring anything from ``train_state.pt``."""
        self._save(tmp_path, {"head": 2.0})
        _, scheduler, manager = self._resume(tmp_path, {})
        rng = torch.get_rng_state()
        with pytest.raises(ValueError, match="but this run uses"):
            manager.load(scheduler=scheduler, exclude_keys=["optimizer"])
        assert scheduler.last_epoch == 0
        assert torch.equal(torch.get_rng_state(), rng)
        model, _, manager = self._resume(tmp_path, {})
        before = [p.detach().clone() for p in model.parameters()]
        step, _, _ = manager.load(exclude_keys=["optimizer"])
        assert step == 3
        assert any(not torch.equal(b, p) for b, p in zip(before, model.parameters(), strict=True))

    @pytest.mark.parametrize("train_state", ["without_scheduler", "absent"])
    def test_no_scheduler_state_to_restore_goes_ahead(self, tmp_path, train_state):
        """Without the optimizer, a load that restores no scheduler state is not checked."""
        model, optimizer, scheduler = self._build("adamw", {"head": 2.0}, seed=0)
        self._train(model, optimizer, scheduler, 3)
        self._manager(tmp_path, model, optimizer, {"head": 2.0}).save(step=3)
        if train_state == "absent":
            (tmp_path / "step_3" / "train_state.pt").unlink()
        _, scheduler, manager = self._resume(tmp_path, {})
        step, _, _ = manager.load(scheduler=scheduler, exclude_keys=["optimizer"])
        assert step == (3 if train_state == "without_scheduler" else 0)
        assert scheduler.last_epoch == 0

    def test_optimizer_state_without_train_state_is_still_checked(self, tmp_path):
        (self._save(tmp_path, {"head": 2.0}) / "train_state.pt").unlink()
        _, scheduler, manager = self._resume(tmp_path, {"head": 3.0})
        with pytest.raises(ValueError, match="but this run uses"):
            manager.load(scheduler=scheduler)
        _, _, scope_and_multipliers = self._resume(
            tmp_path, {"head": 3.0}, weight_decay_scope="exclude_norms_and_biases"
        )
        with pytest.raises(ValueError, match="but this run uses"):
            scope_and_multipliers.load()

    def test_weights_alone_load_under_any_multipliers(self, tmp_path):
        """No optimizer state and no ``train_state.pt``: nothing laid out by group is restored."""
        step_dir = self._save(tmp_path, {"head": 2.0}, steps=0)
        (step_dir / "train_state.pt").unlink()
        (step_dir / "metadata.json").unlink()
        _, scheduler, manager = self._resume(tmp_path, {"layer0": 0.5})
        assert manager.load(scheduler=scheduler)[0] == 0
        assert not any(o.state for o in manager.checkpointed_optimizers)

    def test_a_checkpoint_without_metadata_counts_as_saved_without_multipliers(self, tmp_path):
        (self._save(tmp_path, {}) / "metadata.json").unlink()
        _, scheduler, manager = self._resume(tmp_path, {})
        assert manager.load(scheduler=scheduler)[0] == 3
        _, scheduler, manager = self._resume(tmp_path, {"head": 2.0})
        with pytest.raises(ValueError, match=r"saved with optimizer.lr_multipliers=\{\}"):
            manager.load(scheduler=scheduler)

    def test_a_record_of_a_module_name_with_a_slash_is_read(self, tmp_path):
        """A record keeps any module name, so the load refuses it for its multipliers."""
        step_dir = self._save(tmp_path, {"head": 2.0})
        meta = json.loads((step_dir / "metadata.json").read_text())
        meta["lr_multipliers"] = {"a/b": 2.0}
        meta["lr_module_patterns"] = {"a/b": ["output_head.*"]}
        (step_dir / "metadata.json").write_text(json.dumps(meta))
        _, scheduler, manager = self._resume(tmp_path, {"head": 2.0})
        with pytest.raises(ValueError, match=r"saved with optimizer.lr_multipliers=\{'a/b': 2.0\}"):
            manager.load(scheduler=scheduler)

    @pytest.mark.parametrize(
        ("content", "match"),
        [
            (b"{not json", "cannot read"),
            (b"[]", "does not hold a JSON object"),
            (b'{"step": 3, "lr_multipliers": {"head": -1}}', "records an invalid LR multipliers"),
            (b'{"step": 3, "lr_module_patterns": {"head": "x"}}', "lists of patterns"),
            (b'{"step": 3, "lr_group_modules": "head"}', "a module name or null per"),
            (b'{"step": 3, "lr_group_modules": [null, 2]}', "a module name or null per"),
            (b"[" * 100_000 + b"]" * 100_000, "RecursionError"),
        ],
        ids=["bad-json", "list", "negative", "patterns", "group-modules", "group-module", "deep"],
    )
    @pytest.mark.parametrize("exclude", [[], ["optimizer"]], ids=["resume", "warm-start"])
    def test_an_invalid_record_is_refused(self, tmp_path, content, match, exclude):
        """A run with multipliers cannot tell that the checkpoint's match without the record."""
        (self._save(tmp_path, {"head": 2.0}) / "metadata.json").write_bytes(content)
        _, scheduler, manager = self._resume(tmp_path, {"head": 2.0})
        with pytest.raises(ValueError, match=f"Cannot check the lr_multipliers.*{match}"):
            manager.load(scheduler=scheduler, exclude_keys=exclude)
        assert scheduler.last_epoch == 0

    @pytest.mark.parametrize(
        "content",
        [
            b"{",
            b"[]",
            b'{"step": 3, "lr_multipliers": null}',
            b'{"step": 3, "lr_multipliers": {"head": -1}}',
        ],
        ids=["truncated", "list", "null", "negative"],
    )
    def test_a_run_without_multipliers_loads_past_an_unreadable_record(
        self, tmp_path, caplog, monkeypatch, content
    ):
        """It cannot have been saved with multipliers this run lacks, as far as the load can
        tell, so the load goes ahead with a warning instead of refusing a plain warm start."""
        monkeypatch.setattr(logging.getLogger("kempnerforge"), "propagate", True)
        (self._save(tmp_path, {}) / "metadata.json").write_bytes(content)
        model, scheduler, manager = self._resume(tmp_path, {})
        before = [p.detach().clone() for p in model.parameters()]
        with caplog.at_level(logging.WARNING, logger="kempnerforge.checkpoint.manager"):
            step, _, _ = manager.load(scheduler=scheduler, exclude_keys=["optimizer"])
        assert (step, scheduler.last_epoch) == (3, 3)
        assert any(not torch.equal(b, p) for b, p in zip(before, model.parameters(), strict=True))
        assert [m for m in caplog.messages if "lr_multipliers record" in m] == [
            f"Cannot read the lr_multipliers record of {tmp_path / 'step_3'} "
            f"({self._unreadable(tmp_path / 'step_3', content)}); this run has no "
            "lr_multipliers, so it loads the checkpoint as one saved without them"
        ]

    @staticmethod
    def _unreadable(step_dir, content):
        """What the multipliers check reports about a ``metadata.json`` holding ``content``."""
        path = step_dir / "metadata.json"
        if content == b"[]":
            return f"{path} does not hold a JSON object"
        try:
            meta = json.loads(content)
        except ValueError as e:
            return f"cannot read {path}: {e}"
        try:
            checked_lr_multipliers(meta["lr_multipliers"])
        except ValueError as e:
            return f"{path} records an invalid LR multipliers record: {e}"
        raise AssertionError(content)

    def test_a_run_without_multipliers_refuses_a_record_of_some(self, tmp_path):
        """Only the multipliers decide, so a record of some refuses the load even when the rest
        of it cannot be read."""
        step_dir = self._save(tmp_path, {"head": 2.0})
        meta = json.loads((step_dir / "metadata.json").read_text())
        meta["lr_module_patterns"] = "x"
        (step_dir / "metadata.json").write_text(json.dumps(meta))
        _, scheduler, manager = self._resume(tmp_path, {})
        with pytest.raises(ValueError, match=r"lr_multipliers=\{'head': 2.0\}, but this run uses"):
            manager.load(scheduler=scheduler, exclude_keys=["optimizer"])
        assert scheduler.last_epoch == 0

    @pytest.mark.parametrize("scope", [None, "exclude_norms_and_biases"], ids=["no-scope", "scope"])
    def test_a_resume_reads_the_record_once(self, tmp_path, monkeypatch, scope):
        """One read of ``metadata.json`` and one check of the multipliers serve the optimizer
        state and the scheduler state alike."""
        self._save(tmp_path, {"head": 2.0})
        _, scheduler, manager = self._resume(tmp_path, {"head": 2.0}, weight_decay_scope=scope)
        reads: list[str] = []
        read_text = Path.read_text

        def counted(path, *args, **kwargs):
            if path.name == "metadata.json":
                reads.append(str(path))
            return read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", counted)
        checks: list[object] = []
        check = manager._check_lr_multipliers
        monkeypatch.setattr(
            manager, "_check_lr_multipliers", lambda *a: checks.append(a) or check(*a)
        )
        assert manager.load(scheduler=scheduler)[0] == 3
        assert len(reads) == 1
        assert len(checks) == 1
        assert scheduler.last_epoch == 3

    @staticmethod
    def _as_rank(monkeypatch, manager, rank, sent=()):
        """Run ``manager`` as ``rank`` of two, the other rank adding nothing to the all-reduce.

        ``sent`` lists what rank 0's broadcasts deliver to any other rank, in order. Returns the
        collectives issued, in order.
        """
        import kempnerforge.checkpoint.manager as mgr_mod

        calls: list[tuple] = []
        delivered = iter(sent)

        def all_reduce(flags, group=None):
            calls.append(("all_reduce", group, flags.tolist()))

        def broadcast_object_list(objects, src):
            calls.append(("broadcast", src, list(objects)))
            if rank != 0:
                objects[0] = next(delivered)

        monkeypatch.setattr(mgr_mod.dist, "is_initialized", lambda: True)
        monkeypatch.setattr(mgr_mod.dist, "get_rank", lambda group=None: rank)
        monkeypatch.setattr(mgr_mod.dist, "get_world_size", lambda group=None: 2)
        monkeypatch.setattr(mgr_mod.dist, "all_reduce", all_reduce)
        monkeypatch.setattr(mgr_mod.dist, "broadcast_object_list", broadcast_object_list)
        manager._rank = rank
        manager._sync_group = "GLOO_GROUP"
        return calls

    @staticmethod
    def _stop_at_the_model_load(monkeypatch):
        """Make the load stop, with ``_ReachedTheModel``, once it starts loading the model."""
        import kempnerforge.checkpoint.manager as mgr_mod

        def reached(_model):
            raise _ReachedTheModel

        monkeypatch.setattr(mgr_mod, "get_model_state_dict", reached)

    RECORD = {"lr_multipliers": {"head": 2.0}, "lr_module_patterns": {"head": ["output_head.*"]}}

    def test_rank_zero_reads_the_record_and_broadcasts_it(self, tmp_path, monkeypatch):
        """Rank 0 sends the whole ``metadata.json``, once, for every check of the load."""
        step_dir = self._save(tmp_path, {"head": 2.0})
        meta = json.loads((step_dir / "metadata.json").read_text())
        _, scheduler, manager = self._resume(tmp_path, {"head": 2.0})
        calls = self._as_rank(monkeypatch, manager, 0)
        self._stop_at_the_model_load(monkeypatch)
        with pytest.raises(_ReachedTheModel):
            manager.load(scheduler=scheduler)
        assert calls == [
            ("all_reduce", "GLOO_GROUP", [[0, 0], [1, 0]]),
            ("broadcast", 0, [(meta, None)]),
            ("broadcast", 0, [True]),
        ]

    def test_other_ranks_refuse_what_rank_zero_sent(self, tmp_path, monkeypatch):
        """A rank other than 0 never reads the record and raises on the one it receives."""
        self._save(tmp_path, {"head": 2.0})
        (tmp_path / "step_3" / "metadata.json").unlink()
        _, scheduler, manager = self._resume(tmp_path, {})
        _, listed_scheduler, listed = self._resume(tmp_path, {"head": 2.0})
        self._as_rank(monkeypatch, manager, 1, sent=[(self.RECORD, None)])
        with pytest.raises(ValueError, match=r"lr_multipliers=\{'head': 2.0\}, but this run uses"):
            manager.load(scheduler=scheduler)
        self._as_rank(monkeypatch, listed, 1, sent=[(None, "rank 0 could not read it")])
        with pytest.raises(ValueError, match="rank 0 could not read it"):
            listed.load(scheduler=listed_scheduler)

    def test_a_failure_on_rank_zero_is_broadcast(self, tmp_path, monkeypatch):
        self._save(tmp_path, {"head": 2.0})
        _, scheduler, manager = self._resume(tmp_path, {"head": 2.0})

        def fail(_ckpt_dir):
            raise OSError("disk gone")

        monkeypatch.setattr("kempnerforge.checkpoint.manager._read_metadata", fail)
        calls = self._as_rank(monkeypatch, manager, 0)
        with pytest.raises(ValueError, match="Cannot check the lr_multipliers.*disk gone"):
            manager.load(scheduler=scheduler)
        assert calls[-1] == ("broadcast", 0, [(None, "OSError: disk gone")])

    def test_a_refusal_on_another_rank_stops_this_one_before_loading(self, tmp_path, monkeypatch):
        """The ranks agree on the optimizer checks before any rank loads."""
        import kempnerforge.checkpoint.manager as mgr_mod

        self._save(tmp_path, {"head": 2.0})
        model, scheduler, manager = self._resume(tmp_path, {"head": 2.0})
        calls = self._as_rank(monkeypatch, manager, 1, sent=[(self.RECORD, None), True])

        def all_reduce(flags, group=None):
            calls.append(("all_reduce", group, flags.tolist()))
            if flags.dim() == 1:
                flags[0] = 1

        monkeypatch.setattr(mgr_mod.dist, "all_reduce", all_reduce)
        before = [p.detach().clone() for p in model.parameters()]
        with pytest.raises(RuntimeError, match=r"Checkpoint load failed on ranks \[0\]"):
            manager.load(scheduler=scheduler)
        assert calls[-1] == ("all_reduce", "GLOO_GROUP", [0, 0])
        assert all(torch.equal(b, p) for b, p in zip(before, model.parameters(), strict=True))
        assert scheduler.last_epoch == 0

    @pytest.mark.parametrize(
        "multipliers", [{}, {"head": 2.0}, None], ids=["none", "listed", "unset"]
    )
    @pytest.mark.parametrize("exclude", [[], ["optimizer"]], ids=["optimizer", "no-optimizer"])
    def test_every_load_of_optimizer_state_agrees_on_the_checks(
        self, tmp_path, monkeypatch, multipliers, exclude
    ):
        """A rank's checks pass or fail on its own part of the checkpoint, so whether the ranks
        agree on them depends only on the request, which every rank shares."""
        import kempnerforge.checkpoint.manager as mgr_mod

        self._save(tmp_path, multipliers or {})
        model, optimizer, scheduler = self._build("adamw", multipliers or {}, seed=1)
        manager = self._manager(tmp_path, model, optimizer, multipliers)
        calls = self._as_rank(monkeypatch, manager, 0)

        def reached(*_args, **_kwargs):
            raise _ReachedTheModel

        monkeypatch.setattr(mgr_mod.dcp, "load", reached)
        with pytest.raises(_ReachedTheModel):
            manager.load(scheduler=scheduler, exclude_keys=exclude)
        agreed = [c for c in calls if c[0] == "all_reduce" and c[2] == [0, 0]]
        assert agreed == ([] if exclude else [("all_reduce", "GLOO_GROUP", [0, 0])])

    def test_every_rank_checks_the_scheduler_state_rank_zero_sent(self, tmp_path, monkeypatch):
        """Each rank decides from the train state rank 0 broadcast, so all check or none do."""
        step_dir = self._save(tmp_path, {"head": 2.0})
        train_state = torch.load(step_dir / "train_state.pt", weights_only=False)
        _, scheduler, manager = self._resume(tmp_path, {})
        sent = [True, train_state, (self.RECORD, None)]
        calls = self._as_rank(monkeypatch, manager, 1, sent=sent)
        with pytest.raises(ValueError, match="but this run uses"):
            manager.load(scheduler=scheduler, exclude_keys=["model", "optimizer"])
        assert len(calls) == 3
        assert scheduler.last_epoch == 0
        without_scheduler = {k: v for k, v in train_state.items() if k != "scheduler"}
        calls = self._as_rank(monkeypatch, manager, 1, sent=[True, without_scheduler])
        assert manager.load(scheduler=scheduler, exclude_keys=["model", "optimizer"])[0] == 3
        assert len(calls) == 2


class TestLayoutChange:
    """A load that restores the scheduler into param groups laid out otherwise.

    Which groups a run with ``optimizer.lr_multipliers`` has depends on the trainable
    parameters and on ``weight_decay_scope``, since groups with no parameters are left out.
    Each group resumes at its own module's rate whatever the two layouts are.
    """

    MULTIPLIERS = {"vision_encoder": 0.1, "adapter": 5.0}
    LR = 1e-3
    ADAM_LR = 4e-4
    DEFAULT_SCOPE = "exclude_norms_and_biases"
    SCENARIOS = {
        "unfrozen": (({"transformer"}, DEFAULT_SCOPE), (set(), DEFAULT_SCOPE)),
        "scope": ((set(), "all"), (set(), DEFAULT_SCOPE)),
        "swapped": (({"transformer"}, DEFAULT_SCOPE), ({"vision_encoder"}, DEFAULT_SCOPE)),
        "frozen": ((set(), DEFAULT_SCOPE), ({"vision_encoder"}, DEFAULT_SCOPE)),
    }

    @classmethod
    def _build(cls, name, frozen, scope, multipliers=None, lr=LR, adam_lr=ADAM_LR):
        model = _vlm_like()
        for module in frozen:
            getattr(model, module).requires_grad_(False)
        config = OptimizerConfig(
            name=name,
            lr=lr,
            muon_adam_lr=adam_lr,
            fused=False,
            weight_decay_scope=scope,
            lr_multipliers=cls.MULTIPLIERS if multipliers is None else multipliers,
        )
        optimizer = build_optimizer(model, config, module_patterns=VLM_PATTERNS)
        schedule = SchedulerConfig(warmup_steps=2, min_lr_ratio=0.1)
        return model, optimizer, build_scheduler(optimizer, schedule, max_steps=20)

    @staticmethod
    def _train(model, optimizer, scheduler, steps):
        for _ in range(steps):
            _forward(model).backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

    @classmethod
    def _manager(cls, tmp_path, model, optimizer, multipliers=None):
        return CheckpointManager(
            CheckpointConfig(dir=str(tmp_path)),
            model,
            optimizer,
            lr_multipliers=cls.MULTIPLIERS if multipliers is None else multipliers,
            module_patterns=VLM_PATTERNS,
        )

    @staticmethod
    def _group_layout(optimizer):
        optimizers = (optimizer, *getattr(optimizer, "inner_optimizers", ()))
        return [[(g.get("module"), g["weight_decay"]) for g in o.param_groups] for o in optimizers]

    def _assert_module_rates(self, optimizer, scheduler, lr=LR, adam_lr=ADAM_LR):
        """Every group, a Muon fallback's included, steps next at its module's scheduled rate
        from the peak ``lr``, ``adam_lr`` for the fallback."""
        factor = scheduler.lr_lambdas[0](scheduler.last_epoch)

        def expected(groups, base):
            return [base * self.MULTIPLIERS.get(g.get("module"), 1.0) * factor for g in groups]

        groups = optimizer.param_groups
        assert [g["lr"] for g in groups] == pytest.approx(expected(groups, lr), rel=1e-12)
        if isinstance(optimizer, Muon):
            assert optimizer._adam is not None
            fallback = optimizer._adam.param_groups
            assert optimizer._fallback_lrs() == pytest.approx(
                expected(fallback, adam_lr), rel=1e-12
            )

    @pytest.mark.parametrize("scenario", sorted(SCENARIOS))
    @pytest.mark.parametrize("name", RESUMABLE_OPTIMIZERS)
    def test_a_warm_start_resumes_each_group_at_its_module_rate(self, tmp_path, name, scenario):
        saved, resumed = self.SCENARIOS[scenario]
        model, optimizer, scheduler = self._build(name, *saved)
        self._train(model, optimizer, scheduler, 4)
        self._manager(tmp_path, model, optimizer).save(step=4, scheduler=scheduler)
        saved_layout = self._group_layout(optimizer)

        model, optimizer, scheduler = self._build(name, *resumed)
        assert self._group_layout(optimizer) != saved_layout
        step, _, _ = self._manager(tmp_path, model, optimizer).load(
            scheduler=scheduler, exclude_keys=["optimizer"]
        )
        assert (step, scheduler.last_epoch) == (4, 4)
        self._train(model, optimizer, scheduler, 1)
        self._assert_module_rates(optimizer, scheduler)

    @pytest.mark.parametrize("name", RESUMABLE_OPTIMIZERS)
    def test_groups_of_no_module_that_change_in_number_resume_at_their_rate(self, tmp_path, name):
        """With every listed module frozen no group names a module, and the scope sets how
        many groups hold the rest."""
        multipliers = {"vision_encoder": 0.1}
        frozen = {"vision_encoder"}
        model, optimizer, scheduler = self._build(name, frozen, "all", multipliers)
        self._train(model, optimizer, scheduler, 4)
        self._manager(tmp_path, model, optimizer, multipliers).save(step=4, scheduler=scheduler)
        saved_layout = self._group_layout(optimizer)

        model, optimizer, scheduler = self._build(name, frozen, self.DEFAULT_SCOPE, multipliers)
        assert self._group_layout(optimizer) != saved_layout
        step, _, _ = self._manager(tmp_path, model, optimizer, multipliers).load(
            scheduler=scheduler, exclude_keys=["optimizer"]
        )
        assert step == 4
        self._train(model, optimizer, scheduler, 1)
        self._assert_module_rates(optimizer, scheduler)

    LR_CHANGES = {
        "same-layout": ({"transformer", "vision_encoder"}, {"transformer", "vision_encoder"}),
        "unfrozen": ({"transformer", "vision_encoder"}, {"vision_encoder"}),
        "no-module-held": ({"transformer", "vision_encoder"}, {"adapter", "vision_encoder"}),
    }

    @pytest.mark.parametrize("change", sorted(LR_CHANGES))
    @pytest.mark.parametrize("name", RESUMABLE_OPTIMIZERS)
    def test_a_warm_start_with_another_lr_keeps_the_saved_peak_in_every_group(
        self, tmp_path, name, change
    ):
        """The restored schedule wins over ``optimizer.lr``, its peak included, whatever the
        layout: a group of a module the checkpoint holds no group of runs at the saved peak
        times its multiplier, so the multipliers keep their ratio."""
        saved, resumed = self.LR_CHANGES[change]
        model, optimizer, scheduler = self._build(name, saved, self.DEFAULT_SCOPE, adam_lr=None)
        self._train(model, optimizer, scheduler, 4)
        self._manager(tmp_path, model, optimizer).save(step=4, scheduler=scheduler)

        model, optimizer, scheduler = self._build(
            name, resumed, self.DEFAULT_SCOPE, lr=self.LR / 10, adam_lr=None
        )
        step, _, _ = self._manager(tmp_path, model, optimizer).load(
            scheduler=scheduler, exclude_keys=["optimizer"]
        )
        assert (step, scheduler.last_epoch) == (4, 4)
        for _ in range(3):
            self._train(model, optimizer, scheduler, 1)
            self._assert_module_rates(optimizer, scheduler, adam_lr=self.LR)

    def test_the_load_logs_the_peak_each_module_resumes_at(self, tmp_path, caplog, monkeypatch):
        monkeypatch.setattr(logging.getLogger("kempnerforge"), "propagate", True)
        model, optimizer, scheduler = self._build("adamw", {"transformer", "vision_encoder"}, "all")
        self._train(model, optimizer, scheduler, 4)
        self._manager(tmp_path, model, optimizer).save(step=4, scheduler=scheduler)

        model, optimizer, scheduler = self._build(
            "adamw", {"vision_encoder"}, "all", lr=self.LR / 10
        )
        with caplog.at_level(logging.INFO, logger="kempnerforge.checkpoint.state"):
            self._manager(tmp_path, model, optimizer).load(
                scheduler=scheduler, exclude_keys=["optimizer"]
            )
        assert scheduler.base_lrs == [self.LR, self.LR * 5.0]
        assert (
            "Restored scheduler state; each module's param groups run the restored schedule "
            f"from peak lr { ({None: self.LR, 'adapter': self.LR * 5.0}) }"
        ) in caplog.messages
        built = {None: self.LR / 10, "adapter": self.LR / 10 * 5.0}
        warnings = _restore_warnings(caplog)
        assert warnings == [
            f"The restored schedule runs the param groups from peak lr "
            f"{ ({None: self.LR, 'adapter': self.LR * 5.0}) }, while the optimizer was built at "
            f"{built}: a load that restores train_state.pt continues the saved schedule, its "
            "peak included, so a changed optimizer.lr does not apply. A stage with a new "
            "optimizer.lr needs a warm start from the weights alone."
        ]

    @pytest.mark.parametrize(
        ("name", "saved", "resumed"),
        [
            ("adamw", set(), {"vision_encoder"}),
            ("lion", set(), {"adapter"}),
            ("muon", set(), {"vision_encoder"}),
            ("adamw", {"vision_encoder"}, {"transformer"}),
            ("adamw", {"transformer"}, {"vision_encoder"}),
            ("lion", {"transformer"}, {"vision_encoder"}),
        ],
        ids=[
            "last-module-frozen",
            "middle-module-frozen",
            "muon",
            "as-many-groups-named-now",
            "as-many-groups-named-at-save",
            "as-many-groups-named-at-save-lion",
        ],
    )
    def test_a_resume_into_other_groups_is_refused_before_loading(
        self, tmp_path, name, saved, resumed
    ):
        """Saved settings sit at their group's position, so other groups cannot take them."""
        model, optimizer, scheduler = self._build(name, saved, self.DEFAULT_SCOPE)
        self._train(model, optimizer, scheduler, 4)
        self._manager(tmp_path, model, optimizer).save(step=4, scheduler=scheduler)

        model, optimizer, scheduler = self._build(name, resumed, self.DEFAULT_SCOPE)
        before = [p.detach().clone() for p in model.parameters()]
        torch.rand(1)
        rng = torch.get_rng_state()
        with pytest.raises(RuntimeError, match="laid out otherwise than at the save"):
            self._manager(tmp_path, model, optimizer).load(scheduler=scheduler)
        assert all(torch.equal(b, p) for b, p in zip(before, model.parameters(), strict=True))
        assert torch.equal(torch.get_rng_state(), rng)
        assert scheduler.last_epoch == 0
        step, _, _ = self._manager(tmp_path, model, optimizer).load(
            scheduler=scheduler, exclude_keys=["optimizer"]
        )
        assert step == 4
        self._train(model, optimizer, scheduler, 1)
        self._assert_module_rates(optimizer, scheduler)

    @pytest.mark.parametrize("name", ["adamw", "lion"])
    def test_a_resume_into_fewer_groups_of_no_module_is_refused(self, tmp_path, name):
        """With every listed module frozen no group names a module, and freezing every decayed
        weight still drops a group the checkpoint holds."""
        multipliers = {"vision_encoder": 0.1}
        frozen = {"vision_encoder"}
        model, optimizer, scheduler = self._build(name, frozen, self.DEFAULT_SCOPE, multipliers)
        self._train(model, optimizer, scheduler, 4)
        self._manager(tmp_path, model, optimizer, multipliers).save(step=4, scheduler=scheduler)

        model = _vlm_like()
        model.vision_encoder.requires_grad_(False)
        model.transformer[0].weight.requires_grad_(False)
        model.adapter.weight.requires_grad_(False)
        config = OptimizerConfig(name=name, lr=self.LR, fused=False, lr_multipliers=multipliers)
        optimizer = build_optimizer(model, config, module_patterns=VLM_PATTERNS)
        schedule = SchedulerConfig(warmup_steps=2, min_lr_ratio=0.1)
        scheduler = build_scheduler(optimizer, schedule, max_steps=20)
        assert [g.get("module") for g in optimizer.param_groups] == [None]
        manager = self._manager(tmp_path, model, optimizer, multipliers)
        with pytest.raises(RuntimeError, match="holds 2 optimizer param groups, and this run 1"):
            manager.load(scheduler=scheduler)
        assert scheduler.last_epoch == 0
        assert manager.load(scheduler=scheduler, exclude_keys=["optimizer"])[0] == 4
        self._train(model, optimizer, scheduler, 1)
        self._assert_module_rates(optimizer, scheduler)

    def test_the_same_layout_resumes_the_saved_scheduler_state_as_it_is(self, tmp_path):
        model, optimizer, scheduler = self._build("adamw", {"transformer"}, self.DEFAULT_SCOPE)
        self._train(model, optimizer, scheduler, 4)
        self._manager(tmp_path, model, optimizer).save(step=4, scheduler=scheduler)
        model2, optimizer2, scheduler2 = self._build("adamw", {"transformer"}, self.DEFAULT_SCOPE)
        self._manager(tmp_path, model2, optimizer2).load(scheduler=scheduler2)
        assert scheduler2.state_dict() == scheduler.state_dict()

    @pytest.mark.parametrize(
        ("name", "modules"),
        [
            ("adamw", [None, None, "adapter", "adapter"]),
            ("muon", [None, "adapter", None, "adapter"]),
        ],
    )
    def test_the_save_records_the_module_of_each_group(self, tmp_path, name, modules):
        """In the order the checkpoint keeps the groups, a Muon's AdamW fallback after its own."""
        model, optimizer, scheduler = self._build(name, {"vision_encoder"}, self.DEFAULT_SCOPE)
        self._train(model, optimizer, scheduler, 1)
        self._manager(tmp_path, model, optimizer).save(step=1, scheduler=scheduler)
        meta = json.loads((tmp_path / "step_1" / "metadata.json").read_text())
        assert meta["lr_group_modules"] == modules

    @pytest.mark.parametrize(("name", "index"), [("adamw", 2), ("lion", 2), ("muon", 1)])
    def test_a_resume_into_groups_of_other_modules_is_refused_before_loading(
        self, tmp_path, name, index
    ):
        """As many groups with the same settings, but another module's at a position: the
        module each saved group recorded names the change before anything loads."""
        model, optimizer, scheduler = self._build(name, {"vision_encoder"}, self.DEFAULT_SCOPE)
        self._train(model, optimizer, scheduler, 4)
        self._manager(tmp_path, model, optimizer).save(step=4, scheduler=scheduler)

        model, optimizer, scheduler = self._build(name, {"adapter"}, self.DEFAULT_SCOPE)
        manager = self._manager(tmp_path, model, optimizer)
        before = [p.detach().clone() for p in model.parameters()]
        torch.rand(1)
        rng = torch.get_rng_state()
        change = (
            f"has optimizer param group {index} of module 'adapter', where this run's group "
            f"{index} is of module 'vision_encoder'"
        )
        with pytest.raises(RuntimeError, match=re.escape(change)):
            manager.load(scheduler=scheduler)
        assert all(torch.equal(b, p) for b, p in zip(before, model.parameters(), strict=True))
        assert torch.equal(torch.get_rng_state(), rng)
        assert scheduler.last_epoch == 0

    def test_a_group_record_of_another_length_is_refused(self, tmp_path):
        model, optimizer, scheduler = self._build("adamw", {"vision_encoder"}, self.DEFAULT_SCOPE)
        self._train(model, optimizer, scheduler, 4)
        self._manager(tmp_path, model, optimizer).save(step=4, scheduler=scheduler)
        path = tmp_path / "step_4" / "metadata.json"
        meta = json.loads(path.read_text())
        meta["lr_group_modules"] = meta["lr_group_modules"][:-1]
        path.write_text(json.dumps(meta))
        model, optimizer, scheduler = self._build("adamw", {"vision_encoder"}, self.DEFAULT_SCOPE)
        manager = self._manager(tmp_path, model, optimizer)
        with pytest.raises(
            RuntimeError, match="records 3 optimizer param groups, and this run has"
        ):
            manager.load(scheduler=scheduler)

    def test_a_checkpoint_without_the_group_record_resumes_by_its_keys(self, tmp_path):
        """A save that recorded no group modules still resumes into the same groups."""
        model, optimizer, scheduler = self._build("adamw", {"vision_encoder"}, self.DEFAULT_SCOPE)
        self._train(model, optimizer, scheduler, 4)
        self._manager(tmp_path, model, optimizer).save(step=4, scheduler=scheduler)
        path = tmp_path / "step_4" / "metadata.json"
        meta = json.loads(path.read_text())
        del meta["lr_group_modules"]
        path.write_text(json.dumps(meta))
        model2, optimizer2, scheduler2 = self._build(
            "adamw", {"vision_encoder"}, self.DEFAULT_SCOPE
        )
        assert self._manager(tmp_path, model2, optimizer2).load(scheduler=scheduler2)[0] == 4
        assert [(g.get("module"), g["lr"]) for g in optimizer2.param_groups] == [
            (g.get("module"), g["lr"]) for g in optimizer.param_groups
        ]
        for p, p2 in zip(model.parameters(), model2.parameters(), strict=True):
            assert torch.equal(p, p2)
            if p in optimizer.state:
                assert torch.equal(optimizer2.state[p2]["exp_avg"], optimizer.state[p]["exp_avg"])

    def test_the_records_describe_the_ownership_of_nested_modules(self, tmp_path):
        """The saved multipliers, patterns and group modules pin down which module owns each
        parameter, so the same ones resume into the same groups and other patterns are
        refused."""
        multipliers = {"transformer": 0.5, "inner": 2.0}
        nested = {**VLM_PATTERNS, "inner": ["transformer.0.*"]}

        def build(patterns):
            model = _vlm_like()
            config = OptimizerConfig(lr=self.LR, fused=False, lr_multipliers=multipliers)
            optimizer = build_optimizer(model, config, module_patterns=patterns)
            schedule = SchedulerConfig(warmup_steps=2, min_lr_ratio=0.1)
            return model, optimizer, build_scheduler(optimizer, schedule, max_steps=20)

        def manager(model, optimizer, patterns):
            return CheckpointManager(
                CheckpointConfig(dir=str(tmp_path)),
                model,
                optimizer,
                lr_multipliers=multipliers,
                module_patterns=patterns,
            )

        model, optimizer, scheduler = build(nested)
        self._train(model, optimizer, scheduler, 3)
        manager(model, optimizer, nested).save(step=3, scheduler=scheduler)
        meta = json.loads((tmp_path / "step_3" / "metadata.json").read_text())
        assert meta["lr_group_modules"] == [None, None, "inner", "inner", "transformer"]
        assert meta["lr_module_patterns"] == {
            "inner": ["transformer.0.*"],
            "transformer": ["transformer", "transformer.*"],
        }
        model2, optimizer2, scheduler2 = build(nested)
        assert manager(model2, optimizer2, nested).load(scheduler=scheduler2)[0] == 3
        assert _layout(model2, optimizer2) == _layout(model, optimizer)
        moved = {**nested, "inner": ["transformer.1.*"]}
        model3, optimizer3, scheduler3 = build(moved)
        with pytest.raises(ValueError, match="saved with the module patterns"):
            manager(model3, optimizer3, moved).load(scheduler=scheduler3)

    def test_a_save_under_pipeline_parallelism_records_no_group_modules(self, tmp_path):
        """Rank 0 holds one stage's groups, so the record would not describe the others'."""
        model, optimizer, scheduler = self._build("adamw", set(), self.DEFAULT_SCOPE)
        self._train(model, optimizer, scheduler, 1)
        CheckpointManager(
            CheckpointConfig(dir=str(tmp_path)),
            model,
            optimizer,
            pp_rank=0,
            lr_multipliers=self.MULTIPLIERS,
            module_patterns=VLM_PATTERNS,
        ).save(step=1, scheduler=scheduler)
        meta = json.loads((tmp_path / "step_1" / "metadata.json").read_text())
        assert meta["lr_multipliers"] == self.MULTIPLIERS
        assert "lr_group_modules" not in meta


class TestSchedulerStateLayout:
    """The train state records each scheduled group's module and restores by it."""

    @staticmethod
    def _optimizer(groups):
        """An optimizer whose param groups are given as ``(module, lr)`` pairs."""
        param_groups = []
        for module, lr in groups:
            group = {"params": [nn.Parameter(torch.zeros(1))], "lr": lr}
            if module is not None:
                group["module"] = module
            param_groups.append(group)
        return torch.optim.SGD(param_groups, lr=1.0)

    def _scheduler(self, groups):
        optimizer = self._optimizer(groups)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 / (step + 1))

    def test_groups_without_a_module_record_nothing(self):
        scheduler = self._scheduler([(None, 0.1), (None, 0.1)])
        state = build_train_state(step=1, tokens_seen=0, scheduler=scheduler)
        assert set(state) == {"step", "tokens_seen", "rng", "scheduler"}

    def test_the_record_lists_every_group_and_is_no_extra(self):
        state = build_train_state(
            step=1, tokens_seen=0, scheduler=self._scheduler([(None, 0.1), ("a", 0.2)])
        )
        assert state["scheduler_group_modules"] == [None, "a"]
        extra = restore_train_state(state, scheduler=self._scheduler([(None, 0.1), ("a", 0.2)]))[2]
        assert extra == {}

    def test_a_state_saved_without_a_record_counts_its_groups_as_moduleless(self):
        """A checkpoint from before groups named modules lays out onto named groups."""
        saved = self._scheduler([(None, 0.3), (None, 0.3)])
        saved.step()
        saved.step()
        state = build_train_state(step=2, tokens_seen=0, scheduler=saved)
        assert "scheduler_group_modules" not in state
        resumed = self._scheduler([(None, 0.1), ("a", 0.2), ("a", 0.2)])
        restore_train_state(state, scheduler=resumed, lr_multipliers={"a": 2.0})
        assert resumed.base_lrs == [0.3, 0.6, 0.6]
        assert resumed.last_epoch == 2
        assert resumed._last_lr == pytest.approx([0.1, 0.2, 0.2])

    def test_a_group_of_a_module_with_no_saved_group_takes_the_saved_peak(self):
        """Its rates are the first saved group's, divided by that group's multiplier and
        multiplied by its own, so the multipliers keep their ratio under the saved peak."""
        saved = self._scheduler([("a", 0.5), ("a", 0.5)])
        saved.step()
        saved.step()
        state = build_train_state(step=2, tokens_seen=0, scheduler=saved)
        resumed = self._scheduler([(None, 0.01), ("b", 0.03), ("a", 0.05)])
        restore_train_state(state, scheduler=resumed, lr_multipliers={"a": 5.0, "b": 3.0})
        assert resumed.base_lrs == pytest.approx([0.1, 0.3, 0.5], rel=1e-15)
        assert resumed._last_lr == pytest.approx([0.1 / 3, 0.3 / 3, 0.5 / 3], rel=1e-15)
        resumed.optimizer.step()
        resumed.step()
        assert [g["lr"] for g in resumed.optimizer.param_groups] == pytest.approx(
            [0.1 / 4, 0.3 / 4, 0.5 / 4], rel=1e-15
        )

    def test_a_saved_peak_unchanged_gives_a_new_group_its_built_rate_exactly(self):
        state = build_train_state(step=0, tokens_seen=0, scheduler=self._scheduler([(None, 0.1)]))
        resumed = self._scheduler([(None, 0.1), ("b", 0.1 * 3.0)])
        restore_train_state(state, scheduler=resumed, lr_multipliers={"b": 3.0})
        assert resumed.base_lrs == [0.1, 0.1 * 3.0]

    @pytest.mark.parametrize("multipliers", [None, {"b": 3.0}], ids=["none", "without-a"])
    def test_a_layout_that_needs_a_missing_multiplier_restores_nothing(self, multipliers):
        state = build_train_state(step=2, tokens_seen=0, scheduler=self._scheduler([("a", 0.5)]))
        torch.rand(1)
        rng = torch.get_rng_state()
        resumed = self._scheduler([(None, 0.1), ("b", 0.3)])
        with pytest.raises(ValueError, match="needs the multiplier of 'a'"):
            restore_train_state(state, scheduler=resumed, lr_multipliers=multipliers)
        assert torch.equal(torch.get_rng_state(), rng)
        assert resumed.last_epoch == 0

    def test_a_state_of_no_group_cannot_be_laid_out(self):
        saved = self._scheduler([(None, 0.1)]).state_dict()
        saved.update(base_lrs=[], _last_lr=[], lr_lambdas=[])
        state = {"step": 2, "scheduler": saved, "scheduler_group_modules": []}
        with pytest.raises(ValueError, match="holds no param group to lay out"):
            restore_train_state(state, scheduler=self._scheduler([("a", 0.1)]), lr_multipliers={})

    def test_the_restore_logs_the_peak_of_each_module(self, caplog, monkeypatch):
        monkeypatch.setattr(logging.getLogger("kempnerforge"), "propagate", True)
        state = build_train_state(
            step=0, tokens_seen=0, scheduler=self._scheduler([(None, 0.1), ("a", 0.2), ("a", 0.2)])
        )
        resumed = self._scheduler([(None, 0.1), ("a", 0.2), ("a", 0.2)])
        with caplog.at_level(logging.INFO, logger="kempnerforge.checkpoint.state"):
            restore_train_state(state, scheduler=resumed)
        assert (
            "Restored scheduler state; each module's param groups run the restored schedule "
            "from peak lr {None: 0.1, 'a': 0.2}"
        ) in caplog.messages
        assert not _restore_warnings(caplog)

    def test_a_restored_peak_other_than_the_built_one_warns(self, caplog, monkeypatch):
        monkeypatch.setattr(logging.getLogger("kempnerforge"), "propagate", True)
        state = build_train_state(
            step=0, tokens_seen=0, scheduler=self._scheduler([(None, 0.1), ("a", 0.5)])
        )
        resumed = self._scheduler([(None, 0.01), ("a", 0.05)])
        with caplog.at_level(logging.INFO, logger="kempnerforge.checkpoint.state"):
            restore_train_state(state, scheduler=resumed, lr_multipliers={"a": 5.0})
        assert resumed.base_lrs == [0.1, 0.5]
        warnings = _restore_warnings(caplog)
        assert len(warnings) == 1
        assert warnings[0].startswith(
            "The restored schedule runs the param groups from peak lr {None: 0.1, 'a': 0.5}, "
            "while the optimizer was built at {None: 0.01, 'a': 0.05}"
        )

    def test_a_derived_peak_off_by_rounding_does_not_warn(self, caplog, monkeypatch):
        """Its peak comes from a saved group's rate divided by that group's multiplier, which
        can differ from the built rate in the last bit when ``optimizer.lr`` did not change."""
        monkeypatch.setattr(logging.getLogger("kempnerforge"), "propagate", True)
        state = build_train_state(step=0, tokens_seen=0, scheduler=self._scheduler([("a", 0.3)]))
        resumed = self._scheduler([(None, 0.1), ("a", 0.3)])
        with caplog.at_level(logging.INFO, logger="kempnerforge.checkpoint.state"):
            restore_train_state(state, scheduler=resumed, lr_multipliers={"a": 3.0})
        assert resumed.base_lrs[0] != 0.1
        assert resumed.base_lrs == pytest.approx([0.1, 0.3], rel=1e-15)
        assert not _restore_warnings(caplog)

    @pytest.mark.parametrize("kind", ["lambda", "step"])
    def test_without_a_module_the_restore_logs_as_before(self, caplog, monkeypatch, kind):
        """No group names a module: the restored base rates win silently, as on main, even
        when they differ from the built ones."""
        monkeypatch.setattr(logging.getLogger("kempnerforge"), "propagate", True)

        def build(groups):
            if kind == "lambda":
                return self._scheduler(groups)
            return torch.optim.lr_scheduler.StepLR(self._optimizer(groups), step_size=1)

        state = build_train_state(step=0, tokens_seen=0, scheduler=build([(None, 0.1)]))
        resumed = build([(None, 0.01)])
        with caplog.at_level(logging.INFO, logger="kempnerforge.checkpoint.state"):
            restore_train_state(state, scheduler=resumed)
        assert resumed.base_lrs == [0.1]
        assert "Restored scheduler state" in caplog.messages
        assert not any("peak lr" in message for message in caplog.messages)
        assert not _restore_warnings(caplog)

    @pytest.mark.parametrize(
        "record",
        [["a"], ("a", None), [None, 3], [None, "a", "a"]],
        ids=["short", "tuple", "not-a-name", "long"],
    )
    def test_a_record_that_does_not_fit_the_state_restores_nothing(self, record):
        state = build_train_state(
            step=2, tokens_seen=0, scheduler=self._scheduler([(None, 0.1), ("a", 0.2)])
        )
        state["scheduler_group_modules"] = record
        torch.rand(1)
        rng = torch.get_rng_state()
        resumed = self._scheduler([("a", 0.2)])
        with pytest.raises(ValueError, match="one entry per param group"):
            restore_train_state(state, scheduler=resumed)
        assert torch.equal(torch.get_rng_state(), rng)
        assert resumed.last_epoch == 0

    def test_one_module_saved_at_two_rates_is_refused(self):
        state = build_train_state(
            step=0, tokens_seen=0, scheduler=self._scheduler([("a", 0.1), ("a", 0.2)])
        )
        with pytest.raises(ValueError, match=r"base rates 0.1 and 0.2"):
            restore_train_state(state, scheduler=self._scheduler([("a", 0.1)]))

    def test_a_scheduler_of_another_kind_without_modules_loads_as_saved(self):
        """Only a ``LambdaLR`` is laid out when no group names a module, so another keeps the
        positional load it always had, whatever the number of groups."""
        saved = torch.optim.lr_scheduler.StepLR(self._optimizer([(None, 0.1), (None, 0.1)]), 1)
        state = build_train_state(step=0, tokens_seen=0, scheduler=saved)
        resumed = torch.optim.lr_scheduler.StepLR(self._optimizer([(None, 0.1)]), 1)
        restore_train_state(state, scheduler=resumed)
        assert resumed.base_lrs == [0.1, 0.1]

    def test_only_a_lambda_lr_state_is_laid_out_for_other_groups(self):
        def step_lr(groups):
            return torch.optim.lr_scheduler.StepLR(self._optimizer(groups), step_size=1)

        saved = step_lr([(None, 0.1), ("a", 0.2)])
        state = build_train_state(step=0, tokens_seen=0, scheduler=saved)
        same = step_lr([(None, 0.1), ("a", 0.2)])
        restore_train_state(state, scheduler=same)
        assert same.state_dict() == saved.state_dict()
        with pytest.raises(ValueError, match="only the state of a LambdaLR"):
            restore_train_state(state, scheduler=step_lr([("a", 0.2)]))

    class _Stateful:
        """A scheduler with no optimizer, which saves ``state`` and keeps what it loads."""

        def __init__(self, state):
            self.state = state
            self.loaded = None

        def state_dict(self):
            return dict(self.state)

        def load_state_dict(self, state):
            self.loaded = state

    def test_a_state_without_base_rates_loads_as_it_is(self):
        state = build_train_state(step=0, tokens_seen=0, scheduler=self._Stateful({"count": 3}))
        assert set(state) == {"step", "tokens_seen", "rng", "scheduler"}
        state["scheduler_group_modules"] = ["a"]
        restored = self._Stateful({})
        restore_train_state(state, scheduler=restored)
        assert restored.loaded == {"count": 3}

    def test_a_scheduler_without_an_optimizer_takes_only_a_moduleless_state(self):
        saved = {"base_lrs": [0.1], "count": 3}
        state = build_train_state(step=0, tokens_seen=0, scheduler=self._Stateful(saved))
        restored = self._Stateful({})
        restore_train_state(state, scheduler=restored)
        assert restored.loaded == saved
        state["scheduler_group_modules"] = ["a"]
        with pytest.raises(ValueError, match="only the state of a LambdaLR"):
            restore_train_state(state, scheduler=self._Stateful({}))
