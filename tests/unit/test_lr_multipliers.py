"""Per-module learning rates: ``optimizer.lr_multipliers``.

Each module listed there trains at ``optimizer.lr`` times its multiplier, in param groups of
its own, under the one learning-rate schedule. With nothing listed, nothing changes.
"""

from __future__ import annotations

import json
import math
import re

import pytest
import torch
from torch import nn

from kempnerforge.checkpoint.manager import CheckpointManager
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
        ids=["not-a-dict", "int-key", "empty-key", "bool", "str", "zero", "negative", "nan", "inf"],
    )
    def test_invalid_multipliers_are_refused(self, value, match):
        with pytest.raises(ValueError, match=match):
            OptimizerConfig(lr_multipliers=value)
        with pytest.raises(ValueError, match=match):
            checked_lr_multipliers(value)

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

    @pytest.mark.parametrize(
        ("arch", "inner"),
        [("cross_attention", "cross_attention"), ("mot", "mot"), ("moma", "moma")],
    )
    def test_nested_aliases_are_refused_when_the_config_is_built(
        self, tiny_vlm_configs, arch, inner
    ):
        """An arch's own alias lies inside ``transformer``, which the config can tell alone."""
        mc, vision, adapter, _ = tiny_vlm_configs
        with pytest.raises(ValueError, match=rf"lists '{inner}' and 'transformer', whose"):
            JobConfig(
                model=mc,
                vision_encoder=vision,
                adapter=adapter,
                vlm=VLMConfig.for_arch(arch, max_text_len=32),
                optimizer=OptimizerConfig(lr_multipliers={"transformer": 0.5, inner: 2.0}),
            )

    def test_two_aliases_with_one_pattern_are_refused(self, tiny_vlm_configs):
        mc, vision, adapter, vlm = tiny_vlm_configs
        vlm.module_patterns["connector"] = ["adapter.*"]
        with pytest.raises(ValueError, match="lists 'adapter' and 'connector', whose"):
            JobConfig(
                model=mc,
                vision_encoder=vision,
                adapter=adapter,
                vlm=vlm,
                optimizer=OptimizerConfig(lr_multipliers={"adapter": 2.0, "connector": 2.0}),
            )

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
        assert module_learning_rates(optimizer) == {"adapter": 0.02, "vision_encoder": 0.005}

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
        assert module_learning_rates(optimizer) == {"adapter": 0.02}
        held = {id(p) for g in optimizer.param_groups for p in g["params"]}
        assert not any(id(p) in held for p in model.vision_encoder.parameters())

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

    def test_a_parameter_in_two_listed_modules_is_refused(self):
        config = OptimizerConfig(lr_multipliers={"transformer": 0.5, "first": 2.0})
        patterns = {**VLM_PATTERNS, "first": ["transformer.0.*"]}
        with pytest.raises(
            ValueError, match=r"'transformer.0.weight' is in more than one module.*'first'"
        ):
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

    def test_overlapping_aliases_of_one_arch_are_refused(self):
        """``cross_attention`` lies inside ``transformer``, so listing both is ambiguous."""
        patterns = VLMConfig.for_arch("cross_attention").module_patterns
        model = nn.Module()
        model.transformer = nn.Module()
        model.transformer.cross_attention_layers = nn.Linear(4, 4)
        config = OptimizerConfig(lr_multipliers={"transformer": 0.5, "cross_attention": 2.0})
        with pytest.raises(ValueError, match="in more than one module"):
            build_optimizer(model, config, module_patterns=patterns)
        only = OptimizerConfig(lr_multipliers={"cross_attention": 2.0})
        optimizer = build_optimizer(model, only, module_patterns=patterns)
        assert module_learning_rates(optimizer) == {"cross_attention": pytest.approx(6e-4)}


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
            "adapter": pytest.approx(0.01),
            "adapter/adamw": pytest.approx(1.5e-4),
            "vision_encoder": pytest.approx(0.0025),
            "vision_encoder/adamw": pytest.approx(3.75e-5),
        }
        _forward(model).backward()
        optimizer.step()
        fallback = {g["module"]: g["lr"] for g in optimizer._adam.param_groups if "module" in g}
        assert fallback == {
            "adapter": pytest.approx(1.5e-4),
            "vision_encoder": pytest.approx(3.75e-5),
        }

    def test_a_module_on_the_fallback_alone_is_logged(self):
        """A module with no matrix for Muon still reports its rate, through its fallback."""
        model = _vlm_like()
        patterns = {**VLM_PATTERNS, "norms": ["transformer.1.*"]}
        config = OptimizerConfig(
            name="muon", lr=0.02, muon_adam_lr=3e-4, lr_multipliers={"norms": 0.5}
        )
        optimizer = build_optimizer(model, config, module_patterns=patterns)
        assert not any(g.get("module") == "norms" for g in optimizer.param_groups)
        assert module_learning_rates(optimizer) == {"norms/adamw": pytest.approx(1.5e-4)}


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

    @pytest.mark.parametrize(
        ("content", "match"),
        [
            (b"{not json", "cannot read"),
            (b"[]", "does not hold a JSON object"),
            (b'{"step": 3, "lr_multipliers": {"head": -1}}', "records an invalid LR multipliers"),
            (b'{"step": 3, "lr_module_patterns": {"head": "x"}}', "lists of patterns"),
            (b"[" * 100_000 + b"]" * 100_000, "RecursionError"),
        ],
        ids=["bad-json", "list", "negative", "patterns", "too-deep"],
    )
    def test_an_invalid_record_is_refused(self, tmp_path, content, match):
        (self._save(tmp_path, {"head": 2.0}) / "metadata.json").write_bytes(content)
        _, scheduler, manager = self._resume(tmp_path, {"head": 2.0})
        with pytest.raises(ValueError, match=f"Cannot check the lr_multipliers.*{match}"):
            manager.load(scheduler=scheduler)

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

    def test_rank_zero_reads_the_record_and_broadcasts_it(self, tmp_path, monkeypatch):
        self._save(tmp_path, {"head": 2.0})
        _, scheduler, manager = self._resume(tmp_path, {"head": 2.0})
        calls = self._as_rank(monkeypatch, manager, 0)
        self._stop_at_the_model_load(monkeypatch)
        with pytest.raises(_ReachedTheModel):
            manager.load(scheduler=scheduler)
        record = ({"head": 2.0}, {"head": ["output_head.*"]})
        assert calls == [
            ("all_reduce", "GLOO_GROUP", [[0, 0], [1, 0]]),
            ("broadcast", 0, [(record, None)]),
        ]

    def test_other_ranks_refuse_what_rank_zero_sent(self, tmp_path, monkeypatch):
        """A rank other than 0 never reads the record and raises on the one it receives."""
        self._save(tmp_path, {"head": 2.0})
        (tmp_path / "step_3" / "metadata.json").unlink()
        _, scheduler, manager = self._resume(tmp_path, {})
        record = ({"head": 2.0}, {"head": ["output_head.*"]})
        self._as_rank(monkeypatch, manager, 1, sent=[(record, None)])
        with pytest.raises(ValueError, match=r"lr_multipliers=\{'head': 2.0\}, but this run uses"):
            manager.load(scheduler=scheduler)
        self._as_rank(monkeypatch, manager, 1, sent=[(None, "rank 0 could not read it")])
        with pytest.raises(ValueError, match="rank 0 could not read it"):
            manager.load(scheduler=scheduler)

    def test_a_failure_on_rank_zero_is_broadcast(self, tmp_path, monkeypatch):
        self._save(tmp_path, {"head": 2.0})
        _, scheduler, manager = self._resume(tmp_path, {"head": 2.0})

        def fail(_ckpt_dir):
            raise OSError("disk gone")

        monkeypatch.setattr("kempnerforge.checkpoint.manager._read_lr_multipliers", fail)
        calls = self._as_rank(monkeypatch, manager, 0)
        with pytest.raises(ValueError, match="Cannot check the lr_multipliers.*disk gone"):
            manager.load(scheduler=scheduler)
        assert calls[-1] == ("broadcast", 0, [(None, "OSError: disk gone")])

    def test_every_rank_checks_the_scheduler_state_rank_zero_sent(self, tmp_path, monkeypatch):
        """Each rank decides from the train state rank 0 broadcast, so all check or none do."""
        step_dir = self._save(tmp_path, {"head": 2.0})
        train_state = torch.load(step_dir / "train_state.pt", weights_only=False)
        record = ({"head": 2.0}, {"head": ["output_head.*"]})
        _, scheduler, manager = self._resume(tmp_path, {})
        calls = self._as_rank(monkeypatch, manager, 1, sent=[True, train_state, (record, None)])
        with pytest.raises(ValueError, match="but this run uses"):
            manager.load(scheduler=scheduler, exclude_keys=["model", "optimizer"])
        assert len(calls) == 3
        assert scheduler.last_epoch == 0
        without_scheduler = {k: v for k, v in train_state.items() if k != "scheduler"}
        calls = self._as_rank(monkeypatch, manager, 1, sent=[True, without_scheduler])
        assert manager.load(scheduler=scheduler, exclude_keys=["model", "optimizer"])[0] == 3
        assert len(calls) == 2
