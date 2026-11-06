"""Tests for resolving ``vlm.arch`` to its config class, from a TOML, the CLI and code."""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Callable
from pathlib import Path

import pytest

from kempnerforge.config.loader import load_config
from kempnerforge.config.registry import Registry, registry
from kempnerforge.config.schema import (
    AdapterConfig,
    DistributedConfig,
    JobConfig,
    ModelConfig,
    TrainConfig,
    VisionEncoderConfig,
)
from kempnerforge.config.vlm import VLMConfig

ARCHS = ("joint_decoder", "cross_attention", "mot", "moma")


def _vlm_toml(tmp_path: Path, vlm_lines: str) -> str:
    """Write a minimal VLM job whose ``[vlm]`` section holds ``vlm_lines``."""
    path = tmp_path / "vlm.toml"
    path.write_text(
        f"""\
[model]
dim = 64
n_layers = 2
n_heads = 4
n_kv_heads = 4
vocab_size = 256
max_seq_len = 128

[vision_encoder]
type = "random"
num_tokens = 16

[vlm]
max_text_len = 32
{vlm_lines}
"""
    )
    return str(path)


UNKNOWN_ARCH_SITES: dict[str, Callable[[Path], object]] = {
    "toml": lambda tmp_path: load_config(_vlm_toml(tmp_path, 'arch = "nope"'), cli_args=[]),
    "cli": lambda tmp_path: load_config(cli_args=["--vlm.arch=nope"]),
    "cli-over-toml": lambda tmp_path: load_config(
        _vlm_toml(tmp_path, 'arch = "moma"'), cli_args=["--vlm.arch=nope"]
    ),
    "constructor": lambda tmp_path: VLMConfig(arch="nope"),
    "for_arch": lambda tmp_path: VLMConfig.for_arch("nope"),
}


@pytest.mark.parametrize("build", UNKNOWN_ARCH_SITES.values(), ids=UNKNOWN_ARCH_SITES.keys())
def test_an_unknown_arch_lists_only_the_registered_arches(tmp_path, build):
    with pytest.raises(ValueError) as excinfo:
        build(tmp_path)
    message = str(excinfo.value)
    registered = sorted(registry.list_vlm_configs())
    assert message == f"Unknown vlm.arch: 'nope'. Registered: {registered}."
    assert "Reserved" not in message


class TestArchOverride:
    """``--vlm.arch`` cannot change the config class the TOML's ``[vlm]`` section built."""

    @pytest.mark.parametrize(("preset", "override"), list(itertools.permutations(ARCHS, 2)))
    def test_an_override_to_another_arch_is_rejected(self, tmp_path, preset, override):
        toml = _vlm_toml(tmp_path, f'arch = "{preset}"')
        with pytest.raises(ValueError) as excinfo:
            load_config(toml, cli_args=[f"--vlm.arch={override}"])
        message = str(excinfo.value)
        built = type(VLMConfig.for_arch(preset)).__name__
        assert message.startswith(f"vlm.arch={override!r} cannot be applied as an override")
        assert f"built as {built} (arch {preset!r})" in message
        assert f'Set arch = "{override}" in the TOML\'s [vlm] section instead.' in message

    def test_an_override_of_the_default_arch_is_rejected(self, tmp_path):
        """A ``[vlm]`` section without ``arch`` is built as the joint decoder."""
        with pytest.raises(ValueError, match="built as JointDecoderConfig"):
            load_config(_vlm_toml(tmp_path, ""), cli_args=["--vlm.arch=moma"])

    def test_the_arch_is_checked_before_the_other_keys(self, tmp_path):
        """A field of the overriding arch's class is not reported as an unknown key."""
        toml = _vlm_toml(tmp_path, 'arch = "joint_decoder"')
        args = ["--vlm.arch=cross_attention", "--vlm.cross_attention_every_n_layers=2"]
        with pytest.raises(ValueError, match="cannot be applied as an override"):
            load_config(toml, cli_args=args)

    @pytest.mark.parametrize("arch", ARCHS)
    def test_an_override_to_the_same_arch_keeps_the_class(self, tmp_path, arch):
        toml = _vlm_toml(tmp_path, f'arch = "{arch}"')
        config = load_config(toml, cli_args=[f"--vlm.arch={arch}", "--vlm.max_text_len=16"])
        assert config.vlm is not None
        assert type(config.vlm) is type(VLMConfig.for_arch(arch))
        assert (config.vlm.arch, config.vlm.max_text_len) == (arch, 16)

    @pytest.mark.parametrize("arch", ARCHS)
    def test_an_override_without_a_vlm_section_builds_the_arch_class(self, arch):
        config = load_config(
            cli_args=[f"--vlm.arch={arch}", "--vision_encoder.type=random", "--vlm.max_text_len=16"]
        )
        assert config.vlm is not None
        assert type(config.vlm) is type(VLMConfig.for_arch(arch))
        assert config.vlm.arch == arch


class TestArchMatchesClass:
    """A ``VLMConfig`` carries its class's arch, however it is constructed."""

    @pytest.mark.parametrize("arch", ARCHS)
    def test_for_arch_builds_the_class_of_the_arch(self, arch):
        config = VLMConfig.for_arch(arch)
        assert config.arch == type(config).arch == arch

    def test_the_base_class_is_the_joint_decoder(self):
        assert VLMConfig().arch == "joint_decoder"
        assert VLMConfig(arch="joint_decoder").arch == "joint_decoder"

    @pytest.mark.parametrize(("built", "replaced"), list(itertools.permutations(ARCHS, 2)))
    def test_replacing_the_arch_is_rejected(self, built, replaced):
        config = VLMConfig.for_arch(built)
        with pytest.raises(ValueError) as excinfo:
            dataclasses.replace(config, arch=replaced)
        assert str(excinfo.value).startswith(
            f"vlm.arch={replaced!r} does not match {type(config).__name__}.arch ({built!r})."
        )

    @pytest.mark.parametrize("arch", [a for a in ARCHS if a != "joint_decoder"])
    def test_the_base_class_rejects_another_arch(self, arch):
        with pytest.raises(ValueError, match=rf"vlm.arch='{arch}' does not match VLMConfig.arch"):
            VLMConfig(arch=arch)

    def test_a_subclass_must_default_arch_to_its_registered_name(self, tmp_path, isolated_registry):
        @isolated_registry.register_vlm_config("kf416_unset_arch")
        @dataclasses.dataclass
        class UnsetArchConfig(VLMConfig):
            """Registered under a name, but keeps the base class's arch default."""

        with pytest.raises(
            ValueError,
            match=r"vlm.arch='kf416_unset_arch' does not match UnsetArchConfig.arch "
            r"\('joint_decoder'\).*must default arch to the name it is registered under",
        ):
            load_config(_vlm_toml(tmp_path, 'arch = "kf416_unset_arch"'), cli_args=[])


def _declare(reg: Registry, name: str, style: str) -> type[VLMConfig]:
    """Register a ``VLMConfig`` subclass that defaults ``arch`` to ``name``, declared with
    ``slots=True`` or with a ``default_factory``."""
    if style == "slots":

        @reg.register_vlm_config(name)
        @dataclasses.dataclass(slots=True)
        class SlotsConfig(VLMConfig):
            arch: str = name

        return SlotsConfig

    @reg.register_vlm_config(name)
    @dataclasses.dataclass
    class FactoryConfig(VLMConfig):
        arch: str = dataclasses.field(default_factory=lambda: name)

    return FactoryConfig


class TestDeclaredArch:
    """A class's arch is its ``arch`` field's default, however the class declares it."""

    def test_the_in_tree_classes_declare_their_arch(self):
        assert VLMConfig.declared_arch() == "joint_decoder"
        for arch in ARCHS:
            assert type(VLMConfig.for_arch(arch)).declared_arch() == arch

    @pytest.mark.parametrize("style", ["slots", "default_factory"])
    def test_a_subclass_that_defaults_arch_to_its_name_builds(
        self, tmp_path, isolated_registry, style
    ):
        name = f"kf416_{style}"
        cls = _declare(isolated_registry, name, style)
        assert cls.declared_arch() == name
        config = VLMConfig.for_arch(name)
        assert (type(config), config.arch) == (cls, name)
        loaded = load_config(_vlm_toml(tmp_path, f'arch = "{name}"'), cli_args=[])
        assert loaded.vlm is not None
        assert (type(loaded.vlm), loaded.vlm.arch) == (cls, name)
        with pytest.raises(
            ValueError,
            match=rf"vlm.arch='joint_decoder' does not match {cls.__name__}.arch \('{name}'\)",
        ):
            dataclasses.replace(config, arch="joint_decoder")

    def test_a_subclass_without_an_arch_default_raises_type_error(self, isolated_registry):
        """A bare ``arch: str`` inherits the base default; ``field()`` with none removes it."""

        @isolated_registry.register_vlm_config("kf416_no_default")
        @dataclasses.dataclass
        class NoDefaultConfig(VLMConfig):
            arch: str = dataclasses.field()

        with pytest.raises(TypeError, match="NoDefaultConfig declares arch without a default"):
            NoDefaultConfig(arch="kf416_no_default")

    def test_parallelism_errors_name_a_slots_class_by_its_arch(self, isolated_registry):
        cls = _declare(isolated_registry, "kf416_slots_job", "slots")
        job = JobConfig(
            model=ModelConfig(dim=64, n_layers=2, n_heads=4, vocab_size=256, max_seq_len=128),
            train=TrainConfig(seq_len=64),
            distributed=DistributedConfig(dp_shard=1, tp=2),
            vision_encoder=VisionEncoderConfig(type="random", num_tokens=8),
            adapter=AdapterConfig(),
            vlm=cls(max_text_len=32),
        )
        with pytest.raises(
            ValueError, match=r"vlm.arch='kf416_slots_job' does not support tensor parallelism"
        ):
            job.validate(world_size=2)
