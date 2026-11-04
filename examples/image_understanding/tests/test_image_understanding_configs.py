"""Tests for what this example ships: its configs and its entry point."""

from __future__ import annotations

import importlib.util
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kempnerforge.config.loader import load_config
from kempnerforge.model.adapter import pooled_token_count

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = EXAMPLE_ROOT / "configs"
CONFIGS = sorted(CONFIG_DIR.glob("*.toml"))
CONFIG_IDS = [p.name for p in CONFIGS]
SIGLIP2_PATCH = 16


def _strings(node: Any, key_path: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _strings(value, f"{key_path}.{key}" if key_path else key)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _strings(value, f"{key_path}[{index}]")
    elif isinstance(node, str):
        yield key_path, node


def _assert_no_machine_specific_paths(path: Path) -> None:
    raw = tomllib.loads(path.read_text())
    for key, value in _strings(raw):
        assert not value.startswith("/"), f"{path.name}: {key} is an absolute path: {value!r}"
        assert "~" not in value, f"{path.name}: {key} references a home directory: {value!r}"


def test_config_dir_is_populated() -> None:
    assert CONFIGS, f"no configs found under {CONFIG_DIR}"


@pytest.mark.parametrize("path", CONFIGS, ids=CONFIG_IDS)
def test_config_loads_and_validates(path: Path) -> None:
    config = load_config(str(path), cli_args=[])
    assert config.is_vlm is True
    config.validate(world_size=4)


@pytest.mark.parametrize("path", CONFIGS, ids=CONFIG_IDS)
def test_every_preset_is_the_recipe_architecture(path: Path) -> None:
    """Joint-Decoder over a Qwen3-shaped backbone, with the pooled connector."""
    config = load_config(str(path), cli_args=[])
    assert config.vlm is not None and config.adapter is not None
    assert config.vision_encoder is not None
    assert config.vlm.arch == "joint_decoder"
    assert config.vision_encoder.type == "siglip2"
    assert (config.adapter.type, config.adapter.pool_window) == ("avgpool", 2)
    assert config.model.qk_norm is True
    assert str(config.model.norm_type) == "rmsnorm"
    assert config.model.is_moe is False
    assert config.data.tokenizer_path == "Qwen/Qwen3-0.6B"
    assert config.video is None


@pytest.mark.parametrize("path", CONFIGS, ids=CONFIG_IDS)
def test_config_has_no_machine_specific_paths(path: Path) -> None:
    _assert_no_machine_specific_paths(path)


def test_machine_specific_path_check_fires(tmp_path: Path) -> None:
    """Self-test for the scan above, so a passing sweep means something."""
    bad = tmp_path / "bad.toml"
    bad.write_text('[checkpoint]\ndir = "/absolute/cluster/share/run"\n')
    with pytest.raises(AssertionError, match="absolute path"):
        _assert_no_machine_specific_paths(bad)

    home = tmp_path / "home.toml"
    home.write_text('[checkpoint]\ndir = "~/scratch/run"\n')
    with pytest.raises(AssertionError, match="home directory"):
        _assert_no_machine_specific_paths(home)


@pytest.mark.parametrize("stage", [1, 2])
def test_siglip2_views_and_text_fit_the_context(stage: int) -> None:
    """A 512 px view pooled 2x2 is 256 tokens, and the budget covers it plus the text."""
    path = CONFIG_DIR / f"qwen3_0.6b_siglip2_stage{stage}.toml"
    config = load_config(str(path), cli_args=[])
    assert config.vision_encoder is not None and config.adapter is not None
    assert config.vlm is not None
    assert config.vision_encoder.path == "google/siglip2-so400m-patch16-512"
    assert (config.vision_encoder.num_tokens, config.vision_encoder.feature_dim) == (0, 0)
    assert config.data.hf_image_size == 512
    patches = (config.data.hf_image_size // SIGLIP2_PATCH) ** 2
    visual = pooled_token_count(patches, config.adapter.pool_window)
    assert visual == 256
    assert visual + config.vlm.max_text_len == config.model.max_seq_len == 512
    assert config.train.seq_len == config.model.max_seq_len


def test_stage1_trains_the_vision_side_only() -> None:
    config = load_config(str(CONFIG_DIR / "qwen3_0.6b_siglip2_stage1.toml"), cli_args=[])
    assert config.vlm is not None
    assert [(s.module, s.frozen) for s in config.vlm.freeze] == [("transformer", True)]
    assert config.checkpoint.load_path == "path-to-init-checkpoint"
    assert config.checkpoint.exclude_from_loading == ["optimizer"]


def test_stage2_trains_everything_from_a_stage1_checkpoint() -> None:
    config = load_config(str(CONFIG_DIR / "qwen3_0.6b_siglip2_stage2.toml"), cli_args=[])
    assert config.vlm is not None
    assert config.vlm.freeze == []
    assert config.checkpoint.load_path == "path-to-stage1-checkpoint"
    assert config.checkpoint.exclude_from_loading == ["optimizer"]


@pytest.mark.parametrize(("stage", "precision"), [(1, "fp32"), (2, "bf16")])
def test_stage_sets_the_encoder_precision_apart_from_the_backbone(
    stage: int, precision: str
) -> None:
    """The recipe trains SigLIP2 in fp32 in stage 1 and in bf16 in stage 2; the backbone in bf16."""
    path = CONFIG_DIR / f"qwen3_0.6b_siglip2_stage{stage}.toml"
    config = load_config(str(path), cli_args=[])
    assert config.vision_encoder is not None
    assert config.vision_encoder.mixed_precision == precision
    assert config.train.mixed_precision == "bf16"


@pytest.mark.parametrize(
    ("name", "world_size", "cli_args", "global_batch"),
    [
        ("qwen3_0.6b_siglip2_stage1.toml", 16, [], 256),
        ("qwen3_0.6b_siglip2_stage1.toml", 4, ["--train.grad_accum_steps=4"], 256),
        ("qwen3_0.6b_siglip2_stage2.toml", 16, [], 512),
        ("qwen3_0.6b_siglip2_stage2.toml", 4, ["--train.grad_accum_steps=8"], 512),
    ],
)
def test_stage_presets_validate_at_their_global_batch(
    name: str, world_size: int, cli_args: list[str], global_batch: int
) -> None:
    config = load_config(str(CONFIG_DIR / name), cli_args=cli_args)
    config.validate(world_size=world_size)
    assert config.train.batch_size * config.train.grad_accum_steps * world_size == global_batch


def _load_entry_point() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_image_understanding_example_train", EXAMPLE_ROOT / "train.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_entry_point_passes_the_loaded_config_through(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_entry_point()
    seen: dict[str, Any] = {}
    monkeypatch.setattr(module, "run_training", lambda config: seen.update(config=config))
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", str(CONFIG_DIR / "qwen3_0.6b_siglip2_stage1.toml"), "--train.max_steps=3"],
    )
    module.main()
    assert seen["config"].is_vlm is True
    assert seen["config"].train.max_steps == 3


def test_entry_point_requires_a_config(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_entry_point()
    monkeypatch.setattr(sys, "argv", ["train.py"])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 1
