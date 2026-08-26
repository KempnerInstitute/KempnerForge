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

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = EXAMPLE_ROOT / "configs"
CONFIGS = sorted(CONFIG_DIR.glob("*.toml"))
CONFIG_IDS = [p.name for p in CONFIGS]


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
    """Refuse rather than pass vacuously: an empty glob would make every
    parametrized case below disappear silently if the directory moved."""
    assert CONFIGS, f"no configs found under {CONFIG_DIR}"


@pytest.mark.parametrize("path", CONFIGS, ids=CONFIG_IDS)
def test_config_loads_and_validates(path: Path) -> None:
    config = load_config(str(path), cli_args=[])
    assert config.is_vlm is True
    config.validate(world_size=4)


@pytest.mark.parametrize("path", CONFIGS, ids=CONFIG_IDS)
def test_config_has_no_machine_specific_paths(path: Path) -> None:
    _assert_no_machine_specific_paths(path)


def test_machine_specific_path_check_fires(tmp_path: Path) -> None:
    """Self-test for the scan above, so a passing sweep means something."""
    bad = tmp_path / "bad.toml"
    bad.write_text('[video]\ndata_root = "/absolute/cluster/share/webvid-10m"\n')
    with pytest.raises(AssertionError, match="absolute path"):
        _assert_no_machine_specific_paths(bad)

    home = tmp_path / "home.toml"
    home.write_text('[checkpoint]\ndir = "~/scratch/run"\n')
    with pytest.raises(AssertionError, match="home directory"):
        _assert_no_machine_specific_paths(home)


def test_vlm_debug_toml() -> None:
    """Regression: parallel [vision_encoder] / [adapter] / [vlm] tables load
    correctly, and list[FreezeSpec] inside VLMConfig instantiates each freeze
    entry via __post_init__."""
    config = load_config(str(CONFIG_DIR / "vlm_debug.toml"), cli_args=[])
    assert config.is_vlm is True
    assert config.vision_encoder is not None
    assert config.vlm is not None
    assert config.vision_encoder.type == "random"
    assert config.vision_encoder.num_tokens == 64
    assert len(config.vlm.freeze) == 1
    assert config.vlm.freeze[0].module == "vision_encoder"
    assert config.vlm.freeze[0].frozen is True
    config.validate(world_size=1)


def test_vlm_7b_siglip2_toml() -> None:
    config = load_config(str(CONFIG_DIR / "vlm_7b_siglip2.toml"), cli_args=[])
    assert config.is_vlm is True
    assert config.vision_encoder is not None
    assert config.vlm is not None
    assert config.vision_encoder.type == "siglip2"
    # num_tokens defaults to 0 = "infer from encoder at build time". The
    # encoder probes 196 (14x14 patches, no CLS) for this path; the build-time
    # max_seq_len cross-check in build_vlm_wrapper enforces
    # 196 + 2048 = 2244 <= max_seq_len=2304.
    assert config.vision_encoder.num_tokens == 0
    config.validate(world_size=4)


def _load_entry_point() -> Any:
    spec = importlib.util.spec_from_file_location("_vlm_example_train", EXAMPLE_ROOT / "train.py")
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
        ["train.py", str(CONFIG_DIR / "vlm_debug.toml"), "--train.max_steps=3"],
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
