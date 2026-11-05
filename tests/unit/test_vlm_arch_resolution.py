"""Tests for resolving ``vlm.arch`` to its config class, from a TOML, the CLI and code."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from kempnerforge.config.loader import load_config
from kempnerforge.config.registry import registry
from kempnerforge.config.vlm import VLMConfig


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
