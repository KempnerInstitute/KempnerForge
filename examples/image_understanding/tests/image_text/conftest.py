"""Puts the example root on ``sys.path`` and provides small image fixtures."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

QWEN = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="session")
def tokenizer() -> Any:
    """The Qwen3 tokenizer, loaded once for every renderer test."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(QWEN)


@pytest.fixture
def write_image(tmp_path: Path) -> Callable[..., Path]:
    """Writes a small image under ``tmp_path`` and returns its path."""

    def write(relative: str, size: tuple[int, int] = (8, 6), mode: str = "RGB") -> Path:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        color = {"RGB": (200, 30, 10), "L": 90, "RGBA": (10, 200, 30, 128)}[mode]
        Image.new(mode, size, color).save(path, format="PNG")
        return path

    return write
