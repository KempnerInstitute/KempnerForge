"""What the model is given to answer a document.

``document_inputs`` is the one place a document becomes model inputs: the prompt
the training renderer puts before the assistant's reply, tokenized, and the
document's images as that renderer preprocesses them, every view of each. It
takes documents with one turn and any number of images, none included; anything
else is refused with ``UnsupportedDocumentError``. ``build_prompt`` alone decides
the prompt's text. ``RenderSettings`` holds what a training mixture changes about
rendering and attention, read from its ``[render]`` and ``[packing]`` tables.
"""

from __future__ import annotations

import threading
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from image_text.document import Document
from image_text.render import ChatRenderer

from kempnerforge.config.job import JobConfig
from kempnerforge.data.vlm_dataset import build_tokenizer

_ONE_TOKENIZER_CALL_AT_A_TIME = threading.Lock()


class UnsupportedDocumentError(ValueError):
    """A document outside what the evaluator turns into model inputs."""


@dataclass(frozen=True)
class RenderSettings:
    """The ``ChatRenderer`` options a run trained with, and whether a view's tokens saw each other.

    ``render`` holds ``ChatRenderer`` keyword options (``train_reply_prefix``,
    ``lead_space``, ...) and ``view_groups`` says each image view's tokens attended
    to each other in both directions, as a mixture's ``[packing]`` sets it.
    """

    render: Mapping[str, Any] = field(default_factory=dict)
    view_groups: bool = False

    @classmethod
    def from_mixture(cls, path: str | Path | None) -> RenderSettings:
        """The settings of the mixture file at ``path``, or the defaults without one."""
        if path is None:
            return cls()
        with open(path, "rb") as file:
            mixture = tomllib.load(file)
        packing = mixture.get("packing", {})
        unknown = sorted(set(packing) - {"view_groups"})
        if unknown:
            raise ValueError(
                f"the mixture's [packing] holds options the evaluator lacks: {unknown}"
            )
        return cls(dict(mixture.get("render", {})), bool(packing.get("view_groups", False)))


@dataclass(frozen=True)
class ModelInputs:
    """One document as the model sees it.

    ``pixel_values`` is ``(N, 3, H, W)``, every view of every image of the
    document as the renderer preprocesses it (``N = 0`` without an image), and
    ``prompt_ids`` holds the tokens the reply is decoded after.
    """

    pixel_values: torch.Tensor
    prompt_ids: tuple[int, ...]


def check_renderer_config(config: JobConfig) -> None:
    """Refuses a ``config`` the training renderer cannot be built from.

    Raises:
        ValueError: the config has no ``[vlm]`` section or ``data.tokenizer_path`` is not set.
    """
    if config.vlm is None:
        raise ValueError("the config has no [vlm] section, so there is no renderer to build")
    if not config.data.tokenizer_path:
        raise ValueError(
            "data.tokenizer_path is not set; the evaluator renders prompts with the "
            "tokenizer the checkpoint was trained with"
        )


def build_renderer(config: JobConfig, settings: RenderSettings | None = None) -> ChatRenderer:
    """The training renderer for ``config`` with ``settings``' options.

    It takes the run's tokenizer, image size and tiling, and the text length an
    unpacked run lays out; a packed run's text is uncut.
    """
    check_renderer_config(config)
    assert config.vlm is not None
    options: dict[str, Any] = {"answer_only": True}
    options.update((settings or RenderSettings()).render)
    return ChatRenderer(
        build_tokenizer(config.data.tokenizer_path),
        max_text_len=None if config.data.pack_sequences else config.vlm.max_text_len,
        image_size=config.data.hf_image_size,
        max_images=None,
        tile_grid=config.vlm.tile_grid,
        tile_global=config.vlm.tile_global,
        **options,
    )


def build_prompt(renderer: ChatRenderer, document: Document) -> str:
    """The text ``renderer`` puts before the assistant's reply to a one-turn ``document``."""
    return renderer.prompt(document)


def prompt_token_ids(renderer: ChatRenderer, document: Document) -> tuple[int, ...]:
    """The tokens of ``build_prompt``, tokenized the way ``renderer`` tokenizes a sample.

    The tokenizer is used by one thread at a time, so several may call this.
    """
    with _ONE_TOKENIZER_CALL_AT_A_TIME:
        prompt = build_prompt(renderer, document)
        return tuple(renderer.tokenizer(prompt, add_special_tokens=False)["input_ids"])


def document_inputs(renderer: ChatRenderer, document: Document) -> ModelInputs:
    """The inputs the model answers ``document`` from, as ``renderer`` lays training out.

    The prompt is ``prompt_token_ids`` and the pixels are what the renderer puts
    in a sample of this document. A document with several turns raises
    ``UnsupportedDocumentError``. Several threads may call this.
    """
    if len(document.turns) != 1:
        raise UnsupportedDocumentError(
            f"the evaluator takes documents with one turn, got {len(document.turns)}"
        )
    pixels = renderer.pixels(document)
    return ModelInputs(
        pixel_values=pixels.reshape(-1, *pixels.shape[-3:]),
        prompt_ids=prompt_token_ids(renderer, document),
    )
