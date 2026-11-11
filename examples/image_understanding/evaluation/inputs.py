"""What the model is given to answer a document.

``document_inputs`` is the one place a document becomes model inputs: the prompt
the training renderer puts before the assistant's reply, tokenized, and the
image as that renderer preprocesses it. It takes documents with one image and
one turn; anything else is refused with ``UnsupportedDocumentError``.
``build_prompt`` alone decides the prompt's text.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import torch
from image_text.document import Document
from image_text.render import ChatRenderer

from kempnerforge.config.job import JobConfig
from kempnerforge.data.vlm_dataset import build_tokenizer, pil_to_tensor

_REPLY = "x"
_ONE_TOKENIZER_CALL_AT_A_TIME = threading.Lock()


class UnsupportedDocumentError(ValueError):
    """A document outside what the evaluator turns into model inputs."""


@dataclass(frozen=True)
class ModelInputs:
    """One document as the model sees it.

    ``pixel_values`` is the image as the renderer preprocesses it, ``(3, H, W)``,
    and ``prompt_ids`` holds the tokens the reply is decoded after.
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


def build_renderer(config: JobConfig) -> ChatRenderer:
    """The training renderer for ``config``: its tokenizer, text length and image size."""
    check_renderer_config(config)
    assert config.vlm is not None
    return ChatRenderer(
        build_tokenizer(config.data.tokenizer_path),
        max_text_len=config.vlm.max_text_len,
        image_size=config.data.hf_image_size,
        answer_only=True,
    )


def build_prompt(renderer: ChatRenderer, document: Document) -> str:
    """The text ``renderer`` puts before the assistant's reply to a one-turn ``document``."""
    text, spans = renderer.text_and_reply_spans(document, [_REPLY])
    return text[: spans[0][0]]


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
    in a sample of this image. A document without exactly one image and one turn
    raises ``UnsupportedDocumentError``. Several threads may call this.
    """
    if len(document.images) != 1:
        raise UnsupportedDocumentError(
            f"the evaluator takes documents with one image, got {len(document.images)}"
        )
    if len(document.turns) != 1:
        raise UnsupportedDocumentError(
            f"the evaluator takes documents with one turn, got {len(document.turns)}"
        )
    pixel_values = pil_to_tensor(
        document.images[0], renderer.image_size, renderer.image_mean, renderer.image_std
    )
    return ModelInputs(pixel_values=pixel_values, prompt_ids=prompt_token_ids(renderer, document))
