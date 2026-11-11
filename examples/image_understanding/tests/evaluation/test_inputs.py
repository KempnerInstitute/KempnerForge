"""Tests for turning a document into the inputs the model answers it from."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from evaluation.inputs import (
    UnsupportedDocumentError,
    build_prompt,
    build_renderer,
    document_inputs,
    prompt_token_ids,
)
from image_text.document import Document, Turn
from image_text.render import ChatRenderer
from PIL import Image

from kempnerforge.config.job import JobConfig
from kempnerforge.config.loader import load_config

THINKING_PROMPT = (
    "<|im_start|>user\nWhat is shown?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


def _image(color: tuple[int, int, int] = (255, 0, 0)) -> Image.Image:
    return Image.new("RGB", (12, 9), color)


def _document(
    user: str = "what is shown ?",
    answers: tuple[str, ...] = ("cat", "a dog"),
    kind: Any = "qa",
    images: int = 1,
) -> Document:
    return Document(images=(_image(),) * images, turns=(Turn(user, answers),), kind=kind)


def _renderer(tokenizer: Any, **overrides: Any) -> ChatRenderer:
    kwargs: dict[str, Any] = {"max_text_len": 64, "image_size": 16, "answer_only": True}
    kwargs.update(overrides)
    return ChatRenderer(tokenizer, **kwargs)


@pytest.fixture(scope="module")
def tokenizer(tokenizer_path: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path)


class TestBuildRenderer:
    def test_the_renderer_takes_the_training_text_length_and_image_size(
        self, write_config: Callable[..., Path]
    ) -> None:
        renderer = build_renderer(load_config(write_config(), cli_args=[]))
        assert (renderer.max_text_len, renderer.image_size) == (32, 16)
        assert renderer.tokenizer.eos_token == "<|im_end|>"
        assert renderer.turn_end == "<|im_end|>"

    def test_a_config_without_a_vlm_or_a_tokenizer_is_refused(
        self, write_config: Callable[..., Path]
    ) -> None:
        with pytest.raises(ValueError, match="no \\[vlm\\] section"):
            build_renderer(JobConfig())
        config = load_config(write_config(), cli_args=["--data.tokenizer_path="])
        with pytest.raises(ValueError, match="data.tokenizer_path is not set"):
            build_renderer(config)


class TestBuildPrompt:
    def test_the_prompt_is_the_training_text_before_the_reply(self, tokenizer: Any) -> None:
        renderer = _renderer(tokenizer)
        prompt = build_prompt(renderer, _document())
        assert prompt == "<|im_start|>user\nwhat is shown ?<|im_end|>\n<|im_start|>assistant\n"
        for answer in ("cat", "a dog"):
            assert renderer.text(_document(), [answer]) == f"{prompt}{answer}<|im_end|>"

    def test_the_prompt_does_not_depend_on_the_reference_answers(self, tokenizer: Any) -> None:
        renderer = _renderer(tokenizer)
        marked = _document(answers=("<think>", "a<|im_end|>b"))
        assert build_prompt(renderer, marked) == build_prompt(renderer, _document())

    def test_a_one_turn_conversation_is_prompted_up_to_its_reply(self, tokenizer: Any) -> None:
        renderer = _renderer(tokenizer)
        document = _document(kind="conversation")
        prompt = build_prompt(renderer, document)
        assert prompt.endswith("<|im_start|>assistant\n")
        assert renderer.text(document, ["cat"]).startswith(f"{prompt}cat<|im_end|>")

    def test_the_image_placeholder_stays_in_the_prompt(self, tokenizer: Any) -> None:
        renderer = _renderer(tokenizer, image_placeholder="image")
        assert build_prompt(renderer, _document()).startswith(
            "<|im_start|>user\nimage what is shown ?"
        )

    @pytest.mark.parametrize("kind", ["qa", "conversation"])
    def test_the_recipe_prompt_ends_with_the_empty_think_block(
        self, bpe_tokenizer: Any, kind: str
    ) -> None:
        document = _document(user="What is shown?", kind=kind)
        assert build_prompt(_renderer(bpe_tokenizer), document) == THINKING_PROMPT


class TestPromptTokenIds:
    @pytest.mark.parametrize("kind", ["qa", "conversation"])
    @pytest.mark.parametrize("answer", ["a cat", "42", "Yes.", "(a)", "-3.5", "A\nB", "!"])
    @pytest.mark.parametrize("user", ["What is shown?", "How many?\n", "what is 2+2 ="])
    def test_the_prompt_tokens_start_the_training_sample(
        self, bpe_tokenizer: Any, user: str, answer: str, kind: str
    ) -> None:
        """Tokenized alone, the prompt is the sample's tokens up to the reply, whatever follows."""
        renderer = _renderer(bpe_tokenizer)
        document = _document(user=user, answers=(answer,), kind=kind)
        prompt_ids = list(prompt_token_ids(renderer, document))
        assert renderer.token_ids(document, [answer])[: len(prompt_ids)] == prompt_ids
        sample = renderer.render(document, [answer])
        assert sample["input_ids"][: len(prompt_ids)].tolist() == prompt_ids
        assert (sample["labels"][: len(prompt_ids) - 1] == -100).all()
        assert sample["labels"][len(prompt_ids) - 1] != -100


class TestDocumentInputs:
    def test_inputs_are_the_prompt_tokens_and_the_renderers_pixels(self, tokenizer: Any) -> None:
        renderer = _renderer(tokenizer)
        inputs = document_inputs(renderer, _document())
        assert inputs.prompt_ids == prompt_token_ids(renderer, _document())
        assert tokenizer.decode(list(inputs.prompt_ids)) == (
            "<|im_start|> user what is shown ? <|im_end|> <|im_start|> assistant"
        )
        sample = renderer.render(_document(), ["cat"])
        assert torch.equal(inputs.pixel_values, sample["pixel_values"])
        assert inputs.pixel_values.shape == (3, 16, 16)

    def test_pixels_follow_the_renderers_size_and_normalization(self, tokenizer: Any) -> None:
        renderer = _renderer(
            tokenizer, image_size=8, image_mean=(0.1, 0.2, 0.3), image_std=(0.5, 0.25, 0.2)
        )
        inputs = document_inputs(renderer, _document())
        assert torch.equal(
            inputs.pixel_values, renderer.render(_document(), ["cat"])["pixel_values"]
        )
        assert inputs.pixel_values.shape == (3, 8, 8)

    @pytest.mark.parametrize("images", [0, 2])
    def test_a_document_without_exactly_one_image_is_refused(
        self, tokenizer: Any, images: int
    ) -> None:
        with pytest.raises(UnsupportedDocumentError, match=f"one image, got {images}"):
            document_inputs(_renderer(tokenizer), _document(images=images))

    def test_a_document_with_several_turns_is_refused(self, tokenizer: Any) -> None:
        document = Document(
            images=(_image(),),
            turns=(Turn("how many ?", ("one",)), Turn("which color ?", ("red",))),
            kind="conversation",
        )
        with pytest.raises(UnsupportedDocumentError, match="one turn, got 2"):
            document_inputs(_renderer(tokenizer), document)
