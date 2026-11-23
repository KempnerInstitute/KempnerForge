"""Tests for turning a document into the inputs the model answers it from."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from evaluation.inputs import (
    RenderSettings,
    UnsupportedDocumentError,
    build_prompt,
    build_renderer,
    check_renderer_config,
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

    def test_a_packed_run_renders_uncut_with_its_tiling_and_mixture_options(
        self, write_config: Callable[..., Path]
    ) -> None:
        config = load_config(
            write_config(),
            cli_args=["--data.pack_sequences=true", "--vlm.tile_grid=2", "--model.max_seq_len=64"],
        )
        renderer = build_renderer(
            config, RenderSettings({"lead_space": True, "train_reply_prefix": True})
        )
        assert renderer.max_text_len is None
        assert (renderer.tile_grid, renderer.tile_global, renderer.max_images) == (2, True, None)
        assert renderer.lead_space and renderer.train_reply_prefix and renderer.answer_only

    def test_a_config_without_a_vlm_or_a_tokenizer_is_refused(
        self, write_config: Callable[..., Path]
    ) -> None:
        with pytest.raises(ValueError, match="no \\[vlm\\] section"):
            build_renderer(JobConfig())
        config = load_config(write_config(), cli_args=["--data.tokenizer_path="])
        with pytest.raises(ValueError, match="data.tokenizer_path is not set"):
            build_renderer(config)


class TestRenderSettings:
    def test_a_mixture_gives_its_render_options_and_view_groups(self, tmp_path: Path) -> None:
        path = tmp_path / "mixture.toml"
        path.write_text(
            "[render]\nanswer_only = true\nlead_space = true\n[packing]\nview_groups = true\n"
            '[[source]]\nreader = "vqav2"\n'
        )
        settings = RenderSettings.from_mixture(path)
        assert settings.render == {"answer_only": True, "lead_space": True}
        assert settings.view_groups

    def test_without_a_mixture_nothing_changes(self) -> None:
        assert RenderSettings.from_mixture(None) == RenderSettings()

    def test_a_packing_option_the_evaluator_lacks_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "mixture.toml"
        path.write_text("[packing]\nview_groups = true\nshuffle_rows = 3\n")
        with pytest.raises(ValueError, match="shuffle_rows"):
            RenderSettings.from_mixture(path)


class TestCheckRendererConfig:
    def test_a_config_with_a_vlm_and_a_tokenizer_passes(
        self, write_config: Callable[..., Path]
    ) -> None:
        check_renderer_config(load_config(write_config(), cli_args=[]))

    def test_a_config_without_a_vlm_or_a_tokenizer_is_refused(
        self, write_config: Callable[..., Path]
    ) -> None:
        with pytest.raises(ValueError, match="no \\[vlm\\] section"):
            check_renderer_config(JobConfig())
        config = load_config(write_config(), cli_args=["--data.tokenizer_path="])
        with pytest.raises(ValueError, match="data.tokenizer_path is not set"):
            check_renderer_config(config)


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

    @pytest.mark.parametrize("train_reply_prefix", [False, True])
    @pytest.mark.parametrize("kind", ["qa", "conversation"])
    def test_the_recipe_prompt_ends_with_the_empty_think_block(
        self, bpe_tokenizer: Any, kind: str, train_reply_prefix: bool
    ) -> None:
        """Training the think block or not, the prompt is the generation prompt that holds it."""
        document = _document(user="What is shown?", kind=kind)
        renderer = _renderer(bpe_tokenizer, train_reply_prefix=train_reply_prefix)
        assert build_prompt(renderer, document) == THINKING_PROMPT

    def test_a_lead_space_starts_the_question(self, bpe_tokenizer: Any) -> None:
        renderer = _renderer(bpe_tokenizer, lead_space=True)
        for images in (0, 1):
            prompt = build_prompt(renderer, _document(user="What is shown?", images=images))
            assert prompt == THINKING_PROMPT.replace("user\nWhat", "user\n What")


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
        assert torch.equal(inputs.pixel_values, sample["pixel_values"][None])
        assert inputs.pixel_values.shape == (1, 3, 16, 16)

    def test_pixels_follow_the_renderers_size_and_normalization(self, tokenizer: Any) -> None:
        renderer = _renderer(
            tokenizer, image_size=8, image_mean=(0.1, 0.2, 0.3), image_std=(0.5, 0.25, 0.2)
        )
        inputs = document_inputs(renderer, _document())
        assert torch.equal(
            inputs.pixel_values, renderer.render(_document(), ["cat"])["pixel_values"][None]
        )
        assert inputs.pixel_values.shape == (1, 3, 8, 8)

    @pytest.mark.parametrize(
        ("images", "tile_grid", "views"), [(0, 0, 0), (2, 0, 2), (0, 2, 0), (1, 2, 5), (2, 2, 10)]
    )
    @pytest.mark.parametrize("max_images", [1, None])
    def test_every_view_of_every_image_is_an_input(
        self, tokenizer: Any, images: int, tile_grid: int, views: int, max_images: int | None
    ) -> None:
        if max_images == 1 and images > 1:
            return
        renderer = _renderer(tokenizer, tile_grid=tile_grid, max_images=max_images)
        inputs = document_inputs(renderer, _document(images=images))
        assert inputs.pixel_values.shape == (views, 3, 16, 16)
        if images:
            sample = renderer.render(_document(images=images), ["cat"])["pixel_values"]
            assert torch.equal(inputs.pixel_values, sample.reshape(-1, 3, 16, 16))

    def test_a_document_with_several_turns_is_refused(self, tokenizer: Any) -> None:
        document = Document(
            images=(_image(),),
            turns=(Turn("how many ?", ("one",)), Turn("which color ?", ("red",))),
            kind="conversation",
        )
        with pytest.raises(UnsupportedDocumentError, match="one turn, got 2"):
            document_inputs(_renderer(tokenizer), document)
