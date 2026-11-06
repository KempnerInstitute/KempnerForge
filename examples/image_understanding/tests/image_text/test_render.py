"""Tests for rendering documents into VLM training samples."""

from __future__ import annotations

import random
from typing import Any

import pytest
import torch
from image_text.document import Document, Turn
from image_text.render import ChatRenderer, RenderedDataset, assistant_span_mask
from PIL import Image

QWEN = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def tokenizer() -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(QWEN)


def _image(size: tuple[int, int] = (12, 9)) -> Image.Image:
    return Image.new("RGB", size, (255, 0, 0))


def _qa(user: str = "What is shown?", answers: tuple[str, ...] = ("a cat",)) -> Document:
    return Document(images=(_image(),), turns=(Turn(user, answers),))


def _conversation() -> Document:
    return Document(
        images=(_image(),),
        turns=(Turn("Q1?", ("A1.",)), Turn("Q2?", ("A2.",))),
        kind="conversation",
    )


def _renderer(tokenizer: Any, **overrides: Any) -> ChatRenderer:
    kwargs: dict[str, Any] = {"max_text_len": 64, "image_size": 16, "answer_only": True}
    kwargs.update(overrides)
    return ChatRenderer(tokenizer, **kwargs)


def _trained_text(tokenizer: Any, sample: dict[str, torch.Tensor]) -> str:
    labels = sample["labels"]
    return tokenizer.decode(labels[labels != -100])


class TestAssistantSpanMask:
    def test_marks_each_reply_through_its_turn_end(self) -> None:
        ids = [1, 7, 8, 5, 5, 9, 2, 7, 8, 6, 9, 3]
        mask = assistant_span_mask(ids, generation_prompt=[7, 8], turn_end_id=9)
        assert [i for i, m in enumerate(mask) if m] == [3, 4, 5, 9, 10]

    def test_unterminated_reply_runs_to_the_end(self) -> None:
        mask = assistant_span_mask([7, 8, 5, 5], generation_prompt=[7, 8], turn_end_id=9)
        assert mask == [False, False, True, True]

    def test_without_a_generation_prompt_nothing_is_marked(self) -> None:
        assert not any(assistant_span_mask([1, 2, 3], generation_prompt=[7, 8], turn_end_id=9))


class TestChatRenderer:
    def test_qa_text_is_the_prompt_then_the_answer(self, tokenizer: Any) -> None:
        assert _renderer(tokenizer).text(_qa(), ["a cat"]) == (
            "<|im_start|>user\n What is shown?<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\na cat<|im_end|>"
        )

    def test_conversation_text_is_the_full_chat_template(self, tokenizer: Any) -> None:
        assert _renderer(tokenizer).text(_conversation(), ["A1.", "A2."]) == (
            "<|im_start|>user\n Q1?<|im_end|>\n<|im_start|>assistant\nA1.<|im_end|>\n"
            "<|im_start|>user\nQ2?<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\nA2.<|im_end|>\n"
        )

    def test_image_placeholder_leads_the_first_user_turn(self, tokenizer: Any) -> None:
        renderer = _renderer(tokenizer, image_placeholder="<|vision_start|><|vision_end|>")
        assert renderer.text(_qa(), ["a cat"]).startswith(
            "<|im_start|>user\n<|vision_start|><|vision_end|> What is shown?"
        )

    def test_answer_only_trains_the_reply_and_its_turn_end(self, tokenizer: Any) -> None:
        sample = _renderer(tokenizer).render(_qa(), ["a cat"])
        assert _trained_text(tokenizer, sample) == "<think>\n\n</think>\n\na cat<|im_end|>"

    def test_every_reply_of_a_conversation_is_trained(self, tokenizer: Any) -> None:
        sample = _renderer(tokenizer).render(_conversation(), ["A1.", "A2."])
        assert _trained_text(tokenizer, sample) == (
            "A1.<|im_end|><think>\n\n</think>\n\nA2.<|im_end|>"
        )

    def test_full_loss_trains_every_next_token(self, tokenizer: Any) -> None:
        renderer = _renderer(tokenizer, answer_only=False)
        ids = renderer.token_ids(_qa(), ["a cat"])
        sample = renderer.render(_qa(), ["a cat"])
        n = len(ids)
        assert sample["labels"][: n - 1].tolist() == ids[1:]
        assert (sample["labels"][n - 1 :] == -100).all()

    def test_labels_are_next_token_targets(self, tokenizer: Any) -> None:
        sample = _renderer(tokenizer).render(_qa(), ["a cat"])
        positions = (sample["labels"] != -100).nonzero().flatten()
        for i in positions.tolist():
            assert sample["labels"][i] == sample["input_ids"][i + 1]

    def test_short_documents_are_padded(self, tokenizer: Any) -> None:
        renderer = _renderer(tokenizer)
        n = len(renderer.token_ids(_qa(), ["a cat"]))
        sample = renderer.render(_qa(), ["a cat"])
        assert sample["input_ids"].shape == (64,)
        assert (sample["input_ids"][n:] == renderer.pad_id).all()
        assert (sample["labels"][n - 1 :] == -100).all()

    def test_long_documents_keep_their_head(self, tokenizer: Any) -> None:
        renderer = _renderer(tokenizer, max_text_len=24)
        long_answer = " ".join(["word"] * 40)
        ids = renderer.token_ids(_qa(), [long_answer])
        sample = renderer.render(_qa(), [long_answer])
        assert sample["input_ids"].tolist() == ids[:24]
        assert sample["labels"][23] == -100
        assert (sample["labels"] != -100).sum() > 0

    def test_a_reply_cut_away_by_the_budget_is_refused(self, tokenizer: Any) -> None:
        renderer = _renderer(tokenizer, max_text_len=12)
        with pytest.raises(ValueError, match="trains no tokens"):
            renderer.render(_qa(), ["a cat"])
        assert _renderer(tokenizer, max_text_len=12, answer_only=False).render(_qa(), ["a cat"])

    def test_turn_end_must_be_one_token(self, tokenizer: Any) -> None:
        with pytest.raises(ValueError, match="single token"):
            _renderer(tokenizer, turn_end="</s>")

    def test_an_empty_qa_answer_trains_only_the_empty_think_block(self, tokenizer: Any) -> None:
        renderer = _renderer(tokenizer)
        text = renderer.text(_qa(), [""])
        assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        sample = renderer.render(_qa(), [""])
        assert _trained_text(tokenizer, sample) == "<think>\n\n</think>\n\n"

    def test_pixel_values_are_square_and_normalized(self, tokenizer: Any) -> None:
        pixels = _renderer(tokenizer).render(_qa(), ["a cat"])["pixel_values"]
        assert pixels.shape == (3, 16, 16)
        assert torch.allclose(pixels[0], torch.ones(16, 16))
        assert torch.allclose(pixels[1:], -torch.ones(2, 16, 16))

    def test_multi_image_documents_are_refused(self, tokenizer: Any) -> None:
        doc = Document(images=(_image(), _image()), turns=(Turn("Compare.", ("Same.",)),))
        with pytest.raises(ValueError, match="single-image"):
            _renderer(tokenizer).render(doc, ["Same."])

    def test_answer_count_must_match_turns(self, tokenizer: Any) -> None:
        with pytest.raises(ValueError, match="2 turns but 1 answers"):
            _renderer(tokenizer).text(_conversation(), ["A1."])

    def test_rejects_a_tiny_text_budget(self, tokenizer: Any) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            _renderer(tokenizer, max_text_len=1)


class TestRenderedDataset:
    def test_reply_choice_is_fixed_by_seed_and_index(self, tokenizer: Any) -> None:
        docs = [_qa(answers=tuple(f"answer {i}" for i in range(20)))] * 4
        renderer = _renderer(tokenizer)
        first = [RenderedDataset(docs, renderer, seed=3)[i]["labels"] for i in range(4)]
        again = [RenderedDataset(docs, renderer, seed=3)[i]["labels"] for i in range(4)]
        assert all(torch.equal(a, b) for a, b in zip(first, again, strict=True))
        assert len({tuple(labels.tolist()) for labels in first}) > 1

    def test_draw_matches_a_generator_seeded_the_same_way(self, tokenizer: Any) -> None:
        doc = _qa(answers=("x", "y", "z"))
        renderer = _renderer(tokenizer)
        expected = renderer.render(doc, [random.Random("7:2").choice(("x", "y", "z"))])
        assert torch.equal(
            RenderedDataset([doc] * 3, renderer, seed=7)[2]["labels"], expected["labels"]
        )

    def test_length_follows_the_reader(self, tokenizer: Any) -> None:
        assert len(RenderedDataset([_qa()] * 5, _renderer(tokenizer))) == 5


class TestDocument:
    def test_a_qa_document_has_one_turn(self) -> None:
        with pytest.raises(ValueError, match="exactly one turn"):
            Document(images=(), turns=(Turn("a", ("b",)), Turn("c", ("d",))))

    def test_a_turn_needs_an_answer(self) -> None:
        with pytest.raises(ValueError, match="at least one answer"):
            Turn("a", ())

    def test_unknown_kind_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown document kind"):
            Document(images=(), turns=(Turn("a", ("b",)),), kind="essay")
