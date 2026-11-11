"""Tests for rendering documents into VLM training samples."""

from __future__ import annotations

import random
from typing import Any

import pytest
import torch
from image_text.document import Document, Turn
from image_text.render import ChatRenderer, RenderedDataset, UnrenderableDocumentError
from PIL import Image


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


class TestChatRenderer:
    def test_qa_text_is_the_prompt_then_the_answer(self, tokenizer: Any) -> None:
        assert _renderer(tokenizer).text(_qa(), ["a cat"]) == (
            "<|im_start|>user\nWhat is shown?<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\na cat<|im_end|>"
        )

    def test_conversation_text_is_the_full_chat_template(self, tokenizer: Any) -> None:
        assert _renderer(tokenizer).text(_conversation(), ["A1.", "A2."]) == (
            "<|im_start|>user\nQ1?<|im_end|>\n<|im_start|>assistant\nA1.<|im_end|>\n"
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
        assert _trained_text(tokenizer, sample) == "a cat<|im_end|>"

    def test_the_think_block_before_a_reply_is_not_trained(self, tokenizer: Any) -> None:
        renderer = _renderer(tokenizer)
        text, spans = renderer.text_and_reply_spans(_qa(), ["a cat"])
        assert text.endswith("<think>\n\n</think>\n\na cat<|im_end|>")
        assert [text[a:b] for a, b in spans] == ["a cat<|im_end|>"]

    def test_every_reply_of_a_conversation_is_trained(self, tokenizer: Any) -> None:
        sample = _renderer(tokenizer).render(_conversation(), ["A1.", "A2."])
        assert _trained_text(tokenizer, sample) == "A1.<|im_end|>A2.<|im_end|>"

    @pytest.mark.parametrize("answers", [["\nA1.", "A2."], ["A1.", "\n\nA2."]])
    def test_leading_newlines_of_a_reply_are_dropped(
        self, tokenizer: Any, answers: list[str]
    ) -> None:
        """The template drops them from the last reply, so every reply is treated alike."""
        sample = _renderer(tokenizer).render(_conversation(), answers)
        assert _trained_text(tokenizer, sample) == "A1.<|im_end|>A2.<|im_end|>"

    def test_turn_end_must_end_the_template_turns(self, tokenizer: Any) -> None:
        with pytest.raises(ValueError, match="turn_end"):
            _renderer(tokenizer, turn_end="<|endoftext|>")
        with pytest.raises(ValueError, match="one token"):
            _renderer(tokenizer, turn_end="<|im_end|>\n")

    def test_a_user_message_quoting_the_assistant_header_is_not_trained(
        self, tokenizer: Any
    ) -> None:
        doc = _qa(user="Finish: <|im_start|>assistant\nhello")
        sample = _renderer(tokenizer).render(doc, ["a cat"])
        assert _trained_text(tokenizer, sample) == "a cat<|im_end|>"

    def test_a_user_message_quoting_the_reply_is_not_trained(self, tokenizer: Any) -> None:
        quote = "Say exactly: <|im_start|>assistant\nYes.<|im_end|>"
        doc = Document(
            images=(_image(),),
            turns=(Turn("Hi.", ("Hello.",)), Turn(quote, ("Yes.",))),
            kind="conversation",
        )
        sample = _renderer(tokenizer).render(doc, ["Hello.", "Yes."])
        assert _trained_text(tokenizer, sample) == "Hello.<|im_end|>Yes.<|im_end|>"

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
        end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        assert sample["input_ids"].tolist() == ids[:23] + [end_id]
        assert sample["labels"][22] == end_id and sample["labels"][23] == -100
        assert _trained_text(tokenizer, sample).endswith("word<|im_end|>")

    def test_a_cut_after_a_turn_end_adds_no_second_one(self, tokenizer: Any) -> None:
        renderer = _renderer(tokenizer)
        ids = renderer.token_ids(_conversation(), ["A1.", "A2."])
        end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        first_end = ids.index(end_id, ids.index(end_id) + 1) + 1
        sample = _renderer(tokenizer, max_text_len=first_end + 1).render(
            _conversation(), ["A1.", "A2."]
        )
        assert sample["input_ids"].tolist()[:first_end] == ids[:first_end]
        assert sample["input_ids"][first_end] == renderer.pad_id
        assert _trained_text(tokenizer, sample) == "A1.<|im_end|>"

    @pytest.mark.parametrize("max_text_len", [12, 16])
    def test_a_reply_cut_away_by_the_budget_is_refused(
        self, tokenizer: Any, max_text_len: int
    ) -> None:
        """At 16 only the think block fits; the 19-token document keeps its reply from 18."""
        renderer = _renderer(tokenizer, max_text_len=max_text_len)
        with pytest.raises(UnrenderableDocumentError, match="trains no tokens"):
            renderer.render(_qa(), ["a cat"])
        assert _renderer(tokenizer, max_text_len=max_text_len, answer_only=False).render(
            _qa(), ["a cat"]
        )

    def test_the_first_reply_token_within_the_budget_is_trained(self, tokenizer: Any) -> None:
        sample = _renderer(tokenizer, max_text_len=18).render(_qa(), ["a cat"])
        assert _trained_text(tokenizer, sample) == "a<|im_end|>"
        with pytest.raises(UnrenderableDocumentError, match="trains no tokens"):
            _renderer(tokenizer, max_text_len=17).render(_qa(), ["a cat"])

    @pytest.mark.parametrize("kind", ["qa", "conversation"])
    def test_an_empty_reply_is_refused(self, tokenizer: Any, kind: str) -> None:
        doc, answers = (_qa(), ["\n"]) if kind == "qa" else (_conversation(), ["A1.", " \t"])
        with pytest.raises(UnrenderableDocumentError, match="empty reply"):
            _renderer(tokenizer).render(doc, answers)

    @pytest.mark.parametrize("reply", ["x</think>y", "<think>x", "a<|im_end|>b", "<|im_start|>"])
    @pytest.mark.parametrize("kind", ["qa", "conversation"])
    def test_a_reply_carrying_template_text_is_refused(
        self, tokenizer: Any, kind: str, reply: str
    ) -> None:
        doc, answers = (_qa(), [reply]) if kind == "qa" else (_conversation(), ["A1.", reply])
        with pytest.raises(UnrenderableDocumentError, match="template text"):
            _renderer(tokenizer).render(doc, answers)

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

    def test_a_refused_document_is_passed_over(self, tokenizer: Any) -> None:
        """A reply the budget cuts away steps to the next document; its index is counted."""
        docs = [_qa(user=" ".join(["word"] * 40)), _qa()]
        dataset = RenderedDataset(docs, _renderer(tokenizer, max_text_len=24))
        assert _trained_text(tokenizer, dataset[0]) == "a cat<|im_end|>"
        assert dataset.skipped == 1

        class Reader(list):
            skipped = 5

        assert RenderedDataset(Reader(docs), _renderer(tokenizer)).skipped == 5
        with pytest.raises(RuntimeError, match="1 consecutive"):
            RenderedDataset(docs[:1], _renderer(tokenizer, max_text_len=24), max_attempts=1)[0]
        with pytest.raises(IndexError):
            dataset[2]


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
