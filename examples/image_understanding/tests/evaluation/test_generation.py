"""Tests for greedy decoding of prompts after their images."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from evaluation.generation import Generation, greedy_decode, stop_token_ids
from image_text.render import ChatRenderer

from kempnerforge.config.schema import AdapterConfig, ModelConfig, VisionEncoderConfig, VLMConfig
from kempnerforge.model.vlm import JointDecoderStrategy, VLMWrapper, build_vlm_wrapper

VOCAB_SIZE = 40
MAX_TEXT_LEN = 32
PROMPTS = [[5, 6, 7, 8], [9, 10], [11, 12, 13, 14, 15, 16, 17], [18], [5, 6, 7, 8]]


@pytest.fixture(scope="module")
def model() -> VLMWrapper:
    torch.manual_seed(0)
    return build_vlm_wrapper(
        ModelConfig(
            dim=32, n_layers=2, n_heads=2, vocab_size=VOCAB_SIZE, max_seq_len=48, init_std=0.2
        ),
        VisionEncoderConfig(type="random", feature_dim=24, num_tokens=4),
        AdapterConfig(),
        VLMConfig(max_text_len=MAX_TEXT_LEN),
    ).eval()


@pytest.fixture(scope="module")
def pixels() -> torch.Tensor:
    return torch.randn(len(PROMPTS), 3, 16, 16, generator=torch.Generator().manual_seed(1))


def _decode(
    model: VLMWrapper, pixels: torch.Tensor, prompts: list[list[int]], **overrides: Any
) -> list[Generation]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": 6,
        "stop_ids": (),
        "pad_id": 0,
        "max_text_len": MAX_TEXT_LEN,
    }
    kwargs.update(overrides)
    return greedy_decode(model, pixels, prompts, **kwargs)


def _one_at_a_time(
    model: VLMWrapper, pixels: torch.Tensor, prompt: list[int], steps: int
) -> list[int]:
    """Greedy decoding through the wrapper's own forward, which encodes the image every step."""
    tokens = list(prompt)
    with torch.no_grad():
        for _ in range(steps):
            logits, _ = model(pixels[None], torch.tensor([tokens]))
            tokens.append(int(logits[0, -1].argmax()))
    return tokens[len(prompt) :]


class _Calls:
    """Counts the calls of a module's forward and keeps the batch size of each."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, module: torch.nn.Module) -> None:
        self.batches: list[int] = []
        forward = module.forward

        def counted(*args: Any, **kwargs: Any) -> Any:
            first = args[0] if args else kwargs["tokens"]
            self.batches.append(first.shape[0])
            return forward(*args, **kwargs)

        monkeypatch.setattr(module, "forward", counted)


class TestGreedyDecode:
    def test_a_batch_decodes_what_each_row_decodes_through_the_wrapper(
        self, model: VLMWrapper, pixels: torch.Tensor
    ) -> None:
        """Rows of different lengths, padded into one batch with the images encoded once."""
        generations = _decode(model, pixels, PROMPTS)
        for row, prompt in enumerate(PROMPTS):
            expected = _one_at_a_time(model, pixels[row], prompt, steps=6)
            assert generations[row] == Generation(tuple(expected), stopped=False)
        assert len({generation.token_ids for generation in generations}) > 1

    def test_the_image_changes_the_answer(self, model: VLMWrapper, pixels: torch.Tensor) -> None:
        first, _, _, _, last = _decode(model, pixels, PROMPTS)
        assert PROMPTS[0] == PROMPTS[4]
        assert first.token_ids != last.token_ids

    @pytest.mark.parametrize("groups", [[[0], [1], [2], [3], [4]], [[4, 2], [0, 3, 1]], [[3, 2]]])
    def test_a_row_decodes_the_same_in_any_batch(
        self, model: VLMWrapper, pixels: torch.Tensor, groups: list[list[int]]
    ) -> None:
        together = _decode(model, pixels, PROMPTS)
        for rows in groups:
            apart = _decode(model, pixels[rows], [PROMPTS[row] for row in rows])
            assert apart == [together[row] for row in rows]

    def test_padding_does_not_reach_a_row(self, model: VLMWrapper, pixels: torch.Tensor) -> None:
        assert _decode(model, pixels, PROMPTS, pad_id=3) == _decode(model, pixels, PROMPTS)

    def test_a_stop_token_ends_its_row_and_is_left_out(
        self, model: VLMWrapper, pixels: torch.Tensor
    ) -> None:
        free = _decode(model, pixels, PROMPTS)
        stops = {free[0].token_ids[2], free[2].token_ids[4]}
        generations = _decode(model, pixels, PROMPTS, stop_ids=stops)
        assert generations[0].stopped and generations[2].stopped
        for cut, whole in zip(generations, free, strict=True):
            ends = [step for step, token in enumerate(whole.token_ids) if token in stops]
            assert cut == (Generation(whole.token_ids[: ends[0]], True) if ends else whole)

    def test_a_row_can_stop_before_its_first_token(
        self, model: VLMWrapper, pixels: torch.Tensor
    ) -> None:
        free = _decode(model, pixels, PROMPTS)
        generations = _decode(model, pixels, PROMPTS, stop_ids={free[1].token_ids[0]})
        assert generations[1] == Generation((), stopped=True)

    def test_a_row_without_a_stop_token_runs_to_the_cap(
        self, model: VLMWrapper, pixels: torch.Tensor
    ) -> None:
        for cap in (1, 3):
            generations = _decode(model, pixels, PROMPTS, max_new_tokens=cap)
            assert all(len(generation.token_ids) == cap for generation in generations)
            assert not any(generation.stopped for generation in generations)

    def test_images_are_encoded_once_and_ended_rows_leave_the_batch(
        self, model: VLMWrapper, pixels: torch.Tensor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        free = _decode(model, pixels, PROMPTS)
        stop = free[0].token_ids[2]
        encoder = _Calls(monkeypatch, model.vision_encoder)
        transformer = _Calls(monkeypatch, model.transformer)
        generations = _decode(model, pixels, PROMPTS, stop_ids={stop})
        assert encoder.batches == [len(PROMPTS)]
        still_decoding = [
            sum(len(generation.token_ids) >= step for generation in generations)
            for step in range(6)
        ]
        assert transformer.batches == [rows for rows in still_decoding if rows]
        assert transformer.batches[-1] < transformer.batches[0]

    def test_the_last_position_of_the_text_budget_is_used(
        self, model: VLMWrapper, pixels: torch.Tensor
    ) -> None:
        prompt = list(range(4, 4 + MAX_TEXT_LEN - 2))
        (generation,) = _decode(model, pixels[:1], [prompt], max_new_tokens=2)
        assert list(generation.token_ids) == _one_at_a_time(model, pixels[0], prompt, steps=2)

    @pytest.mark.parametrize(
        ("prompts", "overrides", "message"),
        [
            ([list(range(4, 4 + MAX_TEXT_LEN - 5)), [5]], {}, "do not fit the text budget of 32"),
            ([[5, 6], []], {}, "a prompt has no tokens"),
            ([[5, 6]] * 3, {}, "2 images for 3 prompts"),
            ([], {}, "2 images for 0 prompts"),
            ([[5], [6]], {"max_new_tokens": 0}, "max_new_tokens must be at least 1"),
            ([[5], [6]], {"max_text_len": 45}, "exceed the model's max_seq_len=48"),
        ],
    )
    def test_the_token_budget_is_checked_before_anything_runs(
        self,
        model: VLMWrapper,
        pixels: torch.Tensor,
        monkeypatch: pytest.MonkeyPatch,
        prompts: list[list[int]],
        overrides: dict[str, Any],
        message: str,
    ) -> None:
        encoder = _Calls(monkeypatch, model.vision_encoder)
        with pytest.raises(ValueError, match=message):
            _decode(model, pixels[:2], prompts, **overrides)
        assert encoder.batches == []

    @pytest.mark.parametrize("field", ["modality_ids", "key_padding_mask", "position_ids"])
    def test_a_context_laid_out_along_the_text_is_refused(
        self, model: VLMWrapper, pixels: torch.Tensor, monkeypatch: pytest.MonkeyPatch, field: str
    ) -> None:
        """Such a context fits one text length, so reusing it at the next step would be wrong."""

        class AlongTheText(JointDecoderStrategy):
            def prepare(self, wrapper: Any, pixel_values: Any, input_ids: Any, frame_mask=None):
                context = super().prepare(wrapper, pixel_values, input_ids)
                along = torch.zeros(input_ids.shape[0], 4 + input_ids.shape[1], dtype=torch.long)
                value = along.bool() if field == "key_padding_mask" else along
                return replace(context, **{field: value})

        monkeypatch.setattr(model, "strategy", AlongTheText())
        with pytest.raises(ValueError, match=f"reuses them at every step.*{field}"):
            _decode(model, pixels, PROMPTS)

    def test_a_context_that_keeps_image_positions_in_the_output_is_refused(
        self, model: VLMWrapper, pixels: torch.Tensor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class KeepsImagePositions(JointDecoderStrategy):
            def prepare(self, wrapper: Any, pixel_values: Any, input_ids: Any, frame_mask=None):
                context = super().prepare(wrapper, pixel_values, input_ids)
                return replace(context, output_slice=slice(2, None))

        monkeypatch.setattr(model, "strategy", KeepsImagePositions())
        with pytest.raises(ValueError, match="output_slice past them"):
            _decode(model, pixels, PROMPTS)


class TestStopTokenIds:
    def _renderer(self, tokenizer_path: Path, **tokens: Any) -> ChatRenderer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokens)
        return ChatRenderer(tokenizer, max_text_len=32, image_size=16, answer_only=True)

    def test_the_turn_end_stops_a_reply(self, tokenizer_path: Path) -> None:
        assert stop_token_ids(self._renderer(tokenizer_path)) == {2}

    def test_an_eos_other_than_the_turn_end_stops_it_too(self, tokenizer_path: Path) -> None:
        renderer = self._renderer(tokenizer_path, eos_token="<|endoftext|>")
        assert stop_token_ids(renderer) == {0, 2}

    def test_a_tokenizer_without_eos_stops_at_the_turn_end(self, tokenizer_path: Path) -> None:
        renderer = self._renderer(tokenizer_path)
        renderer.tokenizer.eos_token = None
        assert stop_token_ids(renderer) == {2}
