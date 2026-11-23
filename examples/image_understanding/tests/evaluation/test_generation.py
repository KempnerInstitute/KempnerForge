"""Tests for greedy decoding of documents laid out as packed training rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from evaluation.generation import Generation, footprint, greedy_decode, stop_token_ids
from evaluation.inputs import ModelInputs
from image_text.render import ChatRenderer

from kempnerforge.config.schema import AdapterConfig, ModelConfig, VisionEncoderConfig, VLMConfig
from kempnerforge.data.vlm_dataset import pack_vlm_samples
from kempnerforge.model.vlm import VLMWrapper, build_vlm_wrapper

VOCAB_SIZE = 40
MAX_SEQ_LEN = 48
TOKENS = 4
PROMPTS = [[5, 6, 7, 8], [9, 10], [11, 12, 13, 14, 15, 16, 17], [18], [5, 6, 7, 8]]


@pytest.fixture(scope="module")
def model() -> VLMWrapper:
    torch.manual_seed(0)
    return build_vlm_wrapper(
        ModelConfig(
            dim=32,
            n_layers=2,
            n_heads=2,
            vocab_size=VOCAB_SIZE,
            max_seq_len=MAX_SEQ_LEN,
            init_std=0.2,
        ),
        VisionEncoderConfig(type="random", feature_dim=24, num_tokens=TOKENS),
        AdapterConfig(),
        VLMConfig(max_text_len=32),
    ).eval()


@pytest.fixture(scope="module")
def pixels() -> torch.Tensor:
    return torch.randn(len(PROMPTS), 3, 16, 16, generator=torch.Generator().manual_seed(1))


def _rows(pixels: torch.Tensor, prompts: list[list[int]]) -> list[ModelInputs]:
    return [
        ModelInputs(pixel_values=pixels[row : row + 1], prompt_ids=tuple(prompt))
        for row, prompt in enumerate(prompts)
    ]


def _decode(model: VLMWrapper, rows: list[ModelInputs], **overrides: Any) -> list[Generation]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": 6,
        "stop_ids": (),
        "pad_id": 0,
        "row_len": MAX_SEQ_LEN,
    }
    kwargs.update(overrides)
    return greedy_decode(model, rows, **kwargs)


def _one_at_a_time(
    model: VLMWrapper, pixels: torch.Tensor, prompt: list[int], steps: int
) -> list[int]:
    """Greedy decoding through the wrapper's own unpacked forward, the image a prefix."""
    tokens = list(prompt)
    with torch.no_grad():
        for _ in range(steps):
            logits, _ = model(pixels[None], torch.tensor([tokens]))
            tokens.append(int(logits[0, -1].argmax()))
    return tokens[len(prompt) :]


def _packed_one_at_a_time(
    model: VLMWrapper, views: torch.Tensor, prompt: list[int], steps: int, view_groups: bool
) -> list[int]:
    """Greedy decoding through the wrapper's forward of a packed training row of the document."""
    tokens = list(prompt)
    with torch.no_grad():
        for _ in range(steps):
            sample = {
                "pixel_values": views,
                "input_ids": torch.tensor(tokens),
                "labels": torch.full((len(tokens),), -100),
                "text_len": torch.tensor(len(tokens)),
            }
            batch = pack_vlm_samples(
                [sample],
                seq_len=views.shape[0] * TOKENS + len(tokens),
                tokens_per_image=TOKENS,
                pad_id=0,
            )
            groups = None
            if view_groups:
                groups = torch.full_like(batch["input_ids"], -1)
                for view, start in enumerate(batch["image_starts"].tolist()):
                    groups[0, start : start + TOKENS] = view
            logits, _ = model(
                batch["pixel_values"],
                batch["input_ids"],
                doc_ids=batch["doc_ids"],
                position_ids=batch["position_ids"],
                attention_group_ids=groups,
                image_rows=batch["image_rows"],
                image_starts=batch["image_starts"],
                image_tokens=int(batch["image_tokens"]),
                text_rows=batch["text_rows"],
                text_positions=batch["text_positions"],
            )
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
        generations = _decode(model, _rows(pixels, PROMPTS))
        for row, prompt in enumerate(PROMPTS):
            expected = _one_at_a_time(model, pixels[row], prompt, steps=6)
            assert generations[row] == Generation(tuple(expected), stopped=False)
        assert len({generation.token_ids for generation in generations}) > 1

    @pytest.mark.parametrize("view_groups", [False, True])
    def test_documents_decode_as_their_packed_training_rows(
        self, model: VLMWrapper, view_groups: bool
    ) -> None:
        """Several views, one image and none, each as the packed training forward reads it."""
        generator = torch.Generator().manual_seed(2)
        rows = [
            ModelInputs(torch.randn(3, 3, 16, 16, generator=generator), (5, 6, 7)),
            ModelInputs(torch.zeros(0, 3, 16, 16), (9, 10, 11, 12)),
            ModelInputs(torch.randn(1, 3, 16, 16, generator=generator), (13,)),
        ]
        generations = _decode(model, rows, view_groups=view_groups)
        for row, generation in zip(rows, generations, strict=True):
            expected = _packed_one_at_a_time(
                model, row.pixel_values, list(row.prompt_ids), 6, view_groups
            )
            assert list(generation.token_ids) == expected

    @pytest.mark.parametrize("view_groups", [False, True])
    def test_each_view_is_its_own_group_when_asked(
        self, model: VLMWrapper, monkeypatch: pytest.MonkeyPatch, view_groups: bool
    ) -> None:
        seen: list[Any] = []
        forward = model.transformer.forward

        def kept(*args: Any, **kwargs: Any) -> Any:
            seen.append(kwargs.get("attention_group_ids"))
            return forward(*args, **kwargs)

        monkeypatch.setattr(model.transformer, "forward", kept)
        rows = [
            ModelInputs(torch.zeros(2, 3, 16, 16), (5, 6)),
            ModelInputs(torch.zeros(0, 3, 16, 16), (7,)),
            ModelInputs(torch.zeros(1, 3, 16, 16), (8, 9, 10)),
        ]
        _decode(model, rows, max_new_tokens=1, view_groups=view_groups)
        (groups,) = seen
        if not view_groups:
            assert groups is None
            return
        expected = torch.full((3, 10), -1)
        expected[0, :4], expected[0, 4:8], expected[2, :4] = 0, 1, 2
        assert torch.equal(groups.cpu(), expected)

    def test_the_image_changes_the_answer(self, model: VLMWrapper, pixels: torch.Tensor) -> None:
        first, _, _, _, last = _decode(model, _rows(pixels, PROMPTS))
        assert PROMPTS[0] == PROMPTS[4]
        assert first.token_ids != last.token_ids

    @pytest.mark.parametrize("groups", [[[0], [1], [2], [3], [4]], [[4, 2], [0, 3, 1]], [[3, 2]]])
    def test_a_row_decodes_the_same_in_any_batch(
        self, model: VLMWrapper, pixels: torch.Tensor, groups: list[list[int]]
    ) -> None:
        rows = _rows(pixels, PROMPTS)
        together = _decode(model, rows)
        for chosen in groups:
            apart = _decode(model, [rows[row] for row in chosen])
            assert apart == [together[row] for row in chosen]

    def test_padding_does_not_reach_a_row(self, model: VLMWrapper, pixels: torch.Tensor) -> None:
        rows = _rows(pixels, PROMPTS)
        assert _decode(model, rows, pad_id=3) == _decode(model, rows)

    def test_a_stop_token_ends_its_row_and_is_left_out(
        self, model: VLMWrapper, pixels: torch.Tensor
    ) -> None:
        rows = _rows(pixels, PROMPTS)
        free = _decode(model, rows)
        stops = {free[0].token_ids[2], free[2].token_ids[4]}
        generations = _decode(model, rows, stop_ids=stops)
        assert generations[0].stopped and generations[2].stopped
        for cut, whole in zip(generations, free, strict=True):
            ends = [step for step, token in enumerate(whole.token_ids) if token in stops]
            assert cut == (Generation(whole.token_ids[: ends[0]], True) if ends else whole)

    def test_a_row_can_stop_before_its_first_token(
        self, model: VLMWrapper, pixels: torch.Tensor
    ) -> None:
        rows = _rows(pixels, PROMPTS)
        free = _decode(model, rows)
        generations = _decode(model, rows, stop_ids={free[1].token_ids[0]})
        assert generations[1] == Generation((), stopped=True)

    def test_a_row_without_a_stop_token_runs_to_the_cap(
        self, model: VLMWrapper, pixels: torch.Tensor
    ) -> None:
        for cap in (1, 3):
            generations = _decode(model, _rows(pixels, PROMPTS), max_new_tokens=cap)
            assert all(len(generation.token_ids) == cap for generation in generations)
            assert not any(generation.stopped for generation in generations)

    def test_images_are_encoded_once_and_ended_rows_leave_the_batch(
        self, model: VLMWrapper, pixels: torch.Tensor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = _rows(pixels, PROMPTS)
        free = _decode(model, rows)
        stop = free[0].token_ids[2]
        encoder = _Calls(monkeypatch, model.vision_encoder)
        transformer = _Calls(monkeypatch, model.transformer)
        generations = _decode(model, rows, stop_ids={stop})
        assert encoder.batches == [len(PROMPTS)]
        still_decoding = [
            sum(len(generation.token_ids) >= step for generation in generations)
            for step in range(6)
        ]
        assert transformer.batches == [count for count in still_decoding if count]
        assert transformer.batches[-1] < transformer.batches[0]

    def test_the_last_position_of_the_row_is_used(
        self, model: VLMWrapper, pixels: torch.Tensor
    ) -> None:
        prompt = [4 + index % 30 for index in range(MAX_SEQ_LEN - TOKENS - 2)]
        rows = _rows(pixels[:1], [prompt])
        assert footprint(model, rows[0]) + 2 == MAX_SEQ_LEN
        (generation,) = _decode(model, rows, max_new_tokens=2)
        assert list(generation.token_ids) == _one_at_a_time(model, pixels[0], prompt, steps=2)

    @pytest.mark.parametrize(
        ("rows", "overrides", "message"),
        [
            ([[5] * (MAX_SEQ_LEN - TOKENS - 5), [5]], {}, "do not fit a row of 48"),
            ([[5, 6], []], {}, "a prompt has no tokens"),
            ([], {}, "no document to decode"),
            ([[5], [6]], {"max_new_tokens": 0}, "max_new_tokens must be at least 1"),
            ([[5], [6]], {"row_len": 49}, "exceeds the model's max_seq_len=48"),
        ],
    )
    def test_the_budget_is_checked_before_anything_runs(
        self,
        model: VLMWrapper,
        pixels: torch.Tensor,
        monkeypatch: pytest.MonkeyPatch,
        rows: list[list[int]],
        overrides: dict[str, Any],
        message: str,
    ) -> None:
        encoder = _Calls(monkeypatch, model.vision_encoder)
        with pytest.raises(ValueError, match=message):
            _decode(model, _rows(pixels, rows), **overrides)
        assert encoder.batches == []

    def test_views_of_different_shapes_are_refused(self, model: VLMWrapper) -> None:
        rows = [
            ModelInputs(torch.zeros(1, 3, 16, 16), (5,)),
            ModelInputs(torch.zeros(1, 3, 8, 8), (6,)),
        ]
        with pytest.raises(ValueError, match="views differ in shape"):
            _decode(model, rows)


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
