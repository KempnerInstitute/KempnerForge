"""CPU unit tests for the KempnerForge VLM lmms-eval adapter.

The whole module is skipped when lmms-eval is not installed (it is an optional,
undeclared dependency), so these tests run locally where lmms-eval is present
and skip cleanly in CI. They exercise the adapter's private helpers directly on
a tiny random VLM — no GPU, no real checkpoint, no network.
"""

from __future__ import annotations

import pytest
import torch
from PIL import Image

pytest.importorskip("lmms_eval")

from lmms_eval.protocol import ChatMessages  # noqa: E402

from kempnerforge.data.vlm_dataset import (  # noqa: E402
    DEFAULT_IMAGE_MEAN,
    DEFAULT_IMAGE_STD,
    pil_to_tensor,
)
from kempnerforge.eval.vlm.adapter import (  # noqa: E402
    KempnerForgeVLM,
    _check_generative_arch,
    _frames_to_pixel_values,
    _generate_one,
    _render_request,
    _resolve_gen_kwargs,
)

DEVICE = torch.device("cpu")


class _MockTokenizer:
    """Deterministic tokenizer for decode-loop tests (no HF download).

    ``decode`` renders ids as space-joined integers so stop-string and trimming
    behavior is easy to assert; ``eos_token_id`` is configurable per test.
    """

    pad_token_id = 0

    def __init__(self, eos_token_id: int | None = None) -> None:
        self.eos_token_id = eos_token_id

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        del add_special_tokens
        return {"input_ids": [(ord(c) % 254) + 1 for c in text]}

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(str(int(i)) for i in ids)


def _text(s: str) -> dict:
    return {"type": "text", "text": s}


def _image(img: object) -> dict:
    return {"type": "image", "url": img}


def _chat(content: list[dict], role: str = "user") -> ChatMessages:
    return ChatMessages(messages=[{"role": role, "content": content}])


def _img(size: int = 8) -> Image.Image:
    return Image.new("RGB", (size, size), color=(120, 120, 120))


# ---------------------------------------------------------------------------
# _render_request (message -> prompt text + media guards)
# ---------------------------------------------------------------------------


class TestRenderRequest:
    def test_flattens_text_blocks_in_order(self):
        messages = _chat([_text("Question:"), _image(_img()), _text("What color?")])
        images, prompt = _render_request(messages)
        assert len(images) == 1
        assert prompt == "Question:\nWhat color?"

    def test_video_content_raises(self):
        messages = _chat([_text("describe"), {"type": "video", "url": "clip.mp4"}])
        with pytest.raises(NotImplementedError, match="[Vv]ideo"):
            _render_request(messages)

    def test_audio_content_raises(self):
        messages = _chat([_text("listen"), {"type": "audio", "url": "a.wav"}])
        with pytest.raises(NotImplementedError, match="[Aa]udio"):
            _render_request(messages)

    def test_multi_image_raises(self):
        messages = _chat([_text("compare"), _image(_img()), _image(_img())])
        with pytest.raises(NotImplementedError, match="one image"):
            _render_request(messages)

    def test_no_image_raises(self):
        messages = _chat([_text("text only question")])
        with pytest.raises(NotImplementedError, match="one image"):
            _render_request(messages)

    def test_multi_turn_assistant_raises(self):
        messages = ChatMessages(
            messages=[
                {"role": "user", "content": [_text("q"), _image(_img())]},
                {"role": "assistant", "content": [_text("a")]},
                {"role": "user", "content": [_text("follow up")]},
            ]
        )
        with pytest.raises(NotImplementedError, match="[Mm]ulti-turn"):
            _render_request(messages)


# ---------------------------------------------------------------------------
# _frames_to_pixel_values (preprocessing glue)
# ---------------------------------------------------------------------------


class TestFramesToPixelValues:
    def test_shape_and_dtype(self):
        pv = _frames_to_pixel_values([_img()], image_size=16, device=DEVICE, dtype=torch.float32)
        assert pv.shape == (1, 3, 16, 16)
        assert pv.dtype == torch.float32
        assert pv.device.type == "cpu"

    def test_uses_image_size(self):
        pv = _frames_to_pixel_values([_img()], image_size=32, device=DEVICE, dtype=torch.float32)
        assert pv.shape == (1, 3, 32, 32)

    def test_parity_with_pil_to_tensor(self):
        """The adapter must reuse the exact training preprocessing."""
        img = _img()
        pv = _frames_to_pixel_values([img], image_size=24, device=DEVICE, dtype=torch.float32)
        expected = pil_to_tensor(img, 24, DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD)
        assert torch.allclose(pv[0], expected)


# ---------------------------------------------------------------------------
# _resolve_gen_kwargs (defaults + task overrides)
# ---------------------------------------------------------------------------


class TestResolveGenKwargs:
    def test_defaults_are_greedy(self):
        r = _resolve_gen_kwargs({}, default_max_new_tokens=64)
        assert r == {
            "until": [],
            "max_new_tokens": 64,
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
        }

    def test_task_overrides_max_new_tokens(self):
        assert _resolve_gen_kwargs({"max_new_tokens": 8}, 64)["max_new_tokens"] == 8

    def test_sampling_enabled_by_temperature(self):
        r = _resolve_gen_kwargs({"temperature": 0.7, "top_p": 0.9, "top_k": 5}, 64)
        assert (r["temperature"], r["top_p"], r["top_k"]) == (0.7, 0.9, 5)

    def test_do_sample_false_forces_greedy(self):
        r = _resolve_gen_kwargs({"temperature": 0.7, "do_sample": False}, 64)
        assert r["temperature"] == 0.0 and r["top_p"] == 1.0 and r["top_k"] == 0

    def test_until_string_normalized_to_list(self):
        assert _resolve_gen_kwargs({"until": "\n\n"}, 64)["until"] == ["\n\n"]

    def test_until_list_preserved(self):
        assert _resolve_gen_kwargs({"until": ["a", "b"]}, 64)["until"] == ["a", "b"]


# ---------------------------------------------------------------------------
# _generate_one (cache-less decode loop) on a tiny random VLM
# ---------------------------------------------------------------------------


class TestGenerateOne:
    def _pixels(self) -> torch.Tensor:
        torch.manual_seed(0)
        return torch.randn(1, 3, 16, 16)

    def _prompt(self) -> torch.Tensor:
        return torch.tensor([[5, 9, 12, 3]], dtype=torch.long)

    def test_greedy_is_deterministic(self, tiny_vlm_wrapper):
        pv, pid = self._pixels(), self._prompt()
        r = _resolve_gen_kwargs({"max_new_tokens": 6}, 128)
        out1 = _generate_one(tiny_vlm_wrapper, _MockTokenizer(), pv, pid, r, 64)
        out2 = _generate_one(tiny_vlm_wrapper, _MockTokenizer(), pv, pid, r, 64)
        assert isinstance(out1, str) and out1 == out2

    def test_respects_max_new_tokens(self, tiny_vlm_wrapper):
        r = _resolve_gen_kwargs({"max_new_tokens": 6}, 128)
        out = _generate_one(
            tiny_vlm_wrapper, _MockTokenizer(), self._pixels(), self._prompt(), r, 64
        )
        assert len(out.split()) == 6

    def test_until_trims_continuation(self, tiny_vlm_wrapper):
        pv, pid = self._pixels(), self._prompt()
        one = _generate_one(
            tiny_vlm_wrapper,
            _MockTokenizer(),
            pv,
            pid,
            _resolve_gen_kwargs({"max_new_tokens": 1}, 128),
            64,
        )
        # With decode = space-joined ids, the first space follows the first token,
        # so until=[" "] trims to exactly the first generated token.
        trimmed = _generate_one(
            tiny_vlm_wrapper,
            _MockTokenizer(),
            pv,
            pid,
            _resolve_gen_kwargs({"max_new_tokens": 6, "until": [" "]}, 128),
            64,
        )
        assert trimmed == one and " " not in trimmed

    def test_eos_stops_generation(self, tiny_vlm_wrapper):
        pv, pid = self._pixels(), self._prompt()
        first = _generate_one(
            tiny_vlm_wrapper,
            _MockTokenizer(),
            pv,
            pid,
            _resolve_gen_kwargs({"max_new_tokens": 1}, 128),
            64,
        )
        eos_id = int(first)
        out = _generate_one(
            tiny_vlm_wrapper,
            _MockTokenizer(eos_token_id=eos_id),
            pv,
            pid,
            _resolve_gen_kwargs({"max_new_tokens": 6}, 128),
            64,
        )
        assert out == ""

    def test_length_bound_raises_when_no_room(self, tiny_vlm_wrapper):
        r = _resolve_gen_kwargs({"max_new_tokens": 100}, 128)
        with pytest.raises(ValueError, match="max_new_tokens"):
            _generate_one(tiny_vlm_wrapper, _MockTokenizer(), self._pixels(), self._prompt(), r, 64)


# ---------------------------------------------------------------------------
# Guards: arch + not-implemented methods
# ---------------------------------------------------------------------------


class TestGuards:
    def test_moma_arch_rejected(self):
        with pytest.raises(ValueError, match="non-causal"):
            _check_generative_arch("moma")

    @pytest.mark.parametrize("arch", ["joint_decoder", "cross_attention", "mot"])
    def test_generative_arches_allowed(self, arch):
        _check_generative_arch(arch)  # must not raise

    def test_loglikelihood_not_implemented(self):
        inst = KempnerForgeVLM.__new__(KempnerForgeVLM)  # bypass __init__ (no checkpoint needed)
        with pytest.raises(NotImplementedError, match="loglikelihood"):
            inst.loglikelihood([])

    def test_multi_round_not_implemented(self):
        inst = KempnerForgeVLM.__new__(KempnerForgeVLM)
        with pytest.raises(NotImplementedError, match="multi-round"):
            inst.generate_until_multi_round([])
