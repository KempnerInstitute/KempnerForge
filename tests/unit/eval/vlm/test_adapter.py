"""CPU unit tests for the KempnerForge VLM lmms-eval adapter.

A faithful fake ``lmms_eval`` is injected by ``conftest.py`` (lmms-eval is an optional,
undeclared dependency), so these tests always run — in CI without lmms-eval and locally —
exercising the adapter's helpers, guards, and decode loop on a tiny random VLM: no GPU, no
real checkpoint, no network. Real-package fidelity is pinned by the gated contract test in
``tests/integration/``.
"""

from __future__ import annotations

import json

import pytest
import torch
from lmms_eval.api.instance import Instance
from lmms_eval.protocol import ChatMessages
from PIL import Image

from kempnerforge.config.data import DataConfig
from kempnerforge.config.registry import registry
from kempnerforge.config.schema import AdapterConfig, JobConfig, ModelConfig, VisionEncoderConfig
from kempnerforge.config.video import VideoConfig
from kempnerforge.config.vlm import VLMConfig
from kempnerforge.data.vlm_dataset import (
    DEFAULT_IMAGE_MEAN,
    DEFAULT_IMAGE_STD,
    pil_to_tensor,
)
from kempnerforge.eval.vlm.adapter import (
    KempnerForgeVLM,
    _build_model,
    _check_generative,
    _ContextBudgetError,
    _first_stop,
    _frames_to_pixel_values,
    _generate_batch,
    _load_config,
    _load_weights,
    _log_checkpoint_metadata,
    _render_request,
    _resolve_dtype,
    _resolve_gen_kwargs,
    _to_pil,
)
from kempnerforge.model.vlm import VLMWrapper, build_vlm_wrapper

DEVICE = torch.device("cpu")


# Arch coverage is DERIVED from the registry + the is_generative property (not a
# hardcoded list) so new arches are swept automatically: generative arches get the
# decode/generate sweeps, non-generative ones get the rejection guard. The single
# explicit per-arch truth lives in tests/unit/test_vlm_config.py::TestIsGenerative.
_ALL_VLM_ARCHS = tuple(sorted(registry.list_vlm_configs()))
GENERATIVE_ARCHES = tuple(a for a in _ALL_VLM_ARCHS if VLMConfig.for_arch(a).is_generative)
NON_GENERATIVE_ARCHES = tuple(a for a in _ALL_VLM_ARCHS if not VLMConfig.for_arch(a).is_generative)

# Per-arch BUILD sizing for a tiny CPU wrapper (sizing only, NOT generativity policy):
# CA needs a cross-attention cadence that fits the tiny layer count. Arches without an
# entry build from defaults; a future arch needing knobs fails the build loudly here.
_ARCH_BUILD_KWARGS = {"cross_attention": {"cross_attention_every_n_layers": 2}}


def _vlm_wrapper(arch: str) -> VLMWrapper:
    """Build a tiny CPU ``VLMWrapper`` for a generative arch (no checkpoint).

    Uniform ``n_layers=4`` (so CA has at least one cross-attention block) and
    ``ffn_hidden_dim=128`` (so MoT's per-modality FFN stays tiny) are valid for every
    arch; ``num_image_tokens`` (8) + ``max_text_len`` (32) fits ``max_seq_len`` (64).
    """
    mc = ModelConfig(
        dim=64, n_layers=4, n_heads=4, vocab_size=256, max_seq_len=64, ffn_hidden_dim=128
    )
    vc = VisionEncoderConfig(type="random", feature_dim=96, num_tokens=8)
    lc = VLMConfig.for_arch(arch, max_text_len=32, **_ARCH_BUILD_KWARGS.get(arch, {}))
    return build_vlm_wrapper(mc, vc, AdapterConfig(), lc).eval()


@pytest.fixture
def arch_wrapper(arch):
    """A tiny per-arch ``VLMWrapper``; ``arch`` is provided by ``@pytest.mark.parametrize``."""
    return _vlm_wrapper(arch)


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


class _RecordingLogger:
    """Captures log calls so metadata/warning behavior is asserted without caplog."""

    def __init__(self) -> None:
        self.infos: list[str] = []
        self.warnings: list[str] = []

    def info(self, msg: str) -> None:
        self.infos.append(msg)

    def warning(self, msg: str) -> None:
        self.warnings.append(msg)


def _text(s: str) -> dict:
    return {"type": "text", "text": s}


def _image(img: object) -> dict:
    return {"type": "image", "url": img}


def _video(url: object) -> dict:
    return {"type": "video", "url": url}


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
        images, prompt = _render_request(messages, None)
        assert len(images) == 1
        assert prompt == "Question:\nWhat color?"

    def test_video_content_raises(self):
        messages = _chat([_text("describe"), {"type": "video", "url": "clip.mp4"}])
        with pytest.raises(NotImplementedError, match="[Vv]ideo"):
            _render_request(messages, None)

    def test_audio_content_raises(self):
        messages = _chat([_text("listen"), {"type": "audio", "url": "a.wav"}])
        with pytest.raises(NotImplementedError, match="[Aa]udio"):
            _render_request(messages, None)

    def test_multi_image_raises(self):
        messages = _chat([_text("compare"), _image(_img()), _image(_img())])
        with pytest.raises(NotImplementedError, match="one image"):
            _render_request(messages, None)

    def test_no_image_raises(self):
        messages = _chat([_text("text only question")])
        with pytest.raises(NotImplementedError, match="one image"):
            _render_request(messages, None)

    def test_multi_turn_assistant_raises(self):
        messages = ChatMessages(
            messages=[
                {"role": "user", "content": [_text("q"), _image(_img())]},
                {"role": "assistant", "content": [_text("a")]},
                {"role": "user", "content": [_text("follow up")]},
            ]
        )
        with pytest.raises(NotImplementedError, match="[Mm]ulti-turn"):
            _render_request(messages, None)


class TestRenderRequestVideo:
    """Video-checkpoint rendering (``video_config`` is not None). Decode is stubbed."""

    @pytest.fixture
    def vcfg(self) -> VideoConfig:
        return VideoConfig(max_frames=4, min_frames=1, frame_size=16)

    def test_video_decoded_to_frames(self, monkeypatch, vcfg):
        frames = [_img(), _img(), _img()]
        monkeypatch.setattr(
            "kempnerforge.eval.vlm.adapter.decode_video_frames", lambda path, **kw: frames
        )
        out_frames, prompt = _render_request(_chat([_text("describe"), _video("clip.mp4")]), vcfg)
        assert out_frames is frames
        assert prompt == "describe"

    def test_decode_uses_config_policy(self, monkeypatch, vcfg):
        captured: dict = {}

        def _fake(path, **kw):
            captured["path"] = path
            captured.update(kw)
            return [_img()]

        monkeypatch.setattr("kempnerforge.eval.vlm.adapter.decode_video_frames", _fake)
        _render_request(_chat([_video("c.mp4")]), vcfg)
        assert captured["path"] == "c.mp4"
        assert captured["fps"] == vcfg.fps
        assert captured["min_frames"] == vcfg.min_frames
        assert captured["max_frames"] == vcfg.max_frames
        assert captured["sampling_policy"] == vcfg.sampling_policy

    def test_multiple_videos_raise(self, vcfg):
        with pytest.raises(NotImplementedError, match="[Mm]ultiple videos"):
            _render_request(_chat([_video("a.mp4"), _video("b.mp4")]), vcfg)

    def test_non_path_video_raises(self, vcfg):
        msg = _chat([_video({"video": "x", "start": 0.0})])
        with pytest.raises(NotImplementedError, match="path string"):
            _render_request(msg, vcfg)

    def test_mixed_image_and_video_raise(self, vcfg):
        with pytest.raises(NotImplementedError, match="[Mm]ixed"):
            _render_request(_chat([_image(_img()), _video("c.mp4")]), vcfg)

    def test_single_image_is_one_frame_clip(self, vcfg):
        frames, prompt = _render_request(_chat([_text("q"), _image(_img())]), vcfg)
        assert len(frames) == 1
        assert prompt == "q"

    def test_audio_still_raises(self, vcfg):
        msg = _chat([_text("x"), {"type": "audio", "url": "a.wav"}])
        with pytest.raises(NotImplementedError, match="[Aa]udio"):
            _render_request(msg, vcfg)

    def test_multi_image_raises(self, vcfg):
        with pytest.raises(NotImplementedError, match="exactly one"):
            _render_request(_chat([_image(_img()), _image(_img())]), vcfg)

    def test_multi_turn_raises(self, vcfg):
        msg = ChatMessages(
            messages=[
                {"role": "user", "content": [_text("q"), _video("c.mp4")]},
                {"role": "assistant", "content": [_text("a")]},
                {"role": "user", "content": [_text("again")]},
            ]
        )
        with pytest.raises(NotImplementedError, match="[Mm]ulti-turn"):
            _render_request(msg, vcfg)

    def test_no_frames_decoded_warns(self, monkeypatch, vcfg):
        rec = _RecordingLogger()
        monkeypatch.setattr("kempnerforge.eval.vlm.adapter.logger", rec)
        monkeypatch.setattr(
            "kempnerforge.eval.vlm.adapter.decode_video_frames", lambda path, **kw: []
        )
        frames, _ = _render_request(_chat([_video("c.mp4")]), vcfg)
        assert frames == []
        assert any("zero clip" in m for m in rec.warnings)


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

    def test_accepts_path_string(self, tmp_path):
        """A frame given as a path string is opened and preprocessed identically."""
        img = _img(16)
        path = tmp_path / "frame.png"
        img.save(path)
        from_path = _frames_to_pixel_values(
            [str(path)], image_size=16, device=DEVICE, dtype=torch.float32
        )
        from_pil = _frames_to_pixel_values([img], image_size=16, device=DEVICE, dtype=torch.float32)
        assert from_path.shape == (1, 3, 16, 16)
        assert torch.allclose(from_path, from_pil)


# ---------------------------------------------------------------------------
# _to_pil (str/path -> PIL normalization) + video str-image handling
# ---------------------------------------------------------------------------


class TestToPil:
    def test_passthrough_pil(self):
        img = _img()
        assert _to_pil(img) is img

    def test_opens_str_path(self, tmp_path):
        from PIL import Image

        path = tmp_path / "frame.png"
        _img(16).save(path)
        out = _to_pil(str(path))
        assert isinstance(out, Image.Image)
        assert out.size == (16, 16)


def test_video_checkpoint_accepts_str_image_path(
    tmp_path, monkeypatch, tiny_video_configs, tiny_video_vlm_wrapper
):
    """A single image supplied as a path string runs on a video checkpoint: it is
    opened to PIL before frames_to_clip_tensor (strict PIL-only), so it is processed
    rather than silently skipped by the per-request fault handler."""
    _patch_loaders(monkeypatch, _video_job_config(tiny_video_configs), tiny_video_vlm_wrapper)
    vlm = KempnerForgeVLM(config="x", checkpoint="y", device="cpu", dtype="float32")
    path = tmp_path / "frame.png"
    _img(16).save(path)

    def doc_to_messages(doc):
        del doc
        return [{"role": "user", "content": [_text("describe"), _image(str(path))]}]

    vlm.task_dict = {"t": {"test": {"d0": {}}}}
    inst = Instance(
        request_type="generate_until",
        arguments=("ctx", doc_to_messages, {"max_new_tokens": 3}, "d0", "t", "test"),
        idx=0,
        metadata={"task": "t", "doc_id": "d0", "repeats": 1},
    )
    out = vlm.generate_until([inst])
    assert len(out) == 1 and len(out[0].split()) == 3  # processed, not skipped


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

    def test_explicit_zero_max_new_tokens_honored(self):
        # 0 is a valid explicit value, not a "missing" fallback.
        assert _resolve_gen_kwargs({"max_new_tokens": 0}, 64)["max_new_tokens"] == 0

    def test_explicit_zero_top_p_honored_when_sampling(self):
        # top_p only applies when sampling; an explicit 0.0 must not fall back to 1.0.
        r = _resolve_gen_kwargs({"temperature": 0.5, "top_p": 0.0}, 64)
        assert r["top_p"] == 0.0


# ---------------------------------------------------------------------------
# _generate_batch (cache-less batched decode loop) on a tiny random VLM
# ---------------------------------------------------------------------------


def _pixels(batch: int = 1) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(batch, 3, 16, 16)


@pytest.mark.parametrize("arch", GENERATIVE_ARCHES)
class TestGenerateBatchSingle:
    """B == 1 must reproduce the v1 single-request behavior (every generative arch)."""

    def _prompt(self) -> list[torch.Tensor]:
        return [torch.tensor([5, 9, 12, 3], dtype=torch.long)]

    def test_greedy_is_deterministic(self, arch_wrapper):
        pv, pid = _pixels(), self._prompt()
        r = _resolve_gen_kwargs({"max_new_tokens": 6}, 128)
        out1 = _generate_batch(arch_wrapper, _MockTokenizer(), pv, pid, r, 64)
        out2 = _generate_batch(arch_wrapper, _MockTokenizer(), pv, pid, r, 64)
        assert out1 == out2 and len(out1) == 1 and isinstance(out1[0], str)

    def test_respects_max_new_tokens(self, arch_wrapper):
        r = _resolve_gen_kwargs({"max_new_tokens": 6}, 128)
        out = _generate_batch(arch_wrapper, _MockTokenizer(), _pixels(), self._prompt(), r, 64)
        assert len(out[0].split()) == 6

    def test_until_trims_continuation(self, arch_wrapper):
        pv, pid = _pixels(), self._prompt()
        one = _generate_batch(
            arch_wrapper,
            _MockTokenizer(),
            pv,
            pid,
            _resolve_gen_kwargs({"max_new_tokens": 1}, 128),
            64,
        )[0]
        # decode = space-joined ids, so the first space follows the first token:
        # until=[" "] trims to exactly the first generated token.
        trimmed = _generate_batch(
            arch_wrapper,
            _MockTokenizer(),
            pv,
            pid,
            _resolve_gen_kwargs({"max_new_tokens": 6, "until": [" "]}, 128),
            64,
        )[0]
        assert trimmed == one and " " not in trimmed

    def test_eos_stops_generation(self, arch_wrapper):
        pv, pid = _pixels(), self._prompt()
        first = _generate_batch(
            arch_wrapper,
            _MockTokenizer(),
            pv,
            pid,
            _resolve_gen_kwargs({"max_new_tokens": 1}, 128),
            64,
        )[0]
        out = _generate_batch(
            arch_wrapper,
            _MockTokenizer(eos_token_id=int(first)),
            pv,
            pid,
            _resolve_gen_kwargs({"max_new_tokens": 6}, 128),
            64,
        )
        assert out == [""]

    def test_no_room_raises_context_budget_error(self, arch_wrapper):
        # _generate_batch still guards standalone; generate_until converts this into a
        # per-task skip (see TestGenerateUntilFaultTolerance).
        r = _resolve_gen_kwargs({"max_new_tokens": 100}, 128)
        with pytest.raises(_ContextBudgetError, match="max_new_tokens"):
            _generate_batch(arch_wrapper, _MockTokenizer(), _pixels(), self._prompt(), r, 64)

    def test_overlong_prompt_is_left_truncated(self, arch_wrapper, monkeypatch):
        """A prompt that exceeds the budget (but leaves room) is left-truncated with a warning."""
        rec = _RecordingLogger()
        monkeypatch.setattr("kempnerforge.eval.vlm.adapter.logger", rec)
        # Size the prompt from the wrapper's own num_image_tokens (0 for CA, 8 for JD/MoT)
        # so it overflows the budget by 6 on every arch — no hardcoded image-token count.
        # budget = max_seq_len(64) - num_image_tokens - max_new_tokens(2).
        max_new = 2
        budget = 64 - arch_wrapper.num_image_tokens - max_new
        long_prompt = [torch.arange(1, budget + 7, dtype=torch.long)]
        r = _resolve_gen_kwargs({"max_new_tokens": max_new}, 128)
        out = _generate_batch(arch_wrapper, _MockTokenizer(), _pixels(), long_prompt, r, 64)
        assert len(out) == 1 and len(out[0].split()) == 2
        assert any("left-truncating" in m for m in rec.warnings)


@pytest.mark.parametrize("arch", GENERATIVE_ARCHES)
class TestGenerateBatchMulti:
    """B > 1: right-padding must not change any row's result, and stop
    conditions must be tracked per row (every generative arch)."""

    def _prompts(self) -> list[torch.Tensor]:
        # Deliberately different lengths so right-padding is exercised.
        return [
            torch.tensor([5, 9, 12, 3], dtype=torch.long),
            torch.tensor([7, 2], dtype=torch.long),
            torch.tensor([1, 4, 8, 11, 20, 6], dtype=torch.long),
        ]

    def test_batch_equals_sequential(self, arch_wrapper):
        """Key correctness gate: a right-padded batch yields the same per-row
        continuation as decoding each request alone (greedy, float32)."""
        prompts = self._prompts()
        pv = _pixels(len(prompts))  # (3, 3, 16, 16) — one image per request
        r = _resolve_gen_kwargs({"max_new_tokens": 5}, 128)
        sequential = [
            _generate_batch(arch_wrapper, _MockTokenizer(), pv[i : i + 1], [prompts[i]], r, 64)[0]
            for i in range(len(prompts))
        ]
        batched = _generate_batch(arch_wrapper, _MockTokenizer(), pv, prompts, r, 64)
        assert batched == sequential

    def test_per_row_max_new_tokens(self, arch_wrapper):
        prompts = self._prompts()
        pv = _pixels(len(prompts))
        r = _resolve_gen_kwargs({"max_new_tokens": 4}, 128)
        outs = _generate_batch(arch_wrapper, _MockTokenizer(), pv, prompts, r, 64)
        assert len(outs) == 3 and all(len(o.split()) == 4 for o in outs)

    def test_per_row_eos_independent(self, arch_wrapper):
        """EOS on one row stops only that row; the batch still returns all rows."""
        prompts = self._prompts()
        pv = _pixels(len(prompts))
        first0 = _generate_batch(
            arch_wrapper,
            _MockTokenizer(),
            pv[:1],
            [prompts[0]],
            _resolve_gen_kwargs({"max_new_tokens": 1}, 128),
            64,
        )[0]
        outs = _generate_batch(
            arch_wrapper,
            _MockTokenizer(eos_token_id=int(first0)),
            pv,
            prompts,
            _resolve_gen_kwargs({"max_new_tokens": 5}, 128),
            64,
        )
        assert len(outs) == 3 and outs[0] == ""  # row 0 stops immediately on EOS


# ---------------------------------------------------------------------------
# _generate_batch frame_mask threading (video padded-frame masking)
# ---------------------------------------------------------------------------


class _CaptureModel:
    """Minimal ``VLMWrapper`` stand-in: records the ``frame_mask`` each forward
    receives and returns deterministic zero logits (greedy -> token 0), so the
    decode-loop plumbing can be asserted without a real transformer."""

    def __init__(self, num_image_tokens: int, vocab_size: int = 256) -> None:
        self.num_image_tokens = num_image_tokens
        self._vocab = vocab_size
        self.seen_frame_masks: list[torch.Tensor | None] = []

    def __call__(self, pixel_values, input_ids, frame_mask=None):
        del pixel_values
        self.seen_frame_masks.append(frame_mask)
        b, t = input_ids.shape
        return torch.zeros(b, t, self._vocab), None


def test_generate_batch_threads_frame_mask_for_video():
    """A video batch passes the same (B, F) frame_mask into every model forward, so
    padded-frame visual tokens are masked from attention exactly as in training."""
    model = _CaptureModel(num_image_tokens=16)
    pixel_values = torch.randn(2, 2, 3, 16, 16)  # (B, F=2, 3, H, W)
    frame_mask = torch.tensor([[True, True], [True, False]])  # row 1: frame 2 padded
    prompt_ids = [torch.tensor([5, 9], dtype=torch.long), torch.tensor([7], dtype=torch.long)]
    r = _resolve_gen_kwargs({"max_new_tokens": 3}, 128)
    _generate_batch(model, _MockTokenizer(), pixel_values, prompt_ids, r, 64, frame_mask=frame_mask)
    assert model.seen_frame_masks  # decode actually ran
    assert all(fm is frame_mask for fm in model.seen_frame_masks)


def test_generate_batch_no_frame_mask_for_image():
    """An image batch forwards frame_mask=None so the model keeps its unmasked path."""
    model = _CaptureModel(num_image_tokens=8)
    pixel_values = torch.randn(1, 3, 16, 16)
    prompt_ids = [torch.tensor([5, 9, 12], dtype=torch.long)]
    r = _resolve_gen_kwargs({"max_new_tokens": 2}, 128)
    _generate_batch(model, _MockTokenizer(), pixel_values, prompt_ids, r, 64)
    assert model.seen_frame_masks and all(fm is None for fm in model.seen_frame_masks)


# ---------------------------------------------------------------------------
# Guards: arch + not-implemented methods
# ---------------------------------------------------------------------------


class TestGuards:
    @pytest.mark.parametrize("arch", NON_GENERATIVE_ARCHES)
    def test_non_generative_arches_rejected(self, arch):
        with pytest.raises(ValueError, match="non-causal"):
            _check_generative(VLMConfig.for_arch(arch))

    @pytest.mark.parametrize("arch", GENERATIVE_ARCHES)
    def test_generative_arches_allowed(self, arch):
        _check_generative(VLMConfig.for_arch(arch))  # must not raise

    def test_loglikelihood_not_implemented(self):
        inst = KempnerForgeVLM.__new__(KempnerForgeVLM)  # bypass __init__ (no checkpoint needed)
        with pytest.raises(NotImplementedError, match="loglikelihood"):
            inst.loglikelihood([])

    def test_multi_round_not_implemented(self):
        inst = KempnerForgeVLM.__new__(KempnerForgeVLM)
        with pytest.raises(NotImplementedError, match="multi-round"):
            inst.generate_until_multi_round([])


# ---------------------------------------------------------------------------
# Loader / __init__ helpers (build a VLM JobConfig + patch the heavy loaders so
# the adapter can be constructed on CPU with no checkpoint).
# ---------------------------------------------------------------------------


def _vlm_job_config(tiny_vlm_configs, arch: str | None = None) -> JobConfig:
    mc, vc, ac, lc = tiny_vlm_configs
    if arch is not None:
        # Build the real arch subclass (not a base VLMConfig with a mutated
        # .arch) so per-arch config policy like is_generative is exercised.
        lc = VLMConfig.for_arch(arch, max_text_len=lc.max_text_len)
    return JobConfig(
        model=mc, vision_encoder=vc, adapter=ac, vlm=lc, data=DataConfig(tokenizer_path="mock")
    )


def _video_job_config(tiny_video_configs) -> JobConfig:
    mc, vc, ac, lc, video = tiny_video_configs
    return JobConfig(
        model=mc,
        vision_encoder=vc,
        adapter=ac,
        vlm=lc,
        video=video,
        data=DataConfig(tokenizer_path="mock"),
    )


def _patch_loaders(monkeypatch, job: JobConfig, model) -> None:
    monkeypatch.setattr("kempnerforge.eval.vlm.adapter._load_config", lambda _p: job)
    monkeypatch.setattr("kempnerforge.eval.vlm.adapter._load_weights", lambda *a, **k: model)
    monkeypatch.setattr(
        "kempnerforge.eval.vlm.adapter.build_tokenizer", lambda _p: _MockTokenizer()
    )


# ---------------------------------------------------------------------------
# _resolve_dtype
# ---------------------------------------------------------------------------


class TestResolveDtype:
    @pytest.mark.parametrize(
        "name,expected",
        [("bfloat16", torch.bfloat16), ("float16", torch.float16), ("float32", torch.float32)],
    )
    def test_string_dtypes(self, name, expected):
        assert _resolve_dtype(name) == expected

    def test_passthrough_torch_dtype(self):
        assert _resolve_dtype(torch.float64) == torch.float64

    def test_unsupported_dtype_raises(self):
        with pytest.raises(ValueError, match="Unsupported dtype"):
            _resolve_dtype("float64")


# ---------------------------------------------------------------------------
# _first_stop
# ---------------------------------------------------------------------------


class TestFirstStop:
    def test_no_match_returns_none(self):
        assert _first_stop("hello world", ["xyz"]) is None

    def test_empty_until_returns_none(self):
        assert _first_stop("abc", []) is None

    def test_earliest_of_multiple(self):
        # "cd" at index 2 precedes "ef" at index 4.
        assert _first_stop("abcdef", ["ef", "cd"]) == 2

    def test_single_match_index(self):
        assert _first_stop("a.b", ["."]) == 1


# ---------------------------------------------------------------------------
# _log_checkpoint_metadata
# ---------------------------------------------------------------------------


class TestLogCheckpointMetadata:
    def test_missing_metadata_is_noop(self, tmp_path, monkeypatch):
        rec = _RecordingLogger()
        monkeypatch.setattr("kempnerforge.eval.vlm.adapter.logger", rec)
        _log_checkpoint_metadata(tmp_path)
        assert rec.infos == [] and rec.warnings == []

    def test_valid_metadata_logged(self, tmp_path, monkeypatch):
        rec = _RecordingLogger()
        monkeypatch.setattr("kempnerforge.eval.vlm.adapter.logger", rec)
        (tmp_path / "metadata.json").write_text(json.dumps({"step": 7, "tokens_seen": 1234}))
        _log_checkpoint_metadata(tmp_path)
        assert any("step=7" in m and "tokens_seen=1234" in m for m in rec.infos)

    def test_malformed_metadata_warns_not_raises(self, tmp_path, monkeypatch):
        rec = _RecordingLogger()
        monkeypatch.setattr("kempnerforge.eval.vlm.adapter.logger", rec)
        (tmp_path / "metadata.json").write_text("{not valid json")
        _log_checkpoint_metadata(tmp_path)  # must not raise
        assert any("Could not read" in m for m in rec.warnings)


# ---------------------------------------------------------------------------
# _load_config (VLM-only guard)
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_non_vlm_config_rejected(self, monkeypatch, tiny_job_config):
        monkeypatch.setattr(
            "kempnerforge.eval.vlm.adapter.load_config",
            lambda _p, cli_args=None: tiny_job_config,
        )
        with pytest.raises(ValueError, match="not a VLM config"):
            _load_config("ignored.toml")

    def test_vlm_config_accepted(self, monkeypatch, tiny_vlm_configs):
        job = _vlm_job_config(tiny_vlm_configs)
        monkeypatch.setattr(
            "kempnerforge.eval.vlm.adapter.load_config", lambda _p, cli_args=None: job
        )
        assert _load_config("ignored.toml") is job


# ---------------------------------------------------------------------------
# _load_weights (path resolution / missing checkpoint)
# ---------------------------------------------------------------------------


class TestLoadWeights:
    def test_missing_checkpoint_raises(self, tmp_path, tiny_vlm_configs):
        job = _vlm_job_config(tiny_vlm_configs)
        missing = tmp_path / "no_such_checkpoint"
        with pytest.raises(FileNotFoundError, match="does not exist"):
            _load_weights(job, str(missing), torch.device("cpu"), torch.float32)


# ---------------------------------------------------------------------------
# KempnerForgeVLM.__init__ guards
# ---------------------------------------------------------------------------


class TestInitGuards:
    @pytest.mark.parametrize("arch", NON_GENERATIVE_ARCHES)
    def test_non_generative_fails_fast_before_load(self, monkeypatch, tiny_vlm_configs, arch):
        job = _vlm_job_config(tiny_vlm_configs, arch=arch)
        monkeypatch.setattr("kempnerforge.eval.vlm.adapter._load_config", lambda _p: job)

        def _must_not_load(*args, **kwargs):
            raise AssertionError("model load must not run for an unsupported arch")

        monkeypatch.setattr("kempnerforge.eval.vlm.adapter._load_weights", _must_not_load)
        with pytest.raises(ValueError, match="non-causal"):
            KempnerForgeVLM(config="x", checkpoint="y", device="cpu", dtype="float32")

    def test_ignored_kwargs_warn(self, monkeypatch, tiny_vlm_configs, tiny_vlm_wrapper):
        rec = _RecordingLogger()
        monkeypatch.setattr("kempnerforge.eval.vlm.adapter.logger", rec)
        _patch_loaders(monkeypatch, _vlm_job_config(tiny_vlm_configs), tiny_vlm_wrapper)
        KempnerForgeVLM(config="x", checkpoint="y", device="cpu", dtype="float32", bogus=1)
        assert any("bogus" in m for m in rec.warnings)

    @pytest.mark.parametrize("arch", GENERATIVE_ARCHES)
    def test_init_populates_attrs(self, monkeypatch, tiny_vlm_configs, arch):
        job = _vlm_job_config(tiny_vlm_configs, arch=arch)
        _patch_loaders(monkeypatch, job, _vlm_wrapper(arch))
        vlm = KempnerForgeVLM(
            config="x", checkpoint="y", device="cpu", dtype="float32", batch_size=2
        )
        assert vlm._arch == arch
        assert vlm._max_seq_len == job.model.max_seq_len
        assert vlm._batch_size == 2
        assert vlm._dtype == torch.float32
        assert vlm._is_video is False
        assert vlm._frames_per_clip == 1
        assert vlm._frame_size == job.data.hf_image_size

    def test_dtype_defaults_from_config(self, monkeypatch, tiny_vlm_configs, tiny_vlm_wrapper):
        # No explicit dtype -> use the checkpoint config's train.param_dtype.
        from kempnerforge.config.training import TrainConfig

        mc, vc, ac, lc = tiny_vlm_configs
        job = JobConfig(
            model=mc,
            vision_encoder=vc,
            adapter=ac,
            vlm=lc,
            data=DataConfig(tokenizer_path="mock"),
            train=TrainConfig(mixed_precision="fp32"),
        )
        _patch_loaders(monkeypatch, job, tiny_vlm_wrapper)
        vlm = KempnerForgeVLM(config="x", checkpoint="y", device="cpu")
        assert vlm._dtype == job.train.param_dtype == torch.float32

    def test_explicit_dtype_overrides_config(self, monkeypatch, tiny_vlm_configs, tiny_vlm_wrapper):
        # An explicit dtype wins over the config default.
        _patch_loaders(monkeypatch, _vlm_job_config(tiny_vlm_configs), tiny_vlm_wrapper)
        vlm = KempnerForgeVLM(config="x", checkpoint="y", device="cpu", dtype="float16")
        assert vlm._dtype == torch.float16


# ---------------------------------------------------------------------------
# _build_model (frames_per_clip wiring for image vs video checkpoints)
# ---------------------------------------------------------------------------


class TestBuildModel:
    def test_video_config_sets_frames_per_clip(self, tiny_video_configs):
        job = _video_job_config(tiny_video_configs)
        model = _build_model(job, torch.device("cpu"), torch.float32)
        assert job.video is not None
        assert model.frames_per_clip == job.video.max_frames == 2

    def test_image_config_frames_per_clip_is_one(self, tiny_vlm_configs):
        job = _vlm_job_config(tiny_vlm_configs)
        model = _build_model(job, torch.device("cpu"), torch.float32)
        assert model.frames_per_clip == 1


# ---------------------------------------------------------------------------
# generate_until (end-to-end on a tiny VLM via the fake lmms-eval objects)
# ---------------------------------------------------------------------------


class TestGenerateUntil:
    @pytest.mark.parametrize("arch", GENERATIVE_ARCHES)
    def test_batches_and_restores_order(self, monkeypatch, tiny_vlm_configs, arch):
        _patch_loaders(
            monkeypatch, _vlm_job_config(tiny_vlm_configs, arch=arch), _vlm_wrapper(arch)
        )
        vlm = KempnerForgeVLM(
            config="x", checkpoint="y", device="cpu", dtype="float32", batch_size=2
        )

        img = _img()

        def doc_to_messages(doc):
            return [{"role": "user", "content": [_text(doc["q"]), _image(img)]}]

        vlm.task_dict = {
            "t": {"test": {"d0": {"q": "hi?"}, "d1": {"q": "describe the picture please"}}}
        }
        # Different context strings (args[0]) so the Collator reorders the batch by length;
        # get_original must then restore the original request order.
        specs = [("d0", "c"), ("d1", "cc")]
        instances = [
            Instance(
                request_type="generate_until",
                arguments=(ctx, doc_to_messages, {"max_new_tokens": 3}, doc_id, "t", "test"),
                idx=i,
                metadata={"task": "t", "doc_id": doc_id, "repeats": 1},
            )
            for i, (doc_id, ctx) in enumerate(specs)
        ]

        # Each request decoded alone (greedy → deterministic), in original order.
        singles = [vlm.generate_until([inst])[0] for inst in instances]
        batched = vlm.generate_until(instances)

        assert batched == singles  # batching + reorder + get_original preserve per-request results
        assert all(isinstance(o, str) for o in batched)
        assert all(len(o.split()) == 3 for o in batched)  # greedy emits exactly max_new_tokens


class TestGenerateUntilVideo:
    """End-to-end video decode: stack -> (B, F, 3, H, W) -> forward -> strings."""

    def test_video_batches_and_restores_order(
        self, monkeypatch, tiny_video_configs, tiny_video_vlm_wrapper
    ):
        job = _video_job_config(tiny_video_configs)
        _patch_loaders(monkeypatch, job, tiny_video_vlm_wrapper)
        # Two real frames per clip; zero-padded to frames_per_clip (== 2) downstream.
        monkeypatch.setattr(
            "kempnerforge.eval.vlm.adapter.decode_video_frames",
            lambda path, **kw: [_img(), _img()],
        )
        vlm = KempnerForgeVLM(
            config="x", checkpoint="y", device="cpu", dtype="float32", batch_size=2
        )
        assert vlm._is_video is True
        assert vlm._frames_per_clip == 2

        def doc_to_messages(doc):
            return [{"role": "user", "content": [_text(doc["q"]), _video(doc["v"])]}]

        vlm.task_dict = {
            "t": {
                "test": {
                    "d0": {"q": "what?", "v": "a.mp4"},
                    "d1": {"q": "describe it", "v": "b.mp4"},
                }
            }
        }
        specs = [("d0", "c"), ("d1", "cc")]
        instances = [
            Instance(
                request_type="generate_until",
                arguments=(ctx, doc_to_messages, {"max_new_tokens": 3}, doc_id, "t", "test"),
                idx=i,
                metadata={"task": "t", "doc_id": doc_id, "repeats": 1},
            )
            for i, (doc_id, ctx) in enumerate(specs)
        ]
        # Decoding each request alone must match the batched (stacked, 5-D) result; if the
        # adapter folded frames with cat, the forward would trip the frames-per-clip check.
        singles = [vlm.generate_until([inst])[0] for inst in instances]
        batched = vlm.generate_until(instances)
        assert batched == singles
        assert all(isinstance(o, str) and len(o.split()) == 3 for o in batched)

    def test_video_pixel_values_assembled_5d(self):
        """The video batch is (B, F, 3, H, W) via stack, not (B*F, 3, H, W)."""
        from kempnerforge.data.vlm_dataset import frames_to_clip_tensor

        clip0, _ = frames_to_clip_tensor([_img(), _img()], max_frames=2, frame_size=16)
        clip1, _ = frames_to_clip_tensor([_img()], max_frames=2, frame_size=16)
        pixel_values = torch.stack([clip0, clip1], dim=0)
        assert pixel_values.shape == (2, 2, 3, 16, 16)
        assert pixel_values.ndim == 5


# ---------------------------------------------------------------------------
# generate_until fault tolerance + empty-prompt guard
# ---------------------------------------------------------------------------


class TestGenerateUntilFaultTolerance:
    """A request that fails to render/preprocess is isolated (empty output + a
    warning) so the rest of the batch still scores; an empty flattened prompt is
    guarded rather than crashing the decode."""

    def _vlm(self, monkeypatch, tiny_vlm_configs, batch_size=2):
        _patch_loaders(
            monkeypatch, _vlm_job_config(tiny_vlm_configs), _vlm_wrapper("joint_decoder")
        )
        return KempnerForgeVLM(
            config="x", checkpoint="y", device="cpu", dtype="float32", batch_size=batch_size
        )

    def test_bad_request_skipped_others_complete(self, monkeypatch, tiny_vlm_configs):
        rec = _RecordingLogger()
        monkeypatch.setattr("kempnerforge.eval.vlm.adapter.logger", rec)
        vlm = self._vlm(monkeypatch, tiny_vlm_configs)
        img = _img()

        def doc_to_messages(doc):
            # The "bad" doc carries two images (multi-image -> NotImplementedError in
            # _render_request); the "good" doc carries exactly one.
            content = [_text(doc["q"])] + [_image(img)] * doc["n_images"]
            return [{"role": "user", "content": content}]

        vlm.task_dict = {
            "t": {
                "test": {
                    "bad": {"q": "compare", "n_images": 2},
                    "good": {"q": "describe", "n_images": 1},
                }
            }
        }
        instances = [
            Instance(
                request_type="generate_until",
                arguments=(ctx, doc_to_messages, {"max_new_tokens": 3}, doc_id, "t", "test"),
                idx=i,
                metadata={"task": "t", "doc_id": doc_id, "repeats": 1},
            )
            for i, (doc_id, ctx) in enumerate([("bad", "c"), ("good", "cc")])
        ]
        outs = vlm.generate_until(instances)
        assert outs[0] == ""  # bad request isolated (order restored by get_original)
        assert len(outs[1].split()) == 3  # good request completes normally
        assert any("Skipping request" in m and "doc_id=bad" in m for m in rec.warnings)

    def test_empty_prompt_is_guarded(self, monkeypatch, tiny_vlm_configs):
        rec = _RecordingLogger()
        monkeypatch.setattr("kempnerforge.eval.vlm.adapter.logger", rec)
        vlm = self._vlm(monkeypatch, tiny_vlm_configs, batch_size=1)

        def doc_to_messages(doc):
            del doc
            # One image, no text block -> empty flattened prompt.
            return [{"role": "user", "content": [_image(_img())]}]

        vlm.task_dict = {"t": {"test": {"d0": {}}}}
        inst = Instance(
            request_type="generate_until",
            arguments=("ctx", doc_to_messages, {"max_new_tokens": 3}, "d0", "t", "test"),
            idx=0,
            metadata={"task": "t", "doc_id": "d0", "repeats": 1},
        )
        assert vlm.generate_until([inst]) == [""]
        assert any("empty prompt" in m for m in rec.warnings)

    def test_all_requests_bad_returns_empties(self, monkeypatch, tiny_vlm_configs):
        # Every request in the chunk fails -> generation is short-circuited (no empty
        # stack/cat) and each slot returns "", order preserved.
        monkeypatch.setattr("kempnerforge.eval.vlm.adapter.logger", _RecordingLogger())
        vlm = self._vlm(monkeypatch, tiny_vlm_configs)

        def doc_to_messages(doc):
            del doc
            return [{"role": "user", "content": [_text("x"), _image(_img()), _image(_img())]}]

        vlm.task_dict = {"t": {"test": {"d0": {}, "d1": {}}}}
        instances = [
            Instance(
                request_type="generate_until",
                arguments=(ctx, doc_to_messages, {"max_new_tokens": 3}, doc_id, "t", "test"),
                idx=i,
                metadata={"task": "t", "doc_id": doc_id, "repeats": 1},
            )
            for i, (doc_id, ctx) in enumerate([("d0", "c"), ("d1", "cc")])
        ]
        assert vlm.generate_until(instances) == ["", ""]

    def test_infra_error_propagates(self, monkeypatch, tiny_vlm_configs):
        # A non-per-document error (version drift, CUDA OOM, ...) must surface rather
        # than being silently scored as "".
        vlm = self._vlm(monkeypatch, tiny_vlm_configs, batch_size=1)

        def boom(*_a, **_k):
            raise RuntimeError("boom")

        monkeypatch.setattr("kempnerforge.eval.vlm.adapter._render_request", boom)

        def doc_to_messages(doc):
            del doc
            return [{"role": "user", "content": [_text("hi"), _image(_img())]}]

        vlm.task_dict = {"t": {"test": {"d0": {}}}}
        inst = Instance(
            request_type="generate_until",
            arguments=("ctx", doc_to_messages, {"max_new_tokens": 3}, "d0", "t", "test"),
            idx=0,
            metadata={"task": "t", "doc_id": "d0", "repeats": 1},
        )
        with pytest.raises(RuntimeError, match="boom"):
            vlm.generate_until([inst])

    def test_missing_image_path_skipped(self, monkeypatch, tiny_vlm_configs):
        # A bad image path raises FileNotFoundError (subclass of OSError) inside _to_pil;
        # it is isolated as "" like other per-document failures.
        rec = _RecordingLogger()
        monkeypatch.setattr("kempnerforge.eval.vlm.adapter.logger", rec)
        vlm = self._vlm(monkeypatch, tiny_vlm_configs, batch_size=1)

        def doc_to_messages(doc):
            del doc
            return [{"role": "user", "content": [_text("describe"), _image("/no/such/frame.png")]}]

        vlm.task_dict = {"t": {"test": {"d0": {}}}}
        inst = Instance(
            request_type="generate_until",
            arguments=("ctx", doc_to_messages, {"max_new_tokens": 3}, "d0", "t", "test"),
            idx=0,
            metadata={"task": "t", "doc_id": "d0", "repeats": 1},
        )
        assert vlm.generate_until([inst]) == [""]
        assert any("Skipping request" in m and "doc_id=d0" in m for m in rec.warnings)

    def test_over_budget_chunk_skipped_not_aborted(self, monkeypatch, tiny_vlm_configs):
        # A task whose max_new_tokens over-budgets the context skips its own requests
        # (via _ContextBudgetError) instead of aborting the whole run.
        rec = _RecordingLogger()
        monkeypatch.setattr("kempnerforge.eval.vlm.adapter.logger", rec)
        vlm = self._vlm(monkeypatch, tiny_vlm_configs)  # tiny max_seq_len == 64
        img = _img()

        def doc_to_messages(doc):
            del doc
            return [{"role": "user", "content": [_text("hi"), _image(img)]}]

        vlm.task_dict = {"t": {"test": {"d0": {}, "d1": {}}}}
        instances = [
            Instance(
                request_type="generate_until",
                arguments=(ctx, doc_to_messages, {"max_new_tokens": 1000}, doc_id, "t", "test"),
                idx=i,
                metadata={"task": "t", "doc_id": doc_id, "repeats": 1},
            )
            for i, (doc_id, ctx) in enumerate([("d0", "c"), ("d1", "cc")])
        ]
        assert vlm.generate_until(instances) == ["", ""]
        assert any("Skipping" in m and "max_new_tokens" in m for m in rec.warnings)
