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
from kempnerforge.config.schema import JobConfig
from kempnerforge.data.vlm_dataset import (
    DEFAULT_IMAGE_MEAN,
    DEFAULT_IMAGE_STD,
    pil_to_tensor,
)
from kempnerforge.eval.vlm.adapter import (
    KempnerForgeVLM,
    _check_generative_arch,
    _first_stop,
    _frames_to_pixel_values,
    _generate_batch,
    _load_config,
    _load_weights,
    _log_checkpoint_metadata,
    _render_request,
    _resolve_dtype,
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
# _generate_batch (cache-less batched decode loop) on a tiny random VLM
# ---------------------------------------------------------------------------


def _pixels(batch: int = 1) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(batch, 3, 16, 16)


class TestGenerateBatchSingle:
    """B == 1 must reproduce the v1 single-request behavior."""

    def _prompt(self) -> list[torch.Tensor]:
        return [torch.tensor([5, 9, 12, 3], dtype=torch.long)]

    def test_greedy_is_deterministic(self, tiny_vlm_wrapper):
        pv, pid = _pixels(), self._prompt()
        r = _resolve_gen_kwargs({"max_new_tokens": 6}, 128)
        out1 = _generate_batch(tiny_vlm_wrapper, _MockTokenizer(), pv, pid, r, 64)
        out2 = _generate_batch(tiny_vlm_wrapper, _MockTokenizer(), pv, pid, r, 64)
        assert out1 == out2 and len(out1) == 1 and isinstance(out1[0], str)

    def test_respects_max_new_tokens(self, tiny_vlm_wrapper):
        r = _resolve_gen_kwargs({"max_new_tokens": 6}, 128)
        out = _generate_batch(tiny_vlm_wrapper, _MockTokenizer(), _pixels(), self._prompt(), r, 64)
        assert len(out[0].split()) == 6

    def test_until_trims_continuation(self, tiny_vlm_wrapper):
        pv, pid = _pixels(), self._prompt()
        one = _generate_batch(
            tiny_vlm_wrapper,
            _MockTokenizer(),
            pv,
            pid,
            _resolve_gen_kwargs({"max_new_tokens": 1}, 128),
            64,
        )[0]
        # decode = space-joined ids, so the first space follows the first token:
        # until=[" "] trims to exactly the first generated token.
        trimmed = _generate_batch(
            tiny_vlm_wrapper,
            _MockTokenizer(),
            pv,
            pid,
            _resolve_gen_kwargs({"max_new_tokens": 6, "until": [" "]}, 128),
            64,
        )[0]
        assert trimmed == one and " " not in trimmed

    def test_eos_stops_generation(self, tiny_vlm_wrapper):
        pv, pid = _pixels(), self._prompt()
        first = _generate_batch(
            tiny_vlm_wrapper,
            _MockTokenizer(),
            pv,
            pid,
            _resolve_gen_kwargs({"max_new_tokens": 1}, 128),
            64,
        )[0]
        out = _generate_batch(
            tiny_vlm_wrapper,
            _MockTokenizer(eos_token_id=int(first)),
            pv,
            pid,
            _resolve_gen_kwargs({"max_new_tokens": 6}, 128),
            64,
        )
        assert out == [""]

    def test_length_bound_raises_when_no_room(self, tiny_vlm_wrapper):
        r = _resolve_gen_kwargs({"max_new_tokens": 100}, 128)
        with pytest.raises(ValueError, match="max_new_tokens"):
            _generate_batch(tiny_vlm_wrapper, _MockTokenizer(), _pixels(), self._prompt(), r, 64)

    def test_overlong_prompt_is_left_truncated(self, tiny_vlm_wrapper, monkeypatch):
        """A prompt that exceeds the budget (but leaves room) is left-truncated with a warning."""
        rec = _RecordingLogger()
        monkeypatch.setattr("kempnerforge.eval.vlm.adapter.logger", rec)
        # budget = max_seq_len(64) - image_tokens(8) - max_new_tokens(2) = 54; 60 > 54.
        long_prompt = [torch.arange(1, 61, dtype=torch.long)]
        r = _resolve_gen_kwargs({"max_new_tokens": 2}, 128)
        out = _generate_batch(tiny_vlm_wrapper, _MockTokenizer(), _pixels(), long_prompt, r, 64)
        assert len(out) == 1 and len(out[0].split()) == 2
        assert any("left-truncating" in m for m in rec.warnings)


class TestGenerateBatchMulti:
    """B > 1: right-padding must not change any row's result, and stop
    conditions must be tracked per row."""

    def _prompts(self) -> list[torch.Tensor]:
        # Deliberately different lengths so right-padding is exercised.
        return [
            torch.tensor([5, 9, 12, 3], dtype=torch.long),
            torch.tensor([7, 2], dtype=torch.long),
            torch.tensor([1, 4, 8, 11, 20, 6], dtype=torch.long),
        ]

    def test_batch_equals_sequential(self, tiny_vlm_wrapper):
        """Key correctness gate: a right-padded batch yields the same per-row
        continuation as decoding each request alone (greedy, float32)."""
        prompts = self._prompts()
        pv = _pixels(len(prompts))  # (3, 3, 16, 16) — one image per request
        r = _resolve_gen_kwargs({"max_new_tokens": 5}, 128)
        sequential = [
            _generate_batch(tiny_vlm_wrapper, _MockTokenizer(), pv[i : i + 1], [prompts[i]], r, 64)[
                0
            ]
            for i in range(len(prompts))
        ]
        batched = _generate_batch(tiny_vlm_wrapper, _MockTokenizer(), pv, prompts, r, 64)
        assert batched == sequential

    def test_per_row_max_new_tokens(self, tiny_vlm_wrapper):
        prompts = self._prompts()
        pv = _pixels(len(prompts))
        r = _resolve_gen_kwargs({"max_new_tokens": 4}, 128)
        outs = _generate_batch(tiny_vlm_wrapper, _MockTokenizer(), pv, prompts, r, 64)
        assert len(outs) == 3 and all(len(o.split()) == 4 for o in outs)

    def test_per_row_eos_independent(self, tiny_vlm_wrapper):
        """EOS on one row stops only that row; the batch still returns all rows."""
        prompts = self._prompts()
        pv = _pixels(len(prompts))
        first0 = _generate_batch(
            tiny_vlm_wrapper,
            _MockTokenizer(),
            pv[:1],
            [prompts[0]],
            _resolve_gen_kwargs({"max_new_tokens": 1}, 128),
            64,
        )[0]
        outs = _generate_batch(
            tiny_vlm_wrapper,
            _MockTokenizer(eos_token_id=int(first0)),
            pv,
            prompts,
            _resolve_gen_kwargs({"max_new_tokens": 5}, 128),
            64,
        )
        assert len(outs) == 3 and outs[0] == ""  # row 0 stops immediately on EOS


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


# ---------------------------------------------------------------------------
# Loader / __init__ helpers (build a VLM JobConfig + patch the heavy loaders so
# the adapter can be constructed on CPU with no checkpoint).
# ---------------------------------------------------------------------------


def _vlm_job_config(tiny_vlm_configs, arch: str | None = None) -> JobConfig:
    mc, vc, ac, lc = tiny_vlm_configs
    job = JobConfig(
        model=mc, vision_encoder=vc, adapter=ac, vlm=lc, data=DataConfig(tokenizer_path="mock")
    )
    if arch is not None:
        job.vlm.arch = arch
    return job


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
    def test_moma_fails_fast_before_load(self, monkeypatch, tiny_vlm_configs):
        job = _vlm_job_config(tiny_vlm_configs, arch="moma")
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

    def test_init_populates_attrs(self, monkeypatch, tiny_vlm_configs, tiny_vlm_wrapper):
        job = _vlm_job_config(tiny_vlm_configs)
        _patch_loaders(monkeypatch, job, tiny_vlm_wrapper)
        vlm = KempnerForgeVLM(
            config="x", checkpoint="y", device="cpu", dtype="float32", batch_size=2
        )
        assert vlm._arch == "joint_decoder"
        assert vlm._max_seq_len == job.model.max_seq_len
        assert vlm._batch_size == 2
        assert vlm._dtype == torch.float32


# ---------------------------------------------------------------------------
# generate_until (end-to-end on a tiny VLM via the fake lmms-eval objects)
# ---------------------------------------------------------------------------


class TestGenerateUntil:
    def test_batches_and_restores_order(self, monkeypatch, tiny_vlm_configs, tiny_vlm_wrapper):
        _patch_loaders(monkeypatch, _vlm_job_config(tiny_vlm_configs), tiny_vlm_wrapper)
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
