"""Integration tests for the KempnerForge VLM lmms-eval adapter.

Three tests, all skipped when lmms-eval is absent (optional, undeclared dep):

1. ``test_dcp_roundtrip_generate_until`` — self-contained and CPU-only: builds a
   tiny VLM, saves it via DCP, then loads it back through ``KempnerForgeVLM``
   and runs ``generate_until`` on a synthetic single-image request. Exercises
   the real DCP load path and the full request -> render -> preprocess ->
   decode -> collect pipeline without a GPU, a real checkpoint, or the network.

2. ``test_dcp_roundtrip_video_generate_until`` — the video analogue: a tiny
   ``frames_per_clip == 2`` video VLM, decoder stubbed, validating the video
   build and the ``(B, F, 3, H, W)`` clip path on CPU.

3. ``test_real_task_via_simple_evaluate`` — opt-in (set ``KF_VLM_EVAL_CONFIG``
   and ``KF_VLM_EVAL_CHECKPOINT``): runs a small ``--limit`` slice of a real
   lmms-eval ``generate_until`` task through ``simple_evaluate`` against a real
   checkpoint. Point it at a video config + video task to cover real video.
   Intended for a GPU node; skipped by default.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed.checkpoint as dcp
from PIL import Image

lmms_eval = pytest.importorskip("lmms_eval")
if getattr(lmms_eval, "__file__", None) is None:
    # The unit conftest injects a fake lmms_eval (a bare module with no __file__);
    # these DCP-roundtrip tests need the real package, so skip rather than bind the
    # fake (mirrors tests/integration/test_lmms_eval_contract.py).
    pytest.skip("fake lmms_eval is active; skipping real-package tests", allow_module_level=True)

from lmms_eval.api.instance import Instance  # noqa: E402

from kempnerforge.config.data import DataConfig  # noqa: E402
from kempnerforge.config.schema import JobConfig  # noqa: E402
from kempnerforge.eval.vlm.adapter import KempnerForgeVLM  # noqa: E402
from kempnerforge.model.vlm import build_vlm_wrapper  # noqa: E402


class _MockTokenizer:
    pad_token_id = 0
    eos_token_id = None

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        del add_special_tokens
        return {"input_ids": [(ord(c) % 254) + 1 for c in text]}

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(str(int(i)) for i in ids)


def test_dcp_roundtrip_generate_until(tmp_path, tiny_vlm_configs, monkeypatch):
    mc, vc, ac, lc = tiny_vlm_configs

    # Build a tiny VLM and save it as a DCP checkpoint (single process).
    torch.manual_seed(0)
    model = build_vlm_wrapper(mc, vc, ac, lc).eval()
    ckpt_dir = tmp_path / "step_0"
    dcp.save({"model": model.state_dict()}, checkpoint_id=str(ckpt_dir))

    # A matching JobConfig the loader will rebuild from. Patch _load_config to
    # return it (config TOML parsing is covered by the config tests) and patch
    # the tokenizer builder to avoid an HF download.
    job_config = JobConfig(
        model=mc, vision_encoder=vc, adapter=ac, vlm=lc, data=DataConfig(tokenizer_path="mock")
    )
    monkeypatch.setattr("kempnerforge.eval.vlm.adapter._load_config", lambda _path: job_config)
    monkeypatch.setattr(
        "kempnerforge.eval.vlm.adapter.build_tokenizer", lambda _path: _MockTokenizer()
    )

    vlm = KempnerForgeVLM(
        config="ignored", checkpoint=str(ckpt_dir), device="cpu", dtype="float32", batch_size=2
    )

    # Two synthetic single-image requests with different prompt lengths, decoded
    # as one right-padded batch (batch_size=2), mirroring the chat 6-tuple.
    img = Image.new("RGB", (8, 8), color=(120, 120, 120))

    def doc_to_messages(doc):
        return [
            {
                "role": "user",
                "content": [{"type": "text", "text": doc["q"]}, {"type": "image", "url": img}],
            }
        ]

    vlm.task_dict = {
        "t": {
            "test": {
                "d0": {"q": "What color is this?"},
                "d1": {"q": "Describe this picture in a few words please."},
            }
        }
    }
    instances = [
        Instance(
            request_type="generate_until",
            arguments=("ctx", doc_to_messages, {"max_new_tokens": 3}, doc_id, "t", "test"),
            idx=i,
            metadata={"task": "t", "doc_id": doc_id, "repeats": 1},
        )
        for i, doc_id in enumerate(["d0", "d1"])
    ]

    outputs = vlm.generate_until(instances)
    assert isinstance(outputs, list) and len(outputs) == 2
    assert all(isinstance(o, str) for o in outputs)
    assert all(len(o.split()) == 3 for o in outputs)  # greedy emits exactly max_new_tokens


def test_dcp_roundtrip_video_generate_until(tmp_path, tiny_video_configs, monkeypatch):
    """Video analogue of the image roundtrip: a tiny video VLM (frames_per_clip ==
    2) is DCP-saved and reloaded through ``KempnerForgeVLM``, then evaluated on
    synthetic video requests with the decoder stubbed (no ``av`` / no real clip).
    Exercises the real DCP load, the ``frames_per_clip`` build, and the
    render -> decode -> (B, F, 3, H, W) stack -> forward -> decode video path."""
    mc, vc, ac, lc, video = tiny_video_configs

    torch.manual_seed(0)
    model = build_vlm_wrapper(mc, vc, ac, lc, frames_per_clip=video.max_frames).eval()
    ckpt_dir = tmp_path / "step_0"
    dcp.save({"model": model.state_dict()}, checkpoint_id=str(ckpt_dir))

    job_config = JobConfig(
        model=mc,
        vision_encoder=vc,
        adapter=ac,
        vlm=lc,
        video=video,
        data=DataConfig(tokenizer_path="mock"),
    )
    monkeypatch.setattr("kempnerforge.eval.vlm.adapter._load_config", lambda _path: job_config)
    monkeypatch.setattr(
        "kempnerforge.eval.vlm.adapter.build_tokenizer", lambda _path: _MockTokenizer()
    )
    # Stub video decode so the test needs no av / no real video file.
    frames = [Image.new("RGB", (8, 8), color=(120, 120, 120)) for _ in range(2)]
    monkeypatch.setattr(
        "kempnerforge.eval.vlm.adapter.decode_video_frames", lambda path, **kw: frames
    )

    vlm = KempnerForgeVLM(
        config="ignored", checkpoint=str(ckpt_dir), device="cpu", dtype="float32", batch_size=2
    )
    assert vlm._is_video and vlm._frames_per_clip == 2

    def doc_to_messages(doc):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": doc["q"]},
                    {"type": "video", "url": doc["v"]},
                ],
            }
        ]

    vlm.task_dict = {
        "t": {
            "test": {
                "d0": {"q": "What happens in this clip?", "v": "a.mp4"},
                "d1": {"q": "Describe the video briefly please.", "v": "b.mp4"},
            }
        }
    }
    instances = [
        Instance(
            request_type="generate_until",
            arguments=("ctx", doc_to_messages, {"max_new_tokens": 3}, doc_id, "t", "test"),
            idx=i,
            metadata={"task": "t", "doc_id": doc_id, "repeats": 1},
        )
        for i, doc_id in enumerate(["d0", "d1"])
    ]

    outputs = vlm.generate_until(instances)
    assert isinstance(outputs, list) and len(outputs) == 2
    assert all(isinstance(o, str) for o in outputs)
    assert all(len(o.split()) == 3 for o in outputs)  # greedy emits exactly max_new_tokens


@pytest.mark.skipif(
    not os.environ.get("KF_VLM_EVAL_CONFIG") or not os.environ.get("KF_VLM_EVAL_CHECKPOINT"),
    reason="set KF_VLM_EVAL_CONFIG and KF_VLM_EVAL_CHECKPOINT to run against a real checkpoint",
)
def test_real_task_via_simple_evaluate():
    from lmms_eval.evaluator import simple_evaluate

    config = os.environ["KF_VLM_EVAL_CONFIG"]
    checkpoint = os.environ["KF_VLM_EVAL_CHECKPOINT"]
    task = os.environ.get("KF_VLM_EVAL_TASK", "mmmu_val")

    results = simple_evaluate(
        model="kempnerforge_vlm",
        model_args=f"config={config},checkpoint={checkpoint}",
        tasks=[task],
        limit=2,
    )
    assert results is not None
    assert "results" in results and task in results["results"]
