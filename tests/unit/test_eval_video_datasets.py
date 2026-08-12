"""Unit tests for the evaluation-benchmark video loaders.

These three corpora are read-only evaluation sets, so the behaviour that matters
is that each one renders its question EXACTLY as the benchmark ships it (a
rewritten prompt stops the score being comparable to published numbers) and that
its category label survives onto ``strata`` for stratified sampling.

Fixtures build the same parquet layout the HuggingFace packaging uses, so a
change to that layout fails here rather than silently producing zero samples.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from kempnerforge.data.video_qa_datasets import (
    MLVUDataset,
    PerceptionTestValDataset,
    TempCompassDataset,
)

TOKENIZER = "gpt2"
GEOMETRY = {"max_frames": 4, "min_frames": 1, "fps": 1.0, "frame_size": 64}


def _touch_videos(directory, names):
    os.makedirs(directory, exist_ok=True)
    for name in names:
        with open(os.path.join(directory, name), "wb") as handle:
            handle.write(b"\x00")


@pytest.fixture
def tempcompass_root(tmp_path):
    snap = tmp_path / "hub" / "datasets--lmms-eval--TempCompass" / "snapshots" / "abc"
    for subset, rows in {
        "multi-choice": [
            {"video_id": "v1", "question": "What is he doing?\nA. dunking\nB. dribbling",
             "answer": "A. dunking", "dim": "action"},
            {"video_id": "v2", "question": "Which way?\nA. left\nB. right",
             "answer": "B. right", "dim": "direction"},
        ],
        "yes_no": [
            {"video_id": "v1", "question": "Is he dunking?", "answer": "yes", "dim": "action"},
        ],
        "caption_matching": [
            {"video_id": "v1",
             "question": "Which caption matches?\nOption 1: He dribbles.\nOption 2: He dunks.",
             "answer": "Option 2: He dunks.", "dim": "action"},
        ],
        # `answer` indexes the Information list the prompt shows; `mc_answer`
        # indexes mc_question, which it does not. They disagree here on purpose.
        "captioning": [
            {"video_id": "v1",
             "question": "Describe the video.\nInformation A: dribbling\nInformation B: dunking",
             "answer": "B. dunking", "mc_answer": "A. dunking",
             "mc_question": "What is he doing?\nA. dunking\nB. dribbling", "dim": "action"},
        ],
    }.items():
        (snap / subset).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(snap / subset / "test-0.parquet")
    _touch_videos(tmp_path / "tempcompass" / "videos", ["v1.mp4", "v2.mp4"])
    return str(tmp_path)


@pytest.fixture
def mlvu_root(tmp_path):
    snap = tmp_path / "hub" / "datasets--sy1998--MLVU_dev" / "snapshots" / "abc" / "mlvu"
    snap.mkdir(parents=True)
    pd.DataFrame([
        {"video_name": "needle_1.mp4", "duration": 400.0,
         "question": "What happens?\n(A) x\n(B) y", "candidates": ["x", "y"],
         "answer": "A", "task_type": "needle", "question_id": "Q0"},
        {"video_name": "ego_2", "duration": 90.0,
         "question": "Which first?\n(A) p\n(B) q", "candidates": ["p", "q"],
         "answer": "B", "task_type": "order", "question_id": "Q1"},
    ]).to_parquet(snap / "test-0.parquet")
    _touch_videos(tmp_path / "mlvu", ["needle_1.mp4", "ego_2.mp4"])
    return str(tmp_path)


@pytest.fixture
def perception_root(tmp_path):
    snap = (tmp_path / "hub" / "datasets--lmms-eval--PerceptionTest_Val"
            / "snapshots" / "abc" / "mc_question_val")
    snap.mkdir(parents=True)
    pd.DataFrame([
        {"video_name": "video_1", "question": "Camera moving?", "question_id": 0,
         "options": ["dunno", "moving", "static"], "answer_id": 2,
         "area": "physics", "reasoning": "descriptive", "tag": ["motion"]},
    ]).to_parquet(snap / "validation-0.parquet")
    _touch_videos(tmp_path / "perceptiontest_val" / "videos", ["video_1.mp4"])
    return str(tmp_path)


class TestTempCompass:
    def test_prompt_matches_the_reference_protocol(self, tempcompass_root):
        """The rendered prompt must equal question + the benchmark's own
        post-prompt, byte for byte.

        Copied from lmms_eval/tasks/tempcompass/_default_template_yaml. A generic
        "Answer:" here scores a different input distribution than every published
        TempCompass number, so this asserts the WHOLE string rather than a
        substring -- a containment check would pass on the broken rendering.
        """
        ds = TempCompassDataset(tempcompass_root, TOKENIZER, 64,
                                subset="multi-choice", **GEOMETRY)
        rec = ds._record(0)
        assert rec.prompt == ("What is he doing?\nA. dunking\nB. dribbling"
                              "\nPlease directly give the best option:")
        assert rec.target == " A. dunking"

    def test_yes_no_subset_uses_its_own_post_prompt(self, tempcompass_root):
        ds = TempCompassDataset(tempcompass_root, TOKENIZER, 64, subset="yes_no", **GEOMETRY)
        assert ds._record(0).prompt == "Is he dunking?\nPlease answer yes or no:"

    def test_dim_kept_as_strata(self, tempcompass_root):
        ds = TempCompassDataset(tempcompass_root, TOKENIZER, 64,
                                subset="multi-choice", **GEOMETRY)
        assert ds.strata == ["action", "direction"]

    def test_subset_selects_a_different_file(self, tempcompass_root):
        ds = TempCompassDataset(tempcompass_root, TOKENIZER, 64, subset="yes_no", **GEOMETRY)
        assert len(ds) == 1
        assert ds._record(0).target == " yes"

    def test_unknown_subset_rejected(self, tempcompass_root):
        with pytest.raises(ValueError, match="tempcompass subset"):
            TempCompassDataset(tempcompass_root, TOKENIZER, 64, subset="nope", **GEOMETRY)

    def test_missing_video_dropped(self, tempcompass_root):
        os.remove(os.path.join(tempcompass_root, "tempcompass", "videos", "v2.mp4"))
        ds = TempCompassDataset(tempcompass_root, TOKENIZER, 64,
                                subset="multi-choice", **GEOMETRY)
        assert len(ds) == 1
        assert ds.strata == ["action"]


class TestMLVU:
    def test_prompt_matches_the_reference_protocol(self, mlvu_root):
        """The trailing "Best option: (" is load-bearing -- options render as
        "(A)", so the primer constrains the next token to a bare letter."""
        ds = MLVUDataset(mlvu_root, TOKENIZER, 64, **GEOMETRY)
        rec = ds._record(0)
        assert rec.prompt == ("What happens?\n(A) x\n(B) y"
                              "\nOnly give the best option.\nBest option: (")
        assert rec.target == " A"

    def test_video_name_without_extension_resolves(self, mlvu_root):
        """The parquet mixes 'ego_2' and 'needle_1.mp4' spellings."""
        ds = MLVUDataset(mlvu_root, TOKENIZER, 64, **GEOMETRY)
        assert ds._record(1).video_path.endswith("ego_2.mp4")

    def test_task_type_kept_as_strata(self, mlvu_root):
        ds = MLVUDataset(mlvu_root, TOKENIZER, 64, **GEOMETRY)
        assert ds.strata == ["needle", "order"]


class TestPerceptionTestVal:
    def test_prompt_matches_the_reference_protocol(self, perception_root):
        """No "Question:" prefix and the benchmark's own trailing instruction.
        Routing this through qa_format added a prefix, a different hint and a
        second answer cue -- three independent deltas from the reference."""
        ds = PerceptionTestValDataset(perception_root, TOKENIZER, 64, **GEOMETRY)
        rec = ds._record(0)
        assert rec.prompt == ("Camera moving?\nA. dunno\nB. moving\nC. static"
                              "\nAnswer with the option's letter from the given "
                              "choices directly.")
        assert rec.target == " C"
        assert not rec.prompt.startswith("Question:")

    def test_area_and_reasoning_joined_as_strata(self, perception_root):
        ds = PerceptionTestValDataset(perception_root, TOKENIZER, 64, **GEOMETRY)
        assert ds.strata == ["physics/descriptive"]

    def test_max_samples_caps(self, perception_root_3rows):
        """Needs >1 source row: with a single-row fixture this passes whether or
        not max_samples is implemented at all."""
        ds = PerceptionTestValDataset(perception_root_3rows, TOKENIZER, 64,
                                      max_samples=2, **GEOMETRY)
        assert len(ds) == 2
        assert len(ds.strata) == 2

    def test_missing_video_keeps_strata_aligned(self, perception_root_3rows):
        os.remove(os.path.join(perception_root_3rows, "perceptiontest_val",
                               "videos", "video_2.mp4"))
        ds = PerceptionTestValDataset(perception_root_3rows, TOKENIZER, 64, **GEOMETRY)
        assert len(ds) == 2 and len(ds.strata) == len(ds)
        assert ds.strata == ["physics/descriptive", "memory/explanatory"]


@pytest.fixture
def perception_root_3rows(tmp_path):
    snap = (tmp_path / "hub" / "datasets--lmms-eval--PerceptionTest_Val"
            / "snapshots" / "abc" / "mc_question_val")
    snap.mkdir(parents=True)
    pd.DataFrame([
        {"video_name": f"video_{i}", "question": f"Q{i}?", "question_id": i,
         "options": ["a", "b", "c"], "answer_id": i % 3,
         "area": area, "reasoning": reason, "tag": ["t"]}
        for i, (area, reason) in enumerate(
            [("physics", "descriptive"), ("memory", "explanatory"), ("semantics", "predictive")])
    ]).to_parquet(snap / "validation-0.parquet")
    _touch_videos(tmp_path / "perceptiontest_val" / "videos",
                  ["video_0.mp4", "video_1.mp4", "video_2.mp4"])
    return str(tmp_path)


def test_all_three_are_registered(tempcompass_root, mlvu_root, perception_root):
    """Each registered name must build ITS class.

    The previous version ended in ``or True``, so it asserted nothing at all --
    every name could have resolved to the wrong builder and it stayed green.
    Build through the registry and check the resulting type.
    """
    from types import SimpleNamespace

    from kempnerforge.config import registry

    for name, cls, root, subset in (
        ("tempcompass", TempCompassDataset, tempcompass_root, "yes_no"),
        ("mlvu", MLVUDataset, mlvu_root, ""),
        ("perception_test_val", PerceptionTestValDataset, perception_root, ""),
    ):
        vc = SimpleNamespace(data_root=root, subset=subset, split="", max_samples=0,
                             prompt="", qa_format="mcq_letter", require_video_file=None,
                             sampling_policy="uniform", **GEOMETRY)
        ds = registry.get_video_dataset(name)(vc, TOKENIZER, 64)
        assert type(ds) is cls, f"{name} built {type(ds).__name__}, expected {cls.__name__}"
        assert len(ds) > 0
    with pytest.raises(Exception):
        registry.get_video_dataset("no_such_corpus_xyz")


def test_strata_length_matches_records(tempcompass_root, mlvu_root, perception_root):
    """Alignment invariant, asserted for every loader rather than just one."""
    for ds in (TempCompassDataset(tempcompass_root, TOKENIZER, 64,
                                  subset="multi-choice", **GEOMETRY),
               MLVUDataset(mlvu_root, TOKENIZER, 64, **GEOMETRY),
               PerceptionTestValDataset(perception_root, TOKENIZER, 64, **GEOMETRY)):
        assert len(ds.strata) == len(ds)


def test_tempcompass_caption_matching_prompt_is_golden(tempcompass_root):
    """caption_matching had no test at all, so its post-prompt could be replaced
    with anything and the suite stayed green."""
    ds = TempCompassDataset(tempcompass_root, TOKENIZER, 64, subset="caption_matching",
                            **GEOMETRY)
    assert ds._record(0).prompt == (
        "Which caption matches?\n"
        "Option 1: He dribbles.\n"
        "Option 2: He dunks."
        "\nPlease directly give the best option:"
    )
    assert ds._record(0).target == " Option 2: He dunks."


def test_tempcompass_captioning_prompt_and_gold(tempcompass_root):
    """captioning appends nothing, and scores against `answer`.

    `mc_answer` letters index `mc_question`, which the captioning prompt never
    shows -- it belongs to the upstream ChatGPT re-ask path. Using it here is a
    label error, and this fixture makes the two columns disagree so that swapping
    them fails.
    """
    ds = TempCompassDataset(tempcompass_root, TOKENIZER, 64, subset="captioning", **GEOMETRY)
    rec = ds._record(0)
    assert rec.prompt == (
        "Describe the video.\n"
        "Information A: dribbling\n"
        "Information B: dunking"
    )
    assert rec.target == " B. dunking"          # `answer`
    assert rec.target != " A. dunking"          # `mc_answer`


def test_tempcompass_prompt_overrides_the_post_prompt(tempcompass_root):
    """`prompt` REPLACES the published post-prompt rather than adding to it --
    which is what lets the eval harness render the training format."""
    ds = TempCompassDataset(tempcompass_root, TOKENIZER, 64, subset="multi-choice",
                            prompt="\nAnswer with the option's letter.\nAnswer:", **GEOMETRY)
    assert ds._record(0).prompt == (
        "What is he doing?\nA. dunking\nB. dribbling"
        "\nAnswer with the option's letter.\nAnswer:"
    )
    assert "Please directly give the best option" not in ds._record(0).prompt


def test_perception_test_val_post_prompt_is_the_published_one(perception_root):
    ds = PerceptionTestValDataset(perception_root, TOKENIZER, 64, **GEOMETRY)
    assert ds._record(0).prompt.endswith(
        "\nAnswer with the option's letter from the given choices directly.")
    assert PerceptionTestValDataset._POST_PROMPT == (
        "\nAnswer with the option's letter from the given choices directly.")
