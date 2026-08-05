"""Unit tests for video-dataset mixing: ``[[video.datasets]]`` and its wiring.

Covers config normalization (``sources`` / ``for_source``), the fail-fast
invariants, TOML loading of the array-of-tables form, and ``build_video_data``
returning a weighted ``MixtureDataset``.
"""

from __future__ import annotations

import json

import pytest

from kempnerforge.config.loader import load_config
from kempnerforge.config.video import VideoConfig, VideoDatasetSource

GEOMETRY = {"max_frames": 4, "min_frames": 2, "fps": 2.0, "frame_size": 16}
TOKENIZER = "gpt2"


def _source(**kwargs):
    return VideoDatasetSource(**{"dataset_type": "webvid", "data_root": "/w", **kwargs})


# ---------------------------------------------------------------------------
# VideoDatasetSource validation
# ---------------------------------------------------------------------------


class TestVideoDatasetSource:
    def test_dataset_type_is_required(self):
        with pytest.raises(ValueError, match="require a dataset_type"):
            VideoDatasetSource(data_root="/w")

    def test_data_root_is_required(self):
        with pytest.raises(ValueError, match="requires a data_root"):
            VideoDatasetSource(dataset_type="webvid")

    @pytest.mark.parametrize("weight", [0.0, -1.0])
    def test_non_positive_weight_rejected(self, weight):
        with pytest.raises(ValueError, match="weight must be positive"):
            _source(weight=weight)

    def test_negative_max_samples_rejected(self):
        with pytest.raises(ValueError, match="max_samples must be"):
            _source(max_samples=-1)

    def test_bad_split_rejected(self):
        with pytest.raises(ValueError, match="split must be one of"):
            _source(split="valid")

    def test_metrics_name_defaults_to_type(self):
        assert _source().metrics_name == "webvid"
        assert _source(name="pretrain").metrics_name == "pretrain"


# ---------------------------------------------------------------------------
# VideoConfig normalization
# ---------------------------------------------------------------------------


class TestSources:
    def test_flat_config_yields_one_source(self):
        cfg = VideoConfig(data_root="/w", dataset_type="webvid", split="train", **GEOMETRY)
        (src,) = cfg.sources()
        assert (src.dataset_type, src.data_root, src.split, src.weight) == (
            "webvid",
            "/w",
            "train",
            1.0,
        )

    def test_mixture_inherits_unset_fields_from_video(self):
        cfg = VideoConfig(
            split="validation",
            prompt="Describe the video:",
            qa_format="mcq_text",
            dataset_name="webvid-10M",
            datasets=[
                _source(dataset_type="webvid"),
                _source(dataset_type="nextqa", data_root="/n", split="train", prompt="Answer."),
            ],
            **GEOMETRY,
        )
        a, b = cfg.sources()
        assert (a.split, a.prompt, a.qa_format) == ("validation", "Describe the video:", "mcq_text")
        assert (b.split, b.prompt) == ("train", "Answer.")  # explicit values win

    def test_mixture_inherits_all_inheritable_fields(self):
        # max_samples / require_video_file / subset / text_source must inherit from
        # [video] too, exactly like the single-corpus path passes them through.
        # Otherwise a [video].max_samples smoke cap (or an asr text_source) is
        # silently dropped for every source and the two config forms disagree.
        cfg = VideoConfig(
            max_samples=100,
            require_video_file=True,
            subset="OE",
            text_source="asr",
            datasets=[
                _source(dataset_type="webvid"),  # inherits all four
                _source(
                    dataset_type="nextqa",
                    data_root="/n",
                    max_samples=5,
                    require_video_file=False,
                    subset="MC",
                    text_source="title",
                ),  # explicit values win
            ],
            **GEOMETRY,
        )
        a, b = cfg.sources()
        assert (a.max_samples, a.require_video_file, a.subset, a.text_source) == (
            100,
            True,
            "OE",
            "asr",
        )
        assert (b.max_samples, b.require_video_file, b.subset, b.text_source) == (
            5,
            False,
            "MC",
            "title",
        )

    def test_mixture_require_video_file_none_inherits_video_none(self):
        # Tri-state: source None + [video] None stays None (builder default), not
        # coerced to a bool.
        cfg = VideoConfig(
            datasets=[_source(dataset_type="nextqa", data_root="/n")], **GEOMETRY
        )
        (src,) = cfg.sources()
        assert src.require_video_file is None

    def test_for_source_shares_geometry_and_swaps_corpus_fields(self):
        cfg = VideoConfig(
            max_frames=8,
            min_frames=3,
            fps=1.5,
            frame_size=112,
            sampling_policy="uniform",
            datasets=[_source(dataset_type="nextqa", data_root="/n", subset="OE")],
        )
        (src,) = cfg.sources()
        sub = cfg.for_source(src)
        # Geometry is the shared [video] value ...
        assert (sub.max_frames, sub.min_frames, sub.fps, sub.frame_size) == (8, 3, 1.5, 112)
        # ... corpus fields come from the source ...
        assert (sub.dataset_type, sub.data_root, sub.subset) == ("nextqa", "/n", "OE")
        # ... and the derived view is a plain single-corpus config.
        assert sub.datasets == []

    def test_for_source_leaves_the_parent_untouched(self):
        cfg = VideoConfig(datasets=[_source(dataset_type="nextqa", data_root="/n")], **GEOMETRY)
        cfg.for_source(cfg.sources()[0])
        assert len(cfg.datasets) == 1
        assert cfg.data_root == ""


class TestVideoConfigInvariants:
    def test_data_root_and_datasets_are_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            VideoConfig(data_root="/w", datasets=[_source()], **GEOMETRY)

    def test_unknown_dataset_type_in_a_source_rejected(self):
        with pytest.raises(ValueError, match="dataset_type must be one of"):
            VideoConfig(datasets=[_source(dataset_type="bogus")], **GEOMETRY)

    def test_duplicate_source_names_rejected(self):
        with pytest.raises(ValueError, match="names must be unique"):
            VideoConfig(datasets=[_source(data_root="/a"), _source(data_root="/b")], **GEOMETRY)

    def test_duplicate_types_allowed_with_explicit_names(self):
        cfg = VideoConfig(
            datasets=[
                _source(data_root="/a", name="webvid_a"),
                _source(data_root="/b", name="webvid_b"),
            ],
            **GEOMETRY,
        )
        assert [s.metrics_name for s in cfg.sources()] == ["webvid_a", "webvid_b"]

    def test_unknown_qa_format_rejected(self):
        with pytest.raises(ValueError, match="qa_format must be one of"):
            VideoConfig(qa_format="bogus", **GEOMETRY)

    def test_unknown_qa_format_on_a_source_rejected(self):
        with pytest.raises(ValueError, match="qa_format must be one of"):
            VideoConfig(datasets=[_source(qa_format="bogus")], **GEOMETRY)


# ---------------------------------------------------------------------------
# TOML loading
# ---------------------------------------------------------------------------


class TestTomlLoading:
    def _write(self, tmp_path, body):
        path = tmp_path / "cfg.toml"
        path.write_text(
            "[model]\ndim = 64\nn_layers = 2\nn_heads = 4\nn_kv_heads = 4\nmax_seq_len = 256\n\n"
            "[vision_encoder]\ntype = 'random'\nfeature_dim = 64\nnum_tokens = 16\n\n"
            "[vlm]\narch = 'joint_decoder'\nmax_text_len = 32\n\n" + body
        )
        return path

    def test_array_of_tables_loads(self, tmp_path):
        cfg = load_config(
            self._write(
                tmp_path,
                "[video]\nmax_frames = 4\nfps = 2.0\n\n"
                "[[video.datasets]]\ndataset_type = 'webvid'\ndata_root = '/w'\nweight = 2.0\n\n"
                "[[video.datasets]]\ndataset_type = 'nextqa'\ndata_root = '/n'\n"
                "subset = 'MC'\nweight = 1.0\n",
            ),
            cli_args=[],
        )
        assert cfg.video is not None
        assert [(s.dataset_type, s.weight) for s in cfg.video.sources()] == [
            ("webvid", 2.0),
            ("nextqa", 1.0),
        ]

    def test_geometry_on_a_source_is_rejected_as_an_unknown_key(self, tmp_path):
        # Frame geometry is global; a per-source override would silently desync
        # the visual-token budget, so the loader must reject it.
        with pytest.raises(ValueError, match="Unknown config keys in VideoDatasetSource"):
            load_config(
                self._write(
                    tmp_path,
                    "[video]\nmax_frames = 4\n\n"
                    "[[video.datasets]]\ndataset_type = 'webvid'\ndata_root = '/w'\n"
                    "max_frames = 32\n",
                ),
                cli_args=[],
            )

    def test_flat_form_still_loads(self, tmp_path):
        cfg = load_config(
            self._write(
                tmp_path,
                "[video]\ndata_root = '/w'\ndataset_type = 'webvid'\nmax_frames = 4\n",
            ),
            cli_args=[],
        )
        assert cfg.video is not None and cfg.video.datasets == []
        assert cfg.video.sources()[0].data_root == "/w"


# ---------------------------------------------------------------------------
# build_video_data
# ---------------------------------------------------------------------------


def _perception_root(tmp_path, name, n_questions=2):
    root = tmp_path / name
    vid = "video_0"
    (root / "videos").mkdir(parents=True)
    (root / "videos" / f"{vid}.mp4").write_bytes(b"")
    (root / "mc_question_train.json").write_text(
        json.dumps(
            {
                vid: {
                    "metadata": {"video_id": vid},
                    "mc_question": [
                        {
                            "id": q,
                            "question": f"q{q}",
                            "options": ["a", "b", "c"],
                            "answer_id": 0,
                        }
                        for q in range(n_questions)
                    ],
                }
            }
        )
    )
    return str(root)


class TestBuildVideoData:
    def test_single_corpus_returns_no_mixture(self, tmp_path):
        from kempnerforge.data.video_dataset import build_video_data

        cfg = VideoConfig(
            data_root=_perception_root(tmp_path, "pt", n_questions=3),
            dataset_type="perception_test",
            split="train",
            **GEOMETRY,
        )
        dataset, mixture, weights = build_video_data(cfg, TOKENIZER, max_text_len=32)
        assert mixture is None
        assert len(dataset) == 3
        assert weights == [1.0]

    def test_mixture_concatenates_and_carries_weights(self, tmp_path):
        from kempnerforge.data.video_dataset import build_video_data

        cfg = VideoConfig(
            datasets=[
                VideoDatasetSource(
                    dataset_type="perception_test",
                    data_root=_perception_root(tmp_path, "a", n_questions=3),
                    split="train",
                    name="a",
                    weight=3.0,
                ),
                VideoDatasetSource(
                    dataset_type="perception_test",
                    data_root=_perception_root(tmp_path, "b", n_questions=2),
                    split="train",
                    name="b",
                    weight=1.0,
                ),
            ],
            **GEOMETRY,
        )
        dataset, mixture, weights = build_video_data(cfg, TOKENIZER, max_text_len=32)
        assert mixture is not None
        assert mixture.dataset_names == ["a", "b"]
        assert mixture.cumulative_sizes == [0, 3, 5]
        assert len(dataset) == 5
        assert weights == [3.0, 1.0]

    def test_mixture_tags_samples_with_their_source(self, tmp_path, monkeypatch):
        from PIL import Image

        from kempnerforge.data import video_dataset as vd
        from kempnerforge.data.video_dataset import build_video_data

        monkeypatch.setattr(
            vd,
            "decode_video_frames",
            lambda *a, **k: ([Image.new("RGB", (16, 16))] * 2, [0.0, 1.0]),
        )
        cfg = VideoConfig(
            datasets=[
                VideoDatasetSource(
                    dataset_type="perception_test",
                    data_root=_perception_root(tmp_path, "a", n_questions=2),
                    split="train",
                    name="a",
                ),
                VideoDatasetSource(
                    dataset_type="perception_test",
                    data_root=_perception_root(tmp_path, "b", n_questions=2),
                    split="train",
                    name="b",
                ),
            ],
            **GEOMETRY,
        )
        dataset, _, _ = build_video_data(cfg, TOKENIZER, max_text_len=32)
        assert dataset[0]["dataset_idx"] == 0
        assert dataset[3]["dataset_idx"] == 1


class TestMixtureSamplerWeights:
    """The weights act on the sampler, not the loss — verify the draw counts."""

    def test_draw_counts_follow_the_weights(self):
        from kempnerforge.data.sampler import MixtureSampler

        sampler = MixtureSampler(
            cumulative_sizes=[0, 400, 800],
            weights=[3.0, 1.0],
            num_replicas=1,
            rank=0,
            shuffle=True,
            seed=0,
        )
        drawn = list(sampler)
        from_a = sum(1 for i in drawn if i < 400)
        # 3:1 weighting over an 800-sample epoch -> ~600 from the first corpus.
        assert abs(from_a / len(drawn) - 0.75) < 0.02
