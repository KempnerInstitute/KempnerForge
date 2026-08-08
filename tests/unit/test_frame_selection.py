"""Unit tests for query-aware frame selection (topk / qframe / mdp3).

All CPU, no network: a ``FakeScorer`` with planted embeddings is injected into
the selectors, so no HuggingFace weights are downloaded. The DPP greedy is
checked against a naive greedy that recomputes ``logdet`` each step (the real
correctness property — greedy MAP is a (1-1/e) approximation, so it is NOT
compared to the global-optimum subset). The Markov segment DP is checked
exhaustively for the two-segment case (where the lazy recurrence is provably
exact) and behaviorally for more segments.
"""

from __future__ import annotations

import math
import sys
import types

import pytest
import torch
import torch.nn.functional as F

from kempnerforge.config.registry import registry
from kempnerforge.data import frame_selection as fsmod
from kempnerforge.data.frame_selection import (
    CandidatePoolSpec,
    FrameQueryScorer,
    FrameSelector,
    MDP3Selector,
    QFrameSelector,
    ScorerPreprocess,
    ScorerUnavailableError,
    TopKSelector,
    _dedupe_by_time,
    _derive_seed,
    _dpp_greedy,
    _mdp3_segment_dp,
    _pooled_features,
    apply_frame_selection,
    build_frame_selector,
    decode_candidate_pool,
    make_seed_key,
    select_video_frames,
    uniform_stride_indices,
)
from kempnerforge.data.video_io import sample_timestamps


class FakeScorer:
    """Returns planted embeddings; records calls. No HF, no network."""

    def __init__(self, frame_emb: torch.Tensor, query_emb: torch.Tensor) -> None:
        self._frame_emb = frame_emb.float()
        self._query_emb = query_emb.float()
        self.embed_frames_calls = 0
        self.embed_frame_tensors_calls = 0
        self.embed_uint8_frames_calls = 0
        self.embed_query_calls = 0

    def ensure_loaded(self):
        pass

    def embed_frames(self, frames):  # noqa: ANN001
        self.embed_frames_calls += 1
        return F.normalize(self._frame_emb[: len(frames)], dim=-1)

    def embed_frame_tensors(self, pixel_values):  # noqa: ANN001
        self.embed_frame_tensors_calls += 1
        return F.normalize(self._frame_emb[: pixel_values.shape[0]], dim=-1)

    def preprocess_frame_tensors(self, frames_u8):  # noqa: ANN001
        return frames_u8.to(torch.float32)  # identity-ish: no resize, no stats

    def embed_uint8_frames(self, frames_u8):  # noqa: ANN001
        self.embed_uint8_frames_calls += 1
        return F.normalize(self._frame_emb[: frames_u8.shape[0]], dim=-1)

    def embed_query(self, query):  # noqa: ANN001
        self.embed_query_calls += 1
        return F.normalize(self._query_emb, dim=-1)


def _dummy_frames(n: int) -> list[int]:
    return list(range(n))


# --------------------------------------------------------------------------- #
# Degenerate paths (base class)
# --------------------------------------------------------------------------- #


class TestFrameSelectorFallbacks:
    def setup_method(self):
        # The ``kempnerforge`` logger has propagation disabled once
        # ``metrics.logger`` configures it (so it's off in a full-suite run but on
        # when this file runs alone); re-enable it so caplog (root-attached) can
        # observe the warn-once record. Mirrors the config warning tests.
        import logging

        self._kf_logger = logging.getLogger("kempnerforge")
        self._old_propagate = self._kf_logger.propagate
        self._kf_logger.propagate = True

    def teardown_method(self):
        self._kf_logger.propagate = self._old_propagate

    def _selector(self) -> TopKSelector:
        return TopKSelector(FakeScorer(torch.eye(4), torch.ones(4)))

    def test_empty_frames(self):
        assert self._selector().select([], [], "q", 4) == []

    def test_k_le_zero(self):
        assert self._selector().select(_dummy_frames(4), [0, 1, 2, 3], "q", 0) == []

    def test_n_le_k_returns_identity(self):
        sel = self._selector()
        assert sel.select(_dummy_frames(3), [0, 1, 2], "q", 4) == [0, 1, 2]
        assert sel.select(_dummy_frames(4), [0, 1, 2, 3], "q", 4) == [0, 1, 2, 3]

    def test_empty_query_uniform_stride(self):
        scorer = FakeScorer(torch.eye(8), torch.ones(8))
        sel = TopKSelector(scorer)
        idx = sel.select(_dummy_frames(8), list(range(8)), None, 4)
        assert idx == [0, 2, 4, 6]  # i * n // k
        assert len(set(idx)) == 4
        # The scorer is never consulted on the fallback path.
        assert scorer.embed_frames_calls == 0
        assert scorer.embed_query_calls == 0

    def test_empty_query_warns_once(self, caplog):
        sel = self._selector()
        with caplog.at_level("WARNING"):
            sel.select(_dummy_frames(8), list(range(8)), "", 4)
            sel.select(_dummy_frames(8), list(range(8)), "   ", 4)
        assert sum("empty query" in r.message for r in caplog.records) == 1

    def test_selection_is_sorted(self):
        # Scores decreasing with index -> topk picks high indices, but output is
        # returned in ascending (temporal) order.
        scorer = FakeScorer(torch.eye(6), torch.tensor([6.0, 5, 4, 3, 2, 1]))
        idx = TopKSelector(scorer).select(_dummy_frames(6), list(range(6)), "q", 3)
        assert idx == sorted(idx)


class TestUniformStrideIndices:
    def test_values(self):
        assert uniform_stride_indices(6, 4) == [0, 1, 3, 4]
        assert uniform_stride_indices(8, 4) == [0, 2, 4, 6]

    def test_fallback_consistency(self):
        # The empty-query fallback must produce exactly the shared stride, so
        # worker-side pre-striding and the main-process fallback agree.
        sel = TopKSelector(FakeScorer(torch.eye(8), torch.ones(8)))
        assert sel._fallback_indices(8, None, 4) == uniform_stride_indices(8, 4)
        assert sel._fallback_indices(6, "", 4) == uniform_stride_indices(6, 4)

    def test_take_all_checked_before_stride(self):
        # Workers pre-apply the stride and ship <= k frames with an EMPTY query;
        # the downstream fallback must take-all (n <= k) before considering the
        # empty-query stride, or pre-strided pools would be re-strided.
        sel = TopKSelector(FakeScorer(torch.eye(8), torch.ones(8)))
        assert sel._fallback_indices(4, None, 4) == [0, 1, 2, 3]
        assert sel._fallback_indices(3, "", 4) == [0, 1, 2]


# --------------------------------------------------------------------------- #
# TopK
# --------------------------------------------------------------------------- #


class TestTopKSelector:
    def test_picks_highest_similarity(self):
        # query aligned with increasing weights -> highest-index frames selected.
        scorer = FakeScorer(torch.eye(6), torch.tensor([1.0, 2, 3, 4, 5, 6]))
        idx = TopKSelector(scorer).select(_dummy_frames(6), list(range(6)), "q", 3)
        assert idx == [3, 4, 5]


# --------------------------------------------------------------------------- #
# QFrame (Gumbel-Max top-k)
# --------------------------------------------------------------------------- #


class TestQFrameSelector:
    def _scorer(self, n: int = 4) -> FakeScorer:
        return FakeScorer(torch.eye(n), torch.tensor([1.0, 2.0, 3.0, 4.0]))

    def test_reproducible_across_fresh_selectors(self):
        a = QFrameSelector(self._scorer(), seed=7).select(
            _dummy_frames(4), [0, 1, 2, 3], "q", 2, seed_key="vid42"
        )
        b = QFrameSelector(self._scorer(), seed=7).select(
            _dummy_frames(4), [0, 1, 2, 3], "q", 2, seed_key="vid42"
        )
        assert a == b

    def test_seed_key_sensitivity(self):
        sel = QFrameSelector(self._scorer(), seed=0)
        # Different keys should (for at least one pair) give different draws.
        picks = {
            tuple(sel.select(_dummy_frames(4), [0, 1, 2, 3], "q", 2, seed_key=f"v{i}"))
            for i in range(20)
        }
        assert len(picks) > 1

    def test_sorted_and_exact_k(self):
        idx = QFrameSelector(self._scorer(), seed=1).select(
            _dummy_frames(4), [0, 1, 2, 3], "q", 2, seed_key="v"
        )
        assert idx == sorted(idx)
        assert len(idx) == 2
        assert all(0 <= i < 4 for i in idx)

    def test_high_score_selected_more_often(self):
        # Over many seed keys, the highest-logit frame (3) is selected far more
        # often than the lowest (0).
        sel = QFrameSelector(self._scorer(), seed=0)
        count = {0: 0, 3: 0}
        for i in range(200):
            idx = sel.select(_dummy_frames(4), [0, 1, 2, 3], "q", 2, seed_key=f"k{i}")
            for j in (0, 3):
                if j in idx:
                    count[j] += 1
        assert count[3] > count[0]


class TestDeriveSeed:
    def test_none_key_is_base_seed(self):
        assert _derive_seed(5, None) == 5

    def test_pythonhashseed_independent_and_deterministic(self):
        # Pin the literal SHA-256-derived value. In-process equality alone would
        # stay green if the impl regressed to the PYTHONHASHSEED-salted builtin
        # hash() the code deliberately avoids — which would silently diverge
        # QFrame/mDP3 draws across DP ranks and resumes. The literal is what makes
        # "stable across processes" actually testable here.
        assert _derive_seed(0, "abc") == 965315470085333761

    def test_distinct_keys_distinct_seeds(self):
        assert _derive_seed(0, "a") != _derive_seed(0, "b")
        assert _derive_seed(0, "a") != _derive_seed(1, "a")

    def test_in_uint63_range(self):
        s = _derive_seed(123, "some-video-id")
        assert 0 <= s < (1 << 63)


# --------------------------------------------------------------------------- #
# DPP greedy primitive vs naive greedy
# --------------------------------------------------------------------------- #


def _logdet(kernel: torch.Tensor, subset: list[int]) -> float:
    if not subset:
        return 0.0
    idx = torch.tensor(sorted(subset))
    return float(torch.logdet(kernel[idx][:, idx]))


def _naive_greedy(kernel: torch.Tensor, k: int, condition_on: int | None = None) -> list[int]:
    """Textbook greedy MAP: pick argmax marginal logdet gain each step."""
    n = kernel.shape[0]
    chosen: list[int] = []
    if condition_on is not None:
        chosen.append(condition_on)
    picks: list[int] = []
    while len(picks) < k:
        base = _logdet(kernel, chosen)
        best_j, best_gain = None, float("-inf")
        for j in range(n):
            if j in chosen:
                continue
            gain = _logdet(kernel, chosen + [j]) - base
            if gain > best_gain:
                best_gain, best_j = gain, j
        if best_j is None:
            break
        chosen.append(best_j)
        picks.append(best_j)
    return picks


def _well_conditioned(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, n, generator=g, dtype=torch.float64)
    return a @ a.transpose(0, 1) + n * torch.eye(n, dtype=torch.float64)


class TestDppGreedy:
    def test_matches_naive_greedy(self):
        kernel = _well_conditioned(6, seed=1)
        for k in (1, 2, 3, 4):
            fast, _ = _dpp_greedy(kernel, k)
            assert fast == _naive_greedy(kernel, k)

    def test_condition_on_matches_naive(self):
        kernel = _well_conditioned(6, seed=2)
        for cond in (0, 3, 5):
            fast, _ = _dpp_greedy(kernel, 3, condition_on=cond)
            assert cond not in fast
            assert fast == _naive_greedy(kernel, 3, condition_on=cond)

    def test_gains_are_logdet_curve(self):
        kernel = _well_conditioned(5, seed=3)
        picks, gains = _dpp_greedy(kernel, 3)
        # cumulative gains == logdet of the selected prefix.
        cum = 0.0
        for m in range(len(picks)):
            cum += gains[m]
            assert cum == pytest.approx(_logdet(kernel, picks[: m + 1]), abs=1e-6)

    def test_duplicate_rows_no_raise_distinct(self):
        # Rank-deficient kernel (two identical rows) + no jitter: greedy must
        # still return k distinct indices via the relevance fill, without raising.
        base = _well_conditioned(4, seed=4)
        base[1] = base[0]
        base[:, 1] = base[:, 0]
        picks, _ = _dpp_greedy(base, 3)
        assert len(picks) == 3
        assert len(set(picks)) == 3

    def test_rank1_kernel_exhausts_gain_and_fills(self):
        # A genuinely rank-1 pool (e.g. a static clip: every frame embedding
        # identical). Greedy earns exactly one real marginal gain, then the fill
        # branch must complete to k with log(_MIN_GAIN) markers rather than
        # returning < k. This is the path the duplicate-rows test does NOT reach
        # (its kernel stays full-rank), so it pins _dpp_greedy's fill loop.
        kernel = torch.ones(4, 4, dtype=torch.float64)  # rank 1, PSD
        picks, gains = _dpp_greedy(kernel, 3)
        assert len(picks) == 3
        assert len(set(picks)) == 3
        # One real gain, then two fill markers.
        fill = [g for g in gains if g == pytest.approx(math.log(fsmod._MIN_GAIN))]
        assert len(fill) == 2

    def test_rank1_fill_orders_by_descending_diagonal(self):
        # The fill picks in descending diagonal (relevance) order after the first
        # real pick (argmax diagonal). Diagonal = [1,4,2,3]: first pick is index 1
        # (largest), fill then takes 3 then 2.
        d = torch.tensor([1.0, 4.0, 2.0, 3.0], dtype=torch.float64)
        kernel = torch.outer(d.sqrt(), d.sqrt())  # rank 1, diagonal == d
        picks, _ = _dpp_greedy(kernel, 3)
        assert picks == [1, 3, 2]


# --------------------------------------------------------------------------- #
# mDP3 segment DP
# --------------------------------------------------------------------------- #


def _exhaustive_two_segment(kernel: torch.Tensor, k: int, m: int) -> list[int]:
    """Exact optimum for a two-segment instance (n <= 2m).

    With two segments the lazy DP has no pruning loss: each within-segment
    selection is a deterministic greedy, so enumerating allocations (k1, k2) and
    taking the argmax total score is the exact optimum the DP should reach.
    """
    n = kernel.shape[0]
    assert n <= 2 * m
    seg1 = list(range(0, min(m, n)))
    seg2 = list(range(m, n))
    best_score, best_trace = float("-inf"), []
    for k1 in range(0, min(len(seg1), k) + 1):
        k2 = k - k1
        if k2 < 0 or k2 > len(seg2):
            continue
        sub1 = kernel[seg1][:, seg1]
        picks1, gains1 = _dpp_greedy(sub1, k1)
        trace1 = [seg1[p] for p in picks1]
        cond = trace1[-1] if trace1 else None
        idx2 = seg2 + ([cond] if cond is not None else [])
        sub2 = kernel[idx2][:, idx2]
        cond_pos = len(seg2) if cond is not None else None
        picks2, gains2 = _dpp_greedy(sub2, k2, condition_on=cond_pos)
        trace2 = [idx2[p] for p in picks2]
        total = sum(gains1) + sum(gains2)
        if total > best_score:
            best_score, best_trace = total, sorted(trace1 + trace2)
    return best_trace


class TestMdp3SegmentDp:
    def test_matches_exhaustive_two_segments(self):
        # n = 2m so there are exactly two segments; DP must equal the exact optimum.
        m = 3
        kernel = _well_conditioned(2 * m, seed=5)
        for k in (2, 3, 4):
            assert sorted(_mdp3_segment_dp(kernel, k, m)) == _exhaustive_two_segment(kernel, k, m)

    def test_matches_exhaustive_ragged_final_segment(self):
        # n=5, segment_size=3 -> segments [0,1,2] and [3,4]: the trailing block is
        # PARTIAL (n % segment_size != 0), the common real case since
        # candidate_frames rarely divides evenly. The min-clamp on seg_bounds must
        # handle it, and the two-segment DP is still exactly checkable.
        m = 3
        kernel = _well_conditioned(5, seed=7)
        for k in (2, 3):
            assert sorted(_mdp3_segment_dp(kernel, k, m)) == _exhaustive_two_segment(kernel, k, m)

    def test_exact_k_distinct_sorted_in_range(self):
        kernel = _well_conditioned(12, seed=6)
        out = sorted(_mdp3_segment_dp(kernel, 5, 4))
        assert len(out) == 5
        assert len(set(out)) == 5
        assert all(0 <= i < 12 for i in out)

    def test_spreads_across_segments_when_better(self):
        # Two segments, each with one strongly-relevant, mutually-diverse frame.
        # Sequentiality/diversity should place one pick in each segment.
        emb = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],  # seg1 relevant
                [1.0, 0.05, 0.0, 0.0],  # seg1 near-dup of frame0
                [0.0, 0.0, 1.0, 0.0],  # seg2 relevant, diverse from seg1
                [0.0, 0.0, 1.0, 0.05],  # seg2 near-dup of frame2
            ]
        )
        q = torch.tensor([1.0, 0.0, 1.0, 0.0])  # aligned with both regions
        sel = MDP3Selector(FakeScorer(emb, q), segment_size=2, candidate_frames=4)
        idx = sel.select(_dummy_frames(4), [0, 1, 2, 3], "q", 2, seed_key="v")
        assert idx[0] in (0, 1)
        assert idx[1] in (2, 3)


# --------------------------------------------------------------------------- #
# mDP3 full selector
# --------------------------------------------------------------------------- #


class TestMdp3Selector:
    def test_relevance_steering(self):
        # Orthonormal frames (uniform diversity); query aligned with 0,1 -> those
        # two are the highest-relevance and get selected.
        emb = torch.eye(4)
        q = F.normalize(torch.tensor([1.0, 1.0, 0.0, 0.0]), dim=0)
        sel = MDP3Selector(FakeScorer(emb, q), segment_size=0, candidate_frames=4)
        idx = sel.select(_dummy_frames(4), [0, 1, 2, 3], "q", 2, seed_key="v")
        assert set(idx) == {0, 1}

    def test_diversity_avoids_duplicates(self):
        # Frames 0 and 1 identical; frame 2 distinct; equal relevance (query
        # orthogonal). Selection should include the diverse frame 2, not both dups.
        emb = torch.stack(
            [torch.tensor([1.0, 0, 0]), torch.tensor([1.0, 0, 0]), torch.tensor([0.0, 1, 0])]
        )
        q = torch.tensor([0.0, 0.0, 1.0])  # orthogonal -> equal relevance
        sel = MDP3Selector(FakeScorer(emb, q), segment_size=0, candidate_frames=3)
        idx = sel.select(_dummy_frames(3), [0, 1, 2], "q", 2, seed_key="v")
        assert 2 in idx
        assert idx != [0, 1]

    def test_segment_zero_equals_single_greedy(self):
        emb = torch.randn(8, 16, generator=torch.Generator().manual_seed(9))
        q = torch.randn(16, generator=torch.Generator().manual_seed(10))
        scorer = FakeScorer(emb, q)
        seg0 = MDP3Selector(scorer, segment_size=0).select(_dummy_frames(8), list(range(8)), "q", 3)
        big = MDP3Selector(scorer, segment_size=100).select(
            _dummy_frames(8), list(range(8)), "q", 3
        )
        assert seg0 == big  # segment_size >= n also degenerates to one segment

    def test_lambda_trades_relevance_for_diversity(self):
        # frames 0,1 near-duplicate AND highly relevant; frame 2 diverse but less
        # relevant. D = diag(r ** (1/lam)): small lambda amplifies the relevance
        # ratio so the two relevant near-dups win; large lambda flattens D so
        # diversity dominates and the diverse frame is chosen. This pins the sign
        # of the exponent (1/lam) — inverting it flips the two regimes.
        emb = torch.tensor([[1.0, 0.0, 0.0], [0.985, 0.174, 0.0], [0.3, 0.0, 0.954]])
        q = torch.tensor([1.0, 0.0, 0.0])

        def pick(lam: float) -> list[int]:
            sel = MDP3Selector(FakeScorer(emb, q), segment_size=0, candidate_frames=3, lam=lam)
            return sel.select(_dummy_frames(3), [0, 1, 2], "q", 2, seed_key="v")

        assert 2 not in pick(0.05)  # relevance dominates -> the two near-dups {0,1}
        assert 2 in pick(3.0)  # diversity dominates -> relevant + diverse {0,2}

    def test_cosine_kernel_runs(self):
        emb = torch.randn(6, 8, generator=torch.Generator().manual_seed(11))
        q = torch.randn(8, generator=torch.Generator().manual_seed(12))
        sel = MDP3Selector(FakeScorer(emb, q), segment_size=0, kernel="cosine")
        idx = sel.select(_dummy_frames(6), list(range(6)), "q", 3, seed_key="v")
        assert len(idx) == 3
        assert idx == sorted(idx)

    def test_cosine_kernel_handles_negative_similarity(self):
        # An anti-aligned query gives negative raw cosine; the (1 + s) / 2 shift
        # (and clamp_min on relevance) must keep L PSD and r > 0 so the selection
        # stays finite. Breaking the shift would yield NaN relevance -> a crash or
        # garbage picks; this pins it.
        emb = torch.eye(4)
        q = torch.tensor([-1.0, -1.0, 0.0, 0.0])
        sel = MDP3Selector(FakeScorer(emb, q), segment_size=0, candidate_frames=4, kernel="cosine")
        idx = sel.select(_dummy_frames(4), [0, 1, 2, 3], "q", 2, seed_key="v")
        assert len(idx) == 2
        assert len(set(idx)) == 2
        assert idx == sorted(idx)

    def test_rejects_bad_kernel(self):
        with pytest.raises(ValueError, match="rkhs.*cosine"):
            MDP3Selector(FakeScorer(torch.eye(2), torch.ones(2)), kernel="bogus")


# --------------------------------------------------------------------------- #
# Scorer: chunk-and-pool, failure, pickling
# --------------------------------------------------------------------------- #


class _FakeTokenizer:
    def __init__(self, max_len: int) -> None:
        self.model_max_length = max_len

    def __call__(self, text, add_special_tokens=True, truncation=True):  # noqa: ANN001
        n = len(text)
        extra = 2 if add_special_tokens else 0  # pretend 2 special tokens
        return {"input_ids": list(range(n + extra))}

    def decode(self, ids, skip_special_tokens=True):  # noqa: ANN001
        return "x" * len(ids)


class _FakeProcessor:
    def __init__(self, max_len: int) -> None:
        self.tokenizer = _FakeTokenizer(max_len)
        self.seen_shapes: list[tuple[int, int]] = []

    def __call__(
        self,
        text=None,
        padding=None,
        max_length=None,
        truncation=None,  # noqa: ANN001
        return_tensors=None,
        images=None,
    ):
        input_ids = torch.zeros(len(text), max_length, dtype=torch.long)
        self.seen_shapes.append(tuple(input_ids.shape))
        return {"input_ids": input_ids}


class _FakeTextModel:
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def get_text_features(self, input_ids=None, **_):  # noqa: ANN001
        b = input_ids.shape[0]
        feats = torch.zeros(b, self.dim)
        for i in range(b):
            feats[i, i % self.dim] = 1.0  # distinct one-hot per chunk
        return feats


class TestScorerQueryChunking:
    def _scorer(self, max_len: int, dim: int) -> FrameQueryScorer:
        scorer = FrameQueryScorer("siglip2", "fake/path")
        scorer._model = _FakeTextModel(dim)
        scorer._processor = _FakeProcessor(max_len)
        scorer._text_max_len = max_len
        return scorer

    def test_short_query_single_pass(self):
        scorer = self._scorer(max_len=8, dim=4)
        out = scorer.embed_query("hi")  # 2 tokens <= 8
        assert scorer._processor.seen_shapes == [(1, 8)]
        assert out.shape == (4,)
        assert torch.allclose(out, torch.tensor([1.0, 0, 0, 0]))

    def test_long_query_chunks_are_max_len_and_mean_pooled(self):
        scorer = self._scorer(max_len=4, dim=8)
        # 10 raw tokens; reserve 2 for specials -> chunk_size 2 -> 5 chunks.
        out = scorer.embed_query("x" * 10)
        assert len(scorer._processor.seen_shapes) == 1
        n_chunks, width = scorer._processor.seen_shapes[0]
        assert n_chunks == 5
        assert width == 4  # every chunk padded to the text max length
        expected = F.normalize(
            torch.stack([F.one_hot(torch.tensor(i), 8).float() for i in range(5)]).mean(0), dim=0
        )
        assert torch.allclose(out, expected, atol=1e-6)


class TestPooledFeatures:
    """get_*_features may return a tensor OR a model-output object (SigLIP2 on
    transformers 5.x returns BaseModelOutputWithPooling) — unwrap both."""

    class _Out:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    def test_tensor_passthrough(self):
        t = torch.randn(3, 8)
        assert _pooled_features(t) is t

    def test_pooler_output(self):
        po = torch.randn(2, 8)
        out = self._Out(pooler_output=po, last_hidden_state=torch.randn(2, 5, 8))
        assert torch.equal(_pooled_features(out), po)

    def test_image_embeds(self):
        emb = torch.randn(2, 8)
        assert torch.equal(_pooled_features(self._Out(image_embeds=emb)), emb)

    def test_last_hidden_state_mean_fallback(self):
        lhs = torch.randn(2, 5, 8)
        out = self._Out(last_hidden_state=lhs)
        assert torch.allclose(_pooled_features(out), lhs.mean(dim=1))

    def test_unknown_raises(self):
        with pytest.raises(TypeError, match="Cannot extract pooled features"):
            _pooled_features(object())


def _stub_scorer(
    image_size: int,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    model: object | None = None,
    processor: object | None = None,
) -> FrameQueryScorer:
    """A FrameQueryScorer 'loaded' with stubs: no HF, no network."""
    scorer = FrameQueryScorer("siglip2", "fake/path")
    scorer._model = model if model is not None else object()
    scorer._processor = processor if processor is not None else object()
    scorer._text_max_len = 64
    scorer.preprocess = ScorerPreprocess(image_size=image_size, image_mean=mean, image_std=std)
    return scorer


class TestPreprocessFrameTensors:
    def setup_method(self):
        import logging

        self._kf_logger = logging.getLogger("kempnerforge")
        self._old_propagate = self._kf_logger.propagate
        self._kf_logger.propagate = True

    def teardown_method(self):
        self._kf_logger.propagate = self._old_propagate

    def test_default_config_matches_normalize_frames_exactly(self):
        from kempnerforge.data.vlm_dataset import (
            DEFAULT_IMAGE_MEAN,
            DEFAULT_IMAGE_STD,
            normalize_frames,
        )

        # Matching geometry + default stats: bit-identical to the dataset
        # normalization (the verified default SigLIP2@224 / frame_size=224 path).
        scorer = _stub_scorer(8, DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD)
        g = torch.Generator().manual_seed(0)
        u8 = torch.randint(0, 256, (5, 3, 8, 8), generator=g, dtype=torch.uint8)
        out = scorer.preprocess_frame_tensors(u8)
        assert torch.equal(out, normalize_frames(u8, DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD))
        assert not scorer._warned_resize  # no resize on the matched path

    def test_mismatched_size_interpolates_and_applies_stats(self, caplog):
        # CLIP-like stats + a scorer resolution different from the input.
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        scorer = _stub_scorer(16, mean, std)
        g = torch.Generator().manual_seed(1)
        u8 = torch.randint(0, 256, (4, 3, 8, 8), generator=g, dtype=torch.uint8)
        with caplog.at_level("WARNING"):
            out = scorer.preprocess_frame_tensors(u8)
            scorer.preprocess_frame_tensors(u8)  # warn-once
        assert out.shape == (4, 3, 16, 16)
        x = F.interpolate(u8.float() / 255.0, size=(16, 16), mode="bicubic", antialias=True)
        mean_t = torch.tensor(mean).view(3, 1, 1)
        std_t = torch.tensor(std).view(3, 1, 1)
        assert torch.equal(out, (x - mean_t) / std_t)
        assert sum("resize" in r.message.lower() for r in caplog.records) == 1

    def test_missing_record_raises(self):
        scorer = FrameQueryScorer("siglip2", "fake/path")
        scorer._model = object()  # "loaded" but without a preprocess record
        with pytest.raises(RuntimeError, match="preprocessing record"):
            scorer.preprocess_frame_tensors(torch.zeros(1, 3, 8, 8, dtype=torch.uint8))


class TestValidateFrameGeometry:
    def setup_method(self):
        import logging

        self._kf_logger = logging.getLogger("kempnerforge")
        self._old_propagate = self._kf_logger.propagate
        self._kf_logger.propagate = True

    def teardown_method(self):
        self._kf_logger.propagate = self._old_propagate

    def test_match_logs_info(self, caplog):
        sel = TopKSelector(_stub_scorer(224, (0.5,) * 3, (0.5,) * 3))
        with caplog.at_level("INFO"):
            sel.validate_frame_geometry(224)
        assert any(r.levelname == "INFO" and "matches" in r.message for r in caplog.records)
        assert not any(r.levelname == "WARNING" for r in caplog.records)

    def test_mismatch_warns_never_raises(self, caplog):
        sel = TopKSelector(_stub_scorer(224, (0.5,) * 3, (0.5,) * 3))
        with caplog.at_level("WARNING"):
            sel.validate_frame_geometry(336)  # must not raise
        assert any(r.levelname == "WARNING" and "resize" in r.message for r in caplog.records)

    def test_requires_ensure_loaded(self):
        sel = TopKSelector(FrameQueryScorer("siglip2", "fake/path"))
        with pytest.raises(RuntimeError, match="ensure_loaded"):
            sel.validate_frame_geometry(224)


class _FakeImageModel:
    """get_image_features as a deterministic per-frame function of the pixels."""

    def get_image_features(self, pixel_values=None):  # noqa: ANN001
        return pixel_values.float().mean(dim=(2, 3))  # (b, 3)


class _FakeImageProcessor:
    """Stacks tensor 'images' into a pixel_values batch (no resize/normalize)."""

    def __call__(self, images=None, return_tensors=None):  # noqa: ANN001
        return {"pixel_values": torch.stack(list(images))}


class TestEmbedChunkParity:
    """embed_frames / embed_frame_tensors / embed_uint8_frames all route through
    the one chunk helper and agree on identical inputs (guards the refactor)."""

    def _scorer(self) -> FrameQueryScorer:
        return _stub_scorer(
            4,
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
            model=_FakeImageModel(),
            processor=_FakeImageProcessor(),
        )

    def test_frames_vs_frame_tensors(self):
        # > 2x _EMBED_BATCH frames so the chunk loop runs multiple times.
        n = 2 * fsmod._EMBED_BATCH + 5
        g = torch.Generator().manual_seed(2)
        frames = [torch.rand(3, 4, 4, generator=g) for _ in range(n)]
        scorer = self._scorer()
        via_pil_path = scorer.embed_frames(frames)
        via_tensor_path = scorer.embed_frame_tensors(torch.stack(frames))
        assert via_pil_path.shape == (n, 3)
        assert torch.equal(via_pil_path, via_tensor_path)

    def test_uint8_path_equals_preprocess_then_tensor_path(self):
        g = torch.Generator().manual_seed(3)
        u8 = torch.randint(0, 256, (10, 3, 4, 4), generator=g, dtype=torch.uint8)
        scorer = self._scorer()
        assert torch.equal(
            scorer.embed_uint8_frames(u8),
            scorer.embed_frame_tensors(scorer.preprocess_frame_tensors(u8)),
        )

    def test_output_is_normalized_float32_cpu(self):
        scorer = self._scorer()
        out = scorer.embed_frame_tensors(torch.rand(3, 3, 4, 4))
        assert out.dtype == torch.float32
        assert out.device.type == "cpu"
        assert torch.allclose(out.norm(dim=-1), torch.ones(3), atol=1e-6)


class TestScorerLifecycle:
    def test_load_failure_raises_scorer_unavailable(self, monkeypatch):
        fake_transformers = types.ModuleType("transformers")

        class _AutoModel:
            @staticmethod
            def from_pretrained(path):  # noqa: ANN001
                raise OSError("offline / not cached")

        class _AutoProcessor:
            @staticmethod
            def from_pretrained(path):  # noqa: ANN001
                return object()

        fake_transformers.AutoModel = _AutoModel
        fake_transformers.AutoProcessor = _AutoProcessor
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
        scorer = FrameQueryScorer("siglip2", "fake/path")
        with pytest.raises(ScorerUnavailableError, match="HF_HOME"):
            scorer.ensure_loaded()

    def test_ensure_loaded_is_idempotent(self):
        scorer = FrameQueryScorer("clip", "fake/path")
        scorer._model = object()  # simulate a loaded model
        scorer.ensure_loaded()  # must not attempt a reload (no transformers import)

    def test_selector_ensure_loaded_delegates_to_scorer(self):
        calls = []

        class _Scorer(FakeScorer):
            def ensure_loaded(self):
                calls.append(True)

        TopKSelector(_Scorer(torch.eye(2), torch.ones(2))).ensure_loaded()
        assert calls == [True]

    def test_rejects_bad_scorer_type(self):
        with pytest.raises(ValueError, match="Unknown scorer"):
            FrameQueryScorer("bogus", "p")

    def test_rejects_empty_path(self):
        with pytest.raises(ValueError, match="non-empty"):
            FrameQueryScorer("siglip2", "")


# --------------------------------------------------------------------------- #
# Registry + build + select_video_frames
# --------------------------------------------------------------------------- #


class TestFrameSelectorRegistry:
    @pytest.fixture(autouse=True)
    def _restore_frame_selector_registry(self):
        # These tests register throwaway selectors into the process-global
        # registry. Without teardown a second run in the same interpreter
        # (pytest-repeat, a --lf retry, a double import) makes the first
        # register_frame_selector(...) raise "already registered" before the
        # test's own pytest.raises, flipping it to an unexpected error. Snapshot
        # and restore the frame_selector store around each test.
        store = registry._get_store("frame_selector")
        before = dict(store)
        yield
        store.clear()
        store.update(before)

    def test_builtins_registered(self):
        for name in ("topk", "qframe", "mdp3"):
            assert name in registry.list_frame_selectors()

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="frame_selector"):
            registry.get_frame_selector("does-not-exist")

    def test_double_register_raises(self):
        @registry.register_frame_selector("dup_smoke_selector")
        def _a(scorer, **_):  # noqa: ANN001
            return TopKSelector(scorer)

        with pytest.raises(ValueError, match="already registered"):

            @registry.register_frame_selector("dup_smoke_selector")
            def _b(scorer, **_):  # noqa: ANN001
                return TopKSelector(scorer)

    def test_no_isinstance_ladder(self):
        # A brand-new selector registered from outside builds via the same
        # dispatch, with no core edit.
        sentinel = {}

        @registry.register_frame_selector("ladder_smoke_selector")
        def _build(scorer, **kwargs):  # noqa: ANN001
            sentinel["kwargs"] = kwargs
            return TopKSelector(scorer)

        class _Cfg:
            type = "ladder_smoke_selector"
            scorer = "siglip2"
            scorer_path = "fake/path"

            def extra_kwargs(self):
                return {"candidate_frames": 64, "candidate_fps": 1.5}

        sel = build_frame_selector(_Cfg())
        assert isinstance(sel, TopKSelector)
        assert sentinel["kwargs"] == {"candidate_frames": 64, "candidate_fps": 1.5}


class TestBuildFrameSelector:
    def _cfg(self, **over):
        from kempnerforge.config.frame_selector import FrameSelectorConfig

        return FrameSelectorConfig(**over)

    def test_dispatches_by_type(self):
        assert isinstance(build_frame_selector(self._cfg(type="topk")), TopKSelector)
        assert isinstance(build_frame_selector(self._cfg(type="qframe")), QFrameSelector)
        assert isinstance(build_frame_selector(self._cfg(type="mdp3")), MDP3Selector)

    def test_kwargs_and_candidates_reach_selector(self):
        sel = build_frame_selector(
            self._cfg(type="mdp3", candidate_frames=64, mdp3_lambda=0.5, mdp3_segment_size=8)
        )
        assert isinstance(sel, MDP3Selector)
        assert sel.candidate_frames == 64
        assert sel.lam == 0.5
        assert sel.segment_size == 8

    def test_qframe_kwargs(self):
        sel = build_frame_selector(self._cfg(type="qframe", gumbel_tau=0.5, seed=9))
        assert isinstance(sel, QFrameSelector)
        assert sel.gumbel_tau == 0.5
        assert sel.seed == 9

    def test_device_dtype_forwarded_to_scorer(self):
        sel = build_frame_selector(self._cfg(type="topk"), device="cpu", dtype=torch.float32)
        assert sel._scorer._device == torch.device("cpu")
        assert sel._scorer._dtype == torch.float32


class _RecordingSelector(FrameSelector):
    def __init__(self, indices, **kw):  # noqa: ANN001
        super().__init__(scorer=None, **kw)
        self._indices = indices
        self.calls: list[dict] = []

    def select(self, frames, times, query, k, *, seed_key=None):  # noqa: ANN001
        self.calls.append({"n": len(frames), "query": query, "k": k, "seed_key": seed_key})
        return self._indices


class TestSelectVideoFrames:
    def test_zero_fps_uses_exact_count_clamp(self, monkeypatch):
        seen = {}

        def _decode(path, *, fps, min_frames, max_frames, sampling_policy):  # noqa: ANN001
            seen.update(fps=fps, min_frames=min_frames, max_frames=max_frames)
            frames = list(range(max_frames))
            return frames, [float(i) for i in range(max_frames)]

        monkeypatch.setattr(fsmod, "decode_video_frames", _decode)
        sel = _RecordingSelector([0, 1], candidate_frames=16, candidate_fps=0.0)
        select_video_frames("p.mp4", "q", sel, 2, seed_key="vid")
        assert seen == {"fps": 1.0, "min_frames": 16, "max_frames": 16}

    def test_positive_fps_passthrough(self, monkeypatch):
        seen = {}

        def _decode(path, *, fps, min_frames, max_frames, sampling_policy):  # noqa: ANN001
            seen.update(fps=fps, min_frames=min_frames, max_frames=max_frames)
            return list(range(max_frames)), [float(i) for i in range(max_frames)]

        monkeypatch.setattr(fsmod, "decode_video_frames", _decode)
        sel = _RecordingSelector([0], candidate_frames=32, candidate_fps=2.0)
        select_video_frames("p.mp4", "q", sel, 1)
        assert seen == {"fps": 2.0, "min_frames": 1, "max_frames": 32}

    def test_dedupe_by_time_before_select(self, monkeypatch):
        # decode returns a duplicated tail time -> deduped to distinct frames.
        def _decode(path, **k):  # noqa: ANN001
            return [10, 11, 12, 13], [0.0, 1.0, 1.0, 2.0]

        monkeypatch.setattr(fsmod, "decode_video_frames", _decode)
        sel = _RecordingSelector([0, 2], candidate_frames=8, candidate_fps=0.0)
        frames, times = select_video_frames("p.mp4", "q", sel, 2, seed_key="k")
        assert sel.calls[0]["n"] == 3  # 4 decoded (times 0,1,1,2), the dup-time frame removed
        # Deduped candidates are frames [10, 11, 13] at times [0.0, 1.0, 2.0];
        # the selector's indices [0, 2] pick the first and last of those.
        assert frames == [10, 13]
        assert times == [0.0, 2.0]

    def test_seed_key_and_query_forwarded(self, monkeypatch):
        monkeypatch.setattr(
            fsmod, "decode_video_frames", lambda path, **k: ([0, 1, 2], [0.0, 1.0, 2.0])
        )
        sel = _RecordingSelector([1], candidate_frames=8, candidate_fps=0.0)
        select_video_frames("p.mp4", "the query", sel, 1, seed_key="vid99")
        assert sel.calls[0]["query"] == "the query"
        assert sel.calls[0]["seed_key"] == "vid99"


class TestDedupeByTime:
    def test_keeps_first_occurrence(self):
        f, t = _dedupe_by_time([10, 11, 12], [0.0, 0.0, 1.0])
        assert f == [10, 12]
        assert t == [0.0, 1.0]


class TestCandidateCountMath:
    @pytest.mark.parametrize("cf", [4, 8, 16, 128])
    @pytest.mark.parametrize("duration", [0.5, 3.0, 41.0, 3600.0])
    def test_zero_fps_yields_exactly_candidate_frames(self, cf, duration):
        # The min==max clamp trick used by select_video_frames.
        ts = sample_timestamps(duration, 1.0, cf, cf)
        assert len(ts) == cf

    def test_nonpositive_duration_single_timestamp(self):
        assert sample_timestamps(0.0, 1.0, 16, 16) == [0.0]


# --------------------------------------------------------------------------- #
# Tensor-path selection (select_tensor_frames) + the main-process batch stage
# --------------------------------------------------------------------------- #


class TestSelectTensorFrames:
    """The tensor entry point must select exactly like the PIL entry point."""

    def _planted(self):
        # Distinct rows; query favors high indices so topk has a unique answer.
        return torch.eye(6), torch.tensor([1.0, 2, 3, 4, 5, 6])

    def test_parity_with_pil_select(self):
        frame_emb, query_emb = self._planted()
        times = [float(i) for i in range(6)]
        pil_idx = TopKSelector(FakeScorer(frame_emb, query_emb)).select(
            list(range(6)), times, "q", 3, seed_key="k"
        )
        tensor_idx = TopKSelector(FakeScorer(frame_emb, query_emb)).select_tensor_frames(
            torch.zeros(6, 3, 4, 4), times, "q", 3, seed_key="k"
        )
        assert tensor_idx == pil_idx == [3, 4, 5]

    def test_qframe_seed_parity_across_paths(self):
        frame_emb, query_emb = self._planted()
        times = [float(i) for i in range(6)]
        a = QFrameSelector(FakeScorer(frame_emb, query_emb), seed=7).select(
            list(range(6)), times, "q", 3, seed_key="vid|q"
        )
        b = QFrameSelector(FakeScorer(frame_emb, query_emb), seed=7).select_tensor_frames(
            torch.zeros(6, 3, 4, 4), times, "q", 3, seed_key="vid|q"
        )
        assert a == b

    def test_fallbacks_never_consult_scorer(self):
        scorer = FakeScorer(*self._planted())
        sel = TopKSelector(scorer)
        assert sel.select_tensor_frames(torch.zeros(0, 3, 4, 4), [], "q", 4) == []
        assert sel.select_tensor_frames(torch.zeros(3, 3, 4, 4), [0, 1, 2], "q", 4) == [0, 1, 2]
        assert sel.select_tensor_frames(torch.zeros(6, 3, 4, 4), list(range(6)), "", 3) == [0, 2, 4]
        assert scorer.embed_frame_tensors_calls == 0
        assert scorer.embed_query_calls == 0


def _pool_images(n: int, size: int = 8):
    from PIL import Image

    return [
        Image.new("RGB", (size, size), color=(i * 20 % 255, 7, 250 - i * 9 % 255)) for i in range(n)
    ]


class TestApplyFrameSelection:
    """The batch stage: normalize once, per-sample select, gather the clip."""

    C = 6  # candidate_frames
    K = 4  # max_frames
    SIZE = 8

    def _batch(self, queries: list[str], counts: list[int]):
        from kempnerforge.data.vlm_dataset import pil_to_uint8_tensor

        b = len(counts)
        pool = torch.zeros(b, self.C, 3, self.SIZE, self.SIZE, dtype=torch.uint8)
        times = torch.zeros(b, self.C, dtype=torch.float32)
        for i, n in enumerate(counts):
            for j, img in enumerate(_pool_images(n, self.SIZE)):
                pool[i, j] = pil_to_uint8_tensor(img, self.SIZE)
            times[i, :n] = torch.arange(n, dtype=torch.float32)
        return {
            "candidate_pixels": pool,
            "candidate_count": torch.tensor(counts, dtype=torch.long),
            "candidate_times": times,
            "query": queries,
            "seed_key": [f"clip{i}|{q}" for i, q in enumerate(queries)],
            "input_ids": torch.arange(b * 5, dtype=torch.long).view(b, 5),
            "labels": torch.full((b, 5), -100, dtype=torch.long),
        }

    def _selector(self) -> TopKSelector:
        # Query favors high indices: topk(k=4) over 6 candidates -> [2, 3, 4, 5].
        return TopKSelector(FakeScorer(torch.eye(self.C), torch.tensor([1.0, 2, 3, 4, 5, 6])))

    def test_contract_and_gather(self):
        from kempnerforge.data.vlm_dataset import (
            DEFAULT_IMAGE_MEAN,
            DEFAULT_IMAGE_STD,
            pil_to_tensor,
        )

        batch = self._batch(queries=["q", "q", "q"], counts=[6, 2, 0])
        input_ids_before = batch["input_ids"]
        out = apply_frame_selection(batch, self._selector(), self.K, torch.device("cpu"))

        # Pool keys consumed; model contract produced.
        for key in ("candidate_pixels", "candidate_count", "candidate_times", "query", "seed_key"):
            assert key not in out
        assert out["pixel_values"].shape == (3, self.K, 3, self.SIZE, self.SIZE)
        assert out["pixel_values"].dtype == torch.float32
        assert not out["pixel_values"].is_inference()  # must be usable by autograd
        assert out["frame_mask"].shape == (3, self.K)
        assert out["frame_times"].shape == (3, self.K)
        assert out["input_ids"] is input_ids_before  # text passes through untouched

        # Sample 0 (n=6 > k): scored selection [2,3,4,5]; gathered pixels are
        # bit-identical to the worker-side pil_to_tensor of those frames.
        assert out["frame_mask"][0].all()
        assert out["frame_times"][0].tolist() == [2.0, 3.0, 4.0, 5.0]
        imgs = _pool_images(6, self.SIZE)
        for slot, j in enumerate([2, 3, 4, 5]):
            expected = pil_to_tensor(imgs[j], self.SIZE, DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD)
            assert torch.equal(out["pixel_values"][0, slot], expected)

        # Sample 1 (n=2 <= k): take-all fallback, padded+masked past 2.
        assert out["frame_mask"][1].tolist() == [True, True, False, False]
        assert out["frame_times"][1].tolist() == [0.0, 1.0, 0.0, 0.0]
        assert (out["pixel_values"][1, 2:] == 0).all()

        # Sample 2 (n=0, decode failure): all-masked, all-zero.
        assert not out["frame_mask"][2].any()
        assert (out["pixel_values"][2] == 0).all()
        assert (out["frame_times"][2] == 0).all()

    def test_empty_query_uniform_stride(self):
        batch = self._batch(queries=[""], counts=[6])
        out = apply_frame_selection(batch, self._selector(), self.K, torch.device("cpu"))
        # i * n // k for n=6, k=4 -> frames at times [0, 1, 3, 4].
        assert out["frame_times"][0].tolist() == [0.0, 1.0, 3.0, 4.0]
        assert out["frame_mask"][0].all()

    def test_qframe_selection_is_batchmate_independent(self):
        # The same (pixels, query, seed_key) selects identically whether the
        # sample arrives alone or beside other samples.
        def _sel():
            return QFrameSelector(
                FakeScorer(
                    torch.randn(self.C, 16, generator=torch.Generator().manual_seed(3)),
                    torch.randn(16, generator=torch.Generator().manual_seed(4)),
                ),
                seed=11,
            )

        alone = apply_frame_selection(
            self._batch(queries=["q"], counts=[6]), _sel(), self.K, torch.device("cpu")
        )
        paired = apply_frame_selection(
            self._batch(queries=["q", "other"], counts=[6, 6]), _sel(), self.K, torch.device("cpu")
        )
        assert torch.equal(alone["frame_times"][0], paired["frame_times"][0])
        assert torch.equal(alone["pixel_values"][0], paired["pixel_values"][0])

    def test_query_embedding_cached_per_distinct_query(self):
        # Same query across the batch -> one text forward; distinct -> one each.
        sel = self._selector()
        apply_frame_selection(
            self._batch(queries=["q", "q", "q"], counts=[6, 6, 6]),
            sel,
            self.K,
            torch.device("cpu"),
        )
        assert sel._scorer.embed_query_calls == 1
        assert sel._scorer.embed_uint8_frames_calls == 3  # frames still per-sample

        sel2 = self._selector()
        apply_frame_selection(
            self._batch(queries=["a", "b", "c"], counts=[6, 6, 6]),
            sel2,
            self.K,
            torch.device("cpu"),
        )
        assert sel2._scorer.embed_query_calls == 3

    def test_fallback_samples_never_consult_scorer(self):
        # take-all (n <= k) and empty-query stride skip both towers entirely.
        sel = self._selector()
        apply_frame_selection(
            self._batch(queries=["q", ""], counts=[2, 6]), sel, self.K, torch.device("cpu")
        )
        assert sel._scorer.embed_uint8_frames_calls == 0
        assert sel._scorer.embed_query_calls == 0

    def test_accepts_any_pool_capacity(self):
        # C3: candidate_pixels capacity C is the batch max, not candidate_frames —
        # it may be as small as 1. Behavior must match the padded-capacity batch.
        from kempnerforge.data.vlm_dataset import pil_to_uint8_tensor

        imgs = _pool_images(1, self.SIZE)
        pool = torch.zeros(1, 1, 3, self.SIZE, self.SIZE, dtype=torch.uint8)
        pool[0, 0] = pil_to_uint8_tensor(imgs[0], self.SIZE)
        batch = {
            "candidate_pixels": pool,  # C = 1 << candidate_frames
            "candidate_count": torch.tensor([1], dtype=torch.long),
            "candidate_times": torch.zeros(1, 1, dtype=torch.float32),
            "query": ["q"],
            "seed_key": ["clip0|q"],
        }
        out = apply_frame_selection(batch, self._selector(), self.K, torch.device("cpu"))
        assert out["pixel_values"].shape == (1, self.K, 3, self.SIZE, self.SIZE)
        assert out["frame_mask"][0].tolist() == [True, False, False, False]

        # Scored path with counts < C < candidate-frames-sized pools also works.
        wide = self._batch(queries=["q"], counts=[5])  # C=6, n=5 > k=4 -> scored
        out2 = apply_frame_selection(wide, self._selector(), self.K, torch.device("cpu"))
        assert out2["frame_mask"][0].all()
        # Query favors high indices: topk over 5 candidates -> [1, 2, 3, 4].
        assert out2["frame_times"][0].tolist() == [1.0, 2.0, 3.0, 4.0]


class TestApplyFrameSelectionIsolation:
    """Per-sample failure isolation: one bad sample must not kill the batch."""

    C = 6
    K = 4
    SIZE = 8

    def _batch(self, queries, counts):  # noqa: ANN001
        maker = TestApplyFrameSelection()
        batch = maker._batch(queries, counts)
        batch["labels"] = torch.full((len(counts), 5), 7, dtype=torch.long)
        return batch

    def _failing_selector(self, fail_on_call: int) -> TopKSelector:
        scorer = FakeScorer(torch.eye(self.C), torch.tensor([1.0, 2, 3, 4, 5, 6]))
        original = scorer.embed_uint8_frames

        def _flaky(frames_u8):  # noqa: ANN001
            if scorer.embed_uint8_frames_calls + 1 == fail_on_call:
                scorer.embed_uint8_frames_calls += 1
                raise RuntimeError("transient scorer failure")
            return original(frames_u8)

        scorer.embed_uint8_frames = _flaky
        return TopKSelector(scorer)

    def test_scoring_failure_masks_sample_and_labels(self):
        batch = self._batch(queries=["q", "q"], counts=[6, 6])
        out = apply_frame_selection(
            batch, self._failing_selector(fail_on_call=1), self.K, torch.device("cpu")
        )
        # Failed sample 0: all-masked frames, labels forced to -100.
        assert not out["frame_mask"][0].any()
        assert (out["pixel_values"][0] == 0).all()
        assert (out["labels"][0] == -100).all()
        # Batchmate 1 trains normally on the scored selection.
        assert out["frame_mask"][1].all()
        assert out["frame_times"][1].tolist() == [2.0, 3.0, 4.0, 5.0]
        assert (out["labels"][1] == 7).all()

    def test_selection_failure_masks_sample_and_labels(self):
        class _ExplodingSelector(TopKSelector):
            def _select_scored(self, frame_emb, query_emb, times, k, seed_key):  # noqa: ANN001
                raise RuntimeError("selection math failure")

        sel = _ExplodingSelector(FakeScorer(torch.eye(self.C), torch.arange(6.0)))
        batch = self._batch(queries=["q", ""], counts=[6, 6])
        out = apply_frame_selection(batch, sel, self.K, torch.device("cpu"))
        assert not out["frame_mask"][0].any()
        assert (out["labels"][0] == -100).all()
        # Sample 1 (empty query) never reaches _select_scored: uniform stride.
        assert out["frame_mask"][1].all()
        assert (out["labels"][1] == 7).all()

    def test_scorer_unavailable_still_propagates(self):
        # A scorer that never loaded is systemic misconfiguration, not a
        # per-sample data error: it must fail the run loudly.
        class _UnavailableScorer(FakeScorer):
            def embed_uint8_frames(self, frames_u8):  # noqa: ANN001
                raise ScorerUnavailableError("weights not cached")

        sel = TopKSelector(_UnavailableScorer(torch.eye(self.C), torch.arange(6.0)))
        with pytest.raises(ScorerUnavailableError):
            apply_frame_selection(
                self._batch(queries=["q"], counts=[6]), sel, self.K, torch.device("cpu")
            )


class TestMakeSeedKey:
    def test_format_and_none_query(self):
        assert make_seed_key("/v/clip.mp4", "what happens?") == "/v/clip.mp4|what happens?"
        assert make_seed_key("/v/clip.mp4", None) == "/v/clip.mp4|"
        assert make_seed_key("/v/clip.mp4", "") == "/v/clip.mp4|"


class TestSelectUint8Frames:
    """The training-parity entry point mirrors select/select_tensor_frames."""

    def test_scored_path_uses_uint8_tower_and_matches_planted_topk(self):
        scorer = FakeScorer(torch.eye(6), torch.tensor([1.0, 2, 3, 4, 5, 6]))
        sel = TopKSelector(scorer)
        pool = torch.zeros(6, 3, 8, 8, dtype=torch.uint8)
        idx = sel.select_uint8_frames(pool, [float(i) for i in range(6)], "q", 4)
        assert idx == [2, 3, 4, 5]
        assert scorer.embed_uint8_frames_calls == 1
        assert scorer.embed_frames_calls == 0

    def test_fallbacks_shared_with_select(self):
        scorer = FakeScorer(torch.eye(6), torch.arange(6.0))
        sel = TopKSelector(scorer)
        pool = torch.zeros(6, 3, 8, 8, dtype=torch.uint8)
        assert sel.select_uint8_frames(pool[:2], [0.0, 1.0], "q", 4) == [0, 1]  # take-all
        assert sel.select_uint8_frames(pool, list(range(6)), "  ", 4) == [0, 1, 3, 4]  # stride
        assert scorer.embed_uint8_frames_calls == 0

    def test_select_video_frames_frame_size_routes_through_uint8_path(self, monkeypatch):
        imgs = _pool_images(6, 8)
        monkeypatch.setattr(
            fsmod,
            "decode_candidate_pool",
            lambda path, **kw: (imgs, [float(i) for i in range(6)]),
        )
        scorer = FakeScorer(torch.eye(6), torch.tensor([1.0, 2, 3, 4, 5, 6]))
        sel = TopKSelector(scorer)
        frames, times = select_video_frames("clip.mp4", "q", sel, 4, frame_size=8)
        assert times == [2.0, 3.0, 4.0, 5.0]
        assert frames == [imgs[j] for j in (2, 3, 4, 5)]  # original PILs returned
        assert scorer.embed_uint8_frames_calls == 1  # training pixel path
        assert scorer.embed_frames_calls == 0

        # Without frame_size, the PIL/HF-processor path is used as before.
        scorer2 = FakeScorer(torch.eye(6), torch.tensor([1.0, 2, 3, 4, 5, 6]))
        sel2 = TopKSelector(scorer2)
        select_video_frames("clip.mp4", "q", sel2, 4)
        assert scorer2.embed_frames_calls == 1
        assert scorer2.embed_uint8_frames_calls == 0


class TestDecodeCandidatePool:
    def test_matches_select_video_frames_sizing(self, monkeypatch):
        seen = {}

        def _decode(path, *, fps, min_frames, max_frames, sampling_policy):  # noqa: ANN001
            seen.update(fps=fps, min_frames=min_frames, max_frames=max_frames)
            return list(range(max_frames)), [float(i) for i in range(max_frames)]

        monkeypatch.setattr(fsmod, "decode_video_frames", _decode)
        decode_candidate_pool("p.mp4", candidate_frames=16, candidate_fps=0.0)
        assert seen == {"fps": 1.0, "min_frames": 16, "max_frames": 16}
        decode_candidate_pool("p.mp4", candidate_frames=32, candidate_fps=2.0)
        assert seen == {"fps": 2.0, "min_frames": 1, "max_frames": 32}

    def test_dedupes_by_time(self, monkeypatch):
        monkeypatch.setattr(
            fsmod, "decode_video_frames", lambda path, **k: ([10, 11, 12], [0.0, 1.0, 1.0])
        )
        frames, times = decode_candidate_pool("p.mp4", candidate_frames=8)
        assert frames == [10, 11]
        assert times == [0.0, 1.0]

    def test_spec_defaults_mirror_config(self):
        from kempnerforge.config.frame_selector import FrameSelectorConfig

        cfg = FrameSelectorConfig()
        spec = CandidatePoolSpec.from_config(cfg)
        assert spec.candidate_frames == cfg.candidate_frames
        assert spec.candidate_fps == cfg.candidate_fps
        # The bare-constructor defaults mirror the config defaults too.
        assert spec == CandidatePoolSpec()

    def test_from_config_duck_typed(self):
        class _Cfg:
            candidate_frames = 32
            candidate_fps = 2.0

        spec = CandidatePoolSpec.from_config(_Cfg())
        assert spec == CandidatePoolSpec(candidate_frames=32, candidate_fps=2.0)
