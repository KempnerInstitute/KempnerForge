"""Query-aware video frame selection (the temporal "adapter" seam).

A clip is decoded to a pool of ``candidate_frames`` uniformly sampled stills,
then a ``FrameSelector`` keeps the ``k`` most informative ones *for the query*.
This is the two-stage compose that hosts training-free frame selection without
touching the model: the existing ``sampling_policy`` proposes *when* to sample
candidates; the selector picks *which* to keep. ``k`` equals ``[video].max_frames``,
so the downstream clip packing (``frames_to_clip_tensor``) and the static
``frames_per_clip`` are unchanged — a selector that returns ``<= k`` frames is
zero-padded and masked exactly like a short clip.

Selection runs on the *data* path (train dataset ``__getitem__`` and the eval
request renderer), the only scopes where decoded frames and the query text
coexist. It never runs inside the model.

Selectors shipped:

- ``"topk"`` — cosine top-k of frame/query similarity (the simplest baseline;
  Q-Frame/mDP3 ablation row "+SigLIP").
- ``"qframe"`` — Q-Frame QFS: Gumbel-Max top-k over ``softmax(sim / tau)``
  (ICCV 2025, arXiv 2506.22139). Q-Frame's Multi-Resolution Adaptation is not
  implemented here (KF vision towers are fixed-resolution; the paper disables
  MRA on such models).
- ``"mdp3"`` — Markov-DPP frame selection (arXiv 2501.02885): a conditional
  multi-Gaussian-kernel similarity matrix (query relevance + list-wise
  diversity) with greedy MAP DPP, plus a segmented Markov DP that allocates the
  budget across temporal segments for sequentiality.

Numerics: the DPP math runs in float64 on CPU with an in-kernel jitter for
positive-definiteness. Scoring uses a frozen HuggingFace dual encoder
(SigLIP2 / CLIP) whose weights load lazily on first use. Embeddings always
return as L2-normalized float32 CPU tensors, so the selection math is
independent of the scorer's device/dtype.

Scoring runs on GPU in the process that owns the accelerator; it never runs in
dataloader workers (CUDA is unusable in forked workers, and a CPU so400m
forward is ~80x slower than an H200 one):

- Training: workers decode the candidate pool (``decode_candidate_pool``,
  sized by ``CandidatePoolSpec``) and ship it raw; the main training process
  scores and selects per micro-batch via ``apply_frame_selection``.
- Eval: the harness is single-process, so ``select_video_frames`` runs the
  whole decode-score-select chain inline with a GPU-device scorer.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from kempnerforge.config.registry import registry
from kempnerforge.data.video_io import decode_video_frames
from kempnerforge.data.vlm_dataset import (
    DEFAULT_IMAGE_MEAN,
    DEFAULT_IMAGE_STD,
    normalize_frames,
    pil_to_uint8_tensor,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from PIL.Image import Image as PILImage

logger = logging.getLogger(__name__)

# Supported dual-encoder scorer families (image + text towers with
# get_image_features / get_text_features).
_SCORER_TYPES: tuple[str, ...] = ("siglip2", "clip")
# mDP3 multi-Gaussian kernel bandwidths: alpha_u = 2^i (paper appendix B.3).
_MDP3_LOG2_ALPHAS: tuple[float, ...] = (-3.0, -2.0, 0.0, 1.0, 2.0)
# Frames per vision-encoder forward when embedding candidates.
_EMBED_BATCH: int = 32
# Marginal-gain floor: greedy stops (falls to relevance fill) below this, and
# fill picks are recorded with this log-gain so the segment DP stays finite.
_MIN_GAIN: float = 1e-12
# torch.Generator seeds are uint64; keep to 63 bits to stay signed-safe.
_SEED_MASK: int = (1 << 63) - 1


class ScorerUnavailableError(RuntimeError):
    """The frame-selection scorer weights could not be loaded.

    Raised on a lazy-load failure (e.g. an un-prefetched model on an offline
    node). This is systemic misconfiguration, not a per-sample data error, so
    callers must let it propagate and fail the run loudly rather than degrade
    every sample to an empty clip.
    """


def uniform_stride_indices(n: int, k: int) -> list[int]:
    """Uniform-stride pick of ``k`` of ``n`` indices: ``i * n // k``.

    The canonical no-query fallback, shared between the main-process fallback
    (``_fallback_indices``) and worker-side pool pre-striding so both sides
    agree on which frames a stride keeps.
    """
    return [i * n // k for i in range(k)]


def make_seed_key(path: str, query: str | None) -> str:
    """Canonical per-sample seed key: ``"<path>|<query>"``.

    The single source of the train/eval seed-key format. Pool-mode workers
    (``VideoQADataset._candidate_pool_sample``) and the eval adapter both
    derive the stochastic-selection seed from this string; hand-rolling the
    f-string at either site risks silently de-synchronizing QFrame's Gumbel
    draws — and therefore the selected frames — between training and eval.
    """
    return f"{path}|{query or ''}"


def _derive_seed(base_seed: int, key: str | None) -> int:
    """Deterministic per-sample seed from ``(base_seed, key)``.

    Uses SHA-256 (never the builtin ``hash``, which is salted by
    ``PYTHONHASHSEED``) so a given ``(seed, videoid/path)`` reproduces the same
    draw across processes, ranks, and resumes. ``key=None`` returns the base
    seed. Per-epoch redraw (mixing the epoch into ``key``) is future work.
    """
    if key is None:
        return int(base_seed) & _SEED_MASK
    digest = hashlib.sha256(f"{base_seed}|{key}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & _SEED_MASK


def _pooled_features(out: Any) -> torch.Tensor:
    """Extract a pooled ``(batch, dim)`` embedding from a HF ``get_*_features`` result.

    Some model/transformers-version combos return the pooled tensor directly;
    others (e.g. SigLIP2 ``get_image_features`` on transformers 5.x) return a
    ``BaseModelOutputWithPooling``. Unwrap to ``pooler_output`` /
    ``image_embeds`` / ``text_embeds``, falling back to a mean over
    ``last_hidden_state`` so the scorer is robust across versions.
    """
    if torch.is_tensor(out):
        return out
    for attr in ("pooler_output", "image_embeds", "text_embeds"):
        value = getattr(out, attr, None)
        if value is not None:
            return value
    last_hidden = getattr(out, "last_hidden_state", None)
    if last_hidden is not None:
        return last_hidden.mean(dim=1)
    if isinstance(out, (tuple, list)) and out and torch.is_tensor(out[0]):
        return out[0]
    raise TypeError(f"Cannot extract pooled features from {type(out).__name__}")


@dataclass(frozen=True)
class ScorerPreprocess:
    """Frozen preprocessing geometry/stats derived from the loaded scorer.

    ``image_size`` is the vision tower's true input resolution (primary source:
    ``model.config.vision_config.image_size`` — the position-embedding
    resolution; fallback: the processor's ``crop_size``/``size`` dict).
    ``image_mean``/``image_std`` come from the processor's image processor,
    falling back to ``DEFAULT_IMAGE_MEAN``/``DEFAULT_IMAGE_STD``.
    """

    image_size: int
    image_mean: tuple[float, ...]
    image_std: tuple[float, ...]


class FrameQueryScorer:
    """Frozen HuggingFace dual encoder scoring frames against a text query.

    Weights load lazily on the first ``embed_*`` call, so constructing the
    scorer is cheap and network-free; call ``ensure_loaded`` at startup to
    fail fast instead. The ``embed_*`` methods return L2-normalized float32
    CPU tensors so downstream similarity math is scorer-dtype-independent.
    """

    def __init__(
        self,
        scorer: str,
        path: str,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if scorer not in _SCORER_TYPES:
            raise ValueError(f"Unknown scorer {scorer!r}; choose from {_SCORER_TYPES}")
        if not path:
            raise ValueError("frame-selection scorer path must be non-empty")
        self._scorer = scorer
        self._path = path
        self._device = torch.device(device)
        self._dtype = dtype
        self._model: Any = None
        self._processor: Any = None
        self._text_max_len: int | None = None
        self.preprocess: ScorerPreprocess | None = None
        self._warned_resize = False

    def ensure_loaded(self) -> None:
        """Load the scorer weights now (idempotent).

        The ``embed_*`` methods call this lazily, but a long-lived owner (the
        training loop) should call it once at startup so a missing/broken
        scorer fails the run immediately rather than at the first sample.
        """
        if self._model is not None:
            return
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as e:  # pragma: no cover - only without transformers
            raise ImportError(
                "Frame selection requires `transformers` for the scorer. Install it "
                "or omit the [frame_selector] section."
            ) from e
        # Any load/config failure is systemic misconfiguration, not a per-sample
        # data error: a gated/incompatible repo raises ValueError, an offline node
        # raises OSError, a broken checkpoint raises assorted errors. Wrap them all
        # (except ImportError, handled above) as ScorerUnavailableError so the run
        # fails loudly instead of __getitem__'s broad except degrading every clip.
        # Derive everything into locals and publish _model/_processor/_text_max_len
        # only once all of it succeeds, so a mid-load failure can't leave a
        # half-initialized object that the `self._model is not None` guard freezes.
        try:
            model = AutoModel.from_pretrained(self._path)
            processor = AutoProcessor.from_pretrained(self._path)
            model.eval().requires_grad_(False)
            model.to(device=self._device, dtype=self._dtype)
            text_cfg = getattr(model.config, "text_config", None)
            max_len = getattr(text_cfg, "max_position_embeddings", None)
            if not max_len:
                max_len = getattr(getattr(processor, "tokenizer", None), "model_max_length", 64)
            # SentencePiece/BPE model_max_length can be a huge sentinel; cap defensively.
            text_max_len = int(max_len) if max_len and max_len < 100_000 else 64
            # Derive the scorer's own preprocessing: the vision tower's true
            # input resolution (position-embedding size) and the processor's
            # normalization stats, so preprocess_frame_tensors can feed the
            # image tower exactly what it was trained on.
            vision_cfg = getattr(model.config, "vision_config", None)
            image_size = getattr(vision_cfg, "image_size", None)
            image_proc = getattr(processor, "image_processor", None)
            if not image_size:
                for attr in ("crop_size", "size"):
                    value = getattr(image_proc, attr, None)
                    if isinstance(value, dict):
                        image_size = (
                            value.get("height") or value.get("shortest_edge") or value.get("width")
                        )
                    elif isinstance(value, int):
                        image_size = value
                    if image_size:
                        break
            if not image_size:
                raise ValueError(
                    "could not derive the scorer input resolution from "
                    "model.config.vision_config.image_size or the processor's crop_size/size"
                )
            mean = getattr(image_proc, "image_mean", None) or DEFAULT_IMAGE_MEAN
            std = getattr(image_proc, "image_std", None) or DEFAULT_IMAGE_STD
            preprocess = ScorerPreprocess(
                image_size=int(image_size),
                image_mean=tuple(float(v) for v in mean),
                image_std=tuple(float(v) for v in std),
            )
        except Exception as e:  # noqa: BLE001 - any load/config failure is systemic
            raise ScorerUnavailableError(
                f"Could not load frame-selection scorer {self._path!r}: {e}. "
                "Prefetch it once on a networked node (set HF_HOME to the shared cache and run "
                f'`python -c "from transformers import AutoModel, AutoProcessor; '
                f"AutoModel.from_pretrained('{self._path}'); "
                f"AutoProcessor.from_pretrained('{self._path}')\"`), unset HF_HUB_OFFLINE, or "
                "check the repo id / weights are compatible."
            ) from e
        self._text_max_len = text_max_len
        self.preprocess = preprocess
        self._processor = processor
        self._model = model  # published last: the `_model is not None` load guard

    @torch.inference_mode()
    def _embed_pixel_chunks(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Chunked image-tower forward -> ``(n, d)`` L2-normalized float32 CPU tensor.

        The single embedding loop every image path routes through:
        ``_EMBED_BATCH`` slices -> ``get_image_features`` -> normalize ->
        per-chunk ``.cpu()``. Chunking is per call, so a sample's embeddings
        never depend on its batchmates. Callers must ``ensure_loaded`` first.
        """
        out: list[torch.Tensor] = []
        for start in range(0, pixel_values.shape[0], _EMBED_BATCH):
            chunk = pixel_values[start : start + _EMBED_BATCH].to(
                device=self._device, dtype=self._dtype
            )
            feats = _pooled_features(self._model.get_image_features(pixel_values=chunk))
            out.append(F.normalize(feats.float(), dim=-1).cpu())
        return torch.cat(out, dim=0)

    @torch.inference_mode()
    def embed_frames(self, frames: Sequence[PILImage]) -> torch.Tensor:
        """Embed candidate frames -> ``(n, d)`` L2-normalized float32 CPU tensor."""
        self.ensure_loaded()
        out: list[torch.Tensor] = []
        for start in range(0, len(frames), _EMBED_BATCH):
            batch = list(frames[start : start + _EMBED_BATCH])
            inputs = self._processor(images=batch, return_tensors="pt")
            out.append(self._embed_pixel_chunks(inputs["pixel_values"]))
        return torch.cat(out, dim=0)

    @torch.inference_mode()
    def embed_frame_tensors(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Embed pre-normalized frames -> ``(n, d)`` L2-normalized float32 CPU tensor.

        Tensor twin of ``embed_frames`` for callers that already hold frames as
        normalized ``(n, 3, H, W)`` float tensors: skips the HF processor —
        whose resize/rescale/normalize the caller's preprocessing has already
        applied — and runs the shared chunked image-tower forward.
        """
        self.ensure_loaded()
        return self._embed_pixel_chunks(pixel_values)

    def preprocess_frame_tensors(self, frames_u8: torch.Tensor) -> torch.Tensor:
        """Preprocess raw uint8 frames ``(n, 3, H, W)`` for the scorer's image tower.

        Applies the scorer-derived preprocessing (``ScorerPreprocess`` from
        ``ensure_loaded``): ``x = frames / 255`` in float32, a bicubic
        antialiased resize to the scorer's input resolution ONLY when the input
        geometry differs (warn-once), then ``(x - mean) / std`` with the
        scorer's own normalization stats. When the dataset geometry and stats
        already match the scorer (the default SigLIP2-so400m@224 +
        ``frame_size=224`` config) this is elementwise-identical to
        ``normalize_frames(frames_u8, DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD)``.
        """
        self.ensure_loaded()
        pp = self.preprocess
        if pp is None:
            raise RuntimeError(
                "Scorer preprocessing record missing: ensure_loaded() must derive it "
                "before preprocess_frame_tensors is called."
            )
        x = frames_u8.to(torch.float32) / 255.0
        size = (pp.image_size, pp.image_size)
        if tuple(x.shape[-2:]) != size:
            if not self._warned_resize:
                logger.warning(
                    "Frame-selection scoring resizes frames %sx%s -> scorer resolution %sx%s "
                    "(bicubic, antialiased). Match [video].frame_size to the scorer to skip "
                    "the resize.",
                    x.shape[-2],
                    x.shape[-1],
                    pp.image_size,
                    pp.image_size,
                )
                self._warned_resize = True
            x = F.interpolate(x, size=size, mode="bicubic", antialias=True)
        mean_t = torch.tensor(pp.image_mean, dtype=x.dtype, device=x.device).view(3, 1, 1)
        std_t = torch.tensor(pp.image_std, dtype=x.dtype, device=x.device).view(3, 1, 1)
        return (x - mean_t) / std_t

    @torch.inference_mode()
    def embed_uint8_frames(self, frames_u8: torch.Tensor) -> torch.Tensor:
        """Preprocess + embed raw uint8 frames -> ``(n, d)`` L2-normalized float32 CPU.

        ``preprocess_frame_tensors`` (scorer-derived resize/stats) followed by
        the shared chunked image-tower forward.
        """
        return self._embed_pixel_chunks(self.preprocess_frame_tensors(frames_u8))

    @torch.inference_mode()
    def embed_query(self, query: str) -> torch.Tensor:
        """Embed the query -> ``(d,)`` L2-normalized float32 CPU tensor.

        Queries longer than the text tower's context are split into
        canonically-tokenized chunks (proper special tokens + padding, never
        raw id-splitting), embedded, mean-pooled, and re-normalized (mDP3
        appendix B.3).
        """
        self.ensure_loaded()
        assert self._text_max_len is not None  # set by ensure_loaded
        tokenizer = self._processor.tokenizer
        ids = tokenizer(query, add_special_tokens=False, truncation=False)["input_ids"]
        if len(ids) <= self._text_max_len:
            emb = self._encode_text([query])
            return F.normalize(emb, dim=-1)[0]
        # Reserve room for the special tokens the tokenizer re-adds per chunk.
        with_special = tokenizer(query, add_special_tokens=True, truncation=False)["input_ids"]
        reserve = max(0, len(with_special) - len(ids))
        chunk_size = max(1, self._text_max_len - reserve)
        chunks = [ids[i : i + chunk_size] for i in range(0, len(ids), chunk_size)]
        texts = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]
        embs = self._encode_text(texts)  # (num_chunks, d)
        pooled = embs.mean(dim=0, keepdim=True)
        return F.normalize(pooled, dim=-1)[0]

    def _encode_text(self, texts: list[str]) -> torch.Tensor:
        """Canonically tokenize + embed a list of texts -> ``(len(texts), d)`` float32 CPU."""
        assert self._text_max_len is not None
        inputs = self._processor(
            text=texts,
            padding="max_length",
            max_length=self._text_max_len,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        feats = _pooled_features(self._model.get_text_features(**inputs))
        return feats.float().cpu()


class FrameSelector:
    """Base class for query-aware frame selectors.

    ``select`` centralizes every degenerate path so concrete selectors implement
    only the scoring math in ``_select_scored`` and can never raise on odd
    inputs. ``candidate_frames`` / ``candidate_fps`` are base-class state read by
    ``select_video_frames`` to size the candidate decode.
    """

    def __init__(
        self, scorer: Any, *, candidate_frames: int = 128, candidate_fps: float = 0.0
    ) -> None:
        self._scorer = scorer
        self.candidate_frames = candidate_frames
        self.candidate_fps = candidate_fps
        self._warned_no_query = False

    def ensure_loaded(self) -> None:
        """Load the scorer weights now (idempotent); see ``FrameQueryScorer.ensure_loaded``."""
        self._scorer.ensure_loaded()

    def validate_frame_geometry(self, frame_size: int) -> None:
        """Log how the dataset ``frame_size`` relates to the scorer's input resolution.

        Requires ``ensure_loaded`` to have run (the scorer's preprocessing
        record is derived at load time). Logs INFO on an exact match; logs a
        WARNING — never raises — when scoring will resize ``frame_size`` frames
        to the scorer's resolution.
        """
        pp = getattr(self._scorer, "preprocess", None)
        if pp is None:
            raise RuntimeError(
                "validate_frame_geometry requires loaded scorer weights: call "
                "ensure_loaded() first."
            )
        if int(frame_size) == int(pp.image_size):
            logger.info(
                "Frame-selection scorer input resolution %s matches [video].frame_size %s; "
                "scoring runs on the dataset frames unresized.",
                pp.image_size,
                frame_size,
            )
        else:
            logger.warning(
                "[video].frame_size=%s != frame-selection scorer resolution %s: scoring "
                "will resize each candidate frame (bicubic, antialiased). Set "
                "[video].frame_size=%s to skip the resize, or expect slightly different "
                "scores than a native-resolution scorer.",
                frame_size,
                pp.image_size,
                pp.image_size,
            )

    def embed_pool_u8(self, frames_u8: torch.Tensor) -> torch.Tensor:
        """Embed a raw uint8 candidate pool ``(n, 3, H, W)`` -> ``(n, d)`` float32 CPU."""
        return self._scorer.embed_uint8_frames(frames_u8)

    def embed_query(self, query: str) -> torch.Tensor:
        """Embed the query text -> ``(d,)`` L2-normalized float32 CPU tensor."""
        return self._scorer.embed_query(query)

    def _fallback_indices(self, n: int, query: str | None, k: int) -> list[int] | None:
        """Degenerate-input handling shared by every selection entry point.

        Returns the selection outright when no scoring is needed — empty pool
        (``[]``), pool no larger than the budget (take all), or empty query
        (uniform stride) — and ``None`` when the input genuinely needs scoring.
        """
        if n == 0 or k <= 0:
            return []
        if n <= k:
            return list(range(n))
        if query is None or not query.strip():
            if not self._warned_no_query:
                logger.warning(
                    "Frame selector received an empty query; falling back to uniform stride. "
                    "Selection conditions on the sample's question/instruction prompt, so a "
                    "corpus with no prompt (e.g. captioning with an empty [video].prompt) has "
                    "nothing to select on."
                )
                self._warned_no_query = True
            return uniform_stride_indices(n, k)
        return None

    def select_with_embeddings(
        self,
        frame_emb: torch.Tensor,
        query_emb: torch.Tensor,
        times: Sequence[float],
        k: int,
        seed_key: str | None = None,
    ) -> list[int]:
        """Run the scored selection on precomputed embeddings; sorted-ascending indices.

        The shared post-fallback tail of every entry point: callers are
        responsible for ``_fallback_indices`` first (this method assumes the
        input genuinely needs scoring, i.e. ``frame_emb.shape[0] > k``).
        """
        idx = self._select_scored(frame_emb, query_emb, list(times), k, seed_key)
        return sorted(int(i) for i in idx)

    def select(
        self,
        frames: Sequence[Any],
        times: Sequence[float],
        query: str | None,
        k: int,
        *,
        seed_key: str | None = None,
    ) -> list[int]:
        """Pick ``k`` of ``len(frames)`` PIL candidates; return sorted-ascending indices."""
        fallback = self._fallback_indices(len(frames), query, k)
        if fallback is not None:
            return fallback
        assert query is not None  # narrowed by _fallback_indices
        frame_emb = self._scorer.embed_frames(frames)  # (n, d)
        query_emb = self._scorer.embed_query(query)  # (d,)
        return self.select_with_embeddings(frame_emb, query_emb, times, k, seed_key)

    def select_uint8_frames(
        self,
        frames_u8: torch.Tensor,
        times: Sequence[float],
        query: str | None,
        k: int,
        *,
        seed_key: str | None = None,
    ) -> list[int]:
        """``select`` for candidates held as a raw uint8 ``(n, 3, S, S)`` tensor.

        The training-parity entry point: frames preprocessed exactly like the
        pool-mode training path (``embed_pool_u8`` -> scorer-derived
        preprocessing), so a caller that resizes/quantizes frames the way the
        dataloader workers do (``pil_to_uint8_tensor`` at ``frame_size``) gets
        bit-identical selections to ``apply_frame_selection``.
        """
        fallback = self._fallback_indices(frames_u8.shape[0], query, k)
        if fallback is not None:
            return fallback
        assert query is not None  # narrowed by _fallback_indices
        frame_emb = self.embed_pool_u8(frames_u8)  # (n, d)
        query_emb = self.embed_query(query)  # (d,)
        return self.select_with_embeddings(frame_emb, query_emb, times, k, seed_key)

    def select_tensor_frames(
        self,
        pixel_values: torch.Tensor,
        times: Sequence[float],
        query: str | None,
        k: int,
        *,
        seed_key: str | None = None,
    ) -> list[int]:
        """``select`` for candidates held as a preprocessed ``(n, 3, H, W)`` tensor.

        Same contract and selection math as ``select``; only the frame
        embedding path differs (``embed_frame_tensors`` instead of the
        PIL-consuming ``embed_frames``). The tensors must already carry the
        scorer's expected preprocessing (see ``preprocess_frame_tensors``).
        """
        fallback = self._fallback_indices(pixel_values.shape[0], query, k)
        if fallback is not None:
            return fallback
        assert query is not None  # narrowed by _fallback_indices
        frame_emb = self._scorer.embed_frame_tensors(pixel_values)  # (n, d)
        query_emb = self._scorer.embed_query(query)  # (d,)
        return self.select_with_embeddings(frame_emb, query_emb, times, k, seed_key)

    def _select_scored(
        self,
        frame_emb: torch.Tensor,
        query_emb: torch.Tensor,
        times: list[float],
        k: int,
        seed_key: str | None,
    ) -> list[int]:  # pragma: no cover - interface
        raise NotImplementedError


class TopKSelector(FrameSelector):
    """Cosine top-k of frame/query similarity (simplest query-aware baseline)."""

    def _select_scored(
        self,
        frame_emb: torch.Tensor,
        query_emb: torch.Tensor,
        times: list[float],
        k: int,
        seed_key: str | None,
    ) -> list[int]:
        scores = frame_emb @ query_emb  # (n,)
        return torch.topk(scores, k).indices.tolist()


class QFrameSelector(FrameSelector):
    """Q-Frame QFS: Gumbel-Max top-k over ``softmax(sim / tau)``.

    ``p_i = log softmax(s_i / tau) + g_i`` with ``g_i = -log(-log(u_i))`` and
    ``u ~ Uniform(0, 1)``; the top-k of ``p`` is a sample of k frames without
    replacement from the softmax (paper Eq. QFS). The draw is seeded from
    ``(seed, seed_key)`` so it is reproducible and rank/resume-stable.
    """

    def __init__(
        self,
        scorer: Any,
        *,
        candidate_frames: int = 128,
        candidate_fps: float = 0.0,
        gumbel_tau: float = 0.8,
        seed: int = 0,
    ) -> None:
        super().__init__(scorer, candidate_frames=candidate_frames, candidate_fps=candidate_fps)
        self.gumbel_tau = gumbel_tau
        self.seed = seed

    def _select_scored(
        self,
        frame_emb: torch.Tensor,
        query_emb: torch.Tensor,
        times: list[float],
        k: int,
        seed_key: str | None,
    ) -> list[int]:
        scores = frame_emb @ query_emb  # (n,)
        log_pi = torch.log_softmax(scores / self.gumbel_tau, dim=0).to(torch.float64)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(_derive_seed(self.seed, seed_key))
        u = torch.rand(scores.shape[0], generator=gen, dtype=torch.float64).clamp_(1e-12, 1 - 1e-12)
        gumbel = -torch.log(-torch.log(u))
        perturbed = log_pi + gumbel
        return torch.topk(perturbed, k).indices.tolist()


class MDP3Selector(FrameSelector):
    """Markov-DPP frame selection (arXiv 2501.02885).

    Builds a query-conditional multi-Gaussian-kernel matrix
    ``K = D (L + eps I) D`` with ``L_ij = k(f_i, f_j)`` (list-wise diversity),
    ``r_i = k(f_i, q)`` (query relevance), and ``D = diag(r ** (1 / lam))`` — so
    ``log det(K_S) = (1/lam) sum_i log r_i^2 + log det(L_S)`` (paper Eq. 9). Frame
    selection is greedy MAP DPP; a segmented Markov DP allocates the budget
    across temporal segments for sequentiality (paper Algorithm 1, lazy variant:
    each segment is scored greedily conditioned on the last frame of the running
    selection). ``segment_size == 0`` runs a single plain conditional DPP (the
    paper's no-sequentiality ablation).
    """

    def __init__(
        self,
        scorer: Any,
        *,
        candidate_frames: int = 128,
        candidate_fps: float = 0.0,
        lam: float = 0.2,
        segment_size: int = 32,
        kernel: str = "rkhs",
        log2_alphas: Sequence[float] = _MDP3_LOG2_ALPHAS,
        jitter: float = 1e-6,
    ) -> None:
        super().__init__(scorer, candidate_frames=candidate_frames, candidate_fps=candidate_fps)
        if kernel not in ("rkhs", "cosine"):
            raise ValueError(f"mdp3 kernel must be 'rkhs' or 'cosine' (got {kernel!r})")
        self.lam = lam
        self.segment_size = segment_size
        self.kernel = kernel
        self.log2_alphas = tuple(log2_alphas)
        self.jitter = jitter

    def _rkhs_kernel(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Multi-Gaussian kernel ``mean_u exp(-||a_i-b_j||^2 / (2*2^{alpha_u}))``, ``(m, p)``."""
        # Squared euclidean distances (float64, computed directly for stability).
        a2 = (a * a).sum(dim=1, keepdim=True)  # (m, 1)
        b2 = (b * b).sum(dim=1, keepdim=True).transpose(0, 1)  # (1, p)
        dist2 = (a2 + b2 - 2.0 * (a @ b.transpose(0, 1))).clamp_min(0.0)  # (m, p)
        alphas = torch.tensor([2.0**i for i in self.log2_alphas], dtype=a.dtype, device=a.device)
        kern = torch.exp(-dist2.unsqueeze(-1) / (2.0 * alphas))  # (m, p, U)
        return kern.mean(dim=-1)

    def _select_scored(
        self,
        frame_emb: torch.Tensor,
        query_emb: torch.Tensor,
        times: list[float],
        k: int,
        seed_key: str | None,
    ) -> list[int]:
        emb = frame_emb.to(torch.float64)
        q = query_emb.to(torch.float64)
        n = emb.shape[0]
        if self.kernel == "rkhs":
            sim = self._rkhs_kernel(emb, emb)  # (n, n), diag ~1
            relevance = self._rkhs_kernel(emb, q.unsqueeze(0)).squeeze(1)  # (n,)
        else:  # cosine ablation: (1 + s) / 2 keeps L PSD and r > 0
            sim = (1.0 + emb @ emb.transpose(0, 1)) / 2.0
            relevance = (1.0 + emb @ q) / 2.0
        relevance = relevance.clamp_min(1e-12)
        scale = relevance.pow(1.0 / self.lam)  # D = diag(r ** (1/lam))
        eye = torch.eye(n, dtype=emb.dtype)
        kernel = scale.unsqueeze(1) * (sim + self.jitter * eye) * scale.unsqueeze(0)
        if self.segment_size <= 0 or self.segment_size >= n:
            selected, _ = _dpp_greedy(kernel, k)
            return selected
        return _mdp3_segment_dp(kernel, k, self.segment_size)


def _dpp_greedy(
    kernel: torch.Tensor, k: int, *, condition_on: int | None = None
) -> tuple[list[int], list[float]]:
    """Fast greedy MAP DPP (Chen et al. 2018) with incremental Cholesky.

    Returns ``(selected, gains)`` where ``gains[m]`` is the marginal log-det gain
    of the ``(m+1)``-th pick (so ``cumsum(gains)`` is the log-det curve).
    ``condition_on`` forces an initial (uncounted) insertion — its determinant
    contribution and index are excluded from the returned picks. If the marginal
    gain floors out before ``k`` picks (rank exhaustion), the remainder is filled
    by descending diagonal (relevance), recorded with ``log(_MIN_GAIN)``.
    """
    n = kernel.shape[0]
    selected: list[int] = []
    gains: list[float] = []
    if k <= 0 or n == 0:
        return selected, gains
    d2 = torch.diagonal(kernel).clone()
    max_steps = k + (1 if condition_on is not None else 0)
    cis = torch.zeros((max_steps, n), dtype=kernel.dtype)
    step = 0

    def _insert(j: int) -> None:
        nonlocal step
        if step > 0:
            dots = cis[:step, :].transpose(0, 1) @ cis[:step, j]  # (n,) = <c_i, c_j>
        else:
            dots = torch.zeros(n, dtype=kernel.dtype)
        denom = torch.sqrt(d2[j].clamp_min(0.0))
        e = (kernel[j, :] - dots) / denom
        cis[step, :] = e
        d2.sub_(e * e)
        d2[j] = float("-inf")
        step += 1

    remaining = set(range(n))
    if condition_on is not None:
        c = int(condition_on)
        if float(d2[c]) > _MIN_GAIN:
            _insert(c)
        else:
            d2[c] = float("-inf")
        remaining.discard(c)

    while len(selected) < k and remaining:
        j = int(torch.argmax(d2).item())
        gain = float(d2[j])
        if not math.isfinite(gain) or gain <= _MIN_GAIN:
            break
        _insert(j)
        selected.append(j)
        gains.append(math.log(gain))
        remaining.discard(j)

    if len(selected) < k and remaining:
        diag = torch.diagonal(kernel)
        for i in sorted(remaining, key=lambda x: -float(diag[x])):
            if len(selected) >= k:
                break
            selected.append(i)
            gains.append(math.log(_MIN_GAIN))
    return selected, gains


def _mdp3_segment_dp(kernel: torch.Tensor, k: int, segment_size: int) -> list[int]:
    """Allocate the budget ``k`` across temporal segments (paper Algorithm 1, lazy).

    Segments are consecutive index blocks of ``segment_size``. A dynamic program
    over ``(segment, total_selected)`` keeps, per reachable total, the best-scoring
    running selection trace. Each segment is scored greedily conditioned on the
    last frame of the incoming trace, and the cumulative log-det curve gives the
    reward for every within-segment budget in one greedy pass. Returns the
    exactly-``k`` selection (temporal order restored by the caller's ``sorted``).
    """
    n = kernel.shape[0]
    seg_bounds = [(s, min(s + segment_size, n)) for s in range(0, n, segment_size)]
    # dp[total_selected] = (score, trace) with trace = global indices in pick order.
    dp: dict[int, tuple[float, list[int]]] = {0: (0.0, [])}
    for lo, hi in seg_bounds:
        seg = list(range(lo, hi))
        new_dp: dict[int, tuple[float, list[int]]] = {}
        for c_prev, (score_prev, trace_prev) in dp.items():
            budget = min(len(seg), k - c_prev)
            if budget < 0:
                continue
            cond = trace_prev[-1] if trace_prev else None
            if cond is not None:
                idx = seg + [cond]
                sub = kernel[idx][:, idx]
                cond_pos: int | None = len(seg)
            else:
                idx = seg
                sub = kernel[idx][:, idx]
                cond_pos = None
            picks_local, gains = _dpp_greedy(sub, budget, condition_on=cond_pos)
            cumulative = [0.0]
            for g in gains:
                cumulative.append(cumulative[-1] + g)
            for k_t in range(len(cumulative)):
                total = c_prev + k_t
                score = score_prev + cumulative[k_t]
                trace = trace_prev + [idx[p] for p in picks_local[:k_t]]
                if total not in new_dp or score > new_dp[total][0]:
                    new_dp[total] = (score, trace)
        dp = new_dp
    if k in dp:
        return dp[k][1]
    # n >= k guaranteed upstream, but stay defensive: return the fullest trace.
    best_total = max(dp)
    return dp[best_total][1]


@registry.register_frame_selector("topk")
def _build_topk(
    scorer: Any, *, candidate_frames: int = 128, candidate_fps: float = 0.0, **_: Any
) -> TopKSelector:
    return TopKSelector(scorer, candidate_frames=candidate_frames, candidate_fps=candidate_fps)


@registry.register_frame_selector("qframe")
def _build_qframe(
    scorer: Any,
    *,
    candidate_frames: int = 128,
    candidate_fps: float = 0.0,
    gumbel_tau: float = 0.8,
    seed: int = 0,
    **_: Any,
) -> QFrameSelector:
    return QFrameSelector(
        scorer,
        candidate_frames=candidate_frames,
        candidate_fps=candidate_fps,
        gumbel_tau=gumbel_tau,
        seed=seed,
    )


@registry.register_frame_selector("mdp3")
def _build_mdp3(
    scorer: Any,
    *,
    candidate_frames: int = 128,
    candidate_fps: float = 0.0,
    mdp3_lambda: float = 0.2,
    mdp3_segment_size: int = 32,
    mdp3_kernel: str = "rkhs",
    **_: Any,
) -> MDP3Selector:
    return MDP3Selector(
        scorer,
        candidate_frames=candidate_frames,
        candidate_fps=candidate_fps,
        lam=mdp3_lambda,
        segment_size=mdp3_segment_size,
        kernel=mdp3_kernel,
    )


def build_frame_selector(
    config: Any, device: torch.device | str | None = None, dtype: torch.dtype | None = None
) -> FrameSelector:
    """Build the frame selector selected by ``config.type``.

    Constructs one ``FrameQueryScorer`` (weights load lazily) and dispatches
    through the ``frame_selector`` registry. The config is duck-typed
    (``.type`` / ``.scorer`` / ``.scorer_path`` / ``.extra_kwargs()``) to avoid a
    data->config import cycle, mirroring ``build_adapter`` / ``build_time_embedding``.
    """
    scorer = FrameQueryScorer(
        config.scorer,
        config.scorer_path,
        device=device if device is not None else "cpu",
        dtype=dtype if dtype is not None else torch.float32,
    )
    builder = registry.get_frame_selector(config.type)
    return builder(scorer, **config.extra_kwargs())


def _dedupe_by_time(
    frames: list[PILImage], times: list[float]
) -> tuple[list[PILImage], list[float]]:
    """Drop candidates that repeat a matched timestamp (short clips duplicate the
    tail frame in ``decode_video_frames``); keeps first occurrence, preserving order.
    """
    seen: set[float] = set()
    out_frames: list[PILImage] = []
    out_times: list[float] = []
    for frame, t in zip(frames, times, strict=True):
        key = round(t, 6)
        if key in seen:
            continue
        seen.add(key)
        out_frames.append(frame)
        out_times.append(t)
    return out_frames, out_times


@dataclass(frozen=True)
class CandidatePoolSpec:
    """Geometry of the candidate-pool decode, threaded to training datasets.

    Replaces a live ``FrameSelector`` on the dataset side: workers only decode
    pools (``decode_candidate_pool``), while scoring/selection runs in the
    process that owns the accelerator (``apply_frame_selection``). Mirrors the
    same-named ``[frame_selector]`` fields.
    """

    candidate_frames: int = 128
    candidate_fps: float = 0.0

    @classmethod
    def from_config(cls, config: Any) -> CandidatePoolSpec:
        """Authoritative construction from a ``FrameSelectorConfig``-shaped object.

        Duck-typed (``.candidate_frames`` / ``.candidate_fps``) so the data
        layer never imports the config layer.
        """
        return cls(
            candidate_frames=int(config.candidate_frames),
            candidate_fps=float(config.candidate_fps),
        )


def decode_candidate_pool(
    source: str | bytes,
    *,
    candidate_frames: int,
    candidate_fps: float = 0.0,
    sampling_policy: str = "uniform",
) -> tuple[list[PILImage], list[float]]:
    """Decode the candidate pool a selector chooses from; returns (frames, times).

    The decode is sized *only* by ``candidate_frames`` / ``candidate_fps`` —
    deliberately independent of ``[video].fps`` / ``[video].min_frames`` (which
    govern the non-selector decode; ``JobConfig`` warns if they are set
    alongside a selector). ``candidate_fps == 0`` samples exactly
    ``candidate_frames`` uniformly over the clip (via the ``min==max`` clamp in
    ``sample_timestamps``); ``candidate_fps > 0`` samples at that rate capped
    at ``candidate_frames``. Time-duplicate candidates are dropped, so the pool
    may come back smaller than ``candidate_frames``. Decode exceptions
    propagate (the caller isolates them: per-sample at train, per-request at
    eval).
    """
    if candidate_fps > 0:
        frames, times = decode_video_frames(
            source,
            fps=candidate_fps,
            min_frames=1,
            max_frames=candidate_frames,
            sampling_policy=sampling_policy,
        )
    else:
        frames, times = decode_video_frames(
            source,
            fps=1.0,
            min_frames=candidate_frames,
            max_frames=candidate_frames,
            sampling_policy=sampling_policy,
        )
    return _dedupe_by_time(frames, times)


def select_video_frames(
    path: str,
    query: str | None,
    selector: FrameSelector,
    k: int,
    *,
    sampling_policy: str = "uniform",
    seed_key: str | None = None,
    frame_size: int | None = None,
) -> tuple[list[PILImage], list[float]]:
    """Decode a candidate pool, select ``k`` frames for the query, return them + times.

    The single-process entry point (eval harness, smoke scripts): pool sizing
    comes from the selector's ``candidate_frames`` / ``candidate_fps`` (see
    ``decode_candidate_pool``) and ``k`` is ``[video].max_frames``. The training
    loop splits the same chain across processes instead — workers run
    ``decode_candidate_pool``, the main process runs ``apply_frame_selection``.

    With ``frame_size`` set, candidates are scored through the *training*
    pixel path — ``pil_to_uint8_tensor`` at ``frame_size`` then the scorer's
    derived preprocessing (``select_uint8_frames``), exactly what pool-mode
    workers + ``apply_frame_selection`` do — so selections are bit-identical
    to training for the same clip/query/seed. Without it, candidates go
    through the scorer's HF processor at native resolution (``select``), whose
    resize kernel and float path differ slightly from training. Eval passes
    ``[video].frame_size`` for parity. The *returned* frames are always the
    original decoded PIL frames; ``frame_size`` affects scoring pixels only.
    """
    frames, times = decode_candidate_pool(
        path,
        candidate_frames=selector.candidate_frames,
        candidate_fps=selector.candidate_fps,
        sampling_policy=sampling_policy,
    )
    if frame_size is not None and frames:
        pool_u8 = torch.stack([pil_to_uint8_tensor(f, frame_size) for f in frames])
        idx = selector.select_uint8_frames(pool_u8, times, query, k, seed_key=seed_key)
    else:
        idx = selector.select(frames, times, query, k, seed_key=seed_key)
    return [frames[i] for i in idx], [times[i] for i in idx]


def apply_frame_selection(
    batch: dict[str, Any],
    selector: FrameSelector,
    max_frames: int,
    device: torch.device | str,
) -> dict[str, Any]:
    """Score + select frames for a candidate-pool batch (main-process GPU stage).

    Consumes the pool keys emitted by pool-mode datasets / ``VideoCollator``
    (``candidate_pixels`` ``(B, C, 3, H, W)`` uint8 with any pool capacity
    ``C >= max(counts)`` — C is the batch max and may be smaller than
    ``candidate_frames``; ``candidate_count`` ``(B,)``, ``candidate_times``
    ``(B, C)``, ``query`` / ``seed_key`` string lists) and replaces them
    in-place with the standard model contract on ``device``: ``pixel_values``
    ``(B, max_frames, 3, H, W)`` float32, ``frame_mask`` ``(B, max_frames)``
    bool, ``frame_times`` ``(B, max_frames)`` float32. Text keys pass through
    untouched.

    Runs in three phases:

    - **Embed**: per non-fallback sample, the raw uint8 pool slice is moved to
      ``device`` and embedded via the scorer's own derived preprocessing
      (``embed_pool_u8`` -> ``preprocess_frame_tensors``), per sample so chunk
      boundaries — and therefore embeddings — never depend on batchmates.
      Query embeddings are cached per distinct query string, so repeated
      queries in a batch cost one text forward.
    - **Select**: fallback indices (empty pool / take-all / empty-query
      stride) or ``select_with_embeddings`` on the cached embeddings.
    - **Gather**: only the *selected* frames are normalized for the model
      (``normalize_frames`` with ``DEFAULT_IMAGE_MEAN``/``STD`` — elementwise,
      so bit-identical to worker-side ``pil_to_tensor``); padding slots and
      unselected candidates are never normalized.

    A ``candidate_count`` of 0 (worker-side decode failure) flows through as
    an all-masked sample, mirroring the non-selector skip-with-mask behavior.
    Per-sample scoring/selection failures are isolated the same way: the
    failing sample degrades to all-masked with its labels forced to ``-100``
    (a warning is logged) and its batchmates train normally — a transient
    scorer error must not kill a long run. ``ScorerUnavailableError`` is the
    deliberate exception: the scorer never loaded, which is systemic
    misconfiguration, so it propagates and fails the run loudly.
    """
    pool_u8 = batch.pop("candidate_pixels")  # (B, C, 3, H, W) uint8, stays on CPU
    counts = batch.pop("candidate_count")  # (B,) int64
    pool_times = batch.pop("candidate_times")  # (B, C) float32
    queries: list[str] = batch.pop("query")
    seed_keys: list[str] = batch.pop("seed_key")

    device = torch.device(device)
    b, _, _, h, w = pool_u8.shape
    pixel_values = torch.zeros(b, max_frames, 3, h, w, dtype=torch.float32, device=device)
    frame_mask = torch.zeros(b, max_frames, dtype=torch.bool, device=device)
    frame_times = torch.zeros(b, max_frames, dtype=torch.float32, device=device)

    def _mask_failed_sample(i: int, stage: str, err: Exception) -> None:
        logger.warning(
            "Frame selection %s failed for sample %d (%s); skipping sample "
            "(all-masked frames, labels -100): %s",
            stage,
            i,
            seed_keys[i],
            err,
        )
        if "labels" in batch:
            batch["labels"][i] = -100

    # Phase A — embed: per-sample frame embeddings + a per-batch query cache.
    # Scored samples keep their device-resident uint8 pool slice so Phase C's
    # gather is a device index_select instead of a second host-to-device copy.
    fallbacks: list[list[int] | None] = []
    frame_embs: dict[int, torch.Tensor] = {}
    dev_pools: dict[int, torch.Tensor] = {}
    query_cache: dict[str, torch.Tensor] = {}
    failed: set[int] = set()
    for i in range(b):
        n = int(counts[i])
        fb = selector._fallback_indices(n, queries[i] or None, max_frames)
        fallbacks.append(fb)
        if fb is None:
            try:
                dev_pools[i] = pool_u8[i, :n].to(device)
                frame_embs[i] = selector.embed_pool_u8(dev_pools[i])
                if queries[i] not in query_cache:
                    query_cache[queries[i]] = selector.embed_query(queries[i])
            except ScorerUnavailableError:
                raise
            except Exception as e:  # noqa: BLE001 - per-sample isolation
                _mask_failed_sample(i, "scoring", e)
                failed.add(i)
                dev_pools.pop(i, None)

    for i in range(b):
        if i in failed:
            continue  # all-masked sample, labels already forced to -100
        n = int(counts[i])
        times = pool_times[i, :n].tolist()
        # Phase B — select: fallback or scored selection on cached embeddings.
        fb = fallbacks[i]
        if fb is not None:
            idx = fb
        else:
            try:
                idx = selector.select_with_embeddings(
                    frame_embs[i], query_cache[queries[i]], times, max_frames, seed_keys[i]
                )
            except Exception as e:  # noqa: BLE001 - per-sample isolation
                _mask_failed_sample(i, "selection", e)
                continue
        if not idx:
            continue  # undecodable clip: all-masked sample (labels already -100)
        # Phase C — gather: normalize only the selected frames for the model
        # (dataset stats, elementwise -> bit-identical to worker pil_to_tensor).
        idx_t = torch.tensor(idx, dtype=torch.long)
        src = dev_pools.pop(i, None)
        if src is not None:
            sel = src.index_select(0, idx_t.to(device))
        else:
            sel = pool_u8[i].index_select(0, idx_t).to(device)
        pixel_values[i, : len(idx)] = normalize_frames(
            sel, DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD
        )
        frame_mask[i, : len(idx)] = True
        frame_times[i, : len(idx)] = torch.tensor(
            [times[j] for j in idx], dtype=torch.float32, device=device
        )

    batch["pixel_values"] = pixel_values
    batch["frame_mask"] = frame_mask
    batch["frame_times"] = frame_times
    return batch
