# pyright: reportMissingImports=false
# ^ lmms-eval is an optional, UNDECLARED dependency (it is installed separately,
#   not listed in pyproject.toml). CI type-checks `kempnerforge/` without it
#   installed, so the `lmms_eval` imports below would otherwise raise
#   reportMissingImports. A file-level directive (not a `# type: ignore`, which
#   reportUnnecessaryTypeIgnoreComment would flag as unnecessary in dev where
#   lmms-eval *is* installed) scopes the relaxation to this one module.
"""lmms-eval chat-model adapter wrapping a KempnerForge ``VLMWrapper``.

This module implements ``KempnerForgeVLM``, an lmms-eval ``chat`` model
(``is_simple = False``) that evaluates any KempnerForge VLM checkpoint on the
standard multimodal benchmarks lmms-eval implements as ``generate_until`` tasks
(MMMU, MMBench, ScienceQA, SEED, AI2D, ...). It is loaded directly from a DCP checkpoint,
and is arch-agnostic across the generative VLM arches.
v1 scope and deliberate choices (see docs/how-to/run-vlm-evaluation.md):

- **Generation: no transformer KV cache, single-GPU, batched.** The decode loop
  re-runs the transformer (including the vision encoder + adapter) over the
  growing sequence each step. There is no transformer KV cache
  (``Transformer.forward`` forbids combining ``kv_caches`` with any
  image-conditioning route), and KempnerForge has no image-conditioned KV-cache
  decode path. Requests are decoded in batches
  (``batch_size`` model-arg) by **right-padding** the text to the batch-max
  length — the same layout training uses (image prefix at ``0..n-1``, text
  contiguous from ``n``, trailing pads causally masked) — and reading each
  row's logits at its own last real position. Single-GPU is the validated
  invocation, not a baked-in assumption: rank/world_size come from the lmms
  base (defaults 0/1) and model construction sits behind ``_build_model`` so a
  data-parallel path is a localized future change.

- **Prompt rendering: flatten, no chat template.** KempnerForge pre-training
  uses no chat template / processor and no ``<image>`` placeholder (images are
  conditioned at the embedding level). We render an lmms-eval ``ChatMessages``
  by concatenating its text content blocks in order into a single prompt
  string. This discards role/turn structure and any model-specific template.
  A future enhancement should add repo-wide chat-template support (applied once
  a post-training format exists), at which point this rendering step becomes
  configurable rather than hard-coded to flattening.

- **Arch coverage.** Joint-Decoder, Cross-Attention, and MoT are supported.
  MoMa is NOT: its expert-choice routing is non-causal and cannot
  autoregressively generate, and chat tasks are generation-only. A MoMa
  checkpoint fails fast in ``__init__``.

- **Image and video.** An image checkpoint evaluates exactly one image per
  request; a video checkpoint (a ``[video]`` config) evaluates one video per
  request — decoded to a fixed ``frames_per_clip`` clip via the training
  frame-sampling policy — and also accepts a single image (a 1-frame clip).
  Audio, multi-image, multiple videos, mixed image+video, and multi-turn/few-shot
  requests raise ``NotImplementedError``; ``loglikelihood`` and
  ``generate_until_multi_round`` are not implemented (chat tasks are
  generation-only). Visual input is modeled as an ordered list of frames (a
  single image is the length-1 case).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.protocol import ChatMessages
from lmms_eval.utils import Collator
from tqdm import tqdm

from kempnerforge.config.job import JobConfig
from kempnerforge.config.loader import load_config
from kempnerforge.config.video import VideoConfig
from kempnerforge.config.vlm import VLMConfig
from kempnerforge.data.video_io import decode_video_frames
from kempnerforge.data.vlm_dataset import (
    DEFAULT_IMAGE_MEAN,
    DEFAULT_IMAGE_STD,
    build_tokenizer,
    frames_to_clip_tensor,
    pil_to_tensor,
    resolve_pad_id,
)
from kempnerforge.metrics.logger import get_logger
from kempnerforge.model.generate import sample
from kempnerforge.model.vlm import VLMWrapper, build_vlm_wrapper
from kempnerforge.resilience.elastic import resolve_resume_path

logger = get_logger(__name__)

DEFAULT_MAX_NEW_TOKENS = 128

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    try:
        return _DTYPES[dtype]
    except KeyError:
        raise ValueError(f"Unsupported dtype {dtype!r}; choose from {sorted(_DTYPES)}") from None


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #


def _build_model(config: JobConfig, device: torch.device, dtype: torch.dtype) -> VLMWrapper:
    assert config.vlm is not None, "internal: _build_model requires a VLM config"
    assert config.vision_encoder is not None, "internal: VLM config requires a vision encoder"
    assert config.adapter is not None, "internal: VLM config materializes a default adapter"
    # A video checkpoint bakes num_image_tokens = frames_per_clip * tokens_per_frame
    # into the transformer's residual/positional structure, so the eval must rebuild
    # with the same frames_per_clip the training run used (mirrors scripts/train.py).
    # Image configs default to 1.
    frames_per_clip = config.video.max_frames if config.video is not None else 1
    model = build_vlm_wrapper(
        config.model,
        config.vision_encoder,
        config.adapter,
        config.vlm,
        frames_per_clip=frames_per_clip,
    )
    return model.to(device=device, dtype=dtype)


def _log_checkpoint_metadata(ckpt_path: Path) -> None:
    """Log ``step``/``tokens_seen`` from the plain-JSON ``metadata.json`` if
    present. Never reads ``train_state.pt`` (a pickle behind a UID-ownership
    security gate); only the model weights from the ``.distcp`` shards are
    needed for inference.
    """
    meta_file = ckpt_path / "metadata.json"
    if not meta_file.exists():
        return
    try:
        meta = json.loads(meta_file.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"Could not read {meta_file}: {exc}")
        return
    logger.info(
        f"VLM checkpoint metadata: step={meta.get('step')}, tokens_seen={meta.get('tokens_seen')}"
    )


def _load_config(config_path: str) -> JobConfig:
    config = load_config(config_path, cli_args=[])
    if not config.is_vlm:
        raise ValueError(
            f"{config_path!r} is not a VLM config (config.vlm is None); this evaluation "
            f"path is VLM-only. Use scripts/eval.py for text-model loss/perplexity."
        )
    return config


def _check_generative(vlm_config: VLMConfig) -> None:
    """Fail fast (before building) on arches that cannot autoregressively generate."""
    if not vlm_config.is_generative:
        raise ValueError(
            f"VLM arch {vlm_config.arch!r} cannot be evaluated: its routing is non-causal "
            f"and cannot autoregressively generate, but chat tasks are generation-only. "
            f"Generation support for {vlm_config.arch!r} is a tracked model-side follow-up "
            f"— contact the project owner."
        )


def _load_weights(
    config: JobConfig, checkpoint: str, device: torch.device, dtype: torch.dtype
) -> VLMWrapper:
    """Build a ``VLMWrapper`` and load DCP weights for single-process eval.

    Accepts either a run directory (resolved to its ``latest``/highest
    ``step_N`` via ``resolve_resume_path``) or a specific checkpoint directory
    (used as-is when ``resolve_resume_path`` finds nothing). DCP reshards on
    load, so checkpoints saved under FSDP/PP load into the full model.
    """
    ckpt_path = resolve_resume_path(checkpoint) or Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt_path}")
    if not ckpt_path.is_dir():
        raise NotADirectoryError(f"Checkpoint path is not a directory: {ckpt_path}")
    model = _build_model(config, device, dtype)
    model.eval()

    # Single-process DCP load: build the full (unsharded) model, then load the
    # model shards into its state-dict.
    state_dict = {"model": model.state_dict()}
    dcp.load(state_dict, checkpoint_id=str(ckpt_path))
    model.load_state_dict(state_dict["model"])

    _log_checkpoint_metadata(ckpt_path)
    logger.info(f"Loaded VLM checkpoint from {ckpt_path}")
    return model


# --------------------------------------------------------------------------- #
# Request rendering + preprocessing
# --------------------------------------------------------------------------- #


def _render_request(
    messages: ChatMessages, video_config: VideoConfig | None
) -> tuple[list[Any], str]:
    """Flatten one chat request into ``(frames, prompt_text)``.

    ``frames`` is an ordered list of visual frames; a single image is the
    length-1 case and a decoded video is the multi-frame case. ``video_config``
    is the checkpoint's ``[video]`` config (``None`` for an image checkpoint)
    and selects the mode:

    - **Image checkpoint** (``video_config is None``): exactly one image per
      request; video content raises (an image model cannot evaluate video).
    - **Video checkpoint**: exactly one video — decoded to frames via
      ``video_io.decode_video_frames`` using the checkpoint's frame-sampling
      policy — or, when no video is present, a single image treated as a
      1-frame clip (zero-padded to ``frames_per_clip`` downstream).

    Out-of-scope content (audio, multi-turn/few-shot, multi-image, multiple
    videos, mixed image+video) raises ``NotImplementedError`` so the offending
    task is surfaced rather than silently mishandled. Text content blocks are
    concatenated in message order (newline-joined); role/turn structure is
    intentionally discarded (see the module docstring on flattening).
    """
    images, videos, audios = messages.extract_media()
    if audios:
        raise NotImplementedError(
            "Audio evaluation is not implemented (image/video only). An audio request "
            "reached the KempnerForge VLM adapter; report the task to the project owner."
        )

    roles = [message.role for message in messages.messages]
    if any(role == "assistant" for role in roles) or roles.count("user") > 1:
        raise NotImplementedError(
            "Multi-turn / few-shot requests are not supported (single-turn, zero-shot "
            "only). Report the task to the project owner."
        )

    prompt = "\n".join(
        content.text
        for message in messages.messages
        for content in message.content
        if content.type == "text"
    )

    if video_config is None:
        # Image checkpoint: image-only, exactly one image per request.
        if videos:
            raise NotImplementedError(
                "This is an image checkpoint (no [video] config) and cannot evaluate video. "
                "Use a video checkpoint, or report the task to the project owner."
            )
        if len(images) != 1:
            raise NotImplementedError(
                f"This adapter supports exactly one image per request, got {len(images)}. "
                "Multi-image and text-only requests are out of scope; report the task to "
                "the project owner."
            )
        return images, prompt

    # Video checkpoint: exactly one visual — a video (decoded to frames) or a
    # single image (treated as a 1-frame clip, zero-padded downstream).
    if len(videos) > 1:
        raise NotImplementedError(
            f"Multiple videos per request are not supported, got {len(videos)}. "
            "Report the task to the project owner."
        )
    if videos:
        if images:
            raise NotImplementedError(
                "Mixed image + video content in one request is not supported. "
                "Report the task to the project owner."
            )
        path = videos[0]
        if not isinstance(path, str):
            raise NotImplementedError(
                f"Video content must be a local path string for decoding, got "
                f"{type(path).__name__}. The task may pass clip boundaries or a URL; "
                "report the task to the project owner."
            )
        frames = decode_video_frames(
            path,
            fps=video_config.fps,
            min_frames=video_config.min_frames,
            max_frames=video_config.max_frames,
            sampling_policy=video_config.sampling_policy,
        )
        if not frames:
            logger.warning(
                f"No frames decoded from {path}; evaluating a zero clip (result unreliable)."
            )
        return frames, prompt
    if len(images) == 1:
        # A single image on a video checkpoint: a 1-frame clip (zero-padded to
        # frames_per_clip downstream), consistent with how training pads short clips.
        return images, prompt
    raise NotImplementedError(
        f"A video-checkpoint request must carry exactly one video or one image, got "
        f"{len(images)} images and no video. Multi-image and text-only requests are out "
        "of scope; report the task to the project owner."
    )


def _to_pil(frame: Any) -> Any:
    """Normalize one visual frame to a PIL ``Image``.

    lmms-eval delivers image content as either a ``PIL.Image`` or a path/URL
    string. The training preprocessing (``pil_to_tensor``) is strict PIL-only, so
    a string is opened here (copied so the file handle can close). This lets both
    the image packer (``_frames_to_pixel_values``) and the video packer
    (``frames_to_clip_tensor`` -> ``pil_to_tensor``) accept either form.
    """
    if isinstance(frame, str):
        from PIL import Image

        with Image.open(frame) as im:
            return im.copy()
    return frame


def _frames_to_pixel_values(
    frames: list[Any], image_size: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Convert an ordered list of frames to a ``(num_frames, 3, H, W)`` tensor.

    Reuses the training-time ``pil_to_tensor`` (resize + SigLIP-style
    normalize) as the single source of truth. v1 passes a single image (a
    length-1 list); the list shape is the seam for future video (ordered
    frames). Each frame may be a ``PIL.Image`` or a path string (see ``_to_pil``).
    """
    tensors = [
        pil_to_tensor(_to_pil(frame), image_size, DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD)
        for frame in frames
    ]
    return torch.stack(tensors, dim=0).to(device=device, dtype=dtype)


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


class _ContextBudgetError(ValueError):
    """No room for the prompt: max_new_tokens + image tokens exceed max_seq_len."""


def _resolve_gen_kwargs(gen_kwargs: dict[str, Any], default_max_new_tokens: int) -> dict[str, Any]:
    """Merge task ``gen_kwargs`` over the adapter's fallback defaults.

    Uses explicit ``is None`` checks (not ``x or default``) so an explicit falsy
    task value — ``max_new_tokens=0`` or ``top_p=0.0`` — is honored rather than
    silently replaced by the default.
    """
    until = gen_kwargs.get("until")
    if until is None:
        until = []
    elif isinstance(until, str):
        until = [until]

    mnt = gen_kwargs.get("max_new_tokens")
    max_new_tokens = default_max_new_tokens if mnt is None else int(mnt)
    temp = gen_kwargs.get("temperature")
    temperature = 0.0 if temp is None else float(temp)
    # An explicit do_sample=False forces greedy even if a temperature is given.
    if not gen_kwargs.get("do_sample", temperature > 0):
        temperature = 0.0

    sampling = temperature > 0
    top_k = gen_kwargs.get("top_k")
    top_p = gen_kwargs.get("top_p")
    return {
        "until": [u for u in until if u],
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": (0 if top_k is None else int(top_k)) if sampling else 0,
        "top_p": (1.0 if top_p is None else float(top_p)) if sampling else 1.0,
    }


def _first_stop(text: str, until: list[str]) -> int | None:
    """Index of the earliest occurrence of any stop string in ``text``."""
    cut: int | None = None
    for stop in until:
        idx = text.find(stop)
        if idx != -1 and (cut is None or idx < cut):
            cut = idx
    return cut


@torch.inference_mode()
def _generate_batch(
    model: VLMWrapper,
    tokenizer: Any,
    pixel_values: torch.Tensor,
    prompt_ids: list[torch.Tensor],
    resolved: dict[str, Any],
    max_seq_len: int,
    frame_mask: torch.Tensor | None = None,
) -> list[str]:
    """Batched decode (no transformer KV cache); returns one continuation per request.

    Decodes ``B`` requests together (``pixel_values`` is ``(B, 3, H, W)`` or a
    ``(B, F, 3, H, W)`` video clip, ``prompt_ids`` a list of ``B`` 1-D token
    tensors; ``frame_mask`` is ``(B, F)`` bool for video, masking padded-frame
    visual tokens from attention as in training). There is no transformer KV
    cache and no vision cache: ``model(...)`` re-runs over the growing
    **right-padded** batch each step, re-encoding the vision tower each time.
    Right-padding matches the training
    layout: the image prefix stays at positions ``0..n-1`` and text is contiguous
    from ``n`` for every row (so image/text RoPE distances are consistent across
    rows), and the trailing pads are causally masked, so a batched forward gives
    each row the same real-position logits as decoding it alone. Each row's next
    token is read at its own last real position; EOS / ``max_new_tokens`` / first
    ``until`` match are tracked per row. ``B == 1`` reproduces the single-request
    path exactly.
    """
    until: list[str] = resolved["until"]
    max_new_tokens: int = resolved["max_new_tokens"]
    temperature: float = resolved["temperature"]
    top_k: int = resolved["top_k"]
    top_p: float = resolved["top_p"]
    eos_id = tokenizer.eos_token_id
    pad_id = resolve_pad_id(tokenizer)
    device = pixel_values.device
    batch_size = len(prompt_ids)

    # Length bound: image tokens (in-residual for JD/MoT; 0 for CA) + prompt +
    # generated must fit the context. Reserve room for generation and left-
    # truncate any over-budget prompt (per row).
    num_image_tokens = model.num_image_tokens
    prompt_budget = max_seq_len - num_image_tokens - max_new_tokens
    if prompt_budget <= 0:
        raise _ContextBudgetError(
            f"max_new_tokens ({max_new_tokens}) + image tokens ({num_image_tokens}) leave no "
            f"room for the prompt within max_seq_len ({max_seq_len}); lower --max-new-tokens."
        )
    prompts: list[torch.Tensor] = []
    for ids in prompt_ids:
        if ids.shape[0] > prompt_budget:
            logger.warning(
                f"Prompt ({ids.shape[0]}) + image tokens ({num_image_tokens}) + max_new_tokens "
                f"({max_new_tokens}) exceeds max_seq_len ({max_seq_len}); left-truncating prompt "
                f"to {prompt_budget} tokens. Severe truncation may distort results."
            )
            ids = ids[-prompt_budget:]
        prompts.append(ids)

    generated: list[list[int]] = [[] for _ in range(batch_size)]
    done = [False] * batch_size
    row_index = torch.arange(batch_size, device=device)

    for _ in range(max_new_tokens):
        # Rebuild the right-padded batch from prompt + tokens generated so far.
        seqs = [
            torch.cat([prompts[i], torch.tensor(generated[i], dtype=torch.long, device=device)])
            for i in range(batch_size)
        ]
        real_len = torch.tensor([s.shape[0] for s in seqs], device=device)
        cur_max = int(real_len.max().item())
        input_ids = torch.full((batch_size, cur_max), pad_id, dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            input_ids[i, : s.shape[0]] = s

        logits, _ = model(pixel_values, input_ids, frame_mask=frame_mask)
        # Each row's next-token logits sit at its own last real position (the
        # output is already trimmed to text positions for JD/MoT; CA has no
        # image prefix), not at [-1] (a pad for shorter rows).
        next_logits = logits[row_index, real_len - 1]
        next_tokens = sample(next_logits, temperature, top_k, top_p)

        for i in range(batch_size):
            if done[i]:
                continue
            token_id = int(next_tokens[i].item())
            if eos_id is not None and token_id == eos_id:
                done[i] = True
                continue
            generated[i].append(token_id)
            if len(generated[i]) >= max_new_tokens:
                done[i] = True
            elif until:
                text = tokenizer.decode(generated[i], skip_special_tokens=True)
                if _first_stop(text, until) is not None:
                    done[i] = True
        if all(done):
            break

    outputs: list[str] = []
    for tokens in generated:
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        cut = _first_stop(text, until)
        outputs.append(text[:cut] if cut is not None else text)
    return outputs


# --------------------------------------------------------------------------- #
# Adapter
# --------------------------------------------------------------------------- #


class KempnerForgeVLM(lmms):
    """lmms-eval chat model over a KempnerForge ``VLMWrapper`` (see module docstring).

    Model args (parsed by the base ``create_from_arg_string`` from a
    ``key=value,...`` string):

    - ``config`` (required): path to the KempnerForge TOML the checkpoint was
      trained with.
    - ``checkpoint`` (required): DCP checkpoint directory (a run dir or a
      specific ``step_N`` dir).
    - ``device`` (default ``"cuda"``), ``dtype`` (default: the checkpoint config's
      ``train.param_dtype``; pass e.g. ``"float32"`` to override).
    - ``batch_size`` (default ``1``): number of requests decoded together
      (right-padded), grouped by gen_kwargs.
    - ``max_new_tokens`` (default ``128``): fallback only; task ``gen_kwargs``
      override it.
    """

    is_simple = False

    def __init__(
        self,
        config: str,
        checkpoint: str,
        device: str = "cuda",
        dtype: str | None = None,
        batch_size: int | str = 1,
        max_new_tokens: int | str = DEFAULT_MAX_NEW_TOKENS,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if kwargs:
            logger.warning(f"Ignoring unsupported model_args: {sorted(kwargs)}")

        self._device = torch.device(device)
        self._batch_size = int(batch_size)
        self._default_max_new_tokens = int(max_new_tokens)
        if self._batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self._batch_size}")
        if self._default_max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self._default_max_new_tokens}")
        self._config = _load_config(config)
        assert self._config.vlm is not None  # guaranteed by is_vlm; narrows for the type checker
        # Default the compute dtype to what the checkpoint was trained at
        # (config.train.param_dtype) unless an explicit dtype was passed.
        self._dtype = self._config.train.param_dtype if dtype is None else _resolve_dtype(dtype)
        self._arch = self._config.vlm.arch
        # Fail fast on non-generative arches before building/loading the model.
        _check_generative(self._config.vlm)

        # Video vs image mode is a property of the checkpoint's config. A video
        # checkpoint ([video] config) fixes frames_per_clip and resizes frames to
        # config.video.frame_size; an image checkpoint is a 1-frame clip at
        # config.data.hf_image_size.
        self._is_video = self._config.is_video
        if self._config.video is not None:
            self._frames_per_clip = self._config.video.max_frames
            self._frame_size = self._config.video.frame_size
        else:
            self._frames_per_clip = 1
            self._frame_size = self._config.data.hf_image_size

        self._model = _load_weights(self._config, checkpoint, self._device, self._dtype)
        self._tokenizer = build_tokenizer(self._config.data.tokenizer_path)
        self._max_seq_len = self._config.model.max_seq_len
        logger.info(
            f"KempnerForgeVLM ready: arch={self._arch}, video={self._is_video}, "
            f"frames_per_clip={self._frames_per_clip}, device={self._device}, "
            f"dtype={self._dtype}, max_seq_len={self._max_seq_len}"
        )

    def generate_until(self, requests: list[Instance]) -> list[str]:
        # Group requests by gen_kwargs (a batch must share decode params) and,
        # within a group, sort by context length so similar-length prompts batch
        # together (less padding). Collator.get_original restores request order.
        def _collate(args: tuple[Any, ...]) -> int:
            return -len(args[0]) if isinstance(args[0], str) else 0

        re_ords = Collator(
            [request.args for request in requests],
            _collate,
            group_fn=lambda args: args[2],  # args[2] == gen_kwargs
            grouping=True,
        )
        results: list[str] = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="KempnerForge VLM")
        for chunk in re_ords.get_batched(n=self._batch_size, batch_fn=None):
            # Every request in the chunk shares gen_kwargs (index 2); resolve once.
            resolved = _resolve_gen_kwargs(chunk[0][2], self._default_max_new_tokens)
            # Per-request slots aligned to ``chunk``: ``None`` = filled by generation
            # below; ``""`` = a request that failed to render/preprocess, isolated
            # with a warning so one bad doc does not abort the whole run.
            chunk_outputs: list[str | None] = [None] * len(chunk)
            frames_batch: list[torch.Tensor] = []
            masks_batch: list[torch.Tensor] = []
            prompt_ids: list[torch.Tensor] = []
            for slot, args in enumerate(chunk):
                try:
                    # Chat 6-tuple: (context, doc_to_messages, gen_kwargs, doc_id, task, split).
                    doc = self.task_dict[args[4]][args[5]][args[3]]
                    messages = ChatMessages(messages=args[1](doc))
                    frames, prompt = _render_request(messages, self._config.video)
                    # lmms-eval may deliver image content as a path/URL string;
                    # normalize to PIL so both the image and video packers (strict
                    # pil_to_tensor) accept it.
                    frames = [_to_pil(f) for f in frames]
                    # Mirror training tokenization: no chat template, no <image>
                    # placeholder, add_special_tokens=False (images go via pixel_values).
                    token_ids = self._tokenizer(prompt, add_special_tokens=False)["input_ids"]
                    if not token_ids:
                        # No text to condition on. Image-prefix positions are trained
                        # with -100 labels (and trimmed by output_slice), so there is
                        # no valid position to predict an image-only first token.
                        raise ValueError("empty prompt after flattening (no text content)")
                    if self._is_video:
                        # Fixed (frames_per_clip, 3, H, W) clip + per-frame validity
                        # mask, zero-padded — identical to training. The mask hides
                        # padded-frame visual tokens from attention.
                        clip, fmask = frames_to_clip_tensor(
                            frames, max_frames=self._frames_per_clip, frame_size=self._frame_size
                        )
                        pixels = clip.to(device=self._device, dtype=self._dtype)
                        mask = fmask.to(device=self._device)
                    else:
                        pixels = _frames_to_pixel_values(
                            frames, self._frame_size, self._device, self._dtype
                        )
                        mask = None
                    prompt_tensor = torch.tensor(token_ids, dtype=torch.long, device=self._device)
                except (NotImplementedError, OSError, ValueError) as exc:
                    # Isolate a bad request (unsupported content, decode failure, empty
                    # prompt, ...) so the remaining requests still score.
                    logger.warning(
                        f"Skipping request (task={args[4]}, doc_id={args[3]}): "
                        f"{type(exc).__name__}: {exc}"
                    )
                    chunk_outputs[slot] = ""
                    continue
                # Commit atomically: only reached when every step above succeeded, so a
                # mid-request failure never leaves a partial entry in these lists.
                frames_batch.append(pixels)
                if mask is not None:
                    masks_batch.append(mask)
                prompt_ids.append(prompt_tensor)

            if prompt_ids:
                # Video: (B, F, 3, H, W) via stack (each request is one F-frame clip),
                # with a (B, F) frame mask. Image: (B, 3, H, W) via cat. cat on video
                # would fold frames into the batch and trip the frames-per-clip check.
                if self._is_video:
                    pixel_values = torch.stack(frames_batch, dim=0)
                    frame_mask = torch.stack(masks_batch, dim=0)
                else:
                    pixel_values = torch.cat(frames_batch, dim=0)
                    frame_mask = None
                try:
                    gen_outputs = _generate_batch(
                        self._model,
                        self._tokenizer,
                        pixel_values,
                        prompt_ids,
                        resolved,
                        self._max_seq_len,
                        frame_mask=frame_mask,
                    )
                except _ContextBudgetError as exc:
                    # One task's gen_kwargs over-budgets the context; skip its requests
                    # (they all share gen_kwargs) rather than aborting the whole run.
                    logger.warning(
                        f"Skipping {len(prompt_ids)} request(s) for task {chunk[0][4]}: {exc}"
                    )
                    gen_outputs = [""] * len(prompt_ids)
            else:
                gen_outputs = []
            # Scatter generated continuations back into the surviving (None) slots,
            # preserving alignment with ``chunk`` (skipped slots keep their "").
            gen_iter = iter(gen_outputs)
            outputs = [o if o is not None else next(gen_iter) for o in chunk_outputs]
            for args, output in zip(chunk, outputs, strict=True):
                results.append(output)
                self.cache_hook.add_partial("generate_until", (args[0], args[2]), output)
            pbar.update(len(chunk))
        pbar.close()
        return re_ords.get_original(results)

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        raise NotImplementedError(
            "KempnerForgeVLM is a generation-only chat model; loglikelihood is not supported. "
            "Standard multiple-choice VLM benchmarks run as generate_until tasks in lmms-eval."
        )

    def generate_until_multi_round(self, requests: list[Instance]) -> list[str]:
        raise NotImplementedError("KempnerForgeVLM does not support multi-round generation in v1.")
