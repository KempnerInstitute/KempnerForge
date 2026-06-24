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

- **Generation: cache-less, single-GPU, batch 1.** The decode loop re-runs the
  full ``VLMWrapper.forward`` over the growing sequence each step. There is no
  transformer KV cache (``Transformer.forward`` forbids combining ``kv_caches``
  with any image-conditioning route), and KempnerForge has no
  image-conditioned KV-cache decode path. Single-GPU is the validated
  invocation, not a baked-in assumption: rank/world_size come from the lmms
  base (defaults 0/1) and model construction sits behind ``_build_model`` so a
  data-parallel path is a localized future change.

- **Vision re-encoded per step.** ``ModalityStrategy.prepare`` re-runs the
  vision encoder + adapter internally on every forward, and there is no
  arch-agnostic public seam on ``VLMWrapper`` to pass precomputed image
  features. Encoding the image exactly once per request would require a
  model-side change (a strategy method accepting cached embeds).

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

- **Images only.** A request must carry exactly one image. Video/audio content,
  multi-image, and multi-turn/few-shot requests raise ``NotImplementedError``;
  ``loglikelihood`` and ``generate_until_multi_round`` are not implemented
  (chat tasks are generation-only). Visual input is modeled as an ordered list
  of frames (a single image is the length-1 case) so video is a localized
  future addition.
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
from tqdm import tqdm

from kempnerforge.config.job import JobConfig
from kempnerforge.config.loader import load_config
from kempnerforge.data.vlm_dataset import (
    DEFAULT_IMAGE_MEAN,
    DEFAULT_IMAGE_STD,
    build_tokenizer,
    pil_to_tensor,
)
from kempnerforge.metrics.logger import get_logger
from kempnerforge.model.generate import sample
from kempnerforge.model.vlm import VLMWrapper, build_vlm_wrapper
from kempnerforge.resilience.elastic import resolve_resume_path

logger = get_logger(__name__)

# Arches whose routing cannot autoregressively generate.
UNSUPPORTED_GEN_ARCHS = frozenset({"moma"})

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
    model = build_vlm_wrapper(config.model, config.vision_encoder, config.adapter, config.vlm)
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


def _check_generative_arch(arch: str) -> None:
    """Fail fast (before building) on arches that cannot autoregressively generate."""
    if arch in UNSUPPORTED_GEN_ARCHS:
        raise ValueError(
            f"VLM arch {arch!r} cannot be evaluated: its routing is non-causal and cannot "
            f"autoregressively generate, but chat tasks are generation-only. Supported arches: "
            f"joint_decoder, cross_attention, mot. Generation support for {arch!r} is a tracked "
            f"model-side follow-up — contact the project owner."
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


def _render_request(messages: ChatMessages) -> tuple[list[Any], str]:
    """Flatten one chat request into ``(image_frames, prompt_text)``.

    v1 is single-turn, zero-shot, image-only. Raises ``NotImplementedError``
    for content that violates those assumptions (video/audio, multi-turn or
    few-shot, or anything other than exactly one image) so the offending task
    is surfaced rather than silently mishandled. Text content blocks are
    concatenated in message order (newline-joined); role/turn structure is
    intentionally discarded (see the module docstring on flattening).
    """
    images, videos, audios = messages.extract_media()
    if videos:
        raise NotImplementedError(
            "Video evaluation is not implemented in v1 (image-only). A video request "
            "reached the KempnerForge VLM adapter; report the task to the project owner."
        )
    if audios:
        raise NotImplementedError(
            "Audio evaluation is not implemented in v1 (image-only). An audio request "
            "reached the KempnerForge VLM adapter; report the task to the project owner."
        )

    roles = [message.role for message in messages.messages]
    if any(role == "assistant" for role in roles) or roles.count("user") > 1:
        raise NotImplementedError(
            "Multi-turn / few-shot requests are not supported in v1 (single-turn, "
            "zero-shot only). Report the task to the project owner."
        )
    if len(images) != 1:
        raise NotImplementedError(
            f"v1 supports exactly one image per request, got {len(images)}. Multi-image "
            "and text-only requests are out of scope; report the task to the project owner."
        )

    parts = [
        content.text
        for message in messages.messages
        for content in message.content
        if content.type == "text"
    ]
    return images, "\n".join(parts)


def _frames_to_pixel_values(
    frames: list[Any], image_size: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Convert an ordered list of frames to a ``(num_frames, 3, H, W)`` tensor.

    Reuses the training-time ``pil_to_tensor`` (resize + SigLIP-style
    normalize) as the single source of truth. v1 passes a single image (a
    length-1 list); the list shape is the seam for future video (ordered
    frames). Each frame may be a ``PIL.Image`` or a path string.
    """
    from PIL import Image

    tensors = []
    for frame in frames:
        img = Image.open(frame) if isinstance(frame, str) else frame
        tensors.append(pil_to_tensor(img, image_size, DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD))
    return torch.stack(tensors, dim=0).to(device=device, dtype=dtype)


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


def _resolve_gen_kwargs(gen_kwargs: dict[str, Any], default_max_new_tokens: int) -> dict[str, Any]:
    """Merge task ``gen_kwargs`` over the adapter's fallback defaults."""
    until = gen_kwargs.get("until") or []
    if isinstance(until, str):
        until = [until]

    max_new_tokens = gen_kwargs.get("max_new_tokens") or default_max_new_tokens
    temperature = gen_kwargs.get("temperature") or 0.0
    # An explicit do_sample=False forces greedy even if a temperature is given.
    if not gen_kwargs.get("do_sample", temperature > 0):
        temperature = 0.0

    sampling = temperature > 0
    return {
        "until": [u for u in until if u],
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "top_k": int(gen_kwargs.get("top_k") or 0) if sampling else 0,
        "top_p": float(gen_kwargs.get("top_p") or 1.0) if sampling else 1.0,
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
def _generate_one(
    model: VLMWrapper,
    tokenizer: Any,
    pixel_values: torch.Tensor,
    prompt_ids: torch.Tensor,
    resolved: dict[str, Any],
    max_seq_len: int,
) -> str:
    """Cache-less decode for one request (batch 1); returns the continuation.

    Re-runs ``model(pixel_values, seq)`` over the growing sequence each step
    (no transformer KV cache; vision re-encoded per step — see module
    docstring), selects the next token with KempnerForge's ``sample`` using the
    resolved ``gen_kwargs``, and stops on EOS, ``max_new_tokens``, or the first
    ``until`` match (trimming the continuation at that match).
    """
    until: list[str] = resolved["until"]
    max_new_tokens: int = resolved["max_new_tokens"]
    temperature: float = resolved["temperature"]
    top_k: int = resolved["top_k"]
    top_p: float = resolved["top_p"]
    eos_id = tokenizer.eos_token_id

    num_image_tokens = model.num_image_tokens
    prompt_budget = max_seq_len - num_image_tokens - max_new_tokens
    if prompt_budget <= 0:
        raise ValueError(
            f"max_new_tokens ({max_new_tokens}) + image tokens ({num_image_tokens}) leave no "
            f"room for the prompt within max_seq_len ({max_seq_len}); lower --max-new-tokens."
        )
    if prompt_ids.shape[1] > prompt_budget:
        logger.warning(
            f"Prompt ({prompt_ids.shape[1]}) + image tokens ({num_image_tokens}) + "
            f"max_new_tokens ({max_new_tokens}) exceeds max_seq_len ({max_seq_len}); "
            f"left-truncating prompt to {prompt_budget} tokens. Severe truncation may "
            f"distort results — report the task to the project owner if so."
        )
        prompt_ids = prompt_ids[:, -prompt_budget:]

    seq = prompt_ids
    generated: list[int] = []
    for _ in range(max_new_tokens):
        logits, _ = model(pixel_values, seq)
        next_token = sample(logits[:, -1, :], temperature, top_k, top_p)
        token_id = int(next_token.item())
        if eos_id is not None and token_id == eos_id:
            break
        generated.append(token_id)
        seq = torch.cat([seq, next_token.view(1, 1)], dim=1)
        if until:
            text = tokenizer.decode(generated, skip_special_tokens=True)
            cut = _first_stop(text, until)
            if cut is not None:
                return text[:cut]
    return tokenizer.decode(generated, skip_special_tokens=True)


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
    - ``device`` (default ``"cuda"``), ``dtype`` (default ``"bfloat16"``).
    - ``batch_size`` (default ``1``): recorded for parity; v1 decodes one
      request at a time.
    - ``max_new_tokens`` (default ``128``): fallback only; task ``gen_kwargs``
      override it.
    """

    is_simple = False

    def __init__(
        self,
        config: str,
        checkpoint: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        batch_size: int | str = 1,
        max_new_tokens: int | str = DEFAULT_MAX_NEW_TOKENS,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if kwargs:
            logger.warning(f"Ignoring unsupported model_args: {sorted(kwargs)}")

        self._device = torch.device(device)
        self._dtype = _resolve_dtype(dtype)
        self._batch_size = int(batch_size)
        self._default_max_new_tokens = int(max_new_tokens)

        self._config = _load_config(config)
        assert self._config.vlm is not None  # guaranteed by is_vlm; narrows for the type checker
        self._arch = self._config.vlm.arch
        # Fail fast on non-generative arches before building/loading the model.
        _check_generative_arch(self._arch)

        self._model = _load_weights(self._config, checkpoint, self._device, self._dtype)
        self._tokenizer = build_tokenizer(self._config.data.tokenizer_path)
        self._max_seq_len = self._config.model.max_seq_len
        logger.info(
            f"KempnerForgeVLM ready: arch={self._arch}, device={self._device}, "
            f"dtype={self._dtype}, max_seq_len={self._max_seq_len}"
        )

    def generate_until(self, requests: list[Instance]) -> list[str]:
        results: list[str] = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="KempnerForge VLM")
        for request in requests:
            # Instance.args is an untyped tuple; for chat generate_until it is the
            # 6-tuple (context, doc_to_messages, gen_kwargs, doc_id, task, split).
            args: tuple[Any, ...] = request.args
            context, doc_to_messages, gen_kwargs, doc_id, task, split = args
            doc = self.task_dict[task][split][doc_id]
            messages = ChatMessages(messages=doc_to_messages(doc))

            frames, prompt = _render_request(messages)
            pixel_values = _frames_to_pixel_values(
                frames, self._config.data.hf_image_size, self._device, self._dtype
            )
            # Mirror training tokenization: no chat template, no <image> placeholder,
            # add_special_tokens=False (images are conditioned via pixel_values).
            prompt_ids = torch.tensor(
                [self._tokenizer(prompt, add_special_tokens=False)["input_ids"]],
                dtype=torch.long,
                device=self._device,
            )
            resolved = _resolve_gen_kwargs(gen_kwargs, self._default_max_new_tokens)
            output = _generate_one(
                self._model, self._tokenizer, pixel_values, prompt_ids, resolved, self._max_seq_len
            )

            results.append(output)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), output)
            pbar.update(1)
        pbar.close()
        return results

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        raise NotImplementedError(
            "KempnerForgeVLM is a generation-only chat model; loglikelihood is not supported. "
            "Standard multiple-choice VLM benchmarks run as generate_until tasks in lmms-eval."
        )

    def generate_until_multi_round(self, requests: list[Instance]) -> list[str]:
        raise NotImplementedError("KempnerForgeVLM does not support multi-round generation in v1.")
