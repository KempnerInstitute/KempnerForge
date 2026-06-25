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

- **Generation: cache-less, single-GPU, batched.** The decode loop re-runs the
  full ``VLMWrapper.forward`` over the growing sequence each step. There is no
  transformer KV cache (``Transformer.forward`` forbids combining ``kv_caches``
  with any image-conditioning route), and KempnerForge has no
  image-conditioned KV-cache decode path. Requests are decoded in batches
  (``batch_size`` model-arg) by **right-padding** the text to the batch-max
  length — the same layout training uses (image prefix at ``0..n-1``, text
  contiguous from ``n``, trailing pads causally masked) — and reading each
  row's logits at its own last real position. Single-GPU is the validated
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
from lmms_eval.utils import Collator
from tqdm import tqdm

from kempnerforge.config.job import JobConfig
from kempnerforge.config.loader import load_config
from kempnerforge.data.vlm_dataset import (
    DEFAULT_IMAGE_MEAN,
    DEFAULT_IMAGE_STD,
    build_tokenizer,
    pil_to_tensor,
    resolve_pad_id,
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
        if isinstance(frame, str):
            with Image.open(frame) as im:
                img = im.copy()
        else:
            img = frame
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
def _generate_batch(
    model: VLMWrapper,
    tokenizer: Any,
    pixel_values: torch.Tensor,
    prompt_ids: list[torch.Tensor],
    resolved: dict[str, Any],
    max_seq_len: int,
) -> list[str]:
    """Cache-less batched decode; returns one continuation per request.

    Decodes ``B`` requests together (``pixel_values`` is ``(B, 3, H, W)``,
    ``prompt_ids`` a list of ``B`` 1-D token tensors). Re-runs
    ``model(pixel_values, input_ids)`` over the growing **right-padded** batch
    each step (no transformer KV cache; vision re-encoded per step — see module
    docstring). Right-padding matches the training layout: the image prefix
    stays at positions ``0..n-1`` and text is contiguous from ``n`` for every
    row (so image/text RoPE distances are consistent across rows), and the
    trailing pads are causally masked, so a batched forward gives each row the
    same real-position logits as decoding it alone. Each row's next token is
    read at its own last real position; EOS / ``max_new_tokens`` / first
    ``until`` match are tracked per row. ``B == 1`` reproduces the
    single-request path exactly.
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
        raise ValueError(
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

        logits, _ = model(pixel_values, input_ids)
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
    - ``device`` (default ``"cuda"``), ``dtype`` (default ``"bfloat16"``).
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
        if self._batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self._batch_size}")
        if self._default_max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self._default_max_new_tokens}")
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
            frames_batch: list[torch.Tensor] = []
            prompt_ids: list[torch.Tensor] = []
            for args in chunk:
                # Chat 6-tuple: (context, doc_to_messages, gen_kwargs, doc_id, task, split).
                doc = self.task_dict[args[4]][args[5]][args[3]]
                messages = ChatMessages(messages=args[1](doc))
                frames, prompt = _render_request(messages)
                frames_batch.append(
                    _frames_to_pixel_values(
                        frames, self._config.data.hf_image_size, self._device, self._dtype
                    )
                )
                # Mirror training tokenization: no chat template, no <image>
                # placeholder, add_special_tokens=False (images go via pixel_values).
                prompt_ids.append(
                    torch.tensor(
                        self._tokenizer(prompt, add_special_tokens=False)["input_ids"],
                        dtype=torch.long,
                        device=self._device,
                    )
                )
            pixel_values = torch.cat(frames_batch, dim=0)
            outputs = _generate_batch(
                self._model, self._tokenizer, pixel_values, prompt_ids, resolved, self._max_seq_len
            )
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
