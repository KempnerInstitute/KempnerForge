"""A saved checkpoint as the evaluator loads it: its step directory and its model."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

from kempnerforge.config.job import JobConfig
from kempnerforge.model.vlm import VLMWrapper, build_vlm_wrapper
from kempnerforge.resilience.elastic import resolve_resume_path

SUPPORTED_ARCHES = ("joint_decoder",)

_DCP_METADATA_FILE = ".metadata"
_SIDECAR_FILE = "metadata.json"


def check_supported(config: JobConfig) -> None:
    """Refuses a config whose checkpoints the evaluator cannot decode, before anything is built.

    The decode handles an arch of ``SUPPORTED_ARCHES`` that sees one view of one
    image per document, and it relies on a row's output not depending on the
    rows it is batched with.

    Raises:
        ValueError: the config has no ``[vlm]`` section, names another arch, has
            a ``[video]`` section, tiles images, or caps MoE expert capacity.
    """
    vlm = config.vlm
    if vlm is None:
        raise ValueError("the config has no [vlm] section; the evaluator decodes a VLM checkpoint")
    unsupported = sorted({vlm.arch, type(vlm).arch} - set(SUPPORTED_ARCHES))
    if unsupported:
        raise ValueError(
            f"vlm.arch={unsupported[0]!r} is not supported: the evaluator decodes "
            f"{list(SUPPORTED_ARCHES)} checkpoints"
        )
    if config.video is not None:
        raise ValueError(
            "a [video] config is not supported: the evaluator renders one image per document"
        )
    if config.views_per_sample != 1:
        raise ValueError(
            f"vlm.tile_grid={vlm.tile_grid} is not supported: the renderer makes one view of an "
            f"image, and this model takes {config.views_per_sample}"
        )
    if config.model.is_moe and config.model.moe_capacity_factor > 0:
        raise ValueError(
            f"model.moe_capacity_factor={config.model.moe_capacity_factor} is not supported: a "
            "capped expert drops tokens by what else is in the batch, so an answer would depend "
            "on its batch; evaluate with --model.moe_capacity_factor=0"
        )


def resolve_checkpoint(checkpoint: str | Path) -> Path:
    """The step directory ``checkpoint`` names, with symlinks followed.

    A run directory resolves to its latest complete checkpoint and a step
    directory to itself. Only the weights are loaded, so a step that lacks
    ``train_state.pt`` counts.

    Raises:
        FileNotFoundError: ``checkpoint`` is not a directory, or the directory it
            resolves to holds no saved weights.
    """
    path = Path(checkpoint)
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint {str(checkpoint)!r} is not a directory")
    step_dir = (resolve_resume_path(str(path), require_train_state=False) or path).resolve()
    if not (step_dir / _DCP_METADATA_FILE).is_file():
        raise FileNotFoundError(
            f"{step_dir} holds no saved weights (no {_DCP_METADATA_FILE} file); pass a run "
            "directory or one of its step directories"
        )
    return step_dir


def read_step(step_dir: Path) -> int | None:
    """The training step the checkpoint's ``metadata.json`` records, or ``None`` without one."""
    try:
        step = json.loads((step_dir / _SIDECAR_FILE).read_text()).get("step")
    except (OSError, ValueError, AttributeError):
        return None
    return step if type(step) is int else None


def load_model(config: JobConfig, step_dir: Path, device: torch.device) -> VLMWrapper:
    """The model ``config`` describes with the weights saved in ``step_dir``, in eval mode.

    The language model and the adapter compute in the dtype of
    ``train.mixed_precision``. The image encoder computes in
    ``vision_encoder.mixed_precision`` when that is set and otherwise in the
    dtype it loads in, which is how a training build on one device leaves it.
    The calling process loads the whole model, whatever saved it.
    """
    vision, adapter, vlm = config.vision_encoder, config.adapter, config.vlm
    assert vision is not None and adapter is not None and vlm is not None
    model = build_vlm_wrapper(
        config.model, vision, adapter, vlm, frames_per_clip=config.views_per_sample
    )
    model.transformer.to(device=device, dtype=config.train.param_dtype)
    model.adapter.to(device=device, dtype=config.train.param_dtype)
    model.vision_encoder.to(device=device)
    if vision.compute_dtype is not None:
        model.vision_encoder.to(dtype=vision.compute_dtype)
    state = {"model": model.state_dict()}
    try:
        dcp.load(state, checkpoint_id=str(step_dir), no_dist=True)
    except dcp.CheckpointException as exc:
        causes = "; ".join(f"{type(e).__name__}: {e}" for e, _ in exc.failures.values())
        raise RuntimeError(
            f"{step_dir} does not load into the model this config builds: {causes}"
        ) from exc
    model.load_state_dict(state["model"])
    return model.eval()
