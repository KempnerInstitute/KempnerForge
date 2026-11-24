#!/usr/bin/env python3
"""Write a joint-decoder VLM saved as one ``.pt`` state dict as a checkpoint to warm-start from.

The source holds a Hugging Face SigLIP vision model under
``image_encoder.model.vision_model``, an adapter of an RMS norm and a two-layer
MLP over pooled patches (``adapter.norm`` and ``adapter.proj.{0,2}``, or
``adapter.model.ln_q`` and ``adapter.model.mlp.{0,2}``), and a Qwen3 decoder as
``multimodal_core`` (Hugging Face layer names), ``text_preprocessor.embed`` and
an untied ``text_head``. The target VLM is built from ``--config``, whose
embeddings must be untied, and every one of its tensors is filled from the
source; nothing is written unless each source tensor maps onto its own target
tensor of the same shape and each target tensor is filled. The source's SigLIP attention-pooling head is
dropped when the model has none. The result is one DCP that
``[checkpoint].load_path`` warm-starts from with
``exclude_from_loading = ["optimizer", "train_state"]``.

Usage:
    uv run python examples/image_understanding/scripts/convert_pt_checkpoint.py \\
        --checkpoint path-to-stage-1.pt \\
        --config examples/image_understanding/configs/<config>.toml \\
        --out path-to-init-checkpoint
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_model_state_dict

from kempnerforge.config.loader import load_config
from kempnerforge.model.vlm import build_vlm_wrapper

logger = logging.getLogger(__name__)

_RENAMES = {
    "adapter.norm.weight": "adapter.proj.ln_q.weight",
    "adapter.proj.0.weight": "adapter.proj.proj1.weight",
    "adapter.proj.0.bias": "adapter.proj.proj1.bias",
    "adapter.proj.2.weight": "adapter.proj.proj2.weight",
    "adapter.proj.2.bias": "adapter.proj.proj2.bias",
    "adapter.model.ln_q.weight": "adapter.proj.ln_q.weight",
    "adapter.model.mlp.0.weight": "adapter.proj.proj1.weight",
    "adapter.model.mlp.0.bias": "adapter.proj.proj1.bias",
    "adapter.model.mlp.2.weight": "adapter.proj.proj2.weight",
    "adapter.model.mlp.2.bias": "adapter.proj.proj2.bias",
    "multimodal_core.norm.weight": "transformer.norm.weight",
    "text_preprocessor.embed.weight": "transformer.token_embedding.embedding.weight",
    "text_head.weight": "transformer.output_head.proj.weight",
}
_VISION_PREFIX = ("image_encoder.model.vision_model.", "vision_encoder.vision_tower.")
_POOLING_HEAD = "vision_encoder.vision_tower.head."
_LAYER_KEY = re.compile(r"multimodal_core\.layers\.(\d+)\.(.+)")
_LAYER_LEAVES = {
    "self_attn.q_proj.weight": "attention.q_proj.weight",
    "self_attn.k_proj.weight": "attention.k_proj.weight",
    "self_attn.v_proj.weight": "attention.v_proj.weight",
    "self_attn.o_proj.weight": "attention.o_proj.weight",
    "self_attn.q_norm.weight": "attention.q_norm.weight",
    "self_attn.k_norm.weight": "attention.k_norm.weight",
    "input_layernorm.weight": "attention_norm.weight",
    "post_attention_layernorm.weight": "mlp_norm.weight",
    "mlp.gate_proj.weight": "mlp.gate_proj.weight",
    "mlp.up_proj.weight": "mlp.up_proj.weight",
    "mlp.down_proj.weight": "mlp.down_proj.weight",
}


def map_key(key: str) -> str | None:
    """The target key for a source key, or None when it has no counterpart."""
    if key in _RENAMES:
        return _RENAMES[key]
    source, target = _VISION_PREFIX
    if key.startswith(source):
        return target + key[len(source) :]
    match = _LAYER_KEY.fullmatch(key)
    if match is None or match.group(2) not in _LAYER_LEAVES:
        return None
    return f"transformer.layers.{match.group(1)}.{_LAYER_LEAVES[match.group(2)]}"


def map_state_dict(source: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The source tensors under their target keys, refusing keys without one or sharing one."""
    unmapped = sorted(key for key in source if map_key(key) is None)
    if unmapped:
        raise ValueError(f"{len(unmapped)} source keys have no target: {unmapped[:10]}")
    mapped: dict[str, torch.Tensor] = {}
    for key, tensor in source.items():
        target = map_key(key)
        assert target is not None
        if target in mapped:
            raise ValueError(f"two source keys map to {target}, {key} among them")
        mapped[target] = tensor
    return mapped


def convert(checkpoint: str, config_path: str, out: str) -> Path:
    """Build the VLM of ``config_path``, fill it from ``checkpoint`` and write it as one DCP."""
    config = load_config(config_path, cli_args=[])
    if config.vlm is None or config.vision_encoder is None or config.adapter is None:
        raise ValueError(f"{config_path} does not describe a VLM")
    if config.model.tie_embeddings:
        raise ValueError(
            f"{config_path} ties its embeddings, but the source's embedding and output head "
            "are separate tensors: set model.tie_embeddings = false"
        )
    out_path = Path(out)
    if out_path.exists() and (not out_path.is_dir() or any(out_path.iterdir())):
        raise FileExistsError(f"{out_path} already exists and is not an empty directory")
    source = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(source, dict) or not all(
        isinstance(v, torch.Tensor) for v in source.values()
    ):
        raise ValueError(f"{checkpoint} is not a flat state dict of tensors")
    converted = map_state_dict(source)
    wrapper = build_vlm_wrapper(
        config.model,
        config.vision_encoder,
        config.adapter,
        config.vlm,
        frames_per_clip=config.views_per_sample,
    )
    target = wrapper.state_dict()
    converted = {
        key: tensor
        for key, tensor in converted.items()
        if key in target or not key.startswith(_POOLING_HEAD)
    }
    extra = sorted(converted.keys() - target.keys())
    missing = sorted(target.keys() - converted.keys())
    if extra or missing:
        raise ValueError(
            f"key mismatch: {len(extra)} source tensors absent from the model {extra[:10]}, "
            f"{len(missing)} model tensors missing from the source {missing[:10]}"
        )
    bad_shapes = [
        (key, tuple(converted[key].shape), tuple(target[key].shape))
        for key in sorted(converted)
        if converted[key].shape != target[key].shape
    ]
    if bad_shapes:
        raise ValueError(f"shape mismatch (source, model): {bad_shapes[:10]}")
    wrapper.load_state_dict({k: v.to(target[k].dtype) for k, v in converted.items()}, strict=True)
    state = get_model_state_dict(wrapper)
    out_path.mkdir(parents=True, exist_ok=True)
    dcp.save({"model": state}, checkpoint_id=str(out_path))
    logger.info("Wrote %d tensors from %s to %s", len(state), checkpoint, out_path)
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--checkpoint", required=True, help="source .pt state dict")
    parser.add_argument("--config", required=True, help="target VLM config (TOML)")
    parser.add_argument("--out", required=True, help="output DCP directory (new or empty)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    convert(args.checkpoint, args.config, args.out)


if __name__ == "__main__":
    main()
