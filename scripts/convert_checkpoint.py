#!/usr/bin/env python3
"""Checkpoint format conversion: DCP (KempnerForge) <-> HuggingFace.

Enables:
  - Exporting a KempnerForge DCP checkpoint to HuggingFace format for inference
  - Importing HuggingFace pretrained weights into KempnerForge format for fine-tuning

Usage:
    # DCP → HuggingFace (export for inference)
    uv run python scripts/convert_checkpoint.py dcp-to-hf \
        --dcp-dir checkpoints/step_10000 \
        --hf-dir exports/my_model \
        --config configs/model/llama_7b.toml

    # HuggingFace → DCP (import for fine-tuning)
    uv run python scripts/convert_checkpoint.py hf-to-dcp \
        --hf-dir meta-llama/Llama-3.1-8B \
        --dcp-dir checkpoints/pretrained \
        --config configs/model/llama_7b.toml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

from kempnerforge.config.loader import load_config
from kempnerforge.config.schema import ModelConfig
from kempnerforge.model.transformer import Transformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _kf_to_hf_key(key: str) -> str:
    """Convert a KempnerForge state dict key to HuggingFace format.

    KempnerForge: layers.{i}.attention.q_proj.weight
    HuggingFace:  model.layers.{i}.self_attn.q_proj.weight

    MoE keys:
      KF: layers.{i}.mlp.router.gate.weight → HF: model.layers.{i}.mlp.gate.weight
      KF: layers.{i}.mlp.shared_expert.*    → HF: model.layers.{i}.mlp.shared_experts.*
      KF: layers.{i}.mlp.experts.{j}.*      → HF: model.layers.{i}.mlp.experts.{j}.*
    """
    key = key.replace("token_embedding.embedding.weight", "model.embed_tokens.weight")
    key = key.replace("output_head.proj.weight", "lm_head.weight")
    if key.startswith("layers."):
        key = "model." + key
        key = key.replace(".attention_norm.", ".input_layernorm.")
        key = key.replace(".mlp_norm.", ".post_attention_layernorm.")
        key = key.replace(".attention.", ".self_attn.")
        key = key.replace(".mlp.router.gate.", ".mlp.gate.")
        key = key.replace(".mlp.router.expert_bias", ".mlp.expert_bias")
        key = key.replace(".mlp.shared_expert.", ".mlp.shared_experts.")
    elif key == "norm.weight":
        key = "model.norm.weight"
    return key


def _hf_to_kf_key(key: str) -> str:
    """Convert a HuggingFace state dict key to KempnerForge format."""
    key = key.replace("model.embed_tokens.weight", "token_embedding.embedding.weight")
    key = key.replace("lm_head.weight", "output_head.proj.weight")
    key = key.replace("model.norm.weight", "norm.weight")
    if key.startswith("model.layers."):
        key = key.replace("model.layers.", "layers.", 1)
        key = key.replace(".self_attn.", ".attention.")
        key = key.replace(".input_layernorm.", ".attention_norm.")
        key = key.replace(".post_attention_layernorm.", ".mlp_norm.")
        key = key.replace(".mlp.gate.weight", ".mlp.router.gate.weight")
        key = key.replace(".mlp.expert_bias", ".mlp.router.expert_bias")
        key = key.replace(".mlp.shared_experts.", ".mlp.shared_expert.")
    return key


def _detect_pp_stages(dcp_path: Path) -> list[Path]:
    """Detect pipeline-parallel stage subdirectories (pp0/, pp1/, ...)."""
    pp_dirs = sorted(
        (d for d in dcp_path.iterdir() if d.is_dir() and d.name.startswith("pp")),
        key=lambda d: int(d.name[2:]),
    )
    return pp_dirs


def dcp_to_hf(
    dcp_dir: str,
    hf_dir: str,
    model_config: ModelConfig,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Convert a DCP checkpoint to HuggingFace safetensors format.

    Handles both standard and pipeline-parallel checkpoints (pp0/, pp1/, ...).
    A tensor that still shares storage with one already written after the cast,
    such as a tied output head at float32, is written as a copy, since
    safetensors refuses tensors that share memory.

    Args:
        dcp_dir: Path to DCP checkpoint directory (e.g., checkpoints/step_10000).
        hf_dir: Output directory for HuggingFace model files.
        model_config: Model configuration for building the model skeleton.
        dtype: Export dtype (default: bfloat16), recorded as ``torch_dtype`` in
            the written ``config.json``.
    """
    dcp_path = Path(dcp_dir)
    logger.info(f"Loading DCP checkpoint from {dcp_dir}")

    model = Transformer(model_config)

    pp_dirs = _detect_pp_stages(dcp_path)
    if pp_dirs:
        logger.info(f"Detected {len(pp_dirs)} PP stages: {[d.name for d in pp_dirs]}")
        for pp_path in pp_dirs:
            stage_state = {"model": model.state_dict()}
            dcp.load(stage_state, checkpoint_id=str(pp_path))
            model.load_state_dict(stage_state["model"], strict=False)
    else:
        state_dict = {"model": model.state_dict()}
        dcp.load(state_dict, checkpoint_id=dcp_dir)
        model.load_state_dict(state_dict["model"])

    hf_state = {}
    seen_storage: set[int] = set()
    for kf_key, tensor in model.state_dict().items():
        hf_key = _kf_to_hf_key(kf_key)
        cast = tensor.to(dtype)
        ptr = cast.untyped_storage().data_ptr()
        hf_state[hf_key] = cast.clone() if ptr in seen_storage else cast
        seen_storage.add(ptr)

    hf_path = Path(hf_dir)
    hf_path.mkdir(parents=True, exist_ok=True)

    try:
        from safetensors.torch import save_file

        save_file(hf_state, hf_path / "model.safetensors")
        logger.info(f"Saved HuggingFace checkpoint (safetensors): {hf_path}")
    except ImportError:
        torch.save(hf_state, hf_path / "pytorch_model.bin")
        logger.info(f"Saved HuggingFace checkpoint (torch): {hf_path}")

    hf_config = _build_hf_config(model_config, dtype)
    (hf_path / "config.json").write_text(json.dumps(hf_config, indent=2))
    logger.info(f"Wrote config.json with {len(hf_state)} tensors")


def hf_to_dcp(
    hf_dir: str,
    dcp_dir: str,
    model_config: ModelConfig,
) -> None:
    """Convert a HuggingFace checkpoint to DCP format.

    Supports loading from a local directory or a HuggingFace model ID.

    Args:
        hf_dir: Path to HF model directory or HF model ID.
        dcp_dir: Output directory for DCP checkpoint.
        model_config: Model configuration matching the HF model architecture.
    """
    hf_path = Path(hf_dir)

    hf_state = _load_hf_local(hf_path) if hf_path.exists() else _load_hf_hub(hf_dir)

    logger.info(f"Loaded HuggingFace checkpoint: {len(hf_state)} tensors")

    model = Transformer(model_config)
    kf_state = model.state_dict()

    converted = {}
    unmapped = []
    for hf_key, tensor in hf_state.items():
        kf_key = _hf_to_kf_key(hf_key)
        if kf_key in kf_state:
            converted[kf_key] = tensor
        else:
            unmapped.append(hf_key)

    if unmapped:
        logger.warning(f"Skipped {len(unmapped)} unmapped HF keys: {unmapped[:5]}...")

    missing = set(kf_state.keys()) - set(converted.keys())
    if missing:
        logger.warning(f"Missing {len(missing)} keys (using random init): {list(missing)[:5]}...")

    model.load_state_dict(converted, strict=False)

    dcp_path = Path(dcp_dir)
    dcp_path.mkdir(parents=True, exist_ok=True)
    dcp.save({"model": model.state_dict()}, checkpoint_id=str(dcp_path))
    logger.info(f"Saved DCP checkpoint: {dcp_path}")


def _load_hf_local(path: Path) -> dict[str, torch.Tensor]:
    """Load state dict from a local HuggingFace model directory."""
    safetensor_files = list(path.glob("*.safetensors"))
    if safetensor_files:
        from safetensors.torch import load_file

        state = {}
        for f in sorted(safetensor_files):
            state.update(load_file(f))
        return state

    bin_files = list(path.glob("pytorch_model*.bin"))
    if bin_files:
        state = {}
        for f in sorted(bin_files):
            state.update(torch.load(f, map_location="cpu", weights_only=True))
        return state

    raise FileNotFoundError(f"No model files found in {path}")


def _load_hf_hub(model_id: str) -> dict[str, torch.Tensor]:
    """Load state dict from HuggingFace Hub."""
    from transformers import AutoModelForCausalLM

    logger.info(f"Downloading model from HuggingFace Hub: {model_id}")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    return hf_model.state_dict()


def _build_hf_config(config: ModelConfig, dtype: torch.dtype = torch.bfloat16) -> dict:
    """Build a HuggingFace-compatible config.json from ModelConfig.

    ``torch_dtype`` names ``dtype``, the dtype the weights are written in, so
    ``from_pretrained(..., dtype="auto")`` loads them at that precision.
    """
    if config.dim % config.n_heads != 0:
        raise ValueError(
            f"Cannot export dim ({config.dim}) not divisible by n_heads ({config.n_heads}): "
            "the target architectures require it even when head_dim is set explicitly."
        )
    hf_config = {
        "hidden_size": config.dim,
        "intermediate_size": config.computed_ffn_hidden_dim,
        "num_hidden_layers": config.n_layers,
        "num_attention_heads": config.n_heads,
        "num_key_value_heads": config.n_kv_heads,
        "head_dim": config.head_dim,
        "vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_seq_len,
        "rope_theta": config.rope_theta,
        "rms_norm_eps": config.norm_eps,
        "tie_word_embeddings": config.tie_embeddings,
        "torch_dtype": str(dtype).removeprefix("torch."),
    }

    if config.is_moe:
        hf_config["architectures"] = ["MixtralForCausalLM"]
        hf_config["model_type"] = "mixtral"
        hf_config["num_local_experts"] = config.num_experts
        hf_config["num_experts_per_tok"] = config.moe_top_k
        hf_config["router_aux_loss_coef"] = config.moe_aux_loss_weight
    else:
        hf_config["architectures"] = ["LlamaForCausalLM"]
        hf_config["model_type"] = "llama"

    return hf_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert checkpoints between DCP and HuggingFace formats"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_export = subparsers.add_parser("dcp-to-hf", help="Export DCP checkpoint to HuggingFace")
    p_export.add_argument("--dcp-dir", required=True, help="DCP checkpoint directory")
    p_export.add_argument("--hf-dir", required=True, help="Output HuggingFace directory")
    p_export.add_argument("--config", required=True, help="TOML config file")
    p_export.add_argument(
        "--dtype",
        default=None,
        choices=["float32", "bfloat16", "float16"],
        help="Export dtype; overrides checkpoint.export_dtype from --config "
        "(default: that field, which is bfloat16 unless the TOML sets it)",
    )

    p_import = subparsers.add_parser("hf-to-dcp", help="Import HuggingFace weights to DCP")
    p_import.add_argument("--hf-dir", required=True, help="HF model directory or model ID")
    p_import.add_argument("--dcp-dir", required=True, help="Output DCP directory")
    p_import.add_argument("--config", required=True, help="TOML config file")

    args = parser.parse_args()
    config = load_config(args.config, cli_args=[])

    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }

    if args.command == "dcp-to-hf":
        if args.dtype is not None:
            export_dtype, source = args.dtype, "--dtype"
        else:
            export_dtype, source = config.checkpoint.export_dtype, "checkpoint.export_dtype"
        logger.info(f"Export dtype: {export_dtype} (from {source})")
        dcp_to_hf(
            dcp_dir=args.dcp_dir,
            hf_dir=args.hf_dir,
            model_config=config.model,
            dtype=dtype_map[export_dtype],
        )
    elif args.command == "hf-to-dcp":
        hf_to_dcp(
            hf_dir=args.hf_dir,
            dcp_dir=args.dcp_dir,
            model_config=config.model,
        )


if __name__ == "__main__":
    main()
