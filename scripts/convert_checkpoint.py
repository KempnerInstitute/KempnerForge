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


# ---------------------------------------------------------------------------
# Key mapping: KempnerForge <-> HuggingFace (Llama-style)
# ---------------------------------------------------------------------------

# KempnerForge uses the same naming as Llama models with minor differences.
# This mapping handles the translation.


def _kf_to_hf_key(key: str) -> str:
    """Convert a KempnerForge state dict key to HuggingFace format.

    KempnerForge: layers.{i}.attention.q_proj.weight
    HuggingFace:  model.layers.{i}.self_attn.q_proj.weight
    """
    # Embedding
    key = key.replace("token_embedding.embedding.weight", "model.embed_tokens.weight")
    # Output head
    key = key.replace("output_head.proj.weight", "lm_head.weight")
    # Transformer blocks
    if key.startswith("layers."):
        key = "model." + key
        # Norms (before attention rename to avoid partial matches)
        key = key.replace(".attention_norm.", ".input_layernorm.")
        key = key.replace(".mlp_norm.", ".post_attention_layernorm.")
        # Attention
        key = key.replace(".attention.", ".self_attn.")
    # Final norm (top-level only, after block handling)
    elif key == "norm.weight":
        key = "model.norm.weight"
    return key


def _hf_to_kf_key(key: str) -> str:
    """Convert a HuggingFace state dict key to KempnerForge format."""
    # Embedding
    key = key.replace("model.embed_tokens.weight", "token_embedding.embedding.weight")
    # Output head
    key = key.replace("lm_head.weight", "output_head.proj.weight")
    # Final norm
    key = key.replace("model.norm.weight", "norm.weight")
    # Transformer blocks
    if key.startswith("model.layers."):
        key = key.replace("model.layers.", "layers.", 1)
        # Attention
        key = key.replace(".self_attn.", ".attention.")
        # Norms
        key = key.replace(".input_layernorm.", ".attention_norm.")
        key = key.replace(".post_attention_layernorm.", ".mlp_norm.")
    return key


# ---------------------------------------------------------------------------
# DCP → HuggingFace
# ---------------------------------------------------------------------------


def dcp_to_hf(
    dcp_dir: str,
    hf_dir: str,
    model_config: ModelConfig,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Convert a DCP checkpoint to HuggingFace safetensors format.

    Args:
        dcp_dir: Path to DCP checkpoint directory (e.g., checkpoints/step_10000).
        hf_dir: Output directory for HuggingFace model files.
        model_config: Model configuration for building the model skeleton.
        dtype: Export dtype (default: bfloat16).
    """
    logger.info(f"Loading DCP checkpoint from {dcp_dir}")

    # Build model on CPU to load state into
    model = Transformer(model_config)

    # Load DCP state into model
    state_dict = {"model": model.state_dict()}
    dcp.load(state_dict, checkpoint_id=dcp_dir)
    model.load_state_dict(state_dict["model"])

    # Convert keys and dtype
    hf_state = {}
    for kf_key, tensor in model.state_dict().items():
        hf_key = _kf_to_hf_key(kf_key)
        hf_state[hf_key] = tensor.to(dtype)

    # Save
    hf_path = Path(hf_dir)
    hf_path.mkdir(parents=True, exist_ok=True)

    # Try safetensors first, fall back to torch
    try:
        from safetensors.torch import save_file

        save_file(hf_state, hf_path / "model.safetensors")
        logger.info(f"Saved HuggingFace checkpoint (safetensors): {hf_path}")
    except ImportError:
        torch.save(hf_state, hf_path / "pytorch_model.bin")
        logger.info(f"Saved HuggingFace checkpoint (torch): {hf_path}")

    # Write config.json for HF compatibility
    hf_config = _build_hf_config(model_config)
    (hf_path / "config.json").write_text(json.dumps(hf_config, indent=2))
    logger.info(f"Wrote config.json with {len(hf_state)} tensors")


# ---------------------------------------------------------------------------
# HuggingFace → DCP
# ---------------------------------------------------------------------------


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

    # Load HuggingFace state dict
    hf_state = _load_hf_local(hf_path) if hf_path.exists() else _load_hf_hub(hf_dir)

    logger.info(f"Loaded HuggingFace checkpoint: {len(hf_state)} tensors")

    # Build KempnerForge model
    model = Transformer(model_config)
    kf_state = model.state_dict()

    # Convert keys
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

    # Check for missing keys
    missing = set(kf_state.keys()) - set(converted.keys())
    if missing:
        logger.warning(f"Missing {len(missing)} keys (using random init): {list(missing)[:5]}...")

    # Load converted weights
    model.load_state_dict(converted, strict=False)

    # Save as DCP
    dcp_path = Path(dcp_dir)
    dcp_path.mkdir(parents=True, exist_ok=True)
    dcp.save({"model": model.state_dict()}, checkpoint_id=str(dcp_path))
    logger.info(f"Saved DCP checkpoint: {dcp_path}")


# ---------------------------------------------------------------------------
# HuggingFace loading helpers
# ---------------------------------------------------------------------------


def _load_hf_local(path: Path) -> dict[str, torch.Tensor]:
    """Load state dict from a local HuggingFace model directory."""
    # Try safetensors first
    safetensor_files = list(path.glob("*.safetensors"))
    if safetensor_files:
        from safetensors.torch import load_file

        state = {}
        for f in sorted(safetensor_files):
            state.update(load_file(f))
        return state

    # Fall back to torch .bin files
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
        model_id, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    return hf_model.state_dict()


def _build_hf_config(config: ModelConfig) -> dict:
    """Build a HuggingFace-compatible config.json from ModelConfig."""
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": config.dim,
        "intermediate_size": config.computed_ffn_hidden_dim,
        "num_hidden_layers": config.n_layers,
        "num_attention_heads": config.n_heads,
        "num_key_value_heads": config.n_kv_heads,
        "vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_seq_len,
        "rope_theta": config.rope_theta,
        "rms_norm_eps": config.norm_eps,
        "tie_word_embeddings": config.tie_embeddings,
        "torch_dtype": "bfloat16",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert checkpoints between DCP and HuggingFace formats"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # DCP → HF
    p_export = subparsers.add_parser("dcp-to-hf", help="Export DCP checkpoint to HuggingFace")
    p_export.add_argument("--dcp-dir", required=True, help="DCP checkpoint directory")
    p_export.add_argument("--hf-dir", required=True, help="Output HuggingFace directory")
    p_export.add_argument("--config", required=True, help="TOML config file")
    p_export.add_argument(
        "--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"],
    )

    # HF → DCP
    p_import = subparsers.add_parser("hf-to-dcp", help="Import HuggingFace weights to DCP")
    p_import.add_argument("--hf-dir", required=True, help="HF model directory or model ID")
    p_import.add_argument("--dcp-dir", required=True, help="Output DCP directory")
    p_import.add_argument("--config", required=True, help="TOML config file")

    args = parser.parse_args()
    config = load_config(args.config)

    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }

    if args.command == "dcp-to-hf":
        dcp_to_hf(
            dcp_dir=args.dcp_dir,
            hf_dir=args.hf_dir,
            model_config=config.model,
            dtype=dtype_map[args.dtype],
        )
    elif args.command == "hf-to-dcp":
        hf_to_dcp(
            hf_dir=args.hf_dir,
            dcp_dir=args.dcp_dir,
            model_config=config.model,
        )


if __name__ == "__main__":
    main()
