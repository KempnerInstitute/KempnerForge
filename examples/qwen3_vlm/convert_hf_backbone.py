#!/usr/bin/env python3
"""Build a complete KempnerForge VLM checkpoint from pretrained components.

Builds the VLM via the public factory (vision encoder from HuggingFace, fresh
adapter), copies an HF causal-LM's weights into the transformer backbone, and
saves the whole model as one DCP that ``[checkpoint].load_path`` reads. Tied
embeddings are honored.

The vision encoder is whatever ``[vision_encoder]`` names, built via the registry.
The backbone must be an HF causal LM using the Llama-style naming Qwen3 follows
(``self_attn.{q,k,v,o}_proj``, ``input_layernorm``, ``mlp.{gate,up,down}_proj``)
and a dense arch; anything else fails loudly, naming the unmapped keys.

Usage:
    uv run python examples/qwen3_vlm/convert_hf_backbone.py \\
        --hf-dir Qwen/Qwen3-0.6B \\
        --config examples/qwen3_vlm/configs/<vlm_config>.toml \\
        --out path-to-init-checkpoint
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

from kempnerforge.config.loader import load_config
from kempnerforge.model.vlm import build_vlm_wrapper

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_EMBED_KEY = "token_embedding.embedding.weight"
_HEAD_KEY = "output_head.proj.weight"


def map_key(hf_key: str) -> str | None:
    """Map an HF causal-LM key to its KempnerForge key; None if not a parameter."""
    if hf_key == "model.embed_tokens.weight":
        return _EMBED_KEY
    if hf_key == "lm_head.weight":
        return _HEAD_KEY
    if hf_key == "model.norm.weight":
        return "norm.weight"
    if hf_key.startswith("model.layers."):
        body = hf_key[len("model.") :]  # layers.{i}.<rest>
        body = body.replace(".self_attn.", ".attention.")
        body = body.replace(".input_layernorm.", ".attention_norm.")
        body = body.replace(".post_attention_layernorm.", ".mlp_norm.")
        return body
    return None


def map_state_dict(
    hf_sd: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Apply ``map_key`` across a source state dict -> ``(converted, unmapped)``."""
    converted: dict[str, torch.Tensor] = {}
    unmapped: list[str] = []
    for hf_key, tensor in hf_sd.items():
        kf_key = map_key(hf_key)
        if kf_key is None:
            unmapped.append(hf_key)
        else:
            converted[kf_key] = tensor
    return converted, unmapped


def fill_tied_head(converted: dict[str, torch.Tensor], expected_keys: set[str]) -> bool:
    """Fill a tied output head from the token embedding, in place.

    Returns True when filled: the target ties the head to the embedding and the
    source omitted ``lm_head``, so both keys share one weight.
    """
    if _HEAD_KEY in expected_keys and _HEAD_KEY not in converted and _EMBED_KEY in converted:
        converted[_HEAD_KEY] = converted[_EMBED_KEY]
        return True
    return False


def _load_hf_state_dict(path: str) -> dict[str, torch.Tensor]:
    """Load an HF causal-LM state dict from a local dir or a Hub model id."""
    p = Path(path)
    if p.exists():
        safetensor_files = sorted(p.glob("*.safetensors"))
        if safetensor_files:
            from safetensors.torch import load_file

            state: dict[str, torch.Tensor] = {}
            for f in safetensor_files:
                state.update(load_file(f))
            return state
        bin_files = sorted(p.glob("pytorch_model*.bin"))
        if bin_files:
            state = {}
            for f in bin_files:
                state.update(torch.load(f, map_location="cpu", weights_only=True))
            return state
        raise FileNotFoundError(f"No model weights found in {path}")

    from transformers import AutoModelForCausalLM

    logger.info("Loading HF backbone from Hub: %s", path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, trust_remote_code=True)
    return model.state_dict()


def _assert_out_not_in_source_tree(src: Path, out: Path) -> None:
    src_dir = src.resolve()
    out_r = out.resolve()
    if out_r == src_dir or src_dir in out_r.parents:
        raise ValueError(
            f"refusing to write output into the source model directory ({src_dir}); "
            "choose an --out path outside it."
        )


def convert(hf_dir: str, config_path: str, out: str) -> None:
    """Build the VLM, copy the HF backbone into it, and write a complete DCP."""
    out_path = Path(out)
    hf_path = Path(hf_dir)
    if hf_path.exists():
        _assert_out_not_in_source_tree(hf_path, out_path)

    config = load_config(config_path, cli_args=[])
    if config.vlm is None or config.vision_encoder is None or config.adapter is None:
        raise ValueError("config must define [vlm], [vision_encoder], and [adapter]")
    frames = config.video.max_frames if config.video is not None else 1

    # All on CPU — the factory never touches meta/FSDP.
    logger.info("Building VLM wrapper (vision from HF, adapter fresh)...")
    wrapper = build_vlm_wrapper(
        config.model,
        config.vision_encoder,
        config.adapter,
        config.vlm,
        frames_per_clip=frames,
    )

    converted, unmapped = map_state_dict(_load_hf_state_dict(hf_dir))
    tf_sd = wrapper.transformer.state_dict()
    fill_tied_head(converted, set(tf_sd))

    # Report both sides: unmapped source keys explain which target keys went unfilled.
    def _mismatch(problem: str, keys: set[str]) -> RuntimeError:
        msg = f"{problem}: {sorted(keys)[:10]}"
        if unmapped:
            msg += f"\nUnmapped source keys ({len(unmapped)}): {sorted(unmapped)[:10]}"
        return RuntimeError(msg + "\nSee map_key() — this expects Llama-style HF naming.")

    unexpected = converted.keys() - tf_sd.keys()
    if unexpected:
        raise _mismatch("mapped keys absent from the transformer", unexpected)
    missing = tf_sd.keys() - converted.keys()
    if missing:
        raise _mismatch("transformer keys unfilled by the source", missing)
    bad = [
        (k, tuple(converted[k].shape), tuple(tf_sd[k].shape))
        for k in converted
        if tuple(converted[k].shape) != tuple(tf_sd[k].shape)
    ]
    if bad:
        raise RuntimeError(f"shape mismatches (HF vs transformer): {bad[:8]}")

    converted = {k: v.to(tf_sd[k].dtype) for k, v in converted.items()}
    wrapper.transformer.load_state_dict(converted, strict=True)

    out_path.mkdir(parents=True, exist_ok=True)
    full = wrapper.state_dict()
    dcp.save({"model": full}, checkpoint_id=str(out_path))
    n_tf = sum(1 for k in full if k.startswith("transformer."))
    n_vis = sum(1 for k in full if k.startswith("vision_encoder."))
    n_ad = sum(1 for k in full if k.startswith("adapter."))
    logger.info(
        "Wrote complete VLM DCP (%d tensors: %d transformer, %d vision, %d adapter) -> %s\n"
        "Load it via [checkpoint].load_path + "
        "exclude_from_loading=['optimizer','dataloader'] (weights-only warm start).",
        len(full),
        n_tf,
        n_vis,
        n_ad,
        out_path,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--hf-dir", required=True, help="HF LLM model directory or Hub model id")
    parser.add_argument("--config", required=True, help="Target KF VLM TOML")
    parser.add_argument("--out", required=True, help="Output DCP directory")
    args = parser.parse_args(argv)
    convert(hf_dir=args.hf_dir, config_path=args.config, out=args.out)


if __name__ == "__main__":
    main()
