#!/usr/bin/env python3
"""Build a complete VLM checkpoint from a pretrained Hugging Face decoder backbone.

The VLM is built from the target config (vision encoder per ``[vision_encoder]``,
freshly initialised adapter), its transformer is filled with the source
backbone's weights, and the whole model is written as one DCP that
``[checkpoint].load_path`` warm-starts from with
``exclude_from_loading = ["optimizer"]``.

Nothing is written unless the target config matches the source's own
``config.json``, the files on disk match the source's shard index, every source
weight maps onto the transformer with the right shape, every transformer weight
is filled, and the built model normalises with the source's epsilon.
Initialisation is seeded (``--seed``, default ``[train].seed``), so the same
inputs always produce the same weights.

Usage:
    uv run python examples/vlm/scripts/convert_hf_backbone.py \\
        --hf-dir <model-dir-or-hub-id> \\
        --config examples/vlm/configs/<config>.toml \\
        --out path-to-init-checkpoint
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_model_state_dict

from kempnerforge.config.loader import load_config
from kempnerforge.config.model import ModelConfig
from kempnerforge.model.vlm import build_vlm_wrapper

logger = logging.getLogger(__name__)

SUPPORTED_MODEL_TYPES = ("qwen3",)

_INDEX_FILE = "model.safetensors.index.json"

EMBED_KEY = "token_embedding.embedding.weight"
HEAD_KEY = "output_head.proj.weight"
_HF_EMBED_KEY = "model.embed_tokens.weight"
_HF_HEAD_KEY = "lm_head.weight"

_TOP_LEVEL = {
    _HF_EMBED_KEY: EMBED_KEY,
    "model.norm.weight": "norm.weight",
    _HF_HEAD_KEY: HEAD_KEY,
}
_LAYER_KEY = re.compile(r"model\.layers\.(\d+)\.(.+)")
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


def map_key(hf_key: str) -> str | None:
    """Return the transformer key for a source key, or None if it has no counterpart."""
    if hf_key in _TOP_LEVEL:
        return _TOP_LEVEL[hf_key]
    match = _LAYER_KEY.fullmatch(hf_key)
    if match is None or match.group(2) not in _LAYER_LEAVES:
        return None
    return f"layers.{match.group(1)}.{_LAYER_LEAVES[match.group(2)]}"


def _rope_theta(hf_config: dict[str, Any]) -> tuple[Any, str | None]:
    """The RoPE theta the source applies, and a problem if its two spellings disagree.

    A ``rope_parameters`` entry overrides the top-level field in current
    transformers, while older releases read only the top-level one, so a source
    that sets both to different values applies a different theta depending on
    the release reading it and is refused rather than resolved here.
    """
    top = hf_config.get("rope_theta")
    nested = (hf_config.get("rope_parameters") or {}).get("rope_theta")
    if top is not None and nested is not None and top != nested:
        return nested, (
            f"rope_theta={top!r} but rope_parameters.rope_theta={nested!r}; "
            "which of the two applies depends on the transformers release"
        )
    return (top if nested is None else nested), None


def check_config(hf_config: dict[str, Any], model: ModelConfig) -> None:
    """Raise ``ValueError`` naming every way ``model`` differs from the source's config."""
    model_type = hf_config.get("model_type")
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"source model_type {model_type!r} is not supported "
            f"(supported: {', '.join(SUPPORTED_MODEL_TYPES)})"
        )
    pairs = {
        "hidden_size": ("dim", model.dim),
        "num_hidden_layers": ("n_layers", model.n_layers),
        "num_attention_heads": ("n_heads", model.n_heads),
        "num_key_value_heads": ("n_kv_heads", model.n_kv_heads),
        "head_dim": ("head_dim", model.head_dim),
        "intermediate_size": ("computed_ffn_hidden_dim", model.computed_ffn_hidden_dim),
        "vocab_size": ("vocab_size", model.vocab_size),
        "rms_norm_eps": ("norm_eps", model.norm_eps),
        "rope_theta": ("rope_theta", model.rope_theta),
        "tie_word_embeddings": ("tie_embeddings", model.tie_embeddings),
        "hidden_act": ("activation", str(model.activation)),
    }
    rope_theta, rope_problem = _rope_theta(hf_config)
    problems = [rope_problem] if rope_problem else []
    for hf_key, (kf_name, kf_value) in pairs.items():
        hf_value = rope_theta if hf_key == "rope_theta" else hf_config.get(hf_key)
        if hf_value is None:
            problems.append(f"config.json has no {hf_key}")
        elif hf_value != kf_value:
            problems.append(f"{hf_key}={hf_value!r} but model.{kf_name}={kf_value!r}")
    for spec_key in ("rope_scaling", "rope_parameters"):
        spec = hf_config.get(spec_key) or {}
        rope_type = spec.get("rope_type", spec.get("type", "default"))
        if rope_type != "default":
            problems.append(f"{spec_key} uses rope_type={rope_type!r}; only plain RoPE converts")
    if hf_config.get("use_sliding_window"):
        problems.append("use_sliding_window is set; only full attention converts")
    if hf_config.get("attention_bias"):
        problems.append(
            "attention_bias is set, so the source declares query/key/value/output biases "
            "that the transformer has no parameters for"
        )
    if not model.qk_norm:
        problems.append("the source normalises q/k per head, so model.qk_norm must be true")
    if str(model.norm_type) != "rmsnorm":
        problems.append(f"the source uses RMSNorm but model.norm_type={str(model.norm_type)!r}")
    if model.is_moe:
        problems.append("the source is dense but model.num_experts > 0")
    if problems:
        raise ValueError("target config does not match the source:\n  " + "\n  ".join(problems))


def resolve_source(hf_dir: str, patterns: list[str]) -> Path:
    """The local model directory ``hf_dir``, or the snapshot of Hub id ``hf_dir`` with its
    top-level files matching ``patterns`` fetched (or read from the cache)."""
    path = Path(hf_dir)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    logger.info("Fetching %s of %s from the Hugging Face Hub", patterns, hf_dir)
    return Path(snapshot_download(repo_id=hf_dir, allow_patterns=patterns, ignore_patterns=["*/*"]))


def load_source_config(source: Path) -> dict[str, Any]:
    """Read ``config.json`` in ``source``."""
    config_file = source / "config.json"
    if not config_file.is_file():
        raise FileNotFoundError(f"no config.json in {source}")
    return json.loads(config_file.read_text())


def load_source_weights(source: Path) -> dict[str, torch.Tensor]:
    """Read ``source``'s weights, following its shard index when it has one.

    A sharded source names every file and every tensor in ``_INDEX_FILE``. Reading
    the directory instead would accept a stale or unrelated file in place of a
    shard the index names, so the files and their contents are matched against it
    and any difference refuses the source. Only an unindexed source is read by
    listing its ``*.safetensors`` files.
    """
    from safetensors.torch import load_file

    files = sorted(source.glob("*.safetensors"))
    index_file = source / _INDEX_FILE
    if index_file.is_file():
        return _load_indexed_weights(source, files, json.loads(index_file.read_text()), load_file)
    if not files:
        raise FileNotFoundError(f"no *.safetensors weights in {source}")
    state: dict[str, torch.Tensor] = {}
    for file in files:
        part = load_file(file)
        repeated = state.keys() & part.keys()
        if repeated:
            raise ValueError(f"{file.name} repeats keys from another file: {sorted(repeated)[:5]}")
        state.update(part)
    return state


def _load_indexed_weights(
    source: Path, files: list[Path], index: dict[str, Any], load_file: Any
) -> dict[str, torch.Tensor]:
    """Read exactly the files the index names, with exactly the tensors it assigns them."""
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{source / _INDEX_FILE} has no weight_map")
    keys_by_file: dict[str, set[str]] = {}
    for key, name in weight_map.items():
        keys_by_file.setdefault(name, set()).add(key)
    on_disk = {file.name for file in files}
    absent = sorted(keys_by_file.keys() - on_disk)
    unlisted = sorted(on_disk - keys_by_file.keys())
    if absent or unlisted:
        raise ValueError(
            f"{source / _INDEX_FILE} lists {len(keys_by_file)} shards; {len(absent)} are missing "
            f"{absent[:5]} and {len(unlisted)} files it does not list are present {unlisted[:5]}"
        )
    state: dict[str, torch.Tensor] = {}
    for name, keys in sorted(keys_by_file.items()):
        part = load_file(source / name)
        if set(part) != keys:
            raise ValueError(
                f"{name} holds {len(part)} tensors but the index assigns it {len(keys)}: "
                f"missing {sorted(keys - set(part))[:5]}, unlisted {sorted(set(part) - keys)[:5]}"
            )
        state.update(part)
    return state


def check_norm_eps(transformer: torch.nn.Module, rms_norm_eps: float) -> None:
    """Raise ``ValueError`` if a norm the source's epsilon should reach does not use it.

    The source normalises with one epsilon throughout, so every norm the built
    transformer runs must use it. Which value a norm ends up with is a property
    of the model rather than of the config field it is meant to come from, so it
    is read back off the built modules: a norm that takes its epsilon elsewhere
    changes the outputs while every config value still agrees.
    """
    wrong = {
        name: module.eps
        for name, module in transformer.named_modules()
        if isinstance(getattr(module, "eps", None), float) and module.eps != rms_norm_eps
    }
    if wrong:
        named = sorted(wrong.items())
        raise ValueError(
            f"the source normalises with rms_norm_eps={rms_norm_eps!r}, but {len(wrong)} norms "
            f"of the built transformer use another epsilon: {named[:5]}"
        )


def map_state_dict(
    hf_state: dict[str, torch.Tensor], tie_embeddings: bool
) -> dict[str, torch.Tensor]:
    """Rename source weights to transformer keys; a tied head shares the embedding."""
    state = dict(hf_state)
    if tie_embeddings:
        head, embed = state.pop(_HF_HEAD_KEY, None), state.get(_HF_EMBED_KEY)
        if head is not None and embed is not None and not torch.equal(head, embed):
            raise ValueError(
                f"the source ties its output head, yet {_HF_HEAD_KEY} differs from {_HF_EMBED_KEY}"
            )
    converted: dict[str, torch.Tensor] = {}
    unmapped = []
    for hf_key, tensor in state.items():
        kf_key = map_key(hf_key)
        if kf_key is None:
            unmapped.append(hf_key)
        else:
            converted[kf_key] = tensor
    if unmapped:
        raise ValueError(
            f"{len(unmapped)} source keys have no transformer key: {sorted(unmapped)[:10]}"
        )
    if tie_embeddings and EMBED_KEY in converted:
        converted[HEAD_KEY] = converted[EMBED_KEY]
    return converted


def convert(hf_dir: str, config_path: str, out: str, seed: int | None = None) -> Path:
    """Build the VLM, fill its transformer from the source, and write it as one DCP."""
    config = load_config(config_path, cli_args=[])
    if not config.is_vlm:
        raise ValueError(f"{config_path} has no [vlm] section")
    assert config.vlm is not None and config.vision_encoder is not None
    assert config.adapter is not None
    out_path = Path(out)
    if out_path.exists() and (not out_path.is_dir() or any(out_path.iterdir())):
        raise FileExistsError(f"{out_path} already exists and is not an empty directory")

    source = resolve_source(hf_dir, ["config.json"])
    if source.resolve() in (out_path.resolve(), *out_path.resolve().parents):
        raise ValueError(f"refusing to write inside the source model directory {source.resolve()}")
    hf_config = load_source_config(source)
    check_config(hf_config, config.model)
    weights_dir = resolve_source(hf_dir, ["*.safetensors", f"*{_INDEX_FILE}"])
    hf_state = load_source_weights(weights_dir)
    converted = map_state_dict(hf_state, config.model.tie_embeddings)

    seed = config.train.seed if seed is None else seed
    torch.manual_seed(seed)
    wrapper = build_vlm_wrapper(
        config.model,
        config.vision_encoder,
        config.adapter,
        config.vlm,
        frames_per_clip=config.video.max_frames if config.video is not None else 1,
    )
    check_norm_eps(wrapper.transformer, hf_config["rms_norm_eps"])
    target = wrapper.transformer.state_dict()
    extra = sorted(converted.keys() - target.keys())
    missing = sorted(target.keys() - converted.keys())
    if extra or missing:
        raise ValueError(
            f"key mismatch: {len(extra)} source keys absent from the transformer {extra[:10]}, "
            f"{len(missing)} transformer keys missing from the source {missing[:10]}"
        )
    bad_shapes = [
        (k, tuple(converted[k].shape), tuple(target[k].shape))
        for k in sorted(converted)
        if converted[k].shape != target[k].shape
    ]
    if bad_shapes:
        raise ValueError(f"shape mismatch (source, transformer): {bad_shapes[:10]}")
    wrapper.transformer.load_state_dict(
        {k: v.to(target[k].dtype) for k, v in converted.items()}, strict=True
    )

    state = get_model_state_dict(wrapper)
    out_path.mkdir(parents=True, exist_ok=True)
    dcp.save({"model": state}, checkpoint_id=str(out_path))
    counts = {
        part: sum(k.startswith(f"{part}.") for k in state)
        for part in ("transformer", "vision_encoder", "adapter")
    }
    logger.info("Wrote %d tensors %s (seed %d) to %s", len(state), counts, seed, out_path)
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--hf-dir", required=True, help="local model directory or Hub model id")
    parser.add_argument("--config", required=True, help="target VLM config (TOML)")
    parser.add_argument("--out", required=True, help="output DCP directory (new or empty)")
    parser.add_argument("--seed", type=int, default=None, help="init seed (default: [train].seed)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    convert(args.hf_dir, args.config, args.out, seed=args.seed)


if __name__ == "__main__":
    main()
