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

# A shard index is named for the weights file it indexes, with an optional variant
# before the suffix (``model.safetensors.index.fp16.json``), so it is matched rather
# than named: an index the conversion did not recognise would silently read the
# directory instead, which is what the index exists to prevent.
_INDEX_GLOB = "*.safetensors.index*.json"
# A snapshot directory is named after the commit it holds.
_COMMIT = re.compile(r"[0-9a-f]{40}")

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


def _source_rope(hf_config: dict[str, Any]) -> tuple[Any, Any, dict[str, Any], list[str]]:
    """The RoPE the installed transformers resolves for the source, and how it is ambiguous.

    A theta can be written in three places and a scaling type in two, and which
    one wins has changed between releases. Rather than re-implement that, the
    installed library resolves the config and its answer is what the conversion
    is checked against. A source whose own fields disagree is refused instead of
    resolved, because those are exactly the configs another release could read
    differently; a source that spells its RoPE one way reads the same in every
    release that knows that spelling. The spellings it used come back with the
    answer, so a refusal can say whether the value is the source's or a default.
    """
    problems: list[str] = []
    declared = {
        field: value
        for field, value in (
            ("rope_theta", hf_config.get("rope_theta")),
            (
                "rope_parameters.rope_theta",
                (hf_config.get("rope_parameters") or {}).get("rope_theta"),
            ),
            ("rope_scaling.rope_theta", (hf_config.get("rope_scaling") or {}).get("rope_theta")),
        )
        if value is not None
    }
    if len(set(declared.values())) > 1:
        spelled = ", ".join(f"{field}={value!r}" for field, value in sorted(declared.items()))
        problems.append(
            f"the source gives more than one RoPE theta ({spelled}); "
            "which one applies depends on the transformers release"
        )
    for spec_key in ("rope_scaling", "rope_parameters"):
        spec = hf_config.get(spec_key) or {}
        kinds = {spec[key] for key in ("rope_type", "type") if key in spec}
        if len(kinds) > 1:
            problems.append(
                f"{spec_key} gives both rope_type={spec['rope_type']!r} and type={spec['type']!r}; "
                "which one applies depends on the transformers release"
            )

    from transformers import AutoConfig

    source_config = {k: v for k, v in hf_config.items() if k != "model_type"}
    try:
        resolved = AutoConfig.for_model(hf_config["model_type"], **source_config)
    except Exception as error:  # noqa: BLE001 - any rejection is the source's, and refuses it
        problems.append(f"the installed transformers does not accept the source's config: {error}")
        return None, None, declared, problems
    parameters = getattr(resolved, "rope_parameters", None) or {}
    theta = parameters.get("rope_theta", getattr(resolved, "rope_theta", None))
    rope_type = parameters.get("rope_type", "default")
    return theta, rope_type, declared, problems


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
    rope_theta, rope_type, declared_rope, problems = _source_rope(hf_config)
    if (rope_theta, rope_type) == (None, None):
        # The library refused the config, which is already reported; what it would
        # have resolved for the theta is unknown rather than missing.
        pairs.pop("rope_theta")
    for hf_key, (kf_name, kf_value) in pairs.items():
        hf_value = rope_theta if hf_key == "rope_theta" else hf_config.get(hf_key)
        if hf_value is None:
            problems.append(f"config.json has no {hf_key}")
        elif hf_value == kf_value:
            continue
        elif hf_key == "rope_theta" and not declared_rope:
            # The value is the library's own default, so grepping config.json for it
            # finds nothing.
            problems.append(
                f"config.json gives no RoPE theta, so the installed transformers applies "
                f"{hf_value!r}, but model.{kf_name}={kf_value!r}"
            )
        else:
            problems.append(f"{hf_key}={hf_value!r} but model.{kf_name}={kf_value!r}")
    if rope_type not in (None, "default"):
        problems.append(f"the source uses rope_type={rope_type!r}; only plain RoPE converts")
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


def resolve_source(
    hf_dir: str, patterns: list[str], revision: str | None = None
) -> tuple[Path, str | None]:
    """Resolve ``hf_dir`` to a directory, and to the commit it came from when it is a Hub id.

    A local directory is returned as it is. A Hub id is fetched (or read from the
    cache) for its top-level files matching ``patterns``, at ``revision`` when one
    is given. The commit comes back so that a second fetch of the same source can
    ask for the same one: a repository's default branch can move between two
    requests, which would otherwise check one revision and convert another.
    """
    path = Path(hf_dir)
    if path.is_dir():
        return path, None
    from huggingface_hub import snapshot_download

    logger.info("Fetching %s of %s from the Hugging Face Hub", patterns, hf_dir)
    local = Path(
        snapshot_download(
            repo_id=hf_dir, allow_patterns=patterns, ignore_patterns=["*/*"], revision=revision
        )
    )
    return local, (local.name if _COMMIT.fullmatch(local.name) else None)


def load_source_config(source: Path) -> dict[str, Any]:
    """Read ``config.json`` in ``source``."""
    config_file = source / "config.json"
    if not config_file.is_file():
        raise FileNotFoundError(f"no config.json in {source}")
    return json.loads(config_file.read_text())


def load_source_weights(source: Path) -> dict[str, torch.Tensor]:
    """Read ``source``'s weights, following its shard index when it has one.

    A sharded source names every file and every tensor in its index. Reading the
    directory instead would accept a stale or unrelated file in place of a shard
    the index names, so an indexed source is read through its index alone and any
    difference refuses it. A source carrying more than one index is refused too,
    since which one a loader picks depends on the variant it was asked for. Only
    an unindexed source is read by listing its top-level ``*.safetensors`` files.
    """
    from safetensors.torch import load_file

    files = sorted(source.glob("*.safetensors"))
    indexes = sorted(source.glob(_INDEX_GLOB))
    if len(indexes) > 1:
        raise ValueError(
            f"{source} holds {len(indexes)} shard indexes {[p.name for p in indexes]}; "
            "which one applies depends on the variant a loader asks for"
        )
    if indexes:
        return _load_indexed_weights(
            source, indexes[0], json.loads(indexes[0].read_text()), load_file
        )
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
    source: Path, index_file: Path, index: dict[str, Any], load_file: Any
) -> dict[str, torch.Tensor]:
    """Read exactly the files the index names, with exactly the tensors it assigns them.

    A shard name is read relative to the source directory, as a loader reads it,
    so ``./shard.safetensors`` and a name in a subdirectory both resolve; a name
    that leaves the directory does not. Files the index does not name are left
    alone rather than refused: nothing reads them, so a second weights variant
    beside the indexed one is no reason to decline.
    """
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{index_file} has no weight_map")
    root = source.resolve()
    keys_by_shard: dict[Path, set[str]] = {}
    for key, name in weight_map.items():
        shard = (source / name).resolve()
        if root not in shard.parents:
            raise ValueError(f"{index_file} names {name!r}, which is outside {source}")
        keys_by_shard.setdefault(shard, set()).add(key)
    absent = sorted(str(shard.relative_to(root)) for shard in keys_by_shard if not shard.is_file())
    if absent:
        raise ValueError(
            f"{index_file} lists {len(keys_by_shard)} shards, of which {len(absent)} "
            f"are missing: {absent[:5]}. Only a source's top-level files are fetched "
            "from the Hub, so a shard the index places in a subdirectory is absent here"
        )
    state: dict[str, torch.Tensor] = {}
    for shard, keys in sorted(keys_by_shard.items()):
        part = load_file(shard)
        if set(part) != keys:
            raise ValueError(
                f"{shard.name} holds {len(part)} tensors but the index assigns it {len(keys)}: "
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

    source, commit = resolve_source(hf_dir, ["config.json"])
    if source.resolve() in (out_path.resolve(), *out_path.resolve().parents):
        raise ValueError(f"refusing to write inside the source model directory {source.resolve()}")
    hf_config = load_source_config(source)
    check_config(hf_config, config.model)

    # Everything that depends only on the target is settled before the weights are
    # fetched, so a source this config cannot host costs no download.
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

    weights_dir, _ = resolve_source(hf_dir, ["*.safetensors", _INDEX_GLOB], revision=commit)
    if weights_dir.resolve() != source.resolve():
        raise ValueError(
            f"{hf_dir} resolved to {source} for its config and {weights_dir} for its weights; "
            "the source moved between the two requests"
        )
    hf_state = load_source_weights(weights_dir)
    converted = map_state_dict(hf_state, config.model.tie_embeddings)

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
