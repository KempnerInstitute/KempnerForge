#!/usr/bin/env python3
"""Convert a KempnerInstitute/multimodal Qwen3 VLM checkpoint to KempnerForge DCP.

One-time import tool for **fine-tuning on top of** an externally-trained
(multimodal-repo) Qwen3-backbone VLM. It maps the source's flat ``torch.save``
state dict onto KempnerForge's VLM state-dict layout — the transformer + adapter
are built from a target TOML (one of the ``configs/train/vlm_qwen3_0.6b_*.toml``)
to validate keys/shapes, and the SigLIP vision tower is carried straight from the
source (no vision download) — then writes a DCP directory the warm-start loads:

    [checkpoint]
    load_path = "<out-dir>"
    exclude_from_loading = ["optimizer", "dataloader"]   # weights-only

The mapping is derived purely from the checkpoint's tensor names + shapes and
KempnerForge's own target layout (a clean-room, data-only mapping — no import of
the source repo's code). ``map_key`` is a pure function so it is unit-tested on
key strings alone.

Supported conversions ``(source arch -> target arch)``:
  - ``joint_decoder -> joint_decoder`` — 1:1.
  - ``mot -> mot`` — 1:1 per modality (needs ``--mot-image-index``).
  - ``joint_decoder -> mot`` — **init MoT from a dense JD checkpoint** by
    duplicating each per-layer dense weight into both modality copies (the
    canonical MoT warm-start; no ``--mot-image-index`` since both copies are
    identical). This mirrors how the multimodal repo built its MoT checkpoints
    (they were never trained from scratch — always seeded from JD).

``cross_attention`` differs structurally between the two repos (KF uses separate
K/V from the text dim + a per-block FFN; the source fuses K/V from the vision dim
and shares the decoder-layer FFN) and needs a KF CA arch-alignment first;
``moma`` has no source counterpart. Both raise a clear error.

Optimizer state is never read — the source ``model_*.pt`` is a bare model state
dict, and its ``optimizer_*.pt`` sibling is ignored (fine-tuning starts fresh).

Usage:
    # Joint-Decoder (1:1)
    uv run python scripts/convert_multimodal_checkpoint.py \
        --src /path/to/model_epoch_0_100000.pt \
        --config configs/train/vlm_qwen3_0.6b_joint_decoder.toml \
        --out   /path/to/output/jd_dcp

    # Init a MoT checkpoint from a dense Joint-Decoder checkpoint
    uv run python scripts/convert_multimodal_checkpoint.py \
        --src /path/to/jd_model.pt --from joint_decoder \
        --config configs/train/vlm_qwen3_0.6b_mot.toml \
        --out   /path/to/output/mot_from_jd_dcp

    # Native MoT source (1:1) — state which source modality index is the image
    uv run python scripts/convert_multimodal_checkpoint.py \
        --src /path/to/mot_model.pt \
        --config configs/train/vlm_qwen3_0.6b_mot.toml \
        --out   /path/to/output/mot_dcp \
        --mot-image-index 0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp

from kempnerforge.config.loader import load_config
from kempnerforge.model.adapter import build_adapter
from kempnerforge.model.transformer import Transformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Supported (source arch, target arch) conversions.
SUPPORTED_CONVERSIONS: tuple[tuple[str, str], ...] = (
    ("joint_decoder", "joint_decoder"),
    ("mot", "mot"),
    ("joint_decoder", "mot"),  # init MoT from a dense JD checkpoint (duplicate)
)

# KF MoT modality names (KF MoTConfig default). JD->MoT duplicates each dense
# weight into both; the fanned-out final norm seeds both mot_norms.
MOT_MODALITIES = ("image", "text")

# Source (multimodal repo) top-level prefixes.
_VISION_SRC_PREFIX = "image_encoder.model.vision_model."
_VISION_KF_PREFIX = "vision_encoder.vision_tower."
_CORE_SRC_PREFIX = "multimodal_core."

# Adapter (qwen2_5_vl_patch_merger) suffix map: source ``adapter.model.<suffix>``
# -> KF ``adapter.<suffix>``. ln_q is the merger's pre-projection RMSNorm; mlp.0
# / mlp.2 are the two Linears (mlp.1 is the activation, no params).
_ADAPTER_SUFFIX = {
    "ln_q.weight": "ln_q.weight",
    "mlp.0.weight": "proj1.weight",
    "mlp.0.bias": "proj1.bias",
    "mlp.2.weight": "proj2.weight",
    "mlp.2.bias": "proj2.bias",
}

# Per-modality attention sub-projections shared by the MoT layout.
_MOT_ATTN_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")


# ---------------------------------------------------------------------------
# Pure key mapping (unit-tested; no I/O, no model)
# ---------------------------------------------------------------------------


def _map_jd_layer(body: str) -> str:
    """Map a Joint-Decoder ``layers.{i}.<rest>`` body to KF naming (1:1)."""
    body = body.replace(".self_attn.", ".attention.")
    body = body.replace(".input_layernorm.", ".attention_norm.")
    body = body.replace(".post_attention_layernorm.", ".mlp_norm.")
    return body


def _map_mot_layer(body: str, idx_to_name: dict[str, str]) -> str:
    """Map a native-MoT ``layers.{i}.<rest>`` body (per-modality index) to KF.

    Source ``{m}`` is a numeric modality index (0/1); KF keys it by name
    (image/text). ``idx_to_name`` supplies that resolution.
    """
    parts = body.split(".")
    i = parts[1]
    grp = parts[2]
    if grp == "self_attn":
        sub = parts[3]
        if sub == "input_layer_norm":  # self_attn.input_layer_norm.{m}.weight
            name = idx_to_name[parts[4]]
            return f"layers.{i}.attn_norm.{name}.{'.'.join(parts[5:])}"
        if sub in _MOT_ATTN_PROJ:  # self_attn.{proj}.{m}.weight
            name = idx_to_name[parts[4]]
            return f"layers.{i}.attn.{sub}.{name}.{'.'.join(parts[5:])}"
    elif grp == "feed_forward":
        sub = parts[3]
        if sub == "mlp":  # feed_forward.mlp.{m}.{proj}.weight
            name = idx_to_name[parts[4]]
            return f"layers.{i}.mlp.{name}.{'.'.join(parts[5:])}"
        if sub == "post_attention_layernorm":  # feed_forward.post_attention_layernorm.{m}.weight
            name = idx_to_name[parts[4]]
            return f"layers.{i}.mlp_norm.{name}.{'.'.join(parts[5:])}"
    raise KeyError(f"unrecognized MoT layer key: multimodal_core.{body}")


def _jd_to_mot_layer_one(body: str, name: str) -> str:
    """Map a dense JD ``layers.{i}.<rest>`` body to ONE MoT modality's key.

    The dense weight is copied into modality ``name`` (the caller emits one key
    per modality, so the same dense tensor seeds both the image and text copies).
    """
    parts = body.split(".")
    i = parts[1]
    grp = parts[2]
    if grp == "self_attn":
        sub = parts[3]  # q_proj/k_proj/v_proj/o_proj/q_norm/k_norm
        return f"layers.{i}.attn.{sub}.{name}.{'.'.join(parts[4:])}"
    if grp == "input_layernorm":
        return f"layers.{i}.attn_norm.{name}.{'.'.join(parts[3:])}"
    if grp == "post_attention_layernorm":
        return f"layers.{i}.mlp_norm.{name}.{'.'.join(parts[3:])}"
    if grp == "mlp":
        proj = parts[3]  # gate_proj/up_proj/down_proj
        return f"layers.{i}.mlp.{name}.{proj}.{'.'.join(parts[4:])}"
    raise KeyError(f"unrecognized JD layer key: multimodal_core.{body}")


def map_key(
    src_key: str,
    source_arch: str,
    mot_idx_to_name: dict[str, str] | None = None,
    target_arch: str | None = None,
) -> list[str] | None:
    """Map one source key to zero or more KempnerForge ``VLMWrapper`` keys.

    ``target_arch`` defaults to ``source_arch`` (a 1:1 conversion). Set it
    differently for a cross-arch init, e.g. ``source_arch="joint_decoder"`` +
    ``target_arch="mot"`` duplicates each dense layer weight into both modality
    copies.

    Returns:
        - ``[kf_key, ...]`` — target key(s). One source tensor can seed several
          KF keys (MoT final norm -> shared ``norm`` + both ``mot_norms``;
          JD->MoT layer weight -> both modality copies).
        - ``[]`` — intentionally dropped (e.g. ``text_head.bias``: KF's output
          head is bias-free).
        - ``None`` — unrecognized (the caller surfaces it as unmapped).
    """
    target = target_arch or source_arch

    # Vision tower: identical HF SiglipVisionTransformer submodule on both
    # sides, so a straight prefix swap (all arches).
    if src_key.startswith(_VISION_SRC_PREFIX):
        return [_VISION_KF_PREFIX + src_key[len(_VISION_SRC_PREFIX) :]]

    if src_key == "text_preprocessor.embed.weight":
        return ["transformer.token_embedding.embedding.weight"]
    if src_key == "text_head.weight":
        return ["transformer.output_head.proj.weight"]
    if src_key == "text_head.bias":
        return []  # KF OutputHead has no bias (current Qwen3 checkpoints match)

    if src_key.startswith("adapter.model."):
        suffix = src_key[len("adapter.model.") :]
        mapped = _ADAPTER_SUFFIX.get(suffix)
        return [f"adapter.{mapped}"] if mapped is not None else None

    if src_key == "multimodal_core.norm.weight":
        if target == "mot":
            # Source has one final norm; KF MoT applies per-modality final
            # norms (mot_norms) and keeps the unused shared ``norm`` too.
            return ["transformer.norm.weight"] + [
                f"transformer.mot_norms.{m}.weight" for m in MOT_MODALITIES
            ]
        return ["transformer.norm.weight"]

    if src_key.startswith(_CORE_SRC_PREFIX):
        body = src_key[len(_CORE_SRC_PREFIX) :]  # e.g. "layers.0.self_attn.q_proj.weight"
        if not body.startswith("layers."):
            return None
        if source_arch == "joint_decoder" and target == "joint_decoder":
            return [f"transformer.{_map_jd_layer(body)}"]
        if source_arch == "mot" and target == "mot":
            assert mot_idx_to_name is not None, "mot->mot requires mot_idx_to_name"
            return [f"transformer.{_map_mot_layer(body, mot_idx_to_name)}"]
        if source_arch == "joint_decoder" and target == "mot":
            # Duplicate the dense weight into every modality copy.
            return [f"transformer.{_jd_to_mot_layer_one(body, m)}" for m in MOT_MODALITIES]
        raise NotImplementedError(f"conversion {source_arch!r} -> {target!r} not supported")

    return None


def map_state_dict(
    src_sd: dict[str, torch.Tensor],
    source_arch: str,
    mot_idx_to_name: dict[str, str] | None = None,
    target_arch: str | None = None,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Apply ``map_key`` across a source state dict.

    Returns ``(converted, unmapped)`` where ``converted`` is the KF-keyed state
    dict (one source tensor may seed several KF keys) and ``unmapped`` lists any
    source keys with no mapping (should be empty for a clean source).
    """
    converted: dict[str, torch.Tensor] = {}
    unmapped: list[str] = []
    for src_key, tensor in src_sd.items():
        targets = map_key(src_key, source_arch, mot_idx_to_name, target_arch)
        if targets is None:
            unmapped.append(src_key)
            continue
        for kf_key in targets:
            converted[kf_key] = tensor
    return converted, unmapped


# ---------------------------------------------------------------------------
# Shell: load source, build target, validate, write DCP
# ---------------------------------------------------------------------------

_WRAPPER_KEYS = ("model", "model_state_dict", "state_dict")


def load_source_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """mmap-load a source ``model_*.pt`` and return its flat tensor state dict."""
    try:
        obj = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except Exception as e:  # noqa: BLE001 -- trusted internal ckpt; retry unsafe
        logger.warning("weights_only load failed (%s); retrying weights_only=False", e)
        obj = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if isinstance(obj, dict) and not any(torch.is_tensor(v) for v in obj.values()):
        for k in _WRAPPER_KEYS:
            if k in obj and isinstance(obj[k], dict):
                logger.info("Unwrapped source state dict from ['%s']", k)
                obj = obj[k]
                break
    if not isinstance(obj, dict):
        raise TypeError(f"expected a state-dict mapping in {path}, got {type(obj)}")
    return {k: v for k, v in obj.items() if torch.is_tensor(v)}


def _assert_out_not_in_source_tree(src: Path, out: Path) -> None:
    """Refuse to write anywhere inside the source checkpoint's directory."""
    src_dir = src.resolve().parent
    out_r = out.resolve()
    if out_r == src_dir or src_dir in out_r.parents:
        raise ValueError(
            f"refusing to write output into the source checkpoint tree ({src_dir}). "
            "Choose an --out path outside it; the reference checkpoints must stay intact."
        )


def _infer_feature_dim(src_sd: dict[str, torch.Tensor], config: Any) -> int:
    """Vision feature dim used to size the adapter for the exact shape check.

    Prefer the source adapter's ``ln_q`` (the merger's pre-norm over the vision
    feature dim); fall back to the config's ``vision_encoder.feature_dim`` when set.
    """
    ln_q = src_sd.get("adapter.model.ln_q.weight")
    if ln_q is not None:
        return int(ln_q.shape[0])
    if config.vision_encoder is not None and config.vision_encoder.feature_dim > 0:
        return int(config.vision_encoder.feature_dim)
    raise ValueError(
        "cannot infer vision feature_dim: source has no adapter.model.ln_q.weight and "
        "vision_encoder.feature_dim is 0. Set feature_dim in the config."
    )


def convert(
    src: str,
    config_path: str,
    out: str,
    from_arch: str | None = None,
    mot_image_index: int | None = None,
    dtype: torch.dtype | None = None,
    allow_partial: bool = False,
) -> None:
    """Convert one multimodal ``.pt`` into a KempnerForge DCP directory."""
    src_path = Path(src)
    out_path = Path(out)
    _assert_out_not_in_source_tree(src_path, out_path)

    config = load_config(config_path, cli_args=[])
    target_arch = config.vlm.arch if config.vlm is not None else None
    source_arch = from_arch or target_arch
    if (source_arch, target_arch) not in SUPPORTED_CONVERSIONS:
        raise NotImplementedError(
            f"unsupported conversion {source_arch!r} -> {target_arch!r}. "
            f"Supported: {sorted(SUPPORTED_CONVERSIONS)}. cross_attention needs a KF "
            "CA arch-alignment; moma has no source counterpart."
        )
    assert source_arch is not None and target_arch is not None  # both in SUPPORTED_CONVERSIONS

    # Native MoT source carries numeric modality indices, so we must know which
    # is the image stream. JD->MoT duplicates identical copies, so order is moot.
    mot_idx_to_name: dict[str, str] | None = None
    if source_arch == "mot" and target_arch == "mot":
        if mot_image_index not in (0, 1):
            raise ValueError(
                "mot->mot requires --mot-image-index {0,1}: which source modality "
                "index is the IMAGE stream. Verify against your training convention "
                "(a wrong value silently swaps the image/text experts)."
            )
        text_index = 1 - mot_image_index
        mot_idx_to_name = {str(mot_image_index): "image", str(text_index): "text"}
        logger.warning(
            "MoT modality order: source index %d -> image, %d -> text. "
            "Confirm this matches how the source was trained.",
            mot_image_index,
            text_index,
        )

    logger.info("Loading source checkpoint: %s", src_path)
    src_sd = load_source_state_dict(src_path)
    logger.info("Source tensors: %d (%s -> %s)", len(src_sd), source_arch, target_arch)

    assert config.vlm is not None  # narrowed by the SUPPORTED_CONVERSIONS check above
    if config.adapter is None:
        raise ValueError("VLM config must define an [adapter] section")

    # Expected KF transformer + adapter keys/shapes from a meta build. No vision
    # download: the SigLIP tower is carried straight from the source checkpoint
    # (same HF submodule KF wraps), so building it here would only be discarded.
    feature_dim = _infer_feature_dim(src_sd, config)
    with torch.device("meta"):
        transformer = Transformer(config.model, vlm_config=config.vlm, num_image_tokens=256)
    adapter = build_adapter(config.adapter, in_dim=feature_dim, out_dim=config.model.dim)
    expected: dict[str, tuple[int, ...]] = {
        f"transformer.{k}": tuple(v.shape) for k, v in transformer.state_dict().items()
    }
    expected |= {f"adapter.{k}": tuple(v.shape) for k, v in adapter.state_dict().items()}

    converted, unmapped = map_state_dict(src_sd, source_arch, mot_idx_to_name, target_arch)
    if dtype is not None:
        converted = {k: v.to(dtype) for k, v in converted.items()}

    vision = {k for k in converted if k.startswith(_VISION_KF_PREFIX)}
    nonvision = converted.keys() - vision

    missing = expected.keys() - nonvision
    unexpected = nonvision - expected.keys()
    if unmapped:
        logger.warning("Unmapped source keys (%d): %s", len(unmapped), sorted(unmapped)[:10])
    if unexpected:
        raise RuntimeError(
            f"{len(unexpected)} converted keys are not in the target model (mapping bug): "
            f"{sorted(unexpected)[:10]}"
        )
    if missing:
        msg = f"{len(missing)} target keys unfilled by the source: {sorted(missing)[:10]}"
        if allow_partial:
            logger.warning("%s (continuing: --allow-partial; these keep init)", msg)
        else:
            raise RuntimeError(msg + " (pass --allow-partial to fill them from init)")
    if not vision:
        logger.warning("No vision_encoder.* keys carried from the source (expected the tower).")

    # Shape check on the transformer/adapter keys (catches a wrong-shape slip).
    bad = [
        (k, tuple(converted[k].shape), expected[k])
        for k in nonvision
        if tuple(converted[k].shape) != expected[k]
    ]
    if bad:
        raise RuntimeError(f"shape mismatches (source vs target): {bad[:8]}")

    out_path.mkdir(parents=True, exist_ok=True)
    dcp.save({"model": converted}, checkpoint_id=str(out_path))
    logger.info(
        "Wrote DCP (%d tensors: %d transformer+adapter, %d vision; %s -> %s) -> %s\n"
        "Load it via [checkpoint].load_path + "
        "exclude_from_loading=['optimizer','dataloader'] (weights-only warm start).",
        len(converted),
        len(nonvision),
        len(vision),
        source_arch,
        target_arch,
        out_path,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--src", required=True, help="Source multimodal model_*.pt file")
    parser.add_argument("--config", required=True, help="Target KF TOML (vlm_qwen3_0.6b_*.toml)")
    parser.add_argument(
        "--out", required=True, help="Output DCP directory (outside the source tree)"
    )
    parser.add_argument(
        "--from",
        dest="from_arch",
        default=None,
        choices=["joint_decoder", "mot"],
        help="Source arch (default: same as the target config's arch). "
        "Use --from joint_decoder with a MoT config to init MoT from a dense JD checkpoint.",
    )
    parser.add_argument(
        "--mot-image-index",
        type=int,
        default=None,
        choices=[0, 1],
        help="For a native mot->mot conversion: which source modality index is the image stream",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        choices=["float32", "bfloat16", "float16"],
        help="Optional cast for the written weights (default: keep source dtype)",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Warn instead of failing when some target keys are unfilled (they keep init)",
    )
    args = parser.parse_args(argv)

    dtype_map: dict[str, Any] = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    convert(
        src=args.src,
        config_path=args.config,
        out=args.out,
        from_arch=args.from_arch,
        mot_image_index=args.mot_image_index,
        dtype=dtype_map.get(args.dtype) if args.dtype else None,
        allow_partial=args.allow_partial,
    )


if __name__ == "__main__":
    main()
