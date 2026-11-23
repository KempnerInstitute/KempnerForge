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
import re
from collections.abc import Iterator
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

from kempnerforge.checkpoint.expert_layout import check_saved_experts
from kempnerforge.checkpoint.manager import derived_buffer_owners, drop_derived_model_keys
from kempnerforge.config.loader import load_config
from kempnerforge.config.schema import Activation, ModelConfig, NormType
from kempnerforge.model.transformer import Transformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_MIXTRAL_EXPERT_PROJ = {"gate": "w1", "down": "w2", "up": "w3"}
_KF_EXPERT_PROJ = {mixtral: kf for kf, mixtral in _MIXTRAL_EXPERT_PROJ.items()}
_KF_EXPERT_KEY = re.compile(r"\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.")
_MIXTRAL_EXPERT_KEY = re.compile(r"\.block_sparse_moe\.experts\.(\d+)\.(w[123])\.")
_PACKED_EXPERTS_KEY = re.compile(r"(layers\.\d+\.mlp\.)(gate|up|down)_w")
_MIXTRAL_PACKED_KEY = re.compile(r"(model\.layers\.\d+)\.mlp\.experts\.(gate_up_proj|down_proj)")


def _kf_to_hf_key(key: str) -> str:
    """Convert a KempnerForge state dict key to HuggingFace format.

    KempnerForge: layers.{i}.attention.q_proj.weight
    HuggingFace:  model.layers.{i}.self_attn.q_proj.weight

    MoE keys take the names of a Mixtral checkpoint, which transformers loads:
      KF: layers.{i}.mlp.router.gate.weight → HF: model.layers.{i}.block_sparse_moe.gate.weight
      KF: layers.{i}.mlp.experts.{j}.{gate,down,up}_proj.weight
        → HF: model.layers.{i}.block_sparse_moe.experts.{j}.{w1,w2,w3}.weight

    Mixtral has no shared experts or expert bias, so these keep their own names:
      KF: layers.{i}.mlp.shared_expert.*    → HF: model.layers.{i}.mlp.shared_experts.*
      KF: layers.{i}.mlp.router.expert_bias → HF: model.layers.{i}.mlp.expert_bias
    """
    key = key.replace("token_embedding.embedding.weight", "model.embed_tokens.weight")
    key = key.replace("output_head.proj.weight", "lm_head.weight")
    if key.startswith("layers."):
        key = "model." + key
        key = key.replace(".attention_norm.", ".input_layernorm.")
        key = key.replace(".mlp_norm.", ".post_attention_layernorm.")
        key = key.replace(".attention.", ".self_attn.")
        key = key.replace(".mlp.router.gate.", ".block_sparse_moe.gate.")
        key = key.replace(".mlp.router.expert_bias", ".mlp.expert_bias")
        key = key.replace(".mlp.shared_expert.", ".mlp.shared_experts.")
        key = _KF_EXPERT_KEY.sub(
            lambda m: f".block_sparse_moe.experts.{m[1]}.{_MIXTRAL_EXPERT_PROJ[m[2]]}.", key
        )
    elif key == "norm.weight":
        key = "model.norm.weight"
    return key


def _hf_to_kf_key(key: str) -> str:
    """Convert a HuggingFace state dict key to KempnerForge format.

    Reads the Mixtral names ``_kf_to_hf_key`` writes for MoE layers, and the names it
    wrote before them: ``mlp.gate`` for the router and ``mlp.experts.{j}.*_proj`` for
    the experts.
    """
    key = key.replace("model.embed_tokens.weight", "token_embedding.embedding.weight")
    key = key.replace("lm_head.weight", "output_head.proj.weight")
    key = key.replace("model.norm.weight", "norm.weight")
    if key.startswith("model.layers."):
        key = key.replace("model.layers.", "layers.", 1)
        key = key.replace(".self_attn.", ".attention.")
        key = key.replace(".input_layernorm.", ".attention_norm.")
        key = key.replace(".post_attention_layernorm.", ".mlp_norm.")
        key = key.replace(".block_sparse_moe.gate.", ".mlp.router.gate.")
        key = _MIXTRAL_EXPERT_KEY.sub(
            lambda m: f".mlp.experts.{m[1]}.{_KF_EXPERT_PROJ[m[2]]}_proj.", key
        )
        key = key.replace(".mlp.gate.weight", ".mlp.router.gate.weight")
        key = key.replace(".mlp.expert_bias", ".mlp.router.expert_bias")
        key = key.replace(".mlp.shared_experts.", ".mlp.shared_expert.")
    return key


def _hf_tensors(state_dict: dict[str, torch.Tensor]) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield each tensor of a KempnerForge state dict under its HuggingFace key.

    Packed expert weights, ``mlp.{gate,up,down}_w`` of shape ``(num_experts, in, out)``,
    are split into one ``(out, in)`` copy per expert, under the Mixtral expert names.
    """
    for kf_key, tensor in state_dict.items():
        packed = _PACKED_EXPERTS_KEY.fullmatch(kf_key)
        if packed is None:
            yield _kf_to_hf_key(kf_key), tensor
            continue
        prefix, proj = packed.groups()
        for j, weight in enumerate(tensor.unbind(0)):
            expert_key = _kf_to_hf_key(f"{prefix}experts.{j}.{proj}_proj.weight")
            yield expert_key, weight.t().clone(memory_format=torch.contiguous_format)


def _unpack_mixtral_experts(hf_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Split the experts a transformers 5 Mixtral holds packed into Mixtral's per-expert names.

    A loaded model's state dict, which a hub ID goes through, holds each layer's experts as
    ``mlp.experts.gate_up_proj`` of shape ``(num_experts, 2 * width, dim)``, each expert's
    gate rows then its up rows, and ``mlp.experts.down_proj`` of shape
    ``(num_experts, dim, width)``. Other keys pass through.
    """
    unpacked = {}
    for key, tensor in hf_state.items():
        match = _MIXTRAL_PACKED_KEY.fullmatch(key)
        if match is None:
            unpacked[key] = tensor
            continue
        prefix, proj = match.groups()
        names = ("w1", "w3") if proj == "gate_up_proj" else ("w2",)
        for j, weight in enumerate(tensor.unbind(0)):
            for name, part in zip(names, weight.chunk(len(names)), strict=True):
                unpacked[f"{prefix}.block_sparse_moe.experts.{j}.{name}.weight"] = part
    return unpacked


def _pack_experts(state: dict[str, torch.Tensor], model_state: dict[str, torch.Tensor]) -> None:
    """Stack per-expert weights in ``state`` into the packed tensors ``model_state`` holds.

    For a model built with packed experts, the ``mlp.experts.{j}.{gate,up,down}_proj.weight``
    tensors of a layer are replaced by its ``mlp.{gate,up,down}_w``, once every expert of
    that projection is present.
    """
    for kf_key, packed in model_state.items():
        match = _PACKED_EXPERTS_KEY.fullmatch(kf_key)
        if match is None:
            continue
        prefix, proj = match.groups()
        names = [f"{prefix}experts.{j}.{proj}_proj.weight" for j in range(packed.shape[0])]
        if all(name in state for name in names):
            state[kf_key] = torch.stack([state.pop(name).t() for name in names])


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
    check_architecture: bool = True,
) -> None:
    """Convert a DCP checkpoint to HuggingFace safetensors format.

    Handles both standard and pipeline-parallel checkpoints (pp0/, pp1/, ...).
    A dense model is written as a Llama checkpoint and an MoE model as a Mixtral
    checkpoint, with packed experts split into one tensor per expert; a model that
    its architecture cannot hold is refused before anything is read or written,
    unless ``check_architecture`` is False (see ``_build_hf_config``).
    A tensor that still shares storage with one already written after the cast,
    such as a tied output head at float32, is written as a copy, since
    safetensors refuses tensors that share memory. A checkpoint saved with
    expert parallelism holds every expert, as one saved without it does; one
    that holds fewer experts than ``model_config`` asks for is refused (see
    ``check_saved_experts``). Buffers a module derives from its other state, such as
    the sigmoid router's FP32 masters, are not exported, and a checkpoint saved before
    they existed converts too (see ``derived_buffer_owners``).

    Args:
        dcp_dir: Path to DCP checkpoint directory (e.g., checkpoints/step_10000).
        hf_dir: Output directory for HuggingFace model files.
        model_config: Model configuration for building the model skeleton.
        dtype: Export dtype (default: bfloat16), recorded as ``torch_dtype`` in
            the written ``config.json``.
        check_architecture: Refuse a model that Llama (dense) or Mixtral (MoE) cannot
            hold. False exports it for ``hf_to_dcp`` to read back, with a warning, since
            transformers would load it with those parts dropped or left random.
    """
    hf_config = _build_hf_config(model_config, dtype, check_architecture=check_architecture)
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
        saved = dcp.FileSystemReader(dcp_dir).read_metadata().state_dict_metadata
        check_saved_experts(model, saved, dcp_dir)
        state_dict = {"model": model.state_dict()}
        drop_derived_model_keys(state_dict["model"], model, saved)
        dcp.load(state_dict, checkpoint_id=dcp_dir)
        model.load_state_dict(state_dict["model"])

    derived = derived_buffer_owners(model)
    hf_state = {}
    seen_storage: set[int] = set()
    exported = {key: value for key, value in model.state_dict().items() if key not in derived}
    for hf_key, tensor in _hf_tensors(exported):
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

    (hf_path / "config.json").write_text(json.dumps(hf_config, indent=2))
    logger.info(f"Wrote config.json with {len(hf_state)} tensors")


def hf_to_dcp(
    hf_dir: str,
    dcp_dir: str,
    model_config: ModelConfig,
) -> None:
    """Convert a HuggingFace checkpoint to DCP format.

    Supports loading from a local directory or a HuggingFace model ID. Buffers a module
    derives from its other state, such as the sigmoid router's FP32 masters, which no
    export holds, are derived from the loaded weights.

    Args:
        hf_dir: Path to HF model directory or HF model ID.
        dcp_dir: Output directory for DCP checkpoint.
        model_config: Model configuration matching the HF model architecture.
    """
    hf_path = Path(hf_dir)

    hf_state = _load_hf_local(hf_path) if hf_path.exists() else _load_hf_hub(hf_dir)
    hf_state = _unpack_mixtral_experts(hf_state)

    logger.info(f"Loaded HuggingFace checkpoint: {len(hf_state)} tensors")

    model = Transformer(model_config)
    kf_state = model.state_dict()

    mapped: dict[str, torch.Tensor] = {}
    source: dict[str, str] = {}
    for hf_key, tensor in hf_state.items():
        kf_key = _hf_to_kf_key(hf_key)
        mapped[kf_key] = tensor
        source[kf_key] = hf_key
    _pack_experts(mapped, kf_state)

    converted = {}
    unmapped = []
    for kf_key, tensor in mapped.items():
        if kf_key in kf_state:
            converted[kf_key] = tensor
        else:
            unmapped.append(source.get(kf_key, kf_key))

    if unmapped:
        logger.warning(f"Skipped {len(unmapped)} unmapped HF keys: {unmapped[:5]}...")

    derived = derived_buffer_owners(model)
    missing = kf_state.keys() - converted.keys() - derived.keys()
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


def _architecture_problems(config: ModelConfig) -> list[str]:
    """Each setting of a model that the architecture it exports as cannot hold.

    transformers builds a dense export as Llama and an MoE export as Mixtral: both have
    SwiGLU MLPs or experts, RMSNorm and no query or key norm, and Mixtral routes with
    softmax top-k in every layer, with no shared experts. Weights of any other part
    would be dropped or left at their random init.
    """
    arch = "Mixtral" if config.is_moe else "Llama"
    problems = []
    if config.is_moe and config.moe_shared_experts > 0:
        problems.append(f"moe_shared_experts={config.moe_shared_experts}: Mixtral has none")
    if config.is_moe and config.moe_frequency > 1:
        problems.append(
            f"moe_frequency={config.moe_frequency}: every Mixtral layer is an MoE layer"
        )
    if config.is_moe and config.moe_router != "softmax_topk":
        problems.append(f"moe_router={config.moe_router!r}: Mixtral routes with softmax top-k")
    if config.activation != Activation.silu:
        mlps = "experts" if config.is_moe else "MLPs"
        problems.append(f"activation={str(config.activation)!r}: {arch} {mlps} are SwiGLU")
    if config.norm_type != NormType.rmsnorm:
        problems.append(f"norm_type={str(config.norm_type)!r}: {arch} norms are RMSNorm")
    if config.qk_norm:
        problems.append(f"qk_norm=True: {arch} has no query or key norm")
    return problems


def _build_hf_config(
    config: ModelConfig, dtype: torch.dtype = torch.bfloat16, check_architecture: bool = True
) -> dict:
    """Build a HuggingFace-compatible config.json from ModelConfig.

    ``torch_dtype`` names ``dtype``, the dtype the weights are written in, so
    ``from_pretrained(..., dtype="auto")`` loads them at that precision. A dense model
    becomes a Llama config and an MoE model a Mixtral config, whose
    ``intermediate_size`` is the expert width. With ``check_architecture`` False, a
    model that its architecture cannot hold gets one too, and a warning that names
    what transformers would drop.

    Raises:
        ValueError: ``dim`` is not divisible by ``n_heads``, or, with
            ``check_architecture``, the model has what its architecture lacks: MLPs or
            experts that are not SwiGLU, LayerNorm, or a query and key norm, and for an
            MoE model shared experts, dense layers between its MoE layers, or a router
            other than softmax top-k.
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

    arch = "Mixtral" if config.is_moe else "Llama"
    problems = "; ".join(_architecture_problems(config))
    if problems and check_architecture:
        raise ValueError(
            f"Cannot export this model as {arch}: {problems}. To export it for hf-to-dcp "
            "alone, pass --no-architecture-check (check_architecture=False)."
        )
    if problems:
        logger.warning(
            f"Exporting a model that {arch} cannot hold ({problems}): transformers would "
            "load it with those parts dropped or left random, so read it only with hf-to-dcp"
        )
    if config.is_moe:
        if config.moe_capacity_factor > 0:
            logger.warning(
                f"moe_capacity_factor={config.moe_capacity_factor}: transformers has no expert "
                "capacity, so the export drops no tokens where KempnerForge, in training and "
                "evaluation, drops those past an expert's capacity"
            )
        hf_config["intermediate_size"] = config.computed_expert_ffn_hidden_dim
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
    p_export.add_argument(
        "--no-architecture-check",
        dest="check_architecture",
        action="store_false",
        help="Export a model that Llama (dense) or Mixtral (MoE) cannot hold, for hf-to-dcp "
        "to read back; transformers would load it with those parts dropped or left random",
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
            check_architecture=args.check_architecture,
        )
    elif args.command == "hf-to-dcp":
        hf_to_dcp(
            hf_dir=args.hf_dir,
            dcp_dir=args.dcp_dir,
            model_config=config.model,
        )


if __name__ == "__main__":
    main()
