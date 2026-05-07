"""Mixture-of-Transformers (MoT) operator and block.

Implements Algorithm 1 of Liang et al. (2024) "Mixture-of-Transformers"
and Figure 1c of the multimodal_paper. At every layer, every modality
has its own dense Q/K/V/O projections and a dedicated FFN; a single
global self-attention mixes all modality streams within the layer.

The module exposes three public symbols:

- ``MoTAttention`` — per-modality Q/K/V/O projections, one global SDPA
  over the concatenated multi-modality sequence.
- ``MoTBlock`` — pre-norm block: per-modality norms + MoTAttention +
  per-modality FFN. Identity at construction (zero-init residual).
- ``mot_warm_start_from_text_stack`` — copy dense ``TransformerBlock``
  weights from a source state dict into every per-modality copy of
  every ``MoTBlock`` in a Transformer. JD/text-only -> MoT warm start.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from kempnerforge.config.schema import ModelConfig
from kempnerforge.model.mlp import build_mlp
from kempnerforge.model.moe import MoEMLP, build_moe
from kempnerforge.model.norm import RMSNorm, build_norm
from kempnerforge.model.position import apply_rope


class MoTAttention(nn.Module):
    """Per-modality Q/K/V/O projections; one global SDPA over all modalities.

    State-dict layout (per-modality nesting via ``nn.ModuleDict``):

    .. code::

        q_proj.{m}.weight    # (n_heads * head_dim, dim)
        k_proj.{m}.weight    # (n_kv_heads * head_dim, dim)
        v_proj.{m}.weight    # (n_kv_heads * head_dim, dim)
        o_proj.{m}.weight    # (dim, n_heads * head_dim)
        q_norm.{m}.weight    # (head_dim,) when qk_norm=True
        k_norm.{m}.weight    # (head_dim,) when qk_norm=True

    Initialization: each per-modality ``o_proj.weight`` is zero so the
    operator's contribution to the residual stream is zero at
    construction (warm-start identity).

    Causal mask: a single ``is_causal=True`` over the concatenated
    sequence. With image-then-text concatenation order this matches
    Chameleon-style autoregressive multimodal: image attends causally
    among image; text attends to all earlier image and earlier text.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        modalities: tuple[str, ...],
        head_dim: int | None = None,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        if not modalities:
            raise ValueError("MoTAttention requires at least one modality")
        if n_kv_heads <= 0 or n_heads % n_kv_heads != 0:
            raise ValueError(
                f"MoTAttention: n_heads={n_heads} must be a positive multiple of "
                f"n_kv_heads={n_kv_heads}"
            )
        self.modalities = tuple(modalities)
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim or (dim // n_heads)
        self.n_rep = n_heads // n_kv_heads

        self.q_proj = nn.ModuleDict(
            {m: nn.Linear(dim, n_heads * self.head_dim, bias=False) for m in self.modalities}
        )
        self.k_proj = nn.ModuleDict(
            {m: nn.Linear(dim, n_kv_heads * self.head_dim, bias=False) for m in self.modalities}
        )
        self.v_proj = nn.ModuleDict(
            {m: nn.Linear(dim, n_kv_heads * self.head_dim, bias=False) for m in self.modalities}
        )
        self.o_proj = nn.ModuleDict(
            {m: nn.Linear(n_heads * self.head_dim, dim, bias=False) for m in self.modalities}
        )

        for m in self.modalities:
            nn.init.zeros_(self.o_proj[m].weight)  # type: ignore[reportArgumentType]

        if qk_norm:
            self.q_norm: nn.ModuleDict | None = nn.ModuleDict(
                {m: RMSNorm(self.head_dim) for m in self.modalities}
            )
            self.k_norm: nn.ModuleDict | None = nn.ModuleDict(
                {m: RMSNorm(self.head_dim) for m in self.modalities}
            )
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        streams: dict[str, torch.Tensor],
        rope: dict[str, tuple[torch.Tensor, torch.Tensor]],
        is_causal: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Run per-modality projections, global SDPA, per-modality output.

        Args:
            streams: per-modality input. Keys must equal the
                construction-time modalities. Each value has shape
                ``(batch, seq_m, dim)``.
            rope: per-modality ``(cos, sin)`` RoPE tables. Each
                ``cos[m]`` / ``sin[m]`` has shape
                ``(seq_m, head_dim // 2)`` — counts from position 0
                within that modality's axis.
            is_causal: passed through to
                ``F.scaled_dot_product_attention``.

        Returns:
            per-modality output of shape ``(batch, seq_m, dim)``.
        """
        if set(streams.keys()) != set(self.modalities):
            raise ValueError(
                f"MoTAttention.forward: streams keys {sorted(streams.keys())} "
                f"do not match construction-time modalities {sorted(self.modalities)}"
            )

        batch = next(iter(streams.values())).shape[0]
        qs: list[torch.Tensor] = []
        ks: list[torch.Tensor] = []
        vs: list[torch.Tensor] = []
        lengths: dict[str, int] = {}

        for m in self.modalities:
            x_m = streams[m]
            t_m = x_m.shape[1]
            lengths[m] = t_m

            q_m = self.q_proj[m](x_m).view(batch, t_m, -1, self.head_dim)
            k_m = self.k_proj[m](x_m).view(batch, t_m, -1, self.head_dim)
            v_m = self.v_proj[m](x_m).view(batch, t_m, -1, self.head_dim)

            if self.q_norm is not None:
                q_m = self.q_norm[m](q_m)
                k_m = self.k_norm[m](k_m)  # type: ignore[reportOptionalSubscript,reportOptionalCall]

            # Transpose to (batch, heads, seq, head_dim) for RoPE + SDPA.
            q_m = q_m.transpose(1, 2)
            k_m = k_m.transpose(1, 2)
            v_m = v_m.transpose(1, 2)

            cos_m, sin_m = rope[m]
            q_m = apply_rope(q_m, cos_m, sin_m)
            k_m = apply_rope(k_m, cos_m, sin_m)

            qs.append(q_m)
            ks.append(k_m)
            vs.append(v_m)

        # Concat along the seq dim of (B, heads, T, head_dim).
        q = torch.cat(qs, dim=2)
        k = torch.cat(ks, dim=2)
        v = torch.cat(vs, dim=2)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        # (B, n_heads, total_seq, head_dim) -> (B, total_seq, n_heads, head_dim)
        out = out.transpose(1, 2).contiguous()

        out_streams: dict[str, torch.Tensor] = {}
        offset = 0
        for m in self.modalities:
            t_m = lengths[m]
            o_m = out[:, offset : offset + t_m, :, :].reshape(batch, t_m, -1)
            out_streams[m] = self.o_proj[m](o_m)
            offset += t_m
        return out_streams


class MoTBlock(nn.Module):
    """Modality-aware transformer block: per-modality norms + MoTAttention + per-modality FFN.

    State-dict layout (per-modality nesting):

    .. code::

        attn_norm.{m}.weight     # RMSNorm or LayerNorm per modality
        attn.q_proj.{m}.weight   # ... see MoTAttention
        mlp_norm.{m}.weight
        mlp.{m}.gate_proj.weight # SwiGLU per modality (or up_proj/down_proj for standard)

    Initialization: per-modality ``mlp.{m}.down_proj.weight`` is zero
    so the FFN contribution is zero at construction. Combined with
    ``MoTAttention``'s zero-init ``o_proj``, the block is identity at
    construction (warm-start property: a fresh MoT block is bit-equal
    to passing the residual through unchanged).

    MoE branches (when the layer index hits ``moe_frequency``) skip the
    zero-init since ``MoEMLP`` does not have a single ``down_proj``;
    identity-at-construction is not a hard requirement for MoT — the
    warm-start helper will overwrite these weights from a source state
    dict, and from-scratch runs simply train the residual.
    """

    def __init__(
        self,
        config: ModelConfig,
        modalities: tuple[str, ...],
        layer_idx: int,
    ) -> None:
        super().__init__()
        if not modalities:
            raise ValueError("MoTBlock requires at least one modality")
        self.layer_idx = layer_idx
        self.modalities = tuple(modalities)

        self.attn_norm = nn.ModuleDict(
            {
                m: build_norm(config.norm_type, config.dim, eps=config.norm_eps)
                for m in self.modalities
            }
        )
        self.attn = MoTAttention(
            dim=config.dim,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,  # type: ignore[reportArgumentType]
            modalities=self.modalities,
            head_dim=config.head_dim,
            qk_norm=config.qk_norm,
        )
        self.mlp_norm = nn.ModuleDict(
            {
                m: build_norm(config.norm_type, config.dim, eps=config.norm_eps)
                for m in self.modalities
            }
        )

        use_moe = config.is_moe and ((layer_idx + 1) % config.moe_frequency == 0)
        if use_moe:
            self.mlp = nn.ModuleDict(
                {
                    m: build_moe(
                        dim=config.dim,
                        hidden_dim=config.computed_ffn_hidden_dim,
                        num_experts=config.num_experts,
                        top_k=config.moe_top_k,
                        activation=config.activation,
                        router_type=config.moe_router,
                        shared_experts=config.moe_shared_experts,
                        capacity_factor=config.moe_capacity_factor,
                        gradient_scale=config.moe_gradient_scale,
                        sequence_aux_loss_weight=config.moe_sequence_aux_loss_weight,
                        bias_schedule=config.moe_bias_schedule,
                        packed_experts=config.moe_packed_experts,
                    )
                    for m in self.modalities
                }
            )
        else:
            self.mlp = nn.ModuleDict(
                {
                    m: build_mlp(
                        dim=config.dim,
                        hidden_dim=config.computed_ffn_hidden_dim,
                        activation=config.activation,
                    )
                    for m in self.modalities
                }
            )

        # Zero-init per-modality down_proj on dense FFNs for warm-start identity.
        # MoE FFNs have no single down_proj — they keep registry-default init.
        for m in self.modalities:
            mlp_m = self.mlp[m]
            if isinstance(mlp_m, MoEMLP):
                continue
            if hasattr(mlp_m, "down_proj"):
                nn.init.zeros_(mlp_m.down_proj.weight)  # type: ignore[union-attr]

    def forward(
        self,
        streams: dict[str, torch.Tensor],
        rope: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        if set(streams.keys()) != set(self.modalities):
            raise ValueError(
                f"MoTBlock.forward: streams keys {sorted(streams.keys())} "
                f"do not match construction-time modalities {sorted(self.modalities)}"
            )
        normed_attn = {m: self.attn_norm[m](streams[m]) for m in self.modalities}
        attn_out = self.attn(normed_attn, rope, is_causal=True)
        post_attn = {m: streams[m] + attn_out[m] for m in self.modalities}
        normed_mlp = {m: self.mlp_norm[m](post_attn[m]) for m in self.modalities}
        mlp_out = {m: self.mlp[m](normed_mlp[m]) for m in self.modalities}
        return {m: post_attn[m] + mlp_out[m] for m in self.modalities}


# ---------------------------------------------------------------------------
# Warm-start helper
# ---------------------------------------------------------------------------


def _copy_weight(
    target_module: nn.Module,
    src_tensor: torch.Tensor,
    src_key: str,
    modality: str,
) -> None:
    """Copy ``src_tensor`` into ``target_module.weight`` in place, shape-checked.

    Handles both plain ``torch.Tensor`` targets and FSDP2 ``DTensor`` targets:
    when the target is a ``DTensor``, ``src_tensor`` is sharded to match the
    target's mesh + placements before the in-place copy.
    """
    target = cast(torch.Tensor, target_module.weight)
    if target.shape != src_tensor.shape:
        raise ValueError(
            f"mot_warm_start: shape mismatch for source key '{src_key}' -> "
            f"modality '{modality}': source {tuple(src_tensor.shape)} vs "
            f"target {tuple(target.shape)}"
        )
    src_cast = src_tensor.to(dtype=target.dtype, device=target.device)
    # DTensor path: shard the source to the target's placement, then copy
    # local-shard-to-local-shard. ``DTensor.copy_`` does not accept a plain
    # tensor under FSDP2.
    if hasattr(target, "_local_tensor") and hasattr(target, "device_mesh"):
        from torch.distributed.tensor import distribute_tensor  # noqa: PLC0415

        src_d = distribute_tensor(src_cast, target.device_mesh, target.placements)  # type: ignore[attr-defined]
        target._local_tensor.copy_(src_d._local_tensor)  # type: ignore[attr-defined]
    else:
        target.data.copy_(src_cast)


def mot_warm_start_from_text_stack(
    transformer: nn.Module,
    source_state_dict: Mapping[str, torch.Tensor],
) -> None:
    """Copy dense ``TransformerBlock`` weights into every per-modality copy
    inside each ``MoTBlock`` in ``transformer.layers``.

    Use case: warm-start a fresh MoT training run from a JD or text-only
    checkpoint. The caller loads the source state dict (e.g., via
    ``torch.load`` or DCP), then calls this helper to translate dense
    block keys into per-modality copies.

    Translation, per layer index ``i`` and per modality ``m``:

    .. code::

        layers.{i}.attention_norm.weight    -> layers.{i}.attn_norm.{m}.weight
        layers.{i}.attention.q_proj.weight  -> layers.{i}.attn.q_proj.{m}.weight
        layers.{i}.attention.k_proj.weight  -> layers.{i}.attn.k_proj.{m}.weight
        layers.{i}.attention.v_proj.weight  -> layers.{i}.attn.v_proj.{m}.weight
        layers.{i}.attention.o_proj.weight  -> layers.{i}.attn.o_proj.{m}.weight
        # qk_norm only:
        layers.{i}.attention.q_norm.weight  -> layers.{i}.attn.q_norm.{m}.weight
        layers.{i}.attention.k_norm.weight  -> layers.{i}.attn.k_norm.{m}.weight
        layers.{i}.mlp_norm.weight          -> layers.{i}.mlp_norm.{m}.weight
        layers.{i}.mlp.<proj>.weight        -> layers.{i}.mlp.{m}.<proj>.weight
            for <proj> in {gate_proj, up_proj, down_proj}

    Plus the final norm (when present on the target):

    .. code::

        norm.weight -> mot_norms.{m}.weight

    Source keys may optionally have a ``transformer.`` prefix; both are
    accepted. MoE FFN branches are skipped (their state dict layout is
    incompatible with a dense ``mlp.<proj>`` translation).

    No-op when ``transformer`` has no ``MoTBlock`` layers.
    Idempotent: repeated calls with the same source produce identical state.
    Shape-checked: raises ``ValueError`` on per-tensor shape mismatch.
    """
    layers = getattr(transformer, "layers", None)
    if layers is None:
        return
    mot_layers = {idx: layer for idx, layer in layers.items() if isinstance(layer, MoTBlock)}
    if not mot_layers:
        return

    state: dict[str, torch.Tensor] = {}
    for k, v in source_state_dict.items():
        canonical = k[len("transformer.") :] if k.startswith("transformer.") else k
        state[canonical] = v

    with torch.no_grad():
        for idx, layer in mot_layers.items():
            modalities = layer.modalities
            attn_translations: list[tuple[str, nn.ModuleDict]] = [
                (f"layers.{idx}.attention_norm.weight", layer.attn_norm),
                (f"layers.{idx}.attention.q_proj.weight", layer.attn.q_proj),
                (f"layers.{idx}.attention.k_proj.weight", layer.attn.k_proj),
                (f"layers.{idx}.attention.v_proj.weight", layer.attn.v_proj),
                (f"layers.{idx}.attention.o_proj.weight", layer.attn.o_proj),
                (f"layers.{idx}.mlp_norm.weight", layer.mlp_norm),
            ]
            if layer.attn.q_norm is not None:
                attn_translations.append(
                    (f"layers.{idx}.attention.q_norm.weight", layer.attn.q_norm)
                )
                attn_translations.append(
                    (f"layers.{idx}.attention.k_norm.weight", layer.attn.k_norm)  # type: ignore[arg-type]
                )

            for src_key, target_dict in attn_translations:
                if src_key not in state:
                    continue
                src_tensor = state[src_key]
                for m in modalities:
                    _copy_weight(target_dict[m], src_tensor, src_key, m)

            for proj in ("gate_proj", "up_proj", "down_proj"):
                src_key = f"layers.{idx}.mlp.{proj}.weight"
                if src_key not in state:
                    continue
                src_tensor = state[src_key]
                for m in modalities:
                    mlp_m = layer.mlp[m]
                    if isinstance(mlp_m, MoEMLP) or not hasattr(mlp_m, proj):
                        continue
                    _copy_weight(getattr(mlp_m, proj), src_tensor, src_key, m)

        mot_norms = getattr(transformer, "mot_norms", None)
        if mot_norms is not None and len(mot_norms) > 0 and "norm.weight" in state:
            src_tensor = state["norm.weight"]
            for m in mot_norms:
                _copy_weight(mot_norms[m], src_tensor, "norm.weight", m)
