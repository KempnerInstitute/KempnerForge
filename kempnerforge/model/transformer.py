"""Transformer model for KempnerForge.

Architecture: Llama-style pre-norm transformer.
  Token Embedding → [TransformerBlock × N] → Final Norm → Output Head

Design choices:
  - ModuleDict (not ModuleList) for layers — preserves FQNs for DCP checkpointing.
  - Embedding and output head are optional (can be None for PP middle stages).
  - Forward is a simple loop over blocks — pipeline-parallelism friendly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn

from kempnerforge.config.registry import registry
from kempnerforge.config.schema import ModelConfig
from kempnerforge.config.vlm import CrossAttentionConfig, MoMaConfig, MoTConfig, VLMConfig
from kempnerforge.model.attention import Attention, KVCache
from kempnerforge.model.cross_attention import CrossAttentionBlock
from kempnerforge.model.embedding import OutputHead, TokenEmbedding
from kempnerforge.model.init import init_weights
from kempnerforge.model.masking import build_attention_mask
from kempnerforge.model.mlp import build_mlp
from kempnerforge.model.modality import ModalityContext
from kempnerforge.model.moe import MoEMLP, build_moe
from kempnerforge.model.moma import ExpertChoiceMoE, MoMaBlock, MoMaFFN
from kempnerforge.model.mot import MoTBlock

if TYPE_CHECKING:
    from torch.nn.attention.flex_attention import BlockMask
from kempnerforge.model.norm import build_norm
from kempnerforge.model.position import precompute_rope_frequencies


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture.

    Structure: norm → attention → residual, norm → mlp → residual
    """

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        self.attention_norm = build_norm(config.norm_type, config.dim, eps=config.norm_eps)
        self.attention = Attention(
            dim=config.dim,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,  # type: ignore[reportArgumentType]
            head_dim=config.head_dim,
            qk_norm=config.qk_norm,
            sdpa_backend=config.sdpa_backend,
            norm_eps=config.norm_eps,
        )

        self.mlp_norm = build_norm(config.norm_type, config.dim, eps=config.norm_eps)

        use_moe = config.is_moe and ((layer_idx + 1) % config.moe_frequency == 0)
        if use_moe:
            self.mlp = build_moe(
                dim=config.dim,
                hidden_dim=config.computed_expert_ffn_hidden_dim,
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
        else:
            self.mlp = build_mlp(
                dim=config.dim,
                hidden_dim=config.computed_ffn_hidden_dim,
                activation=config.activation,
            )

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        *,
        kv_cache: KVCache | None = None,
        attn_mask: torch.Tensor | BlockMask | None = None,
    ) -> torch.Tensor:
        """Run attention and the MLP over the residual stream.

        Args:
            x: Residual stream, shape ``(batch, seq_len, dim)``.
            rope_cos: RoPE cosine frequencies for this block's positions.
            rope_sin: RoPE sine frequencies for this block's positions.
            kv_cache: Optional KV cache for incremental generation.
            attn_mask: The one mask from
                ``kempnerforge.model.masking.build_attention_mask``, built once
                per forward and shared by every block.

        Returns:
            The updated residual stream, shape ``(batch, seq_len, dim)``.
        """
        x = x + self.attention(
            self.attention_norm(x),
            rope_cos,
            rope_sin,
            kv_cache=kv_cache,
            attn_mask=attn_mask,
        )
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Transformer(nn.Module):
    """Full transformer model built from ModelConfig.

    Embedding → TransformerBlocks → Norm → Output Head

    ``include_token_embedding=False`` / ``include_output_head=False`` build the
    model without that module: the attribute is ``None`` and registers no
    state-dict keys. Without the embedding, pass pre-embedded inputs via
    ``modality.inputs_embeds``; without the head, ``forward`` returns the final
    hidden states. ``tie_embeddings=True`` requires both modules.
    ``generate()`` rejects a model built without either module.
    Tensor parallelism is not supported for such a model.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        vlm_config: VLMConfig | None = None,
        num_image_tokens: int = 0,
        include_token_embedding: bool = True,
        include_output_head: bool = True,
    ) -> None:
        super().__init__()
        self.config = config

        self.token_embedding: TokenEmbedding | None = (
            TokenEmbedding(config.vocab_size, config.dim) if include_token_embedding else None
        )

        self._mot_modalities: tuple[str, ...] = ()
        self._mot_n_image: int = 0
        self._moma_modalities: tuple[str, ...] = ()
        if isinstance(vlm_config, MoTConfig):
            text_n_kv_heads = config.n_kv_heads if config.n_kv_heads is not None else config.n_heads
            img_n_heads, img_n_kv_heads = vlm_config.resolved_image_heads(
                config.n_heads, text_n_kv_heads
            )
            if img_n_heads != config.n_heads or img_n_kv_heads != text_n_kv_heads:
                raise ValueError(
                    "MoT v1 requires equal head counts across modalities (single global SDPA): "
                    f"image=({img_n_heads}, {img_n_kv_heads}) vs "
                    f"text=({config.n_heads}, {text_n_kv_heads})"
                )
            self._mot_modalities = vlm_config.mot_modalities
            self._mot_n_image = num_image_tokens
            self.layers = nn.ModuleDict(
                {
                    str(i): MoTBlock(config, modalities=self._mot_modalities, layer_idx=i)
                    for i in range(config.n_layers)
                }
            )
        elif isinstance(vlm_config, MoMaConfig):
            self._moma_modalities = vlm_config.moma_modalities
            experts_per_modality = dict(vlm_config.moma_experts_per_modality)
            capacity_factor_per_modality = {
                m: vlm_config.effective_capacity_factor(m) for m in self._moma_modalities
            }
            self.layers = nn.ModuleDict(
                {
                    str(i): MoMaBlock(
                        config,
                        modalities=self._moma_modalities,
                        experts_per_modality=experts_per_modality,
                        capacity_factor_per_modality=capacity_factor_per_modality,
                        gumbel_noise=vlm_config.moma_gumbel_noise,
                        layer_idx=i,
                    )
                    for i in range(config.n_layers)
                }
            )
        else:
            self.layers = nn.ModuleDict(
                {str(i): TransformerBlock(config, layer_idx=i) for i in range(config.n_layers)}
            )

        self.cross_attention_layers: nn.ModuleDict = nn.ModuleDict()
        self._ca_cadence: int = 0
        if isinstance(vlm_config, CrossAttentionConfig):
            self._ca_cadence = vlm_config.cross_attention_every_n_layers
            n_h, n_kv = vlm_config.resolved_heads(config.n_heads)
            ca_head_dim = config.head_dim if config.head_dim_override else None
            num_ca_blocks = config.n_layers // self._ca_cadence
            for k in range(num_ca_blocks):
                self.cross_attention_layers[str(k)] = CrossAttentionBlock(
                    dim=config.dim,
                    n_heads=n_h,
                    n_kv_heads=n_kv,
                    ffn_hidden_dim=config.computed_ffn_hidden_dim,
                    norm_type=config.norm_type,
                    activation=config.activation,
                    head_dim=ca_head_dim,
                    norm_eps=config.norm_eps,
                )

        self.norm = build_norm(config.norm_type, config.dim, eps=config.norm_eps)
        self.mot_norms: nn.ModuleDict = nn.ModuleDict()
        if self._mot_modalities:
            self.mot_norms = nn.ModuleDict(
                {
                    m: build_norm(config.norm_type, config.dim, eps=config.norm_eps)
                    for m in self._mot_modalities
                }
            )
            for p in self.norm.parameters():
                p.requires_grad_(False)

        self.output_head: OutputHead | None = (
            OutputHead(config.dim, config.vocab_size) if include_output_head else None
        )

        if config.tie_embeddings:
            if self.token_embedding is None or self.output_head is None:
                raise ValueError(
                    "tie_embeddings=True requires both the token embedding and the output "
                    f"head (got include_token_embedding={include_token_embedding}, "
                    f"include_output_head={include_output_head}); keep both or set "
                    "tie_embeddings=False"
                )
            self.output_head.tie_weights(self.token_embedding)

        self._rope_cos = None
        self._rope_sin = None
        if not any(p.is_meta for p in self.parameters()):
            self._rope_cos, self._rope_sin = precompute_rope_frequencies(
                head_dim=config.head_dim,
                max_seq_len=config.max_seq_len,
                theta=config.rope_theta,
            )
            init_weights(self, config)

    def init_weights_and_freqs(self) -> None:
        """Initialize weights and RoPE frequencies after meta-device materialization.

        Called after ``model.to_empty(device=...)`` to fill in parameter values
        and compute RoPE frequency table. Safe to call on already-initialized models
        (skips if freqs are already computed).
        """
        if self._rope_cos is None:
            self._rope_cos, self._rope_sin = precompute_rope_frequencies(
                head_dim=self.config.head_dim,
                max_seq_len=self.config.max_seq_len,
                theta=self.config.rope_theta,
            )
        init_weights(self, self.config)

    def set_moe_step(self, step: int, max_steps: int) -> None:
        """Set training step on all MoE routers for adaptive bias scheduling."""
        from kempnerforge.model.router import SigmoidTopKRouter

        for layer in self.layers.values():
            if isinstance(layer.mlp, MoEMLP) and isinstance(layer.mlp.router, SigmoidTopKRouter):
                layer.mlp.router.set_step(step, max_steps)

    def get_moe_aux_loss(self) -> torch.Tensor:
        """Collect auxiliary losses from all MoE layers. Returns 0 if dense."""
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.layers.values():
            if isinstance(layer.mlp, MoEMLP):
                total = total + layer.mlp.aux_loss
        return total

    def get_moe_router_z_loss(self) -> torch.Tensor:
        """Collect router z-losses (ST-MoE) from all MoE layers. Returns 0 if dense."""
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.layers.values():
            if isinstance(layer.mlp, MoEMLP):
                total = total + layer.mlp.z_loss
        return total

    def get_expert_counts(self) -> dict[int, torch.Tensor]:
        """Collect per-layer expert utilization for flat MoE layers.

        Returns ``{layer_idx: (num_experts,) tensor}`` for layers whose MLP
        is a ``MoEMLP`` (the standard, single-pool MoE). Returns ``{}`` for
        dense models and for MoMa: MoMa's per-modality groups have a
        different shape (per modality, per expert) and surface through
        ``get_moma_expert_counts`` instead.
        """
        counts = {}
        for name, layer in self.layers.items():
            if isinstance(layer.mlp, MoEMLP):
                counts[int(name)] = layer.mlp.expert_counts
        return counts

    def get_moma_expert_counts(self) -> dict[int, dict[str, torch.Tensor]]:
        """Collect per-layer, per-modality expert utilization for MoMa layers.

        Returns ``{layer_idx: {modality: (num_experts_for_modality,) tensor}}``
        for every layer whose MLP is a ``MoMaFFN``; returns ``{}`` otherwise.
        Each modality's expert count tensor reflects the most recent forward
        through that layer's expert-choice router (paper Figure 5-style
        utilization). Counts on a fresh model (no forward yet) are the
        router's init zeros.
        """
        counts: dict[int, dict[str, torch.Tensor]] = {}
        for name, layer in self.layers.items():
            if isinstance(layer.mlp, MoMaFFN):
                counts[int(name)] = {
                    m: cast(ExpertChoiceMoE, layer.mlp.experts[m]).expert_counts
                    for m in layer.mlp.modalities
                }
        return counts

    def _validate_modality_ids(
        self, modality_ids: torch.Tensor | None, residual_shape: tuple[int, int]
    ) -> None:
        """Check the per-token modality tag the MoMa and MoT paths route on.

        A no-op for every other arch. Runs before the attention mask is built,
        so an arch missing its tags says so instead of failing later on an
        unrelated input.

        Args:
            modality_ids: The tag from the ``ModalityContext``, or ``None``.
            residual_shape: ``(batch, seq_len)`` of the residual stream.

        Raises:
            ValueError: If the arch routes on ``modality_ids`` and it is absent
                or does not cover the residual stream.
        """
        arch = "MoMa" if self._moma_modalities else "MoT" if self._mot_modalities else ""
        if not arch:
            return
        if modality_ids is None:
            raise ValueError(
                f"{arch} model requires modality.modality_ids (got None). Build the "
                f"ModalityContext via {arch}Strategy or set modality_ids explicitly."
            )
        if tuple(modality_ids.shape) != residual_shape:
            raise ValueError(
                f"modality.modality_ids shape {tuple(modality_ids.shape)} does not "
                f"match residual shape {residual_shape}"
            )

    def forward(
        self,
        tokens: torch.Tensor | None = None,
        *,
        modality: ModalityContext | None = None,
        kv_caches: list[KVCache] | None = None,
        doc_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_group_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Exactly one of ``tokens`` or ``modality.inputs_embeds`` must be
        provided. Modality-injection routes (``prefix_embeds``,
        ``output_slice``, ``image_features``, ``image_mask``,
        ``modality_ids``) are grouped on the optional
        ``ModalityContext`` arg; see ``kempnerforge/model/modality.py``
        for the full intra-context invariant table.

        Args:
            tokens: Integer token ids, shape ``(batch, seq_len)``.
            modality: Optional ``ModalityContext`` bundling pre-embedded
                inputs, prefix embeds, output slicing, image features,
                and modality routing tags for VLM arches. ``None`` is the
                pure text-only forward. It may also carry ``position_ids``
                and ``attention_group_ids`` (see those arguments) or the
                rotation itself as ``rope_cos`` and ``rope_sin``,
                ``(batch, seq_len, head_dim // 2)`` over the residual, for
                position schemes that are not one integer per token; the
                rotation excludes ``position_ids``, covers the appended
                tokens under ``kv_caches`` as ``position_ids`` does, is
                applied without the model's own tables, and is unsupported
                for MoT.
            kv_caches: Optional list of KVCache (one per layer) for
                generation. The tokens of this forward are appended after
                the cached ones; without caller positions or a rotation
                their RoPE positions continue from the cache fill level.
                Cross-arg invariant:
                ``kv_caches`` forbids ``modality.prefix_embeds``,
                ``modality.output_slice``, ``modality.image_features``,
                ``modality.modality_ids``, ``modality.key_padding_mask``,
                ``doc_ids`` and ``attention_group_ids`` (all training-only).
            doc_ids: Optional per-token document IDs for packed sequences,
                shape ``(batch, seq_len)`` over the residual stream. Enables
                block-diagonal causal attention that isolates documents within
                packed sequences.
            position_ids: Optional explicit per-token RoPE positions, shape
                ``(batch, seq_len)`` matching the residual reaching attention
                (including any ``modality.prefix_embeds``), dtype int64,
                values in ``[0, model.max_seq_len)``. Rows may differ, so
                positions can e.g. restart at zero at each document of a
                packed sequence. ``None`` (default) assigns contiguous
                positions starting at the KV-cache fill level, exactly as
                before. With ``kv_caches`` the tensor covers the tokens of
                this forward alone, the ones being appended, so a caller
                that positions its own tokens does so at prefill and at
                every decode step alike. Unsupported for MoT, which derives
                per-modality positions internally. A ``ModalityContext`` may carry the same tensor
                as ``modality.position_ids`` instead, which is how a strategy
                chooses the positions of the residual it lays out; passing
                both raises.
            attention_group_ids: Optional per-token visibility group ids over
                the residual stream, shape ``(batch, seq_len)``, int64;
                negative means no group. Tokens sharing a nonnegative id see
                each other bidirectionally (e.g. prefix-LM spans):
                ``visible(q, k) = same_document AND (k <= q OR
                same_nonnegative_group)``. Without ``doc_ids`` the row is one
                document. ``None`` leaves attention unchanged; all-negative ids
                give the same visibility as ``None`` but still build a mask, so
                ``None`` is the unmasked fast path. On MoT, ``doc_ids`` and
                group ids need ``vlm.mot_modalities`` in residual order,
                ``("image", "text")``. May likewise come from
                ``modality.attention_group_ids``, not both.

        Returns:
            Logits tensor of shape ``(batch, out_seq_len, vocab_size)``
            where ``out_seq_len == seq_len`` normally or the sliced
            length when ``modality.output_slice`` is set. Built with
            ``include_output_head=False``, returns the final hidden
            states ``(batch, out_seq_len, dim)`` instead.
        """
        inputs_embeds = modality.inputs_embeds if modality is not None else None
        prefix_embeds = modality.prefix_embeds if modality is not None else None
        output_slice = modality.output_slice if modality is not None else None
        image_features = modality.image_features if modality is not None else None
        image_mask = modality.image_mask if modality is not None else None
        modality_ids = modality.modality_ids if modality is not None else None
        key_padding_mask = modality.key_padding_mask if modality is not None else None
        position_ids = _argument_or_context("position_ids", position_ids, modality)
        rope_cos = modality.rope_cos if modality is not None else None
        rope_sin = modality.rope_sin if modality is not None else None
        attention_group_ids = _argument_or_context(
            "attention_group_ids", attention_group_ids, modality
        )

        if (tokens is None) == (inputs_embeds is None):
            raise ValueError(
                "Transformer.forward requires exactly one of tokens or modality.inputs_embeds"
            )
        if tokens is not None and self.token_embedding is None:
            raise ValueError(
                "tokens were provided but the model was built with "
                "include_token_embedding=False; pass embedded inputs via modality.inputs_embeds"
            )
        if kv_caches is not None:
            if output_slice is not None:
                raise ValueError(
                    "modality.output_slice is training-only; cannot be combined with kv_caches"
                )
            if prefix_embeds is not None:
                raise ValueError(
                    "modality.prefix_embeds is training-only; cannot be combined with kv_caches"
                )
            if image_features is not None:
                raise ValueError(
                    "modality.image_features is training-only; cannot be combined with kv_caches"
                )
            if modality_ids is not None:
                raise ValueError(
                    "modality.modality_ids is training-only; cannot be combined with kv_caches"
                )
            if key_padding_mask is not None:
                raise ValueError(
                    "modality.key_padding_mask is training-only; cannot be combined with kv_caches"
                )
            if doc_ids is not None:
                raise ValueError("doc_ids is training-only; cannot be combined with kv_caches")
            if attention_group_ids is not None:
                raise ValueError(
                    "attention_group_ids is training-only; cannot be combined with kv_caches"
                )
        text = tokens if tokens is not None else inputs_embeds
        assert text is not None
        batch = text.shape[0]
        seq_len = text.shape[1] + (prefix_embeds.shape[1] if prefix_embeds is not None else 0)
        if rope_cos is not None and position_ids is not None:
            raise ValueError("modality.rope_cos/rope_sin cannot be combined with position_ids")
        if position_ids is not None:
            if self._mot_modalities:
                raise ValueError(
                    "position_ids is not supported for MoT models, which derive "
                    "per-modality positions internally"
                )
            if position_ids.dtype != torch.int64:
                raise ValueError(
                    f"position_ids.dtype must be torch.int64, got {position_ids.dtype}"
                )
            if position_ids.ndim != 2:
                raise ValueError(
                    f"position_ids must have shape (batch, seq_len), got {position_ids.ndim} dims"
                )
            if tuple(position_ids.shape) != (batch, seq_len):
                raise ValueError(
                    f"position_ids shape {tuple(position_ids.shape)} does not match the "
                    f"residual shape {(batch, seq_len)} reaching attention"
                )
            if self._rope_cos is None:
                raise ValueError(
                    "RoPE tables not initialized; call init_weights_and_freqs() after to_empty()"
                )
            n_positions = self._rope_cos.shape[0]
            if bool((position_ids < 0).any()) or bool((position_ids >= n_positions).any()):
                raise ValueError(
                    f"position_ids must be in [0, {n_positions}) (the precomputed RoPE "
                    f"table range, model.max_seq_len); got min {int(position_ids.min())}, "
                    f"max {int(position_ids.max())}"
                )
        if rope_cos is not None:
            assert rope_sin is not None
            if self._mot_modalities:
                raise ValueError(
                    "modality.rope_cos/rope_sin is not supported for MoT models, which derive "
                    "per-modality positions internally"
                )
            expected = (batch, seq_len, self.config.head_dim // 2)
            if tuple(rope_cos.shape) != expected:
                raise ValueError(
                    f"modality.rope_cos shape {tuple(rope_cos.shape)} does not match the "
                    f"residual rotation shape {expected} (batch, seq_len, head_dim // 2)"
                )
        if modality_ids is not None and modality_ids.dtype != torch.long:
            raise ValueError(
                f"modality.modality_ids.dtype must be torch.long, got {modality_ids.dtype}"
            )
        if attention_group_ids is not None and attention_group_ids.dtype != torch.long:
            raise ValueError(
                f"attention_group_ids.dtype must be torch.long, got {attention_group_ids.dtype}"
            )
        mot_order = self._mot_modalities
        has_ids = doc_ids is not None or attention_group_ids is not None
        if has_ids and mot_order and mot_order != ("image", "text"):
            raise ValueError(
                "doc_ids and attention_group_ids need vlm.mot_modalities = ('image', 'text'), "
                f"the residual order, got {mot_order!r}: MoT attends over its streams in that "
                "order, so a mask over the residual would not line up with it."
            )
        self._validate_modality_ids(modality_ids, (batch, seq_len))
        attn_mask = build_attention_mask(
            backend=self.config.attention_backend,
            batch=batch,
            seq_len=seq_len,
            device=text.device,
            doc_ids=doc_ids,
            group_ids=attention_group_ids,
            key_padding_mask=key_padding_mask,
        )

        h = (
            self.token_embedding(tokens)  # type: ignore[reportOptionalCall]
            if tokens is not None
            else inputs_embeds
        )
        assert h is not None

        if prefix_embeds is not None:
            if prefix_embeds.dtype != h.dtype:
                prefix_embeds = prefix_embeds.to(h.dtype)
            h = torch.cat([prefix_embeds, h], dim=1)

        start_pos = kv_caches[0].seq_len if kv_caches is not None else 0

        if rope_cos is not None:
            assert rope_sin is not None
            cos, sin = rope_cos.to(h.device), rope_sin.to(h.device)
        else:
            if self._rope_cos is None or self._rope_sin is None:
                raise ValueError(
                    "RoPE tables not initialized; call init_weights_and_freqs() after to_empty()"
                )
            if self._rope_cos.device != h.device:
                self._rope_cos = self._rope_cos.to(h.device)
                self._rope_sin = self._rope_sin.to(h.device)
            if position_ids is not None:
                position_ids = position_ids.to(h.device)
                cos = self._rope_cos[position_ids]
                sin = self._rope_sin[position_ids]
            else:
                cos = self._rope_cos[start_pos : start_pos + seq_len]
                sin = self._rope_sin[start_pos : start_pos + seq_len]

        if self._moma_modalities:
            assert modality_ids is not None
            for layer in self.layers.values():
                h = layer(
                    h,
                    cos,
                    sin,
                    modality_ids,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                )
            h = self.norm(h)
        elif self._mot_modalities:
            n_image = self._mot_n_image
            t_image = n_image
            t_text = h.shape[1] - n_image
            streams: dict[str, torch.Tensor] = {
                "image": h[:, :t_image, :],
                "text": h[:, t_image:, :],
            }
            rope = {
                "image": (cos[:t_image], sin[:t_image]),
                "text": (cos[:t_text], sin[:t_text]),
            }
            for layer in self.layers.values():
                streams = layer(streams, rope, attn_mask=attn_mask)
            streams = {m: self.mot_norms[m](streams[m]) for m in self._mot_modalities}
            h = torch.cat([streams["image"], streams["text"]], dim=1)
        else:
            ca_iter = iter(self.cross_attention_layers.values()) if self._ca_cadence else None
            for i, layer in enumerate(self.layers.values()):
                cache = kv_caches[i] if kv_caches is not None else None
                h = layer(h, cos, sin, kv_cache=cache, attn_mask=attn_mask)
                if ca_iter is not None and (i + 1) % self._ca_cadence == 0:
                    ca = next(ca_iter, None)
                    if ca is not None:
                        if image_features is None:
                            raise ValueError(
                                "Cross-Attention block fired but modality.image_features is None. "
                                "Cross-Attention models require image_features in the "
                                "ModalityContext."
                            )
                        if image_features.dtype != h.dtype:
                            image_features = image_features.to(h.dtype)
                        h = ca(h, image_features, image_mask)
            h = self.norm(h)

        if output_slice is not None:
            h = h[:, output_slice, :]

        if self.output_head is not None:
            h = self.output_head(h)

        return h


def _argument_or_context(
    name: str, argument: torch.Tensor | None, modality: ModalityContext | None
) -> torch.Tensor | None:
    """The forward argument ``name`` or the ``ModalityContext`` field of that name, never both."""
    from_context = getattr(modality, name) if modality is not None else None
    if argument is not None and from_context is not None:
        raise ValueError(
            f"{name} was passed to Transformer.forward and set on modality.{name}; give one"
        )
    return argument if argument is not None else from_context


@registry.register_model("transformer")
def _build_transformer(model_config: ModelConfig) -> Transformer:
    return Transformer(model_config)
