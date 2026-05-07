"""Transformer model for KempnerForge.

Architecture: Llama-style pre-norm transformer.
  Token Embedding → [TransformerBlock × N] → Final Norm → Output Head

Design choices:
  - ModuleDict (not ModuleList) for layers — preserves FQNs for DCP checkpointing.
  - Embedding and output head are optional (can be None for PP middle stages).
  - Forward is a simple loop over blocks — pipeline-parallelism friendly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from kempnerforge.config.registry import registry
from kempnerforge.config.schema import ModelConfig
from kempnerforge.config.vlm import CrossAttentionConfig, MoTConfig
from kempnerforge.model.attention import Attention, KVCache
from kempnerforge.model.cross_attention import CrossAttentionBlock
from kempnerforge.model.embedding import OutputHead, TokenEmbedding
from kempnerforge.model.init import init_weights
from kempnerforge.model.mlp import build_mlp
from kempnerforge.model.modality import ModalityContext
from kempnerforge.model.moe import MoEMLP, build_moe
from kempnerforge.model.mot import MoTBlock
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
        )

        self.mlp_norm = build_norm(config.norm_type, config.dim, eps=config.norm_eps)

        # MoE placement: with moe_frequency=1, all layers are MoE.
        # With moe_frequency=2, layers 1,3,5... are MoE (layer 0 stays dense).
        use_moe = config.is_moe and ((layer_idx + 1) % config.moe_frequency == 0)
        if use_moe:
            self.mlp = build_moe(
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
        doc_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-norm attention with residual
        x = x + self.attention(
            self.attention_norm(x), rope_cos, rope_sin, kv_cache=kv_cache, doc_ids=doc_ids
        )
        # Pre-norm MLP with residual
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Transformer(nn.Module):
    """Full transformer model built from ModelConfig.

    Embedding → TransformerBlocks → Norm → Output Head
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding (can be None for PP middle stages)
        self.token_embedding: TokenEmbedding | None = TokenEmbedding(config.vocab_size, config.dim)

        # MoT branch: build MoTBlocks instead of TransformerBlocks. v1
        # enforces equal head counts across modalities (single global
        # SDPA over the concatenated multi-modality sequence).
        self._mot_modalities: tuple[str, ...] = ()
        self._mot_n_image: int = 0
        if isinstance(config.vlm, MoTConfig):
            text_n_kv_heads = config.n_kv_heads if config.n_kv_heads is not None else config.n_heads
            img_n_heads, img_n_kv_heads = config.vlm.resolved_image_heads(
                config.n_heads, text_n_kv_heads
            )
            if img_n_heads != config.n_heads or img_n_kv_heads != text_n_kv_heads:
                raise ValueError(
                    "MoT v1 requires equal head counts across modalities (single global SDPA): "
                    f"image=({img_n_heads}, {img_n_kv_heads}) vs "
                    f"text=({config.n_heads}, {text_n_kv_heads})"
                )
            self._mot_modalities = config.vlm.mot_modalities
            self._mot_n_image = config.vlm.num_tokens
            self.layers = nn.ModuleDict(
                {
                    str(i): MoTBlock(config, modalities=self._mot_modalities, layer_idx=i)
                    for i in range(config.n_layers)
                }
            )
        else:
            # Transformer blocks — use ModuleDict to preserve FQNs for DCP
            self.layers = nn.ModuleDict(
                {str(i): TransformerBlock(config, layer_idx=i) for i in range(config.n_layers)}
            )

        # Cross-Attention layers (only populated when vlm is a
        # CrossAttentionConfig). Empty ModuleDict registers no
        # state_dict keys, so JD checkpoints load unchanged on builds
        # where this dict ends up empty.
        self.cross_attention_layers: nn.ModuleDict = nn.ModuleDict()
        self._ca_cadence: int = 0
        if isinstance(config.vlm, CrossAttentionConfig):
            self._ca_cadence = config.vlm.cross_attention_every_n_layers
            n_h, n_kv = config.vlm.resolved_heads(config.n_heads)
            num_ca_blocks = config.n_layers // self._ca_cadence
            for k in range(num_ca_blocks):
                self.cross_attention_layers[str(k)] = CrossAttentionBlock(
                    dim=config.dim,
                    n_heads=n_h,
                    n_kv_heads=n_kv,
                    ffn_hidden_dim=config.computed_ffn_hidden_dim,
                    norm_type=config.norm_type,
                    activation=config.activation,
                )

        # Final normalization. Used by the non-MoT path. MoT uses
        # per-modality ``mot_norms`` instead; ``self.norm`` is built
        # regardless so cross-arch DCP loads can carry ``norm.weight``
        # uniformly. When MoT is active, ``self.norm`` is unused in
        # forward and is frozen so it does not appear as an orphan
        # trainable parameter (its grad would always be ``None``).
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

        # Output head (can be None for PP non-final stages)
        self.output_head: OutputHead | None = OutputHead(config.dim, config.vocab_size)

        # Weight tying
        can_tie = self.token_embedding is not None and self.output_head is not None
        if config.tie_embeddings and can_tie:
            self.output_head.tie_weights(self.token_embedding)

        # Precompute RoPE cos/sin tables — stored as plain attributes (not buffers)
        # so model.to(bf16) doesn't cast them from float32.
        # Skip when on meta device (no data); call init_weights_and_freqs() later.
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

    def get_expert_counts(self) -> dict[int, torch.Tensor]:
        """Collect per-layer expert utilization. Returns {} if dense."""
        counts = {}
        for name, layer in self.layers.items():
            if isinstance(layer.mlp, MoEMLP):
                counts[int(name)] = layer.mlp.expert_counts
        return counts

    def forward(
        self,
        tokens: torch.Tensor | None = None,
        *,
        modality: ModalityContext | None = None,
        kv_caches: list[KVCache] | None = None,
        doc_ids: torch.Tensor | None = None,
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
                pure text-only forward.
            kv_caches: Optional list of KVCache (one per layer) for
                generation. When provided, RoPE positions are offset by
                the current cache fill level. Cross-arg invariant:
                ``kv_caches`` forbids ``modality.prefix_embeds``,
                ``modality.output_slice``, ``modality.image_features``,
                and ``modality.modality_ids``
                (all training-only).
            doc_ids: Optional per-token document IDs for packed sequences,
                shape ``(batch, seq_len)``. Enables block-diagonal causal
                attention that isolates documents within packed sequences.

        Returns:
            Logits tensor of shape ``(batch, out_seq_len, vocab_size)``
            where ``out_seq_len == seq_len`` normally or the sliced
            length when ``modality.output_slice`` is set.
        """
        inputs_embeds = modality.inputs_embeds if modality is not None else None
        prefix_embeds = modality.prefix_embeds if modality is not None else None
        output_slice = modality.output_slice if modality is not None else None
        image_features = modality.image_features if modality is not None else None
        image_mask = modality.image_mask if modality is not None else None
        modality_ids = modality.modality_ids if modality is not None else None

        if (tokens is None) == (inputs_embeds is None):
            raise ValueError(
                "Transformer.forward requires exactly one of tokens or modality.inputs_embeds"
            )
        if kv_caches is not None:
            if output_slice is not None:
                raise ValueError(
                    "modality.output_slice is training-only; cannot be combined with kv_caches"
                )
            if prefix_embeds is not None:
                # If we ever allowed both, the RoPE slice below would start at
                # start_pos (the cache's text fill level) but seq_len would
                # include the prefix — positions would double-count the prefix
                # at every decode step. Training-only.
                raise ValueError(
                    "modality.prefix_embeds is training-only; cannot be combined with kv_caches"
                )
            if image_features is not None:
                raise ValueError(
                    "modality.image_features is training-only; cannot be combined with kv_caches"
                )
            if modality_ids is not None:
                # MoT routes per-token through per-modality projections via
                # modality_ids; KV-cache decode has no equivalent semantics
                # (cache positions are pre-routed). Training-only for v1.
                raise ValueError(
                    "modality.modality_ids is training-only; cannot be combined with kv_caches"
                )
        # modality_ids dtype/shape checks. Shape against the residual seq_len
        # is checked downstream (when the residual is built); dtype is
        # checked here so the error fires before any compute.
        if modality_ids is not None and modality_ids.dtype != torch.long:
            raise ValueError(
                f"modality.modality_ids.dtype must be torch.long, got {modality_ids.dtype}"
            )

        h = (
            self.token_embedding(tokens)  # type: ignore[reportOptionalCall]
            if tokens is not None
            else inputs_embeds
        )
        assert h is not None  # narrowed by the XOR check above

        if prefix_embeds is not None:
            # Cast to the text-embedding dtype so the concat does not promote.
            if prefix_embeds.dtype != h.dtype:
                prefix_embeds = prefix_embeds.to(h.dtype)
            h = torch.cat([prefix_embeds, h], dim=1)

        seq_len = h.shape[1]

        # Determine position offset from KV cache fill level
        start_pos = kv_caches[0].seq_len if kv_caches is not None else 0

        # Slice RoPE frequencies for current positions (device transfer cached)
        if self._rope_cos.device != h.device:  # type: ignore[reportOptionalMemberAccess]
            self._rope_cos = self._rope_cos.to(h.device)  # type: ignore[reportOptionalMemberAccess]
            self._rope_sin = self._rope_sin.to(h.device)  # type: ignore[reportOptionalMemberAccess]
        cos = self._rope_cos[start_pos : start_pos + seq_len]  # type: ignore[reportOptionalSubscript]
        sin = self._rope_sin[start_pos : start_pos + seq_len]  # type: ignore[reportOptionalSubscript]

        # MoT path: position-based image-then-text split, per-modality
        # streams through the MoTBlock stack, single global SDPA per
        # layer. modality_ids is required (presence + shape checked
        # against the residual). v1 uses position-based routing; the
        # tags are validated for shape but not value-matched against
        # positions, so a future per-token scatter/gather can land
        # without changing the public interface.
        if self._mot_modalities:
            if modality_ids is None:
                raise ValueError(
                    "MoT model requires modality.modality_ids (got None). Build the "
                    "ModalityContext via MoTStrategy or set modality_ids explicitly."
                )
            if modality_ids.shape != h.shape[:2]:
                raise ValueError(
                    f"modality.modality_ids shape {tuple(modality_ids.shape)} does not "
                    f"match residual shape {tuple(h.shape[:2])}"
                )
            n_image = self._mot_n_image
            t_image = n_image
            t_text = h.shape[1] - n_image
            streams: dict[str, torch.Tensor] = {
                "image": h[:, :t_image, :],
                "text": h[:, t_image:, :],
            }
            # Per-modality RoPE: each modality counts positions from 0
            # within its own axis. Image and text share the same RoPE
            # table since head_dim is shared.
            rope = {
                "image": (cos[:t_image], sin[:t_image]),
                "text": (cos[:t_text], sin[:t_text]),
            }
            for layer in self.layers.values():
                streams = layer(streams, rope)
            streams = {m: self.mot_norms[m](streams[m]) for m in self._mot_modalities}
            # Re-concat in image-then-text order to match the residual
            # layout the rest of forward expects (output_slice + head).
            h = torch.cat([streams["image"], streams["text"]], dim=1)
        else:
            # Transformer blocks. When the model has cross-attention layers
            # (CrossAttentionConfig + nonzero cadence), a CrossAttentionBlock
            # fires after text block index i iff (i+1) % _ca_cadence == 0.
            # _ca_cadence == 0 (text-only / Joint-Decoder) makes the inner
            # branch dead, so the JD path stays bit-equal to today's.
            ca_iter = iter(self.cross_attention_layers.values()) if self._ca_cadence else None
            for i, layer in enumerate(self.layers.values()):
                cache = kv_caches[i] if kv_caches is not None else None
                h = layer(h, cos, sin, kv_cache=cache, doc_ids=doc_ids)
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
            # Final norm
            h = self.norm(h)

        # Optional slice before the output head (training-only kwarg)
        if output_slice is not None:
            h = h[:, output_slice, :]

        # Output projection
        if self.output_head is not None:
            h = self.output_head(h)

        return h


@registry.register_model("transformer")
def _build_transformer(model_config: ModelConfig) -> Transformer:
    return Transformer(model_config)
