"""Mixture of Modality-Aware Experts (MoMa) operator, FFN, and block.

Implements Lin et al. 2024 ("MoMa: Efficient Early-Fusion Pre-training with
Mixture of Modality-Aware Experts", arXiv:2407.21770) on top of KempnerForge's
existing VLM stack.

Architecture at a glance, per transformer layer:

- Pre-norm ``Attention`` (the standard module, shared Q/K/V/O across
  modalities) running a single global SDPA over the concatenated image+text
  sequence.
- Pre-norm ``MoMaFFN``: a ``ModuleDict`` of per-modality ``ExpertChoiceMoE``
  groups dispatched by ``modality_ids``. Each group's MoE uses
  Expert-Choice + Sigmoid routing (paper §2.2): each expert independently
  picks its top-``k_e`` tokens by sigmoid score, and the token output is
  the sum across experts that selected it, weighted by their sigmoid
  scores (Eq. 1). Optional Gumbel-Sigmoid noise during training (Eq. 5).

Differs from MoT (also in this codebase): MoT has *per-modality* Q/K/V/O
projections and per-modality FFN. MoMa has *shared* Q/K/V/O and per-modality
*MoE* FFN groups (multiple experts per modality, learned routing within
each group). Both share the residual-stream layout (image tokens prepended
to text) and ``modality_ids`` tagging mechanism.

The module exposes four public symbols:

- ``ExpertChoiceSigmoidRouter`` — per-modality gate (``W_g^M``), Sigmoid
  scoring, optional Gumbel noise, and per-expert top-``k_e`` token
  selection.
- ``ExpertChoiceMoE`` — composes a router + ``num_experts`` SwiGLU
  experts; forward(x) returns the sigmoid-weighted expert combination.
- ``MoMaFFN`` — holds one ``ExpertChoiceMoE`` per modality and dispatches
  tokens via ``modality_ids``.
- ``MoMaBlock`` — pre-norm block: shared ``Attention`` + ``MoMaFFN``.

Inference note: expert-choice routing is non-causal (each expert's
top-``k_e`` depends on all tokens in the batch). v1 supports training only;
autoregressive generation requires auxiliary routers (paper §2.4), deferred
to a follow-up PR.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from kempnerforge.config.schema import ModelConfig
from kempnerforge.model.attention import Attention
from kempnerforge.model.mlp import build_mlp
from kempnerforge.model.norm import build_norm


def _gumbel_like(x: torch.Tensor) -> torch.Tensor:
    """Sample Gumbel(0, 1) noise with the same shape and dtype as ``x``.

    Uses the standard inverse-CDF trick on a clamped uniform sample. The
    clamps avoid ``log(0)`` from drawing exactly zero (rare but possible
    at low precisions) or ``log(1)`` (where ``-log(u) == 0`` makes the
    outer ``log`` blow up). The intermediate parenthesization is load-
    bearing: ``(-torch.log(u)).clamp_min(...)`` clamps the *positive*
    side, whereas ``-torch.log(u).clamp_min(...)`` would clamp the
    negative ``log(u)`` first (collapsing everything to ``-1e-9`` and
    then triggering ``log`` of a negative number = NaN).
    """
    u = torch.rand_like(x).clamp_min(1e-9)
    return -torch.log((-torch.log(u)).clamp_min(1e-9))


class ExpertChoiceSigmoidRouter(nn.Module):
    """Expert-Choice + Sigmoid router for one modality group (Lin et al. 2024 §2.2).

    Scoring: ``score = sigmoid(W_g x)`` per token-expert pair (independent
    across experts because Sigmoid does not normalize). Optional Gumbel
    perturbation during training: ``Gumbel-Sigmoid(x) = sigmoid(x + G' - G'')``
    with independent Gumbel(0, 1) samples ``G', G''`` (paper Eq. 5).

    Selection: each expert independently picks its top-``k_e`` tokens by
    score (``torch.topk`` on the (expert, token) score matrix). This is
    the inverse of token-choice routing: there a token picks experts;
    here an expert picks tokens. A token can be picked by 0, 1, or more
    experts (the residual stream carries the unmodified token through
    when no expert picks it).

    ``capacity_factor`` controls ``k_e`` as ``k_e = ceil(c_e * N)`` where
    ``N`` is the number of tokens of this modality in the current batch.
    The paper's default ``c_e = 1/|E^M|`` gives ``k_e ≈ N/|E^M|`` so each
    expert sees the average load (perfect balance under EC routing).
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        capacity_factor: float,
        gumbel_noise: bool = True,
    ) -> None:
        super().__init__()
        if num_experts <= 0:
            raise ValueError(
                f"ExpertChoiceSigmoidRouter: num_experts must be positive (got {num_experts})"
            )
        if capacity_factor <= 0:
            raise ValueError(
                "ExpertChoiceSigmoidRouter: capacity_factor must be positive "
                f"(got {capacity_factor})"
            )
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.gumbel_noise = gumbel_noise
        # Tracked for metrics / debugging (analogous to MoEMLP.expert_counts).
        self.expert_counts: torch.Tensor = torch.zeros(num_experts)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Route tokens to experts via expert-choice.

        Args:
            x: ``(N, D)`` token representations for one modality group.

        Returns:
            ``topk_scores``: ``(E, k_e)`` sigmoid scores of the tokens each
                expert selected.
            ``topk_indices``: ``(E, k_e)`` token indices into ``x`` that
                each expert selected. ``k_e`` is computed from
                ``capacity_factor * N``, capped by ``N``.
        """
        if x.dim() != 2:
            raise ValueError(
                f"ExpertChoiceSigmoidRouter.forward expects (N, D); got shape {tuple(x.shape)}"
            )
        n_tokens, _ = x.shape
        if n_tokens == 0:
            # Empty modality slice — return empty selections; caller handles.
            empty_scores = x.new_zeros(self.num_experts, 0)
            empty_indices = torch.zeros(self.num_experts, 0, dtype=torch.long, device=x.device)
            return empty_scores, empty_indices

        logits = self.gate(x)  # (N, E)
        if self.training and self.gumbel_noise:
            logits = logits + _gumbel_like(logits) - _gumbel_like(logits)
        scores = torch.sigmoid(logits)  # (N, E), independent per expert

        k_e = max(1, math.ceil(self.capacity_factor * n_tokens))
        k_e = min(k_e, n_tokens)

        # scores.t(): (E, N). For each expert (row), select the top-k_e tokens.
        topk_scores, topk_indices = torch.topk(scores.t(), k=k_e, dim=1)

        # Per-expert utilization metric: how many tokens this expert handled.
        # Always k_e (EC routing guarantees this), but recording for parity
        # with the MoEMLP API.
        with torch.no_grad():
            counts = torch.full(
                (self.num_experts,), float(k_e), device=x.device, dtype=torch.float32
            )
            self.expert_counts = counts.detach()

        return topk_scores, topk_indices


class ExpertChoiceMoE(nn.Module):
    """Expert-Choice MoE for one modality group.

    Composes an ``ExpertChoiceSigmoidRouter`` with ``num_experts`` SwiGLU
    expert MLPs. Forward: each expert selects top-``k_e`` tokens, runs its
    MLP on those tokens, and contributes ``sigmoid_score * MLP(x)`` to the
    output. Tokens not picked by any expert receive zero contribution from
    this MoE block (the outer residual skip preserves them).

    State-dict layout (FQN-stable):

    .. code::

        router.gate.weight    # (num_experts, dim) — gate Linear
        experts.0.gate_proj.weight
        experts.0.up_proj.weight
        experts.0.down_proj.weight
        experts.1...
        ...
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        capacity_factor: float,
        activation: str = "silu",
        gumbel_noise: bool = True,
    ) -> None:
        super().__init__()
        if num_experts <= 0:
            raise ValueError(f"ExpertChoiceMoE: num_experts must be positive (got {num_experts})")
        self.dim = dim
        self.num_experts = num_experts
        self.router = ExpertChoiceSigmoidRouter(
            dim=dim,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
            gumbel_noise=gumbel_noise,
        )
        self.experts = nn.ModuleList(
            [
                build_mlp(dim=dim, hidden_dim=hidden_dim, activation=activation)
                for _ in range(num_experts)
            ]
        )

    @property
    def expert_counts(self) -> torch.Tensor:
        """Per-expert token count from the most recent forward (metrics)."""
        return self.router.expert_counts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expert-choice MoE forward over one modality group.

        Args:
            x: ``(N, D)`` token representations.

        Returns:
            ``(N, D)`` output where each token has accumulated weighted
            outputs from every expert that selected it (zero contribution
            from this block when no expert selected the token).
        """
        if x.dim() != 2:
            raise ValueError(f"ExpertChoiceMoE.forward expects (N, D); got shape {tuple(x.shape)}")
        n_tokens, _ = x.shape
        if n_tokens == 0:
            return x  # Pass through empty modality slice.

        topk_scores, topk_indices = self.router(x)  # (E, k_e) each

        out = torch.zeros_like(x)
        # Sequential per-expert dispatch (the codebase's MoEMLP fallback
        # uses the same sequential loop pattern). Grouped-GEMM EC dispatch
        # is a future optimization once the operator is stable.
        for e in range(self.num_experts):
            token_idx = topk_indices[e]  # (k_e,)
            token_scores = topk_scores[e]  # (k_e,)
            x_e = x.index_select(0, token_idx)  # (k_e, D)
            out_e = self.experts[e](x_e)  # (k_e, D)
            weighted = token_scores.unsqueeze(-1) * out_e  # (k_e, D)
            # index_add (non-in-place) accumulates contributions from
            # multiple experts that picked the same token. Autograd-safe
            # for non-unique indices.
            out = out.index_add(0, token_idx, weighted)
        return out


class MoMaFFN(nn.Module):
    """Per-modality MoE FFN groups dispatched by ``modality_ids``.

    Holds one ``ExpertChoiceMoE`` per modality (keys: modality name).
    Forward dispatches tokens by ``modality_ids`` (level-1 deterministic
    routing), runs each modality's EC-MoE (level-2 learned routing),
    then scatters per-modality outputs back to their original positions.

    Modality index convention: ``self.modalities[i]`` corresponds to
    ``modality_ids == i``. With the default ``("image", "text")``,
    ``modality_ids == 0`` selects the image expert group and
    ``modality_ids == 1`` selects the text expert group.
    """

    def __init__(
        self,
        config: ModelConfig,
        modalities: tuple[str, ...],
        experts_per_modality: dict[str, int],
        capacity_factor_per_modality: dict[str, float],
        gumbel_noise: bool = True,
    ) -> None:
        super().__init__()
        if not modalities:
            raise ValueError("MoMaFFN requires at least one modality")
        missing_experts = set(modalities) - set(experts_per_modality.keys())
        if missing_experts:
            raise ValueError(
                f"MoMaFFN: experts_per_modality missing entries for {sorted(missing_experts)}"
            )
        missing_cap = set(modalities) - set(capacity_factor_per_modality.keys())
        if missing_cap:
            raise ValueError(
                f"MoMaFFN: capacity_factor_per_modality missing entries for {sorted(missing_cap)}"
            )
        self.modalities = tuple(modalities)
        self.experts = nn.ModuleDict(
            {
                m: ExpertChoiceMoE(
                    dim=config.dim,
                    hidden_dim=config.computed_expert_ffn_hidden_dim,
                    num_experts=experts_per_modality[m],
                    capacity_factor=capacity_factor_per_modality[m],
                    activation=config.activation,
                    gumbel_noise=gumbel_noise,
                )
                for m in self.modalities
            }
        )

    def forward(
        self,
        x: torch.Tensor,
        modality_ids: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dispatch tokens by modality and run per-modality EC-MoE.

        Args:
            x: ``(B, S, D)`` residual stream.
            modality_ids: ``(B, S)`` long tensor. ``modality_ids == i``
                routes that token to ``self.modalities[i]``'s expert
                group.
            key_padding_mask: Optional ``(B, S)`` bool mask; ``False`` positions
                (e.g. padded video frames) are excluded from the expert-choice
                routing so they neither consume expert capacity nor perturb which
                real tokens the experts select. ``None`` = all positions routed.

        Returns:
            ``(B, S, D)`` tensor with each modality's positions filled
            by its EC-MoE output. Positions whose modality has no tokens
            assigned by any expert get zeros (the outer residual skip
            preserves them).
        """
        if x.dim() != 3:
            raise ValueError(f"MoMaFFN.forward expects (B, S, D); got shape {tuple(x.shape)}")
        if modality_ids.dim() != 2 or modality_ids.shape != x.shape[:2]:
            raise ValueError(
                f"MoMaFFN.forward: modality_ids shape {tuple(modality_ids.shape)} does not "
                f"match (B, S) = {tuple(x.shape[:2])}"
            )
        if modality_ids.dtype != torch.long:
            raise ValueError(
                f"MoMaFFN.forward: modality_ids dtype must be torch.long (got {modality_ids.dtype})"
            )

        b, s, d = x.shape
        x_flat = x.reshape(b * s, d)
        mod_flat = modality_ids.reshape(b * s)
        valid_flat = key_padding_mask.reshape(b * s) if key_padding_mask is not None else None
        out = torch.zeros_like(x_flat)

        # ``total_routed`` counts positions matching *some* modality group (the
        # modality_ids range check below); with well-formed modality_ids it
        # equals b*s. We accumulate Python ints from ``idx.numel()`` (tensor
        # metadata, no host sync) rather than a ``.all()`` reduction that would
        # force a device->host sync every step. The per-modality routing set
        # additionally drops padded positions so they never compete for capacity.
        total_routed = 0
        for i, m in enumerate(self.modalities):
            # nonzero() avoids the boolean-mask copy and gives a 1-D index
            # tensor we can feed to index_select + scatter.
            mod_idx = (mod_flat == i).nonzero(as_tuple=False).squeeze(-1)  # (N_m,)
            total_routed += mod_idx.numel()
            idx = mod_idx
            if valid_flat is not None:
                # Drop padded (e.g. blank-frame) positions from the expert-choice
                # competition: padded tokens must not consume expert capacity or
                # change which real tokens the experts pick. They get zero FFN
                # output; the outer residual skip carries them through unchanged.
                keep = valid_flat.index_select(0, mod_idx).nonzero(as_tuple=False).squeeze(-1)
                idx = mod_idx.index_select(0, keep)
            if idx.numel() == 0:
                continue
            x_m = x_flat.index_select(0, idx)  # (N_routed, D)
            y_m = self.experts[m](x_m)  # (N_routed, D)
            # The modality groups partition the position space, so indices are
            # unique across iterations; index_copy on disjoint indices is safe
            # and autograd-friendly.
            out = out.index_copy(0, idx, y_m)

        if total_routed != b * s:
            raise ValueError(
                f"MoMaFFN.forward: modality_ids contains out-of-range values; "
                f"{b * s - total_routed} of {b * s} positions did not match any "
                f"modality (allowed values: 0..{len(self.modalities) - 1} for "
                f"modalities {self.modalities!r})"
            )
        return out.view(b, s, d)


class MoMaBlock(nn.Module):
    """Pre-norm transformer block: shared ``Attention`` + ``MoMaFFN``.

    Operates on a single residual tensor ``(B, S, D)`` like the dense
    ``TransformerBlock`` (unlike ``MoTBlock`` which operates on a
    per-modality dict of streams). The only structural difference from
    ``TransformerBlock`` is the FFN: ``MoMaFFN`` instead of a dense MLP
    or a flat MoE.

    State-dict layout:

    .. code::

        attention_norm.weight
        attention.q_proj.weight
        attention.k_proj.weight
        attention.v_proj.weight
        attention.o_proj.weight
        # qk_norm only:
        attention.q_norm.weight
        attention.k_norm.weight
        mlp_norm.weight
        mlp.experts.{m}.router.gate.weight
        mlp.experts.{m}.experts.0.gate_proj.weight
        mlp.experts.{m}.experts.0.up_proj.weight
        mlp.experts.{m}.experts.0.down_proj.weight
        mlp.experts.{m}.experts.1...
        ...
    """

    def __init__(
        self,
        config: ModelConfig,
        modalities: tuple[str, ...],
        experts_per_modality: dict[str, int],
        capacity_factor_per_modality: dict[str, float],
        gumbel_noise: bool,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.modalities = tuple(modalities)

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
        self.mlp = MoMaFFN(
            config,
            modalities=self.modalities,
            experts_per_modality=experts_per_modality,
            capacity_factor_per_modality=capacity_factor_per_modality,
            gumbel_noise=gumbel_noise,
        )

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        modality_ids: torch.Tensor,
        *,
        doc_ids: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-norm attention with residual (shared QKVO, single SDPA).
        # kv_cache is intentionally omitted: EC routing is non-causal in v1.
        x = x + self.attention(
            self.attention_norm(x),
            rope_cos,
            rope_sin,
            doc_ids=doc_ids,
            key_padding_mask=key_padding_mask,
        )
        # Pre-norm MoMa FFN with residual (per-modality EC-MoE groups). The
        # key_padding_mask drops padded positions from expert-choice routing.
        x = x + self.mlp(
            self.mlp_norm(x), modality_ids=modality_ids, key_padding_mask=key_padding_mask
        )
        return x
