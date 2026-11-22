"""MoE router implementations for KempnerForge models."""

from __future__ import annotations

import math
from collections.abc import Collection
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, distribute_tensor

from kempnerforge.config.registry import registry


class SoftmaxTopKRouter(nn.Module):
    """Mixtral-style softmax top-k router with auxiliary load-balancing loss.

    Each token independently selects top_k experts. Routing weights are
    softmax-normalized, then renormalized over the selected top_k.

    Stores ``aux_loss`` and ``expert_counts`` as side effects of each forward
    call for collection by the training loop and metrics.
    """

    def __init__(self, dim: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss = torch.tensor(0.0)
        self.z_loss = torch.tensor(0.0)
        self.expert_counts: torch.Tensor = torch.zeros(num_experts)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Route tokens to experts.

        Args:
            x: (num_tokens, dim) — flattened token representations.

        Returns:
            weights:        (num_tokens, top_k) — renormalized routing weights.
            expert_indices: (num_tokens, top_k) — selected expert indices.
        """
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)

        weights, indices = torch.topk(probs, k=self.top_k, dim=-1)

        weights = weights / weights.sum(dim=-1, keepdim=True)

        num_tokens = x.shape[0]

        one_hot = F.one_hot(indices, num_classes=self.num_experts).float()
        tokens_per_expert = one_hot.sum(dim=(0, 1))
        f = tokens_per_expert / (num_tokens * self.top_k)

        p = probs.mean(dim=0)

        if num_tokens == 0:
            self.aux_loss = self.z_loss = logits.sum() * 0
        else:
            self.aux_loss = self.num_experts * (f * p).sum()
            self.z_loss = (torch.logsumexp(logits, dim=-1) ** 2).mean()
        self.expert_counts = tokens_per_expert.detach()

        return weights, indices


class SigmoidTopKRouter(nn.Module):
    """DeepSeek-V3 style sigmoid router with auxiliary-loss-free balancing.

    Uses per-expert sigmoid scoring (each expert scored independently) instead
    of softmax. Load balancing is maintained by ``expert_bias``, which the
    optimizer never trains: ``balance`` moves it once per optimizer step from a
    running EMA of expert utilization, so no auxiliary loss term is added to the
    training loss by default.

    The bias adjustment nudges under-utilized experts up and over-utilized
    experts down, achieving balance without interfering with the main loss
    gradient signal. A training forward adds its expert counts to
    ``pending_counts``, which ``balance`` consumes. ``derived_buffers`` names the
    buffers a load derives from the other state when a checkpoint lacks them.

    ``always_frozen`` names the parameters that ``freeze_params`` never unfreezes.

    Optional enhancements (all disabled by default for backward compatibility):

    - **Sequence-level auxiliary loss**: lightweight variance-based balance
      penalty (10-100x smaller than Switch Transformer's aux loss) that
      complements bias balancing to prevent degenerate routing in long runs.
    - **Adaptive bias schedule**: decays or warms up the bias update rate
      over training to stabilize routing after initial exploration.
    """

    always_frozen = ("expert_bias",)
    derived_buffers = ("fp32_ema_bits", "fp32_bias_bits")
    expert_ema: torch.Tensor
    pending_counts: torch.Tensor
    fp32_ema_bits: torch.Tensor
    fp32_bias_bits: torch.Tensor

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        bias_update_rate: float = 0.001,
        ema_decay: float = 0.99,
        sequence_aux_loss_weight: float = 0.0,
        bias_schedule: str = "constant",
    ) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.num_experts = num_experts
        self.top_k = top_k
        self.bias_update_rate = bias_update_rate
        self.ema_decay = ema_decay
        self.sequence_aux_loss_weight = sequence_aux_loss_weight
        self.bias_schedule = bias_schedule

        self._step: int = 0
        self._max_steps: int = 1

        self.expert_bias = nn.Parameter(torch.zeros(num_experts), requires_grad=False)
        self.register_buffer("expert_ema", torch.ones(num_experts) / num_experts)
        self.register_buffer(
            "pending_counts", torch.zeros(num_experts, dtype=torch.long), persistent=False
        )
        self.register_buffer("fp32_ema_bits", self.expert_ema.clone().view(torch.int32))
        self.register_buffer("fp32_bias_bits", torch.zeros(num_experts).view(torch.int32))

        self.aux_loss = torch.tensor(0.0)
        self.z_loss = torch.tensor(0.0)
        self.expert_counts: torch.Tensor = torch.zeros(num_experts)

    def reset_parameters(self) -> None:
        """Restore this router's construction-time state.

        ``to_empty()`` on a meta build leaves both holding whatever the
        allocator returns, and ``init_weights`` covers neither: ``expert_bias``
        is one-dimensional and ``expert_ema`` is a buffer.
        """
        nn.init.zeros_(self.expert_bias)
        self.expert_ema.fill_(1.0 / self.num_experts)
        self.pending_counts.zero_()
        ema, bias = self._masters()
        ema.fill_(1.0 / self.num_experts)
        bias.zero_()

    def set_step(self, step: int, max_steps: int) -> None:
        """Set current training step for adaptive bias scheduling."""
        self._step = step
        self._max_steps = max(max_steps, 1)

    def _effective_bias_rate(self) -> float:
        """Compute bias update rate based on schedule and current step."""
        rate = self.bias_update_rate
        if self.bias_schedule == "constant":
            return rate
        progress = self._step / self._max_steps
        if self.bias_schedule == "cosine_decay":
            return rate * 0.5 * (1.0 + math.cos(math.pi * progress))
        if self.bias_schedule == "linear_warmup":
            return rate * min(1.0, progress * 10.0)
        return rate

    def _masters(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The FP32 master values of the EMA and the bias, as views of their bit buffers.

        They are kept as int32 bit patterns so that casting the module to a low precision
        dtype leaves them exact; they are checkpointed with the module.
        """
        return self.fp32_ema_bits.view(torch.float32), self.fp32_bias_bits.view(torch.float32)

    def balance(self, counts: torch.Tensor) -> None:
        """Move ``expert_bias`` toward balance after an optimizer step, and clear the counts.

        ``counts`` are the step's expert counts, summed over its micro-batches and every
        rank: utilization is each expert's share of them, so ranks holding the same tokens,
        as under tensor parallelism, cancel out. The EMA of utilization moves toward this
        step's by ``1 - ema_decay``, and each expert's bias by the effective rate times its
        shortfall from ``1 / num_experts``. The arithmetic runs on the FP32 masters (see
        ``_masters``), so that low-precision storage does not round the small moves away,
        and ``expert_ema`` and ``expert_bias`` are rounded from them; under FSDP2 the bias
        is updated in place on this rank's shard.
        """
        self.pending_counts.zero_()
        total = int(counts.sum())
        if total == 0:
            return
        ema, master = self._masters()
        ema.lerp_(counts.to(ema) / total, 1.0 - self.ema_decay)
        master.add_(self._effective_bias_rate() * (1.0 / self.num_experts - ema))
        bias = self.expert_bias
        with torch.no_grad():
            self.expert_ema.copy_(ema)
            if isinstance(bias, DTensor):
                shard = distribute_tensor(
                    master.to(bias.dtype), bias.device_mesh, bias.placements, src_data_rank=None
                )
                bias.copy_(shard)
            else:
                bias.copy_(master)

    def derive_buffers(self, names: Collection[str]) -> None:
        """Start the named FP32 masters from ``expert_ema`` and ``expert_bias`` as loaded.

        For a load from a checkpoint that lacks them; a master not named keeps its loaded
        value. Deriving the bias master gathers a bias that FSDP2 shards, so every rank
        calls it.
        """
        ema, master = self._masters()
        with torch.no_grad():
            if "fp32_ema_bits" in names:
                ema.copy_(self.expert_ema)
            if "fp32_bias_bits" in names:
                bias = self.expert_bias
                master.copy_(bias.full_tensor() if isinstance(bias, DTensor) else bias)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Load, deriving the FP32 masters from the EMA and the bias when they are absent.

        A checkpoint written before the masters existed holds only ``expert_ema`` and
        ``expert_bias``; the masters then start from those values.
        """
        for bits, source in (("fp32_ema_bits", "expert_ema"), ("fp32_bias_bits", "expert_bias")):
            loaded = state_dict.get(prefix + source)
            if prefix + bits not in state_dict and loaded is not None:
                full = loaded.full_tensor() if isinstance(loaded, DTensor) else loaded
                state_dict[prefix + bits] = full.detach().float().contiguous().view(torch.int32)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Route tokens to experts using sigmoid scoring.

        Args:
            x: (num_tokens, dim) — flattened token representations.

        Returns:
            weights:        (num_tokens, top_k) — routing weights (sigmoid scores).
            expert_indices: (num_tokens, top_k) — selected expert indices.
        """
        logits = self.gate(x)
        scores = torch.sigmoid(logits + self.expert_bias)

        weights, indices = torch.topk(scores, k=self.top_k, dim=-1)

        weights = weights / weights.sum(dim=-1, keepdim=True)

        num_tokens = x.shape[0]
        one_hot = F.one_hot(indices, num_classes=self.num_experts).float()
        tokens_per_expert = one_hot.sum(dim=(0, 1))
        self.expert_counts = tokens_per_expert.detach()

        if num_tokens == 0:
            self.aux_loss = self.z_loss = scores.sum() * 0
            return weights, indices

        if self.training:
            self.pending_counts.add_(tokens_per_expert.detach().long())

        if self.sequence_aux_loss_weight > 0:
            f = tokens_per_expert.detach() / (num_tokens * self.top_k + 1e-8)
            p = scores.mean(dim=0)
            balance_loss = self.num_experts * (f * p).sum()
            self.aux_loss = self.sequence_aux_loss_weight * balance_loss
        else:
            self.aux_loss = torch.tensor(0.0, device=x.device)

        self.z_loss = (torch.logsumexp(logits, dim=-1) ** 2).mean()

        return weights, indices


def _build_softmax_topk(dim: int, num_experts: int, top_k: int) -> SoftmaxTopKRouter:
    return SoftmaxTopKRouter(dim, num_experts, top_k)


def _build_sigmoid_topk(dim: int, num_experts: int, top_k: int, **kwargs: Any) -> SigmoidTopKRouter:
    return SigmoidTopKRouter(dim, num_experts, top_k, **kwargs)


registry.register("router", "softmax_topk", _build_softmax_topk)
registry.register("router", "sigmoid_topk", _build_sigmoid_topk)
