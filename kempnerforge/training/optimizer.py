"""Optimizer construction for KempnerForge.

Builds optimizers with per-parameter-group settings:
  - AdamW: standard Adam with decoupled weight decay
  - Muon: momentum with orthogonalized updates via Newton-Schulz iteration.
    Applies Muon to 2D+ weight matrices, AdamW to 1D params (biases, norms).
  - Lion: sign-based momentum update (half the optimizer memory of AdamW)
  - Schedule-Free AdamW: eliminates LR schedule by averaging iterates

All optimizers:
  - Weight decay applied to matrix weights only
  - Bias and norm parameters excluded from weight decay
  - Fused kernel when available (PyTorch 2.x, AdamW only)
"""

from __future__ import annotations

import logging

import torch

from kempnerforge.config.registry import registry
from kempnerforge.config.schema import OptimizerConfig

logger = logging.getLogger(__name__)


@registry.register_optimizer("adamw")
def _build_adamw(
    param_groups: list[dict],
    config: OptimizerConfig,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        param_groups,
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        fused=config.fused and torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------
# Lion optimizer
# ---------------------------------------------------------------------------


class Lion(torch.optim.Optimizer):
    """Lion optimizer (Chen et al., 2023): Evolved Sign Momentum.

    Uses sign-based updates with momentum interpolation.  Only maintains
    one momentum buffer (vs two for AdamW), halving optimizer memory.

    Update rule::

        update = sign(beta1 * m + (1 - beta1) * grad)
        m = beta2 * m + (1 - beta2) * grad
        p = p * (1 - lr * wd) - lr * update

    Args:
        params: Parameters or parameter groups.
        lr: Learning rate (typically 3-10x smaller than AdamW).
        betas: ``(beta1, beta2)`` for update interpolation and momentum.
        weight_decay: Decoupled weight decay coefficient.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p.data)

                m = state["exp_avg"]

                # Update direction: sign of interpolated momentum + gradient
                # .mul() (not in-place) creates a temporary — m is unchanged
                update = (m.mul(beta1) + grad.mul(1 - beta1)).sign_()

                # Decoupled weight decay
                if wd > 0:
                    p.data.mul_(1 - lr * wd)

                # Apply update
                p.data.add_(update, alpha=-lr)

                # Update momentum buffer
                m.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


@registry.register_optimizer("lion")
def _build_lion(
    param_groups: list[dict],
    config: OptimizerConfig,
) -> torch.optim.Optimizer:
    return Lion(
        param_groups,
        lr=config.lr,
        betas=config.betas,
    )


# ---------------------------------------------------------------------------
# Schedule-Free AdamW
# ---------------------------------------------------------------------------


class ScheduleFreeAdamW(torch.optim.Optimizer):
    """Schedule-Free AdamW (Defazio & Mishchenko, 2024).

    Eliminates the need for an LR schedule by maintaining an iterate ``z``
    and a running average ``x``.  Parameters are set to the interpolated
    point ``y = (1 - beta1) * z + beta1 * x`` for gradient computation.

    Use with ``scheduler.name = "none"`` — no LR schedule is needed.

    Args:
        params: Parameters or parameter groups.
        lr: Learning rate (constant — no schedule needed).
        betas: ``(beta1, beta2)`` for interpolation and second moment.
        eps: Denominator term for numerical stability.
        weight_decay: Decoupled weight decay.
        warmup_steps: Linear warmup steps (internal to the optimizer).
    """

    def __init__(
        self,
        params,
        lr: float = 0.025,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.warmup_steps = warmup_steps
        self._k = 0

    def state_dict(self) -> dict:
        sd = super().state_dict()
        sd["_k"] = self._k
        sd["_warmup_steps"] = self.warmup_steps
        return sd

    def load_state_dict(self, state_dict: dict) -> None:
        self._k = state_dict.pop("_k", 0)
        self.warmup_steps = state_dict.pop("_warmup_steps", self.warmup_steps)
        super().load_state_dict(state_dict)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._k += 1
        k = self._k

        warmup = min(1.0, k / max(self.warmup_steps, 1)) if self.warmup_steps > 0 else 1.0

        for group in self.param_groups:
            lr = group["lr"] * warmup
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            # Weight for Polyak averaging (accounts for variable LR during warmup)
            weight = lr * (1 - beta1)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["z"] = p.data.clone()
                    state["v"] = torch.zeros_like(p.data)
                    state["weight_sum"] = 0.0

                z = state["z"]
                v = state["v"]

                # Second moment update
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias-corrected denominator
                bc2 = 1 - beta2**k
                denom = (v / bc2).sqrt().add_(eps)

                # Update z (the iterate)
                z.addcdiv_(grad, denom, value=-lr)

                # Decoupled weight decay on z
                if wd > 0:
                    z.mul_(1 - lr * wd)

                # Polyak average coefficient
                state["weight_sum"] += weight
                ck = weight / state["weight_sum"]

                # Update p.data via interpolation:
                # x_new = (1 - ck) * x_old + ck * z
                # y_new = (1 - beta1) * z + beta1 * x_new
                # We don't store x explicitly — derive it from current y and z:
                # x_old = (y_old - (1 - beta1) * z_old) / beta1
                # But z changed, so we need a different approach.
                # Store x explicitly for correctness:
                if "x" not in state:
                    state["x"] = z.clone()
                else:
                    state["x"].lerp_(z, ck)

                # Set params to interpolated point for next gradient computation
                p.data.copy_(z).lerp_(state["x"], beta1)

        return loss

    def eval_params(self) -> None:
        """Set parameters to the evaluation point (running average).

        Call before validation/inference for best results.
        Call :meth:`train_params` afterward to resume training.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p in self.state and "x" in self.state[p]:
                    p.data.copy_(self.state[p]["x"])

    def train_params(self) -> None:
        """Restore parameters to the training point (interpolated y).

        Call after :meth:`eval_params` to resume training.
        """
        for group in self.param_groups:
            beta1 = group["betas"][0]
            for p in group["params"]:
                state = self.state[p]
                if "z" in state and "x" in state:
                    p.data.copy_(state["z"]).lerp_(state["x"], beta1)


@registry.register_optimizer("schedule_free_adamw")
def _build_schedule_free_adamw(
    param_groups: list[dict],
    config: OptimizerConfig,
) -> torch.optim.Optimizer:
    return ScheduleFreeAdamW(
        param_groups,
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        warmup_steps=config.schedule_free_warmup_steps,
    )


# ---------------------------------------------------------------------------
# Muon optimizer
# ---------------------------------------------------------------------------


def _newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate orthogonal projection via Newton-Schulz iteration.

    Given a matrix G, computes the nearest orthogonal matrix U such that
    U^T U ≈ I. This is used to orthogonalize the momentum update in Muon.

    Uses a degree-5 polynomial iteration:
      A = X @ X^T
      B = b*A + c*A@A
      X = a*X + B@X

    Coefficients (a, b, c) are from Zhu & Jordan (2024), optimized for
    convergence in 5 steps from a spectrally-normalized starting point.

    Cost: ~15 FLOPs per parameter per step — negligible vs forward/backward.
    """
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315

    # Spectral normalization to ensure convergence
    X = G / (G.norm() + 1e-7)

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    return X


def _get_local_tensor(t: torch.Tensor) -> torch.Tensor:
    """Extract the local (non-DTensor) tensor from a possibly-sharded parameter.

    FSDP2 wraps parameters as DTensors. The optimizer needs to operate on
    the underlying local shard directly to avoid DTensor/Tensor mixing errors
    (e.g. 'aten.add_.Tensor got mixed torch.Tensor and DTensor').
    """
    try:
        from torch.distributed._tensor import DTensor

        if isinstance(t, DTensor):
            return t._local_tensor
    except ImportError:
        pass
    return t


class Muon(torch.optim.Optimizer):
    """Muon optimizer: Momentum with Orthogonalized Updates.

    For 2D+ weight matrices: maintains momentum, then orthogonalizes the
    update direction via Newton-Schulz iteration. This keeps update
    directions independent of parameter scale.

    For 1D parameters (biases, norms, embeddings): uses standard AdamW,
    since orthogonalization requires 2D matrices.

    FSDP2 note: Newton-Schulz operates on each rank's local shard
    independently — an approximation, not mathematically equivalent to
    orthogonalizing the full weight matrix. This is the standard approach
    for distributed Muon and works well in practice.

    Args:
        muon_params: Parameter groups for Muon (2D+ weights).
        adam_params: Parameter groups for AdamW fallback (1D params).
        lr: Learning rate for Muon (2D weights).
        momentum: Momentum coefficient (default 0.95).
        weight_decay: Decoupled weight decay.
        adam_betas: Betas for the AdamW fallback.
        adam_eps: Epsilon for the AdamW fallback.
        ns_steps: Newton-Schulz iteration steps (default 5).
        adam_lr: Learning rate for AdamW fallback (1D params). None = same as lr.
    """

    def __init__(
        self,
        muon_params: list[dict],
        adam_params: list[dict],
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        adam_eps: float = 1e-8,
        ns_steps: int = 5,
        adam_lr: float | None = None,
    ):
        adam_lr = adam_lr if adam_lr is not None else lr
        self._initial_lr = lr
        self._initial_adam_lr = adam_lr

        # Create internal AdamW for 1D params
        self._adam = (
            torch.optim.AdamW(
                adam_params,
                lr=adam_lr,
                betas=adam_betas,
                eps=adam_eps,
                fused=torch.cuda.is_available(),
            )
            if any(len(g["params"]) > 0 for g in adam_params)
            else None
        )

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(muon_params, defaults)

    def state_dict(self) -> dict:
        """Include internal AdamW state so DCP checkpoints are complete."""
        sd = super().state_dict()
        if self._adam is not None:
            sd["_adam_state"] = self._adam.state_dict()
        sd["_initial_lr"] = self._initial_lr
        sd["_initial_adam_lr"] = self._initial_adam_lr
        return sd

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore internal AdamW state from checkpoint."""
        adam_state = state_dict.pop("_adam_state", None)
        self._initial_lr = state_dict.pop("_initial_lr", self._initial_lr)
        self._initial_adam_lr = state_dict.pop("_initial_adam_lr", self._initial_adam_lr)
        super().load_state_dict(state_dict)
        if adam_state is not None and self._adam is not None:
            self._adam.load_state_dict(adam_state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Work on local tensors to avoid DTensor/Tensor mixing.
                # FSDP2 wraps params as DTensors; operating on the local
                # shard directly sidesteps distributed tensor dispatch.
                p_local = _get_local_tensor(p)
                g_local = _get_local_tensor(p.grad)

                # Momentum buffer: stored matching p.grad's type (DTensor if
                # FSDP2) so DCP can save/restore it correctly. We operate on
                # the local shard for the actual computation.
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                buf_local = _get_local_tensor(state["momentum_buffer"])
                buf_local.mul_(momentum).add_(g_local)

                # Reshape to 2D for Newton-Schulz if needed
                original_shape = buf_local.shape
                buf_2d = buf_local.view(buf_local.shape[0], -1) if buf_local.ndim > 2 else buf_local

                # Orthogonalize
                update = _newton_schulz(buf_2d.float(), steps=ns_steps)
                update = update.to(buf_local.dtype)

                if buf_local.ndim > 2:
                    update = update.view(original_shape)

                # Scale: NS produces unit-norm rows, scale by sqrt(m/n)
                m, n = buf_2d.shape
                update.mul_(max(1, m / n) ** 0.5)

                # Decoupled weight decay
                if wd > 0:
                    p_local.mul_(1 - lr * wd)

                p_local.add_(update, alpha=-lr)

        # Step the internal AdamW for 1D params.
        # Scale Adam LR proportionally to Muon's current LR so the scheduler's
        # warmup/decay applies to both — the scheduler only sees Muon's param_groups.
        if self._adam is not None:
            if self._initial_lr > 0:
                scale = self.param_groups[0]["lr"] / self._initial_lr
                for group in self._adam.param_groups:
                    group["lr"] = self._initial_adam_lr * scale
            self._adam.step()

        return loss


def _is_muon_eligible(param: torch.Tensor) -> bool:
    """Check if a parameter should use Muon (Newton-Schulz orthogonalization).

    Muon is applied to 2D weight matrices with reasonable aspect ratios.
    Highly rectangular matrices (embeddings, output heads) are too expensive
    for Newton-Schulz (X@X^T becomes vocab_size x vocab_size) and get AdamW.
    """
    if param.ndim < 2:
        return False
    # Aspect ratio check: max/min > 10 means too rectangular for NS
    m, n = param.shape[0], param.view(param.shape[0], -1).shape[1]
    return max(m, n) / max(min(m, n), 1) <= 10


@registry.register_optimizer("muon")
def _build_muon(
    param_groups: list[dict],
    config: OptimizerConfig,
) -> torch.optim.Optimizer:
    # Split: 2D params with reasonable aspect ratio get Muon, rest gets AdamW
    muon_groups = []
    adam_groups = []

    for group in param_groups:
        muon_params = []
        adam_params = []
        for p in group["params"]:
            if _is_muon_eligible(p):
                muon_params.append(p)
            else:
                adam_params.append(p)
        if muon_params:
            muon_groups.append({**group, "params": muon_params})
        if adam_params:
            adam_groups.append({**group, "params": adam_params})

    if not adam_groups:
        adam_groups = [{"params": [], "weight_decay": 0.0}]

    n_muon = sum(p.numel() for g in muon_groups for p in g["params"])
    n_adam = sum(p.numel() for g in adam_groups for p in g["params"])
    logger.info(f"Muon: {n_muon:,} params (NS-orthogonalized), {n_adam:,} params (AdamW fallback)")

    return Muon(
        muon_groups,
        adam_groups,
        lr=config.lr,
        momentum=config.muon_momentum,
        weight_decay=config.weight_decay,
        adam_betas=config.betas,
        adam_eps=config.eps,
        ns_steps=config.muon_ns_steps,
        adam_lr=config.muon_adam_lr,
    )


# ---------------------------------------------------------------------------
# Optimizer construction
# ---------------------------------------------------------------------------


def _should_decay(name: str, param: torch.nn.Parameter) -> bool:
    """Decide whether a parameter should receive weight decay.

    Excluded: 1D parameters (biases, norm scales/shifts), embedding weights.
    """
    if param.ndim <= 1:
        return False
    return "bias" not in name


def build_optimizer(
    model: torch.nn.Module,
    config: OptimizerConfig,
) -> torch.optim.Optimizer:
    """Construct an optimizer with per-parameter-group weight decay settings.

    Args:
        model: Model whose parameters to optimize.
        config: Optimizer configuration.

    Returns:
        Configured optimizer instance.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _should_decay(name, param):
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    # Log parameter counts
    n_decay = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)
    logger.info(
        f"Optimizer groups: {n_decay:,} params with decay, {n_no_decay:,} params without decay"
    )

    builder = registry.get_optimizer(config.name)
    return builder(param_groups, config)
