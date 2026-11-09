"""Optimizer construction for KempnerForge.

Builds optimizers with per-parameter-group settings:
  - AdamW: standard Adam with decoupled weight decay
  - Muon: momentum with orthogonalized updates via Newton-Schulz iteration.
    Applies Muon to hidden nn.Linear weights, AdamW to the rest (embeddings,
    output head, norms, biases).
  - Lion: sign-based momentum update (half the optimizer memory of AdamW)
  - Schedule-Free AdamW: eliminates LR schedule by averaging iterates

All optimizers:
  - Weight decay follows optimizer.weight_decay_scope: by default every parameter
    with ndim > 1 and no "bias" in its name is decayed, and "all" decays every
    parameter
  - Fused kernel when available (PyTorch 2.x, AdamW only)
"""

from __future__ import annotations

import contextlib
import functools
import logging
from collections.abc import Iterator
from typing import Any, ClassVar

import torch
from torch.distributed._composable.fsdp import FSDPModule

from kempnerforge.config.optimizer import MUON_ADJUST_LR_MODES, MUON_ORTHOGONALIZE_MODES
from kempnerforge.config.registry import registry
from kempnerforge.config.schema import OptimizerConfig
from kempnerforge.model.embedding import OutputHead
from kempnerforge.model.moe import MoEMLP
from kempnerforge.model.moma import ExpertChoiceMoE
from kempnerforge.training.orthogonalize import orthogonalize_exact

logger = logging.getLogger(__name__)


@registry.register_optimizer("adamw")
def _build_adamw(
    param_groups: list[dict],
    config: OptimizerConfig,
    model: torch.nn.Module,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        param_groups,
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        fused=config.fused and torch.cuda.is_available(),
    )


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

                update = (m.mul(beta1) + grad.mul(1 - beta1)).sign_()

                if wd > 0:
                    p.data.mul_(1 - lr * wd)

                p.data.add_(update, alpha=-lr)

                m.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


@registry.register_optimizer("lion")
def _build_lion(
    param_groups: list[dict],
    config: OptimizerConfig,
    model: torch.nn.Module,
) -> torch.optim.Optimizer:
    return Lion(
        param_groups,
        lr=config.lr,
        betas=config.betas,
    )


class ScheduleFreeAdamW(torch.optim.Optimizer):
    """Schedule-Free AdamW (Defazio & Mishchenko, 2024).

    Eliminates the need for an LR schedule by maintaining an iterate ``z``
    and a running average ``x``.  Parameters are set to the interpolated
    point ``y = (1 - beta1) * z + beta1 * x`` for gradient computation.

    Each parameter's ``step`` counts the steps it had a gradient in, and its warmup,
    second-moment bias correction and average all run on that count, so a parameter
    whose first gradient comes late starts as a fresh one does. The count is optimizer
    state, which ``get_optimizer_state_dict`` keeps, so a checkpoint resumes it.

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

    def load_state_dict(self, state_dict: dict) -> None:
        """Load ``state_dict``, also one saved before each parameter kept its own ``step``.

        Such a state carries one optimizer-wide count as ``_k``, which every parameter with
        state resumes from, and its warmup as ``_warmup_steps``.
        """
        state_dict = dict(state_dict)
        legacy_k = state_dict.pop("_k", None)
        self.warmup_steps = state_dict.pop("_warmup_steps", self.warmup_steps)
        super().load_state_dict(state_dict)
        if legacy_k is not None:
            for state in self.state.values():
                if state and "step" not in state:
                    state["step"] = torch.tensor(float(legacy_k))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)
                    state["z"] = p.data.clone()
                    state["v"] = torch.zeros_like(p.data)
                    state["weight_sum"] = 0.0

                state["step"] += 1
                k = state["step"].item()
                warmup = min(1.0, k / self.warmup_steps) if self.warmup_steps > 0 else 1.0
                lr = group["lr"] * warmup
                weight = lr * (1 - beta1)

                z = state["z"]
                v = state["v"]

                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bc2 = 1 - beta2**k
                denom = (v / bc2).sqrt().add_(eps)

                z.addcdiv_(grad, denom, value=-lr)

                if wd > 0:
                    z.mul_(1 - lr * wd)

                state["weight_sum"] += weight
                ck = weight / state["weight_sum"] if state["weight_sum"] > 0 else 0.0

                if "x" not in state:
                    state["x"] = z.clone()
                else:
                    state["x"].lerp_(z, ck)

                p.data.copy_(z).lerp_(state["x"], beta1)

        return loss

    def eval_params(self) -> None:
        """Set parameters to the evaluation point (running average).

        Call before validation/inference for best results.
        Call :meth:`train_params` afterward to resume training.
        """
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p)
                if state and "x" in state:
                    p.data.copy_(state["x"])

    def train_params(self) -> None:
        """Restore parameters to the training point (interpolated y), bit for bit.

        Call after :meth:`eval_params` to resume training. A parameter without state
        stays as it is, and no state is created for it.
        """
        for group in self.param_groups:
            beta1 = group["betas"][0]
            for p in group["params"]:
                state = self.state.get(p)
                if state and "z" in state and "x" in state:
                    p.data.copy_(state["z"]).lerp_(state["x"], beta1)


def _reshard(model: torch.nn.Module | None) -> None:
    """Reshard every FSDP2 unit in ``model``, so its next forward gathers the shards anew."""
    if model is None:
        return
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()


@contextlib.contextmanager
def evaluation_point(
    optimizer: torch.optim.Optimizer, model: torch.nn.Module | None = None
) -> Iterator[None]:
    """Hold the parameters at ``optimizer``'s evaluation point inside the block.

    An optimizer that evaluates at another point than it trains at, as
    :class:`ScheduleFreeAdamW` does, has ``eval_params`` and ``train_params``: the block
    runs between them, and the parameters leave it at the training point again, bit for
    bit, also when the swap or the block raises. Any other optimizer leaves them as they
    are.

    The swap writes the sharded parameters. An FSDP2 unit that does not reshard after its
    forward (``reshard_after_forward`` False or an int, or the root under the default
    None) stays gathered after a forward in the block, holding ``x``, and the next
    training forward would reuse it. So pass ``model`` whenever the block runs a forward:
    every FSDP2 unit in it is resharded before the swap in and after the swap back, and
    each forward gathers the point it should see. Resharding a resharded unit does nothing.
    """
    to_eval = getattr(optimizer, "eval_params", None)
    to_train = getattr(optimizer, "train_params", None)
    if to_eval is None or to_train is None:
        yield
        return
    _reshard(model)
    try:
        to_eval()
        yield
    finally:
        to_train()
        _reshard(model)


@registry.register_optimizer("schedule_free_adamw")
def _build_schedule_free_adamw(
    param_groups: list[dict],
    config: OptimizerConfig,
    model: torch.nn.Module,
) -> torch.optim.Optimizer:
    return ScheduleFreeAdamW(
        param_groups,
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        warmup_steps=config.schedule_free_warmup_steps,
    )


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
    A tall G is iterated as G^T, so ``A`` is min(rows, cols) square.
    """
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315

    X = G / (G.norm() + 1e-7)
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    return X.T if tall else X


def _orthogonalize_matrix(matrix: torch.Tensor, steps: int) -> torch.Tensor:
    """Newton-Schulz in fp32, returned in ``matrix``'s dtype."""
    return _newton_schulz(matrix.float(), steps=steps).to(matrix.dtype)


def _muon_update_scale(adjust_lr: str, rows: int, cols: int) -> float:
    """Factor on Muon's orthogonalized update for a ``rows`` x ``cols`` matrix."""
    if adjust_lr == "match_rms_adamw":
        return 0.2 * max(rows, cols) ** 0.5
    if adjust_lr == "spectral_unclamped":
        return (rows / cols) ** 0.5
    return max(1, rows / cols) ** 0.5


def _get_local_tensor(t: torch.Tensor) -> torch.Tensor:
    """Extract the local (non-DTensor) tensor from a possibly-sharded parameter.

    FSDP2 wraps parameters as DTensors. The optimizer needs to operate on
    the underlying local shard directly to avoid DTensor/Tensor mixing errors
    (e.g. 'aten.add_.Tensor got mixed torch.Tensor and DTensor').
    """
    try:
        from torch.distributed._tensor import DTensor

        if isinstance(t, DTensor):
            return t.to_local()
    except ImportError:
        pass
    return t


class Muon(torch.optim.Optimizer):
    """Muon optimizer: Momentum with Orthogonalized Updates.

    For 2D+ weight matrices: maintains momentum, then orthogonalizes the
    update direction via Newton-Schulz iteration. This keeps update
    directions independent of parameter scale.

    For the rest (embeddings, the output head, norms, biases): uses
    standard AdamW.

    FSDP2 note: Newton-Schulz operates on each rank's local shard
    independently — an approximation, not mathematically equivalent to
    orthogonalizing the full weight matrix. This is the standard approach
    for distributed Muon and works well in practice.

    Args:
        muon_params: Parameter groups for Muon (the hidden ``nn.Linear`` weights).
        adam_params: Parameter groups for AdamW fallback (every other parameter).
        lr: Learning rate for Muon.
        momentum: Momentum coefficient (default 0.95).
        weight_decay: Decoupled weight decay.
        adam_betas: Betas for the AdamW fallback.
        adam_eps: Epsilon for the AdamW fallback.
        ns_steps: Newton-Schulz iteration steps (default 5).
        adam_lr: Learning rate for AdamW fallback. None = same as lr.
        nesterov: Feed Newton-Schulz the Nesterov look-ahead instead of the momentum buffer.
        adjust_lr: Update scale: "original" sqrt(max(1, rows/cols)), "match_rms_adamw"
            0.2 * sqrt(max(rows, cols)), or "spectral_unclamped" sqrt(rows/cols).
        orthogonalize: "local" runs Newton-Schulz on each rank's shard, "exact" on the full
            matrix.
    """

    LEGACY_GROUP_DEFAULTS: ClassVar[dict[str, Any]] = {
        "orthogonalize": "local",
        "nesterov": False,
        "adjust_lr": "original",
    }

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
        nesterov: bool = False,
        adjust_lr: str = "original",
        orthogonalize: str = "local",
    ):
        if adjust_lr not in MUON_ADJUST_LR_MODES:
            raise ValueError(f"adjust_lr must be one of {MUON_ADJUST_LR_MODES}, got {adjust_lr!r}")
        if orthogonalize not in MUON_ORTHOGONALIZE_MODES:
            raise ValueError(
                f"orthogonalize must be one of {MUON_ORTHOGONALIZE_MODES}, got {orthogonalize!r}"
            )
        adam_lr = adam_lr if adam_lr is not None else lr
        self._initial_lr = lr
        self._initial_adam_lr = adam_lr

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

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            nesterov=nesterov,
            adjust_lr=adjust_lr,
            orthogonalize=orthogonalize,
        )
        super().__init__(muon_params, defaults)

    @property
    def inner_optimizers(self) -> tuple[torch.optim.Optimizer, ...]:
        """Optimizers holding parameters that Muon's own ``param_groups`` do not list."""
        return (self._adam,) if self._adam is not None else ()

    def state_dict(self) -> dict:
        """Include internal AdamW state in the plain ``state_dict()``."""
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

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Fill in the ``LEGACY_GROUP_DEFAULTS`` that a state saved before them lacks."""
        super().__setstate__(state)
        for group in self.param_groups:
            for key, value in self.LEGACY_GROUP_DEFAULTS.items():
                group.setdefault(key, value)

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero both Muon's own parameters and the inner AdamW's.

        Only the Muon half is registered with the base class, so the inherited
        ``zero_grad`` would leave every delegated parameter accumulating.
        """
        super().zero_grad(set_to_none=set_to_none)
        if self._adam is not None:
            self._adam.zero_grad(set_to_none=set_to_none)

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
            nesterov = group["nesterov"]
            adjust_lr = group["adjust_lr"]
            orthogonalize = group["orthogonalize"]

            params = [p for p in group["params"] if p.grad is not None]
            inputs = []
            for p in params:
                g_local = _get_local_tensor(p.grad)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                buf_local = _get_local_tensor(state["momentum_buffer"])
                buf_local.mul_(momentum).add_(g_local)
                inputs.append(g_local.add(buf_local, alpha=momentum) if nesterov else buf_local)

            matrices = [x.view(x.shape[0], -1) for x in inputs]
            newton_schulz = functools.partial(_orthogonalize_matrix, steps=ns_steps)
            if orthogonalize == "exact":
                orthos = orthogonalize_exact(params, matrices, newton_schulz)
            else:
                orthos = [newton_schulz(matrix) for matrix in matrices]

            for p, x, ortho in zip(params, inputs, orthos, strict=True):
                p_local = _get_local_tensor(p)
                update = ortho.view(x.shape)

                if orthogonalize == "exact":
                    m, n = p.shape[0], p.shape[1:].numel()
                else:
                    m, n = ortho.shape
                update.mul_(_muon_update_scale(adjust_lr, m, n))

                if wd > 0:
                    p_local.mul_(1 - lr * wd)

                p_local.add_(update, alpha=-lr)

        if self._adam is not None:
            if self._initial_lr > 0:
                scale = self.param_groups[0]["lr"] / self._initial_lr
                for group in self._adam.param_groups:
                    group["lr"] = self._initial_adam_lr * scale
            self._adam.step()

        return loss


def _muon_matrices(model: torch.nn.Module, *, routers: bool) -> set[torch.Tensor]:
    """Return the ``nn.Linear`` weights Muon orthogonalizes.

    The output head's go to AdamW, and so do the MoE routers' unless ``routers``.
    """
    adamw: set[torch.Tensor] = set()
    for module in model.modules():
        if isinstance(module, OutputHead):
            adamw.update(module.parameters())
        elif not routers and isinstance(module, MoEMLP | ExpertChoiceMoE):
            adamw.update(module.router.parameters())
    return {
        module.weight
        for module in model.modules()
        if isinstance(module, torch.nn.Linear) and module.weight not in adamw
    }


@registry.register_optimizer("muon")
def _build_muon(
    param_groups: list[dict],
    config: OptimizerConfig,
    model: torch.nn.Module,
) -> torch.optim.Optimizer:
    on_muon = _muon_matrices(model, routers=config.muon_router)
    muon_groups = []
    adam_groups = []

    for group in param_groups:
        muon_params = []
        adam_params = []
        for p in group["params"]:
            if p in on_muon:
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
        nesterov=config.muon_nesterov,
        adjust_lr=config.muon_adjust_lr,
        orthogonalize=config.muon_orthogonalize,
    )


def _should_decay(name: str, param: torch.nn.Parameter, scope: str) -> bool:
    """Decide whether a parameter should receive weight decay.

    Args:
        name: Fully qualified parameter name.
        param: The parameter itself.
        scope: ``"exclude_norms_and_biases"`` keeps decay off every parameter of
            rank 1 or 0 (norm scales and shifts, and scalars) and any parameter
            whose name contains ``"bias"``. Everything else is decayed, which
            includes the token embedding, the output head and any parameter of
            higher rank such as packed expert weights or a convolution kernel.
            ``"all"`` decays every parameter, which is what ``torch.optim.AdamW``
            does by default.

    Returns:
        True when the parameter belongs in the weight-decay group.
    """
    if scope == "all":
        return True
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
        if _should_decay(name, param, config.weight_decay_scope):
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    n_decay = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)
    logger.info(
        f"Optimizer groups: {n_decay:,} params with decay, {n_no_decay:,} params without decay"
    )

    builder = registry.get_optimizer(config.name)
    return builder(param_groups, config, model)
