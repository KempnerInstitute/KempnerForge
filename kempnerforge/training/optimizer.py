"""Optimizer construction for KempnerForge.

Builds optimizers with per-parameter-group settings:
  - AdamW: standard Adam with decoupled weight decay
  - Muon: momentum with orthogonalized updates via Newton-Schulz iteration.
    Applies Muon to 2D+ weight matrices, AdamW to 1D params (biases, norms).
  - Lion: sign-based momentum update (half the optimizer memory of AdamW)
  - Schedule-Free AdamW: eliminates LR schedule by averaging iterates

All optimizers:
  - Weight decay follows optimizer.weight_decay_scope: by default every parameter
    with ndim > 1 and no "bias" in its name is decayed, and "all" decays every
    parameter
  - optimizer.lr_multipliers gives each listed module its own groups, at lr
    times the module's multiplier
  - Fused kernel when available (PyTorch 2.x, AdamW only)
"""

from __future__ import annotations

import fnmatch
import functools
import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, ClassVar

import torch

from kempnerforge.config.optimizer import MUON_ADJUST_LR_MODES, MUON_ORTHOGONALIZE_MODES
from kempnerforge.config.registry import registry
from kempnerforge.config.schema import OptimizerConfig
from kempnerforge.training.orthogonalize import orthogonalize_exact

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

                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bc2 = 1 - beta2**k
                denom = (v / bc2).sqrt().add_(eps)

                z.addcdiv_(grad, denom, value=-lr)

                if wd > 0:
                    z.mul_(1 - lr * wd)

                state["weight_sum"] += weight
                ck = weight / state["weight_sum"]

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

    X = G / (G.norm() + 1e-7)

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    return X


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

    For 1D parameters (biases, norms, embeddings): uses standard AdamW,
    since orthogonalization requires 2D matrices.

    FSDP2 note: Newton-Schulz operates on each rank's local shard
    independently — an approximation, not mathematically equivalent to
    orthogonalizing the full weight matrix. This is the standard approach
    for distributed Muon and works well in practice.

    The AdamW fallback follows Muon's schedule: each step sets every AdamW
    group's rate to its base rate, its own ``lr`` or else ``adam_lr``, times
    the ratio of Muon's first group's current rate to that group's base rate.

    Args:
        muon_params: Parameter groups for Muon (2D+ weights). A group's own
            ``lr`` overrides ``lr``.
        adam_params: Parameter groups for AdamW fallback (1D params). A
            group's own ``lr`` overrides ``adam_lr``.
        lr: Learning rate for Muon (2D weights).
        momentum: Momentum coefficient (default 0.95).
        weight_decay: Decoupled weight decay.
        adam_betas: Betas for the AdamW fallback.
        adam_eps: Epsilon for the AdamW fallback.
        ns_steps: Newton-Schulz iteration steps (default 5).
        adam_lr: Learning rate for AdamW fallback (1D params). None = same as lr.
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
        self._lead_group_lr: float | None = muon_params[0].get("lr") if muon_params else None
        self._adam_group_lrs: list[float | None] = [group.get("lr") for group in adam_params]

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
            for group, lr in zip(self._adam.param_groups, self._fallback_lrs(), strict=True):
                group["lr"] = lr
            self._adam.step()

        return loss

    def _fallback_lrs(self) -> list[float]:
        """The rate each AdamW fallback group steps at, from Muon's current schedule."""
        assert self._adam is not None
        lead_lr = self._initial_lr if self._lead_group_lr is None else self._lead_group_lr
        if lead_lr <= 0:
            return [group["lr"] for group in self._adam.param_groups]
        scale = self.param_groups[0]["lr"] / lead_lr
        return [
            (self._initial_adam_lr if base_lr is None else base_lr) * scale
            for base_lr in self._adam_group_lrs
        ]


def _is_muon_eligible(param: torch.Tensor) -> bool:
    """Check if a parameter should use Muon (Newton-Schulz orthogonalization).

    Muon is applied to 2D weight matrices with reasonable aspect ratios.
    Highly rectangular matrices (embeddings, output heads) are too expensive
    for Newton-Schulz (X@X^T becomes vocab_size x vocab_size) and get AdamW.
    """
    if param.ndim < 2:
        return False
    m, n = param.shape[0], param.view(param.shape[0], -1).shape[1]
    return max(m, n) / max(min(m, n), 1) <= 10


@registry.register_optimizer("muon")
def _build_muon(
    param_groups: list[dict],
    config: OptimizerConfig,
) -> torch.optim.Optimizer:
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

    adam_lr_ratio = 1.0 if config.muon_adam_lr is None else config.muon_adam_lr / config.lr
    for group in adam_groups:
        if "lr" in group:
            group["lr"] *= adam_lr_ratio

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


def _module_owners(
    model: torch.nn.Module,
    modules: Iterable[str],
    module_patterns: Mapping[str, Sequence[str]] | None,
) -> dict[torch.Tensor, str]:
    """Map every parameter of ``model`` that one of ``modules`` matches to that module.

    Names are matched as the module that ``torch.compile`` wrapped, if any, names them.

    Raises:
        ValueError: ``module_patterns`` is missing or lacks one of ``modules``, one of
            ``modules`` matches no parameter, or a parameter matches two of them.
    """
    modules = sorted(modules)
    if module_patterns is None:
        raise ValueError(
            "optimizer.lr_multipliers names modules, so build_optimizer needs module_patterns"
        )
    unknown = [module for module in modules if module not in module_patterns]
    if unknown:
        raise ValueError(
            f"optimizer.lr_multipliers names unknown modules {unknown}; the modules of "
            f"module_patterns are {sorted(module_patterns)}"
        )
    owners: dict[torch.Tensor, str] = {}
    for name, param in getattr(model, "_orig_mod", model).named_parameters():
        matched = [
            module
            for module in modules
            if any(fnmatch.fnmatch(name, pattern) for pattern in module_patterns[module])
        ]
        if len(matched) > 1:
            raise ValueError(
                f"Parameter {name!r} is in more than one module of optimizer.lr_multipliers: "
                f"{matched}. List modules whose patterns share no parameter."
            )
        if matched:
            owners[param] = matched[0]
    unmatched = sorted(set(modules) - set(owners.values()))
    if unmatched:
        raise ValueError(
            f"optimizer.lr_multipliers names modules {unmatched} whose module_patterns match "
            "no parameter of the model"
        )
    return owners


def _split_by_module(
    model: torch.nn.Module,
    param_groups: list[dict],
    config: OptimizerConfig,
    module_patterns: Mapping[str, Sequence[str]] | None,
) -> list[dict]:
    """Give each module of ``config.lr_multipliers`` its own copy of every group.

    A module's copy holds the module's parameters, sets ``"lr"`` to ``config.lr`` times the
    module's multiplier and names the module under ``"module"``. Parameters outside every
    listed module stay in the original groups. The groups come out in a fixed order, the
    original ones first and then each module's by module name, and empty groups are dropped.
    """
    owners = _module_owners(model, config.lr_multipliers, module_patterns)
    split = [
        {**group, "params": [p for p in group["params"] if p not in owners]}
        for group in param_groups
    ]
    for module in sorted(config.lr_multipliers):
        lr = config.lr * config.lr_multipliers[module]
        groups = [
            {
                **group,
                "params": [p for p in group["params"] if owners.get(p) == module],
                "lr": lr,
                "module": module,
            }
            for group in param_groups
        ]
        n_params = sum(p.numel() for group in groups for p in group["params"])
        logger.info(f"Optimizer module {module}: {n_params:,} trainable params at lr {lr:.3g}")
        split.extend(groups)
    return [group for group in split if group["params"]]


def module_learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    """Return the rate each module ``build_optimizer`` gave its own groups steps at next.

    Keyed by module name. Under Muon, ``<module>`` is the rate of the module's orthogonalized
    matrices and ``<module>/adamw`` that of its AdamW fallback, each present when the module
    has such parameters. Empty when ``optimizer.lr_multipliers`` was empty.
    """
    rates = {group["module"]: group["lr"] for group in optimizer.param_groups if "module" in group}
    if isinstance(optimizer, Muon) and optimizer._adam is not None:
        fallback = zip(optimizer._adam.param_groups, optimizer._fallback_lrs(), strict=True)
        rates.update({f"{g['module']}/adamw": lr for g, lr in fallback if "module" in g})
    return rates


def build_optimizer(
    model: torch.nn.Module,
    config: OptimizerConfig,
    module_patterns: Mapping[str, Sequence[str]] | None = None,
) -> torch.optim.Optimizer:
    """Construct an optimizer with per-parameter-group weight decay settings.

    With ``config.lr_multipliers`` set, each listed module's parameters get their own
    groups at ``config.lr`` times the module's multiplier, named by ``"module"``; the
    scheduler then scales every group's own rate by the same factor.

    Args:
        model: Model whose parameters to optimize.
        config: Optimizer configuration.
        module_patterns: Module name to the fnmatch patterns of its parameter names, such
            as ``VLMConfig.module_patterns``. Required when ``config.lr_multipliers`` is set.

    Returns:
        Configured optimizer instance.

    Raises:
        ValueError: ``config.lr_multipliers`` names a module ``module_patterns`` lacks or
            one that matches no parameter, or two of its modules share a parameter.
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
    if config.lr_multipliers:
        param_groups = _split_by_module(model, param_groups, config, module_patterns)

    n_decay = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)
    logger.info(
        f"Optimizer groups: {n_decay:,} params with decay, {n_no_decay:,} params without decay"
    )

    builder = registry.get_optimizer(config.name)
    return builder(param_groups, config)
