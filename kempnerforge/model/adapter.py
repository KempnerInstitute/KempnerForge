"""Vision-to-LLM adapter modules (the "connector").

The adapter projects vision features (shape ``(B, num_tokens, feature_dim)``)
into the LLM embedding space (shape ``(B, out_tokens, model.dim)``). It sits
between the vision encoder and the transformer in ``VLMWrapper``.

Two families:

- **Projection adapters** keep the token count (``out_tokens == num_tokens``):
  ``mlp_2layer`` (default, the canonical LLaVA-family 2-layer MLP) and
  ``linear`` (single ``nn.Linear``, an ablation baseline).
- **Pooling adapters** reduce the token count by pooling the square patch grid
  before projecting: ``avgpool`` (window-average, the cheapest reducer) and
  ``attentional_pool`` (Molmo2-style per-window multi-head attention with the
  window mean as query). Pooling is what makes many-frame video fit the
  sequence budget: a 27×27 SigLIP grid (729 tokens) pools to 81 tokens at a
  3×3 window.

Every adapter is a ``VisionAdapter`` exposing ``output_num_tokens(n_in)`` so the
build path can size the residual stream (and MoT's positional split) without a
dry-run forward. Adapters register themselves under the ``adapter`` registry
category.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from kempnerforge.config.registry import registry

_ADAPTER_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "relu": nn.ReLU,
}

# Registry keys of adapters that pool the patch grid (reduce token count). The
# config layer (``AdapterConfig.output_num_tokens``) consults this to predict
# the post-adapter token count without building the module. Keep in sync with
# the registered pooling builders below.
POOLING_ADAPTER_TYPES: tuple[str, ...] = ("avgpool", "attentional_pool")

# Pooling adapters whose ``forward`` cannot pool ragged edge windows (so their
# token count must reject a non-divisible grid at config/build time). Both
# ``avgpool`` and ``attentional_pool`` now mask partial edge windows, so this is
# empty -- kept as a seam for a future connector that genuinely needs divisibility.
DIVISIBLE_ONLY_POOL_TYPES: tuple[str, ...] = ()


def pooled_token_count(
    num_input_tokens: int, window: int, *, require_divisible: bool = False
) -> int:
    """Token count out of a ``window×window`` pool over a square patch grid.

    A vision encoder emits ``num_input_tokens`` patch tokens laid out on a
    square ``grid × grid`` map (``grid = sqrt(num_input_tokens)``). Pooling with
    a ``window × window`` kernel and ceil edges yields ``ceil(grid/window) ** 2``
    tokens; edge windows that do not fill the kernel pool only the patches they
    cover (Molmo2 §A: "the bottom and far-right image patches are pooled with a
    reduced number of patches").

    Connectors that genuinely cannot pool ragged edges may pass
    ``require_divisible=True`` to raise when ``grid`` is not divisible by
    ``window``, rejecting a ragged config at config/build time rather than
    deterministically failing in ``forward`` at the first step. (Today both
    pooling connectors handle ragged edges, so none set it.)

    This is the single source of truth for the post-pool count: it must equal
    the pooling adapters' actual ``forward`` output length, because the build
    path uses it to size MoT's positional split.
    """
    if window <= 0:
        raise ValueError(f"pool window must be positive (got {window})")
    if num_input_tokens <= 0:
        raise ValueError(f"num_input_tokens must be positive (got {num_input_tokens})")
    grid = _grid_side(num_input_tokens)
    if require_divisible and grid % window != 0:
        raise ValueError(
            f"this pooling connector requires the patch grid ({grid}x{grid}) be "
            f"divisible by the pool window ({window}); got a ragged grid "
            f"(num_tokens={num_input_tokens}). Use a ragged-capable connector "
            "(avgpool or attentional_pool), or pick a divisible window."
        )
    per_side = math.ceil(grid / window)
    return per_side * per_side


def _grid_side(num_tokens: int) -> int:
    """Side length of the square patch grid, or raise if not a perfect square."""
    grid = math.isqrt(num_tokens)
    if grid * grid != num_tokens:
        raise ValueError(
            f"pooling requires a square patch grid, but num_tokens={num_tokens} is "
            "not a perfect square. Use a vision encoder that strips any CLS token so "
            "the patch tokens form a square grid."
        )
    return grid


class VisionAdapter(nn.Module):
    """Base class for vision→LLM adapters (the connector).

    Contract: ``forward`` maps ``(B, N, in_dim) -> (B, M, out_dim)`` where
    ``M == output_num_tokens(N)``. Projection adapters keep ``M == N``; pooling
    adapters reduce it. ``output_num_tokens`` lets the build path size the
    residual stream and MoT's positional split without a dry-run forward, and
    must agree exactly with the forward output length.
    """

    def output_num_tokens(self, num_input_tokens: int) -> int:
        """Tokens emitted per image given ``num_input_tokens`` patch tokens in.

        Identity by default (projection adapters); pooling adapters override.
        """
        return num_input_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class MLP2LayerAdapter(VisionAdapter):
    """2-layer MLP from image-feature dim to LLM embedding dim.

    Architecture: ``Linear(in_dim, hidden) -> activation -> Linear(hidden, out_dim)``.
    ``hidden_dim=None`` defaults to ``out_dim``. Keeps the token count.

    ``reset_parameters`` is provided so callers that materialize adapters
    from meta can re-initialize weights with the standard Linear defaults.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int | None = None,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError("MLP2LayerAdapter in_dim and out_dim must be positive")
        if activation not in _ADAPTER_ACTIVATIONS:
            raise ValueError(
                f"Unknown adapter activation: {activation!r}. Options: {list(_ADAPTER_ACTIVATIONS)}"
            )
        hidden = hidden_dim if hidden_dim and hidden_dim > 0 else out_dim
        self.proj1 = nn.Linear(in_dim, hidden, bias=True)
        self.act = _ADAPTER_ACTIVATIONS[activation]()
        self.proj2 = nn.Linear(hidden, out_dim, bias=True)

    def reset_parameters(self) -> None:
        """Re-run ``nn.Linear`` default init on both projections.

        Used after ``to_empty(device=...)`` on a meta-device build.
        """
        self.proj1.reset_parameters()
        self.proj2.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj2(self.act(self.proj1(x)))


class LinearAdapter(VisionAdapter):
    """Single ``nn.Linear`` from image-feature dim to LLM embedding dim.

    No activation, no hidden layer. Keeps the token count. Useful as an
    ablation baseline against ``MLP2LayerAdapter``.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError("LinearAdapter in_dim and out_dim must be positive")
        self.proj = nn.Linear(in_dim, out_dim, bias=True)

    def reset_parameters(self) -> None:
        self.proj.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class AvgPoolAdapter(VisionAdapter):
    """Average-pool a square patch grid by a window, then project.

    ``(B, N, in_dim)`` patch tokens (``N == grid**2``) are averaged over
    ``window × window`` spatial windows (ceil edges; partial edge windows
    average only the real patches they cover), giving ``(B, M, in_dim)`` with
    ``M == ceil(grid/window)**2``, then a ``Linear`` maps ``in_dim -> out_dim``.

    The cheapest token-count reducer (LLaVA-NeXT / sibling-repo style). ``window``
    is overridable per ``forward`` call so one connector can pool images (e.g.
    2×2) and video frames (3×3) with the same projection weights.
    """

    def __init__(self, in_dim: int, out_dim: int, pool_window: int = 2) -> None:
        super().__init__()
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError("AvgPoolAdapter in_dim and out_dim must be positive")
        if pool_window <= 0:
            raise ValueError(f"AvgPoolAdapter pool_window must be positive (got {pool_window})")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pool_window = pool_window
        self.proj = nn.Linear(in_dim, out_dim, bias=True)

    def reset_parameters(self) -> None:
        self.proj.reset_parameters()

    def output_num_tokens(self, num_input_tokens: int) -> int:
        return pooled_token_count(num_input_tokens, self.pool_window)

    def forward(self, x: torch.Tensor, pool_window: int | None = None) -> torch.Tensor:
        w = pool_window if pool_window is not None else self.pool_window
        if w <= 0:
            raise ValueError(f"pool_window must be positive (got {w})")
        b, n, c = x.shape
        grid = _grid_side(n)
        per = math.ceil(grid / w)
        padded = per * w
        x = x.view(b, grid, grid, c)
        if padded != grid:
            pad = padded - grid
            # F.pad pads from the last dim backward: (C:0,0)(W:0,pad)(H:0,pad).
            x = F.pad(x, (0, 0, 0, pad, 0, pad))
            mask = torch.ones(b, grid, grid, 1, dtype=x.dtype, device=x.device)
            mask = F.pad(mask, (0, 0, 0, pad, 0, pad))
        else:
            mask = torch.ones(b, padded, padded, 1, dtype=x.dtype, device=x.device)
        # Group into windows and average over real (unpadded) cells only.
        sums = x.view(b, per, w, per, w, c).sum(dim=(2, 4))  # (B, per, per, C)
        counts = mask.view(b, per, w, per, w, 1).sum(dim=(2, 4)).clamp_(min=1)  # (B, per, per, 1)
        pooled = (sums / counts).reshape(b, per * per, c)
        return self.proj(pooled)


class AttentionalPoolAdapter(VisionAdapter):
    """Attentional pooling connector (Molmo2 §3.1).

    For each ``window × window`` patch window, a multi-head attention layer
    pools the window's patches into one vector, using the **mean of the window's
    patches as the query** and the patches themselves as keys/values; the result
    is projected ``in_dim -> out_dim``. Output length is ``ceil(grid/window)**2``.

    ``window`` is overridable per ``forward`` call (shared params across image
    2×2 and video 3×3 pooling, per the paper). Ragged grids are supported: a
    partial edge window pools only its real patches (the padded patches are
    masked out of the window's K/V), matching ``avgpool`` and Molmo2 §A.
    """

    def __init__(
        self, in_dim: int, out_dim: int, pool_window: int = 2, pool_heads: int = 16
    ) -> None:
        super().__init__()
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError("AttentionalPoolAdapter in_dim and out_dim must be positive")
        if pool_window <= 0:
            raise ValueError(
                f"AttentionalPoolAdapter pool_window must be positive (got {pool_window})"
            )
        if pool_heads <= 0:
            raise ValueError(
                f"AttentionalPoolAdapter pool_heads must be positive (got {pool_heads})"
            )
        if in_dim % pool_heads != 0:
            raise ValueError(
                f"AttentionalPoolAdapter in_dim ({in_dim}) must be divisible by "
                f"pool_heads ({pool_heads})"
            )
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pool_window = pool_window
        self.pool_heads = pool_heads
        self.head_dim = in_dim // pool_heads
        self.q_proj = nn.Linear(in_dim, in_dim, bias=True)
        self.k_proj = nn.Linear(in_dim, in_dim, bias=True)
        self.v_proj = nn.Linear(in_dim, in_dim, bias=True)
        self.o_proj = nn.Linear(in_dim, in_dim, bias=True)
        self.out_proj = nn.Linear(in_dim, out_dim, bias=True)

    def reset_parameters(self) -> None:
        for layer in (self.q_proj, self.k_proj, self.v_proj, self.o_proj, self.out_proj):
            layer.reset_parameters()

    def output_num_tokens(self, num_input_tokens: int) -> int:
        return pooled_token_count(num_input_tokens, self.pool_window)

    def forward(self, x: torch.Tensor, pool_window: int | None = None) -> torch.Tensor:
        w = pool_window if pool_window is not None else self.pool_window
        if w <= 0:
            raise ValueError(f"pool_window must be positive (got {w})")
        b, n, c = x.shape
        grid = _grid_side(n)
        per = math.ceil(grid / w)
        padded = per * w
        k_win = w * w
        x = x.view(b, grid, grid, c)
        # Ragged grid (grid not divisible by w): pad the bottom/right edges to
        # per*w and mask the padded patches out of each edge window's attention,
        # so a window pools only its real patches (Molmo2 §A). Divisible grids
        # skip this and stay bit-identical to the unmasked pooling.
        win_mask: torch.Tensor | None = None
        if padded != grid:
            pad = padded - grid
            valid = torch.ones(b, grid, grid, 1, dtype=torch.bool, device=x.device)
            # F.pad pads the last dim backward: (C:0,0)(W:0,pad)(H:0,pad).
            x = F.pad(x, (0, 0, 0, pad, 0, pad))
            valid = F.pad(valid, (0, 0, 0, pad, 0, pad))  # (B, padded, padded, 1) bool
            # Window order must match `windows` below.
            win_mask = (
                valid.view(b, per, w, per, w, 1)
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(b * per * per, k_win)
            )
        # (B, padded, padded, C) -> windows (B*per*per, w*w, C): each window's patches contiguous.
        windows = (
            x.view(b, per, w, per, w, c).permute(0, 1, 3, 2, 4, 5).reshape(b * per * per, k_win, c)
        )
        m = windows.shape[0]
        if win_mask is None:
            query = windows.mean(dim=1, keepdim=True)  # (M, 1, C) — window mean as query
        else:
            # Query = mean over real patches only (every window has >=1 real patch,
            # so the count is never zero).
            wf = win_mask.unsqueeze(-1).to(windows.dtype)  # (M, k_win, 1)
            query = (windows * wf).sum(dim=1, keepdim=True) / wf.sum(dim=1, keepdim=True).clamp_(
                min=1
            )
        q = self.q_proj(query).view(m, 1, self.pool_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(windows).view(m, k_win, self.pool_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(windows).view(m, k_win, self.pool_heads, self.head_dim).transpose(1, 2)
        # Mask padded patches out of the K/V for edge windows; None -> plain SDPA
        # (the divisible path, bit-identical to before).
        attn_mask = None if win_mask is None else win_mask.view(m, 1, 1, k_win)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  # (M, H, 1, head_dim)
        attn = attn.transpose(1, 2).reshape(m, c)  # (M, C)
        pooled = self.o_proj(attn).view(b, per * per, c)
        return self.out_proj(pooled)


@registry.register_adapter("mlp_2layer")
def _build_mlp_2layer(
    in_dim: int,
    out_dim: int,
    hidden_dim: int | None = None,
    activation: str = "gelu",
    **_: Any,
) -> VisionAdapter:
    return MLP2LayerAdapter(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_dim=hidden_dim,
        activation=activation,
    )


@registry.register_adapter("linear")
def _build_linear(
    in_dim: int,
    out_dim: int,
    **_: Any,
) -> VisionAdapter:
    return LinearAdapter(in_dim=in_dim, out_dim=out_dim)


@registry.register_adapter("avgpool")
def _build_avgpool(
    in_dim: int,
    out_dim: int,
    pool_window: int = 2,
    **_: Any,
) -> VisionAdapter:
    return AvgPoolAdapter(in_dim=in_dim, out_dim=out_dim, pool_window=pool_window)


@registry.register_adapter("attentional_pool")
def _build_attentional_pool(
    in_dim: int,
    out_dim: int,
    pool_window: int = 2,
    pool_heads: int = 16,
    **_: Any,
) -> VisionAdapter:
    return AttentionalPoolAdapter(
        in_dim=in_dim, out_dim=out_dim, pool_window=pool_window, pool_heads=pool_heads
    )


def build_adapter(adapter_config, in_dim: int, out_dim: int) -> VisionAdapter:
    """Dispatch to the registered adapter builder.

    Args:
        adapter_config: ``AdapterConfig`` (or compatible object exposing
            ``type`` and ``extra_kwargs()``).
        in_dim: Source feature dim (the vision encoder's ``feature_dim``).
        out_dim: Target embedding dim (the transformer's ``dim``).

    Returns:
        A ``VisionAdapter`` with signature ``(B, N, in_dim) -> (B, M, out_dim)``,
        where ``M == adapter.output_num_tokens(N)``.
    """
    builder = registry.get_adapter(adapter_config.type)
    return builder(in_dim=in_dim, out_dim=out_dim, **adapter_config.extra_kwargs())
