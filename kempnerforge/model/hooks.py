"""Activation extraction hooks for mechanistic interpretability.

Provides tools for capturing intermediate activations, attention patterns,
and hidden states during inference — essential for probing, CKA analysis,
SVCCA, and other interpretability research.

Usage::

    store = ActivationStore(model, layers=["layers.0.attention", "layers.5.mlp"])
    store.enable()
    model(input_ids)
    act = store.get("layers.0.attention")  # (batch, seq, dim) on CPU
    store.disable()
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from torch.utils.data import DataLoader, Dataset


class ActivationStore:
    """Register forward hooks on named modules to capture activations.

    Captured tensors are moved to CPU to avoid GPU memory pressure.
    Use :meth:`enable` / :meth:`disable` to control when hooks are active.

    Under tensor parallelism every capture is stored as the full tensor. An
    output that is a ``DTensor`` (a block, its attention or MLP, a norm) is
    gathered before it is stored. A module holding a parameter sharded over the
    ``tp`` mesh dimension, such as a projection, is hooked ahead of its other
    forward hooks, among them the tensor-parallel one that turns its
    ``DTensor`` output into this rank's local shard, so its output is gathered
    too. Each gather is one collective over the tensor-parallel group per
    captured module and forward, so every rank of that group must enable the
    store and run the same forward. A module that may run on a local shard
    without holding such a parameter, as ``q_norm`` and ``k_norm`` do, is
    refused by :meth:`enable`.

    Args:
        model: The model to instrument.
        layers: List of module names (dot-separated FQNs) to capture.
            Example: ``["layers.0.attention", "layers.5.mlp", "norm"]``
    """

    def __init__(self, model: nn.Module, layers: list[str] | None = None) -> None:
        self._model = model
        self._layers = list(layers or [])
        self._activations: dict[str, torch.Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHook] = []  # type: ignore[reportAttributeAccessIssue]
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def layer_names(self) -> list[str]:
        return list(self._layers)

    @property
    def activations(self) -> dict[str, torch.Tensor]:
        """Return a copy of captured activations."""
        return dict(self._activations)

    def enable(self) -> None:
        """Register forward hooks on all target layers.

        Raises:
            ValueError: If a target is not a module of the model, or if under
                tensor parallelism it may run on this rank's local shard: it
                holds no parameter sharded over the ``tp`` mesh dimension but
                sits inside a module that directly contains one that does, the
                model root included, as ``q_norm`` sits next to ``q_proj``
                inside ``layers.N.attention``. No hook is registered when it
                raises.
        """
        if self._enabled:
            return
        self._remove_hooks()
        module_map = dict(self._model.named_modules())
        for name in self._layers:
            if name not in module_map:
                raise ValueError(
                    f"Module '{name}' not found in model. "
                    f"Available: {sorted(module_map.keys())[:20]}..."
                )
        owners = _tp_shard_owners(module_map)
        _refuse_local_shard_targets(owners, self._layers)
        for name in self._layers:
            hook = module_map[name].register_forward_hook(
                self._make_hook(name), prepend=name in owners
            )
            self._hooks.append(hook)
        self._enabled = True

    def disable(self) -> None:
        """Remove all hooks and mark as disabled."""
        self._remove_hooks()
        self._enabled = False

    def clear(self) -> None:
        """Clear captured activations (keeps hooks registered)."""
        self._activations.clear()

    def get(self, name: str) -> torch.Tensor | None:
        """Get captured activation for a layer, or None if not captured."""
        return self._activations.get(name)

    def _make_hook(self, name: str):
        def hook(_module, _input, output):
            if isinstance(output, torch.Tensor):
                self._activations[name] = _full_on_cpu(output)
            elif isinstance(output, tuple) and len(output) > 0:
                self._activations[name] = _full_on_cpu(output[0])

        return hook

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __del__(self) -> None:
        self._remove_hooks()


def _full_on_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """``tensor`` detached and on CPU, a ``DTensor`` gathered to its full value first."""
    tensor = tensor.detach()
    if isinstance(tensor, DTensor):
        tensor = tensor.full_tensor()
    return tensor.cpu()


def _shards_over_tp(param: torch.Tensor) -> bool:
    """Whether ``param`` is a ``DTensor`` sharded over a mesh dimension named ``tp``."""
    if not isinstance(param, DTensor):
        return False
    names = param.device_mesh.mesh_dim_names or ()
    return "tp" in names and param.placements[names.index("tp")].is_shard()


def _tp_shard_owners(modules: dict[str, nn.Module]) -> list[str]:
    """Names, in ``modules`` order, of the modules that hold a parameter sharded over ``tp``."""
    return [
        name
        for name, module in modules.items()
        if any(_shards_over_tp(p) for p in module.parameters(recurse=False))
    ]


def _refuse_local_shard_targets(owners: list[str], targets: list[str]) -> None:
    """Raise ``ValueError`` naming the first target that may run on a local TP shard.

    ``owners`` names the modules holding a parameter sharded over the ``tp``
    mesh dimension, whose outputs the store gathers. Any other module inside an
    owner's parent, the model root included, may run on the owner's local shard
    with nothing marking it as one, as ``q_norm`` does next to ``q_proj``.
    """
    for target in targets:
        if target in owners:
            continue
        for owner in owners:
            region = owner.rpartition(".")[0]
            inside = target.startswith(f"{region}.") if region else target != ""
            if inside:
                hint = f" Capture '{region}' or a module enclosing it instead." if region else ""
                raise ValueError(
                    f"Cannot capture '{target}': under tensor parallelism it may run on this "
                    f"rank's local shard, with nothing marking it as one, because it sits next "
                    f"to '{owner}', which holds parameters sharded over the 'tp' mesh "
                    f"dimension.{hint}"
                )


def extract_representations(
    model: nn.Module,
    dataset: Dataset,
    layers: list[str],
    device: torch.device,
    batch_size: int = 32,
    max_samples: int | None = None,
) -> dict[str, torch.Tensor]:
    """Run model over dataset and collect activations from specified layers.

    Returns a dict mapping layer names to tensors of shape
    ``(num_samples, seq_len, hidden_dim)`` (or whatever the layer outputs).
    On a tensor-parallel model every rank of the tensor-parallel group must
    call it with the same dataset, as :class:`ActivationStore` requires.

    Args:
        model: Model to extract from (should already be on ``device``).
        dataset: Map-style dataset returning dicts with ``"input_ids"``.
        layers: Module FQNs to capture (e.g. ``["layers.0.attention"]``).
        device: Device to run inference on.
        batch_size: Batch size for extraction.
        max_samples: Stop after this many samples (None = full dataset).

    Returns:
        Dict of ``{layer_name: Tensor}`` with activations on CPU.
    """
    store = ActivationStore(model, layers=layers)
    store.enable()
    was_training = model.training
    model.eval()

    results: dict[str, list[torch.Tensor]] = {name: [] for name in layers}
    n_collected = 0

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    try:
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                model(input_ids)
                for name in layers:
                    act = store.get(name)
                    if act is not None:
                        results[name].append(act)
                store.clear()
                n_collected += input_ids.shape[0]
                if max_samples is not None and n_collected >= max_samples:
                    break
    finally:
        store.disable()
        model.train(was_training)

    return {
        name: torch.cat(tensors, dim=0)[:max_samples]
        if max_samples is not None
        else torch.cat(tensors, dim=0)
        for name, tensors in results.items()
        if tensors
    }


def save_activations(
    activations: dict[str, torch.Tensor],
    path: str | Path,
) -> None:
    """Save activations to a ``.npz`` file.

    Args:
        activations: Dict mapping layer names to tensors (from
            :class:`ActivationStore` or :func:`extract_representations`).
        path: Output file path. ``.npz`` extension added if missing.
    """
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), **{k: v.numpy() for k, v in activations.items()})  # type: ignore[reportArgumentType]
