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
from torch.utils.data import DataLoader, Dataset


class ActivationStore:
    """Register forward hooks on named modules to capture activations.

    Captured tensors are moved to CPU to avoid GPU memory pressure.
    Use :meth:`enable` / :meth:`disable` to control when hooks are active.

    Args:
        model: The model to instrument.
        layers: List of module names (dot-separated FQNs) to capture.
            Example: ``["layers.0.attention", "layers.5.mlp", "norm"]``
    """

    def __init__(self, model: nn.Module, layers: list[str] | None = None) -> None:
        self._model = model
        self._layers = list(layers or [])
        self._activations: dict[str, torch.Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
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
        """Register forward hooks on all target layers."""
        if self._enabled:
            return
        self._remove_hooks()
        module_map = dict(self._model.named_modules())
        for name in self._layers:
            module = module_map.get(name)
            if module is None:
                raise ValueError(
                    f"Module '{name}' not found in model. "
                    f"Available: {sorted(module_map.keys())[:20]}..."
                )
            hook = module.register_forward_hook(self._make_hook(name))
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
                self._activations[name] = output.detach().cpu()
            elif isinstance(output, tuple) and len(output) > 0:
                self._activations[name] = output[0].detach().cpu()
        return hook

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __del__(self) -> None:
        self._remove_hooks()


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
    np.savez(str(path), **{k: v.numpy() for k, v in activations.items()})
