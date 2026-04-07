"""Loss function registry for KempnerForge."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from kempnerforge.config.registry import registry


@registry.register_loss("cross_entropy")
def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard cross-entropy loss for language modeling."""
    return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
