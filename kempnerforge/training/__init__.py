"""Training loop and optimization for KempnerForge.

Public API:
  - Trainer: Core training loop orchestrator
  - build_optimizer / build_scheduler: Component factories
  - maybe_no_sync: Gradient accumulation helper
"""

from kempnerforge.training.grad import maybe_no_sync, scale_grads_by_token_count
from kempnerforge.training.optimizer import build_optimizer
from kempnerforge.training.scheduler import build_scheduler
from kempnerforge.training.trainer import Trainer

__all__ = [
    "Trainer",
    "build_optimizer",
    "build_scheduler",
    "maybe_no_sync",
    "scale_grads_by_token_count",
]
