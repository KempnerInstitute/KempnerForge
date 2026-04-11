"""Training loop and optimization for KempnerForge.

Public API:
  - build_loss_fn / build_optimizer / build_scheduler: Component factories
  - run_eval: Evaluation loop (loss + perplexity)
  - maybe_no_sync: Gradient accumulation helper
"""

from kempnerforge.training.eval import run_eval
from kempnerforge.training.grad import maybe_no_sync
from kempnerforge.training.loss import build_loss_fn
from kempnerforge.training.optimizer import build_optimizer
from kempnerforge.training.scheduler import build_scheduler

__all__ = [
    "build_loss_fn",
    "build_optimizer",
    "build_scheduler",
    "maybe_no_sync",
    "run_eval",
]
