"""Core training loop for KempnerForge.

The Trainer class orchestrates the full training lifecycle:
  init → build model → apply parallelism → train loop → checkpoint

The training loop is explicit sequential code — no hidden callbacks,
no event system, no framework magic. Researchers can read it top to
bottom or subclass to customize.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from kempnerforge.config.schema import JobConfig
from kempnerforge.distributed.parallel import apply_ac, apply_fsdp2
from kempnerforge.distributed.setup import get_world_info, init_distributed
from kempnerforge.distributed.tensor_parallel import apply_tensor_parallel
from kempnerforge.distributed.utils import clip_grad_norm_
from kempnerforge.metrics.logger import format_metrics, get_logger
from kempnerforge.model.transformer import Transformer
from kempnerforge.training.grad import maybe_no_sync
from kempnerforge.training.optimizer import build_optimizer
from kempnerforge.training.scheduler import build_scheduler

logger = get_logger(__name__)


class Trainer:
    """Training orchestrator.

    Usage:
        config = load_config("configs/train/default.toml")
        trainer = Trainer(config)
        trainer.train()

    Args:
        config: Full job configuration.
    """

    def __init__(self, config: JobConfig) -> None:
        self.config = config

        # 1. Initialize distributed
        rank, local_rank, world_size = get_world_info()
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        self.device_mesh = init_distributed(config.distributed, seed=config.train.seed)
        config.validate(world_size)

        # 2. Build model
        self.model = self._build_model()

        # 3. Build optimizer and scheduler
        self.optimizer = build_optimizer(self.model, config.optimizer)
        self.scheduler = build_scheduler(
            self.optimizer, config.scheduler, max_steps=config.train.max_steps
        )

        # 4. Training state
        self.step = 0
        self.tokens_seen = 0

    def _build_model(self) -> Transformer:
        """Build, parallelize, and optionally compile the model."""
        config = self.config

        model = Transformer(config.model)
        model = model.to(self.device)

        # Apply activation checkpointing (before parallelism)
        apply_ac(model, config.train.activation_checkpointing)

        # Apply tensor parallelism
        if self.device_mesh is not None and "tp" in self.device_mesh.mesh_dim_names:
            apply_tensor_parallel(model, self.device_mesh)

        # Apply FSDP2 (must be last)
        if self.device_mesh is not None:
            apply_fsdp2(model, self.device_mesh)

        # Optional torch.compile
        if config.train.compile_model:
            logger.info("Compiling model with torch.compile...")
            model = torch.compile(model)

        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model built: {n_params:,} parameters")

        return model

    def train_step(self, batch: dict[str, torch.Tensor]) -> tuple[float, float]:
        """Execute a single training step with gradient accumulation.

        Args:
            batch: Dict with "input_ids" and "labels" tensors.

        Returns:
            Tuple of (loss_value, grad_norm).
        """
        config = self.config.train

        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Forward + loss
        logits = self.model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

        # Scale loss for gradient accumulation
        scaled_loss = loss / config.grad_accum_steps
        scaled_loss.backward()

        # Track tokens
        self.tokens_seen += input_ids.numel() * self.world_size

        return loss.item(), 0.0  # grad_norm computed after accumulation

    def train(self, dataloader) -> None:
        """Run the full training loop.

        Args:
            dataloader: Iterable yielding dicts with "input_ids" and "labels".
        """
        config = self.config.train
        log_interval = self.config.metrics.log_interval

        logger.info(
            f"Starting training: max_steps={config.max_steps}, "
            f"batch_size={config.batch_size}, grad_accum={config.grad_accum_steps}, "
            f"world_size={self.world_size}"
        )

        data_iter = iter(dataloader)
        self.model.train()
        step_start = time.perf_counter()

        while self.step < config.max_steps:
            total_loss = 0.0

            # Gradient accumulation loop
            for micro_step in range(config.grad_accum_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

                with maybe_no_sync(self.model, micro_step, config.grad_accum_steps):
                    loss_val, _ = self.train_step(batch)
                    total_loss += loss_val

            # Gradient clipping
            grad_norm = clip_grad_norm_(self.model, config.grad_clip_norm)

            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            self.step += 1
            avg_loss = total_loss / config.grad_accum_steps

            # Logging
            if self.step % log_interval == 0 or self.step == 1:
                step_time = time.perf_counter() - step_start
                tokens_per_sec = (
                    config.batch_size
                    * config.seq_len
                    * config.grad_accum_steps
                    * self.world_size
                    / step_time
                )
                lr = self.scheduler.get_last_lr()[0]

                metrics = {
                    "loss": f"{avg_loss:.4f}",
                    "grad_norm": f"{grad_norm:.3f}"
                    if isinstance(grad_norm, float)
                    else f"{grad_norm.item():.3f}",
                    "lr": f"{lr:.2e}",
                    "tok/s": f"{tokens_per_sec:,.0f}",
                    "tokens": self.tokens_seen,
                }
                logger.info(format_metrics(self.step, metrics))
                step_start = time.perf_counter()

        logger.info(f"Training complete: {self.step} steps, {self.tokens_seen:,} tokens")

    def state_dict(self) -> dict:
        """Return full training state for checkpointing."""
        return {
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore training state from checkpoint."""
        self.step = state["step"]
        self.tokens_seen = state.get("tokens_seen", 0)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        logger.info(f"Resumed from step {self.step}")
