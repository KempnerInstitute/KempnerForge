"""Checkpoint manager for distributed checkpointing.

Uses PyTorch Distributed Checkpoint (DCP) for model and optimizer state,
which supports automatic resharding (save with N GPUs, load with M GPUs).

Non-distributed state (scheduler, dataloader, training meta, RNG) is saved
separately as a torch file and broadcast from rank 0 on load.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import stat
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

from kempnerforge.checkpoint.async_save import AsyncCheckpointer
from kempnerforge.checkpoint.state import build_train_state, restore_train_state
from kempnerforge.config.schema import CheckpointConfig

logger = logging.getLogger(__name__)

# Filename for non-distributed training state within a checkpoint directory
_TRAIN_STATE_FILE = "train_state.pt"
_METADATA_FILE = "metadata.json"


def _load_train_state(path: Path) -> dict[str, Any]:
    """Load ``train_state.pt`` under an explicit trust boundary.

    ``train_state.pt`` carries scheduler state, dataloader state, and a
    caller-supplied ``extra`` dict, so it is loaded with ``weights_only=False``
    (i.e. full pickle). Any object in the file whose class defines
    ``__reduce__`` runs arbitrary Python during ``torch.load``. On shared
    filesystems this is a real attack surface: anyone who can write into
    another user's checkpoint directory gets code execution in that user's
    training process on next resume.

    Refuses to load files not owned by the current UID and warns when the
    file is group- or world-writable. This does not defend against a
    same-UID compromise — if the attacker can write as you, they already
    win — but it closes the common "group-writable shared checkpoint dir"
    foot-gun and makes the trust boundary visible.

    Checkpoints imported from outside the lab (HuggingFace Hub, colleague
    transfers, etc.) will fail this check and must be either chown'd to the
    current user after inspection or converted to a weights-only-safe form.
    """
    st = path.stat()
    uid = os.getuid()
    if st.st_uid != uid:
        raise PermissionError(
            f"Refusing to load {path}: owned by uid={st.st_uid}, current uid={uid}. "
            f"train_state.pt is a pickle and loading it executes arbitrary Python. "
            f"If you trust this checkpoint, chown it to the current user after inspection."
        )
    if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        logger.warning(
            f"{path} is group/world-writable (mode={oct(st.st_mode & 0o777)}); "
            f"train_state.pt is a pickle and any writer can inject arbitrary code "
            f"at load time. Consider chmod g-w,o-w on the checkpoint directory."
        )
    return torch.load(path, map_location="cpu", weights_only=False)


class CheckpointManager:
    """Manages save/load/cleanup of distributed checkpoints.

    Each checkpoint is stored in a subdirectory: ``{dir}/step_{N}/``
    containing DCP shards and a non-distributed training state file.

    A ``latest`` symlink always points to the most recent checkpoint.

    Args:
        config: Checkpoint configuration.
        model: The model (FSDP-wrapped or plain).
        optimizer: The optimizer.
    """

    def __init__(
        self,
        config: CheckpointConfig,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        process_group=None,
        pp_rank: int | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.base_dir = Path(config.dir)
        self._rank = dist.get_rank() if dist.is_initialized() else 0
        self._async_ckpt = AsyncCheckpointer(mode=config.async_mode)
        self._process_group = process_group
        self._pp_rank = pp_rank

    def _checkpoint_dir(self, step: int) -> Path:
        return self.base_dir / f"step_{step}"

    def _latest_link(self) -> Path:
        return self.base_dir / "latest"

    def save(
        self,
        step: int,
        tokens_seen: int = 0,
        scheduler: Any | None = None,
        dataloader: Any | None = None,
        extra: dict | None = None,
    ) -> None:
        """Save a checkpoint at the given step.

        Args:
            step: Current training step.
            tokens_seen: Total tokens processed.
            scheduler: LR scheduler to save.
            dataloader: Stateful dataloader to save.
            extra: Additional metadata.
        """
        ckpt_dir = self._checkpoint_dir(step)

        # Create directory (all ranks)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # With PP, each stage has different parameters — save DCP shards to
        # a per-stage subdirectory to avoid .metadata file collisions.
        dcp_dir = ckpt_dir / f"pp{self._pp_rank}" if self._pp_rank is not None else ckpt_dir
        dcp_dir.mkdir(parents=True, exist_ok=True)

        # Save distributed state (model + optimizer) via DCP
        dcp_state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        self._async_ckpt.save(
            dcp_state, checkpoint_id=str(dcp_dir), process_group=self._process_group
        )

        # Save non-distributed state (rank 0 only)
        if self._rank == 0:
            train_state = build_train_state(
                step=step,
                tokens_seen=tokens_seen,
                scheduler=scheduler,
                dataloader=dataloader,
                extra=extra,
            )
            torch.save(train_state, ckpt_dir / _TRAIN_STATE_FILE)

            # Write human-readable metadata
            meta = {"step": step, "tokens_seen": tokens_seen}
            (ckpt_dir / _METADATA_FILE).write_text(json.dumps(meta, indent=2))

            # Update "latest" symlink
            latest = self._latest_link()
            tmp_link = latest.with_suffix(".tmp")
            tmp_link.unlink(missing_ok=True)
            tmp_link.symlink_to(ckpt_dir.name)
            tmp_link.rename(latest)

            logger.info(f"Checkpoint saved: {ckpt_dir} (step={step})")

            # Cleanup old checkpoints
            self._cleanup()

    def wait(self) -> None:
        """Block until any pending async checkpoint save completes."""
        self._async_ckpt.wait()

    def load(
        self,
        path: str | None = None,
        scheduler: Any | None = None,
        dataloader: Any | None = None,
        exclude_keys: list[str] | None = None,
    ) -> tuple[int, int, dict[str, Any]]:
        """Load a checkpoint and restore all state.

        Args:
            path: Checkpoint path. If None, loads from ``config.load_path``
                or the ``latest`` symlink.
            scheduler: LR scheduler to restore.
            dataloader: Stateful dataloader to restore.
            exclude_keys: DCP state keys to skip (e.g., ["optimizer"] for fine-tuning).

        Returns:
            Tuple of (step, tokens_seen, extra) where extra contains any
            additional keys saved via ``build_train_state(extra=...)``.
        """
        ckpt_dir = self._resolve_load_path(path)
        if ckpt_dir is None:
            logger.info("No checkpoint found — starting from scratch")
            return 0, 0, {}

        logger.info(f"Loading checkpoint: {ckpt_dir}")

        # Load distributed state via DCP
        dcp_dir = ckpt_dir / f"pp{self._pp_rank}" if self._pp_rank is not None else ckpt_dir

        dcp_state: dict[str, Any] = {}
        if exclude_keys is None or "model" not in exclude_keys:
            dcp_state["model"] = self.model.state_dict()
        if exclude_keys is None or "optimizer" not in exclude_keys:
            dcp_state["optimizer"] = self.optimizer.state_dict()

        if dcp_state:
            dcp.load(dcp_state, checkpoint_id=str(dcp_dir), process_group=self._process_group)

            if "model" in dcp_state:
                self.model.load_state_dict(dcp_state["model"])
            if "optimizer" in dcp_state:
                self.optimizer.load_state_dict(dcp_state["optimizer"])

        # Load non-distributed state
        train_state_path = ckpt_dir / _TRAIN_STATE_FILE
        if train_state_path.exists():
            train_state = _load_train_state(train_state_path)

            # Broadcast from rank 0 to all ranks
            if dist.is_initialized():
                object_list = [train_state if self._rank == 0 else None]
                dist.broadcast_object_list(object_list, src=0)
                train_state = object_list[0]

            assert train_state is not None, "train_state broadcast failed"
            step, tokens_seen, extra = restore_train_state(
                train_state,
                scheduler=scheduler,
                dataloader=dataloader,
            )
            logger.info(f"Resumed from step {step}, {tokens_seen:,} tokens seen")
            return step, tokens_seen, extra

        return 0, 0, {}

    def _resolve_load_path(self, path: str | None = None) -> Path | None:
        """Resolve the checkpoint path to load from."""
        if path is not None:
            p = Path(path)
            return p if p.exists() else None

        if self.config.load_path:
            p = Path(self.config.load_path)
            return p if p.exists() else None

        latest = self._latest_link()
        if latest.exists():
            return latest.resolve()

        return None

    def _cleanup(self) -> None:
        """Remove old checkpoints beyond the retention limit."""
        keep = self.config.keep_last_n
        if keep <= 0:
            return

        # Find all step_N directories
        ckpt_dirs = sorted(
            (d for d in self.base_dir.iterdir() if d.is_dir() and d.name.startswith("step_")),
            key=lambda d: int(d.name.split("_")[1]),
        )

        # Remove oldest beyond retention
        to_remove = ckpt_dirs[:-keep] if len(ckpt_dirs) > keep else []
        for d in to_remove:
            shutil.rmtree(d)
            logger.info(f"Removed old checkpoint: {d}")
