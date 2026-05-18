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
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

from kempnerforge.checkpoint.async_save import AsyncCheckpointer
from kempnerforge.checkpoint.state import build_train_state, restore_train_state
from kempnerforge.config.schema import AsyncCheckpointMode, CheckpointConfig

logger = logging.getLogger(__name__)

# Filename for non-distributed training state within a checkpoint directory
_TRAIN_STATE_FILE = "train_state.pt"
_METADATA_FILE = "metadata.json"
# DCP writes this file LAST, once all shards are durable. Its presence is the
# authoritative signal that a checkpoint's distributed state is loadable.
_DCP_METADATA_FILE = ".metadata"


def _intersect_freeze_meta_by_module(
    saved: list[dict[str, Any]],
    expected: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter both freeze-metadata lists to the intersection of module keys.

    Used at checkpoint load time to make cross-arch resumes work cleanly:
    a Joint-Decoder checkpoint's ``vlm_freeze`` has only
    ``vision_encoder``; a Cross-Attention config's expected metadata has
    ``vision_encoder`` + ``cross_attention`` (auto-default in
    ``CrossAttentionConfig.module_patterns``). Loading JD into CA: the
    ``cross_attention`` entry is in ``expected`` but not in ``saved``,
    so it gets dropped from ``expected``; the remaining
    ``vision_encoder`` entries compare cleanly.

    Real semantic mismatches on shared keys are preserved. If both
    sides have ``vision_encoder`` but with different ``frozen`` values,
    the filtered lists differ and the caller raises.

    The lists are already canonicalized (sorted, deduped) by
    ``canonical_freeze_meta``, so the filter preserves canonical order.
    """
    saved_keys = {e["module"] for e in saved}
    expected_keys = {e["module"] for e in expected}
    shared = saved_keys & expected_keys
    return (
        [e for e in saved if e["module"] in shared],
        [e for e in expected if e["module"] in shared],
    )


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
        # Dataloader state stashed during load() when the caller cannot yet
        # provide a dataloader object. Applied later via
        # apply_dataloader_state() once the loader is constructed.
        self._pending_dataloader_state: dict[str, Any] | None = None
        # Async-only: a checkpoint whose DCP flush was dispatched but is not
        # yet durable. Its `latest` symlink swap + cleanup are deferred until
        # the flush completes (drained at the next save() or wait()), so
        # `latest` never points at a half-written checkpoint. (step, ckpt_dir).
        self._pending_finalize: tuple[int, Path] | None = None
        # Dedicated gloo group for the manager's own synchronization
        # (the end-of-save barrier). dcp.async_save() runs its collective
        # planning (all_gather/reduce_scatter) on a background thread using
        # the default process group; issuing the save() barrier on that
        # SAME group from the main thread interleaves two collectives on
        # one communicator with nondeterministic per-rank ordering, which
        # deadlocks under tightly spaced async saves. A separate gloo
        # group is independent, so the CPU-only barrier cannot race the
        # in-flight DCP collective. Created on all ranks (every rank
        # constructs CheckpointManager identically). Typed + cast because
        # the torch stubs declare new_group as ProcessGroup | int | None
        # but barrier(group=...) wants ProcessGroup | None.
        self._sync_group: dist.ProcessGroup | None = (
            cast("dist.ProcessGroup", dist.new_group(backend="gloo"))
            if dist.is_initialized()
            else None
        )

    def _checkpoint_dir(self, step: int) -> Path:
        return self.base_dir / f"step_{step}"

    def _latest_link(self) -> Path:
        return self.base_dir / "latest"

    def _dcp_dir(self, ckpt_dir: Path) -> Path:
        """DCP shard directory for a checkpoint (per-stage subdir under PP)."""
        return ckpt_dir / f"pp{self._pp_rank}" if self._pp_rank is not None else ckpt_dir

    def _dcp_complete(self, ckpt_dir: Path) -> bool:
        """True once DCP has written its `.metadata` (all shards durable)."""
        return (self._dcp_dir(ckpt_dir) / _DCP_METADATA_FILE).exists()

    def _commit_latest(self, step: int, ckpt_dir: Path) -> None:
        """Atomically point `latest` at a now-durable checkpoint, then prune.

        Rank 0 only. Called either inline (sync mode, DCP already durable) or
        deferred (async mode, after the flush future has resolved).
        """
        latest = self._latest_link()
        tmp_link = latest.with_suffix(".tmp")
        tmp_link.unlink(missing_ok=True)
        tmp_link.symlink_to(ckpt_dir.name)
        tmp_link.rename(latest)
        logger.info(f"Checkpoint committed: {ckpt_dir} (step={step})")
        self._cleanup()

    def _drain_pending_finalize(self) -> None:
        """Commit a deferred async checkpoint once its flush is durable.

        Invoked after the pending DCP future has been awaited (next save()
        or an explicit wait()/flush). Rank 0 performs the symlink + cleanup;
        other ranks no-op (they never touch the symlink), mirroring save().
        """
        if self._pending_finalize is None:
            return
        step, ckpt_dir = self._pending_finalize
        self._pending_finalize = None
        if self._rank == 0:
            self._commit_latest(step, ckpt_dir)

    def _sync_barrier(self) -> None:
        """Barrier on the manager's dedicated gloo group.

        Used to fence non-rank-0 ranks behind rank-0's metadata writes
        without sharing the default process group that DCP's async-save
        background thread is concurrently issuing collectives on.
        """
        if dist.is_initialized():
            dist.barrier(group=self._sync_group)

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
        # Dispatch the DCP save. For async modes this returns immediately but
        # FIRST awaits the previous in-flight flush, so any deferred
        # `_pending_finalize` from the prior save is now durable. For sync
        # mode (disabled) dcp.save() blocks until THIS checkpoint is durable.
        self._async_ckpt.save(
            dcp_state, checkpoint_id=str(dcp_dir), process_group=self._process_group
        )

        sync_mode = self.config.async_mode == AsyncCheckpointMode.disabled

        # The prior async checkpoint's flush is now guaranteed durable (the
        # dispatch above awaited it). Commit its `latest` symlink + cleanup
        # before doing anything else.
        self._drain_pending_finalize()

        # Save non-distributed state (rank 0 only). These are small synchronous
        # writes; they finish well before `latest` is ever pointed here, so
        # the checkpoint dir is fully populated by commit time.
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
            meta: dict[str, Any] = {"step": step, "tokens_seen": tokens_seen}
            if extra is not None and "vlm_freeze" in extra:
                # Already canonicalized by canonical_freeze_meta(...); stored as
                # a sorted, deduplicated list of {"module", "frozen"} dicts so
                # the comparison on load is reorder-invariant.
                meta["vlm_freeze"] = extra["vlm_freeze"]
            (ckpt_dir / _METADATA_FILE).write_text(json.dumps(meta, indent=2))

        if sync_mode:
            # DCP shards are already durable — commit immediately.
            if self._rank == 0:
                self._commit_latest(step, ckpt_dir)
        else:
            # Async flush still in flight. Defer the `latest` swap + cleanup
            # until it is durable (drained at the next save() or wait()), so
            # a crash mid-flush leaves `latest` on the last GOOD step rather
            # than a half-written one whose DCP `.metadata` is absent.
            self._pending_finalize = (step, ckpt_dir)

        # save() is a collective: non-rank-0 ranks must not return until
        # rank-0 has written train_state.pt + metadata.json (and, in sync
        # mode, advanced `latest`). Without this barrier, post-save hooks or
        # readers on other ranks race rank-0's writes (especially NFS/Lustre).
        # Uses the dedicated gloo group: the async DCP flush dispatched just
        # above runs collectives on the default process group from a
        # background thread, and sharing that group here deadlocks.
        self._sync_barrier()

    def wait(self) -> None:
        """Block until any pending async checkpoint save completes.

        Once the flush is durable, commit its deferred `latest` symlink +
        cleanup. The training loop calls this after the loop exits, so the
        final checkpoint's `latest` is committed before process teardown.
        """
        self._async_ckpt.wait()
        self._drain_pending_finalize()
        if dist.is_initialized():
            dist.barrier()

    def flush_pending_save(self) -> None:
        """Drain any in-flight async save before mutating model state.

        Called from the FreezeStage transition hook in the training
        loop: when a transition fires at step S, any save started at
        step S-1 must have written ``metadata.json`` with the
        pre-transition spec before the transition flips
        ``requires_grad``. Otherwise ``metadata.json`` lands with the
        post-transition spec attached to the pre-transition shards.

        Also commits the deferred `latest` symlink for that save, so a
        transition (or any caller draining the queue) leaves `latest`
        pointed at the now-durable checkpoint.
        """
        self._async_ckpt.wait()
        self._drain_pending_finalize()
        if dist.is_initialized():
            dist.barrier()

    def peek_saved_step(self, path: str | None = None) -> int | None:
        """Read ``step`` from a candidate checkpoint's metadata.json.

        Returns ``None`` if no checkpoint resolves or the metadata is
        missing/unreadable. Used by the training loop on resume to
        compute the expected freeze list (which depends on
        ``saved_step``) before calling ``load``.
        """
        ckpt_dir = self._resolve_load_path(path)
        if ckpt_dir is None:
            return None
        metadata_path = ckpt_dir / _METADATA_FILE
        if not metadata_path.exists():
            return None
        try:
            saved_meta = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        step = saved_meta.get("step")
        return int(step) if step is not None else None

    def load(
        self,
        path: str | None = None,
        scheduler: Any | None = None,
        dataloader: Any | None = None,
        exclude_keys: list[str] | None = None,
        vlm_freeze_expected: list[dict[str, Any]] | None = None,
    ) -> tuple[int, int, dict[str, Any]]:
        """Load a checkpoint and restore all state.

        Args:
            path: Checkpoint path. If None, loads from ``config.load_path``
                or the ``latest`` symlink.
            scheduler: LR scheduler to restore.
            dataloader: Stateful dataloader to restore.
            exclude_keys: DCP state keys to skip (e.g., ["optimizer"] for fine-tuning).
            vlm_freeze_expected: Canonical freeze metadata (output of
                ``canonical_freeze_meta``) for the current run's VLMConfig.
                When both the saved metadata and this argument are set,
                a mismatch raises ``ValueError`` unless the checkpoint
                config has ``ignore_freeze_mismatch=True``, in which case
                the load proceeds with a warning.

        Returns:
            Tuple of (step, tokens_seen, extra) where extra contains any
            additional keys saved via ``build_train_state(extra=...)``.
        """
        ckpt_dir = self._resolve_load_path(path)
        if ckpt_dir is None:
            logger.info("No checkpoint found — starting from scratch")
            return 0, 0, {}

        # When a DCP load will occur, fall back off an interrupted async
        # flush to the newest complete checkpoint so the whole load (DCP
        # shards, train_state.pt, metadata.json) stays consistent on one
        # step. Skipped when DCP is fully excluded (e.g. fine-tuning that
        # loads only train_state), where DCP durability is irrelevant.
        will_load_dcp = (
            exclude_keys is None or "model" not in exclude_keys or "optimizer" not in exclude_keys
        )
        if will_load_dcp:
            ckpt_dir = self._resolve_dcp_load_dir(ckpt_dir, path)

        logger.info(f"Loading checkpoint: {ckpt_dir}")

        # Check VLM freeze metadata BEFORE loading DCP shards so a mismatch
        # surfaces without leaving partial state in the live model.
        #
        # Cross-arch load rule: filter both saved and expected to the
        # intersection of module keys. A JD checkpoint's vlm_freeze has
        # only ``vision_encoder``; a Cross-Attention config's expected
        # has ``vision_encoder`` + ``cross_attention`` (auto-default).
        # Loading JD into CA: ``cross_attention`` is in expected but not
        # saved -> drop from expected; remaining ``vision_encoder``
        # entries compare cleanly. Real semantic mismatches on shared
        # keys (e.g., saved ``vision_encoder=True`` vs expected
        # ``vision_encoder=False``) still raise.
        if vlm_freeze_expected is not None:
            metadata_path = ckpt_dir / _METADATA_FILE
            if metadata_path.exists():
                try:
                    saved_meta = json.loads(metadata_path.read_text())
                except (OSError, json.JSONDecodeError) as e:
                    logger.warning(f"Could not read {metadata_path}: {e}")
                    saved_meta = {}
                saved_vlm_freeze = saved_meta.get("vlm_freeze")
                if saved_vlm_freeze is not None:
                    saved_filt, expected_filt = _intersect_freeze_meta_by_module(
                        saved_vlm_freeze, vlm_freeze_expected
                    )
                    if saved_filt != expected_filt:
                        dropped_from_saved = sorted(
                            {e["module"] for e in saved_vlm_freeze}
                            - {e["module"] for e in saved_filt}
                        )
                        dropped_from_expected = sorted(
                            {e["module"] for e in vlm_freeze_expected}
                            - {e["module"] for e in expected_filt}
                        )
                        cross_arch_note = ""
                        if dropped_from_saved or dropped_from_expected:
                            cross_arch_note = (
                                f" (cross-arch keys ignored: "
                                f"saved-only={dropped_from_saved}, "
                                f"current-only={dropped_from_expected})"
                            )
                        msg = (
                            f"VLM freeze mismatch at {ckpt_dir}: "
                            f"saved={saved_filt}, current={expected_filt}"
                            f"{cross_arch_note}"
                        )
                        if getattr(self.config, "ignore_freeze_mismatch", False):
                            logger.warning(
                                msg + " — proceeding because checkpoint.ignore_freeze_mismatch=True"
                            )
                        else:
                            raise ValueError(
                                msg + " (set checkpoint.ignore_freeze_mismatch=true to override)"
                            )

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

        # Load non-distributed state. On NFS/Lustre, independent stat()
        # calls can disagree briefly across ranks; if some ranks enter
        # this branch and others don't, the broadcast_object_list below
        # hangs. Use a rank-0-authoritative existence check broadcast to
        # all ranks so every rank takes the same branch.
        train_state_path = ckpt_dir / _TRAIN_STATE_FILE
        if dist.is_initialized():
            exists_flag = [train_state_path.exists() if self._rank == 0 else False]
            dist.broadcast_object_list(exists_flag, src=0)
            train_state_exists = bool(exists_flag[0])
        else:
            train_state_exists = train_state_path.exists()

        if train_state_exists:
            train_state = (
                _load_train_state(train_state_path)
                if self._rank == 0 or not dist.is_initialized()
                else None
            )

            # Broadcast from rank 0 to all ranks. PyTorch 2.11's
            # broadcast_object_list does not accept async_op, so a per-op
            # timeout cannot be wired here — this call inherits the 1800s
            # process-group default. A wedged rank will still surface, just
            # later than the other fast-fail paths in this patch.
            if dist.is_initialized():
                object_list = [train_state if self._rank == 0 else None]
                dist.broadcast_object_list(object_list, src=0)
                train_state = object_list[0]

            assert train_state is not None, "train_state broadcast failed"
            # Stash dataloader state if the caller can't yet provide the loader
            # object. Training loops construct the dataloader after load() so
            # apply_dataloader_state() can restore it once it exists.
            if dataloader is None and "dataloader" in train_state:
                self._pending_dataloader_state = train_state["dataloader"]
            step, tokens_seen, extra = restore_train_state(
                train_state,
                scheduler=scheduler,
                dataloader=dataloader,
            )
            logger.info(f"Resumed from step {step}, {tokens_seen:,} tokens seen")
            return step, tokens_seen, extra

        return 0, 0, {}

    def apply_dataloader_state(self, dataloader: Any) -> None:
        """Apply any dataloader state stashed during load().

        Training loops call load() before constructing the dataloader (since
        the dataloader depends on phase/annealing state that load() restores).
        This method applies the stashed state once the loader exists.

        No-op if no state is pending, or if the loader does not support
        ``load_state_dict`` (e.g., plain torch DataLoader for HF streaming).
        """
        if self._pending_dataloader_state is None:
            return
        if dataloader is None or not hasattr(dataloader, "load_state_dict"):
            self._pending_dataloader_state = None
            return
        dataloader.load_state_dict(self._pending_dataloader_state)
        self._pending_dataloader_state = None
        logger.info("Applied stashed dataloader state")

    def _newest_complete_checkpoint(self) -> Path | None:
        """Newest ``step_N`` dir whose DCP shards are durable, or None.

        Defense in depth: even though the Layer-1 deferral keeps `latest`
        off half-written checkpoints, a crash mid-flush (or a checkpoint
        dir left incomplete by older buggy code / external interference)
        can still leave the newest dir without DCP `.metadata`. Falling
        back to the newest COMPLETE dir keeps resume working instead of
        hard-failing in dcp.load with "metadata is None".
        """
        if not self.base_dir.exists():
            return None
        step_dirs = sorted(
            (d for d in self.base_dir.iterdir() if d.is_dir() and d.name.startswith("step_")),
            key=lambda d: int(d.name.split("_")[1]),
            reverse=True,
        )
        for d in step_dirs:
            if self._dcp_complete(d):
                return d
        return None

    def _resolve_load_path(self, path: str | None = None) -> Path | None:
        """Resolve the checkpoint path to load from.

        Returns the raw resolution (explicit path, ``load_path``, or the
        ``latest`` symlink target). The DCP-durability fallback for an
        interrupted async flush is applied separately in ``load()``,
        scoped to the case where DCP state is actually being loaded — so
        it never interferes with DCP-excluded loads (e.g. fine-tuning).
        """
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

    def _resolve_dcp_load_dir(self, resolved: Path, path: str | None) -> Path:
        """Pick the dir to DCP-load from, falling back if interrupted.

        ``resolved`` is the ``_resolve_load_path`` result. When it was
        reached via auto-resume (no explicit path/``load_path``) and its
        DCP shards are not durable — the signature of a crash during an
        async flush — fall back to the newest complete checkpoint so
        resume degrades to "last good step" instead of hard-failing in
        ``dcp.load`` with "metadata is None". An explicitly requested
        path is honored as-is (caller intent; fail loudly if broken).
        """
        explicit = path is not None or bool(self.config.load_path)
        if explicit or self._dcp_complete(resolved):
            return resolved
        fallback = self._newest_complete_checkpoint()
        if fallback is not None and fallback != resolved:
            logger.warning(
                f"`latest` -> {resolved} has no durable DCP metadata "
                f"(likely an interrupted async flush); resuming from "
                f"newest complete checkpoint {fallback} instead."
            )
            return fallback
        return resolved

    def _cleanup(self) -> None:
        """Remove old checkpoints beyond the retention limit.

        Two directories are never removed regardless of retention: the
        current ``latest`` target and the in-flight async checkpoint
        (``_pending_finalize``). Pruning either would let a crash strand
        resume with no loadable checkpoint — the exact failure this fix
        exists to prevent.
        """
        keep = self.config.keep_last_n
        if keep <= 0:
            return

        # Find all step_N directories
        ckpt_dirs = sorted(
            (d for d in self.base_dir.iterdir() if d.is_dir() and d.name.startswith("step_")),
            key=lambda d: int(d.name.split("_")[1]),
        )

        protected: set[Path] = set()
        latest = self._latest_link()
        if latest.exists():
            protected.add(latest.resolve())
        if self._pending_finalize is not None:
            protected.add(self._pending_finalize[1].resolve())

        # Remove oldest beyond retention, but never a protected dir.
        to_remove = ckpt_dirs[:-keep] if len(ckpt_dirs) > keep else []
        for d in to_remove:
            if d.resolve() in protected:
                continue
            shutil.rmtree(d)
            logger.info(f"Removed old checkpoint: {d}")
