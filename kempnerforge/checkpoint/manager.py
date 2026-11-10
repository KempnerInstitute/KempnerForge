"""Checkpoint manager for distributed checkpointing.

Uses PyTorch Distributed Checkpoint (DCP) for model and optimizer state,
which supports automatic resharding (save with N GPUs, load with M GPUs).

Non-distributed state (scheduler, dataloader, training meta, RNG) is saved
separately as a torch file written by rank 0 and broadcast from rank 0 on
load. Scheduler and training meta restore from that single copy on every
rank. Dataloader and RNG state are per-rank: every rank's payload is gathered
into a versioned envelope at save time, and on an exact resume
(``load(exact_resume=True)``) each rank restores its own entry from the
broadcast (see ``kempnerforge.checkpoint.state``). Any other load (a
``load_path`` warm start, evaluation) restores rank 0's single copy on every
rank. A weights-only warm start (``exclude_keys=["optimizer", "train_state"]``)
reads no ``train_state.pt``, so it restores neither and checks no topology.

Resharding model and optimizer state is therefore free, but an exact resume
is not: a rank's loader cursor and RNG streams are valid only on the world
size and data-parallel layout that saved them, so an exact resume under a
changed topology raises on every rank. Setting
``checkpoint.ignore_rank_state_mismatch`` discards the saved loader/RNG state
on every rank instead, leaving step, model, optimizer and scheduler to resume
onto the new topology while data order and random streams restart fresh.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import pickle
import re
import shutil
import stat
import threading
import time
import uuid
from collections.abc import Callable, Container, Iterable, Iterator
from functools import partial
from pathlib import Path
from typing import Any, Protocol, TypeGuard, cast

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE, Metadata
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)

from kempnerforge.checkpoint.async_save import AsyncCheckpointer
from kempnerforge.checkpoint.state import (
    PER_RANK_KEY,
    build_per_rank_envelope,
    build_train_state,
    capture_rank_state,
    dp_coordinates,
    restore_train_state,
    select_rank_state,
)
from kempnerforge.config.checkpoint import is_weights_only, validate_exclude_keys
from kempnerforge.config.optimizer import WEIGHT_DECAY_SCOPES
from kempnerforge.config.schema import AsyncCheckpointMode, CheckpointConfig
from kempnerforge.distributed.utils import agree_across_ranks

logger = logging.getLogger(__name__)

_TRAIN_STATE_FILE = "train_state.pt"
_METADATA_FILE = "metadata.json"
_DCP_METADATA_FILE = ".metadata"
_OPTIM_STATE_PREFIX = "optimizer.state."
_OPTIM_PARAM_GROUPS_PREFIX = "optimizer.param_groups."
_NEVER_STEPPED_KEY = "optimizer_never_stepped"
_NEVER_STEPPED_PREFIX = f"{_NEVER_STEPPED_KEY}."
_OPTIM_PREFIX = "optimizer."
_UNRECORDED_WEIGHT_DECAY_SCOPE = "exclude_norms_and_biases"
_TRANSIENT_REMOVE_ERRNOS = frozenset({errno.ENOTEMPTY, errno.EBUSY, errno.ENOENT})
_REMOVE_RETRY_DELAYS = (0.05, 0.1, 0.2, 0.4)
TrackingSource = Callable[[], list[dict[str, str]]]
CheckpointRecord = Callable[[str, dict[str, str]], None]


class CheckpointRegistrar(Protocol):
    """Records checkpoints with the logging backends, which this module never imports."""

    def checkpoint_fields(self, step: int) -> dict[str, dict[str, str]]:
        """Fields each backend adds to its record of the checkpoint saved at ``step``, by backend.

        Called on rank 0 inside ``save()``, so it must answer from what it holds, never a server.
        """
        ...

    def register_checkpoint(self, step: int, directory: Path, record: CheckpointRecord) -> None:
        """Register the checkpoint at ``directory`` once ``latest`` points at it.

        Called on rank 0 when the checkpoint is committed, so it must not block. ``record`` adds
        fields to a backend's record in that checkpoint's ``metadata.json``, from any thread, later.
        """
        ...


def _saved_optimizer_param_groups(saved_keys: Iterable[str]) -> int:
    """Count the optimizer param groups among a DCP checkpoint's keys."""
    return len(
        {
            key.removeprefix(_OPTIM_PARAM_GROUPS_PREFIX).split(".", 1)[0]
            for key in saved_keys
            if key.startswith(_OPTIM_PARAM_GROUPS_PREFIX)
        }
    )


def _drop_legacy_group_keys(
    optim_state: dict[str, Any],
    optimizers: Iterable[torch.optim.Optimizer],
    saved_keys: Container[str],
) -> dict[str, Any]:
    """Drop from a load template the param-group keys its checkpoint predates.

    An optimizer lists in ``LEGACY_GROUP_DEFAULTS`` the keys its param groups gained
    after its checkpoints were first written, and fills each one in when it loads a
    group without it. Only those keys are dropped, and only from the groups whose
    saved copy lacks them; any other key a checkpoint lacks still fails the load.
    Returns the dropped keys with the values they resume with.
    """
    owners = [optimizer for optimizer in optimizers for _ in optimizer.param_groups]
    dropped: dict[str, Any] = {}
    for index, (group, owner) in enumerate(zip(optim_state["param_groups"], owners, strict=True)):
        for key, value in getattr(owner, "LEGACY_GROUP_DEFAULTS", {}).items():
            if key in group and f"{_OPTIM_PARAM_GROUPS_PREFIX}{index}.{key}" not in saved_keys:
                del group[key]
                dropped[key] = value
    return dropped


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


def _never_stepped_record(optim_state: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Record the parameters the optimizer holds without any state.

    An optimizer keeps state only for parameters that have received a gradient,
    and a checkpoint stores only what it keeps. A missing state and a state that
    never existed look the same once the optimizer is gone, so the save writes
    the distinction down while it still has it: one entry per parameter the
    param groups list and the state omits. The entries are tensors so that they
    stay separate keys in the checkpoint's own index, which is all the load
    reads; the value itself is never loaded.
    """
    state = optim_state["state"]
    return {
        fqn: torch.zeros((), dtype=torch.bool)
        for group in optim_state["param_groups"]
        for fqn in group["params"]
        if not state.get(fqn)
    }


def _params_recorded_never_stepped(fqns: set[str], saved_keys: Iterable[str]) -> set[str]:
    """Return the parameters in ``fqns`` the save recorded as holding no state.

    A record entry names one whole parameter, so the name is the rest of its
    key. A parameter whose state is simply absent is not recorded and stays in
    the load, which then fails on it, and a recorded parameter that also has
    saved state makes the two disagree, so the load fails rather than drop the
    state it does have. Each state key counts toward the longest parameter name
    it starts with, so a state of any layout marks its parameter.
    """
    recorded: set[str] = set()
    with_state: set[str] = set()
    for key in saved_keys:
        if key.startswith(_NEVER_STEPPED_PREFIX):
            recorded.add(key.removeprefix(_NEVER_STEPPED_PREFIX))
        elif key.startswith(_OPTIM_STATE_PREFIX):
            name = key.removeprefix(_OPTIM_STATE_PREFIX)
            while name and name not in fqns:
                name = name.rpartition(".")[0]
            with_state.add(name)
    recorded &= fqns
    contradicted = sorted(recorded & with_state)
    if contradicted:
        raise ValueError(
            f"{len(contradicted)} parameters are recorded as never stepped but have saved "
            f"optimizer state: {contradicted[:8]}"
        )
    return recorded


def _holds_optimizer(saved_keys: Iterable[str]) -> bool:
    """Return whether a checkpoint's keys include the optimizer entry, in either layout."""
    return any(key == "optimizer" or key.startswith(_OPTIM_PREFIX) for key in saved_keys)


def missing_train_state(ckpt_dir: Path) -> bool:
    """Whether checkpoint ``ckpt_dir`` holds optimizer state but no ``train_state.pt``.

    ``save()`` writes ``train_state.pt`` with the DCP shards (after them in a synchronous save,
    while an async save's shards still flush), so such a checkpoint is a save that stopped
    before both were written, or one that lost the file, and ``load()`` refuses to
    resume it. A checkpoint without optimizer state, such as an ``hf-to-dcp`` import, needs no
    ``train_state.pt``. The DCP ``.metadata`` of ``ckpt_dir``, or of each ``pp{k}`` stage, is
    read only when the file is absent. One that cannot be read is logged and counted as holding
    no optimizer state, which leaves the load to report it.
    """
    if (ckpt_dir / _TRAIN_STATE_FILE).exists():
        return False
    if (ckpt_dir / _DCP_METADATA_FILE).exists():
        dcp_dirs = [ckpt_dir]
    elif ckpt_dir.is_dir():
        dcp_dirs = sorted(
            d
            for d in ckpt_dir.iterdir()
            if d.is_dir() and d.name.startswith("pp") and (d / _DCP_METADATA_FILE).exists()
        )
    else:
        return False
    for dcp_dir in dcp_dirs:
        try:
            saved_keys = dcp.FileSystemReader(str(dcp_dir)).read_metadata().state_dict_metadata
        except Exception as e:
            logger.warning(
                f"Could not read {dcp_dir / _DCP_METADATA_FILE} to tell whether it holds "
                f"optimizer state: {e}"
            )
            continue
        if _holds_optimizer(saved_keys):
            return True
    return False


def _dcp_keys(dcp_dir: Path) -> dict[str, Any]:
    """Return the keys the DCP index in ``dcp_dir`` lists, each with its storage metadata."""
    return dcp.FileSystemReader(str(dcp_dir)).read_metadata().state_dict_metadata


def _read_weight_decay_scope(ckpt_dir: Path) -> str:
    """Return the weight-decay scope recorded in a checkpoint's ``metadata.json``.

    A checkpoint with no ``metadata.json``, or one without the field, was written under
    ``"exclude_norms_and_biases"``, the only grouping that existed before the scope was
    recorded.

    Raises:
        ValueError: ``metadata.json`` exists but is not a readable JSON object, or it
            records a scope outside ``WEIGHT_DECAY_SCOPES``.
    """
    metadata_path = ckpt_dir / _METADATA_FILE
    if not metadata_path.exists():
        return _UNRECORDED_WEIGHT_DECAY_SCOPE
    try:
        meta = json.loads(metadata_path.read_text())
    except (OSError, ValueError) as e:
        raise ValueError(f"cannot read {metadata_path}: {e}") from e
    if not isinstance(meta, dict):
        raise ValueError(f"{metadata_path} does not hold a JSON object")
    scope = meta.get("weight_decay_scope", _UNRECORDED_WEIGHT_DECAY_SCOPE)
    if scope not in WEIGHT_DECAY_SCOPES:
        raise ValueError(
            f"{metadata_path} records weight_decay_scope={scope!r}, which is not one of "
            f"{WEIGHT_DECAY_SCOPES}"
        )
    return scope


def _reset_to_fresh(optimizer: torch.optim.Optimizer) -> None:
    """Return ``optimizer`` to its freshly built state, keeping its param groups."""
    optimizer.load_state_dict({"state": {}, "param_groups": optimizer.state_dict()["param_groups"]})


class _NeverSteppedStatePlanner(DefaultLoadPlanner):
    """Load planner that leaves out the optimizer state the save recorded as absent.

    A fresh optimizer's template has state for every trainable parameter, but a
    checkpoint has none for parameters that had not received a gradient when it
    was saved. Those are dropped from the load and kept in ``never_stepped`` for
    the caller to confirm against the saved ``param_groups``; every other key
    stays required, including one whose state is missing without a record.
    """

    def __init__(self) -> None:
        super().__init__()
        self.never_stepped: set[str] = set()

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Metadata | None = None,
        is_coordinator: bool = False,
    ) -> None:
        optim_state = state_dict.get("optimizer")
        if optim_state is not None and metadata is not None:
            fqns = set(optim_state["state"])
            self.never_stepped = _params_recorded_never_stepped(fqns, metadata.state_dict_metadata)
            for fqn in self.never_stepped:
                del optim_state["state"][fqn]
        super().set_up_planner(state_dict, metadata, is_coordinator)


def _restore_never_stepped(
    optim_state: dict[str, Any], never_stepped: set[str], where: Path
) -> None:
    """Give the recorded parameters empty optimizer state once the saved optimizer lists them.

    ``optim_state`` holds the loaded ``param_groups``, which list every parameter the saved
    optimizer held, stepped or not. A recorded parameter they do not list was never part of
    that optimizer, so the load fails instead of guessing its state.
    """
    held = {fqn for group in optim_state["param_groups"] for fqn in group["params"]}
    unheld = sorted(never_stepped - held)
    if unheld:
        raise RuntimeError(
            f"Missing optimizer state in {where} for {len(unheld)} trainable parameters "
            f"the saved optimizer did not hold: {unheld[:8]}"
        )
    optim_state["state"].update({fqn: {} for fqn in never_stepped})


def _check_saved_state_matches(
    optim_state: dict[str, Any], saved_keys: Iterable[str], where: Path
) -> None:
    """Fail if a parameter was saved with another optimizer's state than the one it now has."""
    kept = {fqn: set(state) for fqn, state in optim_state["state"].items()}
    saved: dict[str, set[str]] = {}
    for key in saved_keys:
        if not key.startswith(_OPTIM_STATE_PREFIX):
            continue
        name = key.removeprefix(_OPTIM_STATE_PREFIX)
        fqn = name
        while fqn and fqn not in kept:
            fqn = fqn.rpartition(".")[0]
        if fqn:
            saved.setdefault(fqn, set()).add(name.removeprefix(f"{fqn}.").split(".", 1)[0])
    moved = sorted(
        fqn for fqn, entries in saved.items() if entries - kept[fqn] and kept[fqn] - entries
    )
    if moved:
        first = moved[0]
        raise RuntimeError(
            f"{where} holds another optimizer's state for {len(moved)} parameters: "
            f"{moved[:8]}. {first} was saved with "
            f"{sorted(saved[first])} and is now kept with {sorted(kept[first])}: its optimizer "
            "changed since the save, e.g. optimizer.name or optimizer.muon_router changed, "
            "or Muon used to orthogonalize it before it routed parameters by module. "
            "To continue from these weights with a fresh optimizer, set "
            "checkpoint.load_path to this checkpoint and checkpoint.exclude_from_loading = "
            '["optimizer"], with a checkpoint.dir that holds no checkpoint yet.'
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


def _names_backends(tracking: object) -> TypeGuard[list[dict[str, Any]]]:
    """True for a list of mappings that each name a ``backend``, the shape readers select on."""
    return isinstance(tracking, list) and all(
        isinstance(entry, dict)
        and isinstance(entry.get("backend"), str)
        and entry["backend"].strip()
        for entry in tracking
    )


def _string_fields(fields: object) -> bool:
    """True for a mapping of strings to strings, what a tracking record can hold."""
    return isinstance(fields, dict) and all(
        isinstance(key, str) and isinstance(value, str) for key, value in fields.items()
    )


def _replace_file(path: Path, text: str) -> None:
    """Replace ``path`` with ``text`` atomically: a temp file beside it, then a rename over it.

    An existing file keeps its mode, and its replacement is never readable more widely while it is
    written; a new one gets the mode a plain write would give it.
    """
    try:
        mode: int | None = stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        mode = None
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666 if mode is None else 0o600)
    try:
        with os.fdopen(fd, "w") as f:
            if mode is not None:
                os.fchmod(f.fileno(), mode)
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _remove_tree(path: Path, lock: threading.Lock, delays: Iterator[float]) -> None:
    """Remove the directory ``path``, retrying while NFS briefly keeps it non-empty.

    ``ENOTEMPTY``, ``EBUSY`` and ``ENOENT`` are retried while ``path`` exists, each retry
    after waiting the next of ``delays``.

    Args:
        path: The directory to remove.
        lock: Held for each attempt and released during the waits.
        delays: The waits left, in seconds, which removals can share.

    Raises:
        OSError: Any other error, or a retried one once ``path`` is gone or
            ``delays`` is spent.
    """
    while True:
        try:
            with lock:
                shutil.rmtree(path)
            return
        except OSError as e:
            if e.errno not in _TRANSIENT_REMOVE_ERRNOS or not path.exists():
                raise
            delay = next(delays, None)
            if delay is None:
                raise
            time.sleep(delay)


_URI_USERINFO = re.compile(r"(?<=://)[^/?#\s]+@")


def _error_text(error: BaseException) -> str:
    """``error``'s message with the userinfo of every URI in it removed; never raises."""
    try:
        return _URI_USERINFO.sub("", str(error))
    except Exception:
        return f"<unprintable {type(error).__name__}>"


def _attempt(action: Callable[..., object], *args: Any, **kwargs: Any) -> BaseException | None:
    """Call ``action(*args, **kwargs)``; return the error it raised, or ``None``.

    The error is an ``Exception`` or DCP's ``CheckpointException``, which derives from
    ``BaseException`` only; anything else, such as ``KeyboardInterrupt``, propagates.
    """
    try:
        action(*args, **kwargs)
    except (Exception, dcp.CheckpointException) as e:
        return e
    return None


class CheckpointManager:
    """Manages save/load/cleanup of distributed checkpoints.

    Each checkpoint is stored in a subdirectory: ``{dir}/step_{N}/``
    containing DCP shards and a non-distributed training state file.

    A ``latest`` symlink always points to the most recent checkpoint.

    Args:
        config: Checkpoint configuration.
        model: The model (FSDP-wrapped or plain).
        optimizer: The optimizer.
        dp_info: This rank's ``(dp_rank, dp_size)``, the data partition its
            loader reads. Recorded with the per-rank loader/RNG state and
            checked on load, so a resume under a different data-parallel
            layout is refused even at the same world size. None records no
            layout and checks only the world size.
        weight_decay_scope: The run's ``optimizer.weight_decay_scope``. When set, every save
            records it in ``metadata.json``, and a load that restores optimizer state on any
            rank raises ``ValueError`` on every rank, before anything is loaded, when the
            checkpoint was saved under a different scope. A checkpoint that holds no optimizer
            state, such as a converted one or one saved before the first step, loads under any
            scope with a fresh optimizer. A checkpoint that does not record a scope counts as
            ``"exclude_norms_and_biases"``.
    """

    def __init__(
        self,
        config: CheckpointConfig,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        process_group=None,
        pp_rank: int | None = None,
        dp_info: tuple[int, int] | None = None,
        weight_decay_scope: str | None = None,
    ) -> None:
        self._dp_info = dp_coordinates(dp_info) if dp_info is not None else None
        if dp_info is not None and self._dp_info is None:
            raise ValueError(
                f"dp_info must be (dp_rank, dp_size) ints with 0 <= dp_rank < dp_size, "
                f"got {dp_info!r}"
            )
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.checkpointed_optimizers: list[torch.optim.Optimizer] = [
            optimizer,
            *getattr(optimizer, "inner_optimizers", ()),
        ]
        self.base_dir = Path(config.dir)
        self._rank = dist.get_rank() if dist.is_initialized() else 0
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1
        self._async_ckpt = AsyncCheckpointer(mode=config.async_mode)
        self._process_group = process_group
        self._pp_rank = pp_rank
        self._weight_decay_scope = weight_decay_scope
        self._pending_dataloader_state: dict[str, Any] | None = None
        self._pending_finalize: tuple[int, Path] | None = None
        self._inflight_dir: Path | None = None
        self._tracking_source: TrackingSource | None = None
        self._registrar: CheckpointRegistrar | None = None
        self._sidecar_lock = threading.Lock()
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
        """Atomically point `latest` at a now-durable checkpoint, register it, then prune.

        Rank 0 only. Called either inline (sync mode, DCP already durable) or
        deferred (async mode, after the flush future has resolved). Raises only
        while `latest` still names the previous checkpoint: once it names this
        one the checkpoint is committed, so an error while pruning is logged and
        the next commit prunes again.
        """
        latest = self._latest_link()
        tmp_link = latest.with_suffix(".tmp")
        tmp_link.unlink(missing_ok=True)
        tmp_link.symlink_to(ckpt_dir.name)
        tmp_link.rename(latest)
        logger.info(f"Checkpoint committed: {ckpt_dir} (step={step})")
        self._register_committed(step, ckpt_dir)
        try:
            self._cleanup()
        except Exception as e:
            logger.warning(f"Could not prune old checkpoints after committing {ckpt_dir} ({e})")

    def _drain_pending_finalize(
        self, overwriting: Path | None = None, error: BaseException | None = None
    ) -> BaseException | None:
        """Commit a deferred async checkpoint once its flush is durable.

        Invoked on every rank after its pending DCP future has been awaited
        (next save() or an explicit wait()/flush), with the error that wait
        raised on this rank, if any. Under PP every rank first agrees that
        every stage's flush is durable, so a flush that failed on any rank
        raises on every rank and nothing is committed. Rank 0 then performs
        the symlink + cleanup; other ranks never touch the symlink, mirroring
        save(). A checkpoint pending in ``overwriting``, the directory a new
        save's flush is rewriting, is dropped uncommitted and unregistered;
        that save's own deferred commit points `latest` at it and registers it.

        Returns the error this rank still has to report: ``error``, or the one
        rank 0's commit raised.
        """
        if self._pending_finalize is None:
            return error
        step, ckpt_dir = self._pending_finalize
        self._pending_finalize = None
        self._await_every_stage(error)
        if self._rank == 0 and error is None and ckpt_dir != overwriting:
            return _attempt(self._commit_latest, step, ckpt_dir)
        return error

    def _await_every_stage(self, error: BaseException | None) -> None:
        """Under PP, agree that every stage's DCP write is durable; a no-op otherwise.

        Each stage saves through its own process group, so this rank's write
        being durable covers its own stage only; without PP the save spans every
        rank. Every rank calls this once its own write has ended, with the error
        it raised, if any; under PP it then raises on every rank when any rank's
        write failed (see ``agree``).
        """
        if self._pp_rank is not None:
            self.agree(error)

    def agree(self, error: BaseException | None, what: str = "Checkpoint save") -> None:
        """Agree on the manager's dedicated gloo group that every rank's ``what`` succeeded.

        See ``agree_across_ranks``. The group is not the default process group
        that DCP's async-save background thread issues collectives on.
        """
        agree_across_ranks(error, what, group=self._sync_group)

    def set_tracking_source(self, source: TrackingSource | None) -> None:
        """Record the logging runs ``source`` reports in every later ``metadata.json``.

        ``source`` returns one JSON-ready record per live run, and is called on rank 0 only,
        inside ``save()``, so it must return promptly. The ``tracking`` key is omitted when it
        reports none, raises, or returns what JSON cannot write. ``None`` detaches it.
        """
        if source is not None and not callable(source):
            raise TypeError(
                f"tracking source must be callable or None, got {type(source).__name__}"
            )
        self._tracking_source = source

    def set_checkpoint_registrar(self, registrar: CheckpointRegistrar | None) -> None:
        """Record every later checkpoint with the logging backends through ``registrar``.

        Both of its methods run on rank 0 only. ``checkpoint_fields`` is asked inside ``save()``
        when a backend has a live run, and its fields join that save's ``tracking`` records.
        ``register_checkpoint`` gets each checkpoint once it is committed, with a
        ``record(backend, fields)`` that later adds fields to that backend's record, atomically
        and from any thread. A registrar that raises, or answers what a record cannot hold, costs
        only those fields or that registration, never the save or the commit. ``None`` detaches it.
        """
        if registrar is not None and not all(
            callable(getattr(registrar, name, None))
            for name in ("checkpoint_fields", "register_checkpoint")
        ):
            raise TypeError(
                "checkpoint registrar must have callable checkpoint_fields and "
                f"register_checkpoint, or be None; got {type(registrar).__name__}"
            )
        self._registrar = registrar

    def _with_checkpoint_fields(
        self, tracking: list[dict[str, str]], step: int
    ) -> list[dict[str, str]]:
        """``tracking`` plus the fields each backend adds for this save; as given if none."""
        if self._registrar is None:
            return tracking
        try:
            added = self._registrar.checkpoint_fields(step)
            if not isinstance(added, dict):
                raise TypeError(f"expected a dict of fields by backend, got {type(added).__name__}")
            merged = [{**entry, **added.get(entry["backend"], {})} for entry in tracking]
            if not all(_string_fields(entry) for entry in merged):
                raise TypeError("fields must be strings")
        except Exception as e:
            logger.warning(
                "Could not add the checkpoint fields of step %s: %s", step, _error_text(e)
            )
            return tracking
        return merged

    def _sidecar_text(self, meta: dict[str, Any]) -> str:
        """``meta`` as JSON, plus the run's ``weight_decay_scope`` when the manager has one and
        the live tracking runs when the source can supply them.

        Everything that depends on the source sits inside one guard: it runs on rank 0 alone,
        so a raise here would strand the other ranks. A bad source costs the ``tracking`` key.
        """
        if self._weight_decay_scope is not None:
            meta = {**meta, "weight_decay_scope": self._weight_decay_scope}
        if self._tracking_source is not None:
            try:
                tracking = self._tracking_source()
                if tracking:
                    if not _names_backends(tracking):
                        raise TypeError("expected a list of mappings that each name a backend")
                    tracking = self._with_checkpoint_fields(tracking, meta["step"])
                    return json.dumps({**meta, "tracking": tracking}, indent=2)
            except Exception as e:
                logger.warning(
                    "Could not record the tracking runs in %s: %s", _METADATA_FILE, _error_text(e)
                )
        return json.dumps(meta, indent=2)

    def _register_committed(self, step: int, ckpt_dir: Path) -> None:
        """Hand a just-committed checkpoint to the registrar, which must not block."""
        if self._registrar is None:
            return
        try:
            self._registrar.register_checkpoint(
                step, ckpt_dir, partial(self._record, step, ckpt_dir)
            )
        except Exception as e:
            logger.warning(
                "Could not register %s with the tracking backends: %s",
                ckpt_dir.name,
                _error_text(e),
            )

    def _record(self, step: int, ckpt_dir: Path, backend: str, fields: dict[str, str]) -> None:
        """Add ``fields`` to ``backend``'s record in the ``metadata.json`` of step ``step``.

        The registrar's ``record``: callable from any thread, and it never raises. The file is read
        and replaced atomically under the sidecar lock, so a reader always sees a whole file and an
        amend never overwrites a newer save of the step. Only an existing record in that step's own
        sidecar is amended: a checkpoint removed since, or saved without that backend's record, is
        skipped quietly.
        """
        path = ckpt_dir / _METADATA_FILE
        try:
            if not isinstance(backend, str) or not _string_fields(fields):
                raise TypeError("expected a backend name and fields of strings")
            with self._sidecar_lock:
                try:
                    meta = json.loads(path.read_text())
                except FileNotFoundError:
                    logger.debug("%s is gone; the %s fields are not recorded", path, backend)
                    return
                if not isinstance(meta, dict) or meta.get("step") != step:
                    raise ValueError(f"it is not the metadata of step {step}")
                tracking = meta.get("tracking")
                if not _names_backends(tracking) or all(e["backend"] != backend for e in tracking):
                    logger.debug("%s has no %s record; its fields are not recorded", path, backend)
                    return
                meta["tracking"] = [
                    {**entry, **fields} if entry["backend"] == backend else entry
                    for entry in tracking
                ]
                _replace_file(path, json.dumps(meta, indent=2))
        except Exception as e:
            logger.warning(
                "Could not record the %s fields in %s: %s", backend, path, _error_text(e)
            )

    def _gather_rank_states(
        self, dataloader: Any | None
    ) -> tuple[dict[str, Any] | None, BaseException | None]:
        """Collect every rank's own loader/RNG payload into a versioned envelope.

        Collective: every rank calls it once per ``save()``, whatever its own part of
        the save has raised so far. Uses the dedicated gloo group so the gather cannot
        interleave with DCP collectives an in-flight async flush is issuing on the
        default group from its background thread. Each rank captures and serializes
        its payload before the gather, and a rank whose capture raises enters the
        gather with no payload, so no rank is left waiting in it. Entries are keyed by
        each payload's own global rank rather than by gather position, so the envelope
        is correct under any group ordering. Each rank's ``dp_info`` rides along to
        record the data-parallel layout.

        Returns:
            The envelope and ``None``; or ``None`` and the error this rank reports:
            what its own capture raised, a ``RuntimeError`` naming the ranks whose
            capture failed, or what building the envelope raised, which every rank
            meets alike since it builds from the same gathered payloads.
        """
        payload: bytes | None = None
        capture_error: Exception | None = None
        try:
            payload = pickle.dumps(capture_rank_state(dataloader))
        except Exception as e:
            capture_error = e
        mine = (self._rank, self._dp_info, payload)
        gathered: list[Any] = [mine]
        if dist.is_initialized():
            gathered = [None] * self._world_size
            dist.all_gather_object(gathered, mine, group=self._sync_group)
        if capture_error is not None:
            return None, capture_error
        failed = sorted(int(rank) for rank, _, data in gathered if data is None)
        if failed:
            return None, RuntimeError(
                f"Checkpoint save failed on ranks {failed}: they could not capture their "
                f"dataloader/RNG state"
            )
        try:
            return self._build_envelope(gathered), None
        except Exception as e:
            return None, e

    def _build_envelope(self, gathered: list[Any]) -> dict[str, Any]:
        """The per-rank envelope from every rank's ``(rank, dp_info, serialized payload)``."""
        entries = {int(rank): pickle.loads(data) for rank, _, data in gathered}
        coords = {int(rank): dp for rank, dp, _ in gathered}
        layout = coords if all(dp is not None for dp in coords.values()) else None
        return build_per_rank_envelope(entries, world_size=self._world_size, data_parallel=layout)

    def _gather_dp_layout(self) -> dict[int, tuple[int, int]] | None:
        """This run's global rank -> ``(dp_rank, dp_size)``, or None if unknown.

        Collective on the dedicated gloo group: each rank knows only its own
        coordinates, and the envelope check needs the whole layout so every
        rank reaches the same verdict.
        """
        mine = (self._rank, self._dp_info)
        gathered: list[Any] = [mine]
        if dist.is_initialized():
            gathered = [None] * self._world_size
            dist.all_gather_object(gathered, mine, group=self._sync_group)
        coords = {int(rank): dp for rank, dp in gathered}
        return coords if all(dp is not None for dp in coords.values()) else None

    def _per_rank_view(self, train_state: dict[str, Any]) -> dict[str, Any]:
        """This rank's view of the broadcast train state on an exact resume.

        A checkpoint with a per-rank envelope: the view replaces the legacy
        single-copy ``rng``/``dataloader`` keys (always rank 0's) with this
        rank's own entry, selected by global rank. With an envelope it gathers
        the current data-parallel layout (a collective), so every rank must
        call it. ``load()`` calls it right after the train-state broadcast,
        which every rank must reach, and whether to gather depends on the
        broadcast state alone, so no rank waits in the gather for one that
        failed earlier in ``load()``. A topology or schema mismatch refuses
        the load unless ``checkpoint.ignore_rank_state_mismatch`` downgrades
        it to a warning that discards the saved loader/RNG state. The envelope
        check reads only state every rank shares, so all ranks restore,
        refuse, or discard together. A legacy checkpoint (no envelope) keeps
        its old behavior: rank 0's single copy restores on every rank.
        """
        envelope = train_state.get(PER_RANK_KEY)
        if envelope is None:
            if self._world_size > 1 and self._rank == 0:
                logger.info(
                    "Checkpoint has no per-rank state (legacy format); rank 0's "
                    "dataloader/RNG state restores on every rank"
                )
            return train_state
        view = {k: v for k, v in train_state.items() if k not in ("rng", "dataloader")}
        layout = self._gather_dp_layout()
        try:
            entry = select_rank_state(
                envelope, rank=self._rank, world_size=self._world_size, data_parallel=layout
            )
        except ValueError as e:
            if not self.config.ignore_rank_state_mismatch:
                raise
            logger.warning(
                f"Discarding saved per-rank dataloader/RNG state because "
                f"checkpoint.ignore_rank_state_mismatch=true: {e}. Data order and "
                f"random streams restart fresh; model/optimizer/scheduler/step "
                f"still resume."
            )
            return view
        view["rng"] = entry["rng"]
        if "dataloader" in entry:
            view["dataloader"] = entry["dataloader"]
        return view

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

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._inflight_dir = ckpt_dir

        dcp_dir = ckpt_dir / f"pp{self._pp_rank}" if self._pp_rank is not None else ckpt_dir
        dcp_dir.mkdir(parents=True, exist_ok=True)

        dcp_state: dict[str, Any] = {"model": get_model_state_dict(self.model)}
        if any(optimizer.state for optimizer in self.checkpointed_optimizers):
            optim_state = get_optimizer_state_dict(self.model, self.checkpointed_optimizers)
            dcp_state["optimizer"] = optim_state
            dcp_state[_NEVER_STEPPED_KEY] = _never_stepped_record(optim_state)
        flush_error = _attempt(self._async_ckpt.wait)
        error = flush_error
        if flush_error is None:
            error = _attempt(
                self._async_ckpt.save,
                dcp_state,
                checkpoint_id=str(dcp_dir),
                process_group=self._process_group,
            )

        sync_mode = self.config.async_mode == AsyncCheckpointMode.disabled

        drain_error = self._drain_pending_finalize(overwriting=ckpt_dir, error=flush_error)
        if drain_error is not None:
            error = drain_error

        per_rank, gather_error = self._gather_rank_states(dataloader)
        if error is None:
            error = gather_error

        if self._rank == 0 and error is None:
            error = _attempt(
                self._write_train_state,
                ckpt_dir,
                step,
                tokens_seen,
                scheduler,
                dataloader,
                extra,
                per_rank,
            )

        if sync_mode:
            self._await_every_stage(error)
            if self._rank == 0 and error is None:
                error = _attempt(self._commit_latest, step, ckpt_dir)

        self.agree(error)

        if not sync_mode:
            self._pending_finalize = (step, ckpt_dir)

    def _write_train_state(
        self,
        ckpt_dir: Path,
        step: int,
        tokens_seen: int,
        scheduler: Any | None,
        dataloader: Any | None,
        extra: dict | None,
        per_rank: dict[str, Any] | None = None,
    ) -> None:
        """Write ``train_state.pt`` and ``metadata.json`` into ``ckpt_dir``: rank 0's save() part.

        ``extra["vlm_freeze"]``, when present, is recorded in ``metadata.json`` as given:
        ``canonical_freeze_meta`` has already sorted and deduplicated it, so the comparison
        on load does not depend on order.

        ``per_rank`` is the envelope ``save()`` gathered from every rank; omitting it writes
        a legacy train state carrying rank 0's loader/RNG state only.
        """
        train_state = build_train_state(
            step=step,
            tokens_seen=tokens_seen,
            scheduler=scheduler,
            dataloader=dataloader,
            extra=extra,
            per_rank=per_rank,
        )
        torch.save(train_state, ckpt_dir / _TRAIN_STATE_FILE)
        meta: dict[str, Any] = {"step": step, "tokens_seen": tokens_seen}
        if extra is not None and "vlm_freeze" in extra:
            meta["vlm_freeze"] = extra["vlm_freeze"]
        text = self._sidecar_text(meta)
        with self._sidecar_lock:
            _replace_file(ckpt_dir / _METADATA_FILE, text)

    def wait(self) -> None:
        """Block until any pending async checkpoint save completes.

        Once the flush is durable, commit its deferred `latest` symlink +
        cleanup. The training loop calls this after the loop exits, so the
        final checkpoint's `latest` is committed before process teardown.
        A flush that failed on any rank raises on every rank, and nothing is
        committed.
        """
        error = _attempt(self._async_ckpt.wait)
        self.agree(self._drain_pending_finalize(error=error))

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
        pointed at the now-durable checkpoint. A flush that failed on any
        rank raises on every rank, and nothing is committed.
        """
        error = _attempt(self._async_ckpt.wait)
        self.agree(self._drain_pending_finalize(error=error))

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
        exact_resume: bool = False,
    ) -> tuple[int, int, dict[str, Any]]:
        """Load a checkpoint and restore all state.

        Args:
            path: Checkpoint path. If None, loads from ``config.load_path``
                or the ``latest`` symlink.
            scheduler: LR scheduler to restore.
            dataloader: Stateful dataloader to restore.
            exclude_keys: State to skip: the DCP keys ``"model"`` and
                ``"optimizer"`` (e.g. ``["optimizer"]`` for fine-tuning), or
                ``["optimizer", "train_state"]`` for a weights-only warm start
                that loads the model alone and ignores ``train_state.pt`` (step,
                token count, scheduler, dataloader position, RNG) and the saved
                freeze metadata, so the new run starts at step 0 with its own
                schedule, data order and freeze. ``"train_state"`` in any other
                combination raises ``ValueError``.
            vlm_freeze_expected: Canonical freeze metadata (output of
                ``canonical_freeze_meta``) for the current run's VLMConfig.
                When both the saved metadata and this argument are set,
                a mismatch raises ``ValueError`` unless the checkpoint
                config has ``ignore_freeze_mismatch=True``, in which case
                the load proceeds with a warning.
            exact_resume: True only when the run resumes from its own
                checkpoint (auto-resume). Each rank then restores its own
                dataloader/RNG state from the per-rank envelope, which must
                match this run's world size and data-parallel layout. Every
                rank must pass the same value. False (a ``load_path`` warm
                start, evaluation, any other load) restores the single-copy
                ``rng``/``dataloader`` keys (rank 0's) on every rank with no
                topology check. The weights-only ``exclude_keys`` pair reads no
                train state: no rank restores loader/RNG state or enters the
                train-state broadcasts or the layout gather, and with
                ``exact_resume`` it raises.

        Returns:
            Tuple of (step, tokens_seen, extra) where extra contains any
            additional keys saved via ``build_train_state(extra=...)``.

        Raises:
            FileNotFoundError: No checkpoint resolves although one was named, or the
                checkpoint holds optimizer state that this load restores but has no
                ``train_state.pt``. Raised before any state is loaded.
            ValueError: ``exclude_keys`` is not a list of distinct DCP keys, names
                ``"train_state"`` outside ``["optimizer", "train_state"]``, or is that
                pair with ``exact_resume``. Raised before the path is resolved.
        """
        if exclude_keys is not None:
            validate_exclude_keys(exclude_keys, "exclude_keys")
        if exact_resume and is_weights_only(exclude_keys):
            raise ValueError(
                "exact_resume restores each rank's dataloader/RNG state from train_state.pt, "
                "which exclude_keys=['optimizer', 'train_state'] skips: a weights-only warm "
                "start begins a new run at step 0. Load it with exact_resume=False, or drop "
                "'train_state' from exclude_keys to resume."
            )
        ckpt_dir = self._resolve_load_path(path)
        if ckpt_dir is None:
            requested = path if path is not None else self.config.load_path
            if requested:
                raise FileNotFoundError(f"Checkpoint not found: {requested}")
            logger.info("No checkpoint found — starting from scratch")
            return 0, 0, {}

        will_load_dcp = (
            exclude_keys is None or "model" not in exclude_keys or "optimizer" not in exclude_keys
        )
        if will_load_dcp:
            ckpt_dir = self._resolve_dcp_load_dir(ckpt_dir, path)

        logger.info(f"Loading checkpoint: {ckpt_dir}")
        load_train_state = not is_weights_only(exclude_keys)
        if not load_train_state and vlm_freeze_expected is not None:
            logger.warning(
                f"vlm_freeze_expected is not compared with {ckpt_dir}: a weights-only warm start "
                "applies the new run's freeze"
            )

        if vlm_freeze_expected is not None and load_train_state:
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

        dcp_dir = ckpt_dir / f"pp{self._pp_rank}" if self._pp_rank is not None else ckpt_dir

        load_model = exclude_keys is None or "model" not in exclude_keys
        load_optim = exclude_keys is None or "optimizer" not in exclude_keys

        saved_keys = self._read_saved_keys(ckpt_dir, dcp_dir) if load_optim else {}
        fresh_optimizer = load_optim and not _holds_optimizer(saved_keys)
        if fresh_optimizer:
            load_optim = False

        train_state_path = ckpt_dir / _TRAIN_STATE_FILE
        train_state_exists = load_train_state and self._train_state_exists(train_state_path)
        if load_optim and not train_state_exists:
            raise FileNotFoundError(
                f"Cannot resume {ckpt_dir}: it holds optimizer state but no {_TRAIN_STATE_FILE}, "
                f"so loading it would continue its weights and optimizer moments from step 0 and "
                f"restart the LR schedule and the data stream. Its save stopped before writing "
                f"{_TRAIN_STATE_FILE}, or the file was lost. Restore the file, or set a new "
                f"checkpoint.dir with checkpoint.load_path pointing at an earlier complete "
                f"step_N to resume from it, or at this checkpoint with "
                f'checkpoint.exclude_from_loading = ["optimizer"] to start from its weights alone.'
            )

        dcp_state: dict[str, Any] = {}
        optimizers = self._optimizers_saved_in(dcp_dir, saved_keys) if load_optim else []
        if load_model:
            dcp_state["model"] = get_model_state_dict(self.model)
        if load_optim:
            dcp_state["optimizer"] = get_optimizer_state_dict(self.model, optimizers)
            filled = _drop_legacy_group_keys(dcp_state["optimizer"], optimizers, saved_keys)
            if filled:
                logger.warning(
                    f"{dcp_dir} is missing optimizer param-group keys {sorted(filled)}; "
                    f"resuming with {filled}"
                )
            _check_saved_state_matches(dcp_state["optimizer"], saved_keys, dcp_dir)

        if dcp_state:
            planner = _NeverSteppedStatePlanner()
            dcp.load(
                dcp_state,
                checkpoint_id=str(dcp_dir),
                process_group=self._process_group,
                planner=planner,
            )

            if load_optim:
                optim_state = cast("dict[str, Any]", dcp_state["optimizer"])
                _restore_never_stepped(optim_state, planner.never_stepped, dcp_dir)
            if load_model:
                set_model_state_dict(self.model, dcp_state["model"])
            if load_optim:
                set_optimizer_state_dict(
                    self.model, optimizers, optim_state_dict=dcp_state["optimizer"]
                )
            if planner.never_stepped:
                logger.info(
                    f"{len(planner.never_stepped)} parameters the saved optimizer held were never "
                    f"stepped and resume without state: {sorted(planner.never_stepped)[:8]}"
                )
        if fresh_optimizer:
            for optimizer in self.checkpointed_optimizers:
                _reset_to_fresh(optimizer)

        if not load_train_state:
            logger.info(
                f"Weights-only warm start from {ckpt_dir}: its step, schedule, data position "
                "and RNG are not restored, and its freeze metadata is not compared"
            )
            step, tokens_seen, extra = 0, 0, {}
        elif train_state_exists:
            train_state = (
                _load_train_state(train_state_path)
                if self._rank == 0 or not dist.is_initialized()
                else None
            )

            if dist.is_initialized():
                object_list = [train_state if self._rank == 0 else None]
                dist.broadcast_object_list(object_list, src=0)
                train_state = object_list[0]

            assert train_state is not None, "train_state broadcast failed"
            if exact_resume:
                train_state = self._per_rank_view(train_state)
            if dataloader is None and "dataloader" in train_state:
                self._pending_dataloader_state = train_state["dataloader"]
            step, tokens_seen, extra = restore_train_state(
                train_state,
                scheduler=scheduler,
                dataloader=dataloader,
            )
            logger.info(f"Resumed from step {step}, {tokens_seen:,} tokens seen")
        else:
            step, tokens_seen, extra = 0, 0, {}
            logger.info(
                f"{ckpt_dir} has no {_TRAIN_STATE_FILE}; the step counter, token count, LR "
                f"schedule, data position and RNG state start fresh at step 0"
            )

        if fresh_optimizer and step > 0:
            logger.warning(
                f"{dcp_dir} holds no optimizer state at step {step}: resuming with a fresh "
                f"optimizer, so its moments restart from zero and this resume does not "
                f"continue the saved run exactly."
            )
        elif fresh_optimizer:
            logger.info(f"{dcp_dir} holds no optimizer state; the optimizer starts fresh")
        return step, tokens_seen, extra

    def _train_state_exists(self, path: Path) -> bool:
        """Whether ``path`` exists, as rank 0 sees it, on every rank.

        On NFS/Lustre independent stat() calls can disagree briefly across ranks. Ranks that
        disagree take different branches of ``load()``: some raise or skip the train state while
        the others wait in its broadcast and hang. Rank 0 checks, and its answer is broadcast.
        """
        if not dist.is_initialized():
            return path.exists()
        exists_flag = [path.exists() if self._rank == 0 else False]
        dist.broadcast_object_list(exists_flag, src=0)
        return bool(exists_flag[0])

    def _read_saved_keys(self, ckpt_dir: Path, dcp_dir: Path) -> dict[str, Any]:
        """Return the keys this rank's DCP index lists: ``dcp_dir``, its part of ``ckpt_dir``.

        With a weight-decay scope every rank must call this. One all-reduce on the manager's
        gloo group, which every rank enters even when its own read failed, tells every rank
        whether any rank's index holds optimizer state; under PP each stage has its own index.
        If one does, every rank checks the scope before this returns; otherwise none does.

        Raises:
            ValueError: The checkpoint fails the scope check (see ``_check_weight_decay_scope``).
            RuntimeError: Another rank could not read its DCP index.
            Exception: Whatever this rank's read of its DCP index raised.
        """
        if self._weight_decay_scope is None:
            return _dcp_keys(dcp_dir)
        keys: dict[str, Any] = {}
        error: Exception | None = None
        try:
            keys = _dcp_keys(dcp_dir)
        except Exception as e:
            error = e
        if self._optimizer_saved_on_any_rank(error, _holds_optimizer(keys)):
            self._check_weight_decay_scope(ckpt_dir)
        return keys

    def _optimizer_saved_on_any_rank(self, error: Exception | None, holds: bool) -> bool:
        """Return whether any rank's DCP index holds optimizer state, the same on every rank.

        Every rank passes the error its read of its index raised, or ``None``, and whether its
        index holds optimizer state. After one all-reduce on the manager's gloo group a rank
        with an error re-raises it, and every other rank raises a ``RuntimeError`` naming the
        ranks that failed. Without an initialized process group, ``error`` is raised if set.
        """
        if not dist.is_initialized():
            if error is not None:
                raise error
            return holds
        flags = torch.zeros(2, dist.get_world_size(self._sync_group), dtype=torch.int32)
        rank = dist.get_rank(self._sync_group)
        flags[0, rank] = int(error is not None)
        flags[1, rank] = int(holds)
        dist.all_reduce(flags, group=self._sync_group)
        if error is not None:
            raise error
        failed = flags[0].nonzero().flatten().tolist()
        if failed:
            raise RuntimeError(f"Could not read the DCP index of the checkpoint on ranks {failed}")
        return bool(flags[1].any())

    def _check_weight_decay_scope(self, ckpt_dir: Path) -> None:
        """Refuse a checkpoint saved under a weight-decay scope other than this manager's.

        Every rank calls this or none does. Rank 0 reads the checkpoint's scope and broadcasts
        it, or whatever stopped it reading, so every rank raises or proceeds together and none
        is left waiting on the broadcast.

        Raises:
            ValueError: The checkpoint's scope differs from this manager's, or its
                ``metadata.json`` does not hold a valid scope record.
        """
        outcome: list[tuple[str | None, str | None]] = [(None, None)]
        if self._rank == 0 or not dist.is_initialized():
            try:
                outcome[0] = (_read_weight_decay_scope(ckpt_dir), None)
            except ValueError as e:
                outcome[0] = (None, str(e))
            except Exception as e:
                outcome[0] = (None, f"{type(e).__name__}: {e}")
        if dist.is_initialized():
            dist.broadcast_object_list(outcome, src=0)
        saved, error = outcome[0]
        if error is not None:
            raise ValueError(f"Cannot check the weight_decay_scope of {ckpt_dir}: {error}")
        if saved != self._weight_decay_scope:
            raise ValueError(
                f"{ckpt_dir} was saved with optimizer.weight_decay_scope={saved!r}, but this "
                f"run uses {self._weight_decay_scope!r}. The scope decides the optimizer's "
                f"param groups, so the saved optimizer state cannot be loaded into this run's "
                f"groups. Resume with weight_decay_scope={saved!r}, or continue from the "
                f"checkpoint with a fresh optimizer: checkpoint.load_path={str(ckpt_dir)!r} "
                f'with checkpoint.exclude_from_loading=["optimizer"] (add "train_state" to '
                f"also restart the step, schedule and data position) and a checkpoint.dir "
                f"that holds no checkpoint, since training auto-resumes from a checkpoint "
                f"there in place of load_path, optimizer state included."
            )

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
        explicit = bool(self.config.load_path)
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

    def _optimizers_saved_in(
        self, dcp_dir: Path, keys: Iterable[str]
    ) -> list[torch.optim.Optimizer]:
        """Return the optimizers saved in ``dcp_dir``; warn if the inner ones are missing."""
        inner = self.checkpointed_optimizers[1:]
        if not inner or _saved_optimizer_param_groups(keys) != len(self.optimizer.param_groups):
            return self.checkpointed_optimizers
        name = type(self.optimizer).__name__
        n_params = sum(p.numel() for opt in inner for g in opt.param_groups for p in g["params"])
        logger.warning(
            f"{dcp_dir} holds only {name}'s own optimizer state: it was saved before "
            f"{name}'s inner optimizer was checkpointed. Resuming {name}'s state alone; the "
            f"{n_params:,} parameters its inner optimizer updates start from fresh moments, "
            f"so this resume is not bit-exact. Checkpoints saved from now on include them."
        )
        return self.checkpointed_optimizers[:1]

    def _cleanup(self) -> None:
        """Remove old checkpoints beyond the retention limit.

        Never removed regardless of retention: the current ``latest`` target,
        the directory the last ``save()`` writes into (``_inflight_dir``), whose
        async flush may still be running, a deferred async checkpoint
        (``_pending_finalize``), and any dynamic-window milestone
        (``CheckpointConfig.is_dynamic_milestone``), so the dense early-dynamics
        checkpoints survive even with a finite ``keep_last_n``. A save drains
        ``_pending_finalize``, which runs this cleanup, only after its own
        flush is dispatched, so ``_inflight_dir`` is what keeps a checkpoint
        that is not among the newest, as after a rewind, from being deleted
        while it is written.

        A directory that NFS briefly keeps non-empty is retried after 0.05, 0.1,
        0.2 and 0.4 s, waits that all the removals of one cleanup share, so a
        cleanup requests at most 0.75 s of retry sleep. One that still cannot be
        removed is logged and left for the next cleanup.
        """
        keep = self.config.keep_last_n
        if keep <= 0:
            return

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
        if self._inflight_dir is not None:
            protected.add(self._inflight_dir.resolve())

        for d in ckpt_dirs:
            if self.config.is_dynamic_milestone(int(d.name.split("_")[1])):
                protected.add(d.resolve())

        to_remove = ckpt_dirs[:-keep] if len(ckpt_dirs) > keep else []
        delays = iter(_REMOVE_RETRY_DELAYS)
        for d in to_remove:
            if d.resolve() in protected:
                continue
            try:
                _remove_tree(d, self._sidecar_lock, delays)
            except OSError as e:
                logger.warning(
                    f"Could not remove old checkpoint {d} ({e}); the next cleanup retries it"
                )
                continue
            logger.info(f"Removed old checkpoint: {d}")
