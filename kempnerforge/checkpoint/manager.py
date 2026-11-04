"""Checkpoint manager for distributed checkpointing.

Uses PyTorch Distributed Checkpoint (DCP) for model and optimizer state,
which supports automatic resharding (save with N GPUs, load with M GPUs).

Non-distributed state (scheduler, dataloader, training meta, RNG) is saved
separately as a torch file written by rank 0 and broadcast from rank 0 on
load. Dataloader and RNG state are per-rank: every rank's payload is gathered
into a versioned envelope at save time, and on an exact resume
(``load(exact_resume=True)``) each rank restores its own entry from the
broadcast (see ``kempnerforge.checkpoint.state``). Any other load restores
rank 0's single copy on every rank.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import stat
import threading
import uuid
from collections.abc import Callable, Container, Iterable
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
from kempnerforge.config.schema import AsyncCheckpointMode, CheckpointConfig

logger = logging.getLogger(__name__)

_TRAIN_STATE_FILE = "train_state.pt"
_METADATA_FILE = "metadata.json"
_DCP_METADATA_FILE = ".metadata"
_OPTIM_STATE_PREFIX = "optimizer.state."
_OPTIM_PARAM_GROUPS_PREFIX = "optimizer.param_groups."
_NEVER_STEPPED_KEY = "optimizer_never_stepped"
_NEVER_STEPPED_PREFIX = f"{_NEVER_STEPPED_KEY}."
_OPTIM_PREFIX = "optimizer."
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


_URI_USERINFO = re.compile(r"(?<=://)[^/?#\s]+@")


def _error_text(error: BaseException) -> str:
    """``error``'s message with the userinfo of every URI in it removed; never raises."""
    try:
        return _URI_USERINFO.sub("", str(error))
    except Exception:
        return f"<unprintable {type(error).__name__}>"


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
    """

    def __init__(
        self,
        config: CheckpointConfig,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        process_group=None,
        pp_rank: int | None = None,
        dp_info: tuple[int, int] | None = None,
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
        self._pending_dataloader_state: dict[str, Any] | None = None
        self._pending_finalize: tuple[int, Path] | None = None
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
        deferred (async mode, after the flush future has resolved).
        """
        latest = self._latest_link()
        tmp_link = latest.with_suffix(".tmp")
        tmp_link.unlink(missing_ok=True)
        tmp_link.symlink_to(ckpt_dir.name)
        tmp_link.rename(latest)
        logger.info(f"Checkpoint committed: {ckpt_dir} (step={step})")
        self._register_committed(step, ckpt_dir)
        self._cleanup()

    def _drain_pending_finalize(self, overwriting: Path | None = None) -> None:
        """Commit a deferred async checkpoint once its flush is durable.

        Invoked after the pending DCP future has been awaited (next save()
        or an explicit wait()/flush). Rank 0 performs the symlink + cleanup;
        other ranks no-op (they never touch the symlink), mirroring save().
        A checkpoint pending in ``overwriting``, the directory a new save's
        flush is rewriting, is dropped uncommitted and unregistered; that
        save's own deferred commit points `latest` at it and registers it.
        """
        if self._pending_finalize is None:
            return
        step, ckpt_dir = self._pending_finalize
        self._pending_finalize = None
        if self._rank == 0 and ckpt_dir != overwriting:
            self._commit_latest(step, ckpt_dir)

    def _sync_barrier(self) -> None:
        """Barrier on the manager's dedicated gloo group.

        Used to fence non-rank-0 ranks behind rank-0's metadata writes
        without sharing the default process group that DCP's async-save
        background thread is concurrently issuing collectives on.
        """
        if dist.is_initialized():
            dist.barrier(group=self._sync_group)

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
        """``meta`` as JSON, plus the live tracking runs when the source can supply them.

        Everything that depends on the source sits inside one guard: it runs on rank 0 alone,
        so a raise here would strand the other ranks. A bad source costs the ``tracking`` key.
        """
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

    def _gather_rank_states(self, dataloader: Any | None) -> dict[str, Any]:
        """Collect every rank's own loader/RNG payload into a versioned envelope.

        Collective (all ranks call it at the same point in ``save()``). Uses
        the dedicated gloo group so the gather cannot interleave with DCP
        collectives an in-flight async flush is issuing on the default group
        from its background thread. Entries are keyed by each payload's own
        global rank rather than by gather position, so the envelope is correct
        under any group ordering. Each rank's ``dp_info`` rides along to
        record the data-parallel layout.
        """
        mine = (self._rank, self._dp_info, capture_rank_state(dataloader))
        gathered: list[Any] = [mine]
        if dist.is_initialized():
            gathered = [None] * self._world_size
            dist.all_gather_object(gathered, mine, group=self._sync_group)
        entries = {int(rank): state for rank, _, state in gathered}
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
        call it. A topology or schema mismatch refuses the load unless
        ``checkpoint.ignore_rank_state_mismatch`` downgrades it to a warning
        that discards the saved loader/RNG state. The envelope check reads
        only state every rank shares, so all ranks restore, refuse, or discard
        together. A legacy checkpoint (no envelope) keeps its old behavior:
        rank 0's single copy restores on every rank.
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

        dcp_dir = ckpt_dir / f"pp{self._pp_rank}" if self._pp_rank is not None else ckpt_dir
        dcp_dir.mkdir(parents=True, exist_ok=True)

        dcp_state: dict[str, Any] = {"model": get_model_state_dict(self.model)}
        if any(optimizer.state for optimizer in self.checkpointed_optimizers):
            optim_state = get_optimizer_state_dict(self.model, self.checkpointed_optimizers)
            dcp_state["optimizer"] = optim_state
            dcp_state[_NEVER_STEPPED_KEY] = _never_stepped_record(optim_state)
        self._async_ckpt.save(
            dcp_state, checkpoint_id=str(dcp_dir), process_group=self._process_group
        )

        sync_mode = self.config.async_mode == AsyncCheckpointMode.disabled

        self._drain_pending_finalize(overwriting=ckpt_dir)

        per_rank = self._gather_rank_states(dataloader)

        if self._rank == 0:
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

        if sync_mode:
            if self._rank == 0:
                self._commit_latest(step, ckpt_dir)
        else:
            self._pending_finalize = (step, ckpt_dir)

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
        exact_resume: bool = False,
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
            exact_resume: True only when the run resumes from its own
                checkpoint (auto-resume). Each rank then restores its own
                dataloader/RNG state from the per-rank envelope, which must
                match this run's world size and data-parallel layout. Every
                rank must pass the same value. False (a ``load_path`` warm
                start, evaluation, any other load) restores the single-copy
                ``rng``/``dataloader`` keys (rank 0's) on every rank with no
                topology check.

        Returns:
            Tuple of (step, tokens_seen, extra) where extra contains any
            additional keys saved via ``build_train_state(extra=...)``.
        """
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

        dcp_dir = ckpt_dir / f"pp{self._pp_rank}" if self._pp_rank is not None else ckpt_dir

        load_model = exclude_keys is None or "model" not in exclude_keys
        load_optim = exclude_keys is None or "optimizer" not in exclude_keys

        saved_keys = (
            dcp.FileSystemReader(str(dcp_dir)).read_metadata().state_dict_metadata
            if load_optim
            else {}
        )
        fresh_optimizer = load_optim and not _holds_optimizer(saved_keys)
        if fresh_optimizer:
            load_optim = False

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

        if fresh_optimizer and step > 0:
            logger.warning(
                f"{dcp_dir} holds no optimizer state at step {step}: resuming with a fresh "
                f"optimizer, so its moments restart from zero and this resume does not "
                f"continue the saved run exactly."
            )
        elif fresh_optimizer:
            logger.info(f"{dcp_dir} holds no optimizer state; the optimizer starts fresh")
        return step, tokens_seen, extra

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

        Never removed regardless of retention: the current ``latest`` target
        and the in-flight async checkpoint (``_pending_finalize``) — pruning
        either would let a crash strand resume with no loadable checkpoint —
        and any dynamic-window milestone (``CheckpointConfig.is_dynamic_milestone``),
        so the dense early-dynamics checkpoints survive even with a finite
        ``keep_last_n``.
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

        for d in ckpt_dirs:
            if self.config.is_dynamic_milestone(int(d.name.split("_")[1])):
                protected.add(d.resolve())

        to_remove = ckpt_dirs[:-keep] if len(ckpt_dirs) > keep else []
        for d in to_remove:
            if d.resolve() in protected:
                continue
            try:
                with self._sidecar_lock:
                    shutil.rmtree(d)
            except OSError as e:
                logger.warning(
                    f"Could not remove old checkpoint {d} ({e}); the next cleanup retries it"
                )
                continue
            logger.info(f"Removed old checkpoint: {d}")
