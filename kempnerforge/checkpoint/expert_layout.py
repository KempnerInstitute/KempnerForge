"""Expert-parallel experts under their global identity in a DCP state dict.

``apply_expert_parallel`` leaves each EP rank ``num_experts / ep`` experts of every MoE
layer: an unpacked layer keeps them in a ``ModuleList`` numbered from 0, and a packed
layer keeps their rows of each packed weight. Every EP rank's state dict then names and
shapes its experts as if they were the layer's only ones, and DCP, which writes one copy
of each name and offset, would save one rank's experts in place of all of them.

``ExpertLayout`` translates a state dict so that each rank presents its experts as part
of all ``num_experts``: an unpacked expert under its global index, and a packed weight as
a DTensor over ``ep`` and the FSDP mesh whose rows cover every expert. A checkpoint then
holds every expert in the layout an ``ep = 1`` run saves, so DCP loads it at any ``ep``.
The translated tensors share storage with the rank's own, so nothing is copied or
gathered, and a load into them loads the model and optimizer.

``check_saved_experts`` refuses a checkpoint that holds fewer experts than the model, as
one saved with ``ep > 1`` before this layout existed does.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Placement, Shard
from torch.distributed.tensor.placement_types import _StridedShard

from kempnerforge.model.moe import MoEMLP

_WRAPPERS = frozenset({"_checkpoint_wrapped_module", "_orig_mod"})
_EXPERT_NAME = re.compile(r"(?:(?P<layer>.+?)\.)?experts\.(?P<index>\d+)\.(?P<rest>.+)")


def _join(*parts: str) -> str:
    """``parts`` joined with dots, leaving out empty ones, as module names are joined."""
    return ".".join(part for part in parts if part)


def moe_layers(model: torch.nn.Module) -> Iterator[tuple[str, MoEMLP]]:
    """Each ``MoEMLP`` in ``model`` with the name its state-dict entries start with.

    The name is empty when ``model`` is itself the ``MoEMLP``. It leaves out the
    activation-checkpoint and ``torch.compile`` wrappers, as ``get_model_state_dict`` does.
    """
    for name, module in model.named_modules():
        if isinstance(module, MoEMLP):
            yield ".".join(part for part in name.split(".") if part not in _WRAPPERS), module


def expert_layout(model: torch.nn.Module) -> ExpertLayout | None:
    """The layout of ``model``'s expert-parallel layers; None when no layer splits its experts."""
    layers = [(name, moe) for name, moe in moe_layers(model) if moe.ep_world_size > 1]
    return ExpertLayout(layers) if layers else None


def _packed_weights(moe: MoEMLP) -> tuple[str, ...]:
    """A packed layer's expert weights: up and down, and the gate under SwiGLU."""
    return ("up_w", "down_w", "gate_w") if moe._is_swiglu else ("up_w", "down_w")


def _rows_after_ep(placements: Sequence[Placement], ep: int, name: str) -> tuple[Placement, ...]:
    """FSDP's placements of one EP rank's rows, applied after ``ep`` splits the rows.

    ``Shard(0)`` over an FSDP dimension becomes ``_StridedShard(0, split_factor=ep)``,
    which shards each EP rank's block of rows rather than all of them. Other placements
    shard other dimensions or replicate, which ``ep`` leaves as they are.
    """
    result: list[Placement] = []
    for placement in placements:
        if isinstance(placement, _StridedShard):
            raise ValueError(f"{name} is already strided over its rows ({placement})")
        result.append(_StridedShard(0, split_factor=ep) if placement.is_shard(0) else placement)
    return tuple(result)


def _root_mesh(mesh: DeviceMesh) -> DeviceMesh:
    """The mesh that ``mesh`` was sliced from.

    Raises:
        ValueError: This torch's ``DeviceMesh`` has no ``_get_root_mesh``, which arrived
            in torch 2.10 and which a packed expert-parallel checkpoint needs.
    """
    get_root = getattr(mesh, "_get_root_mesh", None)
    if get_root is None:
        raise ValueError(
            "Checkpoints of packed expert-parallel layers need torch 2.10 or newer: torch "
            f"{torch.__version__} has no DeviceMesh._get_root_mesh, which joins the EP mesh "
            "with the FSDP mesh that places each rank's rows"
        )
    return get_root()


def _check_placement(view: DTensor, tensor: torch.Tensor, moe: MoEMLP, name: str) -> None:
    """Refuse ``view`` unless DCP would save its rows where ``tensor``'s rows belong.

    This rank's rows of all ``num_experts`` start at ``local_expert_start``, then
    wherever FSDP's own shard of the rank's block starts. An empty shard has no
    position, so only its size is checked. Older torch versions lay out a strided
    shard otherwise, and the check stops such a save before it writes anything.

    Raises:
        ValueError: DCP would save the view's rows at another size or offset.
    """
    local = view.to_local()
    chunk = view.__create_chunk_list__()[0]
    own = (
        tuple(tensor.__create_chunk_list__()[0].offsets)
        if isinstance(tensor, DTensor)
        else (0,) * local.dim()
    )
    expected = (moe.local_expert_start + own[0], *own[1:])
    misplaced = local.numel() > 0 and tuple(chunk.offsets) != expected
    if tuple(chunk.sizes) != tuple(local.shape) or misplaced:
        raise ValueError(
            f"{name}: DCP would save this rank's {tuple(local.shape)} rows as "
            f"{tuple(chunk.sizes)} at offset {tuple(chunk.offsets)} among all "
            f"{moe.num_experts} experts, but they belong at {expected}; this torch lays "
            "out the view's strided shard other than FSDP2 does, so the save is refused"
        )


def _contiguous_strides(shape: torch.Size) -> tuple[int, ...]:
    """The strides of a contiguous tensor of ``shape``."""
    strides = [1] * len(shape)
    for dim in range(len(shape) - 2, -1, -1):
        strides[dim] = strides[dim + 1] * shape[dim + 1]
    return tuple(strides)


class ExpertLayout:
    """Translates a DCP state dict between one EP rank's experts and all of the layer's.

    ``to_global`` translates the ``"model"`` and ``"optimizer"`` entries of a state
    dict, and the names in every other entry that maps parameter names to values, such
    as the never-stepped record. ``to_local`` undoes it once a load has filled the
    result, so one layout serves one save or one load.

    Args:
        layers: The split ``MoEMLP`` layers, each with the name its state-dict entries
            start with, as ``moe_layers`` gives them.
    """

    def __init__(self, layers: list[tuple[str, MoEMLP]]) -> None:
        self._unpacked = {name: moe for name, moe in layers if not moe.packed_experts}
        self._packed = {
            _join(name, weight): (moe, weight)
            for name, moe in layers
            if moe.packed_experts
            for weight in _packed_weights(moe)
        }
        self._local_names: dict[str, str] = {}
        self._originals: dict[int, tuple[DTensor, torch.Tensor]] = {}
        self._group_params: list[list[str]] = []
        self._meshes: dict[tuple[int, tuple[str, ...]], tuple[DeviceMesh, DeviceMesh]] = {}

    def to_global(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """``state`` with each rank's experts placed among all of the layer's experts.

        Returns a new dict and leaves ``state`` as it is. Each tensor in the result is
        the one it replaces or a view of it, so a load into the result loads them.

        Raises:
            ValueError: The model state lacks the expert weights of a split layer, or the
                optimizer holds a tensor for a packed weight in a shape other than the
                weight's, so the checkpoint could not hold every rank's experts.
        """
        result: dict[str, Any] = {}
        for key, value in state.items():
            if key == "model":
                result[key] = self._model_to_global(value)
            elif key == "optimizer":
                result[key] = self._optimizer_to_global(value)
            elif isinstance(value, Mapping):
                result[key] = {self._global_name(name): item for name, item in value.items()}
            else:
                result[key] = value
        return result

    def to_local(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Undo ``to_global`` on the state dict that a load filled.

        Names go back to this rank's, and each view ``to_global`` made goes back to the
        tensor it views, which holds what the load wrote. Each optimizer param group
        lists this rank's parameters again and keeps the loaded settings.
        """
        result: dict[str, Any] = {}
        for key, value in state.items():
            if key == "model":
                result[key] = {
                    self._local_name(name): self._original(item) for name, item in value.items()
                }
            elif key == "optimizer":
                result[key] = self._optimizer_to_local(value)
            elif isinstance(value, Mapping):
                result[key] = {self._local_name(name): item for name, item in value.items()}
            else:
                result[key] = value
        return result

    def _model_to_global(self, model_state: Mapping[str, Any]) -> dict[str, Any]:
        """The model state with every split expert weight under its global identity."""
        missing = sorted(self._weight_names() - model_state.keys())
        if missing:
            raise ValueError(
                f"The model state dict has no entry for {len(missing)} expert weights of "
                f"layers that expert parallelism splits, such as {missing[0]!r}, so a "
                "checkpoint could not hold every rank's experts"
            )
        return {
            self._global_name(name): (
                self._spread(value, name, name) if name in self._packed else value
            )
            for name, value in model_state.items()
        }

    def _optimizer_to_global(self, optim_state: Mapping[str, Any]) -> dict[str, Any]:
        """The optimizer state with every split expert's state under its global identity."""
        state: dict[str, Any] = {}
        for name, entry in optim_state["state"].items():
            if name in self._packed:
                state[name] = {
                    key: self._spread_state(item, name, key) for key, item in entry.items()
                }
            else:
                state[self._global_name(name)] = entry
        groups = optim_state["param_groups"]
        self._group_params = [list(group["params"]) for group in groups]
        return {
            **optim_state,
            "state": state,
            "param_groups": [
                {**group, "params": self._every_rank(group["params"])} for group in groups
            ],
        }

    def _optimizer_to_local(self, optim_state: Mapping[str, Any]) -> dict[str, Any]:
        """The loaded optimizer state under this rank's names and tensors."""
        state = {
            self._local_name(name): (
                {key: self._original(item) for key, item in entry.items()}
                if isinstance(entry, Mapping)
                else entry
            )
            for name, entry in optim_state["state"].items()
        }
        groups = [
            {**group, "params": params}
            for group, params in zip(optim_state["param_groups"], self._group_params, strict=True)
        ]
        return {**optim_state, "state": state, "param_groups": groups}

    def _weight_names(self) -> set[str]:
        """Every split expert weight's name in this rank's model state dict."""
        names = set(self._packed)
        for layer, moe in self._unpacked.items():
            names.update(
                _join(layer, "experts", param) for param, _ in moe.experts.named_parameters()
            )
        return names

    def _parse(self, name: str) -> tuple[str, MoEMLP, int, str] | None:
        """The layer, its module, the expert index and the rest of a split expert's name."""
        match = _EXPERT_NAME.fullmatch(name)
        if match is None:
            return None
        layer = match["layer"] or ""
        moe = self._unpacked.get(layer)
        if moe is None:
            return None
        return layer, moe, int(match["index"]), match["rest"]

    def _global_name(self, name: str) -> str:
        """``name`` with a split expert's local index replaced by its global one."""
        parsed = self._parse(name)
        if parsed is None:
            return name
        layer, moe, index, rest = parsed
        if index >= moe.num_local_experts:
            raise ValueError(
                f"{name} names local expert {index}, but {layer or 'the model'} holds "
                f"{moe.num_local_experts} experts on this rank"
            )
        global_name = _join(layer, "experts", str(moe.local_expert_start + index), rest)
        self._local_names[global_name] = name
        return global_name

    def _local_name(self, name: str) -> str:
        """The name ``_global_name`` gave ``name`` in place of; ``name`` if it gave none."""
        return self._local_names.get(name, name)

    def _every_rank(self, names: list[str]) -> list[str]:
        """``names`` with each split layer's expert names given for every EP rank's experts.

        Each EP rank's optimizer lists its own experts, and the checkpoint keeps one
        rank's list, so each rank lists all of them in global order where its own begin.
        That is the same list on every rank, and the list an ``ep = 1`` run saves.
        """
        own: dict[str, list[tuple[int, str]]] = {}
        for name in names:
            parsed = self._parse(name)
            if parsed is not None:
                own.setdefault(parsed[0], []).append((parsed[2], parsed[3]))
        result: list[str] = []
        for name in names:
            parsed = self._parse(name)
            if parsed is None:
                result.append(name)
                continue
            layer, moe, _, _ = parsed
            entries = own.pop(layer, None)
            if entries is None:
                continue
            count = moe.num_local_experts
            result += [
                _join(layer, "experts", str(rank * count + index), rest)
                for rank in range(moe.ep_world_size)
                for index, rest in entries
            ]
        return result

    def _spread_state(self, item: Any, name: str, key: str) -> Any:
        """An optimizer state entry of packed weight ``name``, spread like the weight."""
        if isinstance(item, torch.Tensor) and item.dim() > 0:
            return self._spread(item, name, f"the optimizer state {key!r} of {name}")
        return item

    def _spread(self, tensor: torch.Tensor, name: str, what: str) -> DTensor:
        """``tensor``, this rank's rows of packed weight ``name``, as a DTensor of every row.

        Rank ``r`` of the EP mesh holds rows ``r * num_local_experts`` onward, so ``ep``
        shards the rows first and FSDP shards each EP rank's block of rows, as FSDP2 lays
        out a parameter that tensor parallelism shards on the same dimension: FSDP's
        ``Shard(0)`` becomes ``_StridedShard(0, split_factor=ep)``. The view shares
        ``tensor``'s storage.
        """
        moe, weight = self._packed[name]
        shape = getattr(moe, weight).shape
        if tensor.shape != shape:
            raise ValueError(
                f"{what} has shape {tuple(tensor.shape)}, not {tuple(shape)}, the shape of "
                f"this rank's rows of {name}, so it cannot be placed among all "
                f"{moe.num_experts} experts"
            )
        if moe.ep_mesh is None:
            raise ValueError(
                f"{name} is split over ep={moe.ep_world_size}, but its layer records no "
                "ep_mesh to place its rows by; apply_expert_parallel records it"
            )
        global_shape = torch.Size((moe.num_experts, *shape[1:]))
        if isinstance(tensor, DTensor):
            mesh = self._mesh_with_ep(moe.ep_mesh, tensor.device_mesh, name)
            placements = (*_rows_after_ep(tensor.placements, moe.ep_world_size, name), Shard(0))
            local = tensor.to_local()
        else:
            mesh, placements, local = moe.ep_mesh, (Shard(0),), tensor
        view = DTensor.from_local(
            local,
            mesh,
            placements,
            run_check=False,
            shape=global_shape,
            stride=_contiguous_strides(global_shape),
        )
        _check_placement(view, tensor, moe, name)
        self._originals[id(view)] = (view, tensor)
        return view

    def _mesh_with_ep(self, ep_mesh: DeviceMesh, mesh: DeviceMesh, name: str) -> DeviceMesh:
        """The mesh of ``mesh``'s dimensions, which shard a rank's rows, followed by ``ep``.

        ``mesh`` must be those dimensions of the mesh that ``ep_mesh`` comes from. It is
        compared as ``DeviceMesh`` compares meshes, by ranks, layout and names, since a
        DTensor can hold an equal mesh that is another object: DTensor and FSDP2 reuse the
        meshes of earlier, equal layouts.
        """
        root = _root_mesh(ep_mesh)
        dims = mesh.mesh_dim_names or ()
        try:
            if not dims or "ep" in dims:
                raise ValueError("it has no dimension other than ep")
            key = (id(root), dims)
            if key not in self._meshes:
                self._meshes[key] = (root[dims], root[(*dims, "ep")])
            rows, joined = self._meshes[key]
            if rows != mesh:
                raise ValueError(f"those dimensions of {root} are {rows}")
        except (KeyError, ValueError) as e:
            raise ValueError(
                f"{name} is sharded over {mesh}, which cannot be joined with the EP mesh "
                f"{ep_mesh} that its experts are split over ({e}), so its rows cannot be "
                "placed among all the experts"
            ) from e
        return joined

    def _original(self, item: Any) -> Any:
        """The tensor that ``item`` views if ``to_global`` made it; else ``item``."""
        made = self._originals.get(id(item))
        return made[1] if made is not None and made[0] is item else item


def _held_experts(saved: Mapping[str, Any], layer: str, moe: MoEMLP) -> int | None:
    """How many experts of ``layer`` the DCP index ``saved`` holds; None if it holds none."""
    if moe.packed_experts:
        size = getattr(saved.get(_join("model", layer, "up_w")), "size", None)
        return int(size[0]) if size else None
    head = _join("model", layer, "experts.")
    indices = {key[len(head) :].partition(".")[0] for key in saved if key.startswith(head)}
    held = {index for index in indices if index.isdigit()}
    return len(held) if held else None


def check_saved_experts(model: torch.nn.Module, saved: Mapping[str, Any], where: object) -> None:
    """Refuse a checkpoint whose MoE layers hold fewer experts than ``model``'s.

    ``saved`` is the checkpoint's DCP index, its ``state_dict_metadata``. A checkpoint
    saved with ``ep > 1`` before the experts had a global identity holds
    ``num_experts / ep`` experts of each layer: every EP rank wrote its own under the same
    names, and the checkpoint kept one rank's. Loading it at the same ``ep`` would give
    every rank that rank's experts, and at another ``ep`` it fails on the experts it
    lacks, so it is refused here with what it holds. A layer the checkpoint holds no
    expert of is left to the load.

    Raises:
        ValueError: A layer of ``model`` has more experts than the checkpoint holds for it.
    """
    short = [
        (layer, moe, held)
        for layer, moe in moe_layers(model)
        if (held := _held_experts(saved, layer, moe)) is not None and held < moe.num_experts
    ]
    if not short:
        return
    layer, moe, held = short[0]
    total = moe.num_experts
    holds = (
        f"{_join(layer, 'up_w')} has {held} {'row' if held == 1 else 'rows'}"
        if moe.packed_experts
        else f"{layer or 'the model'} holds experts 0-{held - 1}"
        if held > 1
        else f"{layer or 'the model'} holds expert 0"
    )
    layers = "MoE layer" if len(short) == 1 else "MoE layers"
    saved_with = f"ep={total // held}" if total % held == 0 else "expert parallelism"
    raise ValueError(
        f"{where} holds {held} of the {total} experts of {len(short)} {layers} ({holds}), "
        "so it cannot restore this model's experts. That is what a checkpoint saved with "
        f"{saved_with} holds when it predates issue #202's fix: every EP rank saved its "
        "experts under the same names and the checkpoint kept one rank's, so the other "
        "experts' weights were never written. Load a checkpoint saved with ep=1 or after "
        "the fix, or start a new run."
    )
