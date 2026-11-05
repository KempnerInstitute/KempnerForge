"""Expert-parallel checkpoints on CPU over gloo: every expert saved, and loaded at any ``ep``.

Each case builds the model with ``build_parallel_model``, shifts each EP rank's experts
by a different amount so no two ranks' are equal, takes an AdamW step on a batch every
rank shares, and saves it with ``CheckpointManager``. The reference is every rank's live
weights and AdamW moments, gathered under the unsharded model's names, each rank's
experts at their global index. Ranks that differ only in ``dp_replicate`` or ``tp`` hold
the same experts.
"""

from __future__ import annotations

import re
import socket
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor

from kempnerforge.checkpoint.manager import CheckpointManager
from kempnerforge.config.model import ModelConfig
from kempnerforge.config.schema import AsyncCheckpointMode, CheckpointConfig
from kempnerforge.distributed.parallel import build_parallel_model

SEQ = 16
STEP = 1
FP32 = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)
UNPACKED_EXPERT = re.compile(r"(?P<layer>.*\.mlp)\.experts\.(?P<index>\d+)\.(?P<rest>.*)")
PACKED_EXPERT = (".mlp.up_w", ".mlp.down_w", ".mlp.gate_w")
MOMENTS = ("exp_avg", "exp_avg_sq")
PACKED = {"moe_packed_experts": True}
DP_SHARD2_EP2 = (("dp_shard", "ep"), (2, 2))
EP4 = (("dp_shard", "ep"), (1, 4))
HSDP_EP2 = (("dp_replicate", "dp_shard", "ep"), (2, 1, 2))

LAYOUTS = {
    "ep2": (("dp_shard", "ep"), (1, 2), {}),
    "ep2_packed": (("dp_shard", "ep"), (1, 2), PACKED),
    "dp_shard2_ep2": (*DP_SHARD2_EP2, {}),
    "dp_shard2_ep2_packed": (*DP_SHARD2_EP2, PACKED),
    "dp_shard2_ep2_packed_uneven": (*DP_SHARD2_EP2, {**PACKED, "num_experts": 6}),
    "dp_shard2_ep2_packed_one_row": (*DP_SHARD2_EP2, {**PACKED, "num_experts": 2}),
    "dp_replicate2_ep2": (*HSDP_EP2, {}),
    "dp_replicate2_ep2_packed": (*HSDP_EP2, PACKED),
    "ep2_tp2": (("dp_shard", "ep", "tp"), (1, 2, 2), {}),
    "ep2_tp2_packed": (("dp_shard", "ep", "tp"), (1, 2, 2), PACKED),
    "ep4_packed_gelu": (*EP4, {**PACKED, "activation": "gelu"}),
}


def _free_port() -> int:
    """An unused local TCP port for the gloo rendezvous."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _init(rank: int, world: int, port: int) -> None:
    """Join a ``world``-rank gloo group with one intra-op thread per rank.

    The group's timeout ends a stuck collective after two minutes, so a deadlock cannot
    hold the test job for the default half hour.
    """
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world,
        init_method=f"tcp://127.0.0.1:{port}",
        timeout=timedelta(seconds=120),
    )


def _spawn(fn, names: tuple[str, ...], sizes: tuple[int, ...], *args) -> None:
    """Run ``fn(rank, world, port, names, sizes, *args)`` on one gloo rank per mesh slot."""
    world = 1
    for size in sizes:
        world *= size
    mp.spawn(fn, args=(world, _free_port(), names, sizes, *args), nprocs=world)


def _config(overrides: dict) -> ModelConfig:
    """A two-layer model whose layers are both MoE, with four experts unless overridden."""
    fields = {
        "dim": 32,
        "n_layers": 2,
        "n_heads": 2,
        "vocab_size": 64,
        "max_seq_len": SEQ,
        "num_experts": 4,
        "moe_top_k": 2,
        **overrides,
    }
    return ModelConfig(**fields)


def _is_expert(name: str) -> bool:
    return UNPACKED_EXPERT.fullmatch(name) is not None or name.endswith(PACKED_EXPERT)


def _build(
    mesh: DeviceMesh | None, config: ModelConfig, seed: int
) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    """The parallel model built from ``seed``, and a fresh AdamW over it."""
    torch.manual_seed(seed)
    model = build_parallel_model(
        config, torch.device("cpu"), mesh, mp_policy=FP32, param_dtype=torch.float32
    )
    return model, torch.optim.AdamW(model.parameters(), lr=1e-2)


def _train(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, mesh: DeviceMesh | None
) -> None:
    """Shift each EP rank's experts by a different amount, then take one AdamW step.

    Every rank uses the same batch. Without a mesh nothing is shifted.
    """
    if mesh is not None:
        shift = mesh.get_local_rank("ep") + 1.0
        with torch.no_grad():
            for name, p in model.named_parameters():
                if _is_expert(name):
                    (p.to_local() if isinstance(p, DTensor) else p).add_(shift)
    config = model.config
    rows = torch.randint(
        0, config.vocab_size, (2, SEQ + 1), generator=torch.Generator().manual_seed(7)
    )
    logits = model(rows[:, :-1])
    F.cross_entropy(logits.flatten(0, 1).float(), rows[:, 1:].flatten()).backward()
    optimizer.step()
    optimizer.zero_grad()


def _placed(
    key: str, name: str, full: torch.Tensor, mesh: DeviceMesh | None, num_experts: int
) -> dict[str, torch.Tensor]:
    """``full`` under the unsharded model's names; a routed expert's gathered over ``ep``.

    ``key`` is parameter ``name`` or one of its AdamW moments.
    """
    ep = mesh["ep"].size() if mesh is not None and "ep" in mesh.mesh_dim_names else 1
    if ep == 1 or not _is_expert(name):
        return {key: full}
    parts = [torch.empty_like(full) for _ in range(ep)]
    dist.all_gather(parts, full.contiguous(), group=mesh["ep"].get_group())
    match = UNPACKED_EXPERT.fullmatch(name)
    if match is None:
        return {key: torch.cat(parts)}
    layer, index, rest = match["layer"], int(match["index"]), match["rest"] + key[len(name) :]
    count = num_experts // ep
    return {
        f"{layer}.experts.{rank * count + index}.{rest}": part for rank, part in enumerate(parts)
    }


def _gathered(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, mesh: DeviceMesh | None
) -> dict[str, torch.Tensor]:
    """Every weight and AdamW moment, full, under the unsharded model's names.

    DTensors are gathered over their mesh and the routed experts over ``ep`` too, so
    every rank must call this, and every rank gets the whole model.
    """
    num_experts = model.config.num_experts
    result: dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        state = optimizer.state.get(p, {})
        tensors = {name: p.detach(), **{f"{name}.{m}": state[m] for m in MOMENTS if m in state}}
        for key, t in tensors.items():
            full = t.full_tensor() if isinstance(t, DTensor) else t
            result.update(_placed(key, name, full, mesh, num_experts))
    return result


def _mismatches(got: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> list[str]:
    """The names whose tensors differ, or that only one side holds."""
    wrong = sorted(got.keys() ^ expected.keys())
    wrong += [key for key in expected if key in got and not torch.equal(got[key], expected[key])]
    return wrong


def _save(rank, world, port, names, sizes, overrides, root, async_mode) -> None:
    """Train one step and save; then load into a model built from another seed, at the same layout.

    Rank 0 writes the reference and what the load gave.
    """
    _init(rank, world, port)
    try:
        mesh = init_device_mesh("cpu", sizes, mesh_dim_names=names)
        config = _config(overrides)
        model, optimizer = _build(mesh, config, seed=0)
        _train(model, optimizer, mesh)
        reference = _gathered(model, optimizer, mesh)
        ckpt = CheckpointConfig(dir=str(Path(root) / "ckpt"), async_mode=async_mode)
        manager = CheckpointManager(ckpt, model, optimizer)
        manager.save(STEP)
        manager.wait()
        if rank == 0:
            torch.save(reference, Path(root) / "reference.pt")
        resumed, resumed_optimizer = _build(mesh, config, seed=1)
        CheckpointManager(ckpt, resumed, resumed_optimizer).load()
        loaded = _gathered(resumed, resumed_optimizer, mesh)
        if rank == 0:
            torch.save(loaded, Path(root) / "resumed.pt")
    finally:
        dist.destroy_process_group()


def _saved(step_dir: Path, scratch: Path) -> dict[str, torch.Tensor]:
    """Every weight and AdamW moment the checkpoint at ``step_dir`` holds, under its names."""
    dcp_to_torch_save(step_dir, scratch / "saved.pt")
    state = torch.load(scratch / "saved.pt", weights_only=False)
    saved = dict(state["model"])
    for name, entry in state.get("optimizer", {}).get("state", {}).items():
        saved.update({f"{name}.{m}": entry[m] for m in MOMENTS if m in entry})
    return saved


def _held_experts(step_dir: Path, config: ModelConfig) -> dict[str, list[int]]:
    """For each MoE layer, the expert indices its checkpoint index holds, or its rows."""
    index = FileSystemReader(str(step_dir)).read_metadata().state_dict_metadata
    held = {}
    for layer in range(config.n_layers):
        prefix = f"model.layers.{layer}.mlp"
        if config.moe_packed_experts:
            held[prefix] = list(range(index[f"{prefix}.up_w"].size[0]))
        else:
            pattern = re.compile(rf"{re.escape(prefix)}\.experts\.(\d+)\..*")
            held[prefix] = sorted({int(m[1]) for k in index if (m := pattern.fullmatch(k))})
    return held


@pytest.fixture(scope="module")
def saved_layout(tmp_path_factory):
    """Save each layout once, with its same-layout load; the tests read the files."""
    roots: dict[tuple[str, str], Path] = {}

    def run(layout: str, async_mode: str = "disabled") -> Path:
        key = (layout, async_mode)
        if key not in roots:
            root = tmp_path_factory.mktemp(layout)
            names, sizes, overrides = LAYOUTS[layout]
            _spawn(_save, names, sizes, overrides, str(root), async_mode)
            roots[key] = root
        return roots[key]

    return run


def _step_dir(root: Path) -> Path:
    return root / "ckpt" / f"step_{STEP}"


class TestCheckpointHoldsEveryExpert:
    @pytest.mark.parametrize("layout", list(LAYOUTS))
    def test_every_expert_is_saved_once(self, saved_layout, layout, tmp_path):
        """The index holds every expert of every layer, each equal to its rank's weights.

        AdamW's moments are checked with the weights, so a resume restores each rank's
        optimizer state as well.
        """
        root = saved_layout(layout)
        config = _config(LAYOUTS[layout][2])
        experts = list(range(config.num_experts))
        assert _held_experts(_step_dir(root), config) == {
            f"model.layers.{layer}.mlp": experts for layer in range(config.n_layers)
        }
        reference = torch.load(root / "reference.pt")
        saved = _saved(_step_dir(root), tmp_path)
        assert _mismatches({k: saved[k] for k in reference if k in saved}, reference) == []

    def test_async_save_holds_every_expert(self, saved_layout, tmp_path):
        """An async save stages the packed views to CPU and writes every expert's rows."""
        root = saved_layout("dp_shard2_ep2_packed", async_mode=AsyncCheckpointMode.async_.value)
        reference = torch.load(root / "reference.pt")
        saved = _saved(_step_dir(root), tmp_path)
        assert _mismatches({k: saved[k] for k in reference if k in saved}, reference) == []


class TestResumeAtTheSameLayout:
    @pytest.mark.parametrize("layout", list(LAYOUTS))
    def test_each_rank_resumes_its_own_experts(self, saved_layout, layout):
        """A model built from another seed holds every rank's saved weights and moments."""
        root = saved_layout(layout)
        reference = torch.load(root / "reference.pt")
        assert _mismatches(torch.load(root / "resumed.pt"), reference) == []
