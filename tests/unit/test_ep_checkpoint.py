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
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from convert_checkpoint import dcp_to_hf
from generate import _load_model

import kempnerforge.checkpoint.expert_layout as expert_layout_module
import kempnerforge.checkpoint.manager as manager_module
from kempnerforge.checkpoint.manager import CheckpointManager
from kempnerforge.config.model import ModelConfig
from kempnerforge.config.optimizer import OptimizerConfig
from kempnerforge.config.schema import AsyncCheckpointMode, CheckpointConfig, JobConfig
from kempnerforge.distributed.parallel import build_parallel_model
from kempnerforge.training.optimizer import build_optimizer

SEQ = 16
STEP = 1
FP32 = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)
UNPACKED_EXPERT = re.compile(r"(?P<layer>.*\.mlp)\.experts\.(?P<index>\d+)\.(?P<rest>.*)")
PACKED_EXPERT = (".mlp.up_w", ".mlp.down_w", ".mlp.gate_w")
MOMENTS = ("exp_avg", "exp_avg_sq")
LR_PATTERNS = {"layer0": ["layers.0.*"], "layer1": ["layers.1.*"]}
LR_MULTIPLIERS = {"layer0": 2.0, "layer1": 0.5}
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


def _save(rank, world, port, names, sizes, overrides, root, async_mode, unfixed) -> None:
    """Train one step and save; then load into a model built from another seed, at the same layout.

    Rank 0 writes the reference and what the load gave. With ``unfixed`` the save leaves
    the experts as each rank holds them, as before the fix.
    """
    _init(rank, world, port)
    try:
        if unfixed:
            manager_module.expert_layout = lambda model: None
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
        if unfixed:
            return
        resumed, resumed_optimizer = _build(mesh, config, seed=1)
        CheckpointManager(ckpt, resumed, resumed_optimizer).load()
        loaded = _gathered(resumed, resumed_optimizer, mesh)
        if rank == 0:
            torch.save(loaded, Path(root) / "resumed.pt")
    finally:
        dist.destroy_process_group()


def _load(rank, world, port, names, sizes, overrides, step_dir, out) -> None:
    """Load ``step_dir`` at this layout; rank 0 writes what every rank's model then holds."""
    _init(rank, world, port)
    try:
        mesh = init_device_mesh("cpu", sizes, mesh_dim_names=names)
        model, optimizer = _build(mesh, _config(overrides), seed=1)
        CheckpointManager(CheckpointConfig(dir=str(Path(out).parent)), model, optimizer).load(
            path=step_dir
        )
        loaded = _gathered(model, optimizer, mesh)
        if rank == 0:
            torch.save(loaded, out)
    finally:
        dist.destroy_process_group()


def _save_with_lr_groups(rank, world, port, names, sizes, overrides, root) -> None:
    """Save and resume with per-module learning rates, which split the experts among groups.

    Rank 0 writes the reference, what the resume gave, and both optimizers' groups.
    """
    _init(rank, world, port)
    try:
        mesh = init_device_mesh("cpu", sizes, mesh_dim_names=names)
        config = _config(overrides)
        optimizer_config = OptimizerConfig(lr=1e-2, fused=False, lr_multipliers=LR_MULTIPLIERS)
        ckpt = CheckpointConfig(dir=str(Path(root) / "ckpt"))
        built = []
        for seed in (0, 1):
            torch.manual_seed(seed)
            model = build_parallel_model(
                config, torch.device("cpu"), mesh, mp_policy=FP32, param_dtype=torch.float32
            )
            optimizer = build_optimizer(model, optimizer_config, module_patterns=LR_PATTERNS)
            manager = CheckpointManager(
                ckpt, model, optimizer, lr_multipliers=LR_MULTIPLIERS, module_patterns=LR_PATTERNS
            )
            built.append((model, optimizer, manager))
        (model, optimizer, manager), (resumed, resumed_optimizer, resumed_manager) = built
        _train(model, optimizer, mesh)
        reference = _gathered(model, optimizer, mesh)
        manager.save(STEP)
        resumed_manager.load()
        loaded = _gathered(resumed, resumed_optimizer, mesh)
        groups = [
            [(g.get("module"), g["lr"], len(g["params"])) for g in o.param_groups]
            for o in (optimizer, resumed_optimizer)
        ]
        if rank == 0:
            torch.save(reference, Path(root) / "reference.pt")
            torch.save(loaded, Path(root) / "resumed.pt")
            torch.save(groups, Path(root) / "groups.pt")
    finally:
        dist.destroy_process_group()


def _save_on_two_meshes(rank, world, port, names, sizes, overrides, root) -> None:
    """Save and resume twice at one layout, each time on a new mesh.

    DTensor and FSDP2 give the second model's tensors meshes of the first, which are equal
    to the second's but are other objects. Rank 0 writes each save's reference and what its
    load gave.
    """
    _init(rank, world, port)
    try:
        for run in range(2):
            mesh = init_device_mesh("cpu", sizes, mesh_dim_names=names)
            config = _config(overrides)
            model, optimizer = _build(mesh, config, seed=run)
            _train(model, optimizer, mesh)
            reference = _gathered(model, optimizer, mesh)
            ckpt = CheckpointConfig(dir=str(Path(root) / f"ckpt{run}"))
            CheckpointManager(ckpt, model, optimizer).save(STEP)
            resumed, resumed_optimizer = _build(mesh, config, seed=run + 2)
            CheckpointManager(ckpt, resumed, resumed_optimizer).load()
            loaded = _gathered(resumed, resumed_optimizer, mesh)
            if rank == 0:
                torch.save((reference, loaded), Path(root) / f"run{run}.pt")
    finally:
        dist.destroy_process_group()


def _save_drifted(rank, world, port, names, sizes, overrides, root, async_mode) -> None:
    """Shift EP rank 1's copy of every weight outside the routed experts, then save.

    The EP copies then differ, as they drift apart on GPU (#482). Rank 0, which is EP rank
    0, writes its copy. The save runs in ``async_mode`` and is waited for.
    """
    _init(rank, world, port)
    try:
        mesh = init_device_mesh("cpu", sizes, mesh_dim_names=names)
        model, optimizer = _build(mesh, _config(overrides), seed=0)
        _train(model, optimizer, mesh)
        shift = float(mesh.get_local_rank("ep"))
        with torch.no_grad():
            for name, p in model.named_parameters():
                if not _is_expert(name):
                    (p.to_local() if isinstance(p, DTensor) else p).add_(shift)
        copy = _gathered(model, optimizer, mesh)
        ckpt = CheckpointConfig(dir=str(Path(root) / "ckpt"), async_mode=async_mode)
        manager = CheckpointManager(ckpt, model, optimizer)
        manager.save(STEP)
        manager.wait()
        if rank == 0:
            torch.save(copy, Path(root) / "reference.pt")
    finally:
        dist.destroy_process_group()


def _save_misplaced(rank, world, port, names, sizes, overrides, root) -> None:
    """Save with FSDP's placements left as they are, which put the rows dp-major.

    That is the right size at the wrong offset, as older torch versions can lay out the
    strided shard. Ranks whose rows would land elsewhere refuse the save, and the others
    raise with them.
    """
    _init(rank, world, port)
    try:
        expert_layout_module._rows_after_ep = lambda placements, ep, name: tuple(placements)
        mesh = init_device_mesh("cpu", sizes, mesh_dim_names=names)
        model, optimizer = _build(mesh, _config(overrides), seed=0)
        _train(model, optimizer, mesh)
        manager = CheckpointManager(
            CheckpointConfig(dir=str(Path(root) / "ckpt")), model, optimizer
        )
        if mesh.get_local_rank("dp_shard") != mesh.get_local_rank("ep"):
            expected = pytest.raises(ValueError, match="but they belong at")
        else:
            expected = pytest.raises(RuntimeError, match="failed on ranks")
        with expected:
            manager.save(STEP)
    finally:
        dist.destroy_process_group()


def _fail_on_rank_one(rank, world, port, names, sizes, overrides, root) -> None:
    """Fail only rank 1's part of a save, then of a load, before DCP runs.

    Rank 1 raises its own error. Every other rank raises too, naming the operation and
    rank 1, rather than wait in DCP's collectives for a rank that has left.
    """
    _init(rank, world, port)
    try:
        mesh = init_device_mesh("cpu", sizes, mesh_dim_names=names)
        model, optimizer = _build(mesh, _config(overrides), seed=0)
        _train(model, optimizer, mesh)
        manager = CheckpointManager(
            CheckpointConfig(dir=str(Path(root) / "ckpt")), model, optimizer
        )

        def fail(*args, **kwargs):
            raise ValueError("rank 1 alone fails")

        for what, target, attribute in (
            ("save", expert_layout_module.ExpertLayout, "to_global"),
            ("load", manager_module, "check_saved_experts"),
        ):
            original = getattr(target, attribute)
            if rank == 1:
                setattr(target, attribute, fail)
                expected = pytest.raises(ValueError, match="rank 1 alone fails")
            else:
                expected = pytest.raises(
                    RuntimeError, match=rf"Checkpoint {what} failed on ranks \[1\]"
                )
            with expected:
                if what == "save":
                    manager.save(STEP)
                else:
                    manager.load(path=str(_step_dir(Path(root))))
            setattr(target, attribute, original)
            if what == "save":
                manager.save(STEP)
    finally:
        dist.destroy_process_group()


def _refuse(rank, world, port, names, sizes, overrides, step_dir) -> None:
    """Every rank refuses to load ``step_dir`` at this layout, naming what it holds."""
    _init(rank, world, port)
    try:
        mesh = init_device_mesh("cpu", sizes, mesh_dim_names=names)
        model, optimizer = _build(mesh, _config(overrides), seed=1)
        manager = CheckpointManager(
            CheckpointConfig(dir=str(Path(step_dir).parent)), model, optimizer
        )
        with pytest.raises(ValueError, match="holds 2 of the 4 experts"):
            manager.load(path=step_dir)
    finally:
        dist.destroy_process_group()


def _load_one_process(
    config: ModelConfig, step_dir: Path, exclude: list[str] | None = None
) -> dict[str, torch.Tensor]:
    """Load ``step_dir`` into an unsharded model with ``ep = 1``; return what it then holds."""
    model, optimizer = _build(None, config, seed=1)
    manager = CheckpointManager(CheckpointConfig(dir=str(step_dir.parent)), model, optimizer)
    manager.load(path=str(step_dir), exclude_keys=exclude)
    return _gathered(model, optimizer, None)


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
    roots: dict[tuple[str, str, bool], Path] = {}

    def run(layout: str, async_mode: str = "disabled", unfixed: bool = False) -> Path:
        key = (layout, async_mode, unfixed)
        if key not in roots:
            root = tmp_path_factory.mktemp(layout)
            names, sizes, overrides = LAYOUTS[layout]
            _spawn(_save, names, sizes, overrides, str(root), async_mode, unfixed)
            roots[key] = root
        return roots[key]

    return run


def _step_dir(root: Path) -> Path:
    return root / "ckpt" / f"step_{STEP}"


class TestCheckpointHoldsEveryExpert:
    @pytest.mark.parametrize("layout", list(LAYOUTS))
    def test_every_expert_is_saved_once(self, saved_layout, layout, tmp_path):
        """The index holds every expert of every layer, each equal to its rank's weights.

        AdamW's moments are checked with the weights, so the checkpoint holds each rank's
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


class TestSecondMeshOfOneLayout:
    def test_a_second_mesh_saves_and_resumes_every_expert(self, tmp_path):
        """A process that builds a second, equal mesh still saves and resumes every expert."""
        names, sizes, overrides = LAYOUTS["dp_shard2_ep2_packed"]
        _spawn(_save_on_two_meshes, names, sizes, overrides, str(tmp_path))
        config = _config(overrides)
        experts = list(range(config.num_experts))
        for run in range(2):
            step_dir = tmp_path / f"ckpt{run}" / f"step_{STEP}"
            reference, loaded = torch.load(tmp_path / f"run{run}.pt")
            assert _held_experts(step_dir, config) == {
                f"model.layers.{layer}.mlp": experts for layer in range(config.n_layers)
            }
            saved = _saved(step_dir, tmp_path)
            assert _mismatches({k: saved[k] for k in reference if k in saved}, reference) == []
            assert _mismatches(loaded, reference) == []


class TestDriftedCopies:
    @pytest.mark.parametrize("async_mode", ["disabled", AsyncCheckpointMode.async_.value])
    def test_the_checkpoint_holds_one_ep_rank_copy_whole(self, tmp_path, async_mode):
        """Every weight outside the routed experts is EP rank 0's copy, not a mix of copies.

        The async save takes the same planner, so it holds the same copy.
        """
        names, sizes, overrides = LAYOUTS["dp_shard2_ep2"]
        _spawn(_save_drifted, names, sizes, overrides, str(tmp_path), async_mode)
        reference = torch.load(tmp_path / "reference.pt")
        saved = _saved(_step_dir(tmp_path), tmp_path)
        assert _mismatches({k: saved[k] for k in reference if k in saved}, reference) == []


class TestPlacementCheck:
    def test_rows_saved_out_of_place_refuse_the_save(self, tmp_path):
        """A view whose rows DCP would save at the wrong offset is refused on every rank."""
        names, sizes, overrides = LAYOUTS["dp_shard2_ep2_packed"]
        _spawn(_save_misplaced, names, sizes, overrides, str(tmp_path))
        assert not (_step_dir(tmp_path) / ".metadata").exists()


class TestFailureOnOneRank:
    def test_every_rank_raises_and_names_the_operation(self, tmp_path):
        """One rank's failure before DCP raises on every rank, in a save and in a load."""
        names, sizes, overrides = LAYOUTS["ep2_packed"]
        _spawn(_fail_on_rank_one, names, sizes, overrides, str(tmp_path))


class TestResumeAtTheSameLayout:
    @pytest.mark.parametrize("layout", list(LAYOUTS))
    def test_each_rank_resumes_its_own_experts(self, saved_layout, layout):
        """A model built from another seed holds every rank's saved weights and moments."""
        root = saved_layout(layout)
        reference = torch.load(root / "reference.pt")
        assert _mismatches(torch.load(root / "resumed.pt"), reference) == []


class TestPerModuleLearningRates:
    @pytest.mark.parametrize("layout", ["ep2", "dp_shard2_ep2_packed"])
    def test_resume_restores_every_group_and_expert(self, tmp_path, layout):
        """Experts split among per-module groups resume with their weights, moments and groups."""
        names, sizes, overrides = LAYOUTS[layout]
        _spawn(_save_with_lr_groups, names, sizes, overrides, str(tmp_path))
        reference = torch.load(tmp_path / "reference.pt")
        assert _mismatches(torch.load(tmp_path / "resumed.pt"), reference) == []
        saved, resumed = torch.load(tmp_path / "groups.pt")
        assert resumed == saved
        assert {"layer0", "layer1"} <= {module for module, _, _ in saved}


class TestLoadAtAnotherEP:
    @pytest.mark.parametrize("layout", list(LAYOUTS))
    def test_ep1_loads_every_expert(self, saved_layout, layout):
        """One process without EP loads every expert and its moments from an EP checkpoint."""
        root = saved_layout(layout)
        config = _config(LAYOUTS[layout][2])
        reference = torch.load(root / "reference.pt")
        assert _mismatches(_load_one_process(config, _step_dir(root)), reference) == []

    @pytest.mark.parametrize(
        ("saved_as", "target"),
        [
            pytest.param("dp_shard2_ep2", EP4, id="dp_shard2_ep2_to_ep4"),
            pytest.param("dp_shard2_ep2_packed", EP4, id="dp_shard2_ep2_packed_to_ep4"),
            pytest.param("ep4_packed_gelu", DP_SHARD2_EP2, id="ep4_packed_gelu_to_dp_shard2_ep2"),
            pytest.param("dp_shard2_ep2_packed", HSDP_EP2, id="dp_shard2_ep2_packed_to_hsdp_ep2"),
            pytest.param("ep2_tp2", DP_SHARD2_EP2, id="ep2_tp2_to_dp_shard2_ep2"),
        ],
    )
    def test_another_layout_loads_every_rank_its_experts(
        self, saved_layout, saved_as, target, tmp_path
    ):
        """A checkpoint saved at one ``ep`` and FSDP layout loads at another."""
        root = saved_layout(saved_as)
        out = tmp_path / "loaded.pt"
        _spawn(_load, *target, LAYOUTS[saved_as][2], str(_step_dir(root)), str(out))
        assert _mismatches(torch.load(out), torch.load(root / "reference.pt")) == []

    @pytest.mark.parametrize("overrides", [{}, PACKED], ids=["unpacked", "packed"])
    def test_ep_ranks_load_their_experts_from_an_ep1_checkpoint(self, overrides, tmp_path):
        """A checkpoint one process saved gives each EP rank its own slice of the experts."""
        config = _config(overrides)
        model, optimizer = _build(None, config, seed=0)
        _train(model, optimizer, None)
        reference = _gathered(model, optimizer, None)
        CheckpointManager(CheckpointConfig(dir=str(tmp_path / "ckpt")), model, optimizer).save(STEP)
        out = tmp_path / "loaded.pt"
        _spawn(_load, *DP_SHARD2_EP2, overrides, str(_step_dir(tmp_path)), str(out))
        assert _mismatches(torch.load(out), reference) == []


class TestUnfixedCheckpoint:
    """A checkpoint saved as before the fix holds one EP rank's experts under every rank's names."""

    @pytest.mark.parametrize("layout", ["dp_shard2_ep2", "dp_shard2_ep2_packed"])
    def test_load_at_the_same_ep_is_refused(self, saved_layout, layout):
        """Every rank refuses it, rather than load one rank's experts on all of them."""
        root = saved_layout(layout, unfixed=True)
        assert _held_experts(_step_dir(root), _config(LAYOUTS[layout][2])) == {
            f"model.layers.{layer}.mlp": [0, 1] for layer in range(2)
        }
        names, sizes, overrides = LAYOUTS[layout]
        _spawn(_refuse, names, sizes, overrides, str(_step_dir(root)))

    @pytest.mark.parametrize("layout", ["dp_shard2_ep2", "dp_shard2_ep2_packed"])
    def test_load_without_ep_is_refused_with_what_it_holds(self, saved_layout, layout):
        """The error names the experts the checkpoint holds and the ``ep`` that saved them."""
        root = saved_layout(layout, unfixed=True)
        config = _config(LAYOUTS[layout][2])
        holds = "layers.0.mlp.up_w has 2 rows" if config.moe_packed_experts else "experts 0-1"
        with pytest.raises(
            ValueError, match=rf"holds 2 of the 4 experts of 2 MoE layers .*{holds}.*ep=2"
        ):
            _load_one_process(config, _step_dir(root), exclude=["optimizer"])

    @pytest.mark.parametrize("layout", ["dp_shard2_ep2", "dp_shard2_ep2_packed"])
    def test_generate_is_refused(self, saved_layout, layout):
        """The generation script refuses it with the same error, before DCP reads it."""
        root = saved_layout(layout, unfixed=True)
        config = JobConfig(
            model=_config(LAYOUTS[layout][2]),
            checkpoint=CheckpointConfig(load_path=str(_step_dir(root))),
        )
        with pytest.raises(ValueError, match="holds 2 of the 4 experts"):
            _load_model(config, torch.device("cpu"), torch.float32)

    @pytest.mark.parametrize("layout", ["dp_shard2_ep2", "dp_shard2_ep2_packed"])
    def test_dcp_to_hf_is_refused(self, saved_layout, layout, tmp_path):
        """The export refuses it too, since the other experts' weights were never written."""
        root = saved_layout(layout, unfixed=True)
        with pytest.raises(ValueError, match="holds 2 of the 4 experts"):
            dcp_to_hf(str(_step_dir(root)), str(tmp_path / "hf"), _config(LAYOUTS[layout][2]))


class TestExportFromAnEPCheckpoint:
    @pytest.mark.parametrize("layout", ["dp_shard2_ep2", "dp_shard2_ep2_packed", "ep4_packed_gelu"])
    def test_dcp_to_hf_exports_every_expert(self, saved_layout, layout, tmp_path):
        """Every expert of every layer reaches the export with its rank's weights."""
        root = saved_layout(layout)
        config = _config(LAYOUTS[layout][2])
        dcp_to_hf(str(_step_dir(root)), str(tmp_path / "hf"), config, dtype=torch.float32)
        exported = load_file(tmp_path / "hf" / "model.safetensors")
        reference = torch.load(root / "reference.pt")
        experts = {k: v for k, v in reference.items() if _is_expert(k) and not k.endswith(MOMENTS)}
        assert len(experts) == config.n_layers * (
            len(PACKED_EXPERT) - (config.activation == "gelu")
            if config.moe_packed_experts
            else config.num_experts * 3
        )
        assert _mismatches({k: exported[f"model.{k}"] for k in experts}, experts) == []


class TestWithoutEP:
    @pytest.mark.parametrize("overrides", [{}, PACKED], ids=["unpacked", "packed"])
    def test_checkpoint_keeps_the_model_names_and_loads(self, overrides, tmp_path):
        """With ``ep = 1`` the save writes the model's own names, as before, and loads back."""
        config = _config(overrides)
        model, optimizer = _build(None, config, seed=0)
        _train(model, optimizer, None)
        reference = _gathered(model, optimizer, None)
        CheckpointManager(CheckpointConfig(dir=str(tmp_path / "ckpt")), model, optimizer).save(STEP)
        index = FileSystemReader(str(_step_dir(tmp_path))).read_metadata().state_dict_metadata
        names = {f"model.{name}" for name in get_model_state_dict(model)}
        assert {key for key in index if key.startswith("model.")} == names
        assert _mismatches(_load_one_process(config, _step_dir(tmp_path)), reference) == []
