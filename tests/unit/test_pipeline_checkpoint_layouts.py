"""Checkpoints across pipeline layouts: what a run loads from a checkpoint of another layout.

A pipeline stage loads its own layers' weights from a checkpoint saved without pipeline
parallelism, and refuses its optimizer state. A run refuses a stage checkpoint saved at
another pipeline degree, or saved under pipeline parallelism when it has none, naming both
layouts and the dcp-to-hf, hf-to-dcp and load_path route. The cases run in one process, and
across gloo ranks on CPU with real stage modules under FSDP2.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor

from kempnerforge.checkpoint import manager as manager_module
from kempnerforge.checkpoint.manager import CheckpointManager
from kempnerforge.config.checkpoint import CheckpointConfig
from kempnerforge.config.job import JobConfig
from kempnerforge.config.model import ModelConfig
from kempnerforge.config.training import TrainConfig
from kempnerforge.distributed.parallel import apply_fsdp2
from kempnerforge.distributed.pipeline_parallel import build_stage_module
from kempnerforge.model.transformer import Transformer
from kempnerforge.training.entry import build_checkpoint_manager, restore_checkpoint
from kempnerforge.training.runtime import PipelineBundle, RuntimeContext
from tests.gloo_ranks import gloo_group, spawn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from convert_checkpoint import dcp_to_hf, hf_to_dcp

MODEL = ModelConfig(dim=32, n_layers=4, n_heads=4, vocab_size=64, max_seq_len=16)
FP32 = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)
STEP = 3
TOKENS = 96
ROUTE = r"dcp-to-hf.*hf-to-dcp.*checkpoint\.load_path"
WEIGHTS_ONLY = r'exclude_from_loading = \["optimizer"\]'


def _source() -> dict[str, torch.Tensor]:
    """The whole model's weights, the same on every rank."""
    torch.manual_seed(0)
    return Transformer(MODEL).state_dict()


def _stepped(model: torch.nn.Module, lr: float = 0.1) -> torch.optim.Optimizer:
    """AdamW after one step, so it holds state to save; at ``lr=0`` the weights stay put."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    optimizer.step()
    return optimizer


def _differs(model: torch.nn.Module, weights: dict[str, torch.Tensor]) -> bool:
    return any(not torch.equal(value, weights[key]) for key, value in model.state_dict().items())


def _stage(stage: int, stages: int, weights: dict[str, torch.Tensor] | None = None):
    torch.manual_seed(1)
    module = build_stage_module(MODEL, stage, stages)
    if weights is not None:
        module.load_state_dict({key: weights[key] for key in module.state_dict()})
    return module


def _save_stages(ckpt_dir: Path, stages: int, *, record: bool = True) -> dict[str, torch.Tensor]:
    """Save the source weights as ``stages`` stages save them, one stage at a time."""
    weights = _source()
    for k in range(stages):
        stage = _stage(k, stages, weights)
        CheckpointManager(
            CheckpointConfig(dir=str(ckpt_dir)),
            stage,
            _stepped(stage, lr=0.0),
            pp_rank=k,
            pp_size=stages if record else None,
        ).save(STEP, tokens_seen=TOKENS)
    return weights


def _save_flat(ckpt_dir: Path, *, stepped: bool = True) -> dict[str, torch.Tensor]:
    """Save the whole model without pipeline parallelism; return the weights it saved."""
    weights = _source()
    model = Transformer(MODEL)
    model.load_state_dict(weights)
    optimizer = _stepped(model) if stepped else torch.optim.AdamW(model.parameters(), lr=0.1)
    CheckpointManager(CheckpointConfig(dir=str(ckpt_dir)), model, optimizer).save(
        STEP, tokens_seen=TOKENS
    )
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _stage_manager(ckpt_dir: Path, stage: int, stages: int, **kwargs: Any):
    module = _stage(stage, stages)
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.1)
    config = CheckpointConfig(dir=str(ckpt_dir))
    return module, optimizer, CheckpointManager(config, module, optimizer, pp_rank=stage, **kwargs)


class TestManagerArguments:
    @pytest.mark.parametrize(("pp_rank", "pp_size"), [(None, 2), (2, 2), (-1, 2)])
    def test_pp_size_needs_a_pp_rank_inside_it(self, tmp_path, pp_rank, pp_size):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        config = CheckpointConfig(dir=str(tmp_path))
        with pytest.raises(ValueError, match="pp_size needs a pp_rank"):
            CheckpointManager(config, model, optimizer, pp_rank=pp_rank, pp_size=pp_size)


class TestSavedDegree:
    def test_a_pipeline_save_records_its_degree(self, tmp_path):
        _save_stages(tmp_path, 2)
        meta = json.loads((tmp_path / f"step_{STEP}" / "metadata.json").read_text())
        assert meta["pp"] == 2

    def test_a_save_without_pipeline_parallelism_records_no_degree(self, tmp_path):
        """The default save writes what it wrote before: one index at the top, no new key."""
        _save_flat(tmp_path)
        step_dir = tmp_path / f"step_{STEP}"
        meta = json.loads((step_dir / "metadata.json").read_text())
        assert meta == {"step": STEP, "tokens_seen": TOKENS}
        assert (step_dir / ".metadata").is_file()
        assert not [d for d in step_dir.iterdir() if d.name.startswith("pp")]


class TestStageLoadsAFlatCheckpoint:
    @pytest.mark.parametrize("stage", [0, 1])
    def test_each_stage_loads_its_own_layers(self, tmp_path, stage):
        saved = _save_flat(tmp_path / "flat")
        module, optimizer, manager = _stage_manager(tmp_path / "run", stage, 2, pp_size=2)
        assert _differs(module, saved)
        path = str(tmp_path / "flat" / f"step_{STEP}")
        assert manager.load(path=path, exclude_keys=["optimizer"])[:2] == (STEP, TOKENS)
        assert not _differs(module, saved)
        assert not optimizer.state

    def test_a_checkpoint_with_tied_embeddings_gives_the_last_stage_its_head(self, tmp_path):
        """A save without PP keeps both names of a tied weight, so a stage, which never ties,
        loads the output head as the token embedding it was tied to."""
        tied = Transformer(replace(MODEL, tie_embeddings=True))
        CheckpointManager(
            CheckpointConfig(dir=str(tmp_path / "flat")), tied, _stepped(tied, lr=0.0)
        ).save(STEP, tokens_seen=TOKENS)
        module, _, manager = _stage_manager(tmp_path / "run", 1, 2, pp_size=2)
        path = str(tmp_path / "flat" / f"step_{STEP}")
        manager.load(path=path, exclude_keys=["optimizer"])
        assert module.output_head is not None and tied.token_embedding is not None
        assert torch.equal(module.output_head.proj.weight, tied.token_embedding.embedding.weight)

    def test_the_optimizer_state_is_refused_before_anything_loads(self, tmp_path):
        saved = _save_flat(tmp_path / "flat")
        module, _, manager = _stage_manager(tmp_path / "run", 0, 2, pp_size=2)
        before = {key: value.clone() for key, value in module.state_dict().items()}
        expected = rf"saved without pipeline parallelism.*this run has pp=2.*{WEIGHTS_ONLY}"
        with pytest.raises(ValueError, match=expected):
            manager.load(path=str(tmp_path / "flat" / f"step_{STEP}"))
        assert not _differs(module, before)
        assert _differs(module, saved)

    def test_a_stage_without_a_degree_refuses_it_too(self, tmp_path):
        _save_flat(tmp_path / "flat")
        _, _, manager = _stage_manager(tmp_path / "run", 1, 2)
        with pytest.raises(ValueError, match=r"this run is pipeline-parallel"):
            manager.load(path=str(tmp_path / "flat" / f"step_{STEP}"))

    def test_a_checkpoint_without_optimizer_state_loads_whole(self, tmp_path):
        """Saved before the first step, as an hf-to-dcp import is: the optimizer starts fresh."""
        saved = _save_flat(tmp_path / "flat", stepped=False)
        module, optimizer, manager = _stage_manager(tmp_path / "run", 1, 2, pp_size=2)
        assert manager.load(path=str(tmp_path / "flat" / f"step_{STEP}"))[:2] == (STEP, TOKENS)
        assert not _differs(module, saved)
        assert not optimizer.state


class TestStageCheckpointRefusals:
    def test_a_run_without_pipeline_parallelism_refuses_it(self, tmp_path):
        _save_stages(tmp_path / "staged", 2)
        model = Transformer(MODEL)
        manager = CheckpointManager(
            CheckpointConfig(dir=str(tmp_path / "run")),
            model,
            torch.optim.AdamW(model.parameters(), lr=0.1),
        )
        path = str(tmp_path / "staged" / f"step_{STEP}")
        for exclude in (None, ["optimizer"]):
            with pytest.raises(ValueError, match=rf"at pp=2.*no pipeline parallelism.*{ROUTE}"):
                manager.load(path=path, exclude_keys=exclude)

    def test_a_train_state_load_reads_no_shards_and_takes_any_layout(self, tmp_path):
        _save_stages(tmp_path / "staged", 2)
        model = Transformer(MODEL)
        manager = CheckpointManager(
            CheckpointConfig(dir=str(tmp_path / "run")),
            model,
            torch.optim.AdamW(model.parameters(), lr=0.1),
        )
        path = str(tmp_path / "staged" / f"step_{STEP}")
        assert manager.load(path=path, exclude_keys=["model", "optimizer"])[:2] == (STEP, TOKENS)

    @pytest.mark.parametrize("record", [True, False], ids=["recorded", "unrecorded"])
    def test_another_degree_is_refused(self, tmp_path, record):
        _save_stages(tmp_path / "staged", 2, record=record)
        _, _, manager = _stage_manager(tmp_path / "run", 0, 4, pp_size=4)
        saved_as = "at pp=2" if record else "2 stage directories"
        with pytest.raises(ValueError, match=rf"{saved_as}.*this run has pp=4.*{ROUTE}"):
            manager.load(path=str(tmp_path / "staged" / f"step_{STEP}"))


class TestStageCheckpointLoads:
    @pytest.mark.parametrize("record", [True, False], ids=["recorded", "unrecorded"])
    @pytest.mark.parametrize("pp_size", [2, None], ids=["degree", "no-degree"])
    def test_the_same_degree_loads_every_stage(self, tmp_path, record, pp_size):
        """A checkpoint saved before the degree was recorded loads at its own layout."""
        weights = _save_stages(tmp_path / "staged", 2, record=record)
        for stage in (0, 1):
            module, optimizer, manager = _stage_manager(tmp_path / "run", stage, 2, pp_size=pp_size)
            path = str(tmp_path / "staged" / f"step_{STEP}")
            assert manager.load(path=path)[:2] == (STEP, TOKENS)
            assert not _differs(module, weights)
            assert optimizer.state


class TestResumeFallback:
    @staticmethod
    def _step(root: Path, step: int, stages: list[int], *, record: int | None = None) -> Path:
        step_dir = root / f"step_{step}"
        for k in stages:
            (step_dir / f"pp{k}").mkdir(parents=True)
            (step_dir / f"pp{k}" / ".metadata").write_text("")
        meta = {"step": step, **({"pp": record} if record else {})}
        (step_dir / "metadata.json").write_text(json.dumps(meta))
        return step_dir

    def test_a_recorded_stage_that_never_saved_is_noticed(self, tmp_path):
        self._step(tmp_path, 5, [0, 1], record=2)
        torn = self._step(tmp_path, 10, [0], record=2)
        assert CheckpointManager._newest_saved_beside(torn) == tmp_path / "step_5"

    def test_an_unrecorded_stage_that_never_saved_is_noticed_with_the_degree(self, tmp_path):
        self._step(tmp_path, 5, [0, 1])
        torn = self._step(tmp_path, 10, [0])
        assert CheckpointManager._newest_saved_beside(torn) == torn
        assert CheckpointManager._newest_saved_beside(torn, 2) == tmp_path / "step_5"


def _mesh(shape: tuple[int, ...], names: tuple[str, ...]) -> DeviceMesh:
    return init_device_mesh("cpu", shape, mesh_dim_names=names)


def _job(
    rank: int, world: int, mesh: DeviceMesh, ckpt_dir: str, **checkpoint: Any
) -> tuple[JobConfig, RuntimeContext, PipelineBundle | None]:
    """The config, runtime and pipeline of a run on ``mesh``, as training builds them."""
    config = JobConfig(
        model=MODEL,
        train=TrainConfig(batch_size=1, seq_len=16),
        checkpoint=CheckpointConfig(dir=ckpt_dir, **checkpoint),
    )
    runtime = RuntimeContext(
        rank=rank, local_rank=rank, world_size=world, device=torch.device("cpu"), device_mesh=mesh
    )
    if "pp" not in (mesh.mesh_dim_names or ()):
        return config, runtime, None
    config.distributed.pp = mesh["pp"].size()
    pipeline = PipelineBundle(rank=mesh.get_local_rank("pp"), size=mesh["pp"].size(), schedule=None)
    return config, runtime, pipeline


def _model(mesh: DeviceMesh, weights: dict[str, torch.Tensor] | None = None) -> torch.nn.Module:
    """This rank's stage on ``mesh``, or the whole model without ``pp``, under FSDP2."""
    torch.manual_seed(1)
    if "pp" in (mesh.mesh_dim_names or ()):
        model = build_stage_module(MODEL, mesh.get_local_rank("pp"), mesh["pp"].size())
    else:
        model = Transformer(MODEL)
    if weights is not None:
        model.load_state_dict({key: weights[key] for key in model.state_dict()})
    apply_fsdp2(model, mesh, mp_policy=FP32)
    return model


def _full(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.full_tensor() if isinstance(value, DTensor) else value.detach().clone()
        for key, value in model.state_dict().items()
    }


def _train_and_save(
    rank: int, world: int, mesh: DeviceMesh, ckpt_dir: str, lr: float = 0.1
) -> dict[str, torch.Tensor]:
    """Save the source weights after a step at ``lr`` on ``mesh``; return the weights saved."""
    config, runtime, pipeline = _job(rank, world, mesh, ckpt_dir)
    model = _model(mesh, _source())
    manager = build_checkpoint_manager(config, runtime, model, _stepped(model, lr), pipeline)
    manager.save(STEP, tokens_seen=TOKENS)
    manager.wait()
    return _full(model)


def _restore(
    rank: int, world: int, mesh: DeviceMesh, ckpt_dir: str, **checkpoint: Any
) -> tuple[tuple[int, int], dict[str, torch.Tensor]]:
    """Start a run on ``mesh`` as training does; return the step it resumed and its weights."""
    config, runtime, pipeline = _job(rank, world, mesh, ckpt_dir, **checkpoint)
    model = _model(mesh)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    manager = build_checkpoint_manager(config, runtime, model, optimizer, pipeline)
    resumed = restore_checkpoint(config, model, None, manager)
    return resumed, _full(model)


def _assert_holds(loaded: dict[str, torch.Tensor], saved: dict[str, torch.Tensor]) -> None:
    for key, value in loaded.items():
        assert torch.equal(value, saved[key]), key


def _check_a_flat_checkpoint_warm_starts_stages(rank: int, world: int, port: int, root: str):
    with gloo_group(rank, world, port):
        saved = _train_and_save(rank, world, _mesh((world,), ("dp_shard",)), f"{root}/flat")
        staged = _mesh((2, world // 2), ("pp", "dp_shard"))
        resumed, loaded = _restore(
            rank,
            world,
            staged,
            f"{root}/warm",
            load_path=f"{root}/flat/step_{STEP}",
            exclude_from_loading=["optimizer"],
        )
        assert resumed == (STEP, TOKENS)
        assert loaded.keys() < saved.keys()
        _assert_holds(loaded, saved)
        expected = rf"saved without pipeline parallelism.*this run has pp=2.*{WEIGHTS_ONLY}"
        with pytest.raises(ValueError, match=expected):
            _restore(rank, world, staged, f"{root}/flat")


def _check_both_layouts_save_the_same_weights(rank: int, world: int, port: int, root: str):
    with gloo_group(rank, world, port):
        _train_and_save(rank, world, _mesh((world,), ("dp_shard",)), f"{root}/flat", lr=0.0)
        staged = _mesh((2, world // 2), ("pp", "dp_shard"))
        _train_and_save(rank, world, staged, f"{root}/staged", lr=0.0)


def _check_an_import_warm_starts_stages(rank: int, world: int, port: int, root: str):
    with gloo_group(rank, world, port):
        staged = _mesh((world,), ("pp",))
        resumed, loaded = _restore(rank, world, staged, f"{root}/run", load_path=f"{root}/import")
        assert resumed == (0, 0)
        _assert_holds(loaded, _source())


def _check_the_refusals(rank: int, world: int, port: int, root: str):
    """A pp=2 checkpoint, refused by a run without pipeline parallelism and by a pp=4 run,
    recorded or not, and loaded by a pp=2 run when its record is gone."""
    with gloo_group(rank, world, port):
        two = _mesh((2, world // 2), ("pp", "dp_shard"))
        saved = _train_and_save(rank, world, two, f"{root}/two")
        step_dir = Path(root) / "two" / f"step_{STEP}"
        flat = _mesh((world,), ("dp_shard",))
        four = _mesh((world,), ("pp",))
        no_pp = rf"at pp=2.*no pipeline parallelism.*{ROUTE}"
        with pytest.raises(ValueError, match=no_pp):
            _restore(rank, world, flat, f"{root}/flat-run", load_path=str(step_dir))
        with pytest.raises(ValueError, match=no_pp):
            _restore(rank, world, flat, f"{root}/two")
        with pytest.raises(ValueError, match=rf"at pp=2.*this run has pp=4.*{ROUTE}"):
            _restore(rank, world, four, f"{root}/two")
        dist.barrier()
        if rank == 0:
            meta = json.loads((step_dir / "metadata.json").read_text())
            del meta["pp"]
            (step_dir / "metadata.json").write_text(json.dumps(meta))
        dist.barrier()
        with pytest.raises(ValueError, match=rf"2 stage directories.*this run has pp=4.*{ROUTE}"):
            _restore(rank, world, four, f"{root}/two")
        resumed, loaded = _restore(rank, world, two, f"{root}/two")
        assert resumed == (STEP, TOKENS)
        _assert_holds(loaded, saved)


def _unreadable(ckpt_dir: Path) -> Any:
    raise OSError(f"rank 1 cannot read {ckpt_dir}")


def _check_rank_zero_decides(rank: int, world: int, port: int, root: str):
    """Rank 1 cannot read the checkpoint's layout; every rank loads it as rank 0 read it."""
    with gloo_group(rank, world, port):
        two = _mesh((world,), ("pp",))
        saved = _train_and_save(rank, world, two, f"{root}/two")
        if rank == 1:
            manager_module.saved_layout = _unreadable
        resumed, loaded = _restore(rank, world, two, f"{root}/two")
        assert resumed == (STEP, TOKENS)
        _assert_holds(loaded, saved)


class TestAcrossRanks:
    def test_a_flat_checkpoint_warm_starts_every_stage(self, tmp_path):
        """dp_shard=4 save; a pp=2 x dp_shard=2 run loads each stage's layers from it, and
        its auto-resume from it refuses the optimizer state on every rank."""
        spawn(_check_a_flat_checkpoint_warm_starts_stages, 4, str(tmp_path))

    def test_a_stage_checkpoint_exports_as_the_one_saved_without_pp(self, tmp_path):
        """The same weights saved at pp=2 x dp_shard=2 and at dp_shard=4 export the same files."""
        spawn(_check_both_layouts_save_the_same_weights, 4, str(tmp_path))
        for name in ("flat", "staged"):
            dcp_to_hf(
                str(tmp_path / name / f"step_{STEP}"),
                str(tmp_path / f"hf-{name}"),
                MODEL,
                dtype=torch.float32,
            )
        for name in ("model.safetensors", "config.json"):
            exported = (tmp_path / "hf-staged" / name).read_bytes()
            assert exported == (tmp_path / "hf-flat" / name).read_bytes(), name
        staged = tmp_path / "staged" / f"step_{STEP}"
        assert json.loads((staged / "metadata.json").read_text())["pp"] == 2

    def test_an_hf_to_dcp_import_warm_starts_every_stage(self, tmp_path):
        """dcp-to-hf, then hf-to-dcp, then checkpoint.load_path into a pp=2 run."""
        dcp.save({"model": _source()}, checkpoint_id=str(tmp_path / "dcp"))
        dcp_to_hf(str(tmp_path / "dcp"), str(tmp_path / "hf"), MODEL, dtype=torch.float32)
        hf_to_dcp(str(tmp_path / "hf"), str(tmp_path / "import"), MODEL)
        spawn(_check_an_import_warm_starts_stages, 2, str(tmp_path))

    def test_every_rank_refuses_a_layout_it_cannot_load(self, tmp_path):
        spawn(_check_the_refusals, 4, str(tmp_path))

    def test_every_rank_loads_the_layout_rank_0_read(self, tmp_path):
        spawn(_check_rank_zero_decides, 2, str(tmp_path))
