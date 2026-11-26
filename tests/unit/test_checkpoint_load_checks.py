"""What a load checks before it reads any tensor: the model against the checkpoint, and the
VLM freeze against the checkpoint the load settles on.

A checkpoint's DCP index lists every model tensor it holds with its global size. A load refuses,
on every rank and before any tensor is read, a checkpoint that holds model tensors the run's
model does not have, or one of another size, unless ``checkpoint.ignore_extra_model_tensors``
lets it leave the extra ones unread. A VLM resume compares the freeze the checkpoint records with
the run's freeze at the step it records, both from rank 0's one read of its ``metadata.json``.
The cases run in one process, and across gloo ranks on CPU under FSDP2.
"""

from __future__ import annotations

import logging
import pathlib
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy

import kempnerforge.checkpoint.manager as manager_module
import kempnerforge.training.entry as entry
from kempnerforge.checkpoint.manager import CheckpointManager
from kempnerforge.config.adapter import AdapterConfig
from kempnerforge.config.checkpoint import CheckpointConfig
from kempnerforge.config.job import JobConfig
from kempnerforge.config.model import ModelConfig
from kempnerforge.config.training import TrainConfig
from kempnerforge.config.vision import VisionEncoderConfig
from kempnerforge.config.vlm import FreezeSpec, FreezeStage, VLMConfig
from kempnerforge.distributed.parallel import apply_fsdp2
from kempnerforge.model.transformer import Transformer
from kempnerforge.training.freeze import freeze_meta_at_step
from tests.gloo_ranks import gloo_group, spawn

MODEL = ModelConfig(dim=32, n_layers=4, n_heads=4, vocab_size=64, max_seq_len=16)
FP32 = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)
STEP = 3
TOKENS = 96
GROUP_TIMEOUT_S = 30
COMPARISON_FAILURE = "cannot compare the saved sizes"
EXTRA = r"holds \d+ model tensors this run's model does not have, such as \['model\.layers\.2\."
OVERRIDE = r"checkpoint\.ignore_extra_model_tensors = true"


def _stepped(model: torch.nn.Module) -> torch.optim.Optimizer:
    """AdamW after one step, so the checkpoint holds optimizer state."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    optimizer.step()
    optimizer.zero_grad()
    return optimizer


def _save(ckpt_dir: Path, config: ModelConfig = MODEL) -> dict[str, torch.Tensor]:
    """Save a stepped model of ``config``; return the weights it saved."""
    torch.manual_seed(0)
    model = Transformer(config)
    CheckpointManager(CheckpointConfig(dir=str(ckpt_dir)), model, _stepped(model)).save(
        STEP, tokens_seen=TOKENS
    )
    return _snapshot(model)


def _run(ckpt_dir: Path, config: ModelConfig = MODEL, **checkpoint):
    """A fresh model of ``config`` with its optimizer and a manager over ``ckpt_dir``."""
    torch.manual_seed(1)
    model = Transformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    manager = CheckpointManager(CheckpointConfig(dir=str(ckpt_dir), **checkpoint), model, optimizer)
    return model, optimizer, manager


def _snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _holds(model: torch.nn.Module, weights: dict[str, torch.Tensor]) -> bool:
    return all(torch.equal(value, weights[key]) for key, value in model.state_dict().items())


def _manager_warnings(call):
    """Run ``call``; return its result and the warnings the checkpoint manager logged."""
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    target = logging.getLogger("kempnerforge.checkpoint.manager")
    capture = _Capture(level=logging.WARNING)
    target.addHandler(capture)
    try:
        result = call()
    finally:
        target.removeHandler(capture)
    return result, [record.getMessage() for record in records]


class TestModelAgainstTheCheckpoint:
    def test_a_model_with_fewer_layers_is_refused_before_any_tensor_is_read(self, tmp_path):
        """A 2-layer model resumed from a 4-layer checkpoint used to load its layers and
        report the saved step, leaving the other two layers unread without a word."""
        _save(tmp_path)
        model, _, manager = _run(tmp_path, replace(MODEL, n_layers=2))
        before = _snapshot(model)
        with pytest.raises(ValueError, match=rf"step_{STEP} {EXTRA}.*{OVERRIDE}"):
            manager.load()
        assert _holds(model, before)

    def test_a_size_change_is_refused_before_any_tensor_is_read(self, tmp_path):
        """A larger vocabulary used to fail inside DCP's read, after auto-resume had found it."""
        _save(tmp_path)
        model, _, manager = _run(tmp_path, replace(MODEL, vocab_size=128))
        before = _snapshot(model)
        with pytest.raises(ValueError) as caught:
            manager.load()
        message = str(caught.value)
        assert "2 model tensors of another size" in message
        resized = "model.token_embedding.embedding.weight (saved [64, 32], this run [128, 32])"
        assert resized in message
        assert "ignore_extra_model_tensors" not in message
        assert _holds(model, before)

    def test_a_weights_only_warm_start_is_checked_too(self, tmp_path):
        _save(tmp_path)
        _, _, manager = _run(tmp_path, replace(MODEL, n_layers=2))
        with pytest.raises(ValueError, match=EXTRA):
            manager.load(exclude_keys=["optimizer", "train_state"])

    def test_the_override_loads_the_tensors_the_model_has(self, tmp_path):
        saved = _save(tmp_path)
        model, optimizer, manager = _run(
            tmp_path, replace(MODEL, n_layers=2), ignore_extra_model_tensors=True
        )
        (step, tokens, _), warnings = _manager_warnings(manager.load)
        assert (step, tokens) == (STEP, TOKENS)
        assert _holds(model, saved)
        assert optimizer.state
        assert any("model.layers.2." in w and "left unread" in w for w in warnings), warnings

    def test_the_override_does_not_cover_a_size_change(self, tmp_path):
        _save(tmp_path)
        _, _, manager = _run(
            tmp_path, replace(MODEL, vocab_size=128), ignore_extra_model_tensors=True
        )
        with pytest.raises(ValueError, match="model tensors of another size"):
            manager.load()

    def test_the_model_it_was_saved_from_loads_every_tensor(self, tmp_path):
        saved = _save(tmp_path)
        model, optimizer, manager = _run(tmp_path)
        (step, _, _), warnings = _manager_warnings(manager.load)
        assert step == STEP
        assert _holds(model, saved)
        assert optimizer.state
        assert not [w for w in warnings if "model tensors" in w]

    def test_an_index_rank_0_cannot_read_fails_the_load(self, tmp_path, monkeypatch):
        _save(tmp_path)
        _, _, manager = _run(tmp_path)

        def unreadable(dcp_dir):
            raise OSError(f"cannot read {dcp_dir}")

        monkeypatch.setattr("kempnerforge.checkpoint.manager._dcp_keys", unreadable)
        with pytest.raises(RuntimeError, match="Rank 0 could not read the DCP index"):
            manager.load(exclude_keys=["optimizer"])

    @pytest.mark.parametrize("failing", ["_held_model_entries", "_model_misfit"])
    def test_a_check_that_fails_without_a_process_group_raises_its_own_error(
        self, tmp_path, monkeypatch, failing
    ):
        """Listing the model's entries or comparing them fails, with an error that is false in
        a boolean context, as an exception that is also a container can be. The load raises
        that error itself."""
        _save(tmp_path)
        model, _, manager = _run(tmp_path)
        before = _snapshot(model)
        failure = _FalsyError(f"{failing} failed")

        def fail(*_args):
            raise failure

        monkeypatch.setattr(f"kempnerforge.checkpoint.manager.{failing}", fail)
        with pytest.raises(_FalsyError) as caught:
            manager.load()
        assert caught.value is failure
        assert _holds(model, before)


class _FalsyError(TypeError):
    def __bool__(self) -> bool:
        return False


def _failing_comparison(_held, _saved):
    raise TypeError(COMPARISON_FAILURE)


class _ExtraState(torch.nn.Module):
    """A module whose extra state is a dict of tensors of the given sizes, which DCP stores as
    one entry per tensor."""

    def __init__(self, sizes: dict[str, int]) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(2, 2)
        self.sizes = sizes
        self.loaded: dict | None = None

    def get_extra_state(self) -> dict[str, torch.Tensor]:
        return {name: torch.full((size,), 1.0) for name, size in self.sizes.items()}

    def set_extra_state(self, state: dict) -> None:
        self.loaded = state


class TestNestedEntries:
    """An entry that nests, as a module's extra state can, is compared by its leaves."""

    @staticmethod
    def _save(tmp_path, sizes):
        torch.manual_seed(0)
        model = _ExtraState(sizes)
        CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model, _stepped(model)).save(STEP)

    @staticmethod
    def _load(tmp_path, sizes):
        model = _ExtraState(sizes)
        manager = CheckpointManager(
            CheckpointConfig(dir=str(tmp_path)), model, torch.optim.AdamW(model.parameters())
        )
        return model, manager.load

    def test_a_leaf_the_model_does_not_have_is_refused(self, tmp_path):
        self._save(tmp_path, {"kept": 2, "removed": 3})
        model, load = self._load(tmp_path, {"kept": 2})
        with pytest.raises(ValueError, match=r"such as \['model\._extra_state\.removed'\]"):
            load()
        assert model.loaded is None

    def test_a_leaf_of_another_size_is_refused(self, tmp_path):
        self._save(tmp_path, {"kept": 2})
        _, load = self._load(tmp_path, {"kept": 5})
        resized = r"model\._extra_state\.kept \(saved \[2\], this run \[5\]\)"
        with pytest.raises(ValueError, match=resized):
            load()

    def test_the_same_leaves_load(self, tmp_path):
        self._save(tmp_path, {"kept": 2})
        model, load = self._load(tmp_path, {"kept": 2})
        assert load()[0] == STEP
        assert model.loaded is not None and set(model.loaded) == {"kept"}

    def test_an_entry_saved_whole_before_torch_2_4_loads(self, tmp_path, monkeypatch):
        """Before torch 2.4, DCP saved a nested entry that holds no tensor as one value under
        its own key, and its load planner still reads one saved that way."""
        from torch.distributed.checkpoint import _version

        torch.manual_seed(0)
        model = _Counter(7)
        monkeypatch.setattr(_version, "_derived_version", "2_3")
        CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model, _stepped(model)).save(STEP)
        monkeypatch.setattr(_version, "_derived_version", None)
        index = dcp.FileSystemReader(str(tmp_path / f"step_{STEP}")).read_metadata()
        assert "model._extra_state" in index.state_dict_metadata
        loaded = _Counter(0)
        manager = CheckpointManager(
            CheckpointConfig(dir=str(tmp_path)), loaded, torch.optim.AdamW(loaded.parameters())
        )
        assert manager.load(exclude_keys=["optimizer"])[0] == STEP
        assert loaded.counter == 7

    def test_a_state_that_does_not_flatten_raises_its_own_error(self, tmp_path):
        self._save(tmp_path, {"kept": 2})
        model = _ExtraState({"kept": 2})
        model.get_extra_state = lambda: {"a.b": torch.ones(1), "a": {"b": torch.ones(1)}}
        manager = CheckpointManager(
            CheckpointConfig(dir=str(tmp_path)), model, torch.optim.AdamW(model.parameters())
        )
        with pytest.raises(ValueError, match="duplicated flatten key"):
            manager.load()


class _Counter(torch.nn.Module):
    """A module whose extra state is a dict that holds no tensor."""

    def __init__(self, counter: int) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(2, 2)
        self.counter = counter

    def get_extra_state(self) -> dict[str, int]:
        return {"counter": self.counter}

    def set_extra_state(self, state: dict) -> None:
        self.counter = state["counter"]


class TestFreezeWarningsOnRankZero:
    @pytest.mark.parametrize("ignore", [False, True], ids=["unread", "ignored-mismatch"])
    def test_only_rank_0_warns(self, tmp_path, ignore):
        """Every rank decides from the same record; only rank 0 says so."""
        torch.manual_seed(0)
        model = torch.nn.Linear(2, 2)
        config = CheckpointConfig(dir=str(tmp_path), ignore_freeze_mismatch=ignore)
        manager = CheckpointManager(config, model, torch.optim.AdamW(model.parameters()))
        saved = [{"module": "vision_encoder", "frozen": True}]
        expected = [{"module": "vision_encoder", "frozen": False}]
        record = ({"step": 3, "vlm_freeze": saved}, None) if ignore else (None, "no record")
        for rank, warned in ((0, True), (1, False)):
            manager._rank = rank
            _, warnings = _manager_warnings(
                lambda: manager._check_vlm_freeze(record, tmp_path, expected)
            )
            assert bool(warnings) is warned, (rank, warnings)


def _fsdp(config: ModelConfig, mesh) -> torch.nn.Module:
    torch.manual_seed(0)
    model = Transformer(config)
    apply_fsdp2(model, mesh, mp_policy=FP32)
    return model


def _local(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: (value.to_local() if hasattr(value, "to_local") else value).detach().clone()
        for key, value in model.state_dict().items()
    }


def _check_every_rank_refuses_alike(rank: int, world: int, port: int, root: str):
    """Under FSDP2 each rank holds a shard; every rank refuses with rank 0's verdict."""
    with gloo_group(rank, world, port):
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp_shard",))
        model = _fsdp(MODEL, mesh)
        CheckpointManager(CheckpointConfig(dir=root), model, _stepped(model)).save(
            STEP, tokens_seen=TOKENS
        )
        shallow = _fsdp(replace(MODEL, n_layers=2), mesh)
        optimizer = torch.optim.AdamW(shallow.parameters(), lr=0.1)
        manager = CheckpointManager(CheckpointConfig(dir=root), shallow, optimizer)
        before = _local(shallow)
        message = None
        try:
            manager.load()
        except ValueError as error:
            message = str(error)
        messages: list = [None] * world
        dist.all_gather_object(messages, message)
        assert messages[0] is not None, "a 2-layer model loaded a 4-layer checkpoint"
        assert messages == [messages[0]] * world
        assert "model.layers.2." in messages[0]
        assert all(torch.equal(value, before[key]) for key, value in _local(shallow).items())


def _check_a_rank_that_cannot_list_its_model_fails_every_rank(
    rank: int, world: int, port: int, root: str
):
    """Rank 1's state does not flatten. It raises its own error and rank 0 one naming it, and
    neither waits for the other in the model check's collectives."""
    with gloo_group(rank, world, port):
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp_shard",))
        model = _fsdp(MODEL, mesh)
        CheckpointManager(CheckpointConfig(dir=root), model, _stepped(model)).save(
            STEP, tokens_seen=TOKENS
        )
        if rank == 1:

            def duplicated(_state):
                raise ValueError("duplicated flatten key model.a.b")

            manager_module.flatten_state_dict = duplicated
        reloaded = _fsdp(MODEL, mesh)
        optimizer = torch.optim.AdamW(reloaded.parameters(), lr=0.1)
        manager = CheckpointManager(CheckpointConfig(dir=root), reloaded, optimizer)
        outcome = None
        try:
            manager.load()
        except (ValueError, RuntimeError) as error:
            outcome = (type(error).__name__, str(error))
        outcomes: list = [None] * world
        dist.all_gather_object(outcomes, outcome)
        assert outcomes[1] == ("ValueError", "duplicated flatten key model.a.b")
        assert outcomes[0] is not None and outcomes[0][0] == "RuntimeError", outcomes
        expected = "failed on ranks [1]: rank 1: ValueError: duplicated flatten key model.a.b"
        assert expected in outcomes[0][1]


def _check_a_comparison_that_fails_on_rank_0_fails_every_rank(
    rank: int, world: int, port: int, root: str
):
    """Rank 0's comparison raises. Rank 0 raises that error and rank 1 one naming it, and
    neither waits for the other: the loading manager's group gives up after
    ``GROUP_TIMEOUT_S``, so a rank left waiting for the verdict raises that timeout instead."""
    with gloo_group(rank, world, port):
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp_shard",))
        model = _fsdp(MODEL, mesh)
        CheckpointManager(CheckpointConfig(dir=root), model, _stepped(model)).save(
            STEP, tokens_seen=TOKENS
        )
        if rank == 0:
            manager_module._model_misfit = _failing_comparison
        reloaded = _fsdp(MODEL, mesh)
        optimizer = torch.optim.AdamW(reloaded.parameters(), lr=0.1)
        manager = CheckpointManager(
            CheckpointConfig(dir=root), reloaded, optimizer, group_timeout_sec=GROUP_TIMEOUT_S
        )
        before = _local(reloaded)
        outcome = None
        try:
            manager.load()
        except Exception as error:
            outcome = (type(error).__name__, str(error))
        outcomes: list = [None] * world
        dist.all_gather_object(outcomes, outcome)
        assert outcomes[0] == ("TypeError", COMPARISON_FAILURE), outcomes
        expected = (
            "Rank 0 could not compare the checkpoint with this run's model: "
            f"TypeError: {COMPARISON_FAILURE}"
        )
        assert outcomes[1] == ("RuntimeError", expected), outcomes
        assert all(torch.equal(value, before[key]) for key, value in _local(reloaded).items())


def _vlm_job(root: str) -> JobConfig:
    """A VLM run that freezes the vision encoder until step 5 and trains it from then on."""
    return JobConfig(
        model=ModelConfig(dim=64, n_layers=2, n_heads=4, vocab_size=256, max_seq_len=64),
        train=TrainConfig(batch_size=1, seq_len=64, max_steps=10),
        vision_encoder=VisionEncoderConfig(type="random", feature_dim=96, num_tokens=8),
        adapter=AdapterConfig(),
        vlm=VLMConfig(
            max_text_len=32,
            freeze_schedule=[
                FreezeStage(start_step=5, specs=(FreezeSpec("vision_encoder", False),))
            ],
        ),
        checkpoint=CheckpointConfig(dir=root),
    )


def _save_both_sides_of_the_transition(config: JobConfig) -> torch.nn.Module:
    """Steps 3 and 6, each with the freeze ``freeze_meta_at_step`` gives at its step."""
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 4)
    manager = CheckpointManager(config.checkpoint, model, _stepped(model))
    assert config.vlm is not None
    for step in (3, 6):
        manager.save(step, extra={"vlm_freeze": freeze_meta_at_step(step, config.vlm)})
    manager.wait()
    return model


def _resumed(config: JobConfig) -> tuple[int, int]:
    model = torch.nn.Linear(4, 4)
    manager = CheckpointManager(config.checkpoint, model, torch.optim.AdamW(model.parameters()))
    return entry.restore_checkpoint(config, model, None, manager)


class TestFreezeOfTheSettledCheckpoint:
    def test_the_expectation_follows_a_fallback_across_a_freeze_transition(
        self, tmp_path, monkeypatch
    ):
        """The resume resolved step_6, whose shards then turned out incomplete, so load fell
        back to step_3, on the other side of the transition. The expectation used to stay at
        step 6 and refuse the resume."""
        config = _vlm_job(str(tmp_path))
        _save_both_sides_of_the_transition(config)
        (tmp_path / "step_6" / ".metadata").unlink()
        monkeypatch.setattr(entry, "resolve_resume_path", lambda *_: tmp_path / "step_6")
        assert _resumed(config) == (3, 0)

    def test_a_changed_freeze_is_still_refused(self, tmp_path):
        config = _vlm_job(str(tmp_path))
        _save_both_sides_of_the_transition(config)
        assert config.vlm is not None
        config.vlm.freeze_schedule = []
        with pytest.raises(ValueError, match="VLM freeze mismatch"):
            _resumed(config)


def _check_a_rank_that_misses_the_record_resumes_alike(rank: int, world: int, port: int, root: str):
    """Rank 1 misses ``metadata.json`` once, as a transient NFS miss would. Its probe used to
    expect the freeze of step 0 and refuse alone, while rank 0 went on into a collective."""
    with gloo_group(rank, world, port):
        config = _vlm_job(root)
        _save_both_sides_of_the_transition(config)
        dist.barrier()
        if rank == 1:
            read_text = pathlib.Path.read_text
            missed: list[Path] = []

            def miss_once(path, *args, **kwargs):
                if path.name == "metadata.json" and not missed:
                    missed.append(path)
                    raise FileNotFoundError(path)
                return read_text(path, *args, **kwargs)

            pathlib.Path.read_text = miss_once
        outcomes: list = [None] * world
        dist.all_gather_object(outcomes, _resumed(config))
        assert outcomes == [(6, 0)] * world


class TestAcrossRanks:
    def test_every_rank_refuses_a_checkpoint_of_another_model_alike(self, tmp_path):
        spawn(_check_every_rank_refuses_alike, 2, str(tmp_path))

    def test_a_rank_that_misses_the_record_resumes_as_rank_0_read_it(self, tmp_path):
        spawn(_check_a_rank_that_misses_the_record_resumes_alike, 2, str(tmp_path))

    def test_a_rank_that_cannot_list_its_model_fails_every_rank(self, tmp_path):
        spawn(_check_a_rank_that_cannot_list_its_model_fails_every_rank, 2, str(tmp_path))

    def test_a_comparison_that_fails_on_rank_0_fails_every_rank(self, tmp_path):
        spawn(_check_a_comparison_that_fails_on_rank_0_fails_every_rank, 2, str(tmp_path))
