"""The rate, mixture weights and frozen parameters each step runs with follow the schedule.

The step that starts at a data phase's ``start_step`` draws with the phase's ``dataset_weights``
and runs at its ``lr_scale``, and a ``FreezeStage`` applies from its ``start_step``, step 0
included. An optimizer whose state no checkpoint restored, as after a checkpoint saved before
the first update or a ``load_path`` warm start that leaves the optimizer out, steps at the
restored schedule's rate. A resumed run steps with the rate, weights and frozen parameters of
the uninterrupted one. A run without a mixture, on one source or a VLM run, runs at each phase's
``lr_scale`` the same way.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from kempnerforge.checkpoint.manager import CheckpointManager
from kempnerforge.config.adapter import AdapterConfig
from kempnerforge.config.checkpoint import CheckpointConfig
from kempnerforge.config.data import DataConfig, TrainingPhase
from kempnerforge.config.job import JobConfig
from kempnerforge.config.model import ModelConfig
from kempnerforge.config.schema import OptimizerConfig, SchedulerConfig
from kempnerforge.config.training import TrainConfig
from kempnerforge.config.vision import VisionEncoderConfig
from kempnerforge.config.vlm import FreezeSpec, FreezeStage, VLMConfig
from kempnerforge.distributed.parallel import build_parallel_model
from kempnerforge.model.transformer import Transformer
from kempnerforge.training.data_pipeline import DataPipeline, build_phase_state
from kempnerforge.training.freeze import apply_freeze_specs
from kempnerforge.training.loop import (
    StepResult,
    TrainingSession,
    _apply_freeze_stages,
    run_training_loop,
    text_step,
)
from kempnerforge.training.optimizer import build_optimizer
from kempnerforge.training.scheduler import build_scheduler
from tests.unit.test_train_entry import (
    DIM,
    FakeCheckpointManager,
    FakeResumeManager,
    FakeShutdownHandler,
    TinyTextModel,
    _mixture_pipeline,
    _nan_on,
    _runtime,
    _vlm_config,
    make_config,
    make_session,
)


def _frozen(model: nn.Module) -> list[str]:
    return sorted(name for name, p in model.named_parameters() if not p.requires_grad)


class TestPhaseTiming:
    def test_the_step_at_a_phase_start_runs_with_its_weights_and_lr_scale(self):
        """Phases at steps 0 (``a`` weighted 9, ``lr_scale`` 0.5) and 3 (``lr_scale`` 0.25) over
        a constant LR of 0.05: steps 0 to 2 run at 0.025 on ``[9, 3]``, steps 3 to 5 at 0.0125
        on the declared ``[1, 3]``."""
        config = make_config(max_steps=6)
        config.data = DataConfig(
            phases=[
                TrainingPhase(start_step=0, dataset_weights={"a": 9.0}, lr_scale=0.5),
                TrainingPhase(start_step=3, lr_scale=0.25),
            ]
        )
        data = _mixture_pipeline()
        seen: list[tuple[int, float, list[float] | None]] = []

        def step_fn(session: TrainingSession, step: int) -> StepResult:
            calls = session.data.mixture_sampler.calls  # type: ignore[union-attr]
            lr = session.optimizer.param_groups[0]["lr"]
            seen.append((step, lr, calls[-1][0] if calls else None))
            return text_step(session, step)

        torch.manual_seed(0)
        phases = build_phase_state(config, data, step=0)
        run_training_loop(make_session(config, data=data, phases=phases, step_fn=step_fn))
        early, late = (pytest.approx(0.025), [9.0, 3.0]), (pytest.approx(0.0125), [1.0, 3.0])
        assert seen == [(step, *(early if step < 3 else late)) for step in range(6)]


class TestFreezeStageAtStepZero:
    def test_a_fresh_run_freezes_what_a_resume_at_the_same_step_freezes(
        self, tmp_path, monkeypatch
    ):
        """A stage at step 0 that freezes the adapter holds from a fresh run's first step, as it
        does after a resume."""
        import kempnerforge.training.entry as entry

        vlm = VLMConfig(max_text_len=8, freeze_schedule=[FreezeStage(0, (FreezeSpec("adapter"),))])
        model_config = ModelConfig(
            dim=32, n_layers=2, n_heads=2, n_kv_heads=2, vocab_size=64, max_seq_len=16
        )
        vision = VisionEncoderConfig(type="random", feature_dim=16, num_tokens=4)
        config = JobConfig(
            model=model_config,
            train=TrainConfig(batch_size=1, seq_len=16, max_steps=4),
            vision_encoder=vision,
            adapter=AdapterConfig(),
            vlm=vlm,
            checkpoint=CheckpointConfig(dir=str(tmp_path / "fresh")),
        )

        def build() -> Any:
            torch.manual_seed(0)
            return build_parallel_model(
                model_config,
                torch.device("cpu"),
                None,
                vision_config=vision,
                adapter_config=AdapterConfig(),
                vlm_config=vlm,
            )

        fresh = build()
        assert entry.restore_checkpoint(config, fresh, None, FakeResumeManager()) == (0, 0)
        session = SimpleNamespace(config=config, checkpointer=FakeCheckpointManager(), model=fresh)
        for step in (1, 2, 3):
            _apply_freeze_stages(session, step)  # type: ignore[arg-type]

        monkeypatch.setattr(entry, "resolve_resume_path", lambda _d, _n=None: tmp_path / "step_3")
        resumed = build()
        resume = FakeResumeManager(step=3, tokens=0, saved_step=3)
        assert entry.restore_checkpoint(config, resumed, None, resume) == (3, 0)
        assert not any(p.requires_grad for p in fresh.adapter.parameters())
        assert _frozen(fresh) == _frozen(resumed)


class TestRestoreWithoutOptimizerState:
    def test_a_checkpoint_saved_before_the_first_update_resumes_at_its_steps_lr(self, tmp_path):
        """Steps 0 and 1 are skipped for a NaN loss and a shutdown saves step 2, before any
        update. Over a 4-step linear warmup to 0.05 the resumed step 2 runs at 0.025."""

        def run(ckpt_dir: Path, shutdown_after: int | None = None, resume: bool = False):
            config = make_config(max_steps=5)
            config.checkpoint.dir = str(ckpt_dir)
            config.checkpoint.interval = 100
            model = TinyTextModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: s / 4)
            lrs: list[tuple[int, float]] = []
            body = _nan_on({0, 1})

            def step_fn(session: TrainingSession, step: int) -> StepResult:
                lrs.append((step, session.optimizer.param_groups[0]["lr"]))
                return body(session, step)

            session = make_session(
                config,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step_fn=step_fn,
                shutdown_handler=FakeShutdownHandler(after_steps=shutdown_after),
            )
            session.checkpointer = CheckpointManager(config.checkpoint, model, optimizer)
            step = tokens = 0
            if resume:
                step, tokens, _ = session.checkpointer.load(scheduler=scheduler)
                assert not session.checkpointer.restored_optimizer
            run_training_loop(session, step=step, tokens_seen=tokens)
            return lrs

        uninterrupted = run(tmp_path / "full")
        run(tmp_path / "split", shutdown_after=2)
        resumed = run(tmp_path / "split", resume=True)
        assert [step for step, _ in uninterrupted] == [0, 1, 2, 3, 4]
        assert [lr for _, lr in uninterrupted] == pytest.approx([0.0, 0.0125, 0.025, 0.0375, 0.05])
        assert resumed == uninterrupted[2:]

    @pytest.mark.parametrize("warmup_steps", [5, 0])
    @pytest.mark.parametrize("exclude", [None, ["optimizer"]], ids=["full", "no_optimizer"])
    def test_a_load_runs_its_first_step_at_the_restored_schedules_lr(
        self, tmp_path, warmup_steps, exclude
    ):
        """A step-20 checkpoint of a 40-step cosine run, loaded with or without its optimizer:
        step 20 runs at the schedule's rate, not at the one the scheduler set when it was
        built (0 under warmup, the peak without)."""

        def build() -> tuple[Transformer, torch.optim.Optimizer, Any]:
            torch.manual_seed(0)
            model = Transformer(
                ModelConfig(dim=32, n_layers=1, n_heads=2, vocab_size=64, max_seq_len=16)
            )
            optimizer = build_optimizer(model, OptimizerConfig(name="adamw", lr=1e-3, fused=False))
            return (
                model,
                optimizer,
                build_scheduler(optimizer, SchedulerConfig(warmup_steps=warmup_steps), 40),
            )

        def step_fn(session: TrainingSession, step: int) -> StepResult:
            session.model(torch.randint(0, 64, (2, 16))).float().pow(2).mean().backward()
            return StepResult(loss=1.0, grad_norm=0.0, text_tokens=1)

        model, optimizer, scheduler = build()
        for _ in range(20):
            step_fn(SimpleNamespace(model=model), 0)  # type: ignore[arg-type]
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model, optimizer).save(
            20, scheduler=scheduler
        )

        model, optimizer, scheduler = build()
        built_at = optimizer.param_groups[0]["lr"]
        manager = CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model, optimizer)
        manager.load(path=str(tmp_path / "step_20"), scheduler=scheduler, exclude_keys=exclude)
        assert manager.restored_optimizer is (exclude is None)
        scheduled = scheduler.get_last_lr()[0]
        applied: list[float] = []

        def recording_step(session: TrainingSession, step: int) -> StepResult:
            applied.append(session.optimizer.param_groups[0]["lr"])
            return step_fn(session, step)

        session = make_session(
            make_config(max_steps=21),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step_fn=recording_step,
            checkpointer=manager,
        )
        run_training_loop(session, step=20)
        assert scheduled == pytest.approx(6.5e-4 if warmup_steps else 5.5e-4, rel=1e-3)
        assert built_at != pytest.approx(scheduled)
        assert applied == [scheduled]

    def test_a_restored_optimizer_keeps_the_rate_it_was_saved_at(self, tmp_path):
        """Its checkpoint holds the optimizer at a rate other than the schedule's, 0.123 for
        0.05: the loop steps at the restored rate."""
        config = make_config(max_steps=2)
        first = make_session(config)
        first.optimizer.param_groups[0]["momentum"] = 0.9
        first.model(torch.zeros(1, 1, dtype=torch.long)).sum().backward()
        first.optimizer.step()
        first.optimizer.zero_grad()
        first.scheduler.step()
        first.optimizer.param_groups[0]["lr"] = 0.123
        saving = CheckpointManager(
            CheckpointConfig(dir=str(tmp_path)), first.model, first.optimizer
        )
        saving.save(1, scheduler=first.scheduler)

        second = make_session(config)
        manager = CheckpointManager(
            CheckpointConfig(dir=str(tmp_path)), second.model, second.optimizer
        )
        step, tokens, _ = manager.load(path=str(tmp_path / "step_1"), scheduler=second.scheduler)
        assert manager.restored_optimizer
        applied: list[float] = []

        def step_fn(session: TrainingSession, step: int) -> StepResult:
            applied.append(session.optimizer.param_groups[0]["lr"])
            return text_step(session, step)

        second.step_fn = step_fn
        second.checkpointer = manager
        run_training_loop(second, step=step, tokens_seen=tokens)
        assert applied == [0.123]

    def test_an_optimizer_that_holds_state_when_built_starts_on_the_schedule(self, tmp_path):
        """``torch.optim.Adagrad`` holds state from construction. A fresh run with a phase at
        step 0 starts at the phase's scale, and a warm start that leaves the optimizer out at
        the restored schedule's rate, 0.025 over a 4-step warmup to 0.05 at step 2."""

        def adagrad(factor: Any) -> tuple[nn.Module, torch.optim.Optimizer, Any]:
            torch.manual_seed(0)
            model = TinyTextModel()
            optimizer = torch.optim.Adagrad(model.parameters(), lr=0.05)
            return model, optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, factor)

        def first_rate(session: TrainingSession, step: int) -> float:
            applied: list[float] = []

            def step_fn(s: TrainingSession, i: int) -> StepResult:
                applied.append(s.optimizer.param_groups[0]["lr"])
                return text_step(s, i)

            session.step_fn = step_fn
            run_training_loop(session, step=step)
            return applied[0]

        config = make_config(max_steps=3)
        config.data = DataConfig(phases=[TrainingPhase(start_step=0, lr_scale=0.5)])
        data = _mixture_pipeline()
        model, optimizer, scheduler = adagrad(lambda _s: 1.0)
        assert optimizer.state
        fresh = make_session(
            config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            data=data,
            phases=build_phase_state(config, data, step=0),
        )
        assert first_rate(fresh, 0) == pytest.approx(0.025)

        model, optimizer, scheduler = adagrad(lambda s: s / 4)
        for _ in range(2):
            model(torch.zeros(1, 1, dtype=torch.long)).sum().backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model, optimizer).save(
            2, scheduler=scheduler
        )
        model, optimizer, scheduler = adagrad(lambda s: s / 4)
        manager = CheckpointManager(CheckpointConfig(dir=str(tmp_path)), model, optimizer)
        step, _, _ = manager.load(
            path=str(tmp_path / "step_2"), scheduler=scheduler, exclude_keys=["optimizer"]
        )
        warm = make_session(
            make_config(max_steps=3),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpointer=manager,
        )
        assert optimizer.state and not manager.restored_optimizer
        assert first_rate(warm, step) == pytest.approx(0.025)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_encoder = nn.Linear(DIM, DIM)
        self.adapter = nn.Linear(DIM, DIM)
        self.transformer = nn.Linear(DIM, DIM)


_Record = tuple[list[float], list[float] | None, list[str]]


class TestResumeParity:
    """``run_training`` on a VLM config with a data phase and a freeze stage at step 0 and at
    step 3: the build freezes the vision encoder, stage 0 the adapter, and stage 3 unfreezes
    both. Each record is the rate of every param group, the mixture weights last set and the
    frozen parameters as the step starts."""

    MAX_STEPS = 6

    @staticmethod
    def _config() -> JobConfig:
        config = _vlm_config(
            freeze_schedule=(
                FreezeStage(start_step=0, specs=(FreezeSpec("adapter"),)),
                FreezeStage(
                    start_step=3,
                    specs=(FreezeSpec("adapter", False), FreezeSpec("vision_encoder", False)),
                ),
            )
        )
        config.data = DataConfig(
            phases=[
                TrainingPhase(start_step=0, dataset_weights={"a": 9.0}, lr_scale=0.5),
                TrainingPhase(start_step=3, dataset_weights={"b": 7.0}, lr_scale=0.25),
            ]
        )
        config.optimizer.fused = False
        config.scheduler.warmup_steps = 2
        config.scheduler.warmup_start_factor = 0.1
        config.scheduler.decay_steps = 10
        return config

    @classmethod
    def _train(
        cls,
        monkeypatch: pytest.MonkeyPatch,
        ckpt_dir: Path,
        max_steps: int,
        *,
        load_path: Path | None = None,
        exclude: tuple[str, ...] = (),
        nan_steps: frozenset[int] = frozenset(),
        shutdown_after: int | None = None,
    ) -> tuple[dict[int, _Record], dict[str, torch.Tensor]]:
        """Run ``run_training`` and return each step's record and the parameters it ended with."""
        import kempnerforge.training.entry as entry

        config = cls._config()
        config.train.max_steps = max_steps
        config.checkpoint = CheckpointConfig(
            dir=str(ckpt_dir),
            interval=1000,
            load_path=str(load_path) if load_path else None,
            exclude_from_loading=list(exclude),
        )
        torch.manual_seed(0)
        model = _Model()
        assert config.vlm is not None
        apply_freeze_specs(model, config.vlm.freeze, config.vlm.module_patterns)
        data = _mixture_pipeline()
        monkeypatch.setattr(entry, "setup_distributed", lambda _config: _runtime())
        monkeypatch.setattr(
            entry, "ShutdownHandler", lambda **_kw: FakeShutdownHandler(after_steps=shutdown_after)
        )
        monkeypatch.setattr(entry, "build_model", lambda *_a, **_k: (model, None))
        monkeypatch.setattr(entry, "build_profiler", lambda *_a, **_k: None)
        monkeypatch.setattr(entry, "build_data_pipeline", lambda *_a, **_k: data)
        monkeypatch.setattr(entry, "build_eval_dataloader", lambda *_a, **_k: None)
        monkeypatch.setattr(entry, "destroy_distributed", lambda: None)
        records: dict[int, _Record] = {}

        def step_fn(session: TrainingSession, step: int) -> StepResult:
            calls = data.mixture_sampler.calls  # type: ignore[union-attr]
            records[step] = (
                [group["lr"] for group in session.optimizer.param_groups],
                calls[-1][0] if calls else None,
                _frozen(model),
            )
            loss = model.transformer(model.adapter(model.vision_encoder(torch.ones(2, DIM))))
            loss = loss.square().mean()
            loss.backward()
            return StepResult(
                loss=math.nan if step in nan_steps else loss.item(), grad_norm=0.0, text_tokens=1
            )

        entry.run_training(config, step_fn=step_fn)
        return records, {name: p.detach().clone() for name, p in model.named_parameters()}

    def test_a_fresh_run_starts_with_the_step_zero_phase_and_stage(self, tmp_path, monkeypatch):
        records, _ = self._train(monkeypatch, tmp_path / "run", self.MAX_STEPS)
        probe = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=self._config().optimizer.lr)
        scheduler = build_scheduler(probe, self._config().scheduler, self.MAX_STEPS)
        scheduled = []
        for _ in range(self.MAX_STEPS):
            scheduled.append(scheduler.get_last_lr()[0])
            probe.step()
            scheduler.step()
        for step, (lrs, weights, frozen) in records.items():
            phase_scale, phase_weights = (0.5, [9.0, 3.0]) if step < 3 else (0.25, [1.0, 7.0])
            assert lrs == [pytest.approx(scheduled[step] * phase_scale)] * len(lrs), step
            assert weights == phase_weights, step
            expected = [
                "adapter.bias",
                "adapter.weight",
                "vision_encoder.bias",
                "vision_encoder.weight",
            ]
            assert frozen == (expected if step < 3 else []), step

    @pytest.mark.parametrize("stop", [1, 2, 3, 4], ids=lambda s: f"saved_at_{s}")
    def test_a_resume_steps_as_the_uninterrupted_run(self, tmp_path, monkeypatch, stop):
        """Saved before (1, 2), at (3) and after (4) the step-3 boundary, with optimizer state."""
        full, full_params = self._train(monkeypatch, tmp_path / "full", self.MAX_STEPS)
        self._train(monkeypatch, tmp_path / "split", stop)
        resumed, resumed_params = self._train(monkeypatch, tmp_path / "split", self.MAX_STEPS)
        assert sorted(resumed) == list(range(stop, self.MAX_STEPS))
        assert {step: full[step] for step in resumed} == resumed
        for name, param in full_params.items():
            assert torch.equal(resumed_params[name], param), name

    def test_a_resume_from_a_save_before_the_first_update(self, tmp_path, monkeypatch):
        """Steps 0 and 1 are skipped for a NaN loss, and a shutdown saves step 2 with no
        optimizer state: the resumed run steps on as the uninterrupted one does."""
        nan_steps = frozenset({0, 1})
        full, full_params = self._train(
            monkeypatch, tmp_path / "full", self.MAX_STEPS, nan_steps=nan_steps
        )
        self._train(
            monkeypatch, tmp_path / "split", self.MAX_STEPS, nan_steps=nan_steps, shutdown_after=2
        )
        assert (tmp_path / "split" / "step_2").is_dir()
        resumed, resumed_params = self._train(monkeypatch, tmp_path / "split", self.MAX_STEPS)
        assert sorted(resumed) == list(range(2, self.MAX_STEPS))
        assert {step: full[step] for step in resumed} == resumed
        for name, param in full_params.items():
            assert torch.equal(resumed_params[name], param), name

    @pytest.mark.parametrize("stop", [2, 3, 4], ids=lambda s: f"from_{s}")
    def test_a_warm_start_without_the_optimizer_steps_on_the_schedule(
        self, tmp_path, monkeypatch, stop
    ):
        """A ``load_path`` warm start that leaves the optimizer out runs each step at the
        uninterrupted run's rate, weights and frozen parameters, though its moments restart."""
        full, _ = self._train(monkeypatch, tmp_path / "full", self.MAX_STEPS)
        self._train(monkeypatch, tmp_path / "first", stop)
        warm, _ = self._train(
            monkeypatch,
            tmp_path / "second",
            self.MAX_STEPS,
            load_path=tmp_path / "first" / f"step_{stop}",
            exclude=("optimizer",),
        )
        assert sorted(warm) == list(range(stop, self.MAX_STEPS))
        assert {step: full[step] for step in warm} == warm


class TestPhaseWithoutAMixture:
    """A run whose pipeline builds no mixture, on one source or a VLM run, steps at each data
    phase's ``lr_scale`` from its ``start_step`` and keeps its data iterator. The ``run_training``
    runs have phases at step 0 (``lr_scale`` 0.5) and step 3 (0.25), and the VLM run trains the
    adapter at 0.2 of the LR. Each record is the rate of every param group as the step starts."""

    MAX_STEPS = 6

    def test_a_single_source_run_steps_at_each_phases_lr_scale(self):
        """Phases at steps 0 (``lr_scale`` 0.5) and 2 (0.25) over a constant LR of 0.05: steps 0
        and 1 run at 0.025 and steps 2 and 3 at 0.0125, each on the source's next batch."""
        config = make_config(max_steps=4)
        config.data = DataConfig(
            phases=[
                TrainingPhase(start_step=0, lr_scale=0.5),
                TrainingPhase(start_step=2, lr_scale=0.25),
            ]
        )
        data = DataPipeline(dataloader=[{"input_ids": torch.full((2, 4), i)} for i in range(4)])
        seen: list[tuple[int, float, int]] = []

        def step_fn(session: TrainingSession, step: int) -> StepResult:
            batch = session.batches.next_batch()
            lr = session.optimizer.param_groups[0]["lr"]
            seen.append((step, lr, int(batch["input_ids"][0, 0])))
            session.model(batch["input_ids"]).float().pow(2).mean().backward()
            return StepResult(loss=1.0, grad_norm=0.0, text_tokens=1)

        phases = build_phase_state(config, data, step=0)
        run_training_loop(make_session(config, data=data, phases=phases, step_fn=step_fn))
        rates = [0.025, 0.025, 0.0125, 0.0125]
        assert seen == [(step, pytest.approx(rate), step) for step, rate in enumerate(rates)]

    @staticmethod
    def _config(kind: str) -> JobConfig:
        config = make_config() if kind == "text" else _vlm_config()
        config.data = DataConfig(
            phases=[
                TrainingPhase(start_step=0, lr_scale=0.5),
                TrainingPhase(start_step=3, lr_scale=0.25),
            ]
        )
        config.optimizer.fused = False
        config.scheduler.warmup_steps = 2
        config.scheduler.warmup_start_factor = 0.1
        config.scheduler.decay_steps = 10
        if config.vlm is not None:
            config.optimizer.lr_multipliers = {"adapter": 0.2}
        return config

    @classmethod
    def _train(
        cls,
        monkeypatch: pytest.MonkeyPatch,
        kind: str,
        ckpt_dir: Path,
        max_steps: int,
        *,
        load_path: Path | None = None,
        exclude: tuple[str, ...] = (),
    ) -> tuple[dict[int, list[float]], dict[str, torch.Tensor]]:
        """Run ``run_training`` and return each step's rates and the parameters it ended with."""
        import kempnerforge.training.entry as entry

        config = cls._config(kind)
        config.train.max_steps = max_steps
        config.checkpoint = CheckpointConfig(
            dir=str(ckpt_dir),
            interval=1000,
            load_path=str(load_path) if load_path else None,
            exclude_from_loading=list(exclude),
        )
        torch.manual_seed(0)
        model = _Model()
        if config.vlm is not None:
            apply_freeze_specs(model, config.vlm.freeze, config.vlm.module_patterns)
        monkeypatch.setattr(entry, "setup_distributed", lambda _config: _runtime())
        monkeypatch.setattr(entry, "ShutdownHandler", lambda **_kw: FakeShutdownHandler())
        monkeypatch.setattr(entry, "build_model", lambda *_a, **_k: (model, None))
        monkeypatch.setattr(entry, "build_profiler", lambda *_a, **_k: None)
        monkeypatch.setattr(entry, "build_data_pipeline", lambda *_a, **_k: DataPipeline())
        monkeypatch.setattr(entry, "build_eval_dataloader", lambda *_a, **_k: None)
        monkeypatch.setattr(entry, "destroy_distributed", lambda: None)
        records: dict[int, list[float]] = {}

        def step_fn(session: TrainingSession, step: int) -> StepResult:
            records[step] = [group["lr"] for group in session.optimizer.param_groups]
            loss = model.transformer(model.adapter(model.vision_encoder(torch.ones(2, DIM))))
            loss = loss.square().mean()
            loss.backward()
            return StepResult(loss=loss.item(), grad_norm=0.0, text_tokens=1)

        entry.run_training(config, step_fn=step_fn)
        return records, {name: p.detach().clone() for name, p in model.named_parameters()}

    @pytest.mark.parametrize("kind", ["text", "vlm"])
    def test_a_fresh_run_steps_at_the_schedule_times_each_phases_scale(
        self, tmp_path, monkeypatch, kind
    ):
        """Every param group runs at the schedule's rate times its LR multiplier times the
        active phase's ``lr_scale``; the config passes validation."""
        config = self._config(kind)
        config.validate(world_size=1)
        records, _ = self._train(monkeypatch, kind, tmp_path / "run", self.MAX_STEPS)
        probe = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=config.optimizer.lr)
        scheduler = build_scheduler(probe, config.scheduler, self.MAX_STEPS)
        multipliers = [0.2, 1.0] if kind == "vlm" else [1.0]
        assert sorted(records) == list(range(self.MAX_STEPS))
        for step, rates in records.items():
            phase_scale = 0.5 if step < 3 else 0.25
            scheduled = scheduler.get_last_lr()[0] * phase_scale
            assert sorted({round(rate / scheduled, 9) for rate in rates}) == multipliers, step
            probe.step()
            scheduler.step()

    @pytest.mark.parametrize(
        ("kind", "stop"),
        [("text", 2), ("text", 3), ("text", 4), ("vlm", 3)],
        ids=lambda value: str(value),
    )
    def test_a_resume_steps_as_the_uninterrupted_run(self, tmp_path, monkeypatch, kind, stop):
        """Saved before (2), at (3) and after (4) the step-3 boundary, with optimizer state."""
        full, full_params = self._train(monkeypatch, kind, tmp_path / "full", self.MAX_STEPS)
        self._train(monkeypatch, kind, tmp_path / "split", stop)
        resumed, resumed_params = self._train(monkeypatch, kind, tmp_path / "split", self.MAX_STEPS)
        assert sorted(resumed) == list(range(stop, self.MAX_STEPS))
        assert {step: full[step] for step in resumed} == resumed
        for name, param in full_params.items():
            assert torch.equal(resumed_params[name], param), name

    def test_a_warm_start_without_the_optimizer_steps_at_the_phase_lr_scale(
        self, tmp_path, monkeypatch
    ):
        """A ``load_path`` warm start from step 3 that leaves the optimizer out runs each step
        at the uninterrupted run's rates."""
        full, _ = self._train(monkeypatch, "text", tmp_path / "full", self.MAX_STEPS)
        self._train(monkeypatch, "text", tmp_path / "first", 3)
        warm, _ = self._train(
            monkeypatch,
            "text",
            tmp_path / "second",
            self.MAX_STEPS,
            load_path=tmp_path / "first" / "step_3",
            exclude=("optimizer",),
        )
        assert sorted(warm) == list(range(3, self.MAX_STEPS))
        assert {step: full[step] for step in warm} == warm
