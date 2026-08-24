"""Unit tests for the extracted training entry point.

These cover the seams that only became reachable without a GPU once
``scripts/train.py:main()`` was decomposed: the step loop driven by a fake
checkpoint manager, the batch iterator, phase scheduling, the step-body
dispatch, and the data-pipeline builder.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
import torch
from torch import nn

from kempnerforge.config.checkpoint import CheckpointConfig, DynamicCheckpointWindow
from kempnerforge.config.data import DataConfig, TrainingPhase
from kempnerforge.config.job import JobConfig
from kempnerforge.config.model import ModelConfig
from kempnerforge.config.training import TrainConfig
from kempnerforge.resilience.health import NaNDetector
from kempnerforge.training.data_pipeline import (
    DataPipeline,
    PhaseState,
    advance_phases,
    build_data_pipeline,
    build_phase_state,
)
from kempnerforge.training.hooks import HookRunner, TrainingHook
from kempnerforge.training.loop import (
    BatchStream,
    TrainingSession,
    checkpoint_extra,
    pipeline_step,
    run_training_loop,
    select_step_fn,
    text_step,
    vlm_step,
)
from kempnerforge.training.runtime import RuntimeContext

VOCAB = 16
DIM = 8


def _fake_batches(n: int) -> list[dict[str, torch.Tensor]]:
    return [{"input_ids": torch.tensor([i])} for i in range(n)]


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class TinyTextModel(nn.Module):
    """(B, T) ids -> (B, T, V) logits; the contract the text step body uses."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, DIM)
        self.out = nn.Linear(DIM, VOCAB)

    def forward(self, input_ids: torch.Tensor, doc_ids: Any = None) -> torch.Tensor:
        return self.out(self.embed(input_ids))


class FakeCheckpointManager:
    def __init__(self) -> None:
        self.saves: list[tuple[int, int, dict]] = []
        self.waits = 0
        self.flushes = 0

    def save(self, step, tokens_seen=0, scheduler=None, dataloader=None, extra=None) -> None:
        self.saves.append((step, tokens_seen, dict(extra or {})))

    def wait(self) -> None:
        self.waits += 1

    def flush_pending_save(self) -> None:
        self.flushes += 1

    @property
    def saved_steps(self) -> list[int]:
        return [s for s, _, _ in self.saves]


class FakeShutdownHandler:
    """Requests shutdown once ``after_steps`` checks have been made."""

    def __init__(self, after_steps: int | None = None) -> None:
        self.after_steps = after_steps
        self.checks = 0
        self.registered = False
        self.finished = False

    def register(self) -> None:
        self.registered = True

    def should_shutdown(self) -> bool:
        self.checks += 1
        return self.after_steps is not None and self.checks >= self.after_steps

    def finish(self) -> None:
        self.finished = True


class RecordingHook(TrainingHook):
    def __init__(self) -> None:
        self.begins = 0
        self.ends: list[tuple[int, int]] = []
        self.steps: list[int] = []
        self.saves: list[int] = []

    def on_train_begin(self, config: JobConfig) -> None:
        self.begins += 1

    def on_step_end(self, ctx) -> None:
        self.steps.append(ctx.step)

    def on_checkpoint_save(self, step: int, path: str) -> None:
        self.saves.append(step)

    def on_train_end(self, step: int, tokens_seen: int) -> None:
        self.ends.append((step, tokens_seen))


class FakeMixtureDataset:
    def __init__(self, names: list[str]) -> None:
        self.dataset_names = names


class FakeMixtureSampler:
    def __init__(self) -> None:
        self.calls: list[tuple[list[float], float]] = []

    def update_weights(self, weights, temperature) -> None:
        self.calls.append((list(weights), temperature))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_config(**train_kwargs: Any) -> JobConfig:
    train = {
        "batch_size": 2,
        "seq_len": 4,
        "max_steps": 4,
        "grad_accum_steps": 1,
        "compile_model": False,
    }
    train.update(train_kwargs)
    return JobConfig(
        model=ModelConfig(dim=DIM, n_layers=1, n_heads=1, vocab_size=VOCAB, max_seq_len=64),
        train=TrainConfig(**train),  # type: ignore[arg-type]
        checkpoint=CheckpointConfig(dir="unused", interval=2),
    )


def make_session(config: JobConfig, **overrides: Any) -> TrainingSession:
    model = TinyTextModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _s: 1.0)
    kwargs: dict[str, Any] = {
        "config": config,
        "runtime": _runtime(),
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "loss_fn": lambda logits, labels: torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
        ),
        "step_fn": select_step_fn(config),
        "data": DataPipeline(),
        "phases": PhaseState(),
        "checkpointer": FakeCheckpointManager(),
        "tracker": _make_tracker(config),
        "hooks": HookRunner(),
        "nan_detector": NaNDetector(action="warn", max_consecutive=10),
        "shutdown_handler": FakeShutdownHandler(),
    }
    kwargs.update(overrides)
    return TrainingSession(**kwargs)


def _runtime() -> RuntimeContext:
    return RuntimeContext(
        rank=0,
        local_rank=0,
        world_size=1,
        device=torch.device("cpu"),
        device_mesh=None,
    )


def _make_tracker(config: JobConfig):
    from kempnerforge.metrics.tracker import MetricsTracker

    return MetricsTracker(config, num_gpus=1)


# ---------------------------------------------------------------------------
# BatchStream
# ---------------------------------------------------------------------------


class TestBatchStream:
    def test_no_dataloader_reports_no_data(self):
        stream = BatchStream(DataPipeline())
        stream.ensure_started()
        assert stream.has_data is False

    def test_restarts_iterator_at_epoch_boundary(self):
        stream = BatchStream(DataPipeline(dataloader=_fake_batches(2)))
        seen = [int(stream.next_batch()["input_ids"]) for _ in range(5)]
        assert seen == [0, 1, 0, 1, 0]

    def test_next_batch_starts_the_iterator_itself(self):
        # A step body called directly (an example owning its own step_fn)
        # must not have to remember ensure_started().
        stream = BatchStream(DataPipeline(dataloader=_fake_batches(3)))
        assert int(stream.next_batch()["input_ids"]) == 0

    def test_reset_forces_a_fresh_iterator(self):
        stream = BatchStream(DataPipeline(dataloader=_fake_batches(3)))
        assert int(stream.next_batch()["input_ids"]) == 0
        stream.reset()
        assert int(stream.next_batch()["input_ids"]) == 0

    def test_ensure_started_is_idempotent(self):
        stream = BatchStream(DataPipeline(dataloader=_fake_batches(3)))
        stream.ensure_started()
        assert int(stream.next_batch()["input_ids"]) == 0
        stream.ensure_started()
        assert int(stream.next_batch()["input_ids"]) == 1

    def test_empty_shard_raises_a_named_error(self):
        # A bare StopIteration here would kill one rank and leave the others
        # blocked in the next collective until the PG timeout.
        stream = BatchStream(DataPipeline(dataloader=[]))
        with pytest.raises(RuntimeError, match="no batches for this rank"):
            stream.next_batch()

    def test_follows_a_swapped_dataloader(self):
        pipeline = DataPipeline(dataloader=_fake_batches(3))
        stream = BatchStream(pipeline)
        assert int(stream.next_batch()["input_ids"]) == 0
        pipeline.dataloader = [{"input_ids": torch.tensor([99])}]
        stream.ensure_started()
        assert int(stream.next_batch()["input_ids"]) == 99


# ---------------------------------------------------------------------------
# Step-body dispatch
# ---------------------------------------------------------------------------


class TestSessionBatches:
    def test_follows_a_replaced_data_pipeline(self):
        # Replacing session.data must not leave the loop on the synthetic
        # random-token path while a real dataloader sits unused.
        session = make_session(make_config())
        assert session.batches.has_data is False
        session.data = DataPipeline(dataloader=_fake_batches(2))
        assert session.batches.has_data is True
        assert int(session.batches.next_batch()["input_ids"]) == 0


class TestDataPipelineInvariants:
    def test_rejects_a_mixture_without_a_sampler(self):
        with pytest.raises(ValueError, match="must be set together"):
            DataPipeline(mixture_dataset=FakeMixtureDataset(["a"]))  # type: ignore[arg-type]

    def test_rejects_weights_that_miss_a_dataset(self):
        with pytest.raises(ValueError, match="must cover every"):
            DataPipeline(
                mixture_dataset=FakeMixtureDataset(["a", "b"]),  # type: ignore[arg-type]
                mixture_sampler=FakeMixtureSampler(),  # type: ignore[arg-type]
                mixture_weights={"a": 1.0},
            )


class TestSelectStepFn:
    def test_text_only(self):
        assert select_step_fn(make_config()) is text_step

    def test_pipeline_wins_over_everything(self):
        config = make_config()
        config.distributed.pp = 2
        assert select_step_fn(config) is pipeline_step

    def test_vlm(self, tiny_vlm_configs):
        mc, vision, adapter, vlm = tiny_vlm_configs
        config = JobConfig(
            model=mc,
            train=TrainConfig(batch_size=1, seq_len=64, max_steps=1),
            vision_encoder=vision,
            adapter=adapter,
            vlm=vlm,
        )
        assert select_step_fn(config) is vlm_step


class TestTextStep:
    def test_synthetic_batches_produce_gradients(self):
        config = make_config(grad_accum_steps=2)
        session = make_session(config)
        result = session.step_fn(session, 0)
        assert result.loss > 0
        assert result.grad_norm > 0
        assert result.dataset_loss_sums == {}
        assert all(p.grad is not None for p in session.model.parameters())

    def test_reads_from_the_dataloader_when_present(self):
        config = make_config()
        batches = [
            {
                "input_ids": torch.zeros(2, 4, dtype=torch.long),
                "labels": torch.zeros(2, 4, dtype=torch.long),
            }
        ]
        session = make_session(config, data=DataPipeline(dataloader=batches))
        result = session.step_fn(session, 0)
        assert result.loss > 0


class TestVlmTextTokenGuard:
    def test_missing_count_is_loud(self, tiny_vlm_configs):
        from kempnerforge.training.loop import StepResult, _log_periodic_metrics

        mc, vision, adapter, vlm = tiny_vlm_configs
        config = JobConfig(
            model=mc,
            train=TrainConfig(batch_size=1, seq_len=64, max_steps=1),
            vision_encoder=vision,
            adapter=adapter,
            vlm=vlm,
        )
        session = make_session(make_config())
        session.config = config
        with pytest.raises(RuntimeError, match="no text-token count"):
            _log_periodic_metrics(session, StepResult(loss=1.0, grad_norm=0.1), 1)


# ---------------------------------------------------------------------------
# Checkpoint metadata
# ---------------------------------------------------------------------------


class TestCheckpointExtra:
    def test_empty_without_phases_or_run_ids(self):
        assert checkpoint_extra(make_config(), 5, PhaseState()) == {}

    def test_includes_phase_index_when_phases_configured(self):
        phases = PhaseState(phases=[TrainingPhase(start_step=1)], next_idx=1)
        assert checkpoint_extra(make_config(), 5, phases) == {"phase_idx": 1}

    def test_includes_metrics_run_ids(self):
        config = make_config()
        config.metrics.wandb_run_id = "wid"
        config.metrics.mlflow_run_id = "mid"
        extra = checkpoint_extra(config, 5, PhaseState())
        assert extra == {"wandb_run_id": "wid", "mlflow_run_id": "mid"}

    def test_includes_vlm_freeze_metadata(self, tiny_vlm_configs):
        mc, vision, adapter, vlm = tiny_vlm_configs
        config = JobConfig(
            model=mc,
            train=TrainConfig(batch_size=1, seq_len=64, max_steps=1),
            vision_encoder=vision,
            adapter=adapter,
            vlm=vlm,
        )
        assert "vlm_freeze" in checkpoint_extra(config, 3, PhaseState())


# ---------------------------------------------------------------------------
# Phase scheduling
# ---------------------------------------------------------------------------


def _mixture_pipeline() -> DataPipeline:
    return DataPipeline(
        mixture_dataset=FakeMixtureDataset(["a", "b"]),  # type: ignore[arg-type]
        mixture_sampler=FakeMixtureSampler(),  # type: ignore[arg-type]
        mixture_weights={"a": 1.0, "b": 3.0},
    )


class TestPhaseState:
    def test_no_phases_configured(self):
        state = build_phase_state(make_config(), DataPipeline(), step=0)
        assert state.phases == []
        assert state.next_idx == 0
        assert state.lr_scale == 1.0

    def test_phases_are_ordered_by_start_step(self):
        config = make_config()
        config.data = DataConfig(phases=[TrainingPhase(start_step=2), TrainingPhase(start_step=10)])
        state = build_phase_state(config, DataPipeline(), step=0)
        assert [p.start_step for p in state.phases] == [2, 10]

    def test_legacy_anneal_fields_become_one_phase(self):
        config = make_config()
        config.data = DataConfig(anneal_start_step=7, anneal_weights={"a": 2.0})
        state = build_phase_state(config, DataPipeline(), step=0)
        assert len(state.phases) == 1
        assert state.phases[0].start_step == 7
        assert state.phases[0].dataset_weights == {"a": 2.0}

    def test_original_weights_recorded_from_mixture(self):
        config = make_config()
        data = _mixture_pipeline()
        state = build_phase_state(config, data, step=0)
        assert state.original_weights == {"a": 1.0, "b": 3.0}

    def test_resume_re_derives_the_active_phase(self):
        config = make_config()
        config.data = DataConfig(
            phases=[
                TrainingPhase(start_step=5, dataset_weights={"a": 9.0}, lr_scale=0.5),
                TrainingPhase(start_step=50, dataset_weights={"a": 1.0}),
            ]
        )
        data = _mixture_pipeline()
        state = build_phase_state(config, data, step=10)
        assert state.next_idx == 1
        assert state.lr_scale == 0.5
        # Weight not named by the phase falls back to the original.
        assert data.mixture_sampler.calls[-1][0] == [9.0, 3.0]  # type: ignore[union-attr]


class TestAdvancePhases:
    def test_noop_without_a_mixture(self):
        state = PhaseState(phases=[TrainingPhase(start_step=1)])
        assert advance_phases(state, DataPipeline(), 5) is False

    def test_fires_once_then_stays_put(self):
        data = _mixture_pipeline()
        state = build_phase_state(
            _config_with_phases([TrainingPhase(start_step=2, lr_scale=0.25)]), data, step=0
        )
        assert advance_phases(state, data, 1) is False
        assert advance_phases(state, data, 2) is True
        assert state.lr_scale == 0.25
        assert state.next_idx == 1
        assert advance_phases(state, data, 3) is False

    def test_catches_up_across_skipped_phases(self):
        data = _mixture_pipeline()
        state = build_phase_state(
            _config_with_phases(
                [
                    TrainingPhase(start_step=2, lr_scale=0.5),
                    TrainingPhase(start_step=3, lr_scale=0.1),
                ]
            ),
            data,
            step=0,
        )
        assert advance_phases(state, data, 4) is True
        assert state.next_idx == 2
        assert state.lr_scale == 0.1


def _config_with_phases(phases: list[TrainingPhase]) -> JobConfig:
    config = make_config()
    config.data = DataConfig(phases=phases)
    return config


# ---------------------------------------------------------------------------
# The step loop
# ---------------------------------------------------------------------------


class TestRunTrainingLoop:
    def test_runs_to_max_steps_and_accounts_tokens(self):
        config = make_config(max_steps=4)
        session = make_session(config)
        step, tokens_seen = run_training_loop(session)
        assert step == 4
        assert tokens_seen == 4 * config.train.batch_size * config.train.seq_len

    def test_saves_on_the_interval_and_drains_at_the_end(self):
        config = make_config(max_steps=4)
        config.checkpoint.interval = 2
        session = make_session(config)
        run_training_loop(session)
        ckpt: FakeCheckpointManager = session.checkpointer  # type: ignore[assignment]
        # Step 4 is on the schedule, so no extra final save.
        assert ckpt.saved_steps == [2, 4]
        assert ckpt.waits == 1

    def test_final_step_is_saved_when_off_schedule(self):
        config = make_config(max_steps=3)
        config.checkpoint.interval = 2
        session = make_session(config)
        run_training_loop(session)
        ckpt: FakeCheckpointManager = session.checkpointer  # type: ignore[assignment]
        assert ckpt.saved_steps == [2, 3]

    def test_initial_weights_saved_inside_a_dynamic_window(self):
        config = make_config(max_steps=1)
        config.checkpoint.dyn_ckpt_window = DynamicCheckpointWindow(start=0, stop=8)
        session = make_session(config)
        run_training_loop(session)
        ckpt: FakeCheckpointManager = session.checkpointer  # type: ignore[assignment]
        assert ckpt.saved_steps[0] == 0

    def test_resume_skips_the_initial_save(self):
        config = make_config(max_steps=3)
        config.checkpoint.dyn_ckpt_window = DynamicCheckpointWindow(start=0, stop=8)
        session = make_session(config)
        run_training_loop(session, step=2, tokens_seen=99)
        ckpt: FakeCheckpointManager = session.checkpointer  # type: ignore[assignment]
        assert 0 not in ckpt.saved_steps

    def test_resume_continues_the_token_count(self):
        config = make_config(max_steps=3)
        session = make_session(config)
        step, tokens_seen = run_training_loop(session, step=2, tokens_seen=1000)
        assert step == 3
        assert tokens_seen == 1000 + config.train.batch_size * config.train.seq_len

    def test_nan_rollback_stops_without_a_final_save(self):
        config = make_config(max_steps=10)
        session = make_session(
            config,
            loss_fn=lambda logits, labels: logits.sum() * float("nan"),
            nan_detector=NaNDetector(action="warn", max_consecutive=3),
        )
        step, _ = run_training_loop(session)
        ckpt: FakeCheckpointManager = session.checkpointer  # type: ignore[assignment]
        # Two skipped steps, then the third trips the rollback and breaks.
        assert step == 2
        assert ckpt.saves == []
        assert ckpt.waits == 1

    def test_shutdown_writes_an_emergency_checkpoint_and_finishes(self):
        config = make_config(max_steps=10)
        config.checkpoint.interval = 100
        handler = FakeShutdownHandler(after_steps=2)
        session = make_session(config, shutdown_handler=handler)
        step, _ = run_training_loop(session)
        ckpt: FakeCheckpointManager = session.checkpointer  # type: ignore[assignment]
        assert step == 2
        assert ckpt.saved_steps == [2]
        assert handler.finished is True
        assert ckpt.waits == 1

    def test_hooks_fire_for_every_step_and_save(self):
        config = make_config(max_steps=4)
        config.checkpoint.interval = 2
        hook = RecordingHook()
        session = make_session(config, hooks=HookRunner([hook]))
        run_training_loop(session)
        assert hook.begins == 1
        assert hook.steps == [1, 2, 3, 4]
        assert hook.saves == [2, 4]

    def test_exception_path_runs_no_collectives(self):
        # ckpt_mgr.wait() ends in dist.barrier(). Running it while an
        # exception unwinds rendezvouses against ranks still in the loop and
        # hangs the whole job instead of failing one rank fast.
        config = make_config(max_steps=4)

        def boom(_session, step):
            raise RuntimeError("step blew up")

        session = make_session(config, step_fn=boom)
        with pytest.raises(RuntimeError, match="step blew up"):
            run_training_loop(session)
        ckpt: FakeCheckpointManager = session.checkpointer  # type: ignore[assignment]
        assert ckpt.waits == 0
        assert ckpt.saves == []

    def test_profiler_is_stopped_even_when_a_step_raises(self):
        class FakeProfiler:
            def __init__(self) -> None:
                self.started = self.stopped = 0

            def start(self) -> None:
                self.started += 1

            def stop(self) -> None:
                self.stopped += 1

            def step(self) -> None:
                pass

        prof = FakeProfiler()
        session = make_session(
            make_config(max_steps=4),
            step_fn=lambda _s, _i: (_ for _ in ()).throw(RuntimeError("boom")),
            profiler=prof,
            # Non-zero rank so the assertion is about stop(), not the rank-0
            # summary print (which needs a real torch profiler).
            runtime=dataclasses.replace(_runtime(), rank=1),
        )
        with pytest.raises(RuntimeError, match="boom"):
            run_training_loop(session)
        assert (prof.started, prof.stopped) == (1, 1)

    def test_phase_lr_scale_multiplies_the_scheduled_lr(self):
        config = make_config(max_steps=1)
        session = make_session(config, phases=PhaseState(lr_scale=0.5))
        run_training_loop(session)
        assert session.optimizer.param_groups[0]["lr"] == pytest.approx(0.025)


# ---------------------------------------------------------------------------
# run_training wiring
# ---------------------------------------------------------------------------


def _stub_entry(monkeypatch, model: nn.Module) -> FakeCheckpointManager:
    """Replace every phase run_training builds so the orchestration runs on CPU."""
    import kempnerforge.training.entry as entry

    ckpt = FakeCheckpointManager()
    monkeypatch.setattr(entry, "setup_distributed", lambda _config: _runtime())
    monkeypatch.setattr(entry, "ShutdownHandler", lambda **_kw: FakeShutdownHandler())
    monkeypatch.setattr(entry, "build_model", lambda *_a, **_k: (model, None))
    monkeypatch.setattr(
        entry, "build_optimizer", lambda m, _c: torch.optim.SGD(m.parameters(), lr=0.05)
    )
    monkeypatch.setattr(
        entry,
        "build_scheduler",
        lambda o, _c, max_steps: torch.optim.lr_scheduler.LambdaLR(o, lambda _s: 1.0),
    )
    monkeypatch.setattr(entry, "build_checkpoint_manager", lambda *_a, **_k: ckpt)
    monkeypatch.setattr(entry, "restore_checkpoint", lambda *_a, **_k: (0, 0))
    monkeypatch.setattr(entry, "build_profiler", lambda *_a, **_k: None)
    monkeypatch.setattr(entry, "build_data_pipeline", lambda *_a, **_k: DataPipeline())
    monkeypatch.setattr(entry, "build_eval_dataloader", lambda *_a, **_k: None)
    monkeypatch.setattr(entry, "destroy_distributed", lambda: None)
    return ckpt


class TestRunTraining:
    def test_defaults_to_the_registry_selected_step_body(self, monkeypatch):
        from kempnerforge.training.entry import run_training

        model = TinyTextModel()
        before = model.out.weight.detach().clone()
        _stub_entry(monkeypatch, model)
        run_training(make_config(max_steps=2))
        # text_step ran on the synthetic-batch path and the optimizer stepped.
        assert not torch.equal(before, model.out.weight)

    def test_injected_step_fn_replaces_the_default(self, monkeypatch):
        from kempnerforge.training.entry import run_training
        from kempnerforge.training.loop import StepResult

        model = TinyTextModel()
        before = model.out.weight.detach().clone()
        _stub_entry(monkeypatch, model)
        seen: list[int] = []

        def custom_step(_session, step):
            seen.append(step)
            return StepResult(loss=1.0, grad_norm=0.5)

        run_training(make_config(max_steps=3), step_fn=custom_step)
        assert seen == [0, 1, 2]
        # The default body never ran: no backward, so the weights are untouched.
        assert torch.equal(before, model.out.weight)

    def test_injected_hooks_get_the_whole_lifecycle(self, monkeypatch):
        from kempnerforge.training.entry import run_training
        from kempnerforge.training.loop import StepResult

        model = TinyTextModel()
        _stub_entry(monkeypatch, model)
        hook = RecordingHook()
        config = make_config(max_steps=2)

        run_training(
            config,
            step_fn=lambda _s, _i: StepResult(loss=1.0, grad_norm=0.5),
            hooks=HookRunner([hook]),
        )
        assert hook.begins == 1
        assert hook.steps == [1, 2]
        tokens = 2 * config.train.batch_size * config.train.seq_len
        assert hook.ends == [(2, tokens)]


class TestBuildPhaseHelpers:
    def test_checkpoint_manager_has_no_pp_scoping_without_pipeline(self, tmp_path):
        from kempnerforge.training.entry import build_checkpoint_manager

        config = make_config()
        config.checkpoint.dir = str(tmp_path / "ckpt")
        model = TinyTextModel()
        mgr = build_checkpoint_manager(
            config, _runtime(), model, torch.optim.SGD(model.parameters(), lr=0.1), None
        )
        assert mgr._process_group is None
        assert mgr._pp_rank is None

    def test_restore_checkpoint_starts_from_scratch(self, tmp_path):
        from kempnerforge.training.entry import build_checkpoint_manager, restore_checkpoint

        config = make_config()
        config.checkpoint.dir = str(tmp_path / "empty")
        model = TinyTextModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        mgr = build_checkpoint_manager(config, _runtime(), model, optimizer, None)
        assert restore_checkpoint(config, model, None, mgr) == (0, 0)


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------


class TestBuildDataPipeline:
    def test_no_source_yields_an_empty_pipeline(self):
        data = build_data_pipeline(make_config(), _runtime())
        assert data.dataloader is None
        assert data.mixture_dataset is None

    def test_memory_mapped_source(self, mmap_data_dir):
        config = make_config(seq_len=16)
        config.data = DataConfig(dataset_path=mmap_data_dir, num_workers=0)
        data = build_data_pipeline(config, _runtime())
        assert data.dataloader is not None
        assert data.mixture_dataset is None

    def test_mixture_source_exposes_sampler_and_weights(self, mmap_data_dir):
        from kempnerforge.config.data import DatasetSource

        config = make_config(seq_len=16)
        config.data = DataConfig(
            datasets=[
                DatasetSource(name="a", path=mmap_data_dir, weight=1.0),
                DatasetSource(name="b", path=mmap_data_dir, weight=3.0),
            ],
            num_workers=0,
        )
        data = build_data_pipeline(config, _runtime())
        assert data.mixture_dataset is not None
        assert data.mixture_sampler is not None
        assert data.mixture_weights == {"a": 1.0, "b": 3.0}


class TestBuildEvalDataloader:
    def test_disabled_by_default(self):
        from kempnerforge.training.data_pipeline import build_eval_dataloader

        assert build_eval_dataloader(make_config(), _runtime()) is None

    def test_skipped_for_vlm_configs(self, tiny_vlm_configs, mmap_data_dir):
        from kempnerforge.config.eval import EvalConfig
        from kempnerforge.training.data_pipeline import build_eval_dataloader

        mc, vision, adapter, vlm = tiny_vlm_configs
        config = JobConfig(
            model=mc,
            train=TrainConfig(batch_size=1, seq_len=64, max_steps=1),
            eval=EvalConfig(enabled=True, dataset_path=mmap_data_dir),
            vision_encoder=vision,
            adapter=adapter,
            vlm=vlm,
        )
        assert build_eval_dataloader(config, _runtime()) is None
