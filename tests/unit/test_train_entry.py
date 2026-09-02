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
        self.cumulative_sizes = [8 * (i + 1) for i in range(len(names))]

    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0


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

    def test_teardown_is_cooperative_only_on_the_clean_path(self, monkeypatch):
        # destroy_process_group() blocks in _wait_for_pending_works() and
        # shuts NCCL backends down in a deliberate order because
        # ncclCommAbort has been collective. Running it while an exception
        # unwinds meets peers that are still mid-collective.
        import kempnerforge.training.entry as entry
        from kempnerforge.training.entry import run_training
        from kempnerforge.training.loop import StepResult

        destroys: list[int] = []
        closes: list[int] = []

        class CountingTracker:
            def __init__(self, *_a, **_k) -> None:
                pass

            def init_backends(self, _config) -> None:
                pass

            def start_step(self) -> None:
                pass

            def end_step(self, **_kw):
                return None

            def log_eval(self, *_a) -> None:
                pass

            def close(self) -> None:
                closes.append(1)

        _stub_entry(monkeypatch, TinyTextModel())
        monkeypatch.setattr(entry, "MetricsTracker", CountingTracker)
        monkeypatch.setattr(entry, "destroy_distributed", lambda: destroys.append(1))

        def boom(_session, _step):
            raise RuntimeError("step blew up")

        with pytest.raises(RuntimeError, match="step blew up"):
            run_training(make_config(max_steps=2), step_fn=boom)
        assert destroys == []  # left to the launcher on the failure path
        assert closes == [1]  # rank-local, always runs

        run_training(
            make_config(max_steps=2),
            step_fn=lambda _s, _i: StepResult(loss=1.0, grad_norm=0.5),
        )
        assert destroys == [1]
        assert closes == [1, 1]

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


# ---------------------------------------------------------------------------
# Silent-failure pins (#181): logic whose regression produces no error at all
# ---------------------------------------------------------------------------


class TestDpSizeReachesTokenAccounting:
    """`get_dp_info` -> DataPipeline.dp_size -> tokens_in_step.

    A builder that dropped dp_size would silently default it to 1 and
    under-report tokens_seen / tok/s / MFU by the data-parallel factor on
    every multi-GPU run, with training otherwise unaffected.
    """

    def test_builder_records_the_dp_partition(self, monkeypatch, mmap_data_dir):
        import kempnerforge.training.data_pipeline as dp

        monkeypatch.setattr(dp, "get_dp_info", lambda _mesh: (2, 8))
        config = make_config(seq_len=16)
        config.data = DataConfig(dataset_path=mmap_data_dir, num_workers=0)
        data = dp.build_data_pipeline(config, _runtime())
        assert (data.dp_rank, data.dp_size) == (2, 8)

    def test_empty_pipeline_still_records_it(self, monkeypatch):
        import kempnerforge.training.data_pipeline as dp

        monkeypatch.setattr(dp, "get_dp_info", lambda _mesh: (3, 4))
        data = dp.build_data_pipeline(make_config(), _runtime())
        assert data.dataloader is None
        assert (data.dp_rank, data.dp_size) == (3, 4)

    def test_dp_size_multiplies_tokens_seen(self):
        config = make_config(max_steps=2)
        session = make_session(config, data=DataPipeline(dp_size=8))
        _, tokens_seen = run_training_loop(session)
        per_step = config.train.batch_size * config.train.seq_len * config.train.grad_accum_steps
        assert tokens_seen == 2 * per_step * 8

    def test_default_dp_size_is_one_not_zero(self):
        # A default of 0 would silently zero every token count.
        assert DataPipeline().dp_size == 1


class TestPhaseTemperatureReachesSampler:
    """`config.data.mix_temperature` -> PhaseState.temperature -> update_weights.

    These lines are already line-covered, so coverage would never flag a
    regression here; only an assertion on the value can. Dropping the
    propagation silently re-weights the mixture.
    """

    def _config(self, temperature: float) -> JobConfig:
        config = make_config()
        config.data = DataConfig(
            phases=[TrainingPhase(start_step=2, dataset_weights={"a": 9.0})],
            mix_temperature=temperature,
        )
        return config

    def test_build_captures_the_configured_temperature(self):
        state = build_phase_state(self._config(0.25), _mixture_pipeline(), step=0)
        assert state.temperature == 0.25

    def test_transition_passes_it_to_the_sampler(self):
        data = _mixture_pipeline()
        state = build_phase_state(self._config(0.25), data, step=0)
        assert advance_phases(state, data, 2) is True
        weights, temperature = data.mixture_sampler.calls[-1]  # type: ignore[union-attr]
        assert temperature == 0.25
        assert weights == [9.0, 3.0]

    def test_resume_replay_passes_it_too(self):
        data = _mixture_pipeline()
        build_phase_state(self._config(0.5), data, step=5)
        _, temperature = data.mixture_sampler.calls[-1]  # type: ignore[union-attr]
        assert temperature == 0.5


# ---------------------------------------------------------------------------
# Build phases driven under fakes (no GPU, no mesh, no real parallelism)
# ---------------------------------------------------------------------------


class FakeSubMesh:
    def __init__(self, key) -> None:
        self.key = key

    def get_group(self):
        return f"group:{self.key}"


class FakeMesh:
    """Enough DeviceMesh surface for the build phases to run on CPU."""

    def __init__(self, dims=("pp", "dp_shard")) -> None:
        self.mesh_dim_names = dims
        self.sliced: list = []

    def __getitem__(self, key):
        self.sliced.append(key)
        return FakeSubMesh(key)


class FakeStageModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.inited = False
        self.emptied_to = None

    def init_weights_and_freqs(self) -> None:
        self.inited = True

    def to_empty(self, *, device):  # type: ignore[override]
        self.emptied_to = device
        return self


def _pp_runtime(mesh=None) -> RuntimeContext:
    return dataclasses.replace(_runtime(), device_mesh=mesh or FakeMesh())


def _stub_pp(monkeypatch, stage: FakeStageModule, calls: dict):
    """Neutralize every parallelism call build_model makes on the PP path."""
    import kempnerforge.distributed.pipeline_parallel as pp
    import kempnerforge.training.entry as entry

    monkeypatch.setattr(pp, "build_stage_module", lambda cfg, r, s: stage)
    monkeypatch.setattr(pp, "get_pp_rank", lambda _m: 1)
    monkeypatch.setattr(pp, "get_pp_size", lambda _m: 2)
    monkeypatch.setattr(pp, "build_pipeline_stage", lambda *a, **k: "STAGE")

    def _sched(**k):
        calls["schedule_kwargs"] = k
        return "SCHED"

    monkeypatch.setattr(pp, "build_pipeline_schedule", _sched)
    for name in ("apply_tensor_parallel", "apply_float8", "apply_ac", "apply_fsdp2"):
        monkeypatch.setattr(entry, name, lambda *a, _n=name, **k: calls.setdefault(_n, True))
    return entry


class TestBuildModel:
    def test_non_pp_delegates_and_returns_no_pipeline(self, monkeypatch):
        import kempnerforge.training.entry as entry

        seen: dict = {}

        def fake_build(model_config, device, mesh, **kw):
            seen.update(kw)
            seen["device"] = device
            seen["mesh"] = mesh
            return "MODEL"

        monkeypatch.setattr(entry, "build_parallel_model", fake_build)
        config = make_config()
        model, pipeline = entry.build_model(config, _runtime(), lambda a, b: a)

        assert (model, pipeline) == ("MODEL", None)
        # the knobs the loop depends on must be threaded, not defaulted
        assert seen["param_dtype"] == config.train.param_dtype
        assert seen["compile_model"] is config.train.compile_model
        assert seen["fp8"] is config.train.is_fp8
        assert seen["frames_per_clip"] == 1  # no [video] section
        assert seen["mesh"] is None

    def test_non_pp_threads_frames_per_clip_from_video_config(self, monkeypatch):
        import kempnerforge.training.entry as entry
        from kempnerforge.config.video import VideoConfig

        seen: dict = {}
        monkeypatch.setattr(
            entry, "build_parallel_model", lambda *a, **k: seen.update(k) or "MODEL"
        )
        config = make_config()
        config.video = VideoConfig(max_frames=6, min_frames=1, frame_size=16)
        entry.build_model(config, _runtime(), lambda a, b: a)
        assert seen["frames_per_clip"] == 6

    def test_pp_builds_a_bundle_from_the_mesh(self, monkeypatch):
        stage, calls = FakeStageModule(), {}
        entry = _stub_pp(monkeypatch, stage, calls)
        config = make_config()
        config.distributed.pp = 2

        model, pipeline = entry.build_model(config, _pp_runtime(), "LOSS")

        assert pipeline is not None
        assert (pipeline.rank, pipeline.size) == (1, 2)
        assert pipeline.schedule == "SCHED"
        # the schedule must get the real microbatch count and loss fn
        assert calls["schedule_kwargs"]["n_microbatches"] == config.train.grad_accum_steps
        assert calls["schedule_kwargs"]["loss_fn"] == "LOSS"
        assert model is stage

    def test_pp_without_tp_skips_meta_init(self, monkeypatch):
        stage, calls = FakeStageModule(), {}
        entry = _stub_pp(monkeypatch, stage, calls)
        config = make_config()
        config.distributed.pp = 2
        entry.build_model(config, _pp_runtime(FakeMesh(("pp", "dp_shard"))), "LOSS")
        # no tp dim -> the meta-device path must not run
        assert "apply_tensor_parallel" not in calls
        assert stage.inited is False

    def test_pp_with_tp_takes_the_meta_device_path(self, monkeypatch):
        stage, calls = FakeStageModule(), {}
        entry = _stub_pp(monkeypatch, stage, calls)
        config = make_config()
        config.distributed.pp = 2
        entry.build_model(config, _pp_runtime(FakeMesh(("pp", "tp"))), "LOSS")
        assert calls.get("apply_tensor_parallel") is True
        assert stage.inited is True  # init_weights_and_freqs after to_empty
        assert stage.emptied_to is not None

    def test_pp_without_a_mesh_raises_rather_than_asserting(self, monkeypatch):
        stage, calls = FakeStageModule(), {}
        entry = _stub_pp(monkeypatch, stage, calls)
        config = make_config()
        config.distributed.pp = 2
        # RuntimeError, not AssertionError: the guard must survive python -O
        with pytest.raises(RuntimeError, match="requires a device mesh"):
            entry.build_model(config, _runtime(), "LOSS")


class TestBuildCheckpointManager:
    def test_pp_scopes_dcp_to_the_non_pp_dims(self, tmp_path):
        from kempnerforge.training.entry import build_checkpoint_manager
        from kempnerforge.training.runtime import PipelineBundle

        config = make_config()
        config.checkpoint.dir = str(tmp_path / "c")
        model = TinyTextModel()
        mesh = FakeMesh(("pp", "dp_shard"))
        mgr = build_checkpoint_manager(
            config,
            dataclasses.replace(_runtime(), device_mesh=mesh),
            model,
            torch.optim.SGD(model.parameters(), lr=0.1),
            PipelineBundle(rank=1, size=2, schedule=None),
        )
        assert mgr._pp_rank == 1
        assert mgr._process_group == "group:dp_shard"  # "pp" excluded

    def test_pp_with_several_non_pp_dims_uses_a_flattened_group(self, tmp_path):
        from kempnerforge.training.entry import build_checkpoint_manager
        from kempnerforge.training.runtime import PipelineBundle

        config = make_config()
        config.checkpoint.dir = str(tmp_path / "c")
        model = TinyTextModel()
        mesh = FakeMesh(("pp", "dp_shard", "tp"))
        mgr = build_checkpoint_manager(
            config,
            dataclasses.replace(_runtime(), device_mesh=mesh),
            model,
            torch.optim.SGD(model.parameters(), lr=0.1),
            PipelineBundle(rank=0, size=2, schedule=None),
        )
        assert mgr._process_group == "group:('dp_shard', 'tp')"


class FakeResumeManager:
    """CheckpointManager surface restore_checkpoint actually uses."""

    def __init__(self, step=7, tokens=1234, extra=None, saved_step=7) -> None:
        self._ret = (step, tokens, extra or {})
        self._saved_step = saved_step
        self.load_kwargs: dict = {}
        self.peeked: list = []

    def peek_saved_step(self, path=None):
        self.peeked.append(path)
        return self._saved_step

    def load(self, **kw):
        self.load_kwargs = kw
        return self._ret


def _vlm_config(**vlm_kw) -> JobConfig:
    from kempnerforge.config.adapter import AdapterConfig
    from kempnerforge.config.vision import VisionEncoderConfig
    from kempnerforge.config.vlm import VLMConfig

    return JobConfig(
        model=ModelConfig(dim=64, n_layers=2, n_heads=4, vocab_size=256, max_seq_len=64),
        train=TrainConfig(batch_size=1, seq_len=64, max_steps=2),
        vision_encoder=VisionEncoderConfig(type="random", feature_dim=96, num_tokens=8),
        adapter=AdapterConfig(),
        vlm=VLMConfig(max_text_len=32, **vlm_kw),
    )


class TestRestoreCheckpoint:
    def test_no_checkpoint_and_no_load_path_starts_from_scratch(self, monkeypatch):
        import kempnerforge.training.entry as entry

        monkeypatch.setattr(entry, "resolve_resume_path", lambda _d: None)
        mgr = FakeResumeManager()
        assert entry.restore_checkpoint(make_config(), TinyTextModel(), None, mgr) == (0, 0)
        assert mgr.load_kwargs == {}  # load() must not be called at all

    def test_resume_returns_the_managers_step_and_tokens(self, monkeypatch, tmp_path):
        import kempnerforge.training.entry as entry

        monkeypatch.setattr(entry, "resolve_resume_path", lambda _d: tmp_path / "step_7")
        mgr = FakeResumeManager(step=7, tokens=999)
        assert entry.restore_checkpoint(make_config(), TinyTextModel(), "SCHED", mgr) == (7, 999)
        assert mgr.load_kwargs["scheduler"] == "SCHED"
        assert mgr.load_kwargs["vlm_freeze_expected"] is None  # text-only

    def test_load_path_alone_triggers_a_resume(self, monkeypatch):
        import kempnerforge.training.entry as entry

        monkeypatch.setattr(entry, "resolve_resume_path", lambda _d: None)
        config = make_config()
        config.checkpoint.load_path = "/somewhere/step_3"
        mgr = FakeResumeManager(step=3, tokens=30)
        assert entry.restore_checkpoint(config, TinyTextModel(), None, mgr) == (3, 30)
        assert mgr.load_kwargs["path"] is None  # manager resolves load_path itself

    def test_metrics_run_ids_are_restored_onto_the_config(self, monkeypatch, tmp_path):
        import kempnerforge.training.entry as entry

        monkeypatch.setattr(entry, "resolve_resume_path", lambda _d: tmp_path / "s")
        config = make_config()
        mgr = FakeResumeManager(extra={"wandb_run_id": "W1", "mlflow_run_id": "M1"})
        entry.restore_checkpoint(config, TinyTextModel(), None, mgr)
        assert config.metrics.wandb_run_id == "W1"
        assert config.metrics.mlflow_run_id == "M1"

    def test_absent_run_ids_do_not_clobber_existing_ones(self, monkeypatch, tmp_path):
        import kempnerforge.training.entry as entry

        monkeypatch.setattr(entry, "resolve_resume_path", lambda _d: tmp_path / "s")
        config = make_config()
        config.metrics.wandb_run_id = "KEEP"
        entry.restore_checkpoint(config, TinyTextModel(), None, FakeResumeManager(extra={}))
        assert config.metrics.wandb_run_id == "KEEP"

    def test_vlm_resume_computes_expected_freeze_at_the_saved_step(self, monkeypatch, tmp_path):
        import kempnerforge.training.entry as entry
        from kempnerforge.training.freeze import freeze_meta_at_step

        monkeypatch.setattr(entry, "resolve_resume_path", lambda _d: tmp_path / "s")
        config = _vlm_config()
        mgr = FakeResumeManager(saved_step=11)
        entry.restore_checkpoint(config, TinyTextModel(), None, mgr)
        # the peeked step, not the loaded step, drives the expectation
        assert mgr.peeked == [str(tmp_path / "s")]
        assert mgr.load_kwargs["vlm_freeze_expected"] == freeze_meta_at_step(11, config.vlm)

    def test_freeze_schedule_is_reapplied_at_the_resumed_step(self, monkeypatch, tmp_path):
        import kempnerforge.training.entry as entry
        from kempnerforge.config.vlm import FreezeSpec, FreezeStage

        monkeypatch.setattr(entry, "resolve_resume_path", lambda _d: tmp_path / "s")
        applied: list = []
        monkeypatch.setattr(
            entry, "apply_freeze_specs", lambda m, specs, pat: applied.append(list(specs))
        )
        config = _vlm_config(
            freeze_schedule=(
                FreezeStage(start_step=5, specs=(FreezeSpec("vision_encoder", False),)),
            )
        )
        entry.restore_checkpoint(config, TinyTextModel(), None, FakeResumeManager(step=9))
        # step 9 is past start_step 5, so the stage must have been applied
        assert applied and applied[0][0].module == "vision_encoder"

    # -- warm-start exclude vs resume: the asymmetry ------------------------

    def test_warm_start_from_load_path_honors_exclude_from_loading(self, monkeypatch):
        """A converted, weights-only checkpoint has no optimizer/train_state,
        so a warm start must be able to skip the optimizer."""
        import kempnerforge.training.entry as entry

        monkeypatch.setattr(entry, "resolve_resume_path", lambda _d: None)
        config = make_config()
        config.checkpoint.load_path = "/somewhere/converted"
        config.checkpoint.exclude_from_loading = ["optimizer"]
        mgr = FakeResumeManager(step=0, tokens=0)
        entry.restore_checkpoint(config, TinyTextModel(), None, mgr)
        assert mgr.load_kwargs["exclude_keys"] == ["optimizer"]

    def test_resume_ignores_exclude_from_loading(self, monkeypatch, tmp_path):
        """The other direction, and the one that matters: honoring it on a real
        resume would silently drop optimizer moments mid-run."""
        import kempnerforge.training.entry as entry

        monkeypatch.setattr(entry, "resolve_resume_path", lambda _d: tmp_path / "step_7")
        config = make_config()
        config.checkpoint.exclude_from_loading = ["optimizer"]
        mgr = FakeResumeManager(step=7, tokens=999)
        entry.restore_checkpoint(config, TinyTextModel(), None, mgr)
        assert mgr.load_kwargs["exclude_keys"] is None

    def test_resume_ignores_exclude_even_when_load_path_is_also_set(self, monkeypatch, tmp_path):
        """An auto-resumable run keeps a stale load_path in its config; the
        resolved resume path is what decides, not load_path's presence."""
        import kempnerforge.training.entry as entry

        monkeypatch.setattr(entry, "resolve_resume_path", lambda _d: tmp_path / "step_7")
        config = make_config()
        config.checkpoint.load_path = "/somewhere/converted"
        config.checkpoint.exclude_from_loading = ["model", "optimizer"]
        mgr = FakeResumeManager(step=7, tokens=999)
        entry.restore_checkpoint(config, TinyTextModel(), None, mgr)
        assert mgr.load_kwargs["exclude_keys"] is None

    def test_empty_exclude_list_passes_none_not_an_empty_list(self, monkeypatch):
        """CheckpointManager.load branches on `exclude_keys is None`; an empty
        list would take the same branch, but None keeps the intent explicit."""
        import kempnerforge.training.entry as entry

        monkeypatch.setattr(entry, "resolve_resume_path", lambda _d: None)
        config = make_config()
        config.checkpoint.load_path = "/somewhere/converted"
        mgr = FakeResumeManager(step=0, tokens=0)
        entry.restore_checkpoint(config, TinyTextModel(), None, mgr)
        assert mgr.load_kwargs["exclude_keys"] is None

    def test_mot_warm_start_fires_only_at_step_zero(self, monkeypatch, tmp_path):
        import kempnerforge.training.entry as entry
        from kempnerforge.config.adapter import AdapterConfig
        from kempnerforge.config.vision import VisionEncoderConfig
        from kempnerforge.config.vlm import MoTConfig

        monkeypatch.setattr(entry, "resolve_resume_path", lambda _d: tmp_path / "s")
        monkeypatch.setattr(torch, "load", lambda *a, **k: {"model": {"w": 1}})
        copied: list = []
        monkeypatch.setattr(
            entry, "mot_warm_start_from_text_stack", lambda m, src: copied.append(src)
        )
        config = JobConfig(
            model=ModelConfig(dim=64, n_layers=2, n_heads=4, vocab_size=256, max_seq_len=64),
            train=TrainConfig(batch_size=1, seq_len=64, max_steps=2),
            vision_encoder=VisionEncoderConfig(type="random", feature_dim=96, num_tokens=8),
            adapter=AdapterConfig(),
            vlm=MoTConfig(
                max_text_len=32, mot_warm_start_from_text=True, mot_warm_start_path="/x.pt"
            ),
        )
        entry.restore_checkpoint(config, TinyTextModel(), None, FakeResumeManager(step=0))
        assert copied == [{"w": 1}]  # unwrapped from the {"model": ...} envelope

        copied.clear()
        entry.restore_checkpoint(config, TinyTextModel(), None, FakeResumeManager(step=4))
        assert copied == []  # mid-flight resume already has MoT-shaped state


# ---------------------------------------------------------------------------
# Data-pipeline source builders, driven under fakes
# ---------------------------------------------------------------------------


class FakeDataset:
    def __init__(self, n=8, **kw) -> None:
        self.n = n
        self.kwargs = kw

    def __len__(self) -> int:
        return self.n


def _stub_loaders(monkeypatch, seen: dict):
    """Record what the builders construct without touching torch DataLoader."""
    import kempnerforge.training.data_pipeline as dp

    def mk(name):
        def _f(*a, **k):
            seen[name] = {"args": a, "kwargs": k}
            return f"{name}_OBJ"

        return _f

    monkeypatch.setattr(dp, "StatefulDataLoader", mk("stateful"))
    monkeypatch.setattr(dp, "TorchDataLoader", mk("torch_loader"))
    monkeypatch.setattr(dp, "DistributedSampler", mk("sampler"))
    monkeypatch.setattr(dp, "MixtureSampler", mk("mixture_sampler"))
    return dp


class TestHfSourceBuilder:
    def test_eager_path_builds_sampler_and_stateful_loader(self, monkeypatch):
        seen: dict = {}
        dp = _stub_loaders(monkeypatch, seen)
        monkeypatch.setattr(dp, "HuggingFaceDataset", lambda **k: FakeDataset(**k))

        config = make_config()
        config.data = DataConfig(hf_dataset_name="ds", tokenizer_path="tok", num_workers=0)
        data = dp.build_data_pipeline(config, _runtime())

        assert data.dataloader == "stateful_OBJ"
        # the sampler must be built from the DP partition, not world size
        assert seen["sampler"]["kwargs"]["num_replicas"] == data.dp_size
        assert seen["sampler"]["kwargs"]["shuffle"] is True

    def test_streaming_path_shards_by_dp_rank_and_skips_the_sampler(self, monkeypatch):
        seen: dict = {}
        dp = _stub_loaders(monkeypatch, seen)
        monkeypatch.setattr(dp, "get_dp_info", lambda _m: (2, 4))
        made: dict = {}
        monkeypatch.setattr(
            dp, "StreamingHuggingFaceDataset", lambda **k: made.update(k) or FakeDataset()
        )

        config = make_config()
        config.data = DataConfig(
            hf_dataset_name="ds", tokenizer_path="tok", hf_streaming=True, num_workers=0
        )
        data = dp.build_data_pipeline(config, _runtime())

        assert data.dataloader == "torch_loader_OBJ"
        assert "sampler" not in seen  # streaming shards inside the dataset
        assert (made["rank"], made["world_size"]) == (2, 4)

    def test_missing_tokenizer_is_rejected(self):
        import kempnerforge.training.data_pipeline as dp

        config = make_config()
        config.data = DataConfig(hf_dataset_name="ds", num_workers=0)
        with pytest.raises(ValueError, match="tokenizer_path is required"):
            dp.build_data_pipeline(config, _runtime())

    def test_empty_name_raises_rather_than_asserting(self):
        # -O-safe guard: build_data_pipeline dispatches on a truthy name, so
        # this is only reachable by calling the builder directly.
        from kempnerforge.training.data_pipeline import _build_hf_pipeline

        config = make_config()
        config.data = DataConfig(tokenizer_path="tok", num_workers=0)
        with pytest.raises(ValueError, match="hf_dataset_name is required"):
            _build_hf_pipeline(config, 0, 1)


class TestEosResolution:
    def test_none_when_packing_disabled(self):
        from kempnerforge.training.data_pipeline import _resolve_eos_token_id

        config = make_config()
        config.data = DataConfig(dataset_path="/x", pack_sequences=False)
        assert _resolve_eos_token_id(config) is None

    def test_none_when_no_mmap_source(self):
        from kempnerforge.training.data_pipeline import _resolve_eos_token_id

        config = make_config()
        config.data = DataConfig(hf_dataset_name="ds", pack_sequences=True, tokenizer_path="t")
        assert _resolve_eos_token_id(config) is None

    def test_packing_without_a_tokenizer_is_rejected(self):
        from kempnerforge.training.data_pipeline import _resolve_eos_token_id

        config = make_config()
        config.data = DataConfig(dataset_path="/x", pack_sequences=True)
        with pytest.raises(ValueError, match="tokenizer_path is required when pack_sequences"):
            _resolve_eos_token_id(config)

    def test_eos_reaches_the_mmap_dataset(self, monkeypatch, mmap_data_dir):
        import kempnerforge.training.data_pipeline as dp

        seen: dict = {}
        monkeypatch.setattr(dp, "_resolve_eos_token_id", lambda _c: 4242)
        monkeypatch.setattr(dp, "MemoryMappedDataset", lambda **k: seen.update(k) or FakeDataset())
        _stub_loaders(monkeypatch, {})
        config = make_config(seq_len=16)
        config.data = DataConfig(dataset_path=mmap_data_dir, num_workers=0)
        dp.build_data_pipeline(config, _runtime())
        assert seen["eos_token_id"] == 4242
        assert seen["seq_len"] == config.train.seq_len + 1  # +1 for the label shift


class TestVlmSourceBuilders:
    def _vlm_data_config(self, video=False) -> JobConfig:
        config = _vlm_config()
        config.data = DataConfig(hf_dataset_name="coco", tokenizer_path="tok", num_workers=0)
        if video:
            from kempnerforge.config.video import VideoConfig

            config.video = VideoConfig(max_frames=2, min_frames=1, frame_size=16)
        return config

    def _stub_vlm_modules(self, monkeypatch, seen: dict):
        import kempnerforge.data.video_dataset as vid
        import kempnerforge.data.vlm_dataset as vlmd

        monkeypatch.setattr(vlmd, "build_tokenizer", lambda p: "TOK")
        monkeypatch.setattr(vlmd, "resolve_pad_id", lambda tok: 7)
        monkeypatch.setattr(vlmd, "HuggingFaceVLMDataset", lambda **k: FakeDataset(**k))
        monkeypatch.setattr(vlmd, "VLMCollator", lambda **k: seen.setdefault("image_collator", k))
        monkeypatch.setattr(vid, "build_video_dataset", lambda *a: FakeDataset())
        monkeypatch.setattr(vid, "VideoCollator", lambda **k: seen.setdefault("video_collator", k))

    def test_image_path_uses_the_vlm_collator_with_resolved_pad_id(self, monkeypatch):
        seen: dict = {}
        dp = _stub_loaders(monkeypatch, seen)
        self._stub_vlm_modules(monkeypatch, seen)
        data = dp.build_data_pipeline(self._vlm_data_config(), _runtime())
        assert data.dataloader == "stateful_OBJ"
        assert seen["image_collator"]["pad_id"] == 7
        assert "video_collator" not in seen

    def test_video_path_uses_the_video_collator(self, monkeypatch):
        seen: dict = {}
        dp = _stub_loaders(monkeypatch, seen)
        self._stub_vlm_modules(monkeypatch, seen)
        data = dp.build_data_pipeline(self._vlm_data_config(video=True), _runtime())
        assert data.dataloader == "stateful_OBJ"
        assert seen["video_collator"]["pad_id"] == 7
        assert "image_collator" not in seen

    def test_video_without_a_tokenizer_is_rejected(self):
        import kempnerforge.training.data_pipeline as dp

        config = self._vlm_data_config(video=True)
        config.data = DataConfig(num_workers=0)
        with pytest.raises(ValueError, match="Video training requires data.tokenizer_path"):
            dp.build_data_pipeline(config, _runtime())

    def test_image_vlm_without_a_dataset_is_rejected(self):
        import kempnerforge.training.data_pipeline as dp

        config = self._vlm_data_config()
        config.data = DataConfig(num_workers=0)
        with pytest.raises(ValueError, match="VLM training requires"):
            dp.build_data_pipeline(config, _runtime())


class TestEvalDataloaderBuilder:
    def test_mmap_eval_source(self, monkeypatch, mmap_data_dir):
        import kempnerforge.training.data_pipeline as dp
        from kempnerforge.config.eval import EvalConfig

        seen: dict = {}
        _stub_loaders(monkeypatch, seen)
        monkeypatch.setattr(dp, "MemoryMappedDataset", lambda **k: FakeDataset(**k))
        config = make_config(seq_len=16)
        config.eval = EvalConfig(enabled=True, dataset_path=mmap_data_dir)
        assert dp.build_eval_dataloader(config, _runtime()) == "torch_loader_OBJ"
        # eval must not shuffle -- ranks have to see a fixed partition
        assert seen["sampler"]["kwargs"]["shuffle"] is False

    def test_hf_eval_skips_the_broadcast_when_not_distributed(self, monkeypatch):
        import numpy as np

        import kempnerforge.training.data_pipeline as dp
        from kempnerforge.config.eval import EvalConfig

        seen: dict = {}
        _stub_loaders(monkeypatch, seen)

        class FakeHF:
            def __init__(self, **k) -> None:
                self._packed_sequences = [np.zeros(17, dtype=np.int64) for _ in range(3)]

        monkeypatch.setattr(dp, "HuggingFaceDataset", FakeHF)
        monkeypatch.setattr(dp.dist, "is_initialized", lambda: False)

        def _boom(*a, **k):
            raise AssertionError("broadcast must not run single-process")

        monkeypatch.setattr(dp.dist, "broadcast", _boom)

        config = make_config(seq_len=16)
        config.eval = EvalConfig(enabled=True, hf_dataset_name="wikitext")
        assert dp.build_eval_dataloader(config, _runtime()) == "torch_loader_OBJ"

    def test_no_source_configured_returns_none(self):
        import kempnerforge.training.data_pipeline as dp
        from kempnerforge.config.eval import EvalConfig

        config = make_config()
        config.eval = EvalConfig(enabled=False)
        assert dp.build_eval_dataloader(config, _runtime()) is None

    def test_broadcast_tensor_dataset_yields_shifted_pairs(self):
        from kempnerforge.training.data_pipeline import _EvalTensorDataset

        ds = _EvalTensorDataset(torch.arange(10).reshape(2, 5))
        assert len(ds) == 2
        item = ds[0]
        # input_ids/labels are the same row offset by one -- the LM shift
        assert item["input_ids"].tolist() == [0, 1, 2, 3]
        assert item["labels"].tolist() == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Step bodies that normally need PP / a VLM model
# ---------------------------------------------------------------------------


class FakeSchedule:
    def __init__(self, loss=2.0) -> None:
        self.loss = loss
        self.calls: list = []

    def step(self, *args, target=None, losses=None):
        self.calls.append({"args": args, "target": target})
        if losses is not None:
            losses.append(torch.tensor(self.loss))


class TestPipelineStep:
    def _session(self, monkeypatch, rank, size, sched):
        import kempnerforge.training.loop as loop
        from kempnerforge.training.runtime import PipelineBundle

        monkeypatch.setattr(loop, "pp_group", lambda _r: "PPGROUP")
        recorded: dict = {}

        def fake_broadcast(tensor, group_src=None, group=None):
            recorded["group_src"] = group_src
            recorded["group"] = group

        monkeypatch.setattr(loop.dist, "broadcast", fake_broadcast)
        config = make_config()
        config.distributed.pp = size
        session = make_session(
            config, pipeline=PipelineBundle(rank=rank, size=size, schedule=sched)
        )
        return session, recorded

    def test_first_stage_feeds_input_and_target(self, monkeypatch):
        sched = FakeSchedule()
        session, rec = self._session(monkeypatch, rank=0, size=2, sched=sched)
        result = session.step_fn(session, 0)
        assert sched.calls[0]["args"], "first stage must pass the input batch"
        assert sched.calls[0]["target"] is not None
        # loss comes back through the broadcast tensor
        assert isinstance(result.loss, float)
        assert rec["group"] == "PPGROUP"
        assert rec["group_src"] == 1  # last stage is the source

    def test_last_stage_passes_target_only(self, monkeypatch):
        sched = FakeSchedule()
        session, _ = self._session(monkeypatch, rank=1, size=2, sched=sched)
        session.step_fn(session, 0)
        assert sched.calls[0]["args"] == ()
        assert sched.calls[0]["target"] is not None

    def test_middle_stage_passes_neither(self, monkeypatch):
        sched = FakeSchedule()
        session, _ = self._session(monkeypatch, rank=1, size=3, sched=sched)
        session.step_fn(session, 0)
        assert sched.calls[0]["args"] == ()
        assert sched.calls[0]["target"] is None

    def test_selected_without_a_bundle_raises(self):
        from kempnerforge.training.loop import pipeline_step

        session = make_session(make_config())  # pipeline=None
        with pytest.raises(RuntimeError, match="requires TrainingSession.pipeline"):
            pipeline_step(session, 0)


class TinyVlmModel(nn.Module):
    """(pixel_values, input_ids, labels) -> (logits, labels) as VLMWrapper does."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(DIM, VOCAB)
        self.embed = nn.Embedding(VOCAB, DIM)
        self.seen_frame_mask: list = []

    def forward(self, pixel_values, input_ids, labels, frame_mask=None):
        self.seen_frame_mask.append(frame_mask)
        return self.proj(self.embed(input_ids)), labels


class TestVlmStep:
    def _batch(self, with_mask=False):
        b = {
            "pixel_values": torch.zeros(2, 3, 4, 4),
            "input_ids": torch.zeros(2, 4, dtype=torch.long),
            "labels": torch.zeros(2, 4, dtype=torch.long),
        }
        if with_mask:
            b["frame_mask"] = torch.ones(2, 2, dtype=torch.bool)
        return b

    def _session(self, batches):
        config = make_config()
        model = TinyVlmModel()
        session = make_session(config, model=model, data=DataPipeline(dataloader=batches))
        session.step_fn = __import__("kempnerforge.training.loop", fromlist=["vlm_step"]).vlm_step
        return session, model

    def test_counts_unmasked_text_tokens(self):
        session, _ = self._session([self._batch()])
        result = session.step_fn(session, 0)
        assert result.text_tokens == 8  # 2 x 4, none are -100

    def test_ignore_index_positions_are_excluded(self):
        batch = self._batch()
        batch["labels"][:, :2] = -100
        session, _ = self._session([batch])
        result = session.step_fn(session, 0)
        assert result.text_tokens == 4  # half the positions masked out

    def test_frame_mask_is_threaded_when_present(self):
        session, model = self._session([self._batch(with_mask=True)])
        session.step_fn(session, 0)
        assert model.seen_frame_mask[0] is not None

    def test_frame_mask_is_none_for_image_batches(self):
        session, model = self._session([self._batch()])
        session.step_fn(session, 0)
        assert model.seen_frame_mask[0] is None

    def test_synthetic_fallback_is_refused(self):
        session, _ = self._session(None)
        with pytest.raises(RuntimeError, match="requires a real dataloader"):
            session.step_fn(session, 0)


class TestSetupDistributed:
    def test_builds_the_context_and_validates_against_world_size(self, monkeypatch):
        import kempnerforge.training.runtime as rt

        validated: list = []
        monkeypatch.setattr(rt, "get_world_info", lambda: (3, 1, 8))
        monkeypatch.setattr(rt, "init_distributed", lambda cfg, seed: "MESH")
        monkeypatch.setattr(rt, "log_job_info", lambda: None)
        config = make_config()
        monkeypatch.setattr(type(config), "validate", lambda self, ws: validated.append(ws))

        ctx = rt.setup_distributed(config)

        assert (ctx.rank, ctx.local_rank, ctx.world_size) == (3, 1, 8)
        assert ctx.device_mesh == "MESH"
        assert validated == [8]  # validate must see the real world size

    def test_seed_is_threaded_into_init(self, monkeypatch):
        import kempnerforge.training.runtime as rt

        seen: dict = {}
        monkeypatch.setattr(rt, "get_world_info", lambda: (0, 0, 1))
        monkeypatch.setattr(rt, "log_job_info", lambda: None)
        monkeypatch.setattr(
            rt, "init_distributed", lambda cfg, seed: seen.update(seed=seed) or None
        )
        config = make_config()
        config.train.seed = 4242
        rt.setup_distributed(config)
        assert seen["seed"] == 4242


class TestPpGroup:
    def test_derives_the_group_from_the_mesh(self):
        from kempnerforge.training.runtime import pp_group

        mesh = FakeMesh(("pp", "dp_shard"))
        assert pp_group(dataclasses.replace(_runtime(), device_mesh=mesh)) == "group:pp"

    def test_without_a_mesh_raises_rather_than_asserting(self):
        from kempnerforge.training.runtime import pp_group

        with pytest.raises(RuntimeError, match="requires a device mesh"):
            pp_group(_runtime())


# ---------------------------------------------------------------------------
# Loop branches that normally need MoE / VLM / eval / a profiler
# ---------------------------------------------------------------------------


class TinyMoEModel(nn.Module):
    """Text model exposing the MoE hooks the loop reaches through inner_transformer."""

    def __init__(self, aux=0.5, z=0.25, counts=None) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, DIM)
        self.out = nn.Linear(DIM, VOCAB)
        self._aux, self._z = aux, z
        self._counts = counts if counts is not None else {"0": torch.tensor([2.0, 6.0])}
        self.moe_steps: list = []

    def forward(self, input_ids, doc_ids=None):
        return self.out(self.embed(input_ids))

    def set_moe_step(self, step, max_steps) -> None:
        self.moe_steps.append((step, max_steps))

    def get_moe_aux_loss(self):
        return torch.tensor(self._aux)

    def get_moe_router_z_loss(self):
        return torch.tensor(self._z)

    def get_expert_counts(self):
        return self._counts


def _moe_config(z_weight=0.0, aux_weight=0.1, **kw):
    config = make_config(**kw)
    config.model.num_experts = 4
    config.model.moe_aux_loss_weight = aux_weight
    config.model.moe_router_z_loss_weight = z_weight
    return config


class TestMoELossComposition:
    def test_aux_loss_is_added_with_its_weight(self):
        model = TinyMoEModel(aux=0.5, z=0.25)
        plain = make_session(make_config(), model=TinyTextModel())
        base = plain.step_fn(plain, 0).loss

        session = make_session(_moe_config(aux_weight=2.0), model=model)
        torch.manual_seed(0)
        loss = session.step_fn(session, 0).loss
        # aux contributes weight*aux on top of CE; assert it moved, not just ran
        assert loss != base
        assert session.model.moe_steps == [(0, session.config.train.max_steps)]

    def test_z_loss_only_applies_when_its_weight_is_positive(self):
        torch.manual_seed(0)
        off = make_session(_moe_config(z_weight=0.0), model=TinyMoEModel())
        loss_off = off.step_fn(off, 0).loss
        torch.manual_seed(0)
        on = make_session(_moe_config(z_weight=4.0), model=TinyMoEModel())
        loss_on = on.step_fn(on, 0).loss
        assert loss_on == pytest.approx(loss_off + 4.0 * 0.25, abs=1e-4)

    def test_set_moe_step_receives_the_live_step(self):
        session = make_session(_moe_config(), model=TinyMoEModel())
        session.step_fn(session, 7)
        assert session.model.moe_steps == [(7, session.config.train.max_steps)]


class TestMoEMetrics:
    def _run_one_step(self, model, config):
        session = make_session(config, model=model)
        logged: list = []
        session.tracker.log_eval = lambda m, s: logged.append((m, s))  # type: ignore[method-assign]
        run_training_loop(session)
        return logged

    def test_expert_balance_is_min_over_max(self):
        model = TinyMoEModel(counts={"0": torch.tensor([2.0, 8.0])})
        logged = self._run_one_step(model, _moe_config(max_steps=1))
        moe = [m for m, _ in logged if "moe/aux_loss" in m][0]
        assert moe["moe/expert_balance"] == pytest.approx(0.25)
        assert moe["moe/aux_loss"] == pytest.approx(0.5)
        assert moe["moe/router_z_loss"] == pytest.approx(0.25)

    def test_balance_is_omitted_when_there_are_no_counts(self):
        logged = self._run_one_step(TinyMoEModel(counts={}), _moe_config(max_steps=1))
        moe = [m for m, _ in logged if "moe/aux_loss" in m][0]
        assert "moe/expert_balance" not in moe


class TestPerDatasetMetrics:
    def _mixture_session(self, max_steps=1):
        data = _mixture_pipeline()
        batch = {
            "input_ids": torch.zeros(4, 4, dtype=torch.long),
            "labels": torch.zeros(4, 4, dtype=torch.long),
            "dataset_idx": torch.tensor([0, 0, 1, 1]),
        }
        data.dataloader = [batch]
        config = make_config(max_steps=max_steps, batch_size=4)
        session = make_session(config, data=data, phases=PhaseState())
        logged: list = []
        session.tracker.log_eval = lambda m, s: logged.append(m)  # type: ignore[method-assign]
        return session, logged

    def test_step_reports_per_dataset_losses_and_tokens(self):
        session, _ = self._mixture_session()
        result = session.step_fn(session, 0)
        assert set(result.dataset_loss_sums) == {"a", "b"}
        assert result.dataset_loss_counts == {"a": 1, "b": 1}
        # 2 rows per source x seq_len tokens each
        assert result.dataset_token_counts == {"a": 2 * 4, "b": 2 * 4}

    def test_loop_logs_the_mean_and_the_token_totals(self):
        session, logged = self._mixture_session()
        run_training_loop(session)
        ds = [m for m in logged if any(k.startswith("loss/") for k in m)][0]
        assert set(ds) == {"loss/a", "loss/b", "data/a/tokens", "data/b/tokens"}
        assert ds["data/a/tokens"] == 8.0

    def test_a_source_absent_from_the_batch_is_omitted(self):
        session, _ = self._mixture_session()
        session.batches.pipeline.dataloader = [
            {
                "input_ids": torch.zeros(4, 4, dtype=torch.long),
                "labels": torch.zeros(4, 4, dtype=torch.long),
                "dataset_idx": torch.tensor([0, 0, 0, 0]),  # all from "a"
            }
        ]
        result = session.step_fn(session, 0)
        assert set(result.dataset_loss_sums) == {"a"}


class TestLoopPeriodicWork:
    def test_nccl_failure_stops_the_run_without_a_final_save(self, monkeypatch):
        import kempnerforge.training.loop as loop

        monkeypatch.setattr(loop, "check_nccl_health", lambda: False)
        config = make_config(max_steps=6)
        config.train.nccl_health_check_interval = 1
        session = make_session(config)
        step, _ = run_training_loop(session)
        ckpt: FakeCheckpointManager = session.checkpointer  # type: ignore[assignment]
        assert step == 1  # broke on the first health check
        assert ckpt.saves == []  # not a clean completion

    def test_healthy_nccl_does_not_stop_the_run(self, monkeypatch):
        import kempnerforge.training.loop as loop

        monkeypatch.setattr(loop, "check_nccl_health", lambda: True)
        config = make_config(max_steps=3)
        config.train.nccl_health_check_interval = 1
        step, _ = run_training_loop(make_session(config))
        assert step == 3

    def test_eval_runs_on_its_interval_and_feeds_the_hook(self, monkeypatch):
        import kempnerforge.training.loop as loop
        from kempnerforge.config.eval import EvalConfig

        monkeypatch.setattr(loop, "run_eval", lambda *a, **k: {"eval/loss": 1.5})
        config = make_config(max_steps=4)
        config.eval = EvalConfig(enabled=True, interval=2, steps=1, dataset_path="/x")
        hook = RecordingHook()
        session = make_session(config, eval_dataloader=["BATCH"], hooks=HookRunner([hook]))
        evals: list = []
        session.hooks.on_eval_end = lambda m, s: evals.append(s)  # type: ignore[method-assign]
        run_training_loop(session)
        assert evals == [2, 4]  # steps 2 and 4 only

    def test_eval_is_skipped_without_a_dataloader(self, monkeypatch):
        import kempnerforge.training.loop as loop
        from kempnerforge.config.eval import EvalConfig

        def _boom(*a, **k):
            raise AssertionError("run_eval must not be called without a loader")

        monkeypatch.setattr(loop, "run_eval", _boom)
        config = make_config(max_steps=2)
        config.eval = EvalConfig(enabled=True, interval=1, steps=1, dataset_path="/x")
        run_training_loop(make_session(config, eval_dataloader=None))

    def test_profiler_is_started_stepped_and_stopped(self):
        class CountingProfiler:
            def __init__(self) -> None:
                self.started = self.stopped = self.steps = 0

            def start(self) -> None:
                self.started += 1

            def stop(self) -> None:
                self.stopped += 1

            def step(self) -> None:
                self.steps += 1

        prof = CountingProfiler()
        session = make_session(
            make_config(max_steps=3),
            profiler=prof,
            runtime=dataclasses.replace(_runtime(), rank=1),
        )
        run_training_loop(session)
        assert (prof.started, prof.steps, prof.stopped) == (1, 3, 1)

    def test_phase_transition_resets_the_batch_iterator(self):
        data = _mixture_pipeline()
        data.dataloader = [
            {
                "input_ids": torch.zeros(2, 4, dtype=torch.long),
                "labels": torch.zeros(2, 4, dtype=torch.long),
            }
        ]
        config = make_config(max_steps=3)
        config.data = DataConfig(
            phases=[TrainingPhase(start_step=2, dataset_weights={"a": 5.0}, lr_scale=0.5)]
        )
        phases = build_phase_state(config, data, step=0)
        session = make_session(config, data=data, phases=phases)
        run_training_loop(session)
        # the phase fired and re-weighted the sampler
        assert data.mixture_sampler.calls  # type: ignore[union-attr]
        assert phases.lr_scale == 0.5


class TestFreezeStageFence:
    """The stage hook must drain an in-flight save before flipping requires_grad.

    Uses the real vlm_step, because _apply_freeze_stages only runs for VLM
    configs and the loop's VLM metrics require a step body that reports a
    text-token count.
    """

    def _vlm_session(self, monkeypatch, start_step, order):
        import kempnerforge.training.loop as loop
        from kempnerforge.config.vlm import FreezeSpec, FreezeStage

        monkeypatch.setattr(
            loop, "apply_freeze_specs", lambda m, s, p: order.append("freeze") or ["p"]
        )
        config = _vlm_config(
            freeze_schedule=(FreezeStage(start_step=start_step, specs=(FreezeSpec("adapter"),)),)
        )
        config.train.max_steps = 3
        config.train.batch_size, config.train.seq_len = 2, 4
        config.checkpoint.interval = 1000
        batch = {
            "pixel_values": torch.zeros(2, 3, 4, 4),
            "input_ids": torch.zeros(2, 4, dtype=torch.long),
            "labels": torch.zeros(2, 4, dtype=torch.long),
        }
        session = make_session(config, model=TinyVlmModel(), data=DataPipeline(dataloader=[batch]))
        session.step_fn = vlm_step
        session.checkpointer.flush_pending_save = lambda: order.append("flush")  # type: ignore
        return session

    def test_stage_drains_the_async_save_before_flipping_requires_grad(self, monkeypatch):
        order: list = []
        run_training_loop(self._vlm_session(monkeypatch, start_step=2, order=order))
        # the fence: metadata for the in-flight save must land before the flip
        assert order == ["flush", "freeze"]

    def test_no_stage_at_this_step_does_not_flush(self, monkeypatch):
        order: list = []
        run_training_loop(self._vlm_session(monkeypatch, start_step=99, order=order))
        assert order == []


# ---------------------------------------------------------------------------
# Remaining branches
# ---------------------------------------------------------------------------


class TestBatchStreamGuard:
    def test_next_batch_without_a_loader_names_the_contract(self):
        # Reachable by an example writing its own step_fn that forgets has_data.
        with pytest.raises(RuntimeError, match="check has_data first"):
            BatchStream(DataPipeline()).next_batch()


class TestPipelineSyntheticBatches:
    def test_pp_falls_back_to_random_tokens_without_a_dataloader(self, monkeypatch):
        import kempnerforge.training.loop as loop
        from kempnerforge.training.runtime import PipelineBundle

        monkeypatch.setattr(loop, "pp_group", lambda _r: "G")
        monkeypatch.setattr(loop.dist, "broadcast", lambda *a, **k: None)
        sched = FakeSchedule()
        config = make_config()
        config.distributed.pp = 2
        session = make_session(
            config,
            data=DataPipeline(),  # no loader -> synthetic path
            pipeline=PipelineBundle(rank=0, size=2, schedule=sched),
        )
        session.step_fn(session, 0)
        fed = sched.calls[0]["args"][0]
        # grad_accum microbatches concatenated along dim 0
        assert fed.shape == (
            config.train.batch_size * config.train.grad_accum_steps,
            config.train.seq_len,
        )
        assert fed.max().item() < config.model.vocab_size


class TinyVlmMoEModel(TinyVlmModel):
    def set_moe_step(self, step, max_steps) -> None:
        pass

    def get_moe_aux_loss(self):
        return torch.tensor(0.5)

    def get_moe_router_z_loss(self):
        return torch.tensor(0.25)

    def get_expert_counts(self):
        return {}


class TestVlmMoEStep:
    def test_vlm_step_adds_the_moe_terms(self):
        batch = {
            "pixel_values": torch.zeros(2, 3, 4, 4),
            "input_ids": torch.zeros(2, 4, dtype=torch.long),
            "labels": torch.zeros(2, 4, dtype=torch.long),
        }
        # One model instance for both runs, so the CE term is identical and the
        # only difference is the MoE weights.
        model = TinyVlmMoEModel()

        def _loss(aux_weight, z_weight):
            config = _moe_config(z_weight=z_weight, aux_weight=aux_weight)
            config.train.batch_size, config.train.seq_len = 2, 4
            s = make_session(config, model=model, data=DataPipeline(dataloader=[batch]))
            s.step_fn = vlm_step
            return s.step_fn(s, 0).loss

        base = _loss(0.0, 0.0)
        loss = _loss(2.0, 4.0)
        # 2.0*0.5 aux + 4.0*0.25 z on top of the same CE
        assert loss == pytest.approx(base + 2.0, abs=1e-4)


class TestMixtureHfSource:
    def test_hf_source_in_a_mixture_is_built_and_weighted(self, monkeypatch):
        import kempnerforge.training.data_pipeline as dp
        from kempnerforge.config.data import DatasetSource

        seen: dict = {}
        _stub_loaders(monkeypatch, seen)
        monkeypatch.setattr(dp, "HuggingFaceDataset", lambda **k: FakeDataset(**k))

        monkeypatch.setattr(dp, "MixtureDataset", lambda subs, names: FakeMixtureDataset(names))

        config = make_config(seq_len=16)
        config.data = DataConfig(
            datasets=[
                DatasetSource(name="hf", hf_name="wikitext", weight=3.0),
            ],
            tokenizer_path="tok",
            num_workers=0,
        )
        data = dp.build_data_pipeline(config, _runtime())
        assert data.mixture_weights == {"hf": 3.0}
        assert seen["mixture_sampler"]["kwargs"]["weights"] == [3.0]

    def test_hf_source_without_a_tokenizer_is_rejected(self):
        import kempnerforge.training.data_pipeline as dp
        from kempnerforge.config.data import DatasetSource

        config = make_config(seq_len=16)
        config.data = DataConfig(
            datasets=[DatasetSource(name="hf", hf_name="wikitext")], num_workers=0
        )
        with pytest.raises(ValueError, match="tokenizer_path required for HF dataset"):
            dp.build_data_pipeline(config, _runtime())


class TestEosFromTokenizer:
    def test_eos_is_read_off_the_tokenizer(self, monkeypatch, mmap_data_dir):
        import sys
        import types

        from kempnerforge.training.data_pipeline import _resolve_eos_token_id

        fake = types.ModuleType("transformers")

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(path):
                return types.SimpleNamespace(eos_token_id=99)

        fake.AutoTokenizer = AutoTokenizer  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "transformers", fake)

        config = make_config()
        config.data = DataConfig(
            dataset_path=mmap_data_dir, pack_sequences=True, tokenizer_path="tok"
        )
        assert _resolve_eos_token_id(config) == 99


class TestPpBuildFlags:
    def test_fp8_and_compile_are_applied_on_the_pp_path(self, monkeypatch):
        stage, calls = FakeStageModule(), {}
        entry = _stub_pp(monkeypatch, stage, calls)
        compiled: list = []
        monkeypatch.setattr(torch, "compile", lambda m: compiled.append(m) or m)

        config = make_config()
        config.distributed.pp = 2
        config.train.mixed_precision = "fp8"
        config.train.compile_model = True

        entry.build_model(config, _pp_runtime(), "LOSS")
        assert calls.get("apply_float8") is True
        assert compiled and compiled[0] is stage
