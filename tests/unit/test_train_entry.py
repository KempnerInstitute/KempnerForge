"""Unit tests for the extracted training entry point.

These cover the seams that only became reachable without a GPU once
``scripts/train.py:main()`` was decomposed: the step loop driven by a fake
checkpoint manager, the batch iterator, phase scheduling, the step-body
dispatch, and the data-pipeline builder.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator
from pathlib import Path
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
    StepFn,
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
        self.tracking_source = None
        self.checkpoint_registrar = None

    def save(self, step, tokens_seen=0, scheduler=None, dataloader=None, extra=None) -> None:
        self.saves.append((step, tokens_seen, dict(extra or {})))

    def set_tracking_source(self, source) -> None:
        self.tracking_source = source

    def set_checkpoint_registrar(self, registrar) -> None:
        self.checkpoint_registrar = registrar

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
        self.lrs: list[float] = []
        self.saves: list[int] = []

    def on_train_begin(self, config: JobConfig) -> None:
        self.begins += 1

    def on_step_end(self, ctx) -> None:
        self.steps.append(ctx.step)
        self.lrs.append(ctx.lr)

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


def _nan_on(steps: set[int], step_fn: StepFn = text_step) -> StepFn:
    """Wrap a step body so the iterations that start at ``steps`` report a NaN loss."""

    def nan_step(session: TrainingSession, step: int):
        result = step_fn(session, step)
        return dataclasses.replace(result, loss=float("nan")) if step in steps else result

    return nan_step


def _bad_update_on(bad_step: int, step_fn: StepFn = text_step) -> StepFn:
    """Wrap a step body so the iteration that starts at ``bad_step`` makes a bad update.

    The step's loss and the gradient norm it reports stay finite, so the NaN check
    lets it through, but its gradients turn NaN after the norm is taken, so the
    optimizer writes NaN into the weights and every later loss is NaN.
    """

    def bad_update(session: TrainingSession, step: int):
        result = step_fn(session, step)
        if step == bad_step:
            for p in session.model.parameters():
                if p.grad is not None:
                    p.grad.fill_(float("nan"))
        return result

    return bad_update


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
        stream = BatchStream(DataPipeline(dataloader=_fake_batches(3)))
        assert int(stream.next_batch()["input_ids"]) == 0

    def test_reset_forces_a_fresh_iterator(self):
        stream = BatchStream(DataPipeline(dataloader=_fake_batches(3)))
        assert int(stream.next_batch()["input_ids"]) == 0
        stream.reset()
        assert int(stream.next_batch()["input_ids"]) == 0

    def test_close_is_safe_without_a_loader(self):
        BatchStream(DataPipeline()).close()

    def test_close_is_safe_on_a_worker_free_loader(self):
        stream = BatchStream(DataPipeline(dataloader=_fake_batches(3)))
        stream.next_batch()
        stream.close()
        assert stream._iter is None

    def test_close_drains_the_workers_explicitly(self):
        """Shielded workers ignore SIGTERM, so terminate() cannot reap them.

        close() must go through the loader's own sentinel path instead, or the
        interpreter blocks at exit joining daemonic children that refuse to die.
        """

        class _FakeIterator:
            def __init__(self):
                self.shutdown_calls = 0

            def _shutdown_workers(self):
                self.shutdown_calls += 1

        class _FakeInner:
            def __init__(self):
                self._iterator = _FakeIterator()

        class _FakeLoader:
            def __init__(self):
                self._dataloader = _FakeInner()

        loader = _FakeLoader()
        iterator = loader._dataloader._iterator
        stream = BatchStream(DataPipeline(dataloader=loader))

        stream.close()

        assert iterator.shutdown_calls == 1
        assert loader._dataloader._iterator is None

    @pytest.mark.parametrize("alive", [True, False], ids=["stuck", "already-gone"])
    def test_close_kills_a_worker_the_drain_left_behind(self, alive):
        """SIGKILL is the only way out for a worker that ignored the sentinel.

        These workers ignore SIGTERM by design, so terminate() cannot reap one
        that failed to drain. Without a kill the interpreter blocks at exit on a
        daemonic child that refuses to die -- the exact hang the shield would
        otherwise reintroduce by removing the usual escape.
        """

        class _FakeWorker:
            def __init__(self, alive):
                self._alive = alive
                self.killed = 0
                self.joined = 0

            def is_alive(self):
                return self._alive

            def kill(self):
                self.killed += 1
                self._alive = False

            def join(self, timeout=None):
                self.joined += 1

        class _FakeIterator:
            def __init__(self, worker):
                self._workers = [worker]

            def _shutdown_workers(self):
                pass

        class _FakeInner:
            def __init__(self, worker):
                self._iterator = _FakeIterator(worker)

        class _FakeLoader:
            def __init__(self, worker):
                self._dataloader = _FakeInner(worker)

        worker = _FakeWorker(alive)
        BatchStream(DataPipeline(dataloader=_FakeLoader(worker))).close()

        assert worker.killed == (1 if alive else 0)
        assert worker.joined == (1 if alive else 0)

    def test_close_is_idempotent(self):
        stream = BatchStream(DataPipeline(dataloader=_fake_batches(3)))
        stream.next_batch()
        stream.close()
        stream.close()

    def test_ensure_started_is_idempotent(self):
        stream = BatchStream(DataPipeline(dataloader=_fake_batches(3)))
        stream.ensure_started()
        assert int(stream.next_batch()["input_ids"]) == 0
        stream.ensure_started()
        assert int(stream.next_batch()["input_ids"]) == 1

    def test_empty_shard_raises_a_named_error(self):
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


class TestSessionBatches:
    def test_follows_a_replaced_data_pipeline(self):
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


class TestAutocastFallback:
    """``TrainingSession.autocast_dtype`` puts the step body's forward under ``torch.autocast``."""

    @pytest.mark.parametrize(
        ("autocast_dtype", "expected"), [(None, torch.float32), (torch.bfloat16, torch.bfloat16)]
    )
    def test_text_step_forward(
        self, autocast_dtype: torch.dtype | None, expected: torch.dtype
    ) -> None:
        session = make_session(make_config(), autocast_dtype=autocast_dtype)
        seen: list[torch.dtype] = []
        session.model.out.register_forward_hook(lambda _m, _i, out: seen.append(out.dtype))
        session.step_fn(session, 0)
        assert seen == [expected]
        for p in session.model.parameters():
            assert p.dtype == torch.float32
            assert p.grad is not None and p.grad.dtype == torch.float32

    def test_vlm_step_forward(self) -> None:
        model = TinyVlmModel()
        batch = {
            "pixel_values": torch.zeros(2, 3, 4, 4),
            "input_ids": torch.zeros(2, 4, dtype=torch.long),
            "labels": torch.zeros(2, 4, dtype=torch.long),
        }
        session = make_session(
            make_config(),
            model=model,
            data=DataPipeline(dataloader=[batch]),
            step_fn=vlm_step,
            autocast_dtype=torch.bfloat16,
        )
        seen: list[torch.dtype] = []
        model.proj.register_forward_hook(lambda _m, _i, out: seen.append(out.dtype))
        session.step_fn(session, 0)
        assert seen == [torch.bfloat16]

    def test_pipeline_schedule_runs_under_autocast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        enabled: list[bool] = []

        class Schedule(FakeSchedule):
            def step(self, *args: Any, **kwargs: Any) -> None:
                enabled.append(torch.is_autocast_enabled("cpu"))
                super().step(*args, **kwargs)

        session, _ = TestPipelineStep()._session(monkeypatch, rank=0, size=2, sched=Schedule())
        session.autocast_dtype = torch.bfloat16
        session.step_fn(session, 0)
        assert enabled == [True]

    def test_context(self) -> None:
        with make_session(make_config(), autocast_dtype=torch.bfloat16).autocast():
            assert torch.is_autocast_enabled("cpu")
            assert torch.get_autocast_dtype("cpu") == torch.bfloat16
        with make_session(make_config()).autocast():
            assert not torch.is_autocast_enabled("cpu")


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
        assert step == 2
        assert ckpt.saves == []
        assert ckpt.waits == 1

    def test_a_finite_loss_with_nan_gradients_skips_the_step(self):
        """NaN gradients under a finite loss never reach the optimizer."""

        def loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            logits.register_hook(lambda grad: torch.full_like(grad, float("nan")))
            return logits.float().mean()

        config = make_config(max_steps=10)
        session = make_session(
            config, loss_fn=loss_fn, nan_detector=NaNDetector(action="warn", max_consecutive=3)
        )
        before = [p.detach().clone() for p in session.model.parameters()]
        step, _ = run_training_loop(session)
        assert step == 2
        assert session.nan_detector.state.total_nans == 3
        for old, new in zip(before, session.model.parameters(), strict=True):
            assert torch.equal(old, new)

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

    def test_shutdown_on_a_scheduled_step_saves_it_once(self):
        """The scheduled save at step 2, with its save hook, is the emergency checkpoint."""
        config = make_config(max_steps=10)
        config.checkpoint.interval = 2
        ckpt = FakeCheckpointManager()
        handler = FakeShutdownHandler(after_steps=2)
        hook = RecordingHook()
        session = make_session(
            config, checkpointer=ckpt, shutdown_handler=handler, hooks=HookRunner([hook])
        )
        step, _ = run_training_loop(session)
        assert step == 2
        assert ckpt.saved_steps == [2]
        assert hook.saves == [2]
        assert handler.finished is True
        assert ckpt.waits == 2

    def test_shutdown_on_a_scheduled_step_waits_for_it_before_finishing(self, monkeypatch):
        """The scheduled save is flushed while the shutdown watchdog is still armed."""
        config = make_config(max_steps=10)
        config.checkpoint.interval = 2
        ckpt = FakeCheckpointManager()
        handler = FakeShutdownHandler(after_steps=2)
        finished_at_wait: list[bool] = []
        real_wait = ckpt.wait

        def wait() -> None:
            finished_at_wait.append(handler.finished)
            real_wait()

        monkeypatch.setattr(ckpt, "wait", wait)
        session = make_session(config, checkpointer=ckpt, shutdown_handler=handler)
        run_training_loop(session)
        assert finished_at_wait == [False, True]

    def test_shutdown_after_a_scheduled_save_still_saves_its_own_step(self):
        """Step 2 is saved on the schedule; a shutdown at step 3 must still save step 3."""
        config = make_config(max_steps=10)
        config.checkpoint.interval = 2
        ckpt = FakeCheckpointManager()
        handler = FakeShutdownHandler(after_steps=3)
        session = make_session(config, checkpointer=ckpt, shutdown_handler=handler)
        step, _ = run_training_loop(session)
        assert step == 3
        assert ckpt.saved_steps == [2, 3]
        assert handler.finished is True

    @pytest.mark.parametrize(
        "after_steps,expected",
        [(6, [0, 1, 2, 4, 6]), (1, [0, 1])],
        ids=["at-step-6", "at-step-1"],
    )
    def test_shutdown_inside_a_dynamic_window_saves_each_step_once(self, after_steps, expected):
        """Inside the window milestones decide: step 1 is saved once and step 6 is still saved."""
        config = make_config(max_steps=10)
        config.checkpoint.interval = 2
        config.checkpoint.dyn_ckpt_window = DynamicCheckpointWindow(start=0, stop=8)
        ckpt = FakeCheckpointManager()
        handler = FakeShutdownHandler(after_steps=after_steps)
        session = make_session(config, checkpointer=ckpt, shutdown_handler=handler)
        step, _ = run_training_loop(session)
        assert step == after_steps
        assert ckpt.saved_steps == expected
        assert handler.finished is True

    @pytest.mark.parametrize("mode", ["disabled", "async", "async_with_pinned_mem"])
    def test_shutdown_on_a_scheduled_step_commits_that_one_save(self, monkeypatch, tmp_path, mode):
        """The real manager saves the step once and has committed it when the loop returns."""
        from kempnerforge.checkpoint.manager import CheckpointManager
        from kempnerforge.config.checkpoint import AsyncCheckpointMode

        config = make_config(max_steps=10)
        config.checkpoint.dir = str(tmp_path)
        config.checkpoint.interval = 2
        config.checkpoint.async_mode = AsyncCheckpointMode(mode)
        session = make_session(config, shutdown_handler=FakeShutdownHandler(after_steps=2))
        mgr = CheckpointManager(config.checkpoint, session.model, session.optimizer)
        saved: list[int] = []
        real_save = mgr.save

        def counting_save(step: int, **kwargs: Any) -> None:
            saved.append(step)
            real_save(step, **kwargs)

        monkeypatch.setattr(mgr, "save", counting_save)
        session.checkpointer = mgr
        step, _ = run_training_loop(session)
        assert step == 2
        assert saved == [2]
        assert (tmp_path / "latest").resolve() == (tmp_path / "step_2").resolve()
        assert (tmp_path / "step_2" / ".metadata").is_file()

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


class TestNanSkippedStep:
    """A step skipped for a non-finite loss still crosses the step boundary.

    It skips the optimizer update, but the LR schedule, the step-keyed
    transitions and the shutdown check move with the step counter as on a
    finite step.
    """

    def test_lr_schedule_stays_on_the_step_counter(self):
        config = make_config(max_steps=6)
        hook = RecordingHook()
        session = make_session(config, step_fn=_nan_on({1, 3}), hooks=HookRunner([hook]))
        session.scheduler = torch.optim.lr_scheduler.LambdaLR(
            session.optimizer, lambda s: 1.0 / (1 + s)
        )
        step, _ = run_training_loop(session)
        assert step == 6
        assert session.scheduler.last_epoch == 6
        assert hook.steps == [1, 3, 5, 6]
        assert hook.lrs == pytest.approx([0.05 / (1 + s) for s in hook.steps])

    def test_nan_stop_leaves_the_schedule_on_the_returned_step(self):
        config = make_config(max_steps=10)
        session = make_session(
            config,
            step_fn=_nan_on(set(range(10))),
            nan_detector=NaNDetector(action="warn", max_consecutive=3),
        )
        step, _ = run_training_loop(session)
        assert step == 2
        assert session.scheduler.last_epoch == 2

    def test_skipped_step_adds_no_tokens(self):
        """``tokens_seen`` counts the two steps that updated the model, not the skipped one."""
        config = make_config(max_steps=3)
        step, tokens_seen = run_training_loop(make_session(config, step_fn=_nan_on({1})))
        assert step == 3
        assert tokens_seen == 2 * config.train.batch_size * config.train.seq_len

    def test_skipped_step_runs_no_eval_nccl_check_profiler_step_or_metrics(self, monkeypatch):
        """Step 2 is due for eval and the NCCL check but is skipped, so only step 1's work runs."""
        import kempnerforge.training.loop as loop
        from kempnerforge.config.eval import EvalConfig

        calls: list[str] = []
        monkeypatch.setattr(loop, "run_eval", lambda *a, **k: calls.append("eval") or {})
        monkeypatch.setattr(loop, "check_nccl_health", lambda: calls.append("nccl") or True)

        class RecordingProfiler:
            def start(self) -> None:
                pass

            def stop(self) -> None:
                pass

            def step(self) -> None:
                calls.append("profiler")

        config = make_config(max_steps=2)
        config.eval = EvalConfig(enabled=True, interval=2, steps=1, dataset_path="/x")
        config.train.nccl_health_check_interval = 2
        session = make_session(
            config,
            step_fn=_nan_on({1}),
            eval_dataloader=["BATCH"],
            profiler=RecordingProfiler(),
            runtime=dataclasses.replace(_runtime(), rank=1),
        )
        logged: list[int] = []
        end_step = session.tracker.end_step
        monkeypatch.setattr(
            session.tracker, "end_step", lambda **kw: logged.append(kw["step"]) or end_step(**kw)
        )
        run_training_loop(session)
        assert calls == ["profiler"]
        assert logged == [1]

    def test_skipped_last_step_makes_no_final_save(self):
        """A run whose last step is skipped is not a clean finish, so it writes no final save."""
        config = make_config(max_steps=3)
        config.checkpoint.interval = 100
        ckpt = FakeCheckpointManager()
        step, _ = run_training_loop(make_session(config, checkpointer=ckpt, step_fn=_nan_on({2})))
        assert step == 3
        assert ckpt.saves == []

    def test_phase_lr_scale_applies_to_the_next_steps_lr(self):
        config = make_config(max_steps=1)
        session = make_session(config, phases=PhaseState(lr_scale=0.5), step_fn=_nan_on({0}))
        run_training_loop(session)
        assert session.optimizer.param_groups[0]["lr"] == pytest.approx(0.025)

    def _phase_session(self, phase: TrainingPhase, **overrides: Any) -> TrainingSession:
        """A two-step mixture run with one data phase; ``overrides`` go to ``make_session``."""
        data = _mixture_pipeline()
        data.dataloader = [
            {
                "input_ids": torch.zeros(2, 4, dtype=torch.long),
                "labels": torch.zeros(2, 4, dtype=torch.long),
            }
        ]
        config = make_config(max_steps=2)
        config.data = DataConfig(phases=[phase])
        phases = build_phase_state(config, data, step=0)
        return make_session(config, data=data, phases=phases, **overrides)

    def test_phase_due_on_a_skipped_step_fires_on_it(self, monkeypatch):
        """Step 2 is reached by a NaN step and is the last one, so only that step can fire it."""
        session = self._phase_session(
            TrainingPhase(start_step=2, dataset_weights={"a": 5.0}, lr_scale=0.5),
            step_fn=_nan_on({1}),
        )
        resets: list[int] = []
        reset = session.batches.reset
        monkeypatch.setattr(session.batches, "reset", lambda: resets.append(1) or reset())
        run_training_loop(session)
        assert session.phases.next_idx == 1
        assert session.phases.lr_scale == 0.5
        assert session.data.mixture_sampler.calls == [([5.0, 3.0], 1.0)]
        assert resets == [1]

    def test_a_phase_reached_by_a_skipped_step_scales_the_lr_from_the_same_update(self):
        """The phase's ``lr_scale`` first applies to step 3's update, as in an all-finite run."""

        def lr_at_each_step(nan_steps: set[int]) -> list[float]:
            lrs: list[float] = []
            body = _nan_on(nan_steps)

            def step_fn(session: TrainingSession, step: int):
                lrs.append(session.optimizer.param_groups[0]["lr"])
                return body(session, step)

            session = self._phase_session(
                TrainingPhase(start_step=2, dataset_weights={"a": 5.0}, lr_scale=0.5),
                step_fn=step_fn,
            )
            session.config.train.max_steps = 4
            run_training_loop(session)
            return lrs

        finite = lr_at_each_step(set())
        assert finite == pytest.approx([0.05, 0.05, 0.05, 0.025])
        assert lr_at_each_step({1}) == finite

    def test_shutdown_is_acted_on_at_the_skipped_step(self):
        """A pending shutdown is acted on at the first NaN step, not after the NaN run."""
        config = make_config(max_steps=10)
        ckpt = FakeCheckpointManager()
        handler = FakeShutdownHandler(after_steps=1)
        session = make_session(
            config,
            loss_fn=lambda logits, labels: logits.sum() * float("nan"),
            nan_detector=NaNDetector(action="warn", max_consecutive=3),
            checkpointer=ckpt,
            shutdown_handler=handler,
        )
        step, _ = run_training_loop(session)
        assert step == 1
        assert handler.checks == 1
        assert ckpt.saved_steps == [1]
        assert handler.finished is True
        assert ckpt.waits == 1

    def test_shutdown_on_a_skipped_scheduled_step_saves_it_once(self, monkeypatch):
        """The schedule skips a NaN step's save, so the emergency save is its only one."""
        config = make_config(max_steps=10)
        config.checkpoint.interval = 2
        ckpt = FakeCheckpointManager()
        handler = FakeShutdownHandler(after_steps=2)
        session = make_session(
            config, checkpointer=ckpt, shutdown_handler=handler, step_fn=_nan_on({1})
        )
        epochs: list[int] = []
        real_save = ckpt.save

        def save(step: int, **kwargs: Any) -> None:
            epochs.append(kwargs["scheduler"].last_epoch)
            real_save(step, **kwargs)

        monkeypatch.setattr(ckpt, "save", save)
        step, tokens_seen = run_training_loop(session)
        assert step == 2
        assert ckpt.saves == [(2, tokens_seen, {})]
        assert ckpt.waits == 1
        assert epochs == [2]
        assert handler.finished is True

    def test_nan_stop_is_not_saved_even_with_a_shutdown_pending(self):
        """The NaN that trips the stop ends the run before the shutdown check: no save."""
        config = make_config(max_steps=10)
        ckpt = FakeCheckpointManager()
        handler = FakeShutdownHandler(after_steps=2)
        session = make_session(
            config,
            loss_fn=lambda logits, labels: logits.sum() * float("nan"),
            nan_detector=NaNDetector(action="warn", max_consecutive=2),
            checkpointer=ckpt,
            shutdown_handler=handler,
        )
        step, _ = run_training_loop(session)
        assert step == 1
        assert handler.checks == 1
        assert ckpt.saves == []
        assert handler.finished is False

    def test_emergency_checkpoint_records_the_phase_the_skipped_step_entered(self, monkeypatch):
        """It also saves the dataloader, so the resume continues from the same data position."""
        ckpt = FakeCheckpointManager()
        session = self._phase_session(
            TrainingPhase(start_step=2, dataset_weights={"a": 5.0}),
            checkpointer=ckpt,
            shutdown_handler=FakeShutdownHandler(after_steps=2),
            step_fn=_nan_on({1}),
        )
        loaders: list = []
        real_save = ckpt.save

        def save(step: int, **kwargs: Any) -> None:
            loaders.append(kwargs.get("dataloader"))
            real_save(step, **kwargs)

        monkeypatch.setattr(ckpt, "save", save)
        step, tokens_seen = run_training_loop(session)
        assert step == 2
        assert ckpt.saves == [(2, tokens_seen, {"phase_idx": 1})]
        assert loaders == [session.data.dataloader]
        assert loaders[0] is not None

    def test_resume_from_its_emergency_checkpoint_matches_an_uninterrupted_run(self, tmp_path):
        """The real manager carries the skipped step's weights, schedule and RNG across."""
        from kempnerforge.checkpoint.manager import CheckpointManager

        def run(ckpt_dir, shutdown_after=None, resume=False):
            config = make_config(max_steps=5)
            config.checkpoint.dir = str(ckpt_dir)
            config.checkpoint.interval = 100
            model = TinyTextModel()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0 / (1 + s))
            session = make_session(
                config,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step_fn=_nan_on({1}),
                shutdown_handler=FakeShutdownHandler(after_steps=shutdown_after),
            )
            session.checkpointer = CheckpointManager(config.checkpoint, model, optimizer)
            step, tokens_seen = 0, 0
            if resume:
                step, tokens_seen, _ = session.checkpointer.load(scheduler=scheduler)
            return session, run_training_loop(session, step=step, tokens_seen=tokens_seen)

        torch.manual_seed(0)
        full, full_end = run(tmp_path / "full")
        torch.manual_seed(0)
        _, stop_end = run(tmp_path / "split", shutdown_after=2)
        torch.manual_seed(1)
        resumed, resumed_end = run(tmp_path / "split", resume=True)
        assert stop_end[0] == 2
        assert resumed_end == full_end
        assert resumed.scheduler.last_epoch == full.scheduler.last_epoch == 5
        params = zip(full.model.parameters(), resumed.model.parameters(), strict=True)
        assert all(torch.equal(p, q) for p, q in params)

    def _keep_last_one(
        self, ckpt_dir, step_fn: StepFn, mode: str = "disabled"
    ) -> tuple[int, FakeShutdownHandler]:
        """Save every 2 steps with ``keep_last_n=1`` through the real manager; shut down at 4."""
        from kempnerforge.checkpoint.manager import CheckpointManager
        from kempnerforge.config.checkpoint import AsyncCheckpointMode

        config = make_config(max_steps=10)
        config.checkpoint.dir = str(ckpt_dir)
        config.checkpoint.interval = 2
        config.checkpoint.keep_last_n = 1
        config.checkpoint.async_mode = AsyncCheckpointMode(mode)
        handler = FakeShutdownHandler(after_steps=4)
        session = make_session(config, step_fn=step_fn, shutdown_handler=handler)
        session.checkpointer = CheckpointManager(
            config.checkpoint, session.model, session.optimizer
        )
        step, _ = run_training_loop(session)
        return step, handler

    @pytest.mark.parametrize("mode", ["disabled", "async", "async_with_pinned_mem"])
    def test_shutdown_after_a_bad_update_keeps_the_last_good_checkpoint(
        self, monkeypatch, caplog, tmp_path, mode
    ):
        """The update into step 3 writes NaN weights, so skipped step 4's shutdown saves nothing.

        Saved, those weights would become ``latest`` and, with ``keep_last_n=1``,
        prune step 2, so a resume would load NaN weights.
        """
        import logging

        from kempnerforge.checkpoint.manager import CheckpointManager

        monkeypatch.setattr(logging.getLogger("kempnerforge"), "propagate", True)
        with caplog.at_level(logging.ERROR, logger="kempnerforge"):
            step, handler = self._keep_last_one(tmp_path, _bad_update_on(2), mode)
        assert step == 4
        assert handler.finished is True
        assert sorted(d.name for d in tmp_path.glob("step_*")) == ["step_2"]
        assert (tmp_path / "latest").resolve() == (tmp_path / "step_2").resolve()
        errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert any("parameters are not finite" in m for m in errors)
        config = make_config()
        config.checkpoint.dir = str(tmp_path)
        resumed = TinyTextModel()
        mgr = CheckpointManager(
            config.checkpoint, resumed, torch.optim.SGD(resumed.parameters(), lr=0.05)
        )
        assert mgr.load()[0] == 2
        assert all(torch.isfinite(p).all() for p in resumed.parameters())

    def test_shutdown_on_a_skipped_step_with_finite_weights_still_saves_it(self, tmp_path):
        """A NaN loss alone leaves the weights finite, so skipped step 4 is saved as ``latest``."""
        step, handler = self._keep_last_one(tmp_path, _nan_on({3}))
        assert step == 4
        assert handler.finished is True
        assert (tmp_path / "step_4" / ".metadata").is_file()
        assert (tmp_path / "latest").resolve() == (tmp_path / "step_4").resolve()

    def test_weights_are_checked_only_for_a_shutdown_on_a_skipped_step(self, monkeypatch):
        """The check is a collective pass over every weight, so no other step runs it."""
        import kempnerforge.training.loop as loop

        checks: list[int] = []

        def counting_check(model: nn.Module, device: torch.device) -> bool:
            checks.append(1)
            return True

        monkeypatch.setattr(loop, "params_are_finite", counting_check)

        def run(shutdown_after: int | None) -> tuple[list[int], int]:
            checks.clear()
            ckpt = FakeCheckpointManager()
            session = make_session(
                make_config(max_steps=4),
                checkpointer=ckpt,
                step_fn=_nan_on({1}),
                shutdown_handler=FakeShutdownHandler(after_steps=shutdown_after),
            )
            run_training_loop(session)
            return ckpt.saved_steps, len(checks)

        assert run(None) == ([4], 0)
        assert run(3) == ([3], 0)
        assert run(2) == ([2], 1)


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
        assert torch.equal(before, model.out.weight)

    def test_teardown_is_cooperative_only_on_the_clean_path(self, monkeypatch):
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

            def tracking_entries(self) -> list:
                return []

            def checkpoint_fields(self, _step) -> dict:
                return {}

            def register_checkpoint(self, _step, _directory, _record) -> None:
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
        assert destroys == []
        assert closes == [1]

        run_training(
            make_config(max_steps=2),
            step_fn=lambda _s, _i: StepResult(loss=1.0, grad_norm=0.5),
        )
        assert destroys == [1]
        assert closes == [1, 1]

    def test_checkpoint_manager_reads_the_trackers_live_runs(self, monkeypatch):
        import kempnerforge.training.entry as entry
        from kempnerforge.training.entry import run_training
        from kempnerforge.training.loop import StepResult

        live = [{"backend": "tensorboard", "log_dir": "logs"}]

        class LiveTracker:
            def __init__(self, *_a, **_k) -> None:
                pass

            def init_backends(self, _config) -> None:
                pass

            def start_step(self) -> None:
                pass

            def end_step(self, **_kw):
                return None

            def tracking_entries(self) -> list:
                return live

            def checkpoint_fields(self, _step) -> dict:
                return {}

            def register_checkpoint(self, _step, _directory, _record) -> None:
                pass

            def close(self) -> None:
                pass

        ckpt = _stub_entry(monkeypatch, TinyTextModel())
        monkeypatch.setattr(entry, "MetricsTracker", LiveTracker)
        assert ckpt.tracking_source is None
        run_training(
            make_config(max_steps=1), step_fn=lambda _s, _i: StepResult(loss=1.0, grad_norm=0.5)
        )
        assert ckpt.tracking_source is not None
        assert ckpt.tracking_source() == live

    def test_checkpoint_manager_registers_checkpoints_through_the_tracker(self, monkeypatch):
        import kempnerforge.training.entry as entry
        from kempnerforge.training.entry import run_training
        from kempnerforge.training.loop import StepResult

        built: list = []

        class RegisteringTracker:
            def __init__(self, *_a, **_k) -> None:
                built.append(self)

            def init_backends(self, _config) -> None:
                pass

            def start_step(self) -> None:
                pass

            def end_step(self, **_kw):
                return None

            def tracking_entries(self) -> list:
                return []

            def checkpoint_fields(self, _step) -> dict:
                return {}

            def register_checkpoint(self, _step, _directory, _record) -> None:
                pass

            def close(self) -> None:
                pass

        ckpt = _stub_entry(monkeypatch, TinyTextModel())
        monkeypatch.setattr(entry, "MetricsTracker", RegisteringTracker)
        assert ckpt.checkpoint_registrar is None
        run_training(
            make_config(max_steps=1), step_fn=lambda _s, _i: StepResult(loss=1.0, grad_norm=0.5)
        )
        assert len(built) == 1 and ckpt.checkpoint_registrar is built[0]

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

    def test_checkpoint_manager_records_the_weight_decay_scope(self, tmp_path):
        from kempnerforge.training.entry import build_checkpoint_manager

        config = make_config()
        config.checkpoint.dir = str(tmp_path / "ckpt")
        config.optimizer.weight_decay_scope = "all"
        model = TinyTextModel()
        build_checkpoint_manager(
            config, _runtime(), model, torch.optim.SGD(model.parameters(), lr=0.1), None
        ).save(step=0)
        meta = json.loads((tmp_path / "ckpt" / "step_0" / "metadata.json").read_text())
        assert meta["weight_decay_scope"] == "all"

    def test_restore_checkpoint_starts_from_scratch(self, tmp_path):
        from kempnerforge.training.entry import build_checkpoint_manager, restore_checkpoint

        config = make_config()
        config.checkpoint.dir = str(tmp_path / "empty")
        model = TinyTextModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        mgr = build_checkpoint_manager(config, _runtime(), model, optimizer, None)
        assert restore_checkpoint(config, model, None, mgr) == (0, 0)


class TestWeightDecayScopeWarmStart:
    """Starting a run under either ``weight_decay_scope`` from a checkpoint in ``load_path``."""

    @staticmethod
    def _run(checkpoint_dir, scope, seed):
        from kempnerforge.model.transformer import Transformer
        from kempnerforge.training.entry import build_checkpoint_manager
        from kempnerforge.training.optimizer import build_optimizer

        config = make_config()
        config.checkpoint.dir = str(checkpoint_dir)
        config.optimizer.weight_decay_scope = scope
        config.optimizer.fused = False
        torch.manual_seed(seed)
        model = Transformer(config.model)
        optimizer = build_optimizer(model, config.optimizer)
        manager = build_checkpoint_manager(config, _runtime(), model, optimizer, None)
        return config, model, optimizer, manager

    @pytest.mark.parametrize("scope", ["exclude_norms_and_biases", "all"])
    def test_a_converted_checkpoint_warm_starts_under_either_scope(
        self, tmp_path, monkeypatch, scope
    ):
        """hf-to-dcp writes only the model, no ``metadata.json``; it loads, the optimizer fresh."""
        from kempnerforge.model.transformer import Transformer
        from kempnerforge.training.entry import restore_checkpoint

        monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
        from convert_checkpoint import _kf_to_hf_key, hf_to_dcp

        model_config = make_config().model
        torch.manual_seed(0)
        source = Transformer(model_config)
        hf_dir = tmp_path / "hf"
        hf_dir.mkdir()
        hf_state = {_kf_to_hf_key(k): v for k, v in source.state_dict().items()}
        torch.save(hf_state, hf_dir / "pytorch_model.bin")
        hf_to_dcp(str(hf_dir), str(tmp_path / "pretrained"), model_config)
        assert not (tmp_path / "pretrained" / "metadata.json").exists()

        config, model, optimizer, manager = self._run(tmp_path / "run", scope, seed=1)
        config.checkpoint.load_path = str(tmp_path / "pretrained")
        assert restore_checkpoint(config, model, None, manager) == (0, 0)
        for a, b in zip(source.parameters(), model.parameters(), strict=True):
            assert torch.equal(a, b), "the converted weights were not loaded"
        assert not optimizer.state

    def test_the_refusal_names_a_warm_start_that_works(self, tmp_path):
        """The error's way out needs a ``checkpoint.dir`` with nothing to resume, and says so."""
        from kempnerforge.training.entry import restore_checkpoint

        _, model, optimizer, manager = self._run(tmp_path / "saved", "exclude_norms_and_biases", 0)
        model(torch.randint(0, VOCAB, (2, 4))).sum().backward()
        optimizer.step()
        manager.save(step=1)

        config, model, _, manager = self._run(tmp_path / "saved", "all", seed=1)
        with pytest.raises(ValueError, match="a checkpoint.dir that holds no checkpoint"):
            restore_checkpoint(config, model, None, manager)

        config.checkpoint.load_path = str(tmp_path / "saved" / "step_1")
        config.checkpoint.exclude_from_loading = ["optimizer"]
        with pytest.raises(ValueError, match="but this run uses 'all'"):
            restore_checkpoint(config, model, None, manager)

        config, model, optimizer, manager = self._run(tmp_path / "new", "all", seed=1)
        config.checkpoint.load_path = str(tmp_path / "saved" / "step_1")
        config.checkpoint.exclude_from_loading = ["optimizer"]
        assert restore_checkpoint(config, model, None, manager) == (1, 0)
        assert not optimizer.state


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


class TestTokenDtypeReachesTheDatasets:
    """``data.token_dtype``, a source's own ``token_dtype`` and ``eval.token_dtype``
    decide how the raw ``.bin`` shards behind each dataset are read."""

    SAMPLE = 17

    def _shard(self, directory, dtype: str, start: int, *, metadata: bool = False):
        """Write four samples of tokens as a raw .bin shard; return them sample by sample."""
        import numpy as np

        directory.mkdir()
        tokens = np.arange(start, start + 4 * self.SAMPLE, dtype=dtype)
        tokens.tofile(directory / "shard_000.bin")
        if metadata:
            (directory / "metadata.yaml").write_text(f"tokenized_info:\n  dtype: {dtype}\n")
        return [tokens[i : i + self.SAMPLE].tolist() for i in range(0, tokens.size, self.SAMPLE)]

    @staticmethod
    def _tokens(sample) -> list[int]:
        return torch.cat([sample["input_ids"], sample["labels"][-1:]]).tolist()

    def test_data_token_dtype_reads_the_single_dataset(self, tmp_path):
        expected = self._shard(tmp_path / "web", "uint16", 0)
        config = make_config(seq_len=self.SAMPLE - 1)
        config.data = DataConfig(
            dataset_path=str(tmp_path / "web"),
            file_pattern="*.bin",
            token_dtype="uint16",
            num_workers=0,
        )
        data = build_data_pipeline(config, _runtime())
        assert data.dataloader is not None
        dataset = data.dataloader.dataset
        assert [self._tokens(dataset[i]) for i in range(len(dataset))] == expected

    def test_each_mixture_source_resolves_its_own_dtype(self, tmp_path):
        """One source records its dtype, one sets it, one inherits data.token_dtype."""
        from kempnerforge.config.data import DatasetSource

        recorded = self._shard(tmp_path / "web", "uint16", 0, metadata=True)
        own = self._shard(tmp_path / "code", "uint32", 70_000)
        inherited = self._shard(tmp_path / "books", "uint16", 500)
        config = make_config(seq_len=self.SAMPLE - 1)
        config.data = DataConfig(
            datasets=[
                DatasetSource(name="web", path=str(tmp_path / "web")),
                DatasetSource(name="code", path=str(tmp_path / "code"), token_dtype="uint32"),
                DatasetSource(name="books", path=str(tmp_path / "books")),
            ],
            file_pattern="*.bin",
            token_dtype="uint16",
            num_workers=0,
        )
        data = build_data_pipeline(config, _runtime())
        mixture = data.mixture_dataset
        assert mixture is not None
        bounds = mixture.cumulative_sizes
        got = {
            name: [self._tokens(mixture[i]) for i in range(bounds[k], bounds[k + 1])]
            for k, name in enumerate(mixture.dataset_names)
        }
        assert got == {"web": recorded, "code": own, "books": inherited}

    def test_data_token_dtype_must_agree_with_a_source_metadata(self, tmp_path):
        from kempnerforge.config.data import DatasetSource

        self._shard(tmp_path / "web", "uint16", 0, metadata=True)
        config = make_config(seq_len=self.SAMPLE - 1)
        config.data = DataConfig(
            datasets=[DatasetSource(name="web", path=str(tmp_path / "web"))],
            file_pattern="*.bin",
            token_dtype="uint32",
            num_workers=0,
        )
        with pytest.raises(ValueError, match="records tokenized_info.dtype 'uint16'"):
            build_data_pipeline(config, _runtime())

    def test_eval_token_dtype_reads_the_eval_dataset(self, tmp_path):
        from kempnerforge.config.eval import EvalConfig
        from kempnerforge.training.data_pipeline import build_eval_dataloader

        expected = self._shard(tmp_path / "heldout", "uint16", 300)
        config = make_config(seq_len=self.SAMPLE - 1)
        config.eval = EvalConfig(
            enabled=True,
            dataset_path=str(tmp_path / "heldout"),
            file_pattern="*.bin",
            token_dtype="uint16",
        )
        loader = build_eval_dataloader(config, _runtime())
        assert loader is not None
        dataset = loader.dataset
        assert [self._tokens(dataset[i]) for i in range(len(dataset))] == expected


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

    def test_pipeline_parallel_drops_a_partial_last_batch(self, mmap_data_dir):
        """Pipeline stages are built for one microbatch shape, so a short last batch
        cannot go through them; without PP the last batch is kept."""
        from kempnerforge.config.distributed import DistributedConfig
        from kempnerforge.config.eval import EvalConfig
        from kempnerforge.training.data_pipeline import build_eval_dataloader

        for pp, drops in ((1, False), (2, True)):
            config = dataclasses.replace(
                make_config(batch_size=3),
                eval=EvalConfig(enabled=True, dataset_path=mmap_data_dir),
                distributed=DistributedConfig(pp=pp, dp_shard=1),
            )
            loader = build_eval_dataloader(config, _runtime())
            assert loader is not None
            assert loader.drop_last is drops
            sizes = {len(batch["input_ids"]) for batch in loader}
            assert sizes == ({3} if drops else {3, len(loader.dataset) % 3})

    def test_pipeline_parallel_rejects_an_eval_set_without_a_full_batch(self, mmap_data_dir):
        """Under PP a rank with less than one full batch would have nothing to evaluate, so
        the builder fails before training starts rather than at the first eval; without PP
        the partial batch is kept."""
        from kempnerforge.config.distributed import DistributedConfig
        from kempnerforge.config.eval import EvalConfig
        from kempnerforge.training.data_pipeline import build_eval_dataloader

        def config_for(pp: int, batch_size: int) -> JobConfig:
            return dataclasses.replace(
                make_config(batch_size=batch_size),
                eval=EvalConfig(enabled=True, dataset_path=mmap_data_dir),
                distributed=DistributedConfig(pp=pp, dp_shard=1),
            )

        n_samples = len(build_eval_dataloader(config_for(1, 1), _runtime()).dataset)
        batch_size = n_samples + 1
        loader = build_eval_dataloader(config_for(1, batch_size), _runtime())
        assert [len(batch["input_ids"]) for batch in loader] == [n_samples]
        match = rf"{n_samples} samples, fewer than train\.batch_size \({batch_size}\)"
        with pytest.raises(ValueError, match=match):
            build_eval_dataloader(config_for(2, batch_size), _runtime())


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

    def test_packed_counts_do_not_count_tensor_parallel_replicas(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import kempnerforge.training.loop as loop

        session = make_session(
            make_config(),
            data=DataPipeline(dp_size=2),
            runtime=dataclasses.replace(_runtime(), world_size=8),
        )
        reductions: list[Any] = []

        def reduce_counts(count: torch.Tensor, op: Any) -> None:
            assert count.item() == 7
            reductions.append(op)
            count.fill_(4 * (7 + 11))

        monkeypatch.setattr(loop.dist, "is_initialized", lambda: True)
        monkeypatch.setattr(loop.dist, "all_reduce", reduce_counts)
        result = loop.StepResult(loss=1.0, grad_norm=0.5, tokens=7)
        assert loop._tokens_in_step(session, result) == 18
        assert reductions == [loop.dist.ReduceOp.SUM]

    def test_packed_counts_reach_tokens_seen_without_another_accumulation_factor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import kempnerforge.training.loop as loop

        monkeypatch.setattr(loop.dist, "is_initialized", lambda: False)
        counts = iter((17, 23))

        def step_fn(_session: TrainingSession, _step: int) -> loop.StepResult:
            return loop.StepResult(loss=1.0, grad_norm=0.5, tokens=next(counts))

        session = make_session(
            make_config(max_steps=2, grad_accum_steps=3),
            data=DataPipeline(dp_size=4),
            step_fn=step_fn,
        )
        step, tokens_seen = run_training_loop(session)
        assert (step, tokens_seen) == (2, (17 + 23) * 4)
        assert session.checkpointer.saves[-1][1] == tokens_seen

    def test_default_dp_size_is_one_not_zero(self):
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

    monkeypatch.setattr(pp, "build_stage_module", lambda cfg, r, s, **k: stage)
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
        assert seen["param_dtype"] == config.train.param_dtype
        assert seen["compile_model"] is config.train.compile_model
        assert seen["fp8"] is config.train.is_fp8
        assert seen["frames_per_clip"] == 1
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

    def test_non_pp_keeps_one_frame_per_clip_under_an_image_cap(self, monkeypatch):
        """The cap sizes rows, not the clip the prefix route checks, so eval takes one image."""
        import kempnerforge.training.entry as entry

        seen: dict = {}
        monkeypatch.setattr(
            entry, "build_parallel_model", lambda *a, **k: seen.update(k) or "MODEL"
        )
        config = _vlm_config()
        config.data = DataConfig(hf_max_images=3, num_workers=0)
        entry.build_model(config, _runtime(), lambda a, b: a)
        assert seen["frames_per_clip"] == 1

    def test_pp_builds_a_bundle_from_the_mesh(self, monkeypatch):
        stage, calls = FakeStageModule(), {}
        entry = _stub_pp(monkeypatch, stage, calls)
        config = make_config()
        config.distributed.pp = 2

        model, pipeline = entry.build_model(config, _pp_runtime(), "LOSS")

        assert pipeline is not None
        assert (pipeline.rank, pipeline.size) == (1, 2)
        assert pipeline.schedule == "SCHED"
        assert calls["schedule_kwargs"]["n_microbatches"] == config.train.grad_accum_steps
        assert calls["schedule_kwargs"]["loss_fn"] == "LOSS"
        assert model is stage

    def test_pp_without_tp_skips_meta_init(self, monkeypatch):
        stage, calls = FakeStageModule(), {}
        entry = _stub_pp(monkeypatch, stage, calls)
        config = make_config()
        config.distributed.pp = 2
        entry.build_model(config, _pp_runtime(FakeMesh(("pp", "dp_shard"))), "LOSS")
        assert "apply_tensor_parallel" not in calls
        assert stage.inited is False

    def test_pp_with_tp_takes_the_meta_device_path(self, monkeypatch):
        stage, calls = FakeStageModule(), {}
        entry = _stub_pp(monkeypatch, stage, calls)
        config = make_config()
        config.distributed.pp = 2
        entry.build_model(config, _pp_runtime(FakeMesh(("pp", "tp"))), "LOSS")
        assert calls.get("apply_tensor_parallel") is True
        assert stage.inited is True
        assert stage.emptied_to is not None

    @pytest.mark.parametrize(
        ("fp32_master_weights", "dtype"), [(False, torch.bfloat16), (True, torch.float32)]
    )
    def test_parameters_are_stored_in_the_master_dtype(
        self, fp32_master_weights: bool, dtype: torch.dtype
    ) -> None:
        import kempnerforge.training.entry as entry

        config = make_config(fp32_master_weights=fp32_master_weights)
        model, _ = entry.build_model(config, _runtime(), lambda a, b: a)
        assert {p.dtype for p in model.parameters()} == {dtype}

    @pytest.mark.parametrize("dims", [("pp", "dp_shard"), ("pp", "tp")])
    @pytest.mark.parametrize(
        ("fp32_master_weights", "dtype"), [(False, torch.bfloat16), (True, torch.float32)]
    )
    def test_pp_stage_is_stored_in_the_master_dtype(
        self, monkeypatch, dims: tuple[str, str], fp32_master_weights: bool, dtype: torch.dtype
    ) -> None:
        stage, calls = FakeStageModule(), {}
        entry = _stub_pp(monkeypatch, stage, calls)
        config = make_config(fp32_master_weights=fp32_master_weights)
        config.distributed.pp = 2
        entry.build_model(config, _pp_runtime(FakeMesh(dims)), "LOSS")
        assert {p.dtype for p in stage.parameters()} == {dtype}

    def test_pp_without_a_mesh_raises_rather_than_asserting(self, monkeypatch):
        stage, calls = FakeStageModule(), {}
        entry = _stub_pp(monkeypatch, stage, calls)
        config = make_config()
        config.distributed.pp = 2
        with pytest.raises(RuntimeError, match="requires a device mesh"):
            entry.build_model(config, _runtime(), "LOSS")


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
    @pytest.mark.parametrize(("pp", "expected"), [(1, None), (2, 2), (4, 4)])
    def test_resolution_gets_the_pipeline_degree(self, monkeypatch, pp, expected):
        """The step_N scan needs the stage count to see a stage that never saved."""
        import kempnerforge.training.entry as entry

        seen: list = []
        monkeypatch.setattr(
            entry, "resolve_resume_path", lambda _d, num_stages=None: seen.append(num_stages)
        )
        config = make_config()
        config.distributed.pp = pp
        mgr = FakeResumeManager(step=0, tokens=0)
        assert entry.restore_checkpoint(config, TinyTextModel(), None, mgr) == (0, 0)
        assert seen == [expected]

    def test_no_checkpoint_and_no_load_path_starts_from_scratch(self, monkeypatch):
        import kempnerforge.training.entry as entry

        monkeypatch.setattr(entry, "resolve_resume_path", lambda _d, _n=None: None)
        mgr = FakeResumeManager()
        assert entry.restore_checkpoint(make_config(), TinyTextModel(), None, mgr) == (0, 0)
        assert mgr.load_kwargs == {}

    def test_resume_returns_the_managers_step_and_tokens(self, monkeypatch, tmp_path):
        import kempnerforge.training.entry as entry

        monkeypatch.setattr(entry, "resolve_resume_path", lambda _d, _n=None: tmp_path / "step_7")
        mgr = FakeResumeManager(step=7, tokens=999)
        assert entry.restore_checkpoint(make_config(), TinyTextModel(), "SCHED", mgr) == (7, 999)
        assert mgr.load_kwargs["scheduler"] == "SCHED"
        assert mgr.load_kwargs["vlm_freeze_expected"] is None

    def test_load_path_alone_triggers_a_resume(self, monkeypatch):
        import kempnerforge.training.entry as entry

        monkeypatch.setattr(entry, "resolve_resume_path", lambda _d, _n=None: None)
        config = make_config()
        config.checkpoint.load_path = "/somewhere/step_3"
        mgr = FakeResumeManager(step=3, tokens=30)
        assert entry.restore_checkpoint(config, TinyTextModel(), None, mgr) == (3, 30)
        assert mgr.load_kwargs["path"] is None

    def test_metrics_run_ids_are_restored_onto_the_config(self, monkeypatch, tmp_path):
        import kempnerforge.training.entry as entry

        monkeypatch.setattr(entry, "resolve_resume_path", lambda _d, _n=None: tmp_path / "s")
        config = make_config()
        mgr = FakeResumeManager(extra={"wandb_run_id": "W1", "mlflow_run_id": "M1"})
        entry.restore_checkpoint(config, TinyTextModel(), None, mgr)
        assert config.metrics.wandb_run_id == "W1"
        assert config.metrics.mlflow_run_id == "M1"

    def test_absent_run_ids_do_not_clobber_existing_ones(self, monkeypatch, tmp_path):
        import kempnerforge.training.entry as entry

        monkeypatch.setattr(entry, "resolve_resume_path", lambda _d, _n=None: tmp_path / "s")
        config = make_config()
        config.metrics.wandb_run_id = "KEEP"
        entry.restore_checkpoint(config, TinyTextModel(), None, FakeResumeManager(extra={}))
        assert config.metrics.wandb_run_id == "KEEP"

    def test_vlm_resume_computes_expected_freeze_at_the_saved_step(self, monkeypatch, tmp_path):
        import kempnerforge.training.entry as entry
        from kempnerforge.training.freeze import freeze_meta_at_step

        monkeypatch.setattr(entry, "resolve_resume_path", lambda _d, _n=None: tmp_path / "s")
        config = _vlm_config()
        mgr = FakeResumeManager(saved_step=11)
        entry.restore_checkpoint(config, TinyTextModel(), None, mgr)
        assert mgr.peeked == [str(tmp_path / "s")]
        assert mgr.load_kwargs["vlm_freeze_expected"] == freeze_meta_at_step(11, config.vlm)

    def test_freeze_schedule_is_reapplied_at_the_resumed_step(self, monkeypatch, tmp_path):
        import kempnerforge.training.entry as entry
        from kempnerforge.config.vlm import FreezeSpec, FreezeStage

        monkeypatch.setattr(entry, "resolve_resume_path", lambda _d, _n=None: tmp_path / "s")
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
        assert applied and applied[0][0].module == "vision_encoder"

    @pytest.mark.parametrize(
        ("resume_found", "load_path", "exclude", "want_exclude"),
        [
            (False, "/somewhere/converted", ["optimizer"], ["optimizer"]),
            (False, "/somewhere/converted", ["model"], ["model"]),
            (False, "/somewhere/converted", ["model", "optimizer"], ["model", "optimizer"]),
            (False, "/somewhere/converted", [], None),
            (True, None, ["optimizer"], None),
            (True, None, [], None),
            (True, "/somewhere/converted", ["optimizer"], None),
            (True, "/somewhere/converted", ["model", "optimizer"], None),
            (True, "/somewhere/converted", [], None),
        ],
    )
    def test_exclude_from_loading_truth_table(
        self, monkeypatch, tmp_path, resume_found, load_path, exclude, want_exclude
    ):
        import kempnerforge.training.entry as entry

        resume = tmp_path / "step_7" if resume_found else None
        monkeypatch.setattr(entry, "resolve_resume_path", lambda _d, _n=None: resume)
        config = make_config()
        config.checkpoint.load_path = load_path
        config.checkpoint.exclude_from_loading = exclude
        mgr = FakeResumeManager(step=7 if resume_found else 0, tokens=0)
        entry.restore_checkpoint(config, TinyTextModel(), None, mgr)
        assert mgr.load_kwargs["path"] == (str(resume) if resume_found else None)
        assert mgr.load_kwargs["exclude_keys"] == want_exclude

    def test_resume_with_stale_load_path_restores_full_state(self, monkeypatch, tmp_path):
        """A requeued run keeps its warm-start config; the resume must still
        load the optimizer, or momentum silently resets mid-run."""
        import kempnerforge.training.entry as entry

        monkeypatch.setattr(entry, "resolve_resume_path", lambda _d, _n=None: tmp_path / "step_7")
        config = make_config()
        config.checkpoint.load_path = "/somewhere/converted"
        config.checkpoint.exclude_from_loading = ["optimizer"]
        mgr = FakeResumeManager(step=7, tokens=999)
        assert entry.restore_checkpoint(config, TinyTextModel(), None, mgr) == (7, 999)
        assert mgr.load_kwargs["path"] == str(tmp_path / "step_7")
        assert mgr.load_kwargs["exclude_keys"] is None

    def test_exclude_alone_does_not_trigger_a_load(self, monkeypatch):
        import kempnerforge.training.entry as entry

        monkeypatch.setattr(entry, "resolve_resume_path", lambda _d, _n=None: None)
        config = make_config()
        config.checkpoint.exclude_from_loading = ["optimizer"]
        mgr = FakeResumeManager()
        assert entry.restore_checkpoint(config, TinyTextModel(), None, mgr) == (0, 0)
        assert mgr.load_kwargs == {}

    def test_rank_zero_resolves_with_the_pipeline_degree(self, monkeypatch, tmp_path):
        """The distributed resolution also gets the stage count."""
        import kempnerforge.training.entry as entry

        seen: list = []
        chosen = tmp_path / "step_7"

        def resolve(_d, num_stages=None):
            seen.append(num_stages)
            return chosen

        monkeypatch.setattr(entry, "resolve_resume_path", resolve)
        monkeypatch.setattr(entry.dist, "is_initialized", lambda: True)
        monkeypatch.setattr(entry.dist, "get_rank", lambda: 0)
        monkeypatch.setattr(entry.dist, "broadcast_object_list", lambda lst, src=0: None)
        config = make_config()
        config.distributed.pp = 2
        entry.restore_checkpoint(config, TinyTextModel(), None, FakeResumeManager())
        assert seen == [2]

    @pytest.mark.parametrize("rank", [0, 1])
    def test_only_rank_zero_resolves_the_resume_path(self, monkeypatch, tmp_path, rank):
        """Every rank must resume from the same checkpoint.

        Resolution stats ``.metadata`` to test durability, and on shared
        storage those stats can disagree briefly across ranks. A rank that
        disagrees resumes from a different step, or hangs in the collective
        load when it skips one the other ranks take. Only rank 0 looks, and
        its answer is broadcast.
        """
        import kempnerforge.training.entry as entry

        looked: list = []
        chosen = tmp_path / "step_7"
        monkeypatch.setattr(
            entry, "resolve_resume_path", lambda d, _n=None: looked.append(d) or chosen
        )
        monkeypatch.setattr(entry.dist, "is_initialized", lambda: True)
        monkeypatch.setattr(entry.dist, "get_rank", lambda: rank)
        monkeypatch.setattr(
            entry.dist,
            "broadcast_object_list",
            lambda lst, src=0: lst.__setitem__(0, str(chosen)),
        )

        mgr = FakeResumeManager(step=7, tokens=1234)
        assert entry.restore_checkpoint(make_config(), TinyTextModel(), None, mgr) == (7, 1234)
        assert mgr.load_kwargs["path"] == str(chosen)
        assert len(looked) == (1 if rank == 0 else 0), "non-zero ranks must not stat the tree"

    def test_warm_start_exclude_leaves_the_vlm_freeze_probe_alone(self, monkeypatch):
        """Excluding state keys does not skip the freeze-metadata check; a
        weights-only checkpoint has no metadata.json, so the probe yields 0."""
        import kempnerforge.training.entry as entry
        from kempnerforge.training.freeze import freeze_meta_at_step

        monkeypatch.setattr(entry, "resolve_resume_path", lambda _d, _n=None: None)
        config = _vlm_config()
        config.checkpoint.load_path = "/somewhere/converted"
        config.checkpoint.exclude_from_loading = ["model"]
        mgr = FakeResumeManager(step=0, tokens=0, saved_step=None)
        entry.restore_checkpoint(config, TinyTextModel(), None, mgr)
        assert mgr.peeked == [None]
        assert mgr.load_kwargs["vlm_freeze_expected"] == freeze_meta_at_step(0, config.vlm)
        assert mgr.load_kwargs["exclude_keys"] == ["model"]

    def test_mot_warm_start_fires_only_at_step_zero(self, monkeypatch, tmp_path):
        import kempnerforge.training.entry as entry
        from kempnerforge.config.adapter import AdapterConfig
        from kempnerforge.config.vision import VisionEncoderConfig
        from kempnerforge.config.vlm import MoTConfig

        monkeypatch.setattr(entry, "resolve_resume_path", lambda _d, _n=None: tmp_path / "s")
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
        assert copied == [{"w": 1}]

        copied.clear()
        entry.restore_checkpoint(config, TinyTextModel(), None, FakeResumeManager(step=4))
        assert copied == []


class FakeDataset:
    has_text_only_rows = False

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
        assert "sampler" not in seen
        assert (made["rank"], made["world_size"]) == (2, 4)

    def test_missing_tokenizer_is_rejected(self):
        import kempnerforge.training.data_pipeline as dp

        config = make_config()
        config.data = DataConfig(hf_dataset_name="ds", num_workers=0)
        with pytest.raises(ValueError, match="tokenizer_path is required"):
            dp.build_data_pipeline(config, _runtime())

    def test_empty_name_raises_rather_than_asserting(self):
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
        assert seen["seq_len"] == config.train.seq_len + 1


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
        monkeypatch.setattr(
            vlmd,
            "HuggingFaceVLMDataset",
            lambda **k: seen.setdefault("dataset", k) and FakeDataset(),
        )
        monkeypatch.setattr(vlmd, "VLMCollator", lambda **k: seen.setdefault("image_collator", k))
        monkeypatch.setattr(
            vlmd, "PackedVLMCollator", lambda **k: seen.setdefault("packed_collator", k)
        )
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

    def test_image_path_passes_the_text_only_and_image_cap_settings_to_the_dataset(
        self, monkeypatch
    ):
        seen: dict = {}
        dp = _stub_loaders(monkeypatch, seen)
        self._stub_vlm_modules(monkeypatch, seen)
        config = self._vlm_data_config()
        config.data.hf_allow_text_only = True
        config.data.hf_max_images = 2
        dp.build_data_pipeline(config, _runtime(), tokens_per_image=16)
        assert seen["dataset"]["allow_text_only"] is True
        assert seen["dataset"]["max_images"] == 2
        assert seen["packed_collator"]["max_docs_per_row"] == 1 and "image_collator" not in seen

    def test_image_path_passes_the_text_only_flag_to_the_collator(self, monkeypatch):
        seen: dict = {}
        dp = _stub_loaders(monkeypatch, seen)
        self._stub_vlm_modules(monkeypatch, seen)
        config = self._vlm_data_config()
        config.data.hf_allow_text_only = True
        monkeypatch.setattr(FakeDataset, "has_text_only_rows", True)
        dp.build_data_pipeline(config, _runtime())
        assert seen["image_collator"]["allow_text_only"] is True
        assert seen["dataset"]["allow_text_only"] is True
        monkeypatch.setattr(FakeDataset, "has_text_only_rows", False)
        seen.clear()
        dp.build_data_pipeline(config, _runtime())
        assert seen["image_collator"]["allow_text_only"] is False

    def test_an_image_cap_above_one_without_packing_uses_fixed_rows(self, monkeypatch):
        seen: dict = {}
        dp = _stub_loaders(monkeypatch, seen)
        self._stub_vlm_modules(monkeypatch, seen)
        config = self._vlm_data_config()
        config.data.hf_max_images = 3
        config.train.seq_len = 128
        dp.build_data_pipeline(config, _runtime(), tokens_per_image=16)
        assert seen["packed_collator"] == {
            "pad_id": 7,
            "seq_len": config.train.seq_len,
            "tokens_per_image": 16,
            "max_docs_per_row": 1,
            "allow_text_only": False,
        }
        assert "image_collator" not in seen
        with pytest.raises(ValueError, match="data.hf_max_images on a VLM run needs the built"):
            dp.build_data_pipeline(config, _runtime())

    @pytest.mark.parametrize("pack", [False, True])
    def test_laid_out_rows_must_fit_the_image_cap_and_the_text(self, monkeypatch, pack):
        seen: dict = {}
        dp = _stub_loaders(monkeypatch, seen)
        self._stub_vlm_modules(monkeypatch, seen)
        config = self._vlm_data_config()
        config.data.hf_max_images = 3
        config.data.pack_sequences = pack
        too_many = (config.train.seq_len - config.vlm.max_text_len) // 3 + 1
        with pytest.raises(ValueError, match="over train.seq_len"):
            dp.build_data_pipeline(config, _runtime(), tokens_per_image=too_many)
        assert "dataset" not in seen

    def test_packed_image_path_takes_the_model_token_count(self, monkeypatch):
        import kempnerforge.data.vlm_dataset as vlmd

        seen: dict = {}
        dp = _stub_loaders(monkeypatch, seen)
        self._stub_vlm_modules(monkeypatch, seen)
        monkeypatch.setattr(vlmd, "PackedVLMCollator", lambda **k: seen.setdefault("packed", k))
        config = self._vlm_data_config()
        config.data.pack_sequences = True
        data = dp.build_data_pipeline(config, _runtime(), tokens_per_image=12)
        assert data.dataloader == "stateful_OBJ"
        assert seen["packed"] == {
            "pad_id": 7,
            "seq_len": config.train.seq_len,
            "tokens_per_image": 12,
            "max_docs_per_row": 0,
            "allow_text_only": False,
        }
        assert "image_collator" not in seen
        with pytest.raises(ValueError, match="built model's tokens per image"):
            dp.build_data_pipeline(config, _runtime())

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
        assert item["input_ids"].tolist() == [0, 1, 2, 3]
        assert item["labels"].tolist() == [1, 2, 3, 4]


class FakeSchedule:
    def __init__(self, loss=2.0) -> None:
        self.loss = loss
        self.calls: list = []

    def step(self, *args, target=None, losses=None, **kwargs):
        self.calls.append({"args": args, "target": target, "kwargs": kwargs})
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
            recorded["broadcast_numel"] = tensor.numel()

        def fake_all_reduce(tensor, op=None, group=None):
            recorded["all_reduce_group"] = group

        monkeypatch.setattr(loop.dist, "broadcast", fake_broadcast)
        monkeypatch.setattr(loop.dist, "all_reduce", fake_all_reduce)
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
        assert isinstance(result.loss, float)
        assert rec["group"] == "PPGROUP"
        assert rec["group_src"] == 1

    def test_every_stage_sums_the_grad_norm_and_only_the_loss_is_broadcast(self, monkeypatch):
        """Each stage clips with the whole model's norm, so the broadcast carries the loss alone."""
        sched = FakeSchedule()
        session, rec = self._session(monkeypatch, rank=0, size=2, sched=sched)
        session.step_fn(session, 0)
        assert rec["all_reduce_group"] == "PPGROUP"
        assert rec["broadcast_numel"] == 1

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

        session = make_session(make_config())
        with pytest.raises(RuntimeError, match="requires TrainingSession.pipeline"):
            pipeline_step(session, 0)


class TestPipelineStepPacked:
    """``doc_ids`` reaches the schedule as a kwarg, on every rank.

    Two reasons it cannot be positional. A schedule hands positional args to
    stage 0 only, so later stages would never see them; and a positional arg
    becomes an inter-stage activation, which ``PipelineStage`` marks as
    requiring grad without checking dtype -- so an integer tensor fails as soon
    as a backward pass exists. Shape-level coverage misses both: the pipeline
    still runs, it just attends across document boundaries.
    """

    def _packed_session(self, monkeypatch, rank, size, sched, *, packed=True):
        import kempnerforge.training.loop as loop
        from kempnerforge.training.runtime import PipelineBundle

        monkeypatch.setattr(loop, "pp_group", lambda _r: "PPGROUP")
        monkeypatch.setattr(loop.dist, "broadcast", lambda *a, **k: None)
        monkeypatch.setattr(loop.dist, "all_reduce", lambda *a, **k: None)

        batch = {
            "input_ids": torch.zeros(2, 8, dtype=torch.long),
            "labels": torch.zeros(2, 8, dtype=torch.long),
        }
        if packed:
            batch["doc_ids"] = torch.tensor([[0] * 4 + [1] * 4] * 2)

        config = make_config()
        config.distributed.pp = size
        return make_session(
            config,
            data=DataPipeline(dataloader=[batch] * 4),
            pipeline=PipelineBundle(rank=rank, size=size, schedule=sched),
        )

    @pytest.mark.parametrize("rank,size,label", [(0, 2, "first"), (1, 2, "last"), (1, 3, "middle")])
    def test_doc_ids_arrive_as_a_kwarg(self, monkeypatch, rank, size, label):
        sched = FakeSchedule()
        session = self._packed_session(monkeypatch, rank, size, sched)
        session.step_fn(session, 0)

        call = sched.calls[0]
        assert "doc_ids" in call["kwargs"], f"{label} stage received no doc_ids"
        assert not any(
            isinstance(a, torch.Tensor)
            and a.dtype == torch.long
            and a.dim() == 2
            and torch.equal(a, call["kwargs"]["doc_ids"])
            for a in call["args"][1:]
        ), f"{label} stage passed doc_ids positionally"

    def test_unpacked_batch_sends_no_doc_ids(self, monkeypatch):
        """An unpacked run must not grow a kwarg it never had."""
        sched = FakeSchedule()
        session = self._packed_session(monkeypatch, 0, 2, sched, packed=False)
        session.step_fn(session, 0)
        assert sched.calls[0]["kwargs"] == {}


class TinyVlmModel(nn.Module):
    """(pixel_values, input_ids, labels) -> (logits, labels) as VLMWrapper does."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(DIM, VOCAB)
        self.embed = nn.Embedding(VOCAB, DIM)
        self.seen_frame_mask: list = []
        self.seen_packed: list = []

    def forward(self, pixel_values, input_ids, labels, frame_mask=None, **packed):
        self.seen_frame_mask.append(frame_mask)
        self.seen_packed.append(packed)
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
        assert result.text_tokens == 8

    def test_ignore_index_positions_are_excluded(self):
        batch = self._batch()
        batch["labels"][:, :2] = -100
        session, _ = self._session([batch])
        result = session.step_fn(session, 0)
        assert result.text_tokens == 4

    def test_frame_mask_is_threaded_when_present(self):
        session, model = self._session([self._batch(with_mask=True)])
        session.step_fn(session, 0)
        assert model.seen_frame_mask[0] is not None

    def test_packed_keys_are_threaded_when_present(self):
        batch = self._batch()
        batch["doc_ids"] = torch.zeros(2, 4, dtype=torch.long)
        batch["image_rows"] = torch.tensor([0, 1])
        batch["image_starts"] = torch.tensor([0, 0])
        batch["position_ids"] = torch.zeros(2, 4, dtype=torch.long)
        batch["image_tokens"] = torch.tensor(2)
        session, model = self._session([batch])
        session.step_fn(session, 0)
        packed = model.seen_packed[0]
        assert set(packed) == {
            "doc_ids",
            "position_ids",
            "image_rows",
            "image_starts",
            "image_tokens",
        }
        assert packed["image_tokens"] == 2
        assert packed["doc_ids"].shape == (2, 4)

    def test_reports_tokens_only_for_packed_batches(self):
        session, _ = self._session([self._batch()])
        assert session.step_fn(session, 0).tokens is None
        batch = self._batch()
        batch["doc_ids"] = torch.zeros(2, 4, dtype=torch.long)
        session, _ = self._session([batch])
        assert session.step_fn(session, 0).tokens == 8

    def test_packed_rows_are_marked_dynamic_for_compile(self, monkeypatch):
        import torch._dynamo

        marked: list = []
        monkeypatch.setattr(torch._dynamo, "maybe_mark_dynamic", lambda t, d: marked.append(d))
        batch = self._batch()
        batch["doc_ids"] = torch.zeros(2, 4, dtype=torch.long)
        batch["image_rows"] = torch.tensor([0, 1])
        batch["image_starts"] = torch.tensor([0, 0])
        batch["image_tokens"] = torch.tensor(2)
        session, _ = self._session([batch])
        session.step_fn(session, 0)
        assert marked == []
        session.config.train.compile_model = True
        session.step_fn(session, 0)
        assert marked == [0] * 6

    def test_unpacked_batches_pass_no_packed_keys(self):
        session, model = self._session([self._batch()])
        session.step_fn(session, 0)
        assert model.seen_packed[0] == {}

    def test_frame_mask_is_none_for_image_batches(self):
        session, model = self._session([self._batch()])
        session.step_fn(session, 0)
        assert model.seen_frame_mask[0] is None

    def test_synthetic_fallback_is_refused(self):
        session, _ = self._session(None)
        with pytest.raises(RuntimeError, match="requires a real dataloader"):
            session.step_fn(session, 0)


@pytest.fixture
def restore_matmul_precision() -> Iterator[None]:
    """Restore torch's fp32 matmul precision, both the legacy value and the per-backend ones."""
    backends = (torch.backends.cuda.matmul, torch.backends.mkldnn.matmul)
    legacy = torch.get_float32_matmul_precision()
    saved = [backend.fp32_precision for backend in backends]
    yield
    torch.set_float32_matmul_precision(legacy)
    for backend, value in zip(backends, saved, strict=True):
        backend.fp32_precision = value


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
        assert validated == [8]

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

    @pytest.mark.usefixtures("restore_matmul_precision")
    @pytest.mark.parametrize(("tf32", "expected"), [(True, "high"), (False, "highest")])
    def test_tf32_sets_the_matmul_precision(
        self, monkeypatch: pytest.MonkeyPatch, tf32: bool, expected: str
    ) -> None:
        import kempnerforge.training.runtime as rt

        monkeypatch.setattr(rt, "get_world_info", lambda: (0, 0, 1))
        monkeypatch.setattr(rt, "init_distributed", lambda cfg, seed: None)
        monkeypatch.setattr(rt, "log_job_info", lambda: None)
        torch.set_float32_matmul_precision("highest")
        rt.setup_distributed(make_config(tf32=tf32))
        assert torch.get_float32_matmul_precision() == expected

    @pytest.mark.usefixtures("restore_matmul_precision")
    def test_tf32_off_leaves_an_existing_setting_alone(self) -> None:
        from kempnerforge.training.runtime import apply_matmul_precision

        torch.set_float32_matmul_precision("medium")
        apply_matmul_precision(TrainConfig())
        assert torch.get_float32_matmul_precision() == "medium"


class TestPpGroup:
    def test_derives_the_group_from_the_mesh(self):
        from kempnerforge.training.runtime import pp_group

        mesh = FakeMesh(("pp", "dp_shard"))
        assert pp_group(dataclasses.replace(_runtime(), device_mesh=mesh)) == "group:pp"

    def test_without_a_mesh_raises_rather_than_asserting(self):
        from kempnerforge.training.runtime import pp_group

        with pytest.raises(RuntimeError, match="requires a device mesh"):
            pp_group(_runtime())


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
                "dataset_idx": torch.tensor([0, 0, 0, 0]),
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
        assert step == 1
        assert ckpt.saves == []

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
        assert evals == [2, 4]

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
        assert order == ["flush", "freeze"]

    def test_no_stage_at_this_step_does_not_flush(self, monkeypatch):
        order: list = []
        run_training_loop(self._vlm_session(monkeypatch, start_step=99, order=order))
        assert order == []

    def test_stage_due_on_the_last_finite_step_fires_on_it(self, monkeypatch):
        """Step 2 is the last step, so the stage applies only if it runs after the step counts."""
        order: list = []
        session = self._vlm_session(monkeypatch, start_step=2, order=order)
        session.config.train.max_steps = 2
        run_training_loop(session)
        assert order == ["flush", "freeze"]

    def test_stage_reached_by_a_nan_skipped_step_fires_on_it(self, monkeypatch):
        """Step 2 is reached by a NaN step and is the last one, so only that step can apply it."""
        order: list = []
        session = self._vlm_session(monkeypatch, start_step=2, order=order)
        session.config.train.max_steps = 2
        session.step_fn = _nan_on({1}, vlm_step)
        run_training_loop(session)
        assert order == ["flush", "freeze"]

    def test_emergency_checkpoint_on_that_step_is_written_after_the_flip(self, monkeypatch):
        """Its saved freeze metadata then describes the model it was written from."""
        from kempnerforge.training.freeze import freeze_meta_at_step

        order: list = []
        session = self._vlm_session(monkeypatch, start_step=2, order=order)
        session.step_fn = _nan_on({1}, vlm_step)
        session.shutdown_handler = FakeShutdownHandler(after_steps=2)
        saved: list = []

        def save(step: int, **kwargs: Any) -> None:
            order.append("save")
            saved.append((step, kwargs["extra"]))

        monkeypatch.setattr(session.checkpointer, "save", save)
        run_training_loop(session)
        assert order == ["flush", "freeze", "save"]
        assert saved == [(2, {"vlm_freeze": freeze_meta_at_step(2, session.config.vlm)})]


class TestBatchStreamGuard:
    def test_next_batch_without_a_loader_names_the_contract(self):
        with pytest.raises(RuntimeError, match="check has_data first"):
            BatchStream(DataPipeline()).next_batch()


class TestPipelineSyntheticBatches:
    def test_pp_falls_back_to_random_tokens_without_a_dataloader(self, monkeypatch):
        import kempnerforge.training.loop as loop
        from kempnerforge.training.runtime import PipelineBundle

        monkeypatch.setattr(loop, "pp_group", lambda _r: "G")
        monkeypatch.setattr(loop.dist, "broadcast", lambda *a, **k: None)
        monkeypatch.setattr(loop.dist, "all_reduce", lambda *a, **k: None)
        sched = FakeSchedule()
        config = make_config()
        config.distributed.pp = 2
        session = make_session(
            config,
            data=DataPipeline(),
            pipeline=PipelineBundle(rank=0, size=2, schedule=sched),
        )
        session.step_fn(session, 0)
        fed = sched.calls[0]["args"][0]
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
        model = TinyVlmMoEModel()

        def _loss(aux_weight, z_weight):
            config = _moe_config(z_weight=z_weight, aux_weight=aux_weight)
            config.train.batch_size, config.train.seq_len = 2, 4
            s = make_session(config, model=model, data=DataPipeline(dataloader=[batch]))
            s.step_fn = vlm_step
            return s.step_fn(s, 0).loss

        base = _loss(0.0, 0.0)
        loss = _loss(2.0, 4.0)
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
