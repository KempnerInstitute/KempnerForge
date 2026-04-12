"""Unit tests for training loop hooks."""

from __future__ import annotations

import torch

from kempnerforge.config.job import JobConfig
from kempnerforge.training.hooks import HookRunner, StepContext, TrainingHook

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ctx(step: int = 1, loss: float = 1.0) -> StepContext:
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return StepContext(
        step=step,
        loss=loss,
        grad_norm=0.5,
        lr=1e-3,
        tokens_seen=1024,
        model=model,
        optimizer=optimizer,
    )


# ---------------------------------------------------------------------------
# HookRunner dispatch
# ---------------------------------------------------------------------------


class TestHookRunner:
    def test_empty_runner_no_error(self):
        runner = HookRunner()
        config = JobConfig()
        runner.on_train_begin(config)
        runner.on_step_end(_make_ctx())
        runner.on_eval_end({"eval/loss": 1.0}, step=1)
        runner.on_checkpoint_save(step=1, path="/tmp/ckpt")
        runner.on_train_end(step=100, tokens_seen=10000)

    def test_dispatch_calls_all_hooks(self):
        call_log: list[str] = []

        class LoggingHook(TrainingHook):
            def __init__(self, name: str) -> None:
                self.name = name

            def on_step_end(self, ctx: StepContext) -> None:
                call_log.append(self.name)

        runner = HookRunner([LoggingHook("a"), LoggingHook("b"), LoggingHook("c")])
        runner.on_step_end(_make_ctx())
        assert call_log == ["a", "b", "c"]

    def test_dispatch_order_preserved(self):
        order: list[int] = []

        class OrderHook(TrainingHook):
            def __init__(self, idx: int) -> None:
                self.idx = idx

            def on_train_begin(self, config: JobConfig) -> None:
                order.append(self.idx)

        runner = HookRunner([OrderHook(3), OrderHook(1), OrderHook(2)])
        runner.on_train_begin(JobConfig())
        assert order == [3, 1, 2]

    def test_none_hooks_defaults_to_empty(self):
        runner = HookRunner(None)
        assert runner.hooks == []


# ---------------------------------------------------------------------------
# StepContext
# ---------------------------------------------------------------------------


class TestStepContext:
    def test_ctx_fields(self):
        ctx = _make_ctx(step=42, loss=2.5)
        assert ctx.step == 42
        assert ctx.loss == 2.5
        assert ctx.grad_norm == 0.5
        assert ctx.lr == 1e-3
        assert ctx.tokens_seen == 1024
        assert isinstance(ctx.model, torch.nn.Module)
        assert isinstance(ctx.optimizer, torch.optim.Optimizer)

    def test_ctx_model_access(self):
        ctx = _make_ctx()
        params = list(ctx.model.parameters())
        assert len(params) > 0


# ---------------------------------------------------------------------------
# TrainingHook base class
# ---------------------------------------------------------------------------


class TestTrainingHook:
    def test_base_methods_are_noop(self):
        hook = TrainingHook()
        config = JobConfig()
        hook.on_train_begin(config)
        hook.on_step_end(_make_ctx())
        hook.on_eval_end({}, step=1)
        hook.on_checkpoint_save(step=1, path="/tmp")
        hook.on_train_end(step=100, tokens_seen=10000)

    def test_partial_override(self):
        call_log: list[str] = []

        class StepOnlyHook(TrainingHook):
            def on_step_end(self, ctx: StepContext) -> None:
                call_log.append("step")

        runner = HookRunner([StepOnlyHook()])
        runner.on_train_begin(JobConfig())
        runner.on_step_end(_make_ctx())
        runner.on_eval_end({}, step=1)
        assert call_log == ["step"]


# ---------------------------------------------------------------------------
# All event types fire
# ---------------------------------------------------------------------------


class TestAllEvents:
    def test_all_events_dispatch(self):
        events: list[str] = []

        class AllEventsHook(TrainingHook):
            def on_train_begin(self, config: JobConfig) -> None:
                events.append("train_begin")

            def on_step_end(self, ctx: StepContext) -> None:
                events.append("step_end")

            def on_eval_end(self, metrics: dict[str, float], step: int) -> None:
                events.append("eval_end")

            def on_checkpoint_save(self, step: int, path: str) -> None:
                events.append("checkpoint_save")

            def on_train_end(self, step: int, tokens_seen: int) -> None:
                events.append("train_end")

        runner = HookRunner([AllEventsHook()])
        runner.on_train_begin(JobConfig())
        runner.on_step_end(_make_ctx())
        runner.on_eval_end({"eval/loss": 1.0}, step=1)
        runner.on_checkpoint_save(step=1, path="/tmp/ckpt")
        runner.on_train_end(step=100, tokens_seen=10000)

        assert events == ["train_begin", "step_end", "eval_end", "checkpoint_save", "train_end"]
