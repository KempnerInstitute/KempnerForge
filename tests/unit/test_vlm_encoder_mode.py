"""A vision encoder whose parameters are all frozen runs in eval mode while training (#581)."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from kempnerforge.config.adapter import AdapterConfig
from kempnerforge.config.checkpoint import CheckpointConfig
from kempnerforge.config.job import JobConfig
from kempnerforge.config.model import ModelConfig
from kempnerforge.config.optimizer import OptimizerConfig
from kempnerforge.config.registry import registry
from kempnerforge.config.training import TrainConfig
from kempnerforge.config.vision import VisionEncoderConfig
from kempnerforge.config.vlm import FreezeSpec, FreezeStage, VLMConfig
from kempnerforge.metrics.tracker import MetricsTracker
from kempnerforge.model.vision import VisionEncoder
from kempnerforge.resilience.health import NaNDetector
from kempnerforge.resilience.signal_handler import ShutdownHandler
from kempnerforge.training.data_pipeline import DataPipeline, build_phase_state
from kempnerforge.training.entry import build_checkpoint_manager, build_model, restore_checkpoint
from kempnerforge.training.hooks import HookRunner
from kempnerforge.training.loop import TrainingSession, run_training_loop, vlm_step
from kempnerforge.training.loss import build_loss_fn
from kempnerforge.training.optimizer import build_optimizer
from kempnerforge.training.runtime import RuntimeContext
from kempnerforge.training.scheduler import build_scheduler

_ENCODER = "test_encoder_mode_dropout"
_IMAGE = torch.rand(1, 3, 8, 8)


class _DropoutEncoder(VisionEncoder):
    """An encoder whose features change between calls in training mode."""

    def __init__(self) -> None:
        super().__init__()
        self.num_tokens, self.feature_dim = 4, 16
        self.proj = torch.nn.Linear(3, 16)
        self.drop = torch.nn.Dropout(0.5)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.drop(self.proj(pixel_values.mean(dim=(2, 3))))
        return features.unsqueeze(1).expand(-1, self.num_tokens, -1)


def _config(vlm: VLMConfig, tmp_path, encoder: str = _ENCODER, steps: int = 3) -> JobConfig:
    if _ENCODER not in registry.list_vision_encoders():
        registry.register_vision_encoder(_ENCODER)(lambda path="", **_: _DropoutEncoder())
    return JobConfig(
        model=ModelConfig(
            dim=32, n_layers=1, n_heads=2, n_kv_heads=2, vocab_size=64, max_seq_len=16
        ),
        train=TrainConfig(
            batch_size=2, seq_len=12, max_steps=steps, compile_model=False, mixed_precision="fp32"
        ),
        optimizer=OptimizerConfig(fused=False),
        checkpoint=CheckpointConfig(dir=str(tmp_path / "ckpt"), interval=1000),
        vision_encoder=VisionEncoderConfig(type=encoder, num_tokens=4, feature_dim=16),
        adapter=AdapterConfig(),
        vlm=vlm,
    )


def _session(config: JobConfig, modes: list[tuple[int, bool, bool]]) -> TrainingSession:
    """A one-batch VLM session whose steps record the encoder's mode and repeatability."""
    torch.manual_seed(0)
    runtime = RuntimeContext(0, 0, 1, torch.device("cpu"), None)
    loss_fn = build_loss_fn(config.train)
    model, _ = build_model(config, runtime, loss_fn)
    assert config.vlm is not None
    optimizer = build_optimizer(model, config.optimizer, module_patterns=config.vlm.module_patterns)
    batch = {
        "pixel_values": torch.rand(2, 3, 8, 8),
        "input_ids": torch.randint(0, 64, (2, 8)),
        "labels": torch.randint(0, 64, (2, 8)),
    }
    data = DataPipeline(dataloader=[batch])

    def step_fn(session: TrainingSession, step: int) -> Any:
        encoder = session.model.vision_encoder
        with torch.no_grad():
            repeatable = torch.equal(encoder(_IMAGE), encoder(_IMAGE))
        modes.append((step, encoder.training, repeatable))
        return vlm_step(session, step)

    return TrainingSession(
        config=config,
        runtime=runtime,
        model=model,
        optimizer=optimizer,
        scheduler=build_scheduler(optimizer, config.scheduler, max_steps=config.train.max_steps),
        loss_fn=loss_fn,
        step_fn=step_fn,
        data=data,
        phases=build_phase_state(config, data, 0),
        checkpointer=build_checkpoint_manager(config, runtime, model, optimizer, None),
        tracker=MetricsTracker(config, num_gpus=1),
        hooks=HookRunner(),
        nan_detector=NaNDetector(action="warn", max_consecutive=10),
        shutdown_handler=ShutdownHandler(),
    )


class TestWrapperTrain:
    def test_a_frozen_encoder_stays_in_eval_while_the_rest_trains(self, tmp_path):
        model, _ = build_model(
            _config(VLMConfig(max_text_len=8), tmp_path),
            RuntimeContext(0, 0, 1, torch.device("cpu"), None),
            build_loss_fn(TrainConfig()),
        )
        assert model.train() is model
        assert not model.vision_encoder.training
        assert model.training and model.adapter.training and model.transformer.training
        model.eval()
        assert not any(m.training for m in model.modules())

    @pytest.mark.parametrize("trainable", ["vision_encoder.proj.weight", "all"])
    def test_an_encoder_with_a_trainable_parameter_trains(self, tmp_path, trainable):
        model, _ = build_model(
            _config(VLMConfig(max_text_len=8), tmp_path),
            RuntimeContext(0, 0, 1, torch.device("cpu"), None),
            build_loss_fn(TrainConfig()),
        )
        for name, param in model.vision_encoder.named_parameters(prefix="vision_encoder"):
            param.requires_grad_(trainable in ("all", name))
        model.train()
        assert model.vision_encoder.training

    def test_an_encoder_without_parameters_follows_the_mode(self, tmp_path):
        model, _ = build_model(
            _config(VLMConfig(max_text_len=8), tmp_path, encoder="random"),
            RuntimeContext(0, 0, 1, torch.device("cpu"), None),
            build_loss_fn(TrainConfig()),
        )
        assert not list(model.vision_encoder.parameters())
        model.train()
        assert model.vision_encoder.training
        model.eval()
        assert not model.vision_encoder.training


class TestTrainingLoop:
    def test_the_build_freeze_keeps_the_encoder_in_eval_at_every_step(self, tmp_path):
        modes: list = []
        run_training_loop(_session(_config(VLMConfig(max_text_len=8), tmp_path), modes))
        assert modes == [(0, False, True), (1, False, True), (2, False, True)]

    def test_freeze_stages_switch_the_encoder_to_eval_and_back(self, tmp_path):
        stages = [
            FreezeStage(1, (FreezeSpec("vision_encoder"),)),
            FreezeStage(2, (FreezeSpec("vision_encoder", frozen=False),)),
        ]
        vlm = VLMConfig(max_text_len=8, freeze=[], freeze_schedule=stages)
        modes: list = []
        run_training_loop(_session(_config(vlm, tmp_path), modes))
        assert [(step, training) for step, training, _ in modes] == [
            (0, True),
            (1, False),
            (2, True),
        ]
        assert modes[1][2]

    def test_a_resume_past_a_freeze_stage_runs_the_encoder_in_eval(self, tmp_path):
        """The restore applies the stage's freeze; the loop's train() then puts it in eval."""
        stage = FreezeStage(1, (FreezeSpec("vision_encoder"),))
        vlm = VLMConfig(max_text_len=8, freeze=[], freeze_schedule=[stage])
        run_training_loop(_session(_config(vlm, tmp_path, steps=2), []))
        config = _config(vlm, tmp_path, steps=4)
        modes: list = []
        session = _session(config, modes)
        step, tokens_seen = restore_checkpoint(
            config, session.model, session.scheduler, session.checkpointer
        )
        encoder = session.model.vision_encoder
        assert step == 2 and not any(p.requires_grad for p in encoder.parameters())
        assert encoder.training
        run_training_loop(session, step=step, tokens_seen=tokens_seen)
        assert modes == [(2, False, True), (3, False, True)]
