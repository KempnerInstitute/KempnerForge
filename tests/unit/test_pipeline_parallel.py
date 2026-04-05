"""Unit tests for pipeline parallelism (no GPU required)."""

import pytest
import torch

from kempnerforge.config.schema import ModelConfig, PipelineSchedule
from kempnerforge.distributed.pipeline_parallel import (
    PipelineStageModule,
    build_stage_module,
    compute_layer_assignment,
)

# ---------------------------------------------------------------------------
# Layer assignment
# ---------------------------------------------------------------------------


class TestComputeLayerAssignment:
    def test_even_split(self):
        result = compute_layer_assignment(32, 4)
        assert result == [(0, 8), (8, 16), (16, 24), (24, 32)]

    def test_uneven_split(self):
        # 10 layers / 3 stages → 4, 3, 3 (earlier stages get extra)
        result = compute_layer_assignment(10, 3)
        assert result == [(0, 4), (4, 7), (7, 10)]

    def test_single_stage(self):
        result = compute_layer_assignment(32, 1)
        assert result == [(0, 32)]

    def test_one_layer_per_stage(self):
        result = compute_layer_assignment(4, 4)
        assert result == [(0, 1), (1, 2), (2, 3), (3, 4)]

    def test_two_stages(self):
        result = compute_layer_assignment(7, 2)
        assert result == [(0, 4), (4, 7)]

    def test_pp_exceeds_layers_raises(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            compute_layer_assignment(4, 8)

    def test_all_layers_covered(self):
        """Every layer index appears in exactly one stage."""
        for n_layers in [8, 12, 17, 32]:
            for pp_size in [1, 2, 3, 4]:
                if pp_size > n_layers:
                    continue
                assignments = compute_layer_assignment(n_layers, pp_size)
                all_layers = []
                for start, end in assignments:
                    all_layers.extend(range(start, end))
                assert all_layers == list(range(n_layers))


# ---------------------------------------------------------------------------
# PipelineStageModule
# ---------------------------------------------------------------------------


@pytest.fixture
def small_config():
    return ModelConfig(dim=64, n_layers=8, n_heads=4, vocab_size=256, max_seq_len=32)


class TestPipelineStageModule:
    def test_first_stage_has_embedding(self, small_config):
        module = PipelineStageModule(small_config, stage_id=0, num_stages=2, layer_range=(0, 4))
        assert module.token_embedding is not None
        assert module.output_head is None
        assert module.norm is None
        assert len(module.layers) == 4

    def test_last_stage_has_output(self, small_config):
        module = PipelineStageModule(small_config, stage_id=1, num_stages=2, layer_range=(4, 8))
        assert module.token_embedding is None
        assert module.output_head is not None
        assert module.norm is not None
        assert len(module.layers) == 4

    def test_middle_stage(self):
        config = ModelConfig(dim=64, n_layers=12, n_heads=4, vocab_size=256, max_seq_len=32)
        module = PipelineStageModule(config, stage_id=1, num_stages=3, layer_range=(4, 8))
        assert module.token_embedding is None
        assert module.output_head is None
        assert module.norm is None
        assert len(module.layers) == 4

    def test_single_stage_has_everything(self, small_config):
        module = PipelineStageModule(small_config, stage_id=0, num_stages=1, layer_range=(0, 8))
        assert module.token_embedding is not None
        assert module.output_head is not None
        assert module.norm is not None
        assert len(module.layers) == 8

    def test_layer_keys_match_global_indices(self, small_config):
        """Layer keys must match the full model for DCP checkpoint compatibility."""
        module = PipelineStageModule(small_config, stage_id=1, num_stages=2, layer_range=(4, 8))
        assert list(module.layers.keys()) == ["4", "5", "6", "7"]

    def test_first_stage_forward_shape(self, small_config):
        module = PipelineStageModule(small_config, stage_id=0, num_stages=2, layer_range=(0, 4))
        tokens = torch.randint(0, 256, (2, 16))
        out = module(tokens)
        # Not last stage → hidden states
        assert out.shape == (2, 16, 64)

    def test_last_stage_forward_shape(self, small_config):
        module = PipelineStageModule(small_config, stage_id=1, num_stages=2, layer_range=(4, 8))
        hidden = torch.randn(2, 16, 64)
        out = module(hidden)
        # Last stage → logits
        assert out.shape == (2, 16, 256)

    def test_single_stage_forward_shape(self, small_config):
        module = PipelineStageModule(small_config, stage_id=0, num_stages=1, layer_range=(0, 8))
        tokens = torch.randint(0, 256, (2, 16))
        out = module(tokens)
        # Only stage → logits
        assert out.shape == (2, 16, 256)

    def test_chained_forward_matches_shapes(self, small_config):
        """First stage output feeds into last stage input correctly."""
        stage0 = PipelineStageModule(small_config, stage_id=0, num_stages=2, layer_range=(0, 4))
        stage1 = PipelineStageModule(small_config, stage_id=1, num_stages=2, layer_range=(4, 8))

        tokens = torch.randint(0, 256, (2, 16))
        hidden = stage0(tokens)
        assert hidden.shape == (2, 16, 64)

        logits = stage1(hidden)
        assert logits.shape == (2, 16, 256)

    def test_stage_fewer_params_than_full_model(self, small_config):
        """Each stage has fewer parameters than the full model."""
        from kempnerforge.model.transformer import Transformer

        full_model = Transformer(small_config)
        full_params = sum(p.numel() for p in full_model.parameters())

        stage = PipelineStageModule(small_config, stage_id=0, num_stages=2, layer_range=(0, 4))
        stage_params = sum(p.numel() for p in stage.parameters())
        assert stage_params < full_params


# ---------------------------------------------------------------------------
# build_stage_module
# ---------------------------------------------------------------------------


class TestBuildStageModule:
    def test_first_stage(self, small_config):
        stage = build_stage_module(small_config, pp_rank=0, pp_size=2)
        assert stage.is_first
        assert not stage.is_last
        assert len(stage.layers) == 4

    def test_last_stage(self, small_config):
        stage = build_stage_module(small_config, pp_rank=1, pp_size=2)
        assert not stage.is_first
        assert stage.is_last
        assert len(stage.layers) == 4

    def test_four_stages(self, small_config):
        stages = [build_stage_module(small_config, pp_rank=i, pp_size=4) for i in range(4)]
        for i, s in enumerate(stages):
            assert len(s.layers) == 2
            assert s.is_first == (i == 0)
            assert s.is_last == (i == 3)

    def test_total_layers_across_stages(self, small_config):
        """All stages together cover all layers."""
        all_keys = []
        for i in range(4):
            stage = build_stage_module(small_config, pp_rank=i, pp_size=4)
            all_keys.extend(stage.layers.keys())
        assert all_keys == [str(i) for i in range(8)]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestPipelineScheduleConfig:
    def test_default_schedule(self):
        from kempnerforge.config.schema import DistributedConfig

        dc = DistributedConfig()
        assert dc.pp_schedule == PipelineSchedule.schedule_1f1b

    def test_schedule_values(self):
        assert PipelineSchedule.schedule_1f1b == "1f1b"
        assert PipelineSchedule.gpipe == "gpipe"
        assert PipelineSchedule.interleaved_1f1b == "interleaved_1f1b"
