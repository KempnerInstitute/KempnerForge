"""Unit tests for FP8 mixed precision support."""

from __future__ import annotations

import pytest
import torch

from kempnerforge.config.schema import TrainConfig


class TestFP8Config:
    def test_mixed_precision_fp8_accepted(self):
        tc = TrainConfig(mixed_precision="fp8")
        assert tc.mixed_precision == "fp8"

    def test_fp8_param_dtype_is_bf16(self):
        """FP8 is a compute mode — master weights stay in bf16."""
        tc = TrainConfig(mixed_precision="fp8")
        assert tc.param_dtype == torch.bfloat16

    def test_is_fp8_true(self):
        tc = TrainConfig(mixed_precision="fp8")
        assert tc.is_fp8 is True

    def test_is_fp8_false_for_bf16(self):
        tc = TrainConfig(mixed_precision="bf16")
        assert tc.is_fp8 is False

    def test_other_precisions_unchanged(self):
        assert TrainConfig(mixed_precision="bf16").param_dtype == torch.bfloat16
        assert TrainConfig(mixed_precision="fp16").param_dtype == torch.float16
        assert TrainConfig(mixed_precision="fp32").param_dtype == torch.float32

    def test_fp8_plus_tp_rejected(self):
        """FP8 + TP is not supported — should fail config validation."""
        from kempnerforge.config.schema import DistributedConfig, JobConfig, ModelConfig

        with pytest.raises(ValueError, match="FP8.*Tensor Parallelism.*not yet supported"):
            JobConfig(
                model=ModelConfig(dim=256, n_layers=4, n_heads=4, vocab_size=32000),
                train=TrainConfig(mixed_precision="fp8"),
                distributed=DistributedConfig(tp=2, dp_shard=2),
            ).validate(world_size=4)

    def test_fp8_plus_fsdp_accepted(self):
        """FP8 + FSDP (no TP) should pass config validation."""
        from kempnerforge.config.schema import DistributedConfig, JobConfig, ModelConfig

        config = JobConfig(
            model=ModelConfig(dim=256, n_layers=4, n_heads=4, vocab_size=32000),
            train=TrainConfig(mixed_precision="fp8"),
            distributed=DistributedConfig(dp_shard=4),
        )
        config.validate(world_size=4)  # should not raise


class TestApplyFloat8:
    """Test apply_float8 converts the right modules and skips experts."""

    @pytest.fixture
    def small_transformer(self):
        from kempnerforge.config.schema import ModelConfig
        from kempnerforge.model.transformer import Transformer  # triggers registration

        mc = ModelConfig(dim=64, n_layers=2, n_heads=4, vocab_size=256)
        return Transformer(mc)

    @pytest.fixture
    def moe_transformer(self):
        from kempnerforge.config.schema import ModelConfig
        from kempnerforge.model.transformer import Transformer  # triggers registration

        mc = ModelConfig(
            dim=64, n_layers=2, n_heads=4, vocab_size=256,
            num_experts=4, moe_top_k=2,
        )
        return Transformer(mc)

    def test_apply_float8_converts_linears(self, small_transformer):
        """Dense model: all nn.Linear modules should be converted to Float8Linear."""
        from torchao.float8.float8_linear import Float8Linear

        from kempnerforge.distributed.parallel import apply_float8

        # Count nn.Linear before
        linears_before = sum(
            1 for m in small_transformer.modules() if isinstance(m, torch.nn.Linear)
        )
        assert linears_before > 0

        apply_float8(small_transformer, enable_fsdp_float8_all_gather=False)

        # All nn.Linear should now be Float8Linear
        float8_count = sum(
            1 for m in small_transformer.modules() if isinstance(m, Float8Linear)
        )
        plain_linear_count = sum(
            1 for m in small_transformer.modules()
            if type(m) is torch.nn.Linear
        )
        assert float8_count > 0
        assert plain_linear_count == 0, f"Expected 0 plain nn.Linear, got {plain_linear_count}"

    def test_apply_float8_skips_expert_linears(self, moe_transformer):
        """MoE model: expert Linears stay as nn.Linear, others become Float8Linear."""
        from torchao.float8.float8_linear import Float8Linear

        from kempnerforge.distributed.parallel import apply_float8

        apply_float8(moe_transformer, enable_fsdp_float8_all_gather=False)

        # Expert modules should still have plain nn.Linear
        for layer in moe_transformer.layers.values():
            if hasattr(layer.mlp, "experts"):
                for expert in layer.mlp.experts:
                    for name, m in expert.named_modules():
                        if isinstance(m, (torch.nn.Linear, Float8Linear)):
                            assert type(m) is torch.nn.Linear, (
                                f"Expert sub-module {name} was converted to Float8Linear"
                            )

        # Router gate should stay as plain nn.Linear (small dim, not divisible by 16)
        for layer in moe_transformer.layers.values():
            if hasattr(layer.mlp, "router"):
                gate = layer.mlp.router.gate
                assert type(gate) is torch.nn.Linear, (
                    "Router gate was converted to Float8Linear"
                )

        # Non-expert, non-router modules (attention, output) should be Float8Linear
        float8_count = sum(
            1 for m in moe_transformer.modules() if isinstance(m, Float8Linear)
        )
        assert float8_count > 0, "No Float8Linear modules found — conversion failed"

    def test_apply_float8_forward_backward(self, small_transformer):
        """Float8 model produces finite output and gradients on CPU (emulated)."""
        from kempnerforge.distributed.parallel import apply_float8

        apply_float8(small_transformer, enable_fsdp_float8_all_gather=False)

        x = torch.randint(0, 256, (1, 16))
        out = small_transformer(x)
        assert out.shape == (1, 16, 256)
        assert torch.isfinite(out).all()

        loss = out.sum()
        loss.backward()
        grad_count = sum(
            1 for p in small_transformer.parameters() if p.grad is not None
        )
        assert grad_count > 0

    def test_build_parallel_model_fp8_flag(self):
        """build_parallel_model accepts fp8=True and applies Float8 conversion."""
        from torchao.float8.float8_linear import Float8Linear

        from kempnerforge.config.schema import ModelConfig
        from kempnerforge.distributed.parallel import build_parallel_model

        mc = ModelConfig(dim=64, n_layers=2, n_heads=4, vocab_size=256)
        device = torch.device("cpu")

        model = build_parallel_model(mc, device, device_mesh=None, fp8=True)

        float8_count = sum(
            1 for m in model.modules() if isinstance(m, Float8Linear)
        )
        assert float8_count > 0
