"""Unit tests for FP8 mixed precision support."""

from __future__ import annotations

import logging

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
        config.validate(world_size=4)

    @pytest.mark.parametrize("precision", ["fp8", "bf16"])
    def test_dim_not_divisible_by_16(self, precision):
        """No projection can run in FP8 when model.dim is not a multiple of 16 (issue #240)."""
        from kempnerforge.config.schema import JobConfig, ModelConfig

        config = JobConfig(
            model=ModelConfig(dim=72, n_layers=2, n_heads=4, vocab_size=256),
            train=TrainConfig(mixed_precision=precision),
        )
        if precision == "bf16":
            config.validate(world_size=1)
            return
        with pytest.raises(ValueError, match=r"needs model.dim divisible by 16 \(got 72\)"):
            config.validate(world_size=1)

    def test_unaligned_kv_projection_passes_validation(self):
        """head_dim 8 with one KV head is valid FP8: the filter keeps k_proj and v_proj bf16."""
        from kempnerforge.config.schema import JobConfig, ModelConfig

        JobConfig(
            model=ModelConfig(dim=64, n_layers=2, n_heads=8, n_kv_heads=1, vocab_size=256),
            train=TrainConfig(mixed_precision="fp8"),
        ).validate(world_size=1)


class TestApplyFloat8:
    """Test apply_float8 converts the right modules and skips experts."""

    @pytest.fixture
    def small_transformer(self):
        from kempnerforge.config.schema import ModelConfig
        from kempnerforge.model.transformer import Transformer

        mc = ModelConfig(dim=64, n_layers=2, n_heads=4, vocab_size=256)
        return Transformer(mc)

    @pytest.fixture
    def moe_transformer(self):
        from kempnerforge.config.schema import ModelConfig
        from kempnerforge.model.transformer import Transformer

        mc = ModelConfig(
            dim=64,
            n_layers=2,
            n_heads=4,
            vocab_size=256,
            num_experts=4,
            moe_top_k=2,
        )
        return Transformer(mc)

    def test_apply_float8_converts_linears(self, small_transformer):
        """Dense model: all nn.Linear modules should be converted to Float8Linear."""
        from torchao.float8.float8_linear import Float8Linear

        from kempnerforge.distributed.parallel import apply_float8

        linears_before = sum(
            1 for m in small_transformer.modules() if isinstance(m, torch.nn.Linear)
        )
        assert linears_before > 0

        apply_float8(small_transformer, enable_fsdp_float8_all_gather=False)

        float8_count = sum(1 for m in small_transformer.modules() if isinstance(m, Float8Linear))
        plain_linear_count = sum(
            1 for m in small_transformer.modules() if type(m) is torch.nn.Linear
        )
        assert float8_count > 0
        assert plain_linear_count == 0, f"Expected 0 plain nn.Linear, got {plain_linear_count}"

    def test_apply_float8_skips_expert_linears(self, moe_transformer):
        """MoE model: expert Linears stay as nn.Linear, others become Float8Linear."""
        from torchao.float8.float8_linear import Float8Linear

        from kempnerforge.distributed.parallel import apply_float8

        apply_float8(moe_transformer, enable_fsdp_float8_all_gather=False)

        for layer in moe_transformer.layers.values():
            if hasattr(layer.mlp, "experts"):
                for expert in layer.mlp.experts:
                    for name, m in expert.named_modules():
                        if isinstance(m, (torch.nn.Linear, Float8Linear)):
                            assert type(m) is torch.nn.Linear, (
                                f"Expert sub-module {name} was converted to Float8Linear"
                            )

        for layer in moe_transformer.layers.values():
            if hasattr(layer.mlp, "router"):
                gate = layer.mlp.router.gate
                assert type(gate) is torch.nn.Linear, "Router gate was converted to Float8Linear"

        float8_count = sum(1 for m in moe_transformer.modules() if isinstance(m, Float8Linear))
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
        grad_count = sum(1 for p in small_transformer.parameters() if p.grad is not None)
        assert grad_count > 0

    def test_build_parallel_model_fp8_flag(self):
        """build_parallel_model accepts fp8=True and applies Float8 conversion."""
        from torchao.float8.float8_linear import Float8Linear

        from kempnerforge.config.schema import ModelConfig
        from kempnerforge.distributed.parallel import build_parallel_model

        mc = ModelConfig(dim=64, n_layers=2, n_heads=4, vocab_size=256)
        device = torch.device("cpu")

        model = build_parallel_model(mc, device, device_mesh=None, fp8=True)

        float8_count = sum(1 for m in model.modules() if isinstance(m, Float8Linear))
        assert float8_count > 0


class TestFloat8Alignment:
    """``apply_float8`` keeps each Linear with a dimension not divisible by 16 in high precision.

    ``torch._scaled_mm`` rejects such a weight in the first forward on CUDA (issue #240); the
    CPU kernel does not check, so these tests check the conversion, not the crash.
    """

    @staticmethod
    def _kept(model: torch.nn.Module) -> set[str]:
        """Names, without the layer index, of the Linears left unconverted."""
        return {
            fqn.split(".", 2)[-1] if fqn.startswith("layers.") else fqn
            for fqn, m in model.named_modules()
            if type(m) is torch.nn.Linear
        }

    @pytest.mark.parametrize(
        ("geometry", "kept"),
        [
            (dict(dim=64, n_heads=8, n_kv_heads=1), {"attention.k_proj", "attention.v_proj"}),
            (dict(dim=64, n_heads=4, vocab_size=50257), {"output_head.proj"}),
            (
                dict(dim=64, n_heads=4, ffn_hidden_dim=200),
                {"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"},
            ),
            (dict(dim=64, n_heads=4, n_kv_heads=2), set()),
        ],
        ids=["kv-8x64", "vocab-50257", "ffn-200", "aligned"],
    )
    def test_converts_only_aligned_linears(self, geometry, kept):
        from torchao.float8.float8_linear import Float8Linear

        from kempnerforge.config.schema import ModelConfig
        from kempnerforge.distributed.parallel import apply_float8
        from kempnerforge.model.transformer import Transformer

        model = Transformer(ModelConfig(**{"n_layers": 2, "vocab_size": 256, **geometry}))
        apply_float8(model, enable_fsdp_float8_all_gather=False)
        converted = [m for m in model.modules() if isinstance(m, Float8Linear)]
        assert converted
        assert all(m.in_features % 16 == 0 and m.out_features % 16 == 0 for m in converted)
        assert self._kept(model) == kept

    def test_warns_once_naming_the_kept_layers(self, caplog, monkeypatch):
        from kempnerforge.config.schema import ModelConfig
        from kempnerforge.distributed.parallel import apply_float8
        from kempnerforge.model.transformer import Transformer

        monkeypatch.setattr(logging.getLogger("kempnerforge"), "propagate", True)
        model = Transformer(
            ModelConfig(dim=64, n_layers=3, n_heads=8, n_kv_heads=1, vocab_size=256)
        )
        with caplog.at_level(logging.INFO, logger="kempnerforge.distributed.parallel"):
            apply_float8(model, enable_fsdp_float8_all_gather=False)
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "layers.*.attention.k_proj (8x64) x3" in warnings[0]
        assert "layers.*.attention.v_proj (8x64) x3" in warnings[0]
        assert any("converted 16 of 22 Linear layers" in r.getMessage() for r in caplog.records)

    def test_mixed_model_runs_forward_backward(self):
        """Float8 and high-precision Linears in one block train together (CPU emulation)."""
        from kempnerforge.config.schema import ModelConfig
        from kempnerforge.distributed.parallel import apply_float8
        from kempnerforge.model.transformer import Transformer

        model = Transformer(
            ModelConfig(dim=64, n_layers=2, n_heads=8, n_kv_heads=1, vocab_size=256)
        )
        apply_float8(model, enable_fsdp_float8_all_gather=False)
        out = model(torch.randint(0, 256, (2, 16)))
        out.float().sum().backward()
        assert torch.isfinite(out).all()
        assert all(p.grad is not None for p in model.parameters())
