"""Unit tests for distributed utilities (no GPU/multi-process required)."""

from __future__ import annotations

from kempnerforge.config.schema import ModelConfig
from kempnerforge.distributed.tensor_parallel import _build_block_tp_plan
from kempnerforge.model.transformer import TransformerBlock


def _make_block(activation: str = "silu") -> TransformerBlock:
    """Create a small TransformerBlock on CPU for plan builder testing."""
    config = ModelConfig(dim=64, n_layers=2, n_heads=4, vocab_size=128, activation=activation)
    return TransformerBlock(config, layer_idx=0)


class TestBuildBlockTPPlan:
    def test_plan_with_sequence_parallel(self):
        """Plan should include SequenceParallel for norms and Shard(1) layouts."""
        block = _make_block()
        plan = _build_block_tp_plan(block, sequence_parallel=True)
        assert "attention_norm" in plan
        assert "mlp_norm" in plan
        assert "attention.q_proj" in plan
        assert "attention.o_proj" in plan
        assert "mlp.down_proj" in plan

    def test_plan_without_sequence_parallel(self):
        """Plan should NOT include SequenceParallel norms in basic TP mode."""
        block = _make_block()
        plan = _build_block_tp_plan(block, sequence_parallel=False)
        assert "attention_norm" not in plan
        assert "mlp_norm" not in plan
        # Projections should still be parallelized
        assert "attention.q_proj" in plan
        assert "attention.o_proj" in plan

    def test_swiglu_includes_gate_proj(self):
        """SwiGLU blocks should shard both gate_proj and up_proj."""
        block = _make_block(activation="silu")  # SwiGLU
        plan = _build_block_tp_plan(block, sequence_parallel=False)
        assert "mlp.gate_proj" in plan
        assert "mlp.up_proj" in plan
        assert "mlp.down_proj" in plan

    def test_standard_mlp_excludes_gate_proj(self):
        """Standard MLP blocks should only shard up_proj, not gate_proj."""
        block = _make_block(activation="relu")  # StandardMLP
        plan = _build_block_tp_plan(block, sequence_parallel=False)
        assert "mlp.gate_proj" not in plan
        assert "mlp.up_proj" in plan
        assert "mlp.down_proj" in plan

    def test_all_attention_projections_present(self):
        """All four attention projections should be in the plan."""
        block = _make_block()
        plan = _build_block_tp_plan(block, sequence_parallel=False)
        for proj in [
            "attention.q_proj",
            "attention.k_proj",
            "attention.v_proj",
            "attention.o_proj",
        ]:
            assert proj in plan, f"{proj} missing from plan"

    def test_plan_key_count_with_sp(self):
        """With SP + SwiGLU: 2 norms + 4 attn + 3 mlp = 9 entries."""
        block = _make_block(activation="silu")
        plan = _build_block_tp_plan(block, sequence_parallel=True)
        assert len(plan) == 9

    def test_plan_key_count_without_sp(self):
        """Without SP + SwiGLU: 4 attn + 3 mlp = 7 entries (no norms)."""
        block = _make_block(activation="silu")
        plan = _build_block_tp_plan(block, sequence_parallel=False)
        assert len(plan) == 7
