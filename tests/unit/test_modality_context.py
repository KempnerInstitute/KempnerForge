"""Intra-context invariants on ``ModalityContext``.

Cross-arg invariants involving ``kv_caches`` (a ``Transformer.forward``
argument, not a context field) live in ``test_model.py`` next to the
forward-call sites.
"""

from __future__ import annotations

import pytest
import torch

from kempnerforge.model.modality import ModalityContext


class TestModalityContextInvariants:
    def test_default_construction_is_valid(self):
        """All-None context (text-only forward equivalent) constructs cleanly."""
        ctx = ModalityContext()
        assert ctx.inputs_embeds is None
        assert ctx.prefix_embeds is None
        assert ctx.image_features is None
        assert ctx.image_mask is None
        assert ctx.output_slice is None

    def test_output_slice_alone_is_valid(self):
        """output_slice composes with the tokens path (not constrained intra-context)."""
        ctx = ModalityContext(output_slice=slice(5, None))
        assert ctx.output_slice == slice(5, None)

    def test_inputs_embeds_alone_is_valid(self):
        ctx = ModalityContext(inputs_embeds=torch.zeros(1, 4, 8))
        assert ctx.inputs_embeds is not None

    def test_prefix_embeds_alone_is_valid(self):
        ctx = ModalityContext(prefix_embeds=torch.zeros(1, 4, 8))
        assert ctx.prefix_embeds is not None

    def test_image_features_alone_is_valid(self):
        ctx = ModalityContext(image_features=torch.zeros(1, 4, 8))
        assert ctx.image_features is not None
        assert ctx.image_mask is None

    def test_image_features_with_mask_is_valid(self):
        ctx = ModalityContext(
            image_features=torch.zeros(1, 4, 8),
            image_mask=torch.ones(1, 4, dtype=torch.bool),
        )
        assert ctx.image_mask is not None

    def test_inputs_embeds_and_prefix_embeds_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            ModalityContext(
                inputs_embeds=torch.zeros(1, 4, 8),
                prefix_embeds=torch.zeros(1, 4, 8),
            )

    def test_inputs_embeds_and_image_features_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            ModalityContext(
                inputs_embeds=torch.zeros(1, 4, 8),
                image_features=torch.zeros(1, 4, 8),
            )

    def test_prefix_embeds_and_image_features_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            ModalityContext(
                prefix_embeds=torch.zeros(1, 4, 8),
                image_features=torch.zeros(1, 4, 8),
            )

    def test_image_mask_without_image_features_raises(self):
        with pytest.raises(ValueError, match="image_mask requires image_features"):
            ModalityContext(image_mask=torch.ones(1, 4, dtype=torch.bool))

    def test_key_padding_mask_with_prefix_embeds_is_valid(self):
        ctx = ModalityContext(
            prefix_embeds=torch.zeros(1, 4, 8),
            key_padding_mask=torch.ones(1, 10, dtype=torch.bool),
        )
        assert ctx.key_padding_mask is not None

    def test_key_padding_mask_without_residual_raises(self):
        with pytest.raises(ValueError, match="key_padding_mask requires"):
            ModalityContext(key_padding_mask=torch.ones(1, 10, dtype=torch.bool))

    def test_three_residual_routes_raises(self):
        """Setting all three residual-route fields surfaces the same error."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            ModalityContext(
                inputs_embeds=torch.zeros(1, 4, 8),
                prefix_embeds=torch.zeros(1, 4, 8),
                image_features=torch.zeros(1, 4, 8),
            )


class TestModalityIdsInvariants:
    """modality_ids field for MoT routing.

    Intra-context invariants:

    - modality_ids requires prefix_embeds OR inputs_embeds (routing
      without a residual extension is meaningless).

    Cross-arg invariants (kv_caches incompatibility, dtype, shape) are
    enforced in Transformer.forward and tested in test_model.py.
    """

    def test_modality_ids_with_prefix_embeds_is_valid(self):
        ctx = ModalityContext(
            prefix_embeds=torch.zeros(1, 4, 8),
            modality_ids=torch.zeros(1, 12, dtype=torch.long),
        )
        assert ctx.modality_ids is not None
        assert ctx.modality_ids.dtype == torch.long

    def test_modality_ids_with_inputs_embeds_is_valid(self):
        ctx = ModalityContext(
            inputs_embeds=torch.zeros(1, 4, 8),
            modality_ids=torch.zeros(1, 4, dtype=torch.long),
        )
        assert ctx.modality_ids is not None

    def test_position_ids_and_group_ids_ride_along(self):
        ctx = ModalityContext(
            prefix_embeds=torch.zeros(1, 4, 8),
            position_ids=torch.arange(6).unsqueeze(0),
            attention_group_ids=torch.full((1, 6), -1, dtype=torch.long),
        )
        assert ctx.position_ids is not None and ctx.attention_group_ids is not None

    def test_position_ids_and_group_ids_need_no_residual_route(self):
        ctx = ModalityContext(attention_group_ids=torch.zeros(2, 3, dtype=torch.long))
        assert ctx.attention_group_ids is not None

    @pytest.mark.parametrize("name", ["position_ids", "attention_group_ids"])
    def test_ids_must_be_int64(self, name):
        with pytest.raises(ValueError, match=f"{name}.dtype must be torch.int64"):
            ModalityContext(**{name: torch.zeros(1, 4, dtype=torch.int32)})

    @pytest.mark.parametrize("name", ["position_ids", "attention_group_ids"])
    def test_ids_must_be_two_dimensional(self, name):
        with pytest.raises(ValueError, match=f"{name} must be \\(batch, seq_len\\)"):
            ModalityContext(**{name: torch.zeros(4, dtype=torch.long)})

    def test_rope_tables_ride_along(self):
        ctx = ModalityContext(rope_cos=torch.zeros(1, 6, 8), rope_sin=torch.zeros(1, 6, 8))
        assert ctx.rope_cos is not None and ctx.rope_sin is not None

    def test_rope_tables_come_together(self):
        with pytest.raises(ValueError, match="given together"):
            ModalityContext(rope_cos=torch.zeros(1, 6, 8))

    def test_rope_tables_exclude_position_ids(self):
        with pytest.raises(ValueError, match="both choose the rotation"):
            ModalityContext(
                rope_cos=torch.zeros(1, 6, 8),
                rope_sin=torch.zeros(1, 6, 8),
                position_ids=torch.arange(6).unsqueeze(0),
            )

    def test_rope_tables_must_be_floating_three_dimensional_and_alike(self):
        with pytest.raises(ValueError, match="rope_cos must be floating point"):
            ModalityContext(
                rope_cos=torch.zeros(1, 6, 8, dtype=torch.long), rope_sin=torch.zeros(1, 6, 8)
            )
        with pytest.raises(
            ValueError, match="rope_sin must be \\(batch, seq_len, head_dim // 2\\)"
        ):
            ModalityContext(rope_cos=torch.zeros(1, 6, 8), rope_sin=torch.zeros(6, 8))
        with pytest.raises(ValueError, match="differ in shape"):
            ModalityContext(rope_cos=torch.zeros(1, 6, 8), rope_sin=torch.zeros(1, 5, 8))

    def test_modality_ids_alone_raises(self):
        """Bare modality_ids (no prefix/inputs_embeds) is meaningless."""
        with pytest.raises(ValueError, match="modality_ids requires"):
            ModalityContext(modality_ids=torch.zeros(1, 4, dtype=torch.long))

    def test_modality_ids_only_with_output_slice_raises(self):
        """output_slice is not a residual-extension route, so modality_ids
        with only output_slice still raises."""
        with pytest.raises(ValueError, match="modality_ids requires"):
            ModalityContext(
                output_slice=slice(4, None),
                modality_ids=torch.zeros(1, 8, dtype=torch.long),
            )
