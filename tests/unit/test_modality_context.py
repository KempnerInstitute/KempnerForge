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

    def test_inputs_embeds_and_prefix_embeds_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            ModalityContext(
                inputs_embeds=torch.zeros(1, 4, 8),
                prefix_embeds=torch.zeros(1, 4, 8),
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
