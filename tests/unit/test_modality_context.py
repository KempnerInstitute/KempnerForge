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
