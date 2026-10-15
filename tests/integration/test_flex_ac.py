"""FlexAttention composed with activation checkpointing and torch.compile.

`apply_ac` wraps with NO_REENTRANT checkpointing -- `full` around every
TransformerBlock, `selective` around every Attention -- and the model may then
be wrapped in torch.compile. That nests the flex higher-order op inside the
activation-checkpoint higher-order op inside the model graph, which is the
least-exercised combination this backend has to survive.
"""

from __future__ import annotations

import pytest
import torch

from kempnerforge.config.model import FLEX_BLOCK_SIZE
from kempnerforge.config.schema import ActivationCheckpointing, ModelConfig
from kempnerforge.distributed.parallel import apply_ac
from kempnerforge.model.transformer import Transformer

SEQ = 256
BATCH = 2


def _model(backend: str) -> Transformer:
    torch.manual_seed(0)
    config = ModelConfig(
        dim=128,
        n_layers=2,
        n_heads=4,
        vocab_size=256,
        max_seq_len=SEQ,
        attention_backend=backend,
    )
    return Transformer(config).to("cuda")


def _unwrapped_name(name: str) -> str:
    """Strip the segment `apply_ac` injects into parameter names.

    `checkpoint_wrapper` reparents each wrapped module under
    `_checkpoint_wrapped_module`, so an AC'd model's parameter names no longer
    line up with the plain model's -- the same trap `_orig_mod.` sets under
    torch.compile.
    """
    return name.replace("._checkpoint_wrapped_module", "")


def _batch():
    gen = torch.Generator(device="cuda").manual_seed(0)
    tokens = torch.randint(0, 256, (BATCH, SEQ), device="cuda", generator=gen)
    half = SEQ // 2
    row = [0] * half + [1] * (SEQ - half)
    return tokens, torch.tensor([row] * BATCH, device="cuda"), half


@pytest.mark.gpu
class TestFlexActivationCheckpointing:
    @pytest.mark.parametrize(
        "ac_mode",
        [
            ActivationCheckpointing.none,
            ActivationCheckpointing.selective,
            ActivationCheckpointing.full,
        ],
        ids=["ac_none", "ac_selective", "ac_full"],
    )
    @pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
    def test_matches_plain_sdpa(self, ac_mode, compiled):
        """Flex under every AC mode reproduces the dense-mask SDPA reference.

        The reference is deliberately built without AC: checkpointing is a
        recompute strategy and must not change the answer, so folding it into
        the reference too would hide exactly the failure this looks for.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        tokens, doc_ids, _ = _batch()
        reference = _model("sdpa").eval()
        with torch.no_grad():
            expected = reference(tokens, doc_ids=doc_ids)

        model = _model("flex").eval()
        apply_ac(model, ac_mode)
        run = torch.compile(model) if compiled else model
        with torch.no_grad():
            actual = run(tokens, doc_ids=doc_ids)

        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    @pytest.mark.parametrize(
        "ac_mode",
        [ActivationCheckpointing.selective, ActivationCheckpointing.full],
        ids=["ac_selective", "ac_full"],
    )
    def test_gradients_are_finite_and_match_sdpa(self, ac_mode):
        """Recompute must reproduce the same backward, not merely run."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        tokens, doc_ids, _ = _batch()

        reference = _model("sdpa")
        reference(tokens, doc_ids=doc_ids).sum().backward()
        expected = {n: p.grad.detach().clone() for n, p in reference.named_parameters()}

        model = _model("flex")
        apply_ac(model, ac_mode)
        model(tokens, doc_ids=doc_ids).sum().backward()

        assert {_unwrapped_name(n) for n, _ in model.named_parameters()} == set(expected)
        for name, param in model.named_parameters():
            reference_grad = expected[_unwrapped_name(name)]
            assert param.grad is not None, f"no gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"non-finite gradient for {name}"
            scale = reference_grad.abs().max()
            if scale == 0:
                continue
            drift = ((param.grad - reference_grad).abs().max() / scale).item()
            assert drift < 1e-5, f"{name}: {drift:.2e} of scale {scale:.2e}"

    @pytest.mark.parametrize(
        "ac_mode",
        [ActivationCheckpointing.selective, ActivationCheckpointing.full],
        ids=["ac_selective", "ac_full"],
    )
    def test_documents_stay_isolated(self, ac_mode):
        """Checkpointing recomputes attention; the mask must survive the recompute."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        tokens, doc_ids, boundary = _batch()
        model = _model("flex").eval()
        apply_ac(model, ac_mode)

        perturbed = tokens.clone()
        perturbed[:, :boundary] = (perturbed[:, :boundary] + 1) % 256
        with torch.no_grad():
            base = model(tokens, doc_ids=doc_ids)
            moved = model(perturbed, doc_ids=doc_ids)

        torch.testing.assert_close(base[:, boundary:], moved[:, boundary:], rtol=0, atol=0)
        assert not torch.allclose(base[:, :boundary], moved[:, :boundary])


@pytest.mark.gpu
def test_sequence_length_floor_is_the_block_size():
    """The AC tests above all sit above FLEX_BLOCK_SIZE, which is load-bearing."""
    assert SEQ >= FLEX_BLOCK_SIZE
