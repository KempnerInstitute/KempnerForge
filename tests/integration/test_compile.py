"""Integration test: torch.compile correctness.

Verifies that compiled model produces identical (or nearly identical)
output compared to eager mode. Also tests that compilation doesn't
break the backward pass.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from kempnerforge.config.model import FLEX_BLOCK_SIZE
from kempnerforge.config.schema import ModelConfig
from kempnerforge.model.transformer import Transformer


@pytest.mark.gpu
class TestCompileCorrectness:
    def test_compiled_forward_matches_eager(self):
        """Compiled model produces same logits as eager model."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        config = ModelConfig(dim=128, n_layers=2, n_heads=2, vocab_size=256, max_seq_len=32)
        device = torch.device("cuda")

        model = Transformer(config).to(device).eval()
        tokens = torch.randint(0, 256, (1, 16), device=device)

        with torch.no_grad():
            eager_out = model(tokens)

        compiled_model = torch.compile(model)
        with torch.no_grad():
            compiled_out = compiled_model(tokens)

        max_diff = (eager_out - compiled_out).abs().max().item()
        assert max_diff < 1e-4, f"Compiled output differs: max diff={max_diff}"

    def test_compiled_backward_produces_gradients(self):
        """Compiled model produces valid gradients."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        config = ModelConfig(dim=128, n_layers=2, n_heads=2, vocab_size=256, max_seq_len=32)
        device = torch.device("cuda")

        model = Transformer(config).to(device)
        compiled_model = torch.compile(model)

        tokens = torch.randint(0, 256, (2, 16), device=device)
        labels = torch.randint(0, 256, (2, 16), device=device)

        logits = compiled_model(tokens)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()

        # All parameters should have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_compiled_loss_matches_eager(self):
        """Loss from compiled model matches eager model on same input."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        config = ModelConfig(dim=128, n_layers=2, n_heads=2, vocab_size=256, max_seq_len=32)
        device = torch.device("cuda")

        model = Transformer(config).to(device).eval()
        tokens = torch.randint(0, 256, (2, 16), device=device)
        labels = torch.randint(0, 256, (2, 16), device=device)

        with torch.no_grad():
            eager_logits = model(tokens)
            eager_loss = F.cross_entropy(
                eager_logits.view(-1, eager_logits.size(-1)), labels.view(-1)
            )

        compiled_model = torch.compile(model)
        with torch.no_grad():
            compiled_logits = compiled_model(tokens)
            compiled_loss = F.cross_entropy(
                compiled_logits.view(-1, compiled_logits.size(-1)), labels.view(-1)
            )

        assert abs(eager_loss.item() - compiled_loss.item()) < 1e-4


@pytest.mark.gpu
class TestFlexCompile:
    """FlexAttention packed-attention path, which only exists on CUDA.

    torch 2.11 has no CPU backward for flex and raises as soon as an input
    requires grad, so every gradient and compiled-kernel assertion for that
    backend lives here rather than in ``tests/unit/``.

    Every sequence here is at least ``FLEX_BLOCK_SIZE``. Below one block the
    compiled kernel silently returns wrong results -- documents leak into each
    other -- which is why ``JobConfig.validate`` rejects that configuration;
    see ``tests/unit/test_config.py`` for the guard itself.
    """

    SEQ = 256

    @staticmethod
    def _layouts(seq: int) -> list[list[int]]:
        half = seq // 2
        quarter = seq // 4
        return [
            [0] * seq,
            [0] * half + [1] * (seq - half),
            [0] * quarter + [1] * quarter + [2] * (seq - 2 * quarter),
            [min(i * 8 // seq, 7) for i in range(seq)],
            [0] * (seq - 1) + [1],
        ]

    def _model(self, backend: str, seq: int | None = None, **overrides):
        torch.manual_seed(0)
        config = ModelConfig(
            dim=128,
            n_layers=2,
            n_heads=4,
            vocab_size=256,
            max_seq_len=max(seq or self.SEQ, FLEX_BLOCK_SIZE),
            attention_backend=backend,
            **overrides,
        )
        return Transformer(config).to("cuda")

    def _batch(self, layout_idx: int = 1, seq: int | None = None, seed: int = 0):
        seq = seq or self.SEQ
        gen = torch.Generator(device="cuda").manual_seed(seed)
        tokens = torch.randint(0, 256, (2, seq), device="cuda", generator=gen)
        row = self._layouts(seq)[layout_idx]
        return tokens, torch.tensor([row, row], device="cuda")

    @staticmethod
    def _perturb_first_doc(tokens: torch.Tensor, boundary: int) -> torch.Tensor:
        perturbed = tokens.clone()
        perturbed[:, :boundary] = (perturbed[:, :boundary] + 1) % 256
        return perturbed

    @pytest.mark.parametrize("overrides", [{}, {"n_kv_heads": 2}], ids=["mha", "gqa"])
    def test_forward_matches_sdpa(self, overrides):
        """Flex reproduces the dense-mask SDPA logits it replaces."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        tokens, doc_ids = self._batch()
        with torch.no_grad():
            out_sdpa = self._model("sdpa", **overrides).eval()(tokens, doc_ids=doc_ids)
            out_flex = self._model("flex", **overrides).eval()(tokens, doc_ids=doc_ids)
        torch.testing.assert_close(out_flex, out_sdpa, rtol=2e-5, atol=2e-5)

    @pytest.mark.parametrize("overrides", [{}, {"n_kv_heads": 2}], ids=["mha", "gqa"])
    def test_gradients_match_sdpa(self, overrides):
        """Backward parity -- the assertion CPU cannot make (no flex CPU backward).

        Compared relative to each parameter's own gradient scale rather than with
        a fixed atol: at S=256 gradients reach ~1.8e3, so an absolute bound is
        meaningless. Measured against an fp64 CPU reference, flex is *closer* to
        exact than the dense-mask SDPA path on every parameter (relative error
        3.1e-7..5.4e-7 vs SDPA's 3.2e-7..9.1e-7), and the two fp32 kernels differ
        by 4.9e-7 of full scale. The 1e-5 bound below therefore carries ~20x
        headroom while still catching real divergence, which shows up orders of
        magnitude larger -- the pre-fix compiled bug moved logits by 4e-2.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        tokens, doc_ids = self._batch()
        grads = {}
        for backend in ("sdpa", "flex"):
            model = self._model(backend, **overrides)
            model(tokens, doc_ids=doc_ids).sum().backward()
            grads[backend] = {
                name: p.grad.detach().clone()
                for name, p in model.named_parameters()
                if p.grad is not None
            }

        assert grads["sdpa"].keys() == grads["flex"].keys()
        for name, reference in grads["sdpa"].items():
            actual = grads["flex"][name]
            assert torch.isfinite(actual).all(), f"non-finite grad for {name}"
            scale = reference.abs().max()
            if scale == 0:
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                continue
            # Largest elementwise disagreement, as a fraction of full scale.
            elementwise = ((actual - reference).abs().max() / scale).item()
            # Whole-tensor agreement, which a single bad element cannot hide in.
            relative_norm = ((actual - reference).norm() / reference.norm()).item()
            assert elementwise < 1e-5, f"{name}: elementwise {elementwise:.2e} of scale {scale:.2e}"
            assert relative_norm < 1e-5, f"{name}: relative norm {relative_norm:.2e}"

    @pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
    def test_cross_document_isolation(self, compiled):
        """Perturbing document 0 must not move any document-1 logit, at all.

        The compiled case is a regression test: an earlier revision passed
        eagerly and leaked under ``torch.compile``, which is the failure this
        whole path exists to prevent and the one a loss curve would not reveal.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        model = self._model("flex").eval()
        run = torch.compile(model) if compiled else model
        tokens, doc_ids = self._batch(layout_idx=1)
        boundary = self.SEQ // 2
        with torch.no_grad():
            base = run(tokens, doc_ids=doc_ids)
            moved = run(self._perturb_first_doc(tokens, boundary), doc_ids=doc_ids)
        torch.testing.assert_close(base[:, boundary:], moved[:, boundary:], rtol=0, atol=0)
        assert not torch.allclose(base[:, :boundary], moved[:, :boundary])

    @pytest.mark.parametrize("seq", [FLEX_BLOCK_SIZE, 200, 300])
    def test_sequence_lengths_around_the_block_size(self, seq):
        """Exactly one block, and ragged lengths above it, all stay exact.

        ``create_block_mask`` rounds up to the block size, so 200 and 300 carry
        a partial trailing block; the kernel must still isolate documents there.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        model = self._model("flex", seq=seq).eval()
        compiled = torch.compile(model)
        tokens, doc_ids = self._batch(layout_idx=1, seq=seq)
        boundary = seq // 2
        with torch.no_grad():
            eager_out = model(tokens, doc_ids=doc_ids)
            comp_out = compiled(tokens, doc_ids=doc_ids)
            comp_moved = compiled(self._perturb_first_doc(tokens, boundary), doc_ids=doc_ids)
        torch.testing.assert_close(comp_out, eager_out, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(comp_out[:, boundary:], comp_moved[:, boundary:], rtol=0, atol=0)

    def test_no_recompilation_across_document_layouts(self):
        """A fresh mask_mod closure per step must not retrigger compilation.

        ``build_doc_causal_block_mask`` defines ``mask_mod`` inline, closing over
        that step's ``doc_ids``, so every step hands ``flex_attention`` a new
        closure object. If Dynamo guarded on closure identity that would be a
        recompile per step, which blows ``cache_size_limit`` and silently falls
        back to *eager* flex -- the unfused decomposition that materializes the
        full score matrix, i.e. slower and hungrier than the dense SDPA path this
        replaces, with no error to notice.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        from torch._dynamo.utils import counters

        torch._dynamo.reset()
        counters.clear()

        model = self._model("flex")
        tokens, _ = self._batch()
        layouts = self._layouts(self.SEQ)

        def step(layout_idx):
            row = layouts[layout_idx]
            doc_ids = torch.tensor([row, row], device="cuda")
            model(tokens, doc_ids=doc_ids).sum().backward()
            model.zero_grad(set_to_none=True)

        # Warm up on two different layouts: flex may compile one or two
        # block-sparsity specializations up front, a fixed startup cost rather
        # than a per-step recompile.
        step(0)
        step(1)
        after_warmup = counters["stats"]["unique_graphs"]
        assert after_warmup > 0, "flex_attention did not compile at all"

        for layout_idx in range(2, len(layouts)):
            step(layout_idx)

        assert counters["stats"]["unique_graphs"] == after_warmup, (
            f"recompiled {counters['stats']['unique_graphs'] - after_warmup} time(s) across "
            f"{len(layouts) - 2} document layouts -- the mask_mod closure is being guarded "
            "on identity"
        )

    def test_compiled_model_matches_eager(self):
        """flex composes with an outer torch.compile over the whole model."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        model = self._model("flex").eval()
        tokens, doc_ids = self._batch()
        with torch.no_grad():
            eager_out = model(tokens, doc_ids=doc_ids)
            compiled_out = torch.compile(model)(tokens, doc_ids=doc_ids)
        torch.testing.assert_close(compiled_out, eager_out, rtol=2e-5, atol=2e-5)
