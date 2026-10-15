"""Integration test: bidirectional visibility groups on the CUDA flex path.

GPU-side assertions the CPU unit tier cannot make (torch 2.11 has no CPU
FlexAttention backward): dense-vs-flex gradient parity with groups, group ids
past the int32 range against the dense reference, the compiled kernel at
lengths around the block size, recompile stability across group layouts, and a
16384-token pack against the dense reference.

Mask semantics are covered exhaustively in ``tests/unit/test_attention_groups.py``.
"""

from __future__ import annotations

import pytest
import torch

from kempnerforge.config.model import FLEX_BLOCK_SIZE
from kempnerforge.config.schema import ModelConfig
from kempnerforge.model.transformer import Transformer


def _spans(seq: int, split: int) -> tuple[list[int], list[int], int, int]:
    """Two documents split at ``split``; group id 0 inside document 0, reused at
    the head of document 1. Returns the doc row, the group row, and the bounds
    of the document-0 group."""
    g_start, g_end = split // 2, min(split, split // 2 + 70)
    docs = [0] * split + [1] * (seq - split)
    groups = [-1] * seq
    groups[g_start:g_end] = [0] * (g_end - g_start)
    head_end = min(seq, split + 40)
    groups[split:head_end] = [0] * (head_end - split)
    return docs, groups, g_start, g_end


def _splits(seq: int) -> list[int]:
    """Per-row document boundaries: different rows, so the mask must index the batch."""
    return [(seq * 2) // 3, seq // 3]


def _paired_groups(seq: int, id_a: int, id_b: int) -> torch.Tensor:
    """Group rows over the ``_splits`` documents: ``id_a`` and ``id_b`` on two disjoint
    spans of document 0 (``id_b`` crosses the block edge in the longer row), and
    ``id_b`` reused at the head of document 1."""
    rows = []
    for split in _splits(seq):
        row = [-1] * seq
        a_lo, a_hi = split * 35 // 100, split * 58 // 100
        b_lo, b_hi = split * 65 // 100, split * 88 // 100
        row[a_lo:a_hi] = [id_a] * (a_hi - a_lo)
        row[b_lo:b_hi] = [id_b] * (b_hi - b_lo)
        row[split : split + 20] = [id_b] * 20
        rows.append(row)
    return torch.tensor(rows)


def _relative_errors(actual: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
    """Largest elementwise error as a fraction of full scale, and relative norm."""
    diff = actual.float() - reference.float()
    scale = reference.float().abs().max()
    return (diff.abs().max() / scale).item(), (diff.norm() / reference.float().norm()).item()


def _assert_grads_match(model_a: Transformer, model_b: Transformer, bound: float) -> None:
    grads_a = {n: p.grad for n, p in model_a.named_parameters() if p.grad is not None}
    grads_b = {n: p.grad for n, p in model_b.named_parameters() if p.grad is not None}
    assert grads_a.keys() == grads_b.keys()
    for name, reference in grads_a.items():
        actual = grads_b[name]
        assert torch.isfinite(actual).all(), f"non-finite grad for {name}"
        if reference.abs().max() == 0:
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            continue
        elementwise, relative_norm = _relative_errors(actual, reference)
        assert elementwise < bound, f"{name}: elementwise {elementwise:.2e}"
        assert relative_norm < bound, f"{name}: relative norm {relative_norm:.2e}"


@pytest.mark.gpu
class TestFlexGroups:
    SEQ = 256

    def _model(self, backend: str, seq: int | None = None, **overrides) -> Transformer:
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

    def _batch(self, seq: int | None = None, seed: int = 0):
        seq = seq or self.SEQ
        gen = torch.Generator(device="cuda").manual_seed(seed)
        tokens = torch.randint(0, 256, (2, seq), device="cuda", generator=gen)
        rows = [_spans(seq, split) for split in _splits(seq)]
        doc_ids = torch.tensor([r[0] for r in rows], device="cuda")
        group_ids = torch.tensor([r[1] for r in rows], device="cuda")
        return tokens, doc_ids, group_ids

    @pytest.mark.parametrize("overrides", [{}, {"n_kv_heads": 2}], ids=["mha", "gqa"])
    def test_forward_matches_sdpa(self, overrides):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        tokens, doc_ids, group_ids = self._batch()
        with torch.no_grad():
            out = {
                backend: self._model(backend, **overrides).eval()(
                    tokens, doc_ids=doc_ids, attention_group_ids=group_ids
                )
                for backend in ("sdpa", "flex")
            }
        torch.testing.assert_close(out["flex"], out["sdpa"], rtol=2e-5, atol=2e-5)

    @pytest.mark.parametrize("overrides", [{}, {"n_kv_heads": 2}], ids=["mha", "gqa"])
    def test_gradients_match_sdpa(self, overrides):
        """Backward parity, relative to each parameter's own gradient scale."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        tokens, doc_ids, group_ids = self._batch()
        models = {}
        for backend in ("sdpa", "flex"):
            model = self._model(backend, **overrides)
            model(tokens, doc_ids=doc_ids, attention_group_ids=group_ids).sum().backward()
            models[backend] = model
        _assert_grads_match(models["sdpa"], models["flex"], bound=1e-5)

    @pytest.mark.parametrize(
        ("id_a", "id_b"),
        [(2**31, 1), (2**31 + 1, 2**31 + 2), (5, 2**32 + 5)],
        ids=["2**31", "2**31+small", "low-high-collision"],
    )
    def test_ids_beyond_int32_match_dense_reference(self, id_a, id_b):
        """Group ids that a 32-bit cast would corrupt, against the dense int64 reference.

        Two groups share a document. A cast to int32 makes every id from 2**31 up
        negative, so the group vanishes, and folds 5 and 2**32 + 5 into one group.
        The compiled flex path, eager and under an outer torch.compile, must match
        the dense reference, and the reference must differ from each corrupted
        labelling, so that the parity is informative.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        from torch._dynamo.utils import counters

        torch._dynamo.reset()
        counters.clear()
        tokens, doc_ids, _ = self._batch()
        group_ids = _paired_groups(self.SEQ, id_a, id_b).cuda()
        sdpa = self._model("sdpa").eval()
        flex = self._model("flex").eval()

        def run(model, ids):
            return model(tokens, doc_ids=doc_ids, attention_group_ids=ids)

        with torch.no_grad():
            reference = run(sdpa, group_ids)
            torch.testing.assert_close(run(flex, group_ids), reference, rtol=2e-5, atol=2e-5)
            kernel_graphs = counters["stats"]["unique_graphs"]
            assert kernel_graphs > 0, "flex_attention did not compile"
            compiled = torch.compile(flex)
            torch.testing.assert_close(run(compiled, group_ids), reference, rtol=2e-5, atol=2e-5)
            assert counters["stats"]["unique_graphs"] > kernel_graphs, "torch.compile did not run"
            corrupted = {
                "first group ungrouped": group_ids.masked_fill(group_ids == id_a, -1),
                "second group ungrouped": group_ids.masked_fill(group_ids == id_b, -1),
                "groups merged": group_ids.masked_fill(group_ids == id_b, id_a),
            }
            for label, ids in corrupted.items():
                other = run(sdpa, ids)
                assert not torch.allclose(other, reference, rtol=2e-5, atol=2e-5), label

    @pytest.mark.parametrize("seq", [FLEX_BLOCK_SIZE, 200, 300])
    def test_compiled_lengths_around_the_block_size(self, seq):
        """Exactly one block, and ragged lengths whose groups cross a block edge.

        Under an outer torch.compile, per row: the compiled model matches eager,
        document 1 ignores document 0 despite the reused group id, the group's
        first member sees its last (a block above the diagonal at 200 and 300),
        and tokens before the group see neither.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        model = self._model("flex", seq=seq).eval()
        compiled = torch.compile(model)
        tokens, doc_ids, group_ids = self._batch(seq=seq)

        def run(fn, toks):
            return fn(toks, doc_ids=doc_ids, attention_group_ids=group_ids)

        with torch.no_grad():
            eager_out = run(model, tokens)
            comp_out = run(compiled, tokens)
            torch.testing.assert_close(comp_out, eager_out, rtol=2e-5, atol=2e-5)
            for row, split in enumerate(_splits(seq)):
                _, _, g_start, g_end = _spans(seq, split)
                doc0_moved = tokens.clone()
                doc0_moved[row, :split] = (doc0_moved[row, :split] + 1) % 256
                last_moved = tokens.clone()
                last_moved[row, g_end - 1] = (last_moved[row, g_end - 1] + 1) % 256
                out_doc0 = run(compiled, doc0_moved)[row]
                out_last = run(compiled, last_moved)[row]
                base = comp_out[row]
                torch.testing.assert_close(out_doc0[split:], base[split:], rtol=0, atol=0)
                torch.testing.assert_close(out_last[:g_start], base[:g_start], rtol=0, atol=0)
                assert not torch.allclose(out_last[g_start], base[g_start])

    def test_no_recompilation_across_group_layouts(self):
        """A fresh group layout per step must not retrigger compilation."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        from torch._dynamo.utils import counters

        torch._dynamo.reset()
        counters.clear()

        model = self._model("flex")
        tokens, doc_ids, _ = self._batch()

        def step(layout_seed: int):
            gen = torch.Generator().manual_seed(layout_seed)
            starts = torch.randint(0, self.SEQ - 48, (2, 2), generator=gen)
            group_ids = torch.full((2, self.SEQ), -1, dtype=torch.long)
            for row in range(2):
                for gid, start in enumerate(starts[row].tolist()):
                    group_ids[row, start : start + 40] = gid
            model(
                tokens, doc_ids=doc_ids, attention_group_ids=group_ids.to("cuda")
            ).sum().backward()
            model.zero_grad(set_to_none=True)

        step(0)
        step(1)
        after_warmup = counters["stats"]["unique_graphs"]
        assert after_warmup > 0, "flex_attention did not compile at all"

        for layout_seed in range(2, 6):
            step(layout_seed)

        assert counters["stats"]["unique_graphs"] == after_warmup, (
            f"recompiled {counters['stats']['unique_graphs'] - after_warmup} time(s) across "
            "group layouts"
        )

    def test_pack_16384_matches_dense_reference(self):
        """A production-length pack against the dense-mask SDPA reference.

        Six documents with block-unaligned boundaries, a pad tail under its own
        sentinel document id, and group ids 0 and 1 reused in every document at
        unaligned offsets. Forward and backward parity, then exact isolation and
        visibility checks on the flex model. Measured on H200: logits 5.9e-7 and
        gradients 8.6e-7 of full scale at worst.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        seq = 16384
        bounds = [0, 1000, 3333, 6000, 9001, 12345, 16000, seq]  # last span: pad tail
        doc_row = torch.empty(seq, dtype=torch.long)
        group_row = torch.full((seq,), -1, dtype=torch.long)
        for d, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
            pad = hi == seq
            doc_row[lo:hi] = -1 if pad else d
            if not pad:
                group_row[lo + 7 : lo + 307] = 0
                group_row[lo + 400 : lo + 529] = 1
        doc_ids = doc_row.unsqueeze(0).cuda()
        group_ids = group_row.unsqueeze(0).cuda()
        gen = torch.Generator(device="cuda").manual_seed(0)
        tokens = torch.randint(0, 256, (1, seq), device="cuda", generator=gen)

        models, logits = {}, {}
        for backend in ("sdpa", "flex"):
            model = self._model(backend, seq=seq)
            out = model(tokens, doc_ids=doc_ids, attention_group_ids=group_ids)
            out.float().logsumexp(dim=-1).mean().backward()
            models[backend], logits[backend] = model, out.detach()
        assert torch.isfinite(logits["flex"]).all()
        elementwise, relative_norm = _relative_errors(logits["flex"], logits["sdpa"])
        assert elementwise < 1e-5 and relative_norm < 1e-5, (elementwise, relative_norm)
        _assert_grads_match(models["sdpa"], models["flex"], bound=1e-4)

        flex = models["flex"].eval()

        def moved_at(positions: slice) -> torch.Tensor:
            perturbed = tokens.clone()
            perturbed[:, positions] = (perturbed[:, positions] + 1) % 256
            with torch.no_grad():
                return flex(perturbed, doc_ids=doc_ids, attention_group_ids=group_ids)[0]

        with torch.no_grad():
            base = flex(tokens, doc_ids=doc_ids, attention_group_ids=group_ids)[0]
        # Document 1 moves; every other document, and the pad tail, does not.
        out = moved_at(slice(1000, 3333))
        torch.testing.assert_close(out[:1000], base[:1000], rtol=0, atol=0)
        torch.testing.assert_close(out[3333:], base[3333:], rtol=0, atol=0)
        # In document 2 (from 3333): a late group-0 member reaches the first
        # member, but not the tokens ahead of the group.
        out = moved_at(slice(3333 + 290, 3333 + 291))
        torch.testing.assert_close(out[: 3333 + 7], base[: 3333 + 7], rtol=0, atol=0)
        assert not torch.allclose(out[3333 + 7], base[3333 + 7])
        # A group-1 member reaches group 1's first member, not group 0 before it.
        out = moved_at(slice(3333 + 450, 3333 + 451))
        torch.testing.assert_close(out[: 3333 + 400], base[: 3333 + 400], rtol=0, atol=0)
        assert not torch.allclose(out[3333 + 400], base[3333 + 400])
