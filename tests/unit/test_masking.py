"""Unit tests for attention mask construction.

``kempnerforge.model.masking`` owns every visibility rule, materializes one
mask per forward in the form the configured backend consumes, and
``Attention.forward`` takes that one object. These tests pin all three: the
mask predicate itself (against hand tables and an independent oracle), forward
parity between the flex and dense forms, and the shape of the seam -- the one
mask argument, the single construction shared across layers, and the semantic
ids staying at the ``Transformer`` boundary.

FlexAttention has no CPU backward in torch 2.11, so everything here is
forward-only; gradient parity is covered on GPU in
``tests/integration/test_compile.py``.
"""

from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, create_mask

from kempnerforge.config.schema import ModelConfig
from kempnerforge.config.vlm import MoMaConfig, MoTConfig
from kempnerforge.model import masking
from kempnerforge.model import transformer as transformer_module
from kempnerforge.model.attention import Attention, KVCache
from kempnerforge.model.masking import (
    build_attention_mask,
    build_dense_attention_mask,
    build_doc_causal_block_mask,
    flex_attention_fn,
)
from kempnerforge.model.modality import ModalityContext
from kempnerforge.model.mot import MoTAttention, MoTBlock
from kempnerforge.model.transformer import Transformer, TransformerBlock

CPU = torch.device("cpu")


def dense_doc_causal(doc_ids: torch.Tensor) -> torch.Tensor:
    """Reference mask: causal AND same-document."""
    seq_len = doc_ids.shape[1]
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool).tril()
    return causal.unsqueeze(0) & (doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1))


def elementwise_mask(doc_ids: torch.Tensor) -> torch.Tensor:
    """Expand the BlockMask's predicate to a dense (B, S, S) bool tensor."""
    batch, seq_len = doc_ids.shape
    block_mask = build_doc_causal_block_mask(doc_ids, CPU)
    return create_mask(
        block_mask.mask_mod, B=batch, H=1, Q_LEN=seq_len, KV_LEN=seq_len, device=CPU
    )[:, 0]


class TestBlockMaskStructure:
    @pytest.mark.parametrize(
        "doc_ids",
        [
            pytest.param([[0, 0, 0, 0, 0, 0, 0, 0]], id="single_document"),
            pytest.param([[0, 0, 0, 1, 1, 1, 1, 1]], id="two_documents"),
            pytest.param([[0, 0, 1, 1, 1, 2, 2, 2]], id="three_documents"),
            pytest.param([[0, 1, 2, 3, 4, 5, 6, 7]], id="every_token_its_own_document"),
            pytest.param([[0, 0, 0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 2, 2, 2, 2]], id="batched"),
        ],
    )
    def test_matches_dense_reference(self, doc_ids):
        ids = torch.tensor(doc_ids)
        assert torch.equal(elementwise_mask(ids), dense_doc_causal(ids))

    def test_single_document_is_plain_causal(self):
        ids = torch.zeros(1, 16, dtype=torch.long)
        expected = torch.ones(16, 16, dtype=torch.bool).tril()
        assert torch.equal(elementwise_mask(ids)[0], expected)

    def test_every_token_attends_to_itself(self):
        """No query row is ever fully masked, so softmax cannot produce NaN."""
        ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
        assert elementwise_mask(ids)[0].diagonal().all()
        assert elementwise_mask(ids).any(dim=-1).all()

    @pytest.mark.parametrize("seq_len", [7, 63, 129, 200])
    def test_sequence_length_not_a_multiple_of_block_size(self, seq_len):
        """BLOCK_SIZE is 128; ragged lengths must still mask exactly."""
        ids = torch.zeros(1, seq_len, dtype=torch.long)
        ids[0, seq_len // 2 :] = 1
        assert torch.equal(elementwise_mask(ids), dense_doc_causal(ids))

    def test_accepts_int64_doc_ids_from_the_dataset(self):
        """_compute_packed_output emits int64; the builder casts to int32 internally."""
        ids = torch.tensor([[0, 0, 1, 1]], dtype=torch.int64)
        assert build_doc_causal_block_mask(ids, CPU) is not None
        assert torch.equal(elementwise_mask(ids), dense_doc_causal(ids))


class TestFlexForwardParity:
    """Flex output vs the dense-mask SDPA it replaces, in fp64 for a tight bound."""

    def _qkv(self, batch, n_heads, seq_len, head_dim, seed=0):
        gen = torch.Generator().manual_seed(seed)
        shape = (batch, n_heads, seq_len, head_dim)
        return tuple(torch.randn(shape, dtype=torch.float64, generator=gen) for _ in range(3))

    def test_matches_dense_mask_sdpa(self):
        doc_ids = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2]])
        q, k, v = self._qkv(1, 4, 8, 16)
        block_mask = build_doc_causal_block_mask(doc_ids, CPU)
        out_flex = flex_attention_fn(False)(q, k, v, block_mask=block_mask, scale=16**-0.5)
        out_sdpa = F.scaled_dot_product_attention(
            q, k, v, attn_mask=dense_doc_causal(doc_ids).unsqueeze(1)
        )
        torch.testing.assert_close(out_flex, out_sdpa, rtol=1e-12, atol=1e-12)

    def test_single_document_matches_is_causal_sdpa(self):
        doc_ids = torch.zeros(1, 8, dtype=torch.long)
        q, k, v = self._qkv(1, 4, 8, 16, seed=1)
        block_mask = build_doc_causal_block_mask(doc_ids, CPU)
        out_flex = flex_attention_fn(False)(q, k, v, block_mask=block_mask, scale=16**-0.5)
        out_sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.testing.assert_close(out_flex, out_sdpa, rtol=1e-12, atol=1e-12)

    def test_enable_gqa_matches_manual_kv_expansion(self):
        """Licenses skipping repeat_interleave on the flex path (attention.py)."""
        doc_ids = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2]])
        gen = torch.Generator().manual_seed(2)
        q = torch.randn(1, 4, 8, 16, dtype=torch.float64, generator=gen)
        k = torch.randn(1, 2, 8, 16, dtype=torch.float64, generator=gen)
        v = torch.randn(1, 2, 8, 16, dtype=torch.float64, generator=gen)
        block_mask = build_doc_causal_block_mask(doc_ids, CPU)
        flex = flex_attention_fn(False)
        out_gqa = flex(q, k, v, block_mask=block_mask, scale=16**-0.5, enable_gqa=True)
        out_expanded = flex(
            q,
            k.repeat_interleave(2, dim=1),
            v.repeat_interleave(2, dim=1),
            block_mask=block_mask,
            scale=16**-0.5,
        )
        torch.testing.assert_close(out_gqa, out_expanded, rtol=0, atol=0)


class TestFlexAttentionFn:
    def test_returns_the_same_callable_for_repeated_calls(self):
        """Cached so the compiled artifact is built once, not once per forward."""
        assert flex_attention_fn(False) is flex_attention_fn(False)

    def test_cpu_path_is_uncompiled(self):
        from torch.nn.attention.flex_attention import flex_attention

        assert flex_attention_fn(False) is flex_attention


def visibility_oracle(
    seq_len: int,
    doc_ids: list[int] | None = None,
    group_ids: list[int] | None = None,
    key_padding: list[bool] | None = None,
) -> torch.Tensor:
    """The mask spec, evaluated pair by pair in plain Python.

    ``visible(q, k) = same_document AND (k <= q OR same_nonnegative_group)
    AND valid_key(k)``, then any query row left with nothing visible is
    unmasked so softmax cannot produce NaN.
    """
    rows = []
    for q in range(seq_len):
        row = []
        for k in range(seq_len):
            same_doc = doc_ids is None or doc_ids[q] == doc_ids[k]
            same_group = (
                group_ids is not None and group_ids[q] >= 0 and group_ids[q] == group_ids[k]
            )
            valid = key_padding is None or key_padding[k]
            row.append(same_doc and (k <= q or same_group) and valid)
        rows.append(row if any(row) else [True] * seq_len)
    return torch.tensor(rows, dtype=torch.bool)


HAND_DOCS = [0, 0, 1, 1]
HAND_GROUPS = [-1, 0, 0, -1]
HAND_PADDING = [True, False, True, True]
HAND_TABLES = {
    "causal": ({}, [{0}, {0, 1}, {0, 1, 2}, {0, 1, 2, 3}]),
    "documents": ({"doc_ids": HAND_DOCS}, [{0}, {0, 1}, {2}, {2, 3}]),
    "groups": ({"group_ids": HAND_GROUPS}, [{0}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2, 3}]),
    "key_padding": ({"key_padding": HAND_PADDING}, [{0}, {0}, {0, 2}, {0, 2, 3}]),
    "documents+groups": (
        {"doc_ids": HAND_DOCS, "group_ids": HAND_GROUPS},
        [{0}, {0, 1}, {2}, {2, 3}],
    ),
    "documents+key_padding": (
        {"doc_ids": HAND_DOCS, "key_padding": HAND_PADDING},
        [{0}, {0}, {2}, {2, 3}],
    ),
    "groups+key_padding": (
        {"group_ids": HAND_GROUPS, "key_padding": HAND_PADDING},
        [{0}, {0, 2}, {0, 2}, {0, 2, 3}],
    ),
    "documents+groups+key_padding": (
        {"doc_ids": HAND_DOCS, "group_ids": HAND_GROUPS, "key_padding": HAND_PADDING},
        [{0}, {0}, {2}, {2, 3}],
    ),
}


def _from_visible(visible: list[set[int]]) -> torch.Tensor:
    seq_len = len(visible)
    return torch.tensor([[k in keys for k in range(seq_len)] for keys in visible])


def _long(values: list[int] | None) -> torch.Tensor | None:
    return None if values is None else torch.tensor([values], dtype=torch.long)


def _bool(values: list[bool] | None) -> torch.Tensor | None:
    return None if values is None else torch.tensor([values], dtype=torch.bool)


def _dense(seq_len: int, batch: int = 1, **ids) -> torch.Tensor | None:
    return build_dense_attention_mask(
        batch=batch,
        seq_len=seq_len,
        device=CPU,
        doc_ids=_long(ids.get("doc_ids")),
        group_ids=_long(ids.get("group_ids")),
        key_padding_mask=_bool(ids.get("key_padding")),
    )


class TestOracleIsPinnedByHand:
    """The oracle other tests compare against is itself checked against hand tables."""

    @pytest.mark.parametrize("name", sorted(HAND_TABLES))
    def test_matches_the_hand_table(self, name):
        ids, visible = HAND_TABLES[name]
        assert torch.equal(visibility_oracle(4, **ids), _from_visible(visible))


class TestDenseMaskConstructor:
    """``build_dense_attention_mask`` owns document, group and key-validity semantics."""

    @pytest.mark.parametrize("name", sorted(HAND_TABLES))
    def test_matches_the_hand_table(self, name):
        ids, visible = HAND_TABLES[name]
        mask = _dense(4, **ids)
        if not ids:
            assert mask is None
            return
        assert mask is not None
        assert mask.shape == (1, 1, 4, 4)
        assert torch.equal(mask[0, 0], _from_visible(visible))

    @pytest.mark.parametrize("doc_ids", [None, [0, 0, 0, 1, 1, 2, 2, 2], [0, 1, 2, 3, 4, 5, 6, 7]])
    @pytest.mark.parametrize("group_ids", [None, [-1] * 8, [2, -1, 2, -1, 0, 0, -1, 2]])
    @pytest.mark.parametrize(
        "key_padding", [None, [True] * 8, [True, False, True, True, False, True, True, True]]
    )
    def test_every_combination_matches_the_oracle(self, doc_ids, group_ids, key_padding):
        mask = _dense(8, doc_ids=doc_ids, group_ids=group_ids, key_padding=key_padding)
        expected = visibility_oracle(8, doc_ids, group_ids, key_padding)
        if doc_ids is None and group_ids is None and key_padding is None:
            assert mask is None
            assert torch.equal(expected, torch.ones(8, 8, dtype=torch.bool).tril())
            return
        assert mask is not None
        assert torch.equal(mask[0, 0], expected)

    def test_plain_causal_returns_none_to_keep_the_sdpa_fast_path(self):
        assert _dense(8) is None

    def test_a_fully_padded_row_is_unmasked_rather_than_nan(self):
        mask = _dense(4, key_padding=[False, False, False, False])
        assert mask is not None
        assert mask.all()

    def test_a_document_whose_only_keys_are_invalid_is_unmasked(self):
        mask = _dense(2, doc_ids=[0, 1], key_padding=[True, False])
        assert mask is not None
        assert torch.equal(mask[0, 0], torch.tensor([[True, False], [True, True]]))

    def test_batched_rows_are_masked_independently(self):
        doc_ids = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 2]])
        mask = build_dense_attention_mask(batch=2, seq_len=4, device=CPU, doc_ids=doc_ids)
        assert mask is not None
        assert torch.equal(mask[0, 0], visibility_oracle(4, [0, 0, 1, 1]))
        assert torch.equal(mask[1, 0], visibility_oracle(4, [0, 1, 1, 2]))

    def test_a_group_never_reaches_across_a_document(self):
        shared = _dense(4, doc_ids=[0, 1, 2, 3], group_ids=[7, 7, 7, 7])
        assert shared is not None
        assert torch.equal(shared[0, 0], torch.eye(4, dtype=torch.bool))

    def test_a_group_never_revives_an_invalid_key(self):
        mask = _dense(4, group_ids=[3, 3, 3, 3], key_padding=[True, False, True, True])
        assert mask is not None
        assert not mask[0, 0, :, 1].any()


class TestBuildAttentionMask:
    """The single entry point picks the backend form and nothing else."""

    DOCS = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
    GROUPS = torch.tensor([[-1, 2, 2, -1, -1, 5, 5, -1]])

    def _build(self, backend: str, **kwargs):
        return build_attention_mask(backend=backend, batch=1, seq_len=8, device=CPU, **kwargs)

    def test_flex_returns_a_block_mask_for_documents(self):
        assert isinstance(self._build("flex", doc_ids=self.DOCS), BlockMask)

    def test_flex_returns_a_block_mask_for_groups_without_documents(self):
        mask = self._build("flex", group_ids=self.GROUPS)
        assert isinstance(mask, BlockMask)
        expected = visibility_oracle(8, None, self.GROUPS[0].tolist())
        assert torch.equal(elementwise_mask_of(mask, 1, 8)[0], expected)

    def test_flex_falls_back_to_dense_when_keys_can_be_invalid(self):
        mask = self._build(
            "flex", doc_ids=self.DOCS, key_padding_mask=torch.ones(1, 8, dtype=torch.bool)
        )
        assert isinstance(mask, torch.Tensor)

    def test_sdpa_always_returns_the_dense_form(self):
        assert isinstance(self._build("sdpa", doc_ids=self.DOCS), torch.Tensor)
        assert isinstance(self._build("sdpa", group_ids=self.GROUPS), torch.Tensor)

    @pytest.mark.parametrize("backend", ["sdpa", "flex"])
    def test_no_ids_means_no_mask_on_either_backend(self, backend):
        assert self._build(backend) is None

    @pytest.mark.parametrize("backend", ["sdpa", "flex"])
    def test_both_backends_express_the_same_predicate(self, backend):
        mask = self._build(backend, doc_ids=self.DOCS, group_ids=self.GROUPS)
        expected = visibility_oracle(8, self.DOCS[0].tolist(), self.GROUPS[0].tolist())
        actual = (
            elementwise_mask_of(mask, 1, 8)[0] if isinstance(mask, BlockMask) else mask[0, 0]  # type: ignore[reportOptionalSubscript]
        )
        assert torch.equal(actual, expected)

    def test_rejects_doc_ids_shorter_than_the_residual_stream(self):
        with pytest.raises(
            ValueError, match=r"doc_ids shape \(1, 8\) does not match the residual shape \(1, 12\)"
        ):
            build_attention_mask(backend="flex", batch=1, seq_len=12, device=CPU, doc_ids=self.DOCS)

    def test_groups_without_documents_hand_no_document_ids_to_the_block_mask(self, monkeypatch):
        seen: list[object] = []
        real = masking.build_doc_causal_block_mask

        def spy(doc_ids, device, group_ids=None):
            seen.append(doc_ids)
            return real(doc_ids, device, group_ids)

        monkeypatch.setattr(masking, "build_doc_causal_block_mask", spy)
        self._build("flex", group_ids=self.GROUPS)
        assert len(seen) == 1
        assert seen[0] is None

    def test_the_block_mask_builder_takes_group_ids_alone(self):
        mask = build_doc_causal_block_mask(None, CPU, self.GROUPS)
        expected = visibility_oracle(8, None, self.GROUPS[0].tolist())
        assert torch.equal(elementwise_mask_of(mask, 1, 8)[0], expected)

    def test_the_block_mask_builder_needs_documents_or_groups(self):
        with pytest.raises(ValueError, match="doc_ids or group_ids"):
            build_doc_causal_block_mask(None, CPU)


def elementwise_mask_of(block_mask: BlockMask, batch: int, seq_len: int) -> torch.Tensor:
    """Expand a built ``BlockMask`` to a dense (B, S, S) bool tensor."""
    return create_mask(
        block_mask.mask_mod, B=batch, H=1, Q_LEN=seq_len, KV_LEN=seq_len, device=CPU
    )[:, 0]


class TestAttentionTakesExactlyOneMask:
    """The reviewer's contract: a new mask rule is a constructor input, not a parameter."""

    MASKLESS = ("self", "x", "rope_cos", "rope_sin", "kv_cache")

    def test_forward_signature_is_exactly_the_maskless_arguments_plus_one_mask(self):
        params = list(inspect.signature(Attention.forward).parameters)
        assert params == [*self.MASKLESS, "attn_mask"]

    @pytest.mark.parametrize(
        "retired", ["doc_ids", "attention_group_ids", "group_ids", "key_padding_mask", "block_mask"]
    )
    def test_semantic_ids_are_not_attention_parameters(self, retired):
        assert retired not in inspect.signature(Attention.forward).parameters

    def test_the_one_mask_argument_admits_both_backend_forms(self):
        annotation = inspect.signature(Attention.forward).parameters["attn_mask"].annotation
        assert "BlockMask" in annotation and "Tensor" in annotation

    def test_the_block_passes_the_same_one_mask_through(self):
        params = list(inspect.signature(TransformerBlock.forward).parameters)
        assert [p for p in params if "mask" in p or p.endswith("_ids")] == ["attn_mask"]

    @pytest.mark.parametrize(
        "forward", [MoTAttention.forward, MoTBlock.forward], ids=["attn", "block"]
    )
    def test_mot_takes_the_same_one_mask(self, forward):
        params = list(inspect.signature(forward).parameters)
        assert [p for p in params if "mask" in p or p.endswith("_ids")] == ["attn_mask"]

    def test_the_transformer_boundary_still_takes_semantic_ids(self):
        params = inspect.signature(Transformer.forward).parameters
        assert "doc_ids" in params
        assert "attention_group_ids" in params
        assert "attn_mask" not in params
        assert "block_mask" not in params


def _tiny_model(backend: str, n_layers: int = 3) -> Transformer:
    torch.manual_seed(0)
    return Transformer(
        ModelConfig(
            dim=64,
            n_layers=n_layers,
            n_heads=4,
            vocab_size=64,
            max_seq_len=16,
            attention_backend=backend,
        )
    ).eval()


class TestMaskBuiltOncePerForward:
    """One construction per forward, shared by every layer, on both backends."""

    DOCS = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1]])

    @staticmethod
    def _record_layer_masks(model: Transformer) -> list[object]:
        seen: list[object] = []
        for layer in model.layers.values():
            layer.attention.register_forward_pre_hook(  # type: ignore[reportAttributeAccessIssue]
                lambda _m, _a, kwargs: seen.append(kwargs["attn_mask"]), with_kwargs=True
            )
        return seen

    @pytest.mark.parametrize("backend", ["sdpa", "flex"])
    def test_constructed_once_and_the_same_object_reaches_every_layer(self, backend, monkeypatch):
        built: list[object] = []
        real = masking.build_attention_mask

        def spy(**kwargs):
            mask = real(**kwargs)
            built.append(mask)
            return mask

        monkeypatch.setattr(transformer_module, "build_attention_mask", spy)
        model = _tiny_model(backend)
        seen = self._record_layer_masks(model)
        with torch.no_grad():
            model(torch.zeros(1, 8, dtype=torch.long), doc_ids=self.DOCS)

        assert len(built) == 1
        assert len(seen) == len(model.layers)
        assert all(mask is built[0] for mask in seen)
        assert built[0] is not None

    @pytest.mark.parametrize("backend", ["sdpa", "flex"])
    def test_the_constructors_themselves_run_once(self, backend, monkeypatch):
        calls: list[str] = []
        for name in ("build_dense_attention_mask", "build_doc_causal_block_mask"):
            real = getattr(masking, name)
            monkeypatch.setattr(
                masking,
                name,
                lambda *a, _n=name, _f=real, **kw: (calls.append(_n), _f(*a, **kw))[1],
            )
        model = _tiny_model(backend)
        with torch.no_grad():
            model(torch.zeros(1, 8, dtype=torch.long), doc_ids=self.DOCS)
        assert len(calls) == 1

    def test_an_unmasked_forward_builds_nothing_and_stays_on_the_fast_path(self):
        model = _tiny_model("sdpa")
        seen = self._record_layer_masks(model)
        with torch.no_grad():
            model(torch.zeros(1, 8, dtype=torch.long))
        assert seen == [None] * len(model.layers)


class TestKVCacheDecode:
    """A cache grows the key axis every step, so the mask is re-derived, not shared."""

    @staticmethod
    def _caches(model: Transformer, max_seq_len: int, batch: int = 1) -> list[KVCache]:
        return [
            KVCache(
                batch_size=batch,
                max_seq_len=max_seq_len,
                n_kv_heads=model.config.n_kv_heads,  # type: ignore[reportArgumentType]
                head_dim=model.config.head_dim,
                dtype=torch.float32,
                device=CPU,
            )
            for _ in range(len(model.layers))
        ]

    def test_stepwise_decode_matches_one_full_forward(self):
        model = _tiny_model("sdpa", n_layers=2)
        tokens = torch.randint(0, 64, (1, 6), generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            full = model(tokens)
            caches = self._caches(model, max_seq_len=16)
            stepped = [
                model(tokens[:, : i + 1] if i == 0 else tokens[:, i : i + 1], kv_caches=caches)
                for i in range(6)
            ]
        for position, logits in enumerate(stepped):
            torch.testing.assert_close(logits[:, -1], full[:, position], rtol=1e-5, atol=1e-5)

    def test_every_decode_step_sees_no_shared_mask_and_a_longer_key_axis(self):
        model = _tiny_model("sdpa", n_layers=1)
        layer = next(iter(model.layers.values()))
        seen: list[tuple[object, int]] = []
        layer.attention.register_forward_pre_hook(  # type: ignore[reportAttributeAccessIssue]
            lambda _m, _a, kwargs: seen.append((kwargs["attn_mask"], kwargs["kv_cache"].seq_len)),
            with_kwargs=True,
        )
        tokens = torch.randint(0, 64, (1, 5), generator=torch.Generator().manual_seed(1))
        caches = self._caches(model, max_seq_len=16)
        with torch.no_grad():
            for i in range(5):
                model(tokens[:, i : i + 1], kv_caches=caches)
        assert [mask for mask, _ in seen] == [None] * 5
        assert [length for _, length in seen] == [0, 1, 2, 3, 4]

    @pytest.mark.parametrize("capture", [False, True], ids=["fused", "capture_weights"])
    @pytest.mark.parametrize(
        "chunks", [(3, 2, 3), (1, 7), (5, 1, 2)], ids=["3-2-3", "1-7", "5-1-2"]
    )
    def test_chunks_appended_to_a_filled_cache_match_one_full_forward(self, capture, chunks):
        """Each appended token sees every cached key and the new keys up to its own position."""
        torch.manual_seed(0)
        model = Transformer(
            ModelConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=64, max_seq_len=16)
        ).eval()
        for layer in model.layers.values():
            layer.attention.capture_attention_weights = capture
        tokens = torch.randint(0, 64, (2, 8), generator=torch.Generator().manual_seed(2))
        caches = self._caches(model, max_seq_len=16, batch=2)
        with torch.no_grad():
            full = model(tokens)
            parts, start = [], 0
            for length in chunks:
                parts.append(model(tokens[:, start : start + length], kv_caches=caches))
                start += length
        torch.testing.assert_close(torch.cat(parts, dim=1), full, rtol=1e-5, atol=1e-5)
        assert caches[0].seq_len == 8

    def test_captured_weights_of_an_appended_chunk_are_the_full_forwards_rows(self):
        """Captured weights of an appended chunk equal the full forward's rows for those tokens."""
        model = _tiny_model("sdpa", n_layers=1)
        attention = next(iter(model.layers.values())).attention
        attention.capture_attention_weights = True
        tokens = torch.randint(0, 64, (1, 8), generator=torch.Generator().manual_seed(3))
        caches = self._caches(model, max_seq_len=16)
        with torch.no_grad():
            model(tokens)
            full = attention.last_attention_weights
            model(tokens[:, :5], kv_caches=caches)
            model(tokens[:, 5:], kv_caches=caches)
            appended = attention.last_attention_weights
        assert appended.shape == (1, 4, 3, 8)
        torch.testing.assert_close(appended, full[:, :, 5:, :], rtol=1e-5, atol=1e-6)

    @staticmethod
    def _capturing_model(capture: bool) -> tuple[Transformer, list[None]]:
        """A one-layer model whose attention records every call it receives."""
        model = _tiny_model("sdpa", n_layers=1)
        attention = next(iter(model.layers.values())).attention
        attention.capture_attention_weights = capture  # type: ignore[reportAttributeAccessIssue]
        calls: list[None] = []
        attention.register_forward_pre_hook(lambda *_: calls.append(None))  # type: ignore[reportAttributeAccessIssue]
        return model, calls

    @pytest.mark.parametrize("capture", [False, True], ids=["fused", "capture_weights"])
    def test_doc_ids_are_refused_with_a_cache_before_any_compute(self, capture):
        model, calls = self._capturing_model(capture)
        caches = self._caches(model, max_seq_len=16)
        with pytest.raises(ValueError, match="doc_ids is training-only"):
            model(
                torch.zeros(1, 4, dtype=torch.long),
                kv_caches=caches,
                doc_ids=torch.zeros(1, 4, dtype=torch.long),
            )
        assert calls == []
        assert caches[0].seq_len == 0

    @pytest.mark.parametrize("capture", [False, True], ids=["fused", "capture_weights"])
    def test_a_key_padding_mask_is_refused_with_a_cache_before_any_compute(self, capture):
        """The mask is square over the residual stream, so it cannot follow a
        cache's growing key axis; it is refused before any layer runs, on the
        weight-capture path as on the fused one."""
        model, calls = self._capturing_model(capture)
        caches = self._caches(model, max_seq_len=16)
        padding = torch.tensor([[True, False, True, True]])
        modality = ModalityContext(inputs_embeds=torch.randn(1, 4, 64), key_padding_mask=padding)
        with pytest.raises(ValueError, match="modality.key_padding_mask is training-only"):
            model(modality=modality, kv_caches=caches)
        assert calls == []
        assert caches[0].seq_len == 0


def _moma_model(backend: str) -> Transformer:
    torch.manual_seed(0)
    return Transformer(
        ModelConfig(
            dim=64,
            n_layers=2,
            n_heads=4,
            vocab_size=64,
            max_seq_len=16,
            attention_backend=backend,
        ),
        vlm_config=MoMaConfig(
            moma_experts_per_modality={"image": 2, "text": 2}, moma_gumbel_noise=False
        ),
        num_image_tokens=0,
    ).eval()


def _mot_model(backend: str) -> Transformer:
    torch.manual_seed(0)
    return Transformer(
        ModelConfig(
            dim=64,
            n_layers=1,
            n_heads=4,
            vocab_size=64,
            max_seq_len=16,
            attention_backend=backend,
        ),
        vlm_config=MoTConfig(max_text_len=8),
        num_image_tokens=4,
    ).eval()


class TestMaskInputsAreValidatedAgainstTheBatch:
    """A mask input that merely broadcasts is rejected, not silently reused.

    Every id tensor is indexed per (example, position). A ``(1, seq_len)`` input
    in a batch of 2 broadcasts cleanly through the dense mask and applies one
    example's documents, groups or key validity to the whole batch.
    """

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("doc_ids", torch.zeros(1, 8, dtype=torch.long)),
            ("group_ids", torch.zeros(1, 8, dtype=torch.long)),
            ("key_padding_mask", torch.ones(1, 8, dtype=torch.bool)),
        ],
    )
    @pytest.mark.parametrize("backend", ["sdpa", "flex"])
    def test_the_constructor_rejects_an_input_that_only_broadcasts(self, name, value, backend):
        with pytest.raises(ValueError, match="does not match the residual shape"):
            build_attention_mask(backend=backend, batch=2, seq_len=8, device=CPU, **{name: value})

    @pytest.mark.parametrize("backend", ["sdpa", "flex"])
    def test_one_forward_validates_its_mask_inputs_once(self, backend, monkeypatch):
        calls: list[None] = []
        real = masking._validate_mask_inputs

        def spy(**kwargs):
            calls.append(None)
            real(**kwargs)

        monkeypatch.setattr(masking, "_validate_mask_inputs", spy)
        model = _tiny_model(backend, n_layers=2)
        with torch.no_grad():
            model(
                torch.zeros(1, 8, dtype=torch.long),
                doc_ids=torch.zeros(1, 8, dtype=torch.long),
                attention_group_ids=torch.zeros(1, 8, dtype=torch.long),
            )
        assert len(calls) == 1

    def test_a_shape_error_reads_the_same_from_the_forward_and_the_constructor(self):
        bad = torch.zeros(1, 7, dtype=torch.long)
        with pytest.raises(ValueError) as from_forward:
            _tiny_model("sdpa", n_layers=1)(
                torch.zeros(1, 8, dtype=torch.long), attention_group_ids=bad
            )
        with pytest.raises(ValueError) as from_constructor:
            build_attention_mask(backend="sdpa", batch=1, seq_len=8, device=CPU, group_ids=bad)
        assert str(from_forward.value) == str(from_constructor.value)

    @pytest.mark.parametrize("backend", ["sdpa", "flex"])
    def test_short_doc_ids_on_a_text_model_name_the_residual_shape(self, backend):
        model = _tiny_model(backend, n_layers=1)
        with pytest.raises(
            ValueError, match=r"doc_ids shape \(1, 7\) does not match the residual shape \(1, 8\)"
        ) as caught:
            model(torch.zeros(1, 8, dtype=torch.long), doc_ids=torch.zeros(1, 7, dtype=torch.long))
        assert "image" not in str(caught.value)

    def test_a_transformer_forward_rejects_doc_ids_for_one_example_of_a_batch(self):
        model = _tiny_model("sdpa", n_layers=1)
        tokens = torch.randint(0, 64, (2, 8), generator=torch.Generator().manual_seed(0))
        with pytest.raises(ValueError, match="does not match the residual shape"):
            model(tokens, doc_ids=torch.zeros(1, 8, dtype=torch.long))

    def test_a_transformer_forward_rejects_a_key_padding_mask_for_one_example(self):
        model = _tiny_model("sdpa", n_layers=1)
        embeds = torch.randn(2, 8, 64, generator=torch.Generator().manual_seed(1))
        modality = ModalityContext(
            inputs_embeds=embeds, key_padding_mask=torch.ones(1, 8, dtype=torch.bool)
        )
        with pytest.raises(ValueError, match="does not match the residual shape"):
            model(modality=modality)


def _live_mot_model(backend: str, modalities: tuple[str, ...] = ("image", "text")) -> Transformer:
    """A two-layer MoT model with every weight redrawn, so no block is the identity."""
    torch.manual_seed(0)
    model = Transformer(
        ModelConfig(
            dim=64,
            n_layers=2,
            n_heads=4,
            vocab_size=64,
            max_seq_len=16,
            attention_backend=backend,
        ),
        vlm_config=MoTConfig(max_text_len=8, mot_modalities=modalities),
        num_image_tokens=4,
    ).eval()
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn(param.shape, generator=gen) * 0.2)
    return model


class TestMoTGetsTheSharedMask:
    """Every visibility rule reaches the MoT residual path on both backends.

    MoT blocks take the one mask every other block takes, so packed documents
    stay isolated and visibility groups apply. The residual is four image
    tokens then eight text tokens; document 1 is the last four text tokens.
    """

    PREFIX = torch.randn(1, 4, 64, generator=torch.Generator().manual_seed(2))
    TOKENS = torch.randint(0, 64, (1, 8), generator=torch.Generator().manual_seed(3))
    MODALITY_IDS = torch.tensor([[0] * 4 + [1] * 8])
    DOCS = torch.tensor([[0] * 8 + [1] * 4])
    GROUPS = torch.tensor([[0, 0, 0, 0, -1, -1, -1, -1, 1, 1, -1, -1]])

    def _forward(
        self,
        model: Transformer,
        tokens: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        **ids: torch.Tensor,
    ) -> torch.Tensor:
        modality = ModalityContext(
            prefix_embeds=self.PREFIX,
            modality_ids=self.MODALITY_IDS,
            key_padding_mask=key_padding_mask,
        )
        with torch.no_grad():
            return model(self.TOKENS if tokens is None else tokens, modality=modality, **ids)

    @staticmethod
    def _record_block_masks(model: Transformer) -> list[object]:
        seen: list[object] = []
        for layer in model.layers.values():
            layer.attn.register_forward_pre_hook(  # type: ignore[reportAttributeAccessIssue]
                lambda _m, _a, kwargs: seen.append(kwargs.get("attn_mask")), with_kwargs=True
            )
        return seen

    @pytest.mark.parametrize("backend", ["sdpa", "flex"])
    def test_packed_documents_are_isolated(self, backend):
        model = _live_mot_model(backend)
        moved = self.TOKENS.clone()
        moved[:, :4] = (moved[:, :4] + 1) % 64
        base = self._forward(model, doc_ids=self.DOCS)
        out = self._forward(model, moved, doc_ids=self.DOCS)
        torch.testing.assert_close(out[:, 8:], base[:, 8:], rtol=0, atol=0)
        assert not torch.allclose(out[:, 4:8], base[:, 4:8])

    @pytest.mark.parametrize("backend", ["sdpa", "flex"])
    def test_a_group_member_sees_a_later_member(self, backend):
        model = _live_mot_model(backend)
        moved = self.TOKENS.clone()
        moved[:, 5] = (moved[:, 5] + 1) % 64
        causal = [self._forward(model, t, doc_ids=self.DOCS) for t in (self.TOKENS, moved)]
        grouped = [
            self._forward(model, t, doc_ids=self.DOCS, attention_group_ids=self.GROUPS)
            for t in (self.TOKENS, moved)
        ]
        torch.testing.assert_close(causal[1][:, 8], causal[0][:, 8], rtol=0, atol=0)
        assert not torch.allclose(grouped[1][:, 8], grouped[0][:, 8])

    def test_flex_matches_sdpa_with_documents_and_groups(self):
        out = {
            backend: self._forward(
                _live_mot_model(backend), doc_ids=self.DOCS, attention_group_ids=self.GROUPS
            )
            for backend in ("sdpa", "flex")
        }
        torch.testing.assert_close(out["flex"], out["sdpa"], rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("backend", ["sdpa", "flex"])
    def test_every_mot_block_receives_the_one_shared_mask(self, backend):
        model = _live_mot_model(backend)
        seen = self._record_block_masks(model)
        self._forward(model, doc_ids=self.DOCS)
        assert len(seen) == len(model.layers)
        assert seen[0] is not None
        assert all(mask is seen[0] for mask in seen)

    @pytest.mark.parametrize("backend", ["sdpa", "flex"])
    def test_an_unmasked_forward_keeps_the_causal_fast_path(self, backend):
        model = _live_mot_model(backend)
        seen = self._record_block_masks(model)
        self._forward(model)
        assert seen == [None] * len(model.layers)

    @pytest.mark.parametrize("backend", ["sdpa", "flex"])
    def test_a_key_padding_mask_alone_is_the_causal_and_valid_mask(self, backend):
        """The same dense mask MoT built for itself from key validity alone."""
        padding = torch.ones(1, 12, dtype=torch.bool)
        padding[0, :2] = False
        padding[0, 7] = False
        expected = torch.ones(12, 12, dtype=torch.bool).tril().view(1, 1, 12, 12)
        expected = expected & padding.view(1, 1, 1, 12)
        expected = expected | ~expected.any(dim=-1, keepdim=True)
        model = _live_mot_model(backend)
        seen = self._record_block_masks(model)
        self._forward(model, key_padding_mask=padding)
        assert isinstance(seen[0], torch.Tensor)
        assert torch.equal(seen[0], expected)

    @pytest.mark.parametrize("backend", ["sdpa", "flex"])
    def test_rejects_doc_ids_shorter_than_the_residual_stream(self, backend):
        with pytest.raises(
            ValueError, match=r"doc_ids shape \(1, 8\) does not match the residual shape \(1, 12\)"
        ):
            self._forward(_live_mot_model(backend), doc_ids=torch.zeros(1, 8, dtype=torch.long))

    @pytest.mark.parametrize("name", ["doc_ids", "attention_group_ids"])
    def test_a_text_first_modality_order_refuses_documents_and_groups(self, name):
        """Its attention runs over text then image, so a residual-order mask would misalign."""
        model = _live_mot_model("sdpa", modalities=("text", "image"))
        ids = self.DOCS if name == "doc_ids" else self.GROUPS
        with pytest.raises(ValueError, match="mot_modalities"):
            self._forward(model, **{name: ids})

    def test_a_text_first_modality_order_still_runs_without_them(self):
        model = _live_mot_model("sdpa", modalities=("text", "image"))
        assert self._forward(model).shape == (1, 12, 64)


class TestInputsAreValidatedBeforeTheMaskIsBuilt:
    """Validate inputs, then build the mask, then run the layers.

    An arch that is missing its modality tags must say so, rather than dying
    inside the mask builder on an unrelated bad input.
    """

    @pytest.mark.parametrize(
        ("factory", "message"),
        [(_moma_model, "MoMa model requires"), (_mot_model, "MoT model requires")],
        ids=["moma", "mot"],
    )
    @pytest.mark.parametrize("backend", ["sdpa", "flex"])
    def test_a_missing_modality_tag_is_reported_before_a_bad_doc_ids(
        self, factory, message, backend
    ):
        model = factory(backend)
        tokens = torch.randint(0, 64, (1, 8), generator=torch.Generator().manual_seed(0))
        with pytest.raises(ValueError, match=message):
            model(tokens, doc_ids=torch.zeros(1, 4, dtype=torch.long))

    @pytest.mark.parametrize("backend", ["sdpa", "flex"])
    def test_a_misshaped_modality_tag_is_reported_before_a_bad_doc_ids(self, backend):
        model = _moma_model(backend)
        embeds = torch.randn(1, 8, 64, generator=torch.Generator().manual_seed(2))
        modality = ModalityContext(
            inputs_embeds=embeds, modality_ids=torch.zeros(1, 5, dtype=torch.long)
        )
        with pytest.raises(ValueError, match="modality.modality_ids shape"):
            model(modality=modality, doc_ids=torch.zeros(1, 4, dtype=torch.long))


class TestMoMaGetsTheSharedMask:
    """Document isolation reaches the MoMa residual path on both backends.

    MoMa blocks take the same one mask every other block takes, so selecting the
    flex backend cannot quietly drop the block-diagonal mask and leave MoMa
    attending across document boundaries.
    """

    DOCS = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]])
    GROUPS = torch.tensor([[0, 0, 0, -1, -1, -1, 1, 1, 1, -1, -1, -1]])

    @staticmethod
    def _forward(
        model: Transformer, doc_ids: torch.Tensor | None, group_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        embeds = torch.randn(1, 12, 64, generator=torch.Generator().manual_seed(1))
        modality = ModalityContext(
            inputs_embeds=embeds, modality_ids=torch.zeros(1, 12, dtype=torch.long)
        )
        with torch.no_grad():
            return model(modality=modality, doc_ids=doc_ids, attention_group_ids=group_ids)

    def test_flex_isolates_documents_exactly_as_sdpa_does(self):
        packed = {b: self._forward(_moma_model(b), self.DOCS) for b in ("sdpa", "flex")}
        torch.testing.assert_close(packed["flex"], packed["sdpa"], rtol=1e-5, atol=1e-5)

    def test_groups_reach_moma_on_both_backends(self):
        grouped = {
            b: self._forward(_moma_model(b), self.DOCS, self.GROUPS) for b in ("sdpa", "flex")
        }
        torch.testing.assert_close(grouped["flex"], grouped["sdpa"], rtol=1e-5, atol=1e-5)
        ungrouped = self._forward(_moma_model("sdpa"), self.DOCS)
        assert not torch.allclose(grouped["sdpa"], ungrouped, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("backend", ["sdpa", "flex"])
    def test_packing_actually_changes_the_moma_forward(self, backend):
        model = _moma_model(backend)
        assert not torch.allclose(
            self._forward(model, self.DOCS), self._forward(model, None), rtol=1e-5, atol=1e-5
        )

    @pytest.mark.parametrize("backend", ["sdpa", "flex"])
    def test_every_moma_block_receives_the_one_shared_mask(self, backend):
        model = _moma_model(backend)
        seen: list[object] = []
        for layer in model.layers.values():
            layer.attention.register_forward_pre_hook(  # type: ignore[reportAttributeAccessIssue]
                lambda _m, _a, kwargs: seen.append(kwargs["attn_mask"]), with_kwargs=True
            )
        self._forward(model, self.DOCS)
        assert len(seen) == len(model.layers)
        assert seen[0] is not None
        assert all(mask is seen[0] for mask in seen)


def test_dynamic_flex_shapes_toggle_swaps_and_restores_the_compiled_helpers():
    from kempnerforge.model import masking

    static = masking.flex_attention_fn(True)
    try:
        masking.use_dynamic_flex_shapes(True)
        assert masking.flex_attention_fn(True) is masking._FLEX_DYNAMIC
        assert masking._CREATE_BLOCK_MASK_COMPILED is masking._CREATE_BLOCK_MASK_DYNAMIC
        masking.use_dynamic_flex_shapes(True)
        assert masking.flex_attention_fn(True) is masking._FLEX_DYNAMIC
    finally:
        masking.use_dynamic_flex_shapes(False)
    assert masking.flex_attention_fn(True) is static
    assert masking._CREATE_BLOCK_MASK_COMPILED is masking._CREATE_BLOCK_MASK_STATIC
