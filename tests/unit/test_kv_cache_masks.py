"""Masked prefill and per-row key validity under a KV cache."""

from __future__ import annotations

import pytest
import torch

from kempnerforge.config.schema import ModelConfig
from kempnerforge.model.attention import KVCache
from kempnerforge.model.masking import build_cached_attention_mask
from kempnerforge.model.modality import ModalityContext
from kempnerforge.model.transformer import Transformer

CPU = torch.device("cpu")
LENGTHS = (10, 7)
GENERATED = 4


def _model(backend: str) -> Transformer:
    torch.manual_seed(0)
    config = ModelConfig(
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=64,
        max_seq_len=32,
        attention_backend=backend,
    )
    return Transformer(config).eval()


def _caches(model: Transformer, batch: int) -> list[KVCache]:
    return [
        KVCache(batch, 32, 2, model.config.head_dim, torch.float32, CPU)
        for _ in range(len(model.layers))
    ]


def _image(row: int, slots: int = 3) -> ModalityContext:
    """Three inline embeddings over the first slots of ``row``, in a batch of that row alone."""
    gen = torch.Generator().manual_seed(row)
    return ModalityContext(
        inline_embeds=torch.randn(slots, 64, generator=gen),
        inline_rows=torch.zeros(slots, dtype=torch.long),
        inline_positions=torch.arange(slots),
    )


class TestCachedDecodeMatchesUncached:
    """A padded batch prefilled with groups and images, then decoded, matches each row alone."""

    @pytest.mark.parametrize("backend", ["sdpa", "flex"])
    @pytest.mark.parametrize(
        "prefill_valid", [False, True], ids=["causal_prefill", "valid_prefill"]
    )
    def test_logits_match_each_row_run_uncached(self, backend, prefill_valid):
        model = _model(backend)
        gen = torch.Generator().manual_seed(1)
        rows = [torch.randint(1, 64, (n + GENERATED,), generator=gen) for n in LENGTHS]
        lengths = torch.tensor(LENGTHS)
        width = max(LENGTHS)

        with torch.no_grad():
            expected = []
            for index, row in enumerate(rows):
                groups = torch.full((1, row.numel()), -1)
                groups[0, :3] = index
                expected.append(
                    model(row.unsqueeze(0), modality=_image(index), attention_group_ids=groups)[0]
                )

            ids = torch.zeros(2, width, dtype=torch.long)
            groups = torch.full((2, width), -1)
            for index, (row, n) in enumerate(zip(rows, LENGTHS, strict=True)):
                ids[index, :n] = row[:n]
                groups[index, :3] = index
            images = [_image(index) for index in range(2)]
            context = ModalityContext(
                inline_embeds=torch.cat([image.inline_embeds for image in images]),  # type: ignore[reportArgumentType]
                inline_rows=torch.tensor([0, 0, 0, 1, 1, 1]),
                inline_positions=torch.tensor([0, 1, 2, 0, 1, 2]),
                output_rows=torch.arange(2),
                output_positions=lengths - 1,
            )
            caches = _caches(model, 2)
            prompt_keys = torch.arange(width).unsqueeze(0) < lengths.unsqueeze(1)
            logits = model(
                ids,
                modality=context,
                attention_group_ids=groups,
                kv_caches=caches,
                valid_keys=prompt_keys if prefill_valid else None,
            )
            for index, n in enumerate(LENGTHS):
                torch.testing.assert_close(logits[0, index], expected[index][n - 1])

            for step in range(GENERATED - 1):
                token = torch.stack([row[n + step] for row, n in zip(rows, LENGTHS, strict=True)])
                valid = torch.cat([prompt_keys, torch.ones(2, step + 1, dtype=torch.bool)], dim=1)
                logits = model(
                    token.unsqueeze(1),
                    position_ids=(lengths + step).unsqueeze(1),
                    kv_caches=caches,
                    valid_keys=valid,
                )
                for index, n in enumerate(LENGTHS):
                    torch.testing.assert_close(logits[index, 0], expected[index][n + step])

            keep = torch.tensor([1])
            for cache in caches:
                cache.keep_rows(keep)
            step = GENERATED - 1
            valid = torch.cat([prompt_keys[keep], torch.ones(1, step + 1, dtype=torch.bool)], dim=1)
            logits = model(
                rows[1][LENGTHS[1] + step].view(1, 1),
                position_ids=torch.tensor([[LENGTHS[1] + step]]),
                kv_caches=caches,
                valid_keys=valid,
            )
            torch.testing.assert_close(logits[0, 0], expected[1][LENGTHS[1] + step])


class TestRefusals:
    def test_groups_need_empty_caches(self):
        model = _model("sdpa")
        caches = _caches(model, 1)
        tokens = torch.randint(0, 64, (1, 4))
        groups = torch.zeros(1, 4, dtype=torch.long)
        with torch.no_grad():
            model(tokens, attention_group_ids=groups, kv_caches=caches)
            with pytest.raises(ValueError, match="filled kv_caches"):
                model(tokens, attention_group_ids=groups, kv_caches=caches)
        assert caches[0].seq_len == 4

    def test_valid_keys_need_caches(self):
        model = _model("sdpa")
        with pytest.raises(ValueError, match="valid_keys requires kv_caches"):
            model(
                torch.zeros(1, 4, dtype=torch.long), valid_keys=torch.ones(1, 4, dtype=torch.bool)
            )

    @pytest.mark.parametrize(
        "valid",
        [torch.ones(1, 5, dtype=torch.bool), torch.ones(1, 6, dtype=torch.long)],
        ids=["short", "not_bool"],
    )
    def test_valid_keys_cover_the_keys_after_the_forward(self, valid):
        model = _model("sdpa")
        caches = _caches(model, 1)
        with torch.no_grad():
            model(torch.zeros(1, 4, dtype=torch.long), kv_caches=caches)
            with pytest.raises(ValueError, match="valid_keys must be bool of shape"):
                model(torch.zeros(1, 2, dtype=torch.long), kv_caches=caches, valid_keys=valid)
        assert caches[0].seq_len == 4


class TestBuildCachedAttentionMask:
    def test_lower_right_causal_narrowed_by_validity(self):
        valid = torch.tensor([[True, False, True, True, True]])
        mask = build_cached_attention_mask(seq_len=2, valid_keys=valid)
        expected = torch.tensor([[True, False, True, True, False], [True, False, True, True, True]])
        assert mask.shape == (1, 1, 2, 5)
        assert torch.equal(mask[0, 0], expected)

    def test_a_query_without_a_key_is_unmasked(self):
        mask = build_cached_attention_mask(
            seq_len=1, valid_keys=torch.zeros(1, 3, dtype=torch.bool)
        )
        assert mask.all()

    def test_rejects_too_few_keys_and_other_dtypes(self):
        with pytest.raises(ValueError, match="fewer than"):
            build_cached_attention_mask(seq_len=3, valid_keys=torch.ones(1, 2, dtype=torch.bool))
        with pytest.raises(ValueError, match="bool tensor"):
            build_cached_attention_mask(seq_len=1, valid_keys=torch.ones(1, 2))


class TestKeepRows:
    def test_keeps_the_rows_in_order(self):
        cache = KVCache(3, 4, 1, 2, torch.float32, CPU)
        k = torch.arange(3, dtype=torch.float32).view(3, 1, 1, 1).expand(3, 1, 2, 2)
        cache.update(k, -k)
        cache.keep_rows(torch.tensor([2, 0]))
        keys, values = cache.update(torch.zeros(2, 1, 1, 2), torch.zeros(2, 1, 1, 2))
        assert keys.shape == (2, 1, 3, 2)
        assert torch.equal(keys[:, 0, 0, 0], torch.tensor([2.0, 0.0]))
        assert torch.equal(values[:, 0, 0, 0], torch.tensor([-2.0, -0.0]))
