"""Integration test: tokens appended to a filled KV cache on the CUDA SDPA backends.

Chunks appended to a filled cache match one full forward under each CUDA SDPA
backend setting, kernels the CPU unit tier does not reach. The masking contract
itself is covered in ``tests/unit/test_masking.py``.
"""

from __future__ import annotations

import pytest
import torch

from kempnerforge.config.schema import ModelConfig
from kempnerforge.model.attention import KVCache
from kempnerforge.model.transformer import Transformer


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("sdpa_backend", "dtype", "tol"),
    [
        ("auto", torch.bfloat16, 2e-2),
        ("flash", torch.bfloat16, 2e-2),
        ("efficient", torch.bfloat16, 2e-2),
        ("efficient", torch.float32, 1e-5),
        ("math", torch.float32, 1e-5),
    ],
)
def test_chunks_appended_to_a_filled_cache_match_one_full_forward(sdpa_backend, dtype, tol):
    """Each appended token sees every cached key and the new keys up to its own position."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    torch.manual_seed(0)
    config = ModelConfig(
        dim=256,
        n_layers=4,
        n_heads=8,
        n_kv_heads=2,
        vocab_size=512,
        max_seq_len=128,
        sdpa_backend=sdpa_backend,
    )
    model = Transformer(config).to(device, dtype).eval()
    tokens = torch.randint(0, 512, (2, 96), generator=torch.Generator().manual_seed(1)).to(device)
    caches = [
        KVCache(
            batch_size=2,
            max_seq_len=128,
            n_kv_heads=2,
            head_dim=config.head_dim,
            dtype=dtype,
            device=device,
        )
        for _ in range(config.n_layers)
    ]
    with torch.no_grad():
        full = model(tokens)
        appended = [
            model(tokens[:, a:b], kv_caches=caches) for a, b in ((0, 40), (40, 72), (72, 96))
        ]
    torch.testing.assert_close(torch.cat(appended, dim=1), full, rtol=tol, atol=tol)
    assert caches[0].seq_len == 96
