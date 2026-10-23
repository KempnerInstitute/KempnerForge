"""Core model benchmarks: forward pass, forward+backward, attention, MLP."""

from __future__ import annotations

from dataclasses import replace

import torch

from benchmarks.micro.runner import BenchmarkResult, print_results, run_benchmark
from kempnerforge.config.model import ModelConfig
from kempnerforge.model.transformer import Transformer

# 125M model for benchmarking (small enough to run quickly, large enough to be meaningful)
BENCH_CONFIG = ModelConfig(
    dim=768,
    n_layers=12,
    n_heads=12,
    vocab_size=32000,
    max_seq_len=2048,
)

BATCH = 4
SEQ = 512
DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16


def run_forward_benchmarks() -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []

    model = Transformer(BENCH_CONFIG).to(device=DEVICE, dtype=DTYPE).eval()
    toks_per_iter = BATCH * SEQ

    # --- Forward only ---
    input_ids = torch.randint(0, BENCH_CONFIG.vocab_size, (BATCH, SEQ), device=DEVICE)

    def forward_fn(m=model, ids=input_ids):
        with torch.no_grad():
            m(ids)

    results.append(
        run_benchmark(
            forward_fn,
            name="forward (125M, bs=4, seq=512)",
            tokens_per_iter=toks_per_iter,
        )
    )

    # --- Forward + Backward ---
    model.train()

    def forward_backward_fn(m=model, ids=input_ids):
        logits = m(ids)
        loss = logits.sum()
        loss.backward()
        m.zero_grad()

    results.append(
        run_benchmark(
            forward_backward_fn,
            name="forward+backward (125M)",
            tokens_per_iter=toks_per_iter,
        )
    )

    # --- Attention only (varying seq lengths) ---
    from kempnerforge.model.attention import Attention
    from kempnerforge.model.position import precompute_rope_frequencies

    attn = Attention(dim=768, n_heads=12, n_kv_heads=12).to(device=DEVICE, dtype=DTYPE)
    rope_cos, rope_sin = precompute_rope_frequencies(head_dim=64, max_seq_len=2048)
    rope_cos = rope_cos.to(device=DEVICE, dtype=DTYPE)
    rope_sin = rope_sin.to(device=DEVICE, dtype=DTYPE)

    for seq_len in [512, 1024, 2048]:
        x = torch.randn(BATCH, seq_len, 768, device=DEVICE, dtype=DTYPE)
        cos_sl = rope_cos[:seq_len]
        sin_sl = rope_sin[:seq_len]

        def attn_fn(a=attn, x=x, c=cos_sl, s=sin_sl):
            with torch.no_grad():
                a(x, c, s)

        results.append(
            run_benchmark(
                attn_fn,
                name=f"attention (seq={seq_len})",
                tokens_per_iter=BATCH * seq_len,
            )
        )

    # --- MLP only ---
    from kempnerforge.model.mlp import build_mlp

    mlp = build_mlp(
        BENCH_CONFIG.dim, BENCH_CONFIG.computed_ffn_hidden_dim, BENCH_CONFIG.activation.value
    ).to(device=DEVICE, dtype=DTYPE)
    x_mlp = torch.randn(BATCH, SEQ, 768, device=DEVICE, dtype=DTYPE)

    def mlp_fn(m=mlp, x=x_mlp):
        with torch.no_grad():
            m(x)

    results.append(run_benchmark(mlp_fn, name="MLP (SwiGLU, 125M)", tokens_per_iter=toks_per_iter))

    del model, attn, mlp
    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Packed-sequence attention: dense-mask SDPA vs FlexAttention
# ---------------------------------------------------------------------------

# max_seq_len has to cover the longest sweep point. head_dim = 768 / 12 = 64,
# comfortably inside FlexAttention's supported 16..256.
PACKED_CONFIG = ModelConfig(
    dim=768,
    n_layers=12,
    n_heads=12,
    vocab_size=32000,
    max_seq_len=8192,
)

# Batch shrinks as the sequence grows so each point moves a comparable number of
# tokens and the dense (B, 1, S, S) mask stays addressable at S=8192.
PACKED_SWEEP = [(512, 16), (2048, 8), (8192, 2)]
DOCS_PER_SEQ = 8


def _doc_ids(batch: int, seq_len: int, docs: int, device: torch.device) -> torch.Tensor:
    """Split each row into `docs` equal-length documents."""
    per_doc = seq_len // docs
    row = torch.arange(seq_len, device=device) // per_doc
    return row.clamp(max=docs - 1).unsqueeze(0).expand(batch, -1).contiguous()


def _packed_case(backend: str, seq_len: int, batch: int, docs: int | None, n_kv_heads=None):
    """Build (model, tokens, kwargs) for one benchmark arm."""
    config = replace(PACKED_CONFIG, attention_backend=backend, n_kv_heads=n_kv_heads)
    model = Transformer(config).to(device=DEVICE, dtype=DTYPE)
    tokens = torch.randint(0, config.vocab_size, (batch, seq_len), device=DEVICE)
    kwargs = {} if docs is None else {"doc_ids": _doc_ids(batch, seq_len, docs, DEVICE)}
    return model, tokens, kwargs


def run_packed_attention_benchmarks() -> list[BenchmarkResult]:
    """Forward+backward throughput for the three packed-attention paths.

    Three arms per shape:

    * ``unpacked``   -- no doc_ids, so SDPA takes its ``is_causal`` fast path.
      The speed-of-light reference; it does no cross-document masking at all.
    * ``sdpa+packed`` -- today's packing. A dense (B, 1, S, S) bool mask is built
      per layer, which is not a FlashAttention-2 shape, so SDPA drops to the
      mem-efficient kernel.
    * ``flex+packed`` -- the same mask as a FlexAttention BlockMask, built once
      per forward, with fully-masked blocks skipped rather than computed.

    The comparison that matters is arm 3 against arm 2; arm 1 bounds how much
    of the gap is recoverable. Peak memory matters as much as tok/s here: the
    dense mask is what flex deletes.
    """
    results: list[BenchmarkResult] = []

    def bench(label, backend, seq_len, batch, docs, n_kv_heads=None):
        model, tokens, kwargs = _packed_case(backend, seq_len, batch, docs, n_kv_heads)
        model.train()

        def step(m=model, ids=tokens, kw=kwargs):
            m(ids, **kw).sum().backward()
            m.zero_grad(set_to_none=True)

        # Above the default warmup: the first flex call pays Triton compilation,
        # which must not land inside the timed window.
        results.append(
            run_benchmark(
                step, warmup=6, iterations=10, name=label, tokens_per_iter=batch * seq_len
            )
        )
        del model, tokens, kwargs
        torch.cuda.empty_cache()

    for seq_len, batch in PACKED_SWEEP:
        shape = f"S={seq_len}, b={batch}, {DOCS_PER_SEQ} docs"
        bench(f"unpacked causal      ({shape})", "sdpa", seq_len, batch, None)
        bench(f"sdpa + packed        ({shape})", "sdpa", seq_len, batch, DOCS_PER_SEQ)
        bench(f"flex + packed        ({shape})", "flex", seq_len, batch, DOCS_PER_SEQ)

    # Document count at fixed shape: more documents means a sparser block-diagonal
    # mask, so flex should pull further ahead while dense SDPA is unaffected.
    for docs in (2, 16, 64):
        shape = f"S=2048, b=8, {docs} docs"
        bench(f"sdpa + packed        ({shape})", "sdpa", 2048, 8, docs)
        bench(f"flex + packed        ({shape})", "flex", 2048, 8, docs)

    # GQA: flex passes enable_gqa instead of materializing repeated K/V heads.
    shape = f"S=2048, b=8, {DOCS_PER_SEQ} docs, GQA 12/4"
    bench(f"sdpa + packed        ({shape})", "sdpa", 2048, 8, DOCS_PER_SEQ, n_kv_heads=4)
    bench(f"flex + packed        ({shape})", "flex", 2048, 8, DOCS_PER_SEQ, n_kv_heads=4)

    return results


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("Benchmarks require a CUDA GPU")
    print_results(run_forward_benchmarks(), "Forward Benchmarks")
    print_results(run_packed_attention_benchmarks(), "Packed Attention Benchmarks")
