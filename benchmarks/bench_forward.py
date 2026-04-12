"""Core model benchmarks: forward pass, forward+backward, attention, MLP."""

from __future__ import annotations

import torch

from benchmarks.runner import BenchmarkResult, print_results, run_benchmark
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

    results.append(
        run_benchmark(mlp_fn, name="MLP (SwiGLU, 125M)", tokens_per_iter=toks_per_iter)
    )

    del model, attn, mlp
    torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("Benchmarks require a CUDA GPU")
    print_results(run_forward_benchmarks(), "Forward Benchmarks")
