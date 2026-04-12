"""MoE benchmarks: forward+backward, router comparison, grouped GEMM vs loop."""

from __future__ import annotations

import torch

from benchmarks.runner import BenchmarkResult, print_results, run_benchmark
from kempnerforge.config.model import ModelConfig
from kempnerforge.model.transformer import Transformer

BATCH = 4
SEQ = 512
DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16


def _moe_config(
    num_experts: int = 8, top_k: int = 2, router: str = "softmax_topk"
) -> ModelConfig:
    return ModelConfig(
        dim=768,
        n_layers=4,
        n_heads=12,
        vocab_size=32000,
        max_seq_len=2048,
        num_experts=num_experts,
        moe_top_k=top_k,
        moe_router=router,
    )


def run_moe_benchmarks() -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    toks_per_iter = BATCH * SEQ

    # --- MoE forward + backward (8 experts, top-2) ---
    config = _moe_config()
    model = Transformer(config).to(device=DEVICE, dtype=DTYPE)
    input_ids = torch.randint(0, 32000, (BATCH, SEQ), device=DEVICE)

    def moe_fwd_bwd(m=model, ids=input_ids):
        logits = m(ids)
        loss = logits.sum()
        loss.backward()
        m.zero_grad()

    results.append(
        run_benchmark(
            moe_fwd_bwd,
            name="MoE fwd+bwd (8E top-2)",
            tokens_per_iter=toks_per_iter,
        )
    )
    del model
    torch.cuda.empty_cache()

    # --- Router comparison: softmax_topk vs sigmoid_topk ---
    for router in ["softmax_topk", "sigmoid_topk"]:
        config = _moe_config(router=router)
        model = Transformer(config).to(device=DEVICE, dtype=DTYPE).eval()
        input_ids = torch.randint(0, 32000, (BATCH, SEQ), device=DEVICE)

        def router_fn(m=model, ids=input_ids):
            with torch.no_grad():
                m(ids)

        results.append(
            run_benchmark(
                router_fn,
                name=f"MoE forward ({router})",
                tokens_per_iter=toks_per_iter,
            )
        )
        del model
        torch.cuda.empty_cache()

    # --- Grouped GEMM vs loop dispatch ---
    import kempnerforge.model.moe as moe_mod
    from kempnerforge.model.moe import build_moe

    config = _moe_config()
    moe_layer = build_moe(
        dim=config.dim,
        hidden_dim=config.computed_ffn_hidden_dim,
        num_experts=config.num_experts,
        top_k=config.moe_top_k,
        activation=config.activation.value,
        router_type=config.moe_router,
    ).to(device=DEVICE, dtype=DTYPE).eval()
    x = torch.randn(BATCH, SEQ, 768, device=DEVICE, dtype=DTYPE)

    orig_flag = moe_mod._HAS_GROUPED_MM

    # Loop dispatch (force by disabling grouped_mm)
    moe_mod._HAS_GROUPED_MM = False

    def loop_fn(layer=moe_layer, inp=x):
        with torch.no_grad():
            layer(inp)

    results.append(
        run_benchmark(loop_fn, name="MoE dispatch: loop", tokens_per_iter=toks_per_iter)
    )

    # Grouped GEMM (if available)
    moe_mod._HAS_GROUPED_MM = orig_flag
    if orig_flag:

        def gemm_fn(layer=moe_layer, inp=x):
            with torch.no_grad():
                layer(inp)

        results.append(
            run_benchmark(
                gemm_fn,
                name="MoE dispatch: grouped_mm",
                tokens_per_iter=toks_per_iter,
            )
        )

    del moe_layer
    torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("Benchmarks require a CUDA GPU")
    print_results(run_moe_benchmarks(), "MoE Benchmarks")
