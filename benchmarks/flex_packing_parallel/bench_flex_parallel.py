"""FlexAttention vs dense-mask SDPA for packed sequences, per parallelism strategy.

The single-GPU numbers in ``benchmarks/micro/bench_forward.py`` overstate what a
distributed run sees: sharding and tensor parallelism add communication that
attention does not, so the same attention saving is a smaller share of the step.
This sweep measures that directly, one strategy at a time.

Three arms per cell, all at the same per-rank batch:

* ``unpacked``   -- no ``doc_ids``, so SDPA takes its ``is_causal`` fast path.
                    The do-nothing baseline: what packing has to beat to be worth it.
* ``sdpa+pack``  -- ``doc_ids`` with the dense ``(B, 1, S, S)`` mask, rebuilt per layer.
* ``flex+pack``  -- ``doc_ids`` with one FlexAttention ``BlockMask`` per forward.

Two measurement choices that matter:

* **Per-rank batch is fixed**, so ``seq_len`` is the only variable. Scaling batch
  with length would confound the two.
* **Tokens/sec counts global tokens** (``batch x dp_size x seq_len``). TP ranks all
  process the *same* batch, so counting them would inflate throughput by the TP
  degree and make TP look free.

Pipeline parallelism is absent deliberately: ``PipelineStageModule`` does not carry
``doc_ids`` on this branch, so a PP arm would silently measure unpacked attention
rather than the thing being benchmarked.

Usage (see ``run_bench.sbatch``)::

    MODE={fsdp,tp,tp_fsdp} torchrun --nproc_per_node=4 bench_flex_parallel.py
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable

import torch

from kempnerforge.config.schema import ActivationCheckpointing, DistributedConfig, ModelConfig
from kempnerforge.distributed.parallel import build_parallel_model
from kempnerforge.distributed.setup import get_world_info, init_distributed

MODE = os.environ.get("MODE", "fsdp")
WORLD = int(os.environ.get("WORLD_SIZE", "1"))
OUT_DIR = os.environ.get("BENCH_OUT_DIR", ".")

# Same shape as micro/bench_forward.py's PACKED_CONFIG (134M params), so the
# single-GPU table and this one are directly comparable.
BASE: dict[str, int] = dict(dim=768, n_layers=12, n_heads=12, vocab_size=32000, max_seq_len=16384)
MESHES: dict[str, dict[str, int]] = {
    "fsdp": dict(dp_shard=WORLD, tp=1, pp=1),
    "tp": dict(dp_shard=1, tp=WORLD, pp=1),
    "tp_fsdp": dict(dp_shard=max(WORLD // 2, 1), tp=2, pp=1),
}
SEQ_LENS: tuple[int, ...] = (512, 1024, 2048, 4096, 8192, 16384)
BATCH = 2  # per rank, fixed
DOCS = 8

ARMS: tuple[tuple[str, str, bool], ...] = (
    ("unpacked", "sdpa", False),
    ("sdpa+pack", "sdpa", True),
    ("flex+pack", "flex", True),
)


def log(msg: str, rank: int) -> None:
    """Print and append to the per-mode result file, rank 0 only."""
    if rank != 0:
        return
    print(msg, flush=True)
    with open(f"{OUT_DIR}/flex_parallel_{MODE}.txt", "a") as fh:
        fh.write(msg + "\n")


def doc_ids_for(batch: int, seq_len: int, device: torch.device) -> torch.Tensor:
    """``DOCS`` equal-length documents per row, which is the sparsity flex exploits."""
    per = max(seq_len // DOCS, 1)
    row = (torch.arange(seq_len, device=device) // per).clamp(max=DOCS - 1)
    return row.unsqueeze(0).expand(batch, -1).contiguous()


def timed(step: Callable[[], None], warmup: int = 4, iters: int = 8) -> tuple[float, float]:
    """Mean ms/step and peak GB, after enough warmup to pay Triton compilation."""
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(iters):
        step()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters, torch.cuda.max_memory_allocated() / 1e9


def main() -> None:
    rank, local_rank, world = get_world_info()
    dims = MESHES[MODE]
    mesh = init_distributed(DistributedConfig(**dims), seed=42)
    device = torch.device(f"cuda:{local_rank}")
    dp_size = world // (dims["tp"] * dims["pp"])

    log(f"### MODE={MODE} world={world} {dims} dp_size={dp_size} batch/rank={BATCH}", rank)
    log(f"### {torch.cuda.get_device_name(0)}  torch {torch.__version__}  docs/seq={DOCS}", rank)
    log(
        f"{'seq_len':>8} {'unpacked':>11} {'sdpa+pack':>11} {'flex+pack':>11} "
        f"{'flex/sdpa':>10} {'sdpa_GB':>8} {'flex_GB':>8}",
        rank,
    )

    records: list[dict[str, object]] = []
    for seq_len in SEQ_LENS:
        tok_s: dict[str, float] = {}
        peak_gb: dict[str, float] = {}
        for arm, backend, packed in ARMS:
            try:
                torch.manual_seed(0)
                config = ModelConfig(**BASE, attention_backend=backend)
                model = build_parallel_model(
                    config,
                    device,
                    mesh,
                    ac_mode=ActivationCheckpointing.none,
                    param_dtype=torch.bfloat16,
                    compile_model=False,
                )
                tokens = torch.randint(0, config.vocab_size, (BATCH, seq_len), device=device)
                kwargs = {"doc_ids": doc_ids_for(BATCH, seq_len, device)} if packed else {}

                def step(m=model, t=tokens, k=kwargs) -> None:
                    m(t, **k).sum().backward()
                    m.zero_grad(set_to_none=True)

                ms, gb = timed(step)
                tok_s[arm] = (BATCH * dp_size * seq_len) / (ms / 1000)
                peak_gb[arm] = gb
                del model, tokens
            except Exception as exc:  # noqa: BLE001 - one OOM must not end the sweep
                tok_s[arm] = float("nan")
                peak_gb[arm] = float("nan")
                log(f"  {arm} @ {seq_len}: FAILED {type(exc).__name__}: {str(exc)[:60]}", rank)
            torch.cuda.empty_cache()

        ratio = tok_s["flex+pack"] / tok_s["sdpa+pack"]
        log(
            f"{seq_len:>8} {tok_s['unpacked']:>11,.0f} {tok_s['sdpa+pack']:>11,.0f} "
            f"{tok_s['flex+pack']:>11,.0f} {ratio:>9.2f}x {peak_gb['sdpa+pack']:>8.2f} "
            f"{peak_gb['flex+pack']:>8.2f}",
            rank,
        )
        records.append(dict(mode=MODE, seq_len=seq_len, tok_s=tok_s, peak_gb=peak_gb, ratio=ratio))

    if rank == 0:
        with open(f"{OUT_DIR}/flex_parallel_{MODE}.json", "w") as fh:
            json.dump(
                dict(
                    mode=MODE,
                    world=world,
                    mesh=dims,
                    batch_per_rank=BATCH,
                    docs=DOCS,
                    model=BASE,
                    gpu=torch.cuda.get_device_name(0),
                    torch=torch.__version__,
                    rows=records,
                ),
                fh,
                indent=2,
            )


if __name__ == "__main__":
    main()
