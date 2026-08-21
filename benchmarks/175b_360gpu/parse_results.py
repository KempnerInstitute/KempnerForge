"""Parse the 175B / 360-GPU training log into the tables in README.md.

Steady state is the median over the BACK HALF of the run, which discards
`torch.compile` warmup and the thermal ramp. Checkpoint stalls are found by
taking the slowest step in a short window after each checkpoint boundary, and
their cost is reported against the steady-state step time.

Deliberately has no matplotlib dependency, so the numbers can be re-derived
without the plotting extras. Use make_figures.py for the figures.

Usage:  python parse_results.py [log] [--ckpt-interval N]
        (default log: results/175b-360gpu.log)
"""

from __future__ import annotations

import argparse
import re
import statistics as st
from pathlib import Path

# [step N] loss=.. | lr=.. | grad_norm=.. | tok/s=.. | mfu=..% | mem=../..GB | step_time=..s
STEP_RE = re.compile(
    r"\[step (\d+)\] loss=([\d.]+).*?tok/s=([\d,]+).*?mfu=([\d.]+)%.*?"
    r"mem=([\d.]+)/([\d.]+)GB.*?step_time=([\d.]+)s"
)

GPU_PEAK_BF16 = 989.5e12  # H200 SXM dense bf16 peak, FLOP/s per GPU
N_GPU = 360
SEQ_LEN = 4096
BATCH_PER_GPU = 8
DP_SHARD = 90


def parse(log: Path) -> list[dict]:
    rows = []
    for line in log.read_text(errors="replace").splitlines():
        m = STEP_RE.search(line)
        if m:
            rows.append(
                {
                    "step": int(m.group(1)),
                    "loss": float(m.group(2)),
                    "tok_s": float(m.group(3).replace(",", "")),
                    "mfu": float(m.group(4)),
                    "mem": float(m.group(5)),
                    "mem_total": float(m.group(6)),
                    "step_time": float(m.group(7)),
                }
            )
    return rows


def checkpoint_stalls(rows: list[dict], interval: int) -> list[tuple[int, float]]:
    """Slowest step within 12 steps after each checkpoint boundary."""
    out = []
    last = rows[-1]["step"]
    for k in range(interval, last + 1, interval):
        window = [(r["step"], r["step_time"]) for r in rows if k <= r["step"] <= k + 12]
        if window:
            out.append(max(window, key=lambda x: x[1]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", nargs="?", default="results/175b-360gpu.log", type=Path)
    ap.add_argument("--ckpt-interval", type=int, default=300)
    args = ap.parse_args()

    rows = parse(args.log)
    if not rows:
        print(
            f"no step lines found in {args.log}: expected lines matching\n"
            f"  [step N] loss=.. | ... | tok/s=.. | mfu=..% | mem=../..GB | step_time=..s\n"
            "Point this at the training log written by run_175b.sbatch."
        )
        return 1

    back = rows[len(rows) // 2 :]
    mfu = st.median(r["mfu"] for r in back)
    tok_s = st.median(r["tok_s"] for r in back)
    step_s = st.median(r["step_time"] for r in back)
    mem = max(r["mem"] for r in rows)
    mem_total = rows[0]["mem_total"]
    global_batch = BATCH_PER_GPU * DP_SHARD * SEQ_LEN
    tokens = len(rows) * global_batch
    pflops = N_GPU * GPU_PEAK_BF16 * mfu / 100 / 1e15

    print(f"# 175B on {N_GPU} H200 — steady state (median over back half of {len(rows)} steps)\n")
    print("| Metric | Value |")
    print("|--------|-------|")
    print(f"| Sustained MFU | **{mfu:.1f}%** |")
    print(f"| Throughput | **{tok_s:,.0f} tokens/s** |")
    print(f"| Step time | {step_s:.2f} s (global batch {global_batch / 1e6:.2f}M tokens) |")
    print(f"| Aggregate model FLOPs | ~{pflops:.0f} PFLOP/s |")
    print(f"| Peak memory | {mem:.1f} / {mem_total:.0f} GB per GPU |")
    loss_span = f"{rows[0]['loss']:.2f} -> {rows[-1]['loss']:.2f}"
    print(f"| Loss | {loss_span} over {tokens / 1e9:.2f}B tokens |")

    stalls = checkpoint_stalls(rows, args.ckpt_interval)
    if stalls:
        overhead = [t - step_s for _, t in stalls]
        wall = sum(r["step_time"] for r in rows)
        print(f"\n## Asynchronous checkpoint cost (every {args.ckpt_interval} steps)\n")
        print("| Step | Step time | Overhead vs baseline |")
        print("|-----:|----------:|---------------------:|")
        for (s, t), oh in zip(stalls, overhead, strict=True):
            print(f"| {s} | {t:.1f} s | +{oh:.1f} s |")
        print(
            f"\nMean stall **+{st.mean(overhead):.0f} s** per checkpoint; "
            f"{len(stalls)} checkpoints total **{100 * sum(overhead) / wall:.1f}%** of wall-clock "
            f"against a {step_s:.2f} s baseline step."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
