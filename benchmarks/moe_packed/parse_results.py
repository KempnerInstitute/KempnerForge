"""Parse moe_packed benchmark logs into a results table.

Reads moe_packed_results/*.log, extracts steady-state tok/s (mean over
last 10 steps) and peak memory, prints markdown table.
"""

from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

STEP_RE = re.compile(
    r"\[step (\d+)\] loss=\S+ \| lr=\S+ \| grad_norm=\S+ \| "
    r"tok/s=([\d,]+) \| mfu=([\d.]+)% \| mem=([\d.]+)/[\d.]+GB \| step_time=([\d.]+)s"
)


NAME_RE = re.compile(r"^\d+_\d+gpu_moe_e(\d+)_(\w+)_(\w+)$")


def nice_name(stem: str) -> str:
    m = NAME_RE.match(stem)
    if not m:
        return stem
    e, topo, mode = m.groups()
    return f"E={e}, {topo.replace('_', '+')}, {mode}"


def parse_log(path: Path, tail_n: int = 10) -> dict | None:
    text = path.read_text(errors="replace")
    rows = []
    for line in text.splitlines():
        m = STEP_RE.search(line)
        if m:
            step = int(m.group(1))
            toks = int(m.group(2).replace(",", ""))
            mfu = float(m.group(3))
            mem = float(m.group(4))
            st = float(m.group(5))
            rows.append((step, toks, mfu, mem, st))
    if not rows:
        return None
    tail = rows[-tail_n:]
    return {
        "name": path.stem,
        "n_steps": rows[-1][0],
        "tok_s_median": statistics.median(r[1] for r in tail),
        "mfu_median": statistics.median(r[2] for r in tail),
        "mem_peak": max(r[3] for r in rows),
        "step_time_median": statistics.median(r[4] for r in tail),
    }


def main() -> int:
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("moe_packed_results")
    logs = sorted(results_dir.glob("*.log"))
    if not logs:
        print(f"no logs in {results_dir}", file=sys.stderr)
        return 1

    rows = []
    for log in logs:
        r = parse_log(log)
        if r is None:
            print(f"# {log.name}: no steady-state steps", file=sys.stderr)
            continue
        rows.append(r)

    print("| Cell | tok/s (median) | MFU (%) | Peak Mem (GB) | Step Time (s) |")
    print("|------|--------------:|--------:|-------------:|-------------:|")
    for r in rows:
        print(
            f"| {nice_name(r['name'])} | {r['tok_s_median']:,.0f} | {r['mfu_median']:.1f} | "
            f"{r['mem_peak']:.1f} | {r['step_time_median']:.2f} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
