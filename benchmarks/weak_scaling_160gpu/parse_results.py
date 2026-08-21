"""Parse the H200 MFU sweep logs into a results table + scaling efficiency + pulse.

Reads <results_dir>/*.log (one per benchmark config), extracts steady-state
tok/s / MFU / mem / step_time (median over the BACK HALF of each run -- discards
compile + thermal warmup), aggregates `_rN` repeats into mean +/- std, computes
WEAK-scaling efficiency per model (per-GPU batch is held constant, so tokens/step
and tok/s grow linearly with GPU count -- this is weak, not strong, scaling), and
folds in KempnerPulse telemetry from <results_dir>/pulse/*.csv (best-effort).
Prints markdown.

Usage:  python parse_results.py <results_dir>   (default: ./results)
"""

from __future__ import annotations

import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# [step N] loss=.. | lr=.. | grad_norm=.. | tok/s=.. | mfu=..% | mem=../..GB | step_time=..s
# tok/s may be "125,000" or "125k"/"1.2m" depending on formatter -- handle both.
STEP_RE = re.compile(
    r"\[step (\d+)\].*?tok/s=([\d.,kKmM]+)\s*\|\s*mfu=([\d.]+)%\s*\|\s*"
    r"mem=([\d.]+)/[\d.]+GB\s*\|\s*step_time=([\d.]+)s"
)
# 70b_192gpu_tp4_fsdp48_r1  ->  (70b, 192, tp4_fsdp48, 1)
NAME_RE = re.compile(r"^(?:preflight_)?(\d+b|moe\w*)_(\d+)gpu_(.+?)(?:_r(\d+))?$")


# Steady-state = back half of the run (auto-discards compile + thermal warmup).
# A 500-step headline medians over ~250 steady steps; a 30-step point over ~15.
def _steady_window(n: int) -> int:
    return max(5, n // 2)


def _tok(s: str) -> float:
    s = s.replace(",", "").strip()
    mult = 1.0
    if s and s[-1] in "kK":
        mult, s = 1e3, s[:-1]
    elif s and s[-1] in "mM":
        mult, s = 1e6, s[:-1]
    return float(s) * mult


def parse_log(path: Path) -> dict | None:
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        m = STEP_RE.search(line)
        if m:
            rows.append(
                (
                    int(m.group(1)),
                    _tok(m.group(2)),
                    float(m.group(3)),
                    float(m.group(4)),
                    float(m.group(5)),
                )
            )
    if not rows:
        return None
    tail = rows[-_steady_window(len(rows)) :]
    nm = NAME_RE.match(path.stem)
    if nm:
        model, gpus, par, rep = nm.group(1), int(nm.group(2)), nm.group(3), nm.group(4)
    else:
        model, gpus, par, rep = path.stem, 0, "?", None
    return {
        "stem": path.stem,
        "model": model,
        "gpus": gpus,
        "par": par,
        "rep": rep,
        "n_steps": rows[-1][0],
        "tok_s": statistics.median(r[1] for r in tail),
        "mfu": statistics.median(r[2] for r in tail),
        "mem_peak": max(r[3] for r in rows),
        "step_time": statistics.median(r[4] for r in tail),
    }


def pulse_summary(results_dir: Path) -> dict[str, str]:
    """Mean sm_active% / tensor_active% / dram_active% over BUSY GPUs per config,
    from KempnerPulse DCGM CSV exports (header-driven column lookup, so it is
    robust to column order). A GPU is "busy" if gpu_util_pct > 20, so idle ranks
    in an allocation don't dilute the averages."""
    pdir = results_dir / "pulse"
    if not pdir.is_dir():
        return {}
    want = ("sm_active_pct", "tensor_active_pct", "dram_active_pct")
    by_config: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for csv in pdir.glob("*.csv"):
        cfg = re.sub(r"_[^_]+_s\d+$", "", csv.stem)  # strip _<node>_sN
        lines = csv.read_text(errors="replace").splitlines()
        if not lines:
            continue
        header = lines[0].split(",")
        col = {name: header.index(name) for name in ("gpu_util_pct", *want) if name in header}
        if "gpu_util_pct" not in col:
            continue
        for line in lines[1:]:
            cells = line.split(",")
            try:
                if float(cells[col["gpu_util_pct"]]) <= 20:  # skip idle GPUs
                    continue
            except (ValueError, IndexError):
                continue
            for name in want:
                if name in col and col[name] < len(cells):
                    v = cells[col[name]].strip()
                    if re.fullmatch(r"[\d.]+", v):
                        by_config[cfg][name].append(float(v))
    out = {}
    for cfg, metrics in by_config.items():
        parts = [
            f"{k.split('_')[0]}={statistics.mean(v):.0f}%" for k in want if (v := metrics.get(k))
        ]
        if parts:
            n = max((len(v) for v in metrics.values()), default=0)
            out[cfg] = f"{' '.join(parts)} (n={n})"
    return out


def fmt_mean_std(vals: list[float]) -> str:
    if len(vals) == 1:
        return f"{vals[0]:,.0f}" if vals[0] >= 100 else f"{vals[0]:.1f}"
    m, s = statistics.mean(vals), statistics.stdev(vals)
    return f"{m:,.0f}±{s:,.0f}" if m >= 100 else f"{m:.1f}±{s:.1f}"


def main() -> int:
    rdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    logs = sorted(rdir.glob("*.log"))
    if not logs:
        print(f"no logs in {rdir}", file=sys.stderr)
        return 1

    parsed = [r for r in (parse_log(p) for p in logs) if r]
    pulse = pulse_summary(rdir)

    # Aggregate _rN repeats: key = (model, gpus, par)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in parsed:
        groups[(r["model"], r["gpus"], r["par"])].append(r)

    print("# MFU & weak-scaling on H200\n")
    print(
        "Steady-state = median over the back half of each run (discards compile + "
        "thermal warmup). Repeats shown mean±std.\n"
    )
    print("| Model | GPUs | Parallelism | tok/s | MFU % | Mem/GPU GB | Step s | n |")
    print("|-------|-----:|-------------|------:|------:|-----------:|-------:|--:|")
    agg: dict[tuple, dict] = {}
    for (model, gpus, par), rs in sorted(groups.items(), key=lambda k: (k[0][0], k[0][1])):
        toks = [r["tok_s"] for r in rs]
        mfus = [r["mfu"] for r in rs]
        agg[(model, gpus, par)] = {"tok_s": statistics.mean(toks), "mfu": statistics.mean(mfus)}
        mem = max(r["mem_peak"] for r in rs)
        step = statistics.median(r["step_time"] for r in rs)
        print(
            f"| {model} | {gpus} | {par} | {fmt_mean_std(toks)} | {fmt_mean_std(mfus)} | "
            f"{mem:.1f} | {step:.2f} | {len(rs)} |"
        )

    # Weak scaling per model: per-GPU batch is constant, so tok/s should grow
    # linearly with GPU count. Efficiency = actual / ideal-linear, anchored at
    # 32 GPUs (the published envelope edge) when available -- the 8-GPU 70B point
    # is memory-bound (barely fits) and makes a misleadingly low baseline.
    print("\n## Weak-scaling efficiency\n")
    print(
        "_Per-GPU batch held constant; ideal = linear in GPU count. "
        "Anchored at 32 GPUs (or smallest available)._\n"
    )
    for model in sorted({m for m, _, _ in agg}):
        pts = sorted(((g, a) for (mm, g, _), a in agg.items() if mm == model), key=lambda x: x[0])
        if len(pts) < 2:
            continue
        base_g, base = next((p for p in pts if p[0] == 32), pts[0])
        print(f"\n**{model}** (baseline {base_g} GPUs = {base['tok_s']:,.0f} tok/s):\n")
        print("| GPUs | tok/s | MFU % | ideal tok/s | scaling eff % |")
        print("|-----:|------:|------:|------------:|--------------:|")
        for g, a in pts:
            ideal = base["tok_s"] * (g / base_g)
            eff = 100.0 * a["tok_s"] / ideal
            print(f"| {g} | {a['tok_s']:,.0f} | {a['mfu']:.1f} | {ideal:,.0f} | {eff:.0f} |")

    if pulse:
        print("\n## KempnerPulse telemetry (busy GPUs, mid-run)\n")
        print("| Config | SM / Tensor active |")
        print("|--------|--------------------|")
        for cfg in sorted(pulse):
            print(f"| {cfg} | {pulse[cfg]} |")

    print(
        "\n_Note: MFU reported for dense models only; MoE/MoT MFU underestimates ~2x "
        "(see design-doc) -- read MoE rows as tok/s._"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
