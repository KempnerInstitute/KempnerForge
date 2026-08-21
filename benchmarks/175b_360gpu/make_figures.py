"""Render the two figures in README.md from the training log.

Both are derived from results/175b-360gpu.log alone, so they can be regenerated
from what this folder commits. Requires matplotlib (the `dev` dependency group);
parse_results.py produces the numeric tables without it.

Usage:  python make_figures.py [log] [out_dir] [--ckpt-interval N]
        (defaults: results/175b-360gpu.log, figures/)
"""

from __future__ import annotations

import argparse
import re
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STEP_RE = re.compile(
    r"\[step (\d+)\] loss=([\d.]+).*?tok/s=([\d,]+).*?mfu=([\d.]+)%.*?"
    r"mem=([\d.]+)/[\d.]+GB.*?step_time=([\d.]+)s"
)

BLUE = "#1b4d8f"
AMBER = "#bd6209"
GREY = "#8b94a1"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.30,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.autolayout": True,
    }
)


def parse(log: Path) -> dict[str, list]:
    s: dict[str, list] = {"step": [], "loss": [], "tps": [], "mfu": [], "mem": [], "stime": []}
    for line in log.read_text(errors="replace").splitlines():
        m = STEP_RE.search(line)
        if m:
            s["step"].append(int(m.group(1)))
            s["loss"].append(float(m.group(2)))
            s["tps"].append(int(m.group(3).replace(",", "")))
            s["mfu"].append(float(m.group(4)))
            s["mem"].append(float(m.group(5)))
            s["stime"].append(float(m.group(6)))
    return s


def dashboard(s: dict[str, list], out: Path, mfu_ss: float, tps_ss: float) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax[0, 0].plot(s["step"], s["loss"], color=BLUE, lw=1.4)
    ax[0, 0].set(title="Training loss", xlabel="step", ylabel="cross-entropy loss")
    ax[0, 1].plot(s["step"], s["mfu"], color=AMBER, lw=0.9)
    ax[0, 1].axhline(mfu_ss, ls="--", color=GREY)
    ax[0, 1].set(
        title=f"Model-FLOPs Utilization (median {mfu_ss:.1f}%)",
        xlabel="step",
        ylabel="MFU %",
        ylim=(0, 60),
    )
    ax[1, 0].plot(s["step"], [t / 1e3 for t in s["tps"]], color=BLUE, lw=0.9)
    ax[1, 0].axhline(tps_ss / 1e3, ls="--", color=GREY)
    ax[1, 0].set(
        title=f"Throughput (median {tps_ss / 1e3:.0f}K tok/s)",
        xlabel="step",
        ylabel="tok/s (thousands)",
    )
    ax[1, 1].plot(s["step"], s["stime"], color=AMBER, lw=0.9)
    ax[1, 1].set(title="Step time — spikes are async checkpoints", xlabel="step", ylabel="seconds")
    fig.suptitle("175B (Llama-3 arch) on 360 H200 GPUs — training dashboard", fontweight="bold")
    fig.savefig(out / "dashboard.png", bbox_inches="tight")
    plt.close(fig)


def checkpoint_cost(s: dict[str, list], out: Path, step_ss: float, interval: int) -> None:
    cks = []
    for k in range(interval, s["step"][-1] + 1, interval):
        win = [(x, t) for x, t in zip(s["step"], s["stime"], strict=True) if k <= x <= k + 12]
        if win:
            cks.append(max(win, key=lambda x: x[1]))
    if not cks:
        return
    xs = [str(x) for x, _ in cks]
    ys = [t for _, t in cks]
    overhead = [t - step_ss for t in ys]
    wall = sum(s["stime"])
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    bars = ax.bar(xs, ys, color=AMBER, width=0.55, label="checkpoint step")
    ax.axhline(step_ss, ls="--", color=BLUE, label=f"normal step ({step_ss:.1f}s)")
    for r, t in zip(bars, ys, strict=True):
        ax.text(
            r.get_x() + r.get_width() / 2,
            t + 1.5,
            f"{t:.0f}s",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    ax.set(
        title=f"Async-checkpoint cost at 175B (every {interval} steps)",
        xlabel="step",
        ylabel="step time (s)",
        ylim=(0, max(ys) * 1.42),
    )
    ax.legend(loc="upper right")
    ax.text(
        0.02,
        0.95,
        f"mean stall +{st.mean(overhead):.0f}s/ckpt\n"
        f"{len(cks)} ckpts = {100 * sum(overhead) / wall:.1f}% of wall-clock",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="#f3f0e8", ec=GREY),
    )
    fig.savefig(out / "checkpoint_cost.png", bbox_inches="tight")
    plt.close(fig)
    print(
        f"checkpoint_cost.png: n={len(cks)} mean_stall=+{st.mean(overhead):.0f}s "
        f"cost={100 * sum(overhead) / wall:.2f}% of wall"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", nargs="?", default="results/175b-360gpu.log", type=Path)
    ap.add_argument("out", nargs="?", default="figures", type=Path)
    ap.add_argument("--ckpt-interval", type=int, default=300)
    args = ap.parse_args()

    s = parse(args.log)
    if not s["step"]:
        print(
            f"no step lines found in {args.log}: expected lines matching\n"
            f"  [step N] loss=.. | ... | tok/s=.. | mfu=..% | mem=../..GB | step_time=..s\n"
            "Point this at the training log written by run_175b.sbatch."
        )
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    back = slice(max(1, len(s["step"]) // 2), None)
    mfu_ss = st.median(s["mfu"][back])
    tps_ss = st.median(s["tps"][back])
    step_ss = st.median(s["stime"][back])
    print(
        f"parsed {len(s['step'])} steps | steady MFU={mfu_ss:.1f}% "
        f"tok/s={tps_ss:,.0f} step={step_ss:.2f}s"
    )

    dashboard(s, args.out, mfu_ss, tps_ss)
    print("dashboard.png written")
    checkpoint_cost(s, args.out, step_ss, args.ckpt_interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
