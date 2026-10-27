"""Colour results figure for the two-page report.

Reads the canonical results/eval/summary.csv (no other data) and writes
report/fig_results.pdf: mean J, p95 latency (all lookups) and maintenance pings
per hour (log axis) for four policies in three scenarios. Error bars are one
standard deviation across the 20 evaluation seeds. Policies are distinguished
by a colour-blind-safe palette (Okabe-Ito) plus hatching on the two
reference baselines, so the plot also survives black-and-white printing.

Usage: python report/make_figure.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "eval" / "summary.csv"
OUT = Path(__file__).with_name("fig_results.pdf")

SCENARIOS = ["stable", "high_churn", "heldout_medium"]
XLABELS = ["stable", "high\nchurn", "held-out\nmedium"]
# (policy id, legend label, face colour, hatch)
POLICIES = [
    ("kademlia_epoch", "Kademlia-epoch", "#B0B0B0", "////"),
    ("matched_random", "Matched random", "#E69F00", "...."),
    ("tuned_heuristic", "Tuned heuristic", "#0072B2", ""),
    ("rl_mean", "Q-learning (RL)", "#D55E00", ""),
]
PANELS = [
    ("J", "(a) Score $J$", "$J$ (higher is better)"),
    ("p95_latency_all_ms", "(b) p95 latency", "ms (lower is better)"),
    ("pings_per_hour", "(c) Pings per hour", "pings / h (log scale)"),
]


def main() -> None:
    s = pd.read_csv(SUMMARY)
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 6.5, "axes.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "hatch.linewidth": 0.4,
        "xtick.major.size": 2, "ytick.major.size": 2, "xtick.minor.size": 1.2,
        "ytick.minor.size": 1.2,
    })
    fig, axes = plt.subplots(1, 3, figsize=(3.45, 1.72))
    width = 0.2
    x = np.arange(len(SCENARIOS))
    for ax, (metric, title, ylabel) in zip(axes, PANELS):
        for i, (pid, label, face, hatch) in enumerate(POLICIES):
            rows = [s[(s.policy == pid) & (s.scenario == q)].iloc[0] for q in SCENARIOS]
            mean = [r[f"{metric}_mean"] for r in rows]
            std = [r[f"{metric}_std"] for r in rows]
            ax.bar(x + (i - 1.5) * width, mean, width, color=face, edgecolor="black",
                   linewidth=0.4, hatch=hatch, label=label, yerr=std, capsize=0.9,
                   error_kw={"elinewidth": 0.45, "capthick": 0.45, "ecolor": "black"})
        ax.set_xticks(x)
        ax.set_xticklabels(XLABELS, fontsize=5.6)
        ax.set_title(title, fontsize=6.8, pad=2.5)
        ax.set_ylabel(ylabel, fontsize=5.8, labelpad=1.5)
        ax.tick_params(axis="y", labelsize=5.6, pad=1.5)
        ax.tick_params(axis="x", pad=1.5, length=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylim(0.95, 0.99)
    axes[0].set_yticks([0.95, 0.96, 0.97, 0.98])
    axes[1].set_ylim(0, 950)
    axes[2].set_yscale("log")
    axes[2].set_ylim(3, 800)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=5.8,
               handlelength=1.6, handleheight=0.9, columnspacing=0.9, handletextpad=0.4,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.25, w_pad=0.6, rect=(0, 0, 1, 0.9))
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
