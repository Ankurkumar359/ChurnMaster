"""Make report figures from results/train_log.csv and results/eval_results.csv."""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from utils import moving_average


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-log", default="results/train_log.csv")
    p.add_argument("--eval", default="results/eval_results.csv")
    p.add_argument("--bar-churn", type=float, default=4.0)
    p.add_argument("--train-range", type=float, nargs=2, default=[1.0, 4.0])
    p.add_argument("--outdir", default="results")
    a = p.parse_args()

    # 1) training curves
    tl = pd.read_csv(a.train_log)
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.5))
    for axi, col, lab in zip(ax, ["return", "success"], ["Episode return", "Lookup success"]):
        axi.plot(tl[col], alpha=0.25, lw=0.8)
        ma = moving_average(tl[col], 20)
        axi.plot(range(len(tl) - len(ma), len(tl)), ma, lw=1.8)
        axi.set_xlabel("Episode"); axi.set_ylabel(lab); axi.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{a.outdir}/fig_training.png", dpi=200); plt.close(fig)

    # 2) success vs churn
    ev = pd.read_csv(a.eval)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for name, g in ev.groupby("policy", sort=False):
        g = g.sort_values("churn")
        ax.errorbar(g["churn"], g["success_mean"], yerr=g["success_std"], marker="o", ms=3,
                    lw=1.5 if "ours" in name or "ablat" in name.lower() else 1.0, capsize=2, label=name)
    ax.axvspan(*a.train_range, color="grey", alpha=0.12, label="train churn")
    ax.set_xscale("log", base=2); ax.set_xlabel("Churn scale"); ax.set_ylabel("Lookup success")
    ax.grid(alpha=0.3); ax.legend(fontsize=5.5)
    fig.tight_layout(); fig.savefig(f"{a.outdir}/fig_success_vs_churn.png", dpi=200); plt.close(fig)

    # 3) latency + maintenance overhead at one churn level
    g = ev[ev["churn"] == a.bar_churn]
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.5))
    ax[0].barh(g["policy"], g["latency_ms"]); ax[0].set_xlabel("Mean lookup latency (ms)")
    ax[1].barh(g["policy"], g["repl_per_step"]); ax[1].set_xlabel("Replacements / step")
    ax[1].set_yticks([])
    for x in ax: x.grid(alpha=0.3, axis="x"); x.tick_params(labelsize=6)
    fig.tight_layout(); fig.savefig(f"{a.outdir}/fig_overhead.png", dpi=200); plt.close(fig)
    print("saved figures to", a.outdir)


if __name__ == "__main__":
    main()
