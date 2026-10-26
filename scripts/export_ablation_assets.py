"""Copy exploratory ablation figures and tables into the Git-tracked report tree.

``results/`` is gitignored except ``results/figures/`` (force-added). This script
reads the local ablation outputs in ``results/ablations/`` (written by
``python -m churnmaster ablate --suite agent|reward|memoryless``) and writes

* ``results/figures/ablations/*.png``           (copies of the suite figures)
* ``results/figures/ablations/tables/*.md``     (markdown tables for docs/report.md)
* ``results/figures/ablations/assets_manifest.json`` (sha256 of sources and outputs)

It only reads ``results/ablations`` and never touches canonical results. All
statistics are exploratory (TEST split, run 0, episodes 0..19; uncorrected
p-values). Usage: ``python scripts/export_ablation_assets.py``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "results" / "ablations"
DST = REPO / "results" / "figures" / "ablations"
SCENARIOS = ("stable", "high_churn", "heldout_medium")
FIGURES = ("agent_ablation_J.png", "agent_ablation_metrics.png", "reward_sweep_J.png",
           "reward_sweep_p95.png", "reward_sweep_tradeoff.png", "memoryless_J_p95.png")
AGENT_ORDER = ("buffered_control", "shuffled_reward", "gamma0", "no_probe", "drop_tier",
               "drop_inc_fail", "drop_inc_stale", "drop_inc_age", "drop_cand_freshness",
               "drop_cand_faster", "drop_churn")
HEADER = ("Exploratory (not part of the locked evaluation): TEST split, run 0, episodes 0..19, "
          "20 paired seeds; 95% paired-bootstrap CIs; Wilcoxon p-values uncorrected (no Holm).")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ci(r: pd.Series | None, d: int) -> str:
    if r is None:
        return "-"
    return f"{r['mean_diff']:+.{d}f} [{r['ci_lo']:+.{d}f}, {r['ci_hi']:+.{d}f}]"


def pick(df: pd.DataFrame, **kw) -> pd.Series | None:
    m = pd.Series(True, index=df.index)
    for k, v in kw.items():
        m &= df[k] == v
    sub = df[m]
    if len(sub) > 1:
        raise ValueError(f"ambiguous selection {kw}: {len(sub)} rows")
    return None if sub.empty else sub.iloc[0]


def agent_table(c: pd.DataFrame) -> str:
    lines = ["# Agent ablations (Task 12.1)", "", HEADER,
             "Δ = variant − canonical rl_mean (both: mean of 3 runs per seed), except the last "
             "column (variant − tuned_heuristic).", "",
             "| variant | scenario | ΔJ vs canonical [95% CI] | p | Δp95 ms [95% CI] | "
             "Δpings/h [95% CI] | ΔJ vs tuned [95% CI] |", "|---|---|---|---|---|---|---|"]
    for v in AGENT_ORDER:
        for s in SCENARIOS:
            j = pick(c, comparison="vs_canonical", variant=v, scenario=s, metric="J")
            p95 = pick(c, comparison="vs_canonical", variant=v, scenario=s,
                       metric="p95_latency_all_ms")
            pg = pick(c, comparison="vs_canonical", variant=v, scenario=s, metric="pings_per_hour")
            t = pick(c, comparison="vs_tuned_heuristic", variant=v, scenario=s, metric="J")
            lines.append(f"| {v} | {s} | {ci(j, 4)} | "
                         f"{'-' if j is None else format(j['p_uncorrected'], '.2g')} | "
                         f"{ci(p95, 1)} | {ci(pg, 1)} | {ci(t, 4)} |")
    sb = c[c["comparison"] == "shuffled_vs_buffered"]
    if not sb.empty:
        lines += ["", "## shuffled_reward − buffered_control", "",
                  "| scenario | ΔJ [95% CI] | p | Δp95 ms [95% CI] |", "|---|---|---|---|"]
        for s in SCENARIOS:
            j = pick(sb, scenario=s, metric="J")
            p95 = pick(sb, scenario=s, metric="p95_latency_all_ms")
            lines.append(f"| {s} | {ci(j, 4)} | "
                         f"{'-' if j is None else format(j['p_uncorrected'], '.2g')} | {ci(p95, 1)} |")
    lines.append("")
    return "\n".join(lines)


def reward_table(c: pd.DataFrame, summ: pd.DataFrame) -> str:
    rl = summ[summ["kind"] == "rl_mean"]
    cells = rl[["policy", "cell", "w_L", "w_M", "w_R"]].drop_duplicates().sort_values(["w_M", "w_L"])
    lines = ["# Reward-weight sweep (Task 12.2)", "", HEADER,
             "Each cell trains 3 runs with (w_L, w_M, w_R = 1.5·w_M); every policy is SCORED "
             "under the canonical weights (w_L 0.3, w_M 0.1, w_R 0.15). The canonical cell is "
             "w_L 0.3, w_M 0.1. The tuned heuristic was NOT re-tuned per cell.", "",
             "| w_L | w_M | scenario | J | ΔJ vs canonical [95% CI] | p95 ms | "
             "Δp95 vs canonical [95% CI] | pings/h | ΔJ vs tuned [95% CI] | Δp95 vs tuned [95% CI] |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, cell in cells.iterrows():
        name = cell["policy"]                 # "canonical" for the w_L 0.3 / w_M 0.1 cell
        for s in SCENARIOS:
            row = pick(rl[rl["policy"] == name], scenario=s)
            j = pick(c, comparison="vs_canonical", variant=name, scenario=s, metric="J")
            p95 = pick(c, comparison="vs_canonical", variant=name, scenario=s,
                       metric="p95_latency_all_ms")
            tj = pick(c, comparison="vs_tuned_heuristic", variant=name, scenario=s, metric="J")
            tp = pick(c, comparison="vs_tuned_heuristic", variant=name, scenario=s,
                      metric="p95_latency_all_ms")
            jm = "-" if row is None else f"{row['J_mean']:.4f}"
            pm = "-" if row is None else f"{row['p95_latency_all_ms_mean']:.1f}"
            gm = "-" if row is None else f"{row['pings_per_hour_mean']:.1f}"
            tag = " (canonical)" if name == "canonical" else ""
            lines.append(f"| {cell['w_L']:g} | {cell['w_M']:g}{tag} | {s} | {jm} | {ci(j, 4)} | {pm} | "
                         f"{ci(p95, 1)} | {gm} | {ci(tj, 4)} | {ci(tp, 1)} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    if not SRC.is_dir():
        print(f"missing {SRC}; run `python -m churnmaster ablate --suite ...` first", file=sys.stderr)
        return 1
    (DST / "tables").mkdir(parents=True, exist_ok=True)
    sources = {}
    for f in FIGURES:
        src = SRC / "figures" / f
        shutil.copyfile(src, DST / f)
        sources[f"figures/{f}"] = sha(src)
    ac, rc = SRC / "agent" / "comparison.csv", SRC / "reward" / "comparison.csv"
    rs, mm = SRC / "reward" / "summary.csv", SRC / "memoryless" / "summary.md"
    tables = {
        "agent_ablations.md": agent_table(pd.read_csv(ac)),
        "reward_sweep.md": reward_table(pd.read_csv(rc), pd.read_csv(rs)),
        "memoryless.md": mm.read_text(encoding="utf-8"),
    }
    for p in (ac, rc, rs, mm, SRC / "manifest.json"):
        sources[p.relative_to(SRC).as_posix()] = sha(p)
    for name, text in tables.items():
        (DST / "tables" / name).write_text(text, encoding="utf-8", newline="\n")
    outputs = {p.relative_to(DST).as_posix(): sha(p)
               for p in sorted(DST.rglob("*")) if p.is_file() and p.name != "assets_manifest.json"}
    (DST / "assets_manifest.json").write_text(json.dumps(
        {"note": "Exploratory ablation assets copied from results/ablations (local, gitignored) "
                 "by scripts/export_ablation_assets.py; not part of the locked evaluation.",
         "sources_sha256": sources, "outputs_sha256": outputs}, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    print(f"wrote {len(outputs)} files to {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())