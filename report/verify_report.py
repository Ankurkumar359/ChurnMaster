"""Cross-check every number and claim in churnmaster_report.tex against the
canonical results, configs and source code. Read-only; exits non-zero on any
mismatch.

Usage: python report/verify_report.py
"""
from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TEX = (Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("churnmaster_report.tex")).read_text(encoding="utf-8")
EV = ROOT / "results" / "eval"
SCN = ["stable", "high_churn", "heldout_medium"]
fails: list[str] = []
n_ok = 0


def check(cond: bool, what: str) -> None:
    global n_ok
    if cond:
        n_ok += 1
    else:
        fails.append(what)


def in_tex(s: str, what: str | None = None) -> None:
    check(s in TEX, what or f"text not found in tex: {s!r}")


def r(x: float, d: int) -> str:
    return f"{x:.{d}f}"


def sgn(x: float, d: int) -> str:
    return f"{x:+.{d}f}"


# ---------------------------------------------------------------- Table 1 ---
s = pd.read_csv(EV / "summary.csv")
pol = {"Kademlia-epoch": "kademlia_epoch", "Matched random": "matched_random",
       "Tuned heuristic": "tuned_heuristic", "Q-learning (RL)": "rl_mean"}
cells: dict[tuple[str, str, str], str] = {}
for label, p in pol.items():
    row_tex = next((ln for ln in TEX.splitlines() if ln.strip().startswith(label + " ")
                    or ln.strip().startswith(label + "&")), None)
    check(row_tex is not None, f"Table 1 row missing: {label}")
    if row_tex is None:
        continue
    vals = [re.sub(r"\\textbf\{([^}]*)\}", r"\1", v).strip().rstrip("\\").strip()
            for v in row_tex.split("&")[1:]]
    exp = []
    for q in SCN:
        x = s[(s.policy == p) & (s.scenario == q)].iloc[0]
        exp += [r(x.J_mean, 4), r(100 * x.success_rate_mean, 2), r(x.p95_latency_all_ms_mean, 0),
                r(x.pings_per_hour_mean, 1)]
    for i, (got, want) in enumerate(zip(vals, exp)):
        check(got == want, f"Table 1 {label} col {i}: tex {got} vs data {want}")
        cells[(label, SCN[i // 4], ["J", "succ", "p95", "pings"][i % 4])] = got
# bold = column winner among the four policies (exact, not rounded)
for qi, q in enumerate(SCN):
    sub = s[(s.scenario == q) & s.policy.isin(pol.values())].set_index("policy")
    winners = {"J": sub.J_mean.idxmax(), "p95": sub.p95_latency_all_ms_mean.idxmin(),
               "pings": sub.pings_per_hour_mean.idxmin()}
    for label, p in pol.items():
        row_tex = next(ln for ln in TEX.splitlines() if ln.strip().startswith(label))
        parts = [v.strip() for v in row_tex.split("&")[1:]]
        check(not parts[qi * 4 + 1].startswith("\\textbf"), f"success column should not be bold {label} {q}")
        for mi, m in zip([0, 2, 3], ["J", "p95", "pings"]):
            bold = parts[qi * 4 + mi].startswith("\\textbf")
            check(bold == (winners[m] == p), f"Table 1 bold mismatch {label} {q} {m}")

# ------------------------------------------------- Table 1 delta rows (Holm) ---
pt = pd.read_csv(EV / "primary_tests.csv")
for comp, lab in [("kademlia_epoch", "Kademlia-epoch"), ("tuned_heuristic", "Tuned heuristic"),
                  ("matched_random", "Matched random")]:
    line = next(ln for ln in TEX.splitlines() if ln.strip().startswith(f"$\\Delta J$, RL $-$ {lab}"))
    for q in SCN:
        x = pt[(pt.scenario == q) & (pt.comparator == comp)].iloc[0]
        want = f"${sgn(x.mean_diff, 4)}$ [${sgn(x.ci_lo, 4)}$, ${sgn(x.ci_hi, 4)}$]"
        check(want in line, f"Delta row {lab} {q}: expected {want}")
check(bool(pt.significant.all()), "not all 9 primary tests significant")
check(len(pt) == 9, "primary test count != 9")
in_tex(f"p_{{\\mathrm{{Holm}}}}\\le {pt.p_holm.max():.1e}".replace("e-04", "\\times10^{-4}"),
       f"max p_Holm {pt.p_holm.max():.3g} not as stated")

# ping change and ranges
pc = []
for q in SCN:
    k = s[(s.policy == "kademlia_epoch") & (s.scenario == q)].pings_per_hour_mean.iloc[0]
    rl = s[(s.policy == "rl_mean") & (s.scenario == q)].pings_per_hour_mean.iloc[0]
    pc.append(100 * (rl / k - 1))
for v in pc:
    in_tex(f"${v:.1f}\\%$", f"ping change {v:.1f}% missing")
check(round(-pc[1]) == 87 and round(-pc[0]) == 98, "87--98% range")
in_tex("87--98\\%")
kp = s[(s.policy == "kademlia_epoch") & s.scenario.isin(SCN)].pings_per_hour_mean
rp = s[(s.policy == "rl_mean") & s.scenario.isin(SCN)].pings_per_hour_mean
in_tex(f"({kp.min():.0f}--{kp.max():.0f} versus {rp.min():.0f}--{rp.max():.0f})", "ping ranges")
dk = pt[pt.comparator == "kademlia_epoch"].mean_diff
in_tex(f"$\\Delta J={dk.min():+.4f}$ to ${dk.max():+.4f}$", "Delta J vs Kademlia range")
dt = pt[pt.comparator == "tuned_heuristic"].mean_diff
in_tex(f"$\\Delta J={dt.max():.4f}$ to ${dt.min():.4f}$".replace("=-", "=-"), "Delta J vs tuned range")
dm = pt[pt.comparator == "matched_random"].mean_diff
in_tex(f"$\\Delta J={dm.min():+.3f}$ to ${dm.max():+.3f}$", "Delta J vs matched range")

# --------------------------------------------- per-seed derived statements ---
ps = pd.read_csv(EV / "per_seed.csv")
pr = pd.read_csv(EV / "per_run.csv")
succ_k = pr[(pr.metric == "success_rate") & (pr.comparator == "kademlia_epoch")].mean_diff.abs().max()
check(succ_k * 100 <= 0.2, f"per-run success diff vs Kademlia {succ_k*100:.3f} pp > 0.2")
from churnmaster.experiments import stats as st  # noqa: E402

rl = ps[ps.policy.str.startswith("rl_run")].groupby(["scenario", "episode"]).mean(numeric_only=True)
msg, p95k = [], {}
for q in SCN:
    a = rl.loc[q].sort_index()
    b = ps[(ps.policy == "kademlia_epoch") & (ps.scenario == q)].set_index("episode").sort_index()
    msg.append(100 * (a.messages_per_lookup.mean() / b.messages_per_lookup.mean() - 1))
    p95k[q] = st.paired_bootstrap_ci(a.p95_latency_all_ms.values, b.p95_latency_all_ms.values,
                                     n_resamples=10_000, level=0.95, seed=0)
check(all(-3.5 < m < -2.0 for m in msg), f"messages/lookup change {msg} not ~3%")
in_tex("only about 3\\%")
check(p95k["stable"].lo > 0 and p95k["heldout_medium"].lo > 0, "RL p95 > Kademlia not significant")
check(p95k["high_churn"].lo < 0 < p95k["high_churn"].hi, "high_churn p95 diff should include 0")
in_tex(f"(${p95k['stable'].mean:+.0f}$ and ${p95k['heldout_medium'].mean:+.0f}$\\,ms)", "p95 +85/+55")

# ------------------------------------------------------------ hypotheses ---
hy = pd.read_csv(EV / "hypotheses.csv").set_index("id")
for h in ["H1", "H2", "H3", "H4", "H5"]:
    line = next(ln for ln in TEX.splitlines() if ln.strip().startswith(h + " &"))
    check(line.rstrip().rstrip("\\").strip().endswith(hy.loc[h, "verdict"]),
          f"{h} verdict should be {hy.loc[h, 'verdict']}")
ratios = [float(x) for x in re.findall(r"ratio ([0-9.]+)", hy.loc["H3", "evidence"])]
in_tex("ratios " + ", ".join(f"{x:.2f}" for x in ratios[:2]) + f" and {ratios[2]:.2f}", "H3 ratios")
check(round((min(ratios) - 1) * 100) == 18 and round((max(ratios) - 1) * 100) == 38, "18--38%")
in_tex("18--38\\%")


# ---------------------------------------------------- success-rate claims ---
h2 = hy.loc["H2", "evidence"]
mk = re.search(r"kademlia_epoch: dsucc=([-+0-9.e]+) \[([-+0-9.e]+),([-+0-9.e]+)\]", h2)
mt = re.search(r"tuned_heuristic: dsucc=([-+0-9.e]+) \[([-+0-9.e]+),([-+0-9.e]+)\]", h2)
fk = [100 * float(g) for g in mk.groups()]; ft = [100 * float(g) for g in mt.groups()]
in_tex(f"RL $-$ Kademlia-epoch is ${fk[0]:+.2f}$\\,pp [${fk[1]:+.2f}$, ${fk[2]:+.2f}$]", "H2 success diff vs Kademlia")
in_tex(f"RL $-$ tuned heuristic is ${ft[0]:+.2f}$\\,pp [${ft[1]:+.2f}$, ${ft[2]:+.2f}$]", "H2 success diff vs tuned")
rs_ = s[(s.policy == "rl_mean") & s.scenario.isin(SCN)].success_rate_mean * 100
in_tex(f"{rs_.min():.2f}--{rs_.max():.2f}\\% for RL", "RL success range")
# 13 labels = 10 baselines + 3 RL runs
labels = set(ps.policy)
rl_runs = {x for x in labels if x.startswith("rl_run")}
check(len(rl_runs) == 3 and len(labels - rl_runs) == 10, "10 baselines + 3 RL runs")
named = ["kademlia\\_epoch", "matched\\_random", "tuned\\_heuristic", "never\\_replace", "always\\_replace",
         "uniform\\_random", "kademlia\\_std", "the default heuristic", "privileged per-scenario heuristic", "\\pol{oracle}"]
for nm in named:
    in_tex(nm, f"baseline not described: {nm}")
in_tex("the 13 labels are the 10 baselines above plus the 3 independently trained RL runs")
# table order: state table caption precedes evaluation table caption in source
check(TEX.index("\\label{tab:state}") < TEX.index("\\label{tab:main}") < TEX.index("\\label{tab:hyp}"),
      "table source order state < main < hyp")

# ------------------------------------------------- protocol / manifest facts ---
m = json.loads((EV / "eval_manifest.json").read_text(encoding="utf-8"))
check(m["n_jobs"] == 780 and len(ps) == 780, "780 evaluation episodes")
check(ps.policy.nunique() == 13, "13 policy labels")
in_tex("780 episodes ($13$ policy labels $\\times$ 3 scenarios $\\times$ 20 seeds)")
check(m["eval_lock"]["disjoint"]["n_other_seed_states"] == 635 and
      m["eval_lock"]["disjoint"]["overlap"] is False, "635 seed states, no overlap")
in_tex("635 other seed states")
check(m["prior_eval_runs"] == 0, "evaluation ran once")
check(m["stats"]["bootstrap_resamples"] == 10000, "10000 resamples")
in_tex("10\\,000 resamples")

# ----------------------------------------------------------- configs ---------
base = tomllib.loads((ROOT / "configs" / "base.toml").read_text(encoding="utf-8"))
sim, rw, ag = base["sim"], base["reward"], base["agent"]
for cond, what in [
    (sim["n_nodes"] == 128 and sim["id_bits"] == 8, "128 nodes / 8-bit IDs"),
    (sim["k"] == 4 and sim["alpha"] == 2 and sim["replicas"] == 3, "k/alpha/replicas"),
    (sim["cache_size"] == 4 and sim["cache_ttl_s"] == 300, "cache 4, TTL 300 s"),
    (sim["p_loss"] == 0.02 and sim["rebuild_period_s"] == 300, "loss 0.02, rebuild 300 s"),
    (sim["lookup_rate_hz"] == 1.0 and sim["deadline_ms"] == 2000, "1/s lookups, 2000 ms"),
    (sim["warmup_s"] == 600 and sim["measured_s"] == 2400 and sim["epoch_s"] == 30, "episode timing"),
    (sim["window_timeout_s"] == 120, "churn window 120 s"),
    ((rw["w_L"], rw["w_M"], rw["w_R"]) == (0.3, 0.1, 0.15), "reward weights"),
    ((ag["eps_start"], ag["eps_end"], ag["eps_decay_frac"]) == (1.0, 0.05, 0.6), "epsilon schedule"),
    (ag["alpha_exp"] == 0.7 and ag["episodes"] == 200 and ag["runs"] == 3, "alpha_exp/episodes/runs"),
    (ag["validate_every"] == 20, "validate every 20"),
]:
    check(cond, f"config: {what}")
for t in ["128-node", "8-bit", "$k=4$", "$\\alpha_{\\mathrm{L}}=2$", "3 closest online replicas",
          "TTL 300\\,s", "probability 0.02", "every 300\\,s", "1/s", "2000\\,ms deadline",
          "600\\,s of warm-up", "2400\\,s of measurement", "every 30\\,s", "$w_L=0.3$",
          "$w_M=0.1$", "$w_R=0.15$", "from 1.0 to 0.05 over the first 60\\%", "(1+n)^{-0.7}",
          "3 independent runs of 200 episodes", "every 20 episodes on 5 separate seeds"]:
    in_tex(t)
tr = tomllib.loads((ROOT / "configs" / "experiments" / "train.toml").read_text(encoding="utf-8"))
check(tr["experiment"]["validation_seeds"] == 5, "5 validation seeds")
check(len(tr["experiment"]["grid"]["gamma"]) * len(tr["experiment"]["grid"]["alpha_min"]) == 4, "4 grid points")
sel = json.loads((ROOT / "results" / "train" / "grid" / "selected.json").read_text(encoding="utf-8"))
check((sel["gamma"], sel["alpha_min"]) == (0.8, 0.02), "grid selection")
in_tex("$\\gamma=0.8$ and $\\alpha_{\\min}=0.02$")
tune = json.loads((ROOT / "results" / "tune" / "selected.json").read_text(encoding="utf-8"))
check(tune["n_configs"] == 32, "32 heuristic configs")
scen = {q: tomllib.loads((ROOT / "configs" / "scenarios" / f"{q}.toml").read_text(encoding="utf-8"))["scenario"]
        for q in SCN}
want = {"stable": (0.70, 14400, 1800, 1200), "high_churn": (0.30, 3600, 300, 600),
        "heldout_medium": (0.45, 7200, 600, 720)}
for q, w in want.items():
    c = scen[q]
    check((c["stable_share"], c["stable_session_mean_s"], c["flaky_session_mean_s"],
           c["downtime_mean_s"]) == w and c["weibull_shape"] == 0.6, f"scenario {q}")
for t in ["70\\% stable nodes", "4\\,h (stable) and 30\\,min (flaky)", "20\\,min downtimes",
          "30\\%, 1\\,h/5\\,min sessions and 10\\,min downtimes", "45\\%, 2\\,h/10\\,min sessions and 12\\,min",
          "shape 0.6"]:
    in_tex(t)

# ------------------------------------------------------------ state / code ---
from churnmaster.rl import state as S  # noqa: E402

check(S.RADICES == (3, 3, 2, 3, 3, 2, 2) and S.N_STATES == 648 and S.N_REACHABLE == 540, "state space")
check((S.INC_STALE_EDGE_S, S.INC_AGE_MID_EDGE_S, S.INC_AGE_OLD_EDGE_S, S.CAND_FRESH_EDGE_S,
       S.CAND_FASTER_RATIO, S.CHURN_EDGE) == (60.0, 120.0, 1200.0, 30.0, 0.7, 0.10), "state thresholds")
check(S.N_STATES - S.N_REACHABLE == 108, "108 unreachable")
for t in ["3 \\times 3 \\times 2 \\times 3 \\times 3 \\times 2 \\times 2 = 648", "Only 540", "108 combinations",
          "$<60$\\,s", "$<120$, $120$--$1200$, $\\ge 1200$\\,s", "fresh ($<30$\\,s)", "$<0.7\\times$",
          "$<0.10$, $\\ge 0.10$", "$648\\times3$"]:
    in_tex(t)
from churnmaster.rl.reward import lookup_loss  # noqa: E402

check(lookup_loss(False, 5000, 0.3) == 1.3 and abs(lookup_loss(True, 1000, 0.3) - 0.15) < 1e-12, "loss formula")
qsrc = (ROOT / "src" / "churnmaster" / "rl" / "qlearning.py").read_text(encoding="utf-8")
check("target   = r + gamma * max_{a' in A(s')} Q[s', a']" in qsrc and
      "max(alpha_min, (1 + N[s, a]) ** -alpha_exp)" in qsrc, "Q update / step size in code")
check("KEEP < PROBE < REPLACE" in qsrc, "tie order")
diag = json.loads((ROOT / "results" / "diagnostics" / "diagnostics.json").read_text(encoding="utf-8"))
check(f"{diag['identity']['pooled']['max_abs_err_J']:.1e}" == "3.3e-16", "identity error")
in_tex("3.3\\times10^{-16}")
nr = s[(s.policy == "never_replace") & s.scenario.isin(SCN)].success_rate_mean
in_tex(f"{100*nr.min():.1f}--{100*nr.max():.1f}\\%", "no-maintenance success range")

# ------------------------------------------------------------ ablations ------
AB = ROOT / "results" / "ablations"
ac = pd.read_csv(AB / "agent" / "comparison.csv")
sb = ac[(ac.comparison == "shuffled_vs_buffered") & (ac.metric == "J")].mean_diff
check(-0.012 <= sb.min() and sb.max() <= -0.008, f"shuffle effect {list(sb)}")
in_tex("about 0.008--0.012")
g0 = ac[(ac.comparison == "vs_canonical") & (ac.variant == "gamma0") & (ac.metric == "J")]
check(bool((g0.ci_hi < 0).all()) and len(g0) == 3, "gamma0 worse in all three")
vt = ac[(ac.comparison == "vs_tuned_heuristic") & (ac.metric == "J") & (ac.variant != "canonical")]
check(vt.variant.nunique() == 11 and bool((vt.ci_hi < 0).all()), "11 variants all below tuned")
in_tex("None of the 11 variants")
rc = pd.read_csv(AB / "reward" / "comparison.csv")
x = rc[(rc.comparison == "vs_canonical") & (rc.cell == "wL1.0_wM0.1") & (rc.scenario == "stable")
       & (rc.metric == "p95_latency_all_ms")].iloc[0]
in_tex(f"${x.mean_diff:.0f}$\\,ms in stable at $w_L=1.0$", "wL=1.0 stable p95")
rs = pd.read_csv(AB / "reward" / "summary.csv")
hp = rs[(rs.kind == "rl_mean") & (rs.cell == "wL1.0_wM0.1")].pings_per_hour_mean
cp = rs[(rs.kind == "rl_mean") & (rs.policy == "canonical")].pings_per_hour_mean
in_tex(f"{hp.min():.0f}--{hp.max():.0f} instead of {cp.min():.0f}--{cp.max():.0f} pings/h", "sweep ping ranges")
y = rc[(rc.comparison == "vs_tuned_heuristic") & (rc.cell == "wL1.0_wM0.1") & (rc.scenario == "high_churn")
       & (rc.metric == "p95_latency_all_ms")].iloc[0]
check(y.ci_lo > 0, "high_churn p95 above tuned at wL=1.0")
vtj = rc[(rc.comparison == "vs_tuned_heuristic") & (rc.metric == "J") & (rc.cell != "wL0.3_wM0.1")]
pos = vtj[vtj.ci_lo > 0]
check(len(vtj) == 36 and len(pos) == 1 and pos.iloc[0].scenario == "stable", "one cell beats tuned (stable)")
in_tex("36 combinations")
md = (AB / "memoryless" / "summary.md").read_text(encoding="utf-8")
check("| tuned_heuristic | +0.0007 [-0.0004, +0.0018]" in md, "memoryless gap shift")
in_tex("($+0.0007$ [$-0.0004$, $+0.0018$])")

# ------------------------------------------------------------ format checks --
check("thebibliography" in TEX, "references section present")
import re as _re
keys = set(_re.findall(r"\\bibitem\{([^}]+)\}", TEX))
cited = {k.strip() for c in _re.findall(r"\\cite\{([^}]+)\}", TEX) for k in c.split(",")}
check(keys == cited and len(keys) == 5, f"cited {sorted(cited)} vs bib {sorted(keys)}")
for name in ["Deepansh Singh", "Ankur Kumar"]:
    check(name in TEX, f"author name missing: {name}")
for name in ["Bagwari", "Mertia", "Bohra", "Jahanvi", "Vasishth"]:
    check(name not in TEX, f"old author name present: {name}")

print(f"{n_ok} checks passed, {len(fails)} failed")
for f in fails:
    print("  FAIL:", f)
sys.exit(1 if fails else 0)
