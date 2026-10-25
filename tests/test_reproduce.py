"""Reproduce part A: output guard, figures-only regeneration, report plots (Req. 26.3, 26.4)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churnmaster import plots
from churnmaster.cli import main
from churnmaster.experiments.reproduce import (CANONICAL_DIRS, CANONICAL_NAMES, FIGURE_NAMES,
                                               TABLE_NAMES, figures_only, guard_out_dir,
                                               markdown_table, write_reproduce_manifest)

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
SCNS = ("stable", "high_churn")


def _is_png(p: Path) -> bool:
    return p.is_file() and p.read_bytes()[:8] == PNG_MAGIC


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Synthetic results tree (mirrors the real file formats, tiny)
# --------------------------------------------------------------------------- #

def _metrics(rng, n):
    return {"J": rng.uniform(0.9, 0.98, n), "success_rate": rng.uniform(0.98, 1.0, n),
            "p95_latency_all_ms": rng.uniform(300, 900, n),
            "pings_per_hour": rng.uniform(0, 400, n), "evictions_per_hour": rng.uniform(0, 300, n)}


def _validation(rng):
    rows = [{"checkpoint": c, "scenario": s, "episode": e, **{k: v[0] for k, v in _metrics(rng, 1).items()},
             "n_decisions": 640} for c in (0, 20, 40) for s in SCNS for e in range(3)]
    return pd.DataFrame(rows)


def _curves(rng, n=30):
    return pd.DataFrame({"episode": np.arange(n), "scenario": [SCNS[i % 2] for i in range(n)],
                         "epsilon": np.linspace(1, 0.05, n), **_metrics(rng, n),
                         "mean_abs_dq": np.geomspace(0.1, 0.001, n), "n_updates": 700,
                         "n_transitions": 700, "n_decisions": 640, "states_visited": 50})


def _greedy(rng):
    rows, st = [], 0
    acts = ("KEEP", "PROBE", "REPLACE")
    for tier in range(3):
        for f in range(3):
            for s in range(2):
                for churn in range(2):
                    rows.append({"state": st, "tier": tier, "inc_fail": f, "inc_stale": s,
                                 "inc_age": 0, "cand": 0, "cand_faster": 0, "churn": churn,
                                 "reachable": True, "q_KEEP": -2.0, "n_KEEP": 5,
                                 "q_PROBE": -2.1, "n_PROBE": 3, "q_REPLACE": np.nan,
                                 "n_REPLACE": 0, "visits": int(rng.integers(0, 50)),
                                 "greedy": acts[(st + f) % 3]})
                    st += 1
    return pd.DataFrame(rows)


def make_results(root: Path, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    final = root / "train" / "final"
    for k in range(2):
        d = final / f"run{k}"
        d.mkdir(parents=True)
        _validation(rng).to_csv(d / "validation.csv", index=False)
        _curves(rng).to_csv(d / "curves.csv", index=False)
        _greedy(rng).to_csv(d / "greedy_table.csv", index=False)
    refs = [{"policy": p, "scenario": s, "episode": e, **{k: v[0] for k, v in _metrics(rng, 1).items()}}
            for p in ("never_replace", "uniform_random", "kademlia_epoch", "heuristic",
                      "oracle_heuristic", "always_replace") for s in SCNS for e in range(3)]
    pd.DataFrame(refs).to_csv(final / "references.csv", index=False)

    ev = root / "eval"
    ev.mkdir(parents=True)
    pols = [("never_replace", -1), ("kademlia_epoch", -1), ("tuned_heuristic", -1),
            ("rl_run0", 0), ("rl_run1", 1)]
    per = [{"policy": p, "policy_impl": p, "privileged": False, "privileged_fallback": False,
            "reference": False, "rl_run": r, "scenario": s, "split": "eval", "run": 0, "episode": e,
            "shares_decision_points": True, **{k: v[0] for k, v in _metrics(rng, 1).items()},
            "n_lookups": 2000, "n_decisions": 640}
           for p, r in pols for s in SCNS for e in range(4)]
    per = pd.DataFrame(per)
    per.to_csv(ev / "per_seed.csv", index=False)
    summ = []
    for (p, s), g in per.groupby(["policy", "scenario"], sort=False):
        row = {"policy": p, "scenario": s, "policy_impl": p, "shares_decision_points": True,
               "privileged": False, "privileged_fallback": False, "reference": False, "n": len(g)}
        for m in ("J", "success_rate", "p95_latency_all_ms", "pings_per_hour", "evictions_per_hour"):
            row[f"{m}_mean"], row[f"{m}_std"] = g[m].mean(), g[m].std()
        summ.append(row)
    for s in SCNS:
        summ.append({**summ[0], "policy": "rl_mean", "scenario": s, "policy_impl": "rl_mean"})
    pd.DataFrame(summ).to_csv(ev / "summary.csv", index=False)
    pd.DataFrame([{"scenario": s, "comparator": c, "metric": "J", "n": 4, "mean_diff": 0.01,
                   "ci_lo": 0.005, "ci_hi": 0.02, "wilcoxon_stat": 0.0, "p": 0.01, "p_holm": 0.03,
                   "significant": True, "direction": 1}
                  for s in SCNS for c in ("kademlia_epoch", "tuned_heuristic")]
                 ).to_csv(ev / "primary_tests.csv", index=False)
    pd.DataFrame([{"id": "H1", "statement": "RL > matched_random, everywhere", "verdict": "pass",
                   "evidence": "stable: dJ=+0.01 | high_churn: dJ=+0.02"},
                  {"id": "H2", "statement": "non-inferior", "verdict": "inconclusive",
                   "evidence": "n/a"}]).to_csv(ev / "hypotheses.csv", index=False)

    hd = root / "headroom"
    hd.mkdir(parents=True)
    pd.DataFrame([{"policy": p, "scenario": s, "split": "calibration", "run": 0, "episode": e,
                   **{k: v[0] for k, v in _metrics(rng, 1).items()}}
                  for p in ("never_replace", "kademlia_epoch", "oracle_heuristic")
                  for s in SCNS for e in range(3)]).to_csv(hd / "per_seed.csv", index=False)
    ci = {"mean": 0.02, "lo": 0.01, "hi": 0.03, "n": 3, "level": 0.95}
    gates = {"split": "calibration", "n_seeds": 3, "gates": {
        "scenario": "high_churn", "thresholds": {"g1_min_dJ": 0.01},
        "G1": {"passed": True, "route": "J", "best": "oracle_heuristic", "candidates": {
            "oracle_heuristic": {"dJ": ci, "p95_diff_ms": ci, "p95_rel": 0.5, "j_route": True,
                                 "p95_route": True, "passes": True}}},
        "G2": {"dJ": ci, "ping_ratio": 0.1, "passed": True, "route": "J",
               "limited_headroom": False},
        "G2_info": {"stable": {"dJ": ci, "ping_ratio": 0.2, "passed": False}}}}
    (hd / "gates.json").write_text(json.dumps(gates), encoding="utf-8")
    (hd / "manifest.json").write_text(json.dumps({"frozen": True, "eval_lock": {}}), encoding="utf-8")

    dg = root / "diagnostics"
    dg.mkdir(parents=True)
    pd.DataFrame([{"scope": sc, "state": i, "visits": int(rng.integers(0, 200)), "covered": False,
                   "tier": 0, "inc_fail": 0, "inc_stale": 0, "inc_age": 0, "cand": 0,
                   "cand_faster": 0, "churn": 0} for sc in ("high_churn", "stable", "pooled")
                  for i in range(20)]).to_csv(dg / "coverage.csv", index=False)
    (dg / "diagnostics.json").write_text(json.dumps(
        {"coverage": {s: {"min_visits": 50} for s in ("high_churn", "stable", "pooled")}}),
        encoding="utf-8")

    tu = root / "tune"
    tu.mkdir(parents=True)
    rows = []
    for i in range(8):
        r = {"rank": i + 1, "config": f"f={i},po=0", "policy": f"heuristic:f={i},po=0",
             "is_default": i == 6, "J_overall": 0.97 - 0.001 * i,
             "success_rate_overall": 0.999, "p95_latency_all_ms_overall": 400.0 + i,
             "pings_per_hour_overall": 20.0 + 10 * i, "global_best": i == 0}
        for s in SCNS:
            r.update({f"J_mean_{s}": 0.97 - 0.001 * i, f"J_std_{s}": 0.002, f"n_{s}": 10,
                      f"success_rate_mean_{s}": 0.999, f"p95_latency_all_ms_mean_{s}": 400.0,
                      f"pings_per_hour_mean_{s}": 20.0 + 10 * i,
                      f"privileged_best_{s}": i == (0 if s == "stable" else 1)})
        rows.append(r)
    pd.DataFrame(rows).to_csv(tu / "summary.csv", index=False)
    return root


@pytest.fixture
def results(tmp_path):
    return make_results(tmp_path / "res")


def _tree_hashes(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): _sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


EXPECTED_FIGURES = ("validation_curves.png", "training_curves.png", "eval_comparison.png",
                    "eval_success_vs_pings_stable.png", "eval_success_vs_pings_high_churn.png",
                    "greedy_policy_heatmap.png", "headroom_J_p95.png",
                    "diagnostics_state_visits.png", "tuning_scatter.png")


# --------------------------------------------------------------------------- #
# Output guard
# --------------------------------------------------------------------------- #

def test_guard_accepts_tmp_and_figures_dirs(tmp_path):
    assert guard_out_dir(tmp_path / "out") == (tmp_path / "out").resolve()
    assert guard_out_dir(REPO / "results" / "figures").name == "figures"
    assert guard_out_dir(REPO / "results" / "reproduce_quick" / "x").name == "x"


@pytest.mark.parametrize("name", CANONICAL_NAMES)
def test_guard_rejects_canonical_dirs_and_their_contents(name):
    with pytest.raises(ValueError):
        guard_out_dir(REPO / "results" / name)
    with pytest.raises(ValueError):
        guard_out_dir(REPO / "results" / name / "sub")


def test_guard_rejects_results_root_and_parents(monkeypatch):
    monkeypatch.chdir(REPO)
    assert len(CANONICAL_DIRS) == len(CANONICAL_NAMES)
    for bad in ("results", "results/", ".", str(REPO), str(REPO.parent), "results/eval/../train"):
        with pytest.raises(ValueError):
            guard_out_dir(bad)


def test_guard_protects_extra_results_root(results):
    with pytest.raises(ValueError):
        guard_out_dir(results, extra_roots=[results])
    with pytest.raises(ValueError):
        guard_out_dir(results / "eval" / "figs", extra_roots=[results])
    with pytest.raises(ValueError):
        guard_out_dir(results.parent, extra_roots=[results])
    assert guard_out_dir(results / "figures", extra_roots=[results]).name == "figures"


# --------------------------------------------------------------------------- #
# figures_only
# --------------------------------------------------------------------------- #

def test_figures_only_writes_every_figure_table_and_manifest(results, tmp_path):
    before = _tree_hashes(results)
    out = tmp_path / "figs"
    files = figures_only(results, out)

    for name in EXPECTED_FIGURES:
        assert _is_png(out / name), name
    for name in TABLE_NAMES:
        text = (out / "tables" / name).read_text(encoding="utf-8")
        assert text.startswith("# ") and "| --- |" in text, name
    assert files[-1] == out / "figures_manifest.json"
    man = json.loads(files[-1].read_text(encoding="utf-8"))
    assert man["skipped"] == []
    assert {o["path"] for o in man["outputs"]} == (
        set(EXPECTED_FIGURES) | {f"tables/{t}" for t in TABLE_NAMES})
    for o in man["outputs"]:
        assert o["sha256"] == _sha(out / o["path"])
    for rel, h in man["inputs"].items():
        assert h == before[rel]
    assert "eval/per_seed.csv" in man["inputs"] and "tune/summary.csv" in man["inputs"]
    assert "headroom/manifest.json" not in man["inputs"]
    assert _tree_hashes(results) == before  # inputs untouched

    summary = (out / "tables" / "eval_summary.md").read_text(encoding="utf-8")
    assert "rl_mean" in summary and "±" in summary
    top = (out / "tables" / "tuning_top5.md").read_text(encoding="utf-8")
    assert "f=0,po=0" in top and "global best" in top
    assert "f=6,po=0" in top and "default" in top  # default listed although rank 7
    hyp = (out / "tables" / "hypotheses.md").read_text(encoding="utf-8")
    assert "\\|" in hyp  # pipe in evidence is escaped
    gates = (out / "tables" / "headroom_gates.md").read_text(encoding="utf-8")
    assert "G1: PASS" in gates and "oracle_heuristic" in gates


def test_figures_only_skips_missing_inputs(results, tmp_path):
    (results / "tune" / "summary.csv").unlink()
    for d in (results / "train" / "final").glob("run*"):
        (d / "greedy_table.csv").unlink()
    (results / "eval" / "hypotheses.csv").unlink()
    out = tmp_path / "figs"
    with pytest.warns(RuntimeWarning, match="skipped"):
        figures_only(results, out)
    man = json.loads((out / "figures_manifest.json").read_text(encoding="utf-8"))
    skipped = {s["output"] for s in man["skipped"]}
    assert skipped == {"tuning_scatter.png", "greedy_policy_heatmap.png", "tuning_top5.md",
                       "hypotheses.md"}
    assert all("missing input" in s["reason"] for s in man["skipped"])
    assert not (out / "tuning_scatter.png").exists()
    assert _is_png(out / "validation_curves.png") and _is_png(out / "eval_comparison.png")


def test_figures_only_works_without_optional_extras(results, tmp_path):
    (results / "train" / "final" / "references.csv").unlink()
    (results / "eval" / "summary.csv").unlink()
    (results / "diagnostics" / "diagnostics.json").unlink()
    out = tmp_path / "figs"
    with pytest.warns(RuntimeWarning):
        figures_only(results, out)
    man = json.loads((out / "figures_manifest.json").read_text(encoding="utf-8"))
    assert {s["output"] for s in man["skipped"]} == {"eval_summary.md"}
    assert _is_png(out / "validation_curves.png") and _is_png(out / "diagnostics_state_visits.png")


def test_figures_only_refuses_canonical_out(results):
    with pytest.raises(ValueError):
        figures_only(results, results / "eval")
    with pytest.raises(ValueError):
        figures_only(results, results)


def test_cli_figures_only_smoke(results, tmp_path, capsys):
    out = tmp_path / "cli_figs"
    rc = main(["reproduce", "--figures-only", "--results-dir", str(results), "--out", str(out)])
    assert rc == 0
    assert "skipped 0" in capsys.readouterr().out
    assert (out / "figures_manifest.json").is_file()
    assert all(_is_png(out / n) for n in EXPECTED_FIGURES)
    rc = main(["reproduce", "--figures-only", "--results-dir", str(results),
               "--out", str(results / "headroom")])
    assert rc == 1 and "canonical" in capsys.readouterr().err


def test_figure_names_cover_expected():
    assert len(FIGURE_NAMES) == 8 and len(TABLE_NAMES) == 5


def test_markdown_table_escapes_and_formats():
    md = markdown_table(["a", "b"], [["x|y", 0.123456], [True, float("nan")]])
    lines = md.strip().splitlines()
    assert lines[1] == "| --- | --- |"
    assert "x\\|y" in lines[2] and "0.1235" in lines[2] and "nan" in lines[3]


# --------------------------------------------------------------------------- #
# Reproduce manifest
# --------------------------------------------------------------------------- #

def test_write_reproduce_manifest(tmp_path):
    import argparse

    stages = [{"name": "train", "wall_s": 1.5}, {"name": "evaluate", "wall_s": 2.0}]
    ns = argparse.Namespace(quick=True, workers=2, func=print)
    data = write_reproduce_manifest(tmp_path / "m" / "manifest.json", "quick", ns, stages,
                                    {"note": "x"}, configs_dir=CONFIGS,
                                    experiment_path=CONFIGS / "experiments" / "eval.toml")
    on_disk = json.loads((tmp_path / "m" / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk["mode"] == "quick" and on_disk["note"] == "x"
    assert on_disk["args"] == {"quick": True, "workers": 2}
    assert set(on_disk["config_hashes"]) >= {"stable", "high_churn", "heldout_medium"}
    assert all(len(h) == 64 for h in on_disk["config_hashes"].values())
    assert set(on_disk["env_hashes"]) == set(on_disk["config_hashes"])
    assert "base.toml" in on_disk["config_files"]
    for pkg in ("numpy", "scipy", "pandas", "matplotlib", "pytest", "hypothesis"):
        assert on_disk["packages"][pkg]
    assert on_disk["python"] and "git_hash" in on_disk
    assert on_disk["total_wall_s"] == pytest.approx(3.5)
    assert data["stages"] == stages


# --------------------------------------------------------------------------- #
# New plots.py functions
# --------------------------------------------------------------------------- #

def test_plot_validation_curves(tmp_path):
    rng = np.random.default_rng(1)
    refs = make_results(tmp_path / "r") / "train" / "final" / "references.csv"
    p = plots.plot_validation_curves([("run0", _validation(rng)), ("run1", _validation(rng))],
                                     tmp_path / "v.png", references=pd.read_csv(refs))
    assert _is_png(p)
    assert _is_png(plots.plot_validation_curves([("run0", _validation(rng))], tmp_path / "v2.png"))
    with pytest.raises(ValueError):
        plots.plot_validation_curves([], tmp_path / "x.png")


def test_plot_training_curves(tmp_path):
    rng = np.random.default_rng(2)
    assert _is_png(plots.plot_training_curves([("run0", _curves(rng)), ("run1", _curves(rng))],
                                              tmp_path / "t.png", window=5))
    with pytest.raises(ValueError):
        plots.plot_training_curves([("run0", _curves(rng))], tmp_path / "x.png", window=0)


def test_plot_comparison_bars(tmp_path, results):
    df = pd.read_csv(results / "eval" / "per_seed.csv")
    for err in ("ci95", "std"):
        assert _is_png(plots.plot_comparison_bars(df, tmp_path / f"c_{err}.png", error=err))
    with pytest.raises(ValueError):
        plots.plot_comparison_bars(df, tmp_path / "x.png", error="sem")
    with pytest.raises(ValueError):
        plots.plot_comparison_bars(df, tmp_path / "x.png", metrics=("nope",))


def test_greedy_share_matrix_visit_weighted():
    t = pd.DataFrame([
        {"tier": 0, "inc_fail": 2, "inc_stale": 1, "visits": 30, "greedy": "REPLACE", "reachable": True},
        {"tier": 0, "inc_fail": 2, "inc_stale": 1, "visits": 10, "greedy": "KEEP", "reachable": True},
        {"tier": 0, "inc_fail": 2, "inc_stale": 1, "visits": 99, "greedy": "KEEP", "reachable": False},
        {"tier": 0, "inc_fail": 0, "inc_stale": 0, "visits": 0, "greedy": "PROBE", "reachable": True},
        {"tier": 1, "inc_fail": 0, "inc_stale": 0, "visits": 5, "greedy": "PROBE", "reachable": True},
    ])
    share, visits = plots.greedy_share_matrix(t, "REPLACE", 0)
    assert share.shape == (3, 2) and visits[2, 1] == 40
    assert share[2, 1] == pytest.approx(0.75)
    assert np.isnan(share[0, 0]) and visits[0, 0] == 0
    share1, _ = plots.greedy_share_matrix(t, "PROBE", 1)
    assert share1[0, 0] == pytest.approx(1.0)


def test_plot_greedy_heatmap(tmp_path):
    rng = np.random.default_rng(3)
    assert _is_png(plots.plot_greedy_heatmap([_greedy(rng), _greedy(rng)], tmp_path / "g.png"))
    with pytest.raises(ValueError):
        plots.plot_greedy_heatmap([], tmp_path / "x.png")


def test_plot_tuning_scatter(tmp_path, results):
    s = pd.read_csv(results / "tune" / "summary.csv")
    assert _is_png(plots.plot_tuning_scatter(s, tmp_path / "s.png"))
    with pytest.raises(ValueError):
        plots.plot_tuning_scatter(s.iloc[0:0], tmp_path / "x.png")


# --------------------------------------------------------------------------- #
# Part B: quick / full pipelines
# --------------------------------------------------------------------------- #

from churnmaster.experiments import reproduce as rp  # noqa: E402


def _per_seed_table(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [{"policy": p, "scenario": s, "split": "eval", "run": 0, "episode": e,
             **{k: v[0] for k, v in _metrics(rng, 1).items()}, "trace_hash": f"h{p}{s}{e}"}
            for p in ("never_replace", "rl_run0") for s in SCNS for e in range(3)]
    return pd.DataFrame(rows)


def test_compare_per_seed_identical_and_perturbed(tmp_path):
    df = _per_seed_table()
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    df.to_csv(a, index=False)
    df.to_csv(b, index=False)
    same = rp.compare_per_seed(a, b)
    assert same["ok"] and same["sha256_equal"] and same["differing_columns"] == []
    assert all(v == 0.0 for v in same["max_abs_diff"].values())

    # Row order does not matter; values do.
    df.iloc[::-1].to_csv(b, index=False)
    shuffled = rp.compare_per_seed(a, b)
    assert shuffled["ok"] and not shuffled["sha256_equal"]

    pert = df.copy()
    pert.loc[3, "J"] += 1e-9
    pert.loc[5, "pings_per_hour"] -= 2.5
    pert.loc[0, "trace_hash"] = "other"
    pert.to_csv(b, index=False)
    d = rp.compare_per_seed(a, b)
    assert not d["ok"] and d["keys_equal"] and d["columns_equal"]
    assert d["differing_columns"] == ["J", "pings_per_hour", "trace_hash"]
    assert d["max_abs_diff"]["J"] == pytest.approx(1e-9, rel=1e-3)
    assert d["max_abs_diff"]["pings_per_hour"] == pytest.approx(2.5)
    assert d["max_abs_diff"]["success_rate"] == 0.0
    assert d["n_text_diff"]["trace_hash"] == 1

    df.iloc[1:].to_csv(b, index=False)
    missing = rp.compare_per_seed(a, b)
    assert not missing["ok"] and not missing["keys_equal"]
    assert missing["rows_only_reproduced"] == ["never_replace|stable|eval|0|0"]


def test_verify_against_selections(tmp_path):
    canon, out = tmp_path / "canon", tmp_path / "out"
    for root in (canon, out):
        (root / "eval").mkdir(parents=True)
        (root / "tune").mkdir()
        _per_seed_table().to_csv(root / "eval" / "per_seed.csv", index=False)
        (root / "tune" / "selected.json").write_text(json.dumps(
            {"global": "f=2", "per_scenario": {"stable": "f=2"}, "default_config": "d",
             "default_rank": 3, "wall_s": 1.0 if root == canon else 9.0}), encoding="utf-8")
    v = rp.verify_against(out, canon)
    assert v["status"] == "ok"  # wall_s ignored
    assert v["selections"]["tune/selected.json"]["status"] == "ok"
    assert v["selections"]["train/grid/selected.json"]["status"] == "skipped"
    (out / "tune" / "selected.json").write_text(json.dumps(
        {"global": "f=1", "per_scenario": {"stable": "f=2"}, "default_config": "d",
         "default_rank": 3}), encoding="utf-8")
    v = rp.verify_against(out, canon)
    assert v["status"] == "mismatch"
    assert v["selections"]["tune/selected.json"]["differences"] == {
        "global": {"reproduced": "f=1", "canonical": "f=2"}}
    assert rp.verify_against(out, tmp_path / "none")["status"] == "skipped"


def _noop(p):
    return {}


def test_pipeline_records_failed_stage_and_stops(tmp_path, monkeypatch):
    calls = []

    def boom(p):
        calls.append("diagnostics")
        raise RuntimeError("diagnostics exploded")

    monkeypatch.setattr(rp, "_stage_headroom", lambda p: calls.append("headroom") or {"x": 1})
    monkeypatch.setattr(rp, "_stage_diagnostics", boom)
    for name in ("train", "tune", "matched", "evaluate", "figures"):
        monkeypatch.setattr(rp, f"_stage_{name}", lambda p, n=name: calls.append(n) or {})
    res = rp.run_pipeline("quick", tmp_path / "o", configs_dir=CONFIGS, workers=1)
    assert res.exit_code == rp.EXIT_STAGE_FAILED and not res.ok
    assert calls == ["headroom", "diagnostics"]
    man = json.loads((tmp_path / "o" / "reproduce_manifest.json").read_text(encoding="utf-8"))
    assert man["status"] == "failed" and man["exit_code"] == rp.EXIT_STAGE_FAILED
    assert man["failed_stage"] == "diagnostics"
    assert man["not_run"] == ["train", "tune", "matched", "evaluate", "figures"]
    assert [(s["name"], s["status"]) for s in man["stages"]] == [("headroom", "ok"),
                                                                ("diagnostics", "failed")]
    assert man["stages"][0]["x"] == 1
    assert "diagnostics exploded" in man["stages"][1]["error"]
    assert man["settings"] == rp.QUICK_SETTINGS and "git_hash" in man
    assert man["stage_plan"] == list(rp.STAGE_NAMES[:-1])  # no verify in quick mode

    # CLI: same failure -> exit 5; the previous reproduce output dir is reused.
    assert main(["reproduce", "--quick", "--out", str(tmp_path / "o"), "--workers", "1",
                 "--configs-dir", str(CONFIGS)]) == rp.EXIT_STAGE_FAILED


def test_full_pipeline_verify_mismatch_exits_3(tmp_path, monkeypatch):
    canon = tmp_path / "canon"
    (canon / "eval").mkdir(parents=True)
    _per_seed_table().to_csv(canon / "eval" / "per_seed.csv", index=False)

    def fake_eval(p, delta=0.0):
        df = _per_seed_table()
        df.loc[0, "J"] += delta
        (p.out / "eval").mkdir(parents=True)
        df.to_csv(p.out / "eval" / "per_seed.csv", index=False)
        return {}

    monkeypatch.setattr(rp, "_stage_evaluate", fake_eval)
    ok = rp.run_pipeline("full", tmp_path / "o", configs_dir=CONFIGS, workers=1,
                         stages=["evaluate", "verify"], canonical_root=canon)
    assert ok.exit_code == 0 and ok.manifest["verify"]["status"] == "ok"
    monkeypatch.setattr(rp, "_stage_evaluate", lambda p: fake_eval(p, 0.125))
    bad = rp.run_pipeline("full", tmp_path / "o", configs_dir=CONFIGS, workers=1,
                          stages=["evaluate", "verify"], canonical_root=canon)
    assert bad.exit_code == rp.EXIT_VERIFY_MISMATCH
    assert bad.manifest["status"] == "verify_mismatch"
    v = bad.manifest["verify"]
    assert v["per_seed"]["differing_columns"] == ["J"]
    assert v["per_seed"]["max_abs_diff"]["J"] == pytest.approx(0.125)
    assert bad.manifest["stages"][-1]["status"] == "mismatch"


def test_pipeline_refuses_canonical_and_foreign_out(tmp_path):
    for out in (REPO / "results" / "train", REPO / "results", REPO):
        with pytest.raises(ValueError):
            rp.run_pipeline("quick", out, configs_dir=CONFIGS)
    foreign = tmp_path / "f"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="not empty"):
        rp.run_pipeline("quick", foreign, configs_dir=CONFIGS)
    assert (foreign / "keep.txt").is_file()
    with pytest.raises(ValueError, match="unknown settings"):
        rp.run_pipeline("quick", tmp_path / "g", settings_overrides={"nope": 1})
    with pytest.raises(ValueError, match="mode"):
        rp.run_pipeline("medium", tmp_path / "g")


def test_toml_dumps_round_trips_experiment_overrides(tmp_path):
    import tomllib

    from churnmaster.config import load_config, merge, read_toml

    data = merge(read_toml(CONFIGS / "experiments" / "train.toml"),
                 {"sim": {"warmup_s": 60, "measured_s": 240, "p_loss": 0.02}})
    assert tomllib.loads(rp.toml_dumps(data)) == data
    paths = rp._experiment_paths(CONFIGS, tmp_path, {"sim": {"warmup_s": 60, "measured_s": 240}})
    assert set(paths) == {"headroom", "diagnostics", "train", "tune", "matched", "evaluate"}
    for p in paths.values():
        cfg = load_config(CONFIGS / "base.toml", CONFIGS / "scenarios" / "stable.toml", p)
        assert (cfg.sim.warmup_s, cfg.sim.measured_s) == (60, 240)
    cfg = load_config(CONFIGS / "base.toml", CONFIGS / "scenarios" / "stable.toml",
                      paths["evaluate"])
    assert cfg.experiment.name == "eval"
    assert rp._experiment_paths(CONFIGS, tmp_path / "x", None)["diagnostics"] is None


def _canonical_hashes() -> dict[str, str]:
    out = {}
    for d in (*(REPO / c for c in CANONICAL_DIRS), REPO / "results" / "figures"):
        if d.is_dir():
            out.update({f"{d.name}/{k}": v for k, v in _tree_hashes(d).items()})
    return out


@pytest.mark.slow
def test_quick_pipeline_end_to_end_short_episodes(tmp_path):
    from churnmaster.experiments.runner import default_workers

    before = _canonical_hashes()
    out = tmp_path / "quick"
    res = rp.run_pipeline(
        "quick", out, configs_dir=CONFIGS, workers=default_workers(),
        experiment_overrides={"sim": {"warmup_s": 60, "measured_s": 240}},
        settings_overrides={"headroom_seeds": 5, "diag_episodes": 3, "train_episodes": 6,
                            "train_validate_every": 3, "train_validation_seeds": 1,
                            "tune_seeds": 1, "matched_seeds": 1, "eval_seeds": 3})
    man = json.loads((out / "reproduce_manifest.json").read_text(encoding="utf-8"))
    assert res.exit_code == 0, man.get("stages")
    assert man["status"] == "ok" and man["mode"] == "quick"
    assert "git_hash" in man and man["packages"]["numpy"]
    assert set(man["config_hashes"]) >= {"stable", "high_churn", "heldout_medium"}
    assert [s["name"] for s in man["stages"]] == list(rp.STAGE_NAMES[:-1])
    assert all(s["status"] == "ok" and s["wall_s"] >= 0 for s in man["stages"])
    # Stages used OUT's own frozen manifest (with its own eval lock).
    frozen = json.loads((out / "headroom" / "manifest.json").read_text(encoding="utf-8"))
    assert frozen["frozen"] and frozen["eval_lock"]["episodes"] == [0, 1, 2]
    assert len(frozen["eval_runs"]) == 1
    # Tables and figures.
    per = pd.read_csv(out / "eval" / "per_seed.csv")
    assert set(per["episode"]) == {0, 1, 2} and "rl_run0" in set(per["policy"])
    hyp = pd.read_csv(out / "eval" / "hypotheses.csv")
    assert list(hyp["id"]) == ["H1", "H2", "H3", "H4", "H5"]
    for name in ("validation_curves.png", "training_curves.png", "eval_comparison.png",
                 "headroom_J_p95.png", "tuning_scatter.png"):
        assert _is_png(out / "figures" / name), name
    assert (out / "figures" / "tables" / "hypotheses.md").is_file()
    assert _canonical_hashes() == before
