"""Tests for experiments/stats.py paired bootstrap CI (Requirements 22.2, 25.4)."""

import numpy as np
import pandas as pd
import pytest

from churnmaster.experiments.stats import (
    PairedCI,
    paired_bootstrap_ci,
    paired_ci_from_frame,
    paired_differences,
)


def _shifted(shift: float, n: int = 30, seed: int = 123):
    rng = np.random.default_rng(seed)
    b = rng.normal(0.0, 1.0, n)
    a = b + shift + rng.normal(0.0, 0.1, n)
    return a, b


def test_known_shift_ci_contains_shift_and_excludes_zero():
    a, b = _shifted(0.5)
    ci = paired_bootstrap_ci(a, b)
    assert ci.n == 30 and ci.n_resamples == 10_000 and ci.level == 0.95
    assert ci.lo <= 0.5 <= ci.hi
    assert ci.lo <= ci.mean <= ci.hi
    assert ci.excludes_zero


def test_zero_shift_ci_contains_zero():
    a, b = _shifted(0.0)
    ci = paired_bootstrap_ci(a, b)
    assert ci.lo <= 0.0 <= ci.hi
    assert not ci.excludes_zero


def test_negative_shift_excludes_zero():
    a, b = _shifted(-0.5)
    assert paired_bootstrap_ci(a, b).hi < 0


def test_deterministic_for_seed_and_differs_across_seeds():
    a, b = _shifted(0.5)
    assert paired_bootstrap_ci(a, b, seed=7) == paired_bootstrap_ci(a, b, seed=7)
    c1, c2 = paired_bootstrap_ci(a, b, seed=7), paired_bootstrap_ci(a, b, seed=8)
    assert (c1.lo, c1.hi) != (c2.lo, c2.hi)
    assert c1.mean == c2.mean


def test_level_affects_width():
    a, b = _shifted(0.5)
    w90 = paired_bootstrap_ci(a, b, level=0.90)
    w99 = paired_bootstrap_ci(a, b, level=0.99)
    assert (w99.hi - w99.lo) > (w90.hi - w90.lo)


def test_single_pair_degenerate_interval():
    ci = paired_bootstrap_ci([3.0], [1.0])
    assert ci.n == 1 and ci.mean == ci.lo == ci.hi == 2.0
    assert ci.excludes_zero


def test_to_dict():
    d = paired_bootstrap_ci([1.0, 2.0], [0.0, 0.0]).to_dict()
    assert set(d) == {"mean", "lo", "hi", "n", "level", "n_resamples", "excludes_zero"}
    assert isinstance(paired_bootstrap_ci([1.0], [0.0]), PairedCI)


def test_paired_differences_values():
    np.testing.assert_array_equal(paired_differences([3, 5], [1, 1]), [2.0, 4.0])


@pytest.mark.parametrize(
    "a,b,msg",
    [
        ([1.0, 2.0], [1.0], "equal length"),
        ([], [], "at least one"),
        ([1.0, np.nan], [0.0, 0.0], "NaN"),
        ([1.0, 2.0], [0.0, np.inf], "NaN"),
    ],
)
def test_validation_errors(a, b, msg):
    with pytest.raises(ValueError, match=msg):
        paired_differences(a, b)
    with pytest.raises(ValueError, match=msg):
        paired_bootstrap_ci(a, b)


def test_invalid_level():
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1.0, 2.0], [0.0, 0.0], level=1.0)


def _frame(n: int = 10, shift: float = 0.5, seed: int = 1) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    a, b = _shifted(shift, n=n, seed=seed)
    rows = []
    for i in range(n):
        key = {"scenario": "high_churn", "split": "eval", "run": 0, "episode": i}
        rows.append({"policy": "A", **key, "J": a[i]})
        rows.append({"policy": "B", **key, "J": b[i]})
    return pd.DataFrame(rows), a, b


def test_frame_pairing_matches_arrays_and_ignores_row_order():
    df, a, b = _frame()
    ref = paired_bootstrap_ci(a, b, seed=3)
    shuffled = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    got = paired_ci_from_frame(shuffled, "J", "A", "B", seed=3)
    assert got.mean == pytest.approx(ref.mean)
    assert got.lo == pytest.approx(ref.lo) and got.hi == pytest.approx(ref.hi)
    assert got.n == 10


def test_frame_missing_keys_raises():
    df, _, _ = _frame()
    drop = df[(df["policy"] == "B") & (df["episode"] == 3)].index
    with pytest.raises(ValueError, match="same keys"):
        paired_ci_from_frame(df.drop(drop), "J", "A", "B")


def test_frame_unknown_policy_or_metric_raises():
    df, _, _ = _frame()
    with pytest.raises(ValueError, match="no rows"):
        paired_ci_from_frame(df, "J", "A", "Z")
    with pytest.raises(ValueError, match="missing columns"):
        paired_ci_from_frame(df, "nope", "A", "B")


# ---------------------------------------------------------------------------
# Task 9.4: Wilcoxon, Holm, run averaging, primary tests, H1–H5 table
# ---------------------------------------------------------------------------

from churnmaster.experiments.stats import (  # noqa: E402
    average_over_runs,
    holm,
    hypothesis_table,
    per_run_results,
    primary_tests,
    to_markdown,
    wilcoxon_paired,
    write_hypothesis_table,
)


def test_wilcoxon_shifted_vs_unshifted():
    a, b = _shifted(0.5)
    r = wilcoxon_paired(a, b)
    assert r.p < 1e-4 and r.n == 30 and r.n_nonzero == 30 and r.alternative == "two-sided"
    assert wilcoxon_paired(a, b, alternative="greater").p < 1e-4
    assert wilcoxon_paired(a, b, alternative="less").p > 0.99
    a0, b0 = _shifted(0.0, seed=5)
    assert wilcoxon_paired(a0, b0).p > 0.05


def test_wilcoxon_exact_small_sample():
    # d = 1..5 all positive: exact two-sided p = 2 / 2^5
    r = wilcoxon_paired([1, 2, 3, 4, 5], [0, 0, 0, 0, 0])
    assert r.p == pytest.approx(2 / 32)


def test_wilcoxon_all_zero_differences():
    r = wilcoxon_paired([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert r.p == 1.0 and r.statistic == 0.0 and r.n == 3 and r.n_nonzero == 0


def test_wilcoxon_bad_args():
    with pytest.raises(ValueError):
        wilcoxon_paired([1, 2], [0, 0], alternative="bigger")
    with pytest.raises(ValueError, match="equal length"):
        wilcoxon_paired([1, 2], [0])


def test_holm_hand_computed_input_order():
    # sorted: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> cummax 0.06
    res = holm([0.01, 0.04, 0.03, 0.005], alpha=0.05)
    np.testing.assert_allclose([r.p_adj for r in res], [0.03, 0.06, 0.06, 0.02])
    assert [r.reject for r in res] == [True, False, False, True]
    assert [r.name for r in res] == ["0", "1", "2", "3"]
    named = holm({"x": 0.04, "y": 0.001})
    assert [r.name for r in named] == ["x", "y"]
    assert named[0].p_adj == pytest.approx(0.04) and named[1].p_adj == pytest.approx(0.002)


def test_holm_capped_and_monotone():
    assert [r.p_adj for r in holm([0.5, 0.6])] == [1.0, 1.0]
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 9) ** 3
    res = holm(p)
    adj = np.array([r.p_adj for r in res])
    assert np.all(adj >= p) and np.all(adj <= 1.0)
    order = np.argsort(p)
    assert np.all(np.diff(adj[order]) >= 0)
    assert holm([]) == []
    with pytest.raises(ValueError):
        holm([0.1, 1.5])


RL_RUNS = ("rl_run0", "rl_run1", "rl_run2")
SCEN = ("stable", "high_churn", "heldout_medium")
NOISE = {"J": 0.05, "success_rate": 0.001, "p95_latency_all_ms": 2.0, "pings_per_hour": 5.0}


def _eval_frame(offsets, n=20, seed=0, rl_runs=RL_RUNS):
    """Synthetic eval table: shared per-seed base + per-policy offsets + small noise.

    ``offsets(policy, scenario) -> {metric: offset}`` (missing metrics = 0).
    """
    rng = np.random.default_rng(seed)
    policies = [*rl_runs, "kademlia_epoch", "tuned_heuristic", "matched_random", "oracle_heuristic"]
    rows = []
    for sc in SCEN:
        base = {"J": rng.normal(0, 1, n), "success_rate": 0.95 + rng.uniform(0, 0.02, n),
                "p95_latency_all_ms": 500 + rng.normal(0, 20, n), "pings_per_hour": 1000 + rng.normal(0, 30, n)}
        for pol in policies:
            off = offsets(pol, sc)
            vals = {m: base[m] + off.get(m, 0.0) + rng.normal(0, NOISE[m], n) for m in base}
            for i in range(n):
                rows.append({"policy": pol, "scenario": sc, "split": "eval", "run": 0, "episode": i,
                             **{m: float(v[i]) for m, v in vals.items()}})
    return pd.DataFrame(rows)


def _good(pol, sc):
    if pol.startswith("rl_run"):
        return {"J": 0.5, "pings_per_hour": -100.0}
    if pol == "matched_random":
        return {"J": -0.5}
    if pol == "tuned_heuristic":
        return {"p95_latency_all_ms": 5.0}
    if pol == "oracle_heuristic":
        return {"J": 2.0, "p95_latency_all_ms": -200.0}  # excluded from "best baseline"
    return {}


def _bad(pol, sc):
    if pol.startswith("rl_run"):
        return {"J": -0.5, "success_rate": -0.05, "pings_per_hour": 100.0, "p95_latency_all_ms": 100.0}
    return {}


def _null(pol, sc):
    return {}


KW = dict(n_resamples=2000, seed=0)


def test_average_over_runs_mean_and_errors():
    df = _eval_frame(_good)
    avg = average_over_runs(df, RL_RUNS, metrics=["J", "pings_per_hour"])
    assert set(avg["policy"]) == {"rl_mean"} and (avg["run"] == 0).all()
    assert len(avg) == 3 * 20
    keys = ["scenario", "split", "episode"]
    expected = df[df["policy"].isin(RL_RUNS)].groupby(keys)["J"].mean()
    got = avg.set_index(keys)["J"].reindex(expected.index)
    np.testing.assert_allclose(got.to_numpy(), expected.to_numpy())
    with pytest.raises(ValueError, match="missing run"):
        average_over_runs(df, [*RL_RUNS, "rl_run3"])
    drop = df[(df["policy"] == "rl_run1") & (df["episode"] == 4) & (df["scenario"] == "stable")].index
    with pytest.raises(ValueError, match="missing run"):
        average_over_runs(df.drop(drop), RL_RUNS)
    # RL rows labelled with their training run must still average fine
    df2 = df.copy()
    for i, p in enumerate(RL_RUNS):
        df2.loc[df2["policy"] == p, "run"] = i
    avg2 = average_over_runs(df2, RL_RUNS, metrics=["J"])
    np.testing.assert_allclose(avg2["J"].to_numpy(), avg.sort_values(keys)["J"].to_numpy())


def _with_mean(df):
    return pd.concat([df, average_over_runs(df, RL_RUNS)], ignore_index=True)


def test_primary_tests_nine_holm_corrected():
    # clear wins everywhere except RL vs tuned in stable (tuned equally good there)
    def offs(pol, sc):
        if pol == "tuned_heuristic" and sc == "stable":
            return {"J": 0.5}
        return _good(pol, sc)

    df = _with_mean(_eval_frame(offs))
    pt = primary_tests(df, **KW)
    assert len(pt) == 9 and set(pt["scenario"]) == set(SCEN)
    assert (pt["n"] == 20).all()
    assert (pt["p_holm"] >= pt["p"]).all() and (pt["p_holm"] <= 1).all()
    expected_adj = [h.p_adj for h in holm(list(pt["p"]))]
    np.testing.assert_allclose(pt["p_holm"], expected_adj)
    tie = (pt["scenario"] == "stable") & (pt["comparator"] == "tuned_heuristic")
    assert not pt.loc[tie, "significant"].item()
    assert pt.loc[~tie, "significant"].all() and (pt.loc[~tie, "direction"] == 1).all()
    mr = pt[pt["comparator"] == "matched_random"]
    np.testing.assert_allclose(mr["mean_diff"], 1.0, atol=0.1)
    assert ((mr["ci_lo"] < mr["mean_diff"]) & (mr["mean_diff"] < mr["ci_hi"])).all()


def test_per_run_results():
    df = _eval_frame(_good)
    pr = per_run_results(df, RL_RUNS, metric="J", **KW)
    assert len(pr) == 3 * 3 * 3
    assert list(pr["run"].unique()) == [0, 1, 2]
    assert "p_holm" not in pr.columns
    assert (pr["mean_diff"] > 0).all()


def _verdicts(df, **kw):
    t = hypothesis_table(df, RL_RUNS, **KW, **kw)
    assert list(t.columns) == ["id", "statement", "verdict", "evidence"]
    assert list(t["id"]) == ["H1", "H2", "H3", "H4", "H5"]
    assert t["verdict"].isin(["pass", "fail", "inconclusive"]).all()
    return dict(zip(t["id"], t["verdict"])), t


def test_hypothesis_table_all_pass():
    v, t = _verdicts(_eval_frame(_good))
    assert v == {h: "pass" for h in ("H1", "H2", "H3", "H4", "H5")}
    assert "pings_per_hour" in t.loc[t["id"] == "H2", "evidence"].item()
    # oracle excluded from best baseline; tuned/kademlia are the reference
    assert "oracle" not in t.loc[t["id"] == "H3", "evidence"].item()


def test_hypothesis_table_all_fail():
    v, _ = _verdicts(_eval_frame(_bad))
    assert v == {h: "fail" for h in ("H1", "H2", "H3", "H4", "H5")}


def test_hypothesis_table_inconclusive_when_no_effect():
    v, _ = _verdicts(_eval_frame(_null))
    assert v["H1"] == v["H2"] == v["H3"] == v["H4"] == "inconclusive"


def test_h5_fails_when_one_run_significantly_opposite():
    def offs(pol, sc):
        if pol == "rl_run2":
            return {"J": -0.3, "pings_per_hour": -100.0}
        if pol.startswith("rl_run"):
            return {"J": 0.6, "pings_per_hour": -100.0}
        return _good(pol, sc)

    v, _ = _verdicts(_eval_frame(offs))
    assert v["H2"] == "pass" and v["H5"] == "fail"


def test_hypothesis_table_custom_labels():
    df = _eval_frame(_good).replace({"policy": {"tuned_heuristic": "heur_best"}})
    v, t = _verdicts(df, tuned="heur_best")
    assert v["H2"] == "pass" and "heur_best" in t.loc[t["id"] == "H2", "statement"].item()


def test_write_hypothesis_table(tmp_path):
    _, t = _verdicts(_eval_frame(_good))
    csv_p, md_p = tmp_path / "out" / "h.csv", tmp_path / "out" / "h.md"
    write_hypothesis_table(t, csv_p, md_p)
    back = pd.read_csv(csv_p)
    assert list(back["id"]) == ["H1", "H2", "H3", "H4", "H5"]
    assert list(back["verdict"]) == list(t["verdict"])
    md = md_p.read_text(encoding="utf-8")
    lines = [ln for ln in md.splitlines() if ln.startswith("|")]
    assert lines[0] == "| id | statement | verdict | evidence |"
    assert lines[1] == "| --- | --- | --- | --- |"
    assert len(lines) == 2 + 5
    assert to_markdown(pd.DataFrame({"a": ["x|y"]})).splitlines()[2] == "| x\\|y |"
