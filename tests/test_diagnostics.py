"""Tests for experiments/diagnostics.py and stats.two_sample_bootstrap_ci (Requirement 23)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from churnmaster.actions import Action
from churnmaster.experiments.diagnostics import (COMPARISONS, DecisionSample, coverage,
                                                 coverage_from_visits, decision_samples,
                                                 discounted_return, identity_errors,
                                                 j_success_spearman, reward_sanity_from_samples,
                                                 run_diagnostics, state_visits, write_outputs,
                                                 _identity_report)
from churnmaster.experiments.runner import ConfigLoader, FrozenEnvError, freeze_env
from churnmaster.experiments.stats import two_sample_bootstrap_ci
from churnmaster.rl.ledger import Ledger
from churnmaster.rl.state import (CAND_FRESH, CAND_NONE, TIER_OTHER, TIER_SECOND, TIER_TOP,
                                  Features, encode)
from churnmaster.seeding import Split
from churnmaster.sim.observation import DecisionRecord, GTRecord, ObservedLookup

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SHORT_SIM = "[sim]\nwarmup_s = 60\nmeasured_s = 240\n"
K, P, R = int(Action.KEEP), int(Action.PROBE), int(Action.REPLACE)


def st(tier: int, cand: int = CAND_FRESH) -> int:
    return encode(Features(tier, 0, 0, 0, cand, 0, 0))


# --------------------------------------------------------------------------- #
# two-sample bootstrap
# --------------------------------------------------------------------------- #

def test_two_sample_ci_known_shift():
    rng = np.random.default_rng(1)
    a, b = rng.normal(1.0, 1.0, 400), rng.normal(0.0, 1.0, 300)
    ci = two_sample_bootstrap_ci(a, b, n_resamples=4000, seed=3)
    assert ci.mean == pytest.approx(a.mean() - b.mean())
    assert 0.0 < ci.lo < 1.0 < ci.hi and ci.excludes_zero
    assert (ci.n_a, ci.n_b) == (400, 300)
    assert ci == two_sample_bootstrap_ci(a, b, n_resamples=4000, seed=3)   # deterministic
    flipped = two_sample_bootstrap_ci(b, a, n_resamples=4000, seed=3)
    assert flipped.hi < 0.0


def test_two_sample_ci_no_shift_and_degenerate():
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, 1.0, 500)
    ci = two_sample_bootstrap_ci(x, x.copy(), n_resamples=2000)
    assert ci.lo < 0.0 < ci.hi and not ci.excludes_zero
    one = two_sample_bootstrap_ci([2.0], [0.5], n_resamples=50)
    assert one.mean == one.lo == one.hi == 1.5
    for bad in ([], [math.nan]):
        with pytest.raises(ValueError):
            two_sample_bootstrap_ci(bad, [1.0])
    with pytest.raises(ValueError):
        two_sample_bootstrap_ci([1.0], [1.0], level=1.0)


# --------------------------------------------------------------------------- #
# discounted returns and the GT join
# --------------------------------------------------------------------------- #

def test_discounted_return_hand_computed():
    r = {(3, 5): -1.0, (3, 6): -2.0, (3, 7): -4.0, (3, 8): -100.0}
    f = lambda b, k: r.get((b, k), 0.0)
    assert discounted_return(f, 3, 5, gamma=0.5, horizon=3) == pytest.approx(-1 - 1 - 1)
    assert discounted_return(f, 3, 5, gamma=0.9, horizon=1) == -1.0
    assert discounted_return(f, 2, 5, gamma=0.9, horizon=4) == 0.0
    with pytest.raises(ValueError):
        discounted_return(f, 3, 5, horizon=0)


def _dec(b, k, state, action):
    return DecisionRecord(b, k, k * 30_000, state, action, 0, 0, True)


def _gt(b, k, action, inc_online, cand_present):
    return GTRecord.make(k * 30_000, "decision", b, True, epoch_idx=k, action=action,
                         inc_online=inc_online, cand_present=cand_present,
                         cand_online=None, inserted=False, evicted=False)


def test_decision_samples_join_groups_and_truncation():
    s_top = st(TIER_TOP)
    decs = [_dec(7, 0, s_top, P), _dec(6, 0, st(TIER_SECOND, CAND_NONE), K),
            _dec(5, 0, None, None),          # null record: skipped
            _dec(7, 1, s_top, K), _dec(7, 2, s_top, R)]
    gts = [_gt(7, 0, P, False, True), _gt(6, 0, K, True, False),
           _gt(7, 1, K, True, True), _gt(7, 2, R, True, True)]
    rew = {(7, 0): -1.0, (7, 1): -0.5, (7, 2): -0.25, (6, 0): -2.0, (6, 1): -2.0}
    out = decision_samples(decs, gts, lambda b, k: rew.get((b, k), 0.0), gamma=0.5, horizon=2,
                           scenario="x", episode=4)
    # last epoch = 2, so (7, 2) with H=2 is truncated and skipped
    assert [(s.bucket, s.epoch_idx) for s in out] == [(7, 0), (6, 0), (7, 1)]
    a, b, c = out
    assert (a.tier, a.inc_online, a.cand_present, a.action, a.G) == (TIER_TOP, False, True, P, -1.25)
    assert (b.tier, b.inc_online, b.cand_present, b.G) == (TIER_SECOND, True, False, -3.0)
    assert c.G == pytest.approx(-0.5 - 0.125) and c.scenario == "x" and c.episode == 4


def test_decision_samples_rejects_missing_or_mismatched_gt():
    decs = [_dec(7, 0, st(TIER_TOP), P)]
    with pytest.raises(ValueError, match="no GT"):
        decision_samples(decs, [], lambda b, k: 0.0, horizon=1)
    with pytest.raises(ValueError, match="GT action"):
        decision_samples(decs, [_gt(7, 0, K, False, True)], lambda b, k: 0.0, horizon=1)


# --------------------------------------------------------------------------- #
# reward sanity groups and pass flags
# --------------------------------------------------------------------------- #

def _samples(spec):
    """spec: list of (tier, inc_online, cand_present, action, [G values])."""
    out = []
    for tier, on, cand, a, gs in spec:
        for g in gs:
            out.append(DecisionSample("x", 0, 7, 0, st(tier, CAND_FRESH if cand else CAND_NONE),
                                      a, tier, on, cand, float(g)))
    return out


def test_reward_sanity_selects_groups_and_passes():
    spec = [
        # dead, top/second tiers
        (TIER_TOP, False, True, P, [-1.0, -1.2, -0.8]),
        (TIER_SECOND, False, True, R, [-0.5, -0.6, -0.4]),
        (TIER_TOP, False, True, K, [-3.0, -3.2, -2.8]),
        (TIER_SECOND, False, False, K, [-9.0, -9.0]),      # no candidate: only unrestricted
        (TIER_SECOND, False, False, P, [-2.0, -2.0]),
        (TIER_OTHER, False, True, K, [10.0] * 5),           # other tier: excluded from dead group
        # live, any tier
        (TIER_OTHER, True, True, K, [-1.0, -1.1, -0.9]),
        (TIER_OTHER, True, True, R, [-2.0, -2.1, -1.9]),
        (TIER_OTHER, True, False, K, [-50.0]),               # no candidate: excluded
    ]
    rep = reward_sanity_from_samples(_samples(spec), n_resamples=500)
    assert rep.n_samples == 25
    c = rep.comparison("dead_top2_probe_vs_keep")
    assert (c.n_better, c.n_worse) == (5, 5)
    assert c.mean_better == pytest.approx((-3.0 - 4.0) / 5)
    assert c.mean_worse == pytest.approx((-9.0 - 9.0 - 9.0) / 5)
    assert c.passed
    c = rep.comparison("dead_top2_probe_vs_keep_cand")
    assert (c.n_better, c.n_worse, c.mean_worse) == (3, 3, pytest.approx(-3.0)) and c.passed
    c = rep.comparison("dead_top2_replace_vs_keep_cand")
    assert (c.n_better, c.n_worse) == (3, 3) and c.ci.mean == pytest.approx(2.5) and c.passed
    c = rep.comparison("live_keep_vs_replace_cand")
    assert (c.n_better, c.n_worse, c.mean_better) == (3, 3, pytest.approx(-1.0)) and c.passed
    info = rep.comparison("live_top2_keep_vs_replace_cand")
    assert info.ci is None and not info.passed and not info.primary
    assert rep.passed    # only primary comparisons count
    assert {d["name"] for d in rep.to_dict()["comparisons"]} == {x.name for x in COMPARISONS}


def test_reward_sanity_fails_in_wrong_direction_or_when_empty():
    spec = [(TIER_TOP, False, True, P, [-3.0, -3.1, -2.9]),
            (TIER_TOP, False, True, R, [-1.0, -1.0, -1.0]),
            (TIER_TOP, False, True, K, [-1.0, -1.1, -0.9]),     # KEEP better than PROBE
            (TIER_TOP, True, True, K, [-1.0, -1.1, -0.9]),
            (TIER_TOP, True, True, R, [-1.0, -1.1, -0.9])]      # no difference
    rep = reward_sanity_from_samples(_samples(spec), n_resamples=500)
    assert not rep.comparison("dead_top2_probe_vs_keep").passed
    assert rep.comparison("dead_top2_probe_vs_keep").ci.hi < 0
    assert not rep.comparison("dead_top2_replace_vs_keep_cand").passed  # CI straddles 0
    assert not rep.comparison("live_keep_vs_replace_cand").passed
    assert not rep.passed
    assert not reward_sanity_from_samples([], n_resamples=10).passed


# --------------------------------------------------------------------------- #
# identity
# --------------------------------------------------------------------------- #

def test_identity_errors_on_synthetic_ledger():
    led = Ledger(0.3, 0.2, 0.3, 2000, 30_000, 8)
    s = st(TIER_TOP)
    lookups = [ObservedLookup(0, 1_000, 7, True, 100.0, False),
               ObservedLookup(1, 31_000, 7, False, 2000.0, True),
               ObservedLookup(2, 45_000, 6, True, 500.0, True)]
    decisions = [DecisionRecord(7, 0, 0, s, P, 1, 0, False),
                 DecisionRecord(7, 1, 30_000, s, R, 0, 1, True)]
    for rec in [decisions[0], lookups[0], decisions[1], lookups[1], lookups[2]]:
        led.add(rec)
    eJ, eT = identity_errors(led.score(), led, lookups, decisions)
    assert eJ == 0.0 and eT <= 1e-12
    expected_total = led.reward(7, 1) + led.reward(6, 1)
    assert led.total_reward() == pytest.approx(expected_total)
    eJ, _ = identity_errors(led.score() + 0.1, led, lookups, decisions)
    assert eJ == pytest.approx(0.1)
    rep = _identity_report([(0.0, 0.0), (eJ, 0.0)])
    assert (rep.n_episodes, rep.n_passed, rep.passed) == (2, 1, False)
    assert rep.max_abs_err_J == pytest.approx(0.1)
    assert _identity_report([(0.0, 0.0)]).passed


# --------------------------------------------------------------------------- #
# coverage
# --------------------------------------------------------------------------- #

def test_coverage_fraction_and_pass():
    rep = coverage_from_visits({0: 60, 1: 50, 2: 49, 3: 1}, min_visits=50, target=0.8)
    assert rep.n_decisions == 160 and rep.n_states_visited == 4 and rep.n_states_covered == 2
    assert rep.fraction == pytest.approx(110 / 160) and not rep.passed
    assert coverage_from_visits({0: 90, 1: 10}, min_visits=50).passed
    assert coverage_from_visits({0: 80, 1: 20}, min_visits=50).passed      # exactly 0.80
    assert not coverage_from_visits({}).passed


def test_coverage_from_results_uses_first_n_episodes():
    s0, s1 = st(TIER_TOP), st(TIER_OTHER)
    ep = lambda states: SimpleNamespace(decisions=[_dec(7, k, s, K) for k, s in enumerate(states)]
                                        + [_dec(5, 0, None, None)])
    results = [ep([s0] * 3 + [s1]), ep([s0] * 2), ep([s1] * 100)]
    rep = coverage(results, min_visits=5, n_episodes=2)
    assert rep.n_episodes == 2 and rep.visits == {s0: 5, s1: 1}
    assert rep.fraction == pytest.approx(5 / 6) and rep.passed
    assert state_visits(results[2].decisions) == {s1: 100}
    with pytest.raises(ValueError):
        coverage(results, n_episodes=4)


# --------------------------------------------------------------------------- #
# Spearman
# --------------------------------------------------------------------------- #

def test_spearman_monotone_and_constant():
    rows = []
    for scn, sign in (("a", 1.0), ("b", -1.0)):
        for i, pol in enumerate(["p0", "p1", "p2", "p3"]):
            for ep in range(3):
                rows.append({"scenario": scn, "policy": pol, "episode": ep,
                             "J": 0.5 + 0.1 * i + 0.01 * ep,
                             "success_rate": 0.5 + sign * (0.1 * i + 0.01 * ep)})
    res = {(r.level, r.scope): r for r in j_success_spearman(pd.DataFrame(rows))}
    assert res[("policy_means", "a")].rho == pytest.approx(1.0) and res[("policy_means", "a")].n == 4
    assert res[("policy_means", "b")].rho == pytest.approx(-1.0)
    assert res[("per_seed", "a")].rho == pytest.approx(1.0) and res[("per_seed", "a")].n == 12
    assert res[("per_seed", "pooled")].n == 24
    const = pd.DataFrame({"scenario": ["a"] * 4, "policy": list("wxyz"),
                          "J": [0.1, 0.2, 0.3, 0.4], "success_rate": [1.0] * 4})
    assert math.isnan(j_success_spearman(const)[0].rho)
    with pytest.raises(ValueError):
        j_success_spearman(const.drop(columns=["J"]))


# --------------------------------------------------------------------------- #
# Driver: frozen-env refusal and an end-to-end short run
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def short_exp(tmp_path_factory):
    p = tmp_path_factory.mktemp("exp") / "short.toml"
    p.write_text(SHORT_SIM, encoding="utf-8")
    return p


def test_run_diagnostics_refuses_unfrozen_or_mismatched_env(tmp_path, short_exp):
    with pytest.raises(FrozenEnvError):
        run_diagnostics(CONFIGS, tmp_path / "missing.json", experiment_path=short_exp,
                        n_episodes=1, scenarios=("high_churn",))
    # manifest frozen on the full-length env: the short sim has a different env hash
    man = tmp_path / "manifest.json"
    full = ConfigLoader(CONFIGS)
    freeze_env(man, {"high_churn": full("high_churn")})
    with pytest.raises(FrozenEnvError, match="differs"):
        run_diagnostics(CONFIGS, man, experiment_path=short_exp, n_episodes=1,
                        scenarios=("high_churn",))


def test_run_diagnostics_end_to_end_short(tmp_path, short_exp):
    loader = ConfigLoader(CONFIGS, short_exp)
    man = tmp_path / "manifest.json"
    freeze_env(man, {"high_churn": loader("high_churn")})
    rep = run_diagnostics(CONFIGS, man, experiment_path=short_exp, n_episodes=2,
                          scenarios=("high_churn",), n_resamples=200, split=Split.CALIBRATION,
                          spearman_policies=("never_replace", "always_replace", "kademlia_epoch"),
                          spearman_seeds=1)
    assert rep.identity["pooled"].passed and rep.identity["pooled"].n_episodes == 2
    assert rep.sanity["high_churn"].n_samples > 0
    cov = rep.coverage["high_churn"]
    assert cov.n_episodes == 2 and cov.n_decisions == sum(cov.visits.values()) > 0
    assert any(s.level == "policy_means" and s.n == 3 for s in rep.spearman)
    d = rep.to_dict()
    assert set(d["passed"]) == {"identity", "reward_sanity", "coverage"}
    files = write_outputs(rep, tmp_path / "out")
    assert all(f.is_file() for f in files)
    assert json.loads((tmp_path / "out" / "diagnostics.json").read_text())["n_episodes"] == 2
    cov_df = pd.read_csv(tmp_path / "out" / "coverage.csv")
    assert set(cov_df["scope"]) == {"high_churn", "pooled"}
    # Requirement 21.3: identical report with a process pool
    rep2 = run_diagnostics(CONFIGS, man, experiment_path=short_exp, n_episodes=2,
                           scenarios=("high_churn",), n_resamples=200, split=Split.CALIBRATION,
                           spearman_policies=("never_replace", "always_replace", "kademlia_epoch"),
                           spearman_seeds=1, workers=2)
    assert rep2.to_dict() == d
    assert rep2.coverage["high_churn"].visits == cov.visits
