"""Runner and policy factory tests (Requirements 17.4, 21.1, 26.1)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from churnmaster.config import load_config
from churnmaster.experiments.runner import (ConfigLoader, Job, NoMaintenance, job_baseline_rng,
                                            make_policy, policy_env_mode, results_table,
                                            run_episode, run_job, run_jobs, summary_by_policy)
from churnmaster.metrics import summary_hash
from churnmaster.policies.baselines import NeverReplace
from churnmaster.policies.heuristic import DEFAULT_CONFIG, Heuristic, all_configs
from churnmaster.policies.oracle import OracleHeuristic
from churnmaster.seeding import Split

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SHORT_SIM = "[sim]\nwarmup_s = 60\nmeasured_s = 240\n"


@pytest.fixture(scope="module")
def loader(tmp_path_factory):
    exp = tmp_path_factory.mktemp("exp") / "short.toml"
    exp.write_text(SHORT_SIM, encoding="utf-8")
    return ConfigLoader(CONFIGS, exp)


@pytest.fixture(scope="module")
def short_cfg(loader):
    return loader("high_churn")


@pytest.fixture(scope="module")
def compare_df(loader):
    jobs = [Job(p, "high_churn", Split.CALIBRATION, 0, ep)
            for p in ("never_replace", "kademlia_epoch") for ep in (0, 1)]
    return results_table(run_jobs(jobs, cfg_loader=loader))


def test_small_compare_returns_per_seed_metrics(compare_df):
    df = compare_df
    assert len(df) == 4
    assert list(df["policy"]) == ["never_replace"] * 2 + ["kademlia_epoch"] * 2
    assert list(df["episode"]) == [0, 1, 0, 1]
    assert (df["n_lookups"] > 0).all()
    for m in ("success_rate", "p95_latency_all_ms", "pings_per_hour", "J"):
        assert df[m].map(math.isfinite).all()
    assert (df.loc[df["policy"] == "never_replace", "pings_per_hour"] == 0).all()
    assert (df.loc[df["policy"] == "kademlia_epoch", "pings_per_hour"] > 0).all()


def test_same_seed_same_trace_across_policies(compare_df):
    by_ep = compare_df.groupby("episode")
    for _, g in by_ep:
        assert g["trace_hash"].nunique() == 1 and g["workload_hash"].nunique() == 1
    assert compare_df["trace_hash"].nunique() == 2  # different seeds differ


def test_summary_by_policy(compare_df):
    s = summary_by_policy(compare_df)
    assert list(s["policy"]) == ["never_replace", "kademlia_epoch"]
    assert (s["n"] == 2).all()
    g = compare_df[compare_df["policy"] == "kademlia_epoch"]["success_rate"]
    row = s[s["policy"] == "kademlia_epoch"].iloc[0]
    assert row["success_rate_mean"] == pytest.approx(g.mean())
    assert row["success_rate_std"] == pytest.approx(g.std(ddof=1))


def test_oracle_and_kademlia_std_labels(loader):
    res = run_jobs([Job(p, "high_churn", Split.CALIBRATION, 0, 0)
                    for p in ("oracle_heuristic", "kademlia_std", "heuristic")], cfg_loader=loader)
    df = results_table(res)
    assert dict(zip(df["policy"], df["shares_decision_points"])) == {
        "oracle_heuristic": True, "kademlia_std": False, "heuristic": True}
    assert (df["n_decisions"] > 0).all() and df["trace_hash"].nunique() == 1
    assert policy_env_mode("kademlia_std") == "kademlia_std"
    assert policy_env_mode("oracle_heuristic") == "epoch"


def test_policy_factory(short_cfg):
    hc = all_configs()[5]
    p = make_policy(f"heuristic:{hc.id}", short_cfg)
    assert isinstance(p, Heuristic) and p.cfg == hc
    assert make_policy("heuristic", short_cfg).cfg == DEFAULT_CONFIG
    assert isinstance(make_policy("none", short_cfg), NoMaintenance)
    with pytest.raises(ValueError):
        make_policy("bogus", short_cfg)
    with pytest.raises(ValueError):
        make_policy("heuristic:f=3", short_cfg)
    with pytest.raises(ValueError):
        make_policy("oracle_heuristic", short_cfg)  # needs the env
    with pytest.raises(ValueError):
        make_policy("matched_random", short_cfg)  # needs freqs
    with pytest.raises(ValueError):
        run_job(Job("bogus", "high_churn", Split.CALIBRATION), cfg=short_cfg)


def test_policy_object_extension_point_matches_name(short_cfg):
    a = run_job(Job(NeverReplace(), "high_churn", Split.CALIBRATION, 0, 0), cfg=short_cfg)
    b = run_job(Job("never_replace", "high_churn", Split.CALIBRATION, 0, 0), cfg=short_cfg)
    assert a.policy == "never_replace"
    assert summary_hash(a.metrics) == summary_hash(b.metrics)


def test_random_baseline_deterministic_per_job(short_cfg):
    job = Job("uniform_random", "high_churn", Split.CALIBRATION, 0, 0)
    a, b = run_job(job, cfg=short_cfg), run_job(job, cfg=short_cfg)
    assert summary_hash(a.metrics) == summary_hash(b.metrics)
    r0 = job_baseline_rng(short_cfg.root_seed, 0, 0, Split.CALIBRATION, 0).integers(1 << 62, size=4)
    r1 = job_baseline_rng(short_cfg.root_seed, 0, 0, Split.CALIBRATION, 1).integers(1 << 62, size=4)
    assert not np.array_equal(r0, r1)


def test_keep_records_and_none_matches_run_episode(short_cfg):
    res = run_job(Job("none", "high_churn", Split.TEST, 0, 0), cfg=short_cfg, keep_records=True)
    assert res.lookups and res.gts and res.decisions == []
    m, t = run_episode(short_cfg, Split.TEST, 0, 0)
    assert summary_hash(m) == summary_hash(res.metrics) and t.trace_hash == res.trace.trace_hash
    assert run_job(Job("none", "high_churn", Split.TEST), cfg=short_cfg).lookups is None


def test_cfg_scenario_mismatch_raises(short_cfg):
    with pytest.raises(ValueError):
        run_job(Job("none", "stable", Split.TEST), cfg=short_cfg)


def test_loader_merges_experiment_file(loader):
    cfg = loader("stable")
    full = load_config(CONFIGS / "base.toml", CONFIGS / "scenarios" / "stable.toml")
    assert (cfg.sim.warmup_s, cfg.sim.measured_s) == (60, 240)
    assert cfg.scenario == full.scenario


# --------------------------------------------------------------------------- #
# Ledger wiring: Metrics.J == Ledger.score() (Requirements 15.4, 15.5, 20.4)
# --------------------------------------------------------------------------- #

from churnmaster.experiments.runner import POLICY_NAMES  # noqa: E402
from churnmaster.rl.reward import lookup_loss  # noqa: E402
from churnmaster.rl.state import allowed_actions  # noqa: E402

MATCHED_FREQS = {False: [0.6, 0.4, 0.0], True: [0.4, 0.3, 0.3]}
LEDGER_POLICIES = POLICY_NAMES + (f"heuristic:{all_configs()[5].id}",)


@pytest.fixture(scope="module")
def ledger_runs(short_cfg):
    out = {}
    for p in LEDGER_POLICIES:
        kw = {"freqs": MATCHED_FREQS} if p == "matched_random" else {}
        hooked: list = []
        res = run_job(Job(p, "high_churn", Split.CALIBRATION, 0, 0, policy_kw=kw), cfg=short_cfg,
                      keep_records=True, on_transition=hooked.append)
        out[p] = (res, hooked)
    return out


def test_ledger_identity_every_policy(ledger_runs):
    assert set(ledger_runs) >= set(POLICY_NAMES)
    for p, (res, _) in ledger_runs.items():
        assert math.isfinite(res.metrics.J), p
        assert abs(res.metrics.J - res.ledger_score) < 1e-9, p
        row = res.row()
        assert row["ledger_score"] == res.ledger_score and row["n_transitions"] == res.n_transitions


def test_ledger_includes_kademlia_std_costs(ledger_runs, short_cfg):
    res, _ = ledger_runs["kademlia_std"]
    epoch_ms = int(short_cfg.sim.epoch_s) * 1000
    cost_recs = [d for d in res.decisions if d.measured and d.t_ms != d.epoch_idx * epoch_ms]
    assert cost_recs and sum(d.pings for d in cost_recs) > 0
    rw, sim = short_cfg.reward, short_cfg.sim
    meas = [lk for lk in res.lookups if lk.measured]
    j_nocost = 1.0 - sum(lookup_loss(lk.success, lk.latency_ms, rw.w_L, sim.deadline_ms)
                         for lk in meas) / len(meas)
    assert res.ledger_score < j_nocost


def test_ledger_transitions(ledger_runs):
    for p, (res, hooked) in ledger_runs.items():
        if p == "none":
            assert res.n_transitions == 0 and res.transitions == [] and hooked == []
            continue
        assert res.n_transitions > 0, p
        assert hooked == res.transitions and len(hooked) == res.n_transitions, p
        for tr in res.transitions:
            assert tr.a in {int(a) for a in allowed_actions(tr.s)}, p
            assert tr.allowed_next == tuple(int(a) for a in allowed_actions(tr.s_next))


def test_ledger_identity_violation_raises():
    from churnmaster.experiments.runner import _check_identity
    job = Job("none", "high_churn", Split.TEST)
    _check_identity(math.nan, math.nan, job)
    _check_identity(0.5, 0.5 + 1e-12, job)
    with pytest.raises(RuntimeError):
        _check_identity(0.5, 0.5 + 1e-6, job)
    with pytest.raises(RuntimeError):
        _check_identity(0.5, math.nan, job)


# --------------------------------------------------------------------------- #
# Multiprocessing (Requirement 21.3)
# --------------------------------------------------------------------------- #

import pandas as pd  # noqa: E402

from churnmaster.experiments.runner import default_workers  # noqa: E402

MP_POLICIES = ("never_replace", "kademlia_epoch", "uniform_random", "heuristic",
               "oracle_heuristic", "kademlia_std")
MP_JOBS = [Job(p, "high_churn", Split.CALIBRATION, 0, ep) for ep in (0, 1) for p in MP_POLICIES]


@pytest.fixture(scope="module")
def mp_results(loader):
    seq = run_jobs(MP_JOBS, cfg_loader=loader, keep_records=True)
    par = run_jobs(MP_JOBS, cfg_loader=loader, keep_records=True, workers=2)
    return seq, par


def test_multiprocessing_matches_sequential(mp_results):
    seq, par = mp_results
    a, b = results_table(seq), results_table(par)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert list(b["policy"]) == [j.policy for j in MP_JOBS]
    assert list(b["episode"]) == [j.episode for j in MP_JOBS]


def test_multiprocessing_keep_records_match(mp_results):
    for rs, rp in zip(*mp_results):
        assert rs.lookups == rp.lookups and rs.decisions == rp.decisions
        assert rs.gts == rp.gts and rs.transitions == rp.transitions
        assert rs.ledger is not None and rp.ledger is not None
        assert rs.ledger.score() == rp.ledger.score() or (
            math.isnan(rs.ledger.score()) and math.isnan(rp.ledger.score()))
        assert rs.ledger.total_reward() == rp.ledger.total_reward()
        keys = {(d.bucket, d.epoch_idx) for d in rs.decisions}
        assert all(rs.ledger.reward(b, k) == rp.ledger.reward(b, k) for b, k in keys)


def test_workers_one_is_sequential_path(loader):
    jobs = MP_JOBS[:2]
    base = results_table(run_jobs(jobs, cfg_loader=loader))
    for w in (None, 0, 1):
        pd.testing.assert_frame_equal(results_table(run_jobs(jobs, cfg_loader=loader, workers=w)),
                                      base, check_exact=True)


def test_multiprocessing_rejects_unpicklable(loader):
    with pytest.raises(TypeError, match="cfg_loader"):
        run_jobs(MP_JOBS[:2], cfg_loader=lambda s: loader(s), workers=2)

    class Local(NeverReplace):  # local class: not picklable
        pass

    with pytest.raises(TypeError, match="not picklable"):
        run_jobs([Job(Local(), "high_churn", Split.CALIBRATION)] * 2, cfg_loader=loader,
                 workers=2)
    with pytest.raises(ValueError):
        run_jobs(MP_JOBS[:2], cfg_loader=loader, workers=-1)


def test_default_workers():
    w = default_workers()
    assert 1 <= w <= 8
