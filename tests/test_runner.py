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
