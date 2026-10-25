"""Tests for experiments/matched.py (Requirement 17.2, task 9.2)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from churnmaster.experiments.matched import load_matched_freqs, measure_matched_freqs
from churnmaster.experiments.runner import ConfigLoader, FrozenEnvError, Job, freeze_env, run_jobs
from churnmaster.policies.baselines import MatchedRandom, action_counts, action_frequencies
from churnmaster.policies.learned import Learned
from churnmaster.rl.qlearning import QAgent
from churnmaster.seeding import Split

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SHORT = """
[sim]
warmup_s = 60
measured_s = 240
"""
SCENARIOS = ("stable", "heldout_medium")


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    d = tmp_path_factory.mktemp("matched")
    exp = d / "short.toml"
    exp.write_text(SHORT, encoding="utf-8")
    loader = ConfigLoader(CONFIGS, exp)
    man = d / "manifest.json"
    freeze_env(man, {s: loader(s) for s in SCENARIOS})
    qs = []
    for i in range(2):
        ag = QAgent()
        ag.Q = np.random.default_rng(i).normal(size=ag.Q.shape)
        qs.append(ag.save(d / f"run{i}" / "q_final.npy"))
    return d, exp, man, qs, loader


def test_measure_writes_freqs_and_round_trips(setup):
    d, exp, man, qs, loader = setup
    out = d / "out" / "freqs.json"
    freqs = measure_matched_freqs(qs, CONFIGS, man, scenarios=SCENARIOS, n_seeds=1,
                                  experiment_path=exp, out_path=out)
    assert set(freqs) == set(SCENARIOS)
    data = json.loads(out.read_text(encoding="utf-8"))
    for s in SCENARIOS:
        assert data[s]["no_cand"][2] == 0.0
        assert data[s]["split"] == "validation" and data[s]["n_episodes"] == 2
        assert sum(map(sum, data[s]["counts"].values())) > 0
        MatchedRandom(freqs[s], np.random.default_rng(0))
    assert load_matched_freqs(out) == freqs

    # Pooled over both RL runs on validation episode 0 (run index 0).
    pols = [Learned.load(q) for q in qs]
    res = run_jobs([Job(p, "stable", Split.VALIDATION, 0, 0) for p in pols],
                   cfg_loader=loader, keep_records=True)
    decs = [x for r in res for x in r.decisions]
    assert freqs["stable"] == action_frequencies(decs)
    assert data["stable"]["counts"] == {"no_cand": list(action_counts(decs)[False]),
                                        "cand": list(action_counts(decs)[True])}


def test_measure_refuses_unfrozen(setup, tmp_path):
    d, exp, man, qs, _ = setup
    with pytest.raises(FrozenEnvError):
        measure_matched_freqs(qs, CONFIGS, tmp_path / "missing.json", scenarios=SCENARIOS,
                              n_seeds=1, experiment_path=exp, out_path=tmp_path / "f.json")
    with pytest.raises(FrozenEnvError):   # high_churn not in the manifest
        measure_matched_freqs(qs, CONFIGS, man, scenarios=("high_churn",), n_seeds=1,
                              experiment_path=exp)
    assert not (tmp_path / "f.json").exists()
