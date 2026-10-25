"""Tests for policies/learned.py (Requirement 16.4)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from churnmaster.actions import Action
from churnmaster.experiments.runner import ConfigLoader, Job, run_job
from churnmaster.policies.base import Policy
from churnmaster.policies.learned import Learned
from churnmaster.rl.qlearning import QAgent
from churnmaster.rl.state import N_ACTIONS, N_STATES, discretize, encode
from churnmaster.seeding import Split
from churnmaster.sim.observation import DecisionContext

K, P, R = Action.KEEP, Action.PROBE, Action.REPLACE
CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _ctx(cand: bool, **kw) -> DecisionContext:
    base = dict(bucket=7, epoch_idx=0, t_ms=0, inc_consec_fail=0, inc_since_response_s=1.0,
                inc_age_s=5.0, inc_srtt_ms=100.0,
                cand_since_seen_s=5.0 if cand else None,
                cand_last_rtt_ms=50.0 if cand else None, timeout_fraction=0.0)
    base.update(kw)
    return DecisionContext(**base)


def _s(ctx: DecisionContext) -> int:
    return encode(discretize(ctx))


def _random_contexts(rng: np.random.Generator, n: int) -> list[DecisionContext]:
    out = []
    for _ in range(n):
        cand = bool(rng.random() < 0.6)
        out.append(DecisionContext(
            bucket=int(rng.integers(0, 8)), epoch_idx=0, t_ms=0,
            inc_consec_fail=int(rng.integers(0, 4)),
            inc_since_response_s=float(rng.uniform(0, 200)),
            inc_age_s=float(rng.uniform(0, 3000)),
            inc_srtt_ms=None if rng.random() < 0.2 else float(rng.uniform(5, 300)),
            cand_since_seen_s=float(rng.uniform(0, 300)) if cand else None,
            cand_last_rtt_ms=float(rng.uniform(5, 300)) if cand else None,
            timeout_fraction=float(rng.uniform(0, 0.3))))
    return out


def test_fixed_table_choice_and_masking():
    with_c, without_c = _ctx(True), _ctx(False)
    Q = np.zeros((N_STATES, N_ACTIONS))
    Q[_s(with_c)] = [-3.0, -2.0, -1.0]       # REPLACE best
    Q[_s(without_c)] = [-3.0, -2.0, 5.0]     # REPLACE highest but masked (no candidate)
    pol = Learned(Q)
    assert pol.decide(with_c) == R
    assert pol.decide(without_c) == P
    # REPLACE is never chosen without a candidate, whatever its Q value
    Q2 = np.full((N_STATES, N_ACTIONS), -1.0)
    Q2[:, 2] = 100.0
    pol2 = Learned(Q2)
    for ctx in _random_contexts(np.random.default_rng(1), 300):
        a = pol2.decide(ctx)
        assert (a == R) == (ctx.cand_since_seen_s is not None)


def test_ties_keep_then_probe():
    ctx = _ctx(True)
    Q = np.zeros((N_STATES, N_ACTIONS))
    assert Learned(Q).decide(ctx) == K                 # all tied -> KEEP
    Q[_s(ctx)] = [-1.0, 0.0, 0.0]
    assert Learned(Q).decide(ctx) == P                 # PROBE/REPLACE tie -> PROBE
    Q[_s(ctx)] = [-1.0, -1.0, 0.0]
    assert Learned(Q).decide(ctx) == R


def test_agrees_with_agent_greedy_on_random_table():
    rng = np.random.default_rng(7)
    agent = QAgent()
    agent.Q[...] = rng.integers(-3, 1, size=(N_STATES, N_ACTIONS)).astype(float)  # many ties
    pol = Learned.from_agent(agent, name="rl_run0")
    assert pol.name == "rl_run0"
    assert [int(pol.greedy_table[s]) for s in range(N_STATES)] == \
        [int(agent.greedy(s)) for s in range(N_STATES)]
    np.testing.assert_array_equal(pol.greedy_table, agent.greedy_table())
    for ctx in _random_contexts(rng, 500):
        assert pol.decide(ctx) == agent.greedy(_s(ctx))


def test_frozen_copy_semantics():
    ctx = _ctx(True)
    agent = QAgent()
    agent.Q[_s(ctx)] = [0.0, -1.0, -1.0]
    pol = Learned(agent)
    agent.Q[_s(ctx)] = [-5.0, -1.0, 0.0]
    agent.mask[_s(ctx), 0] = False
    assert pol.decide(ctx) == K
    for arr in (pol.Q, pol.mask, pol.greedy_table):
        assert not arr.flags.writeable
        with pytest.raises(ValueError):
            arr[0] = 1


def test_bad_shapes_raise():
    with pytest.raises(ValueError):
        Learned(np.zeros((10, 3)))
    with pytest.raises(ValueError):
        Learned(np.zeros((N_STATES, N_ACTIONS)), mask=np.ones((10, 3), dtype=bool))


def test_load_from_saved_agent(tmp_path):
    agent = QAgent()
    agent.Q[...] = np.random.default_rng(3).normal(size=(N_STATES, N_ACTIONS))
    agent.save(tmp_path / "q")
    pol = Learned.load(tmp_path / "q", name="loaded")
    assert pol.name == "loaded"
    np.testing.assert_array_equal(pol.greedy_table, agent.greedy_table())


def test_pickle_round_trip_and_protocol():
    rng = np.random.default_rng(11)
    pol = Learned(rng.normal(size=(N_STATES, N_ACTIONS)), name="rl_run1")
    pol2 = pickle.loads(pickle.dumps(pol))
    assert isinstance(pol, Policy) and isinstance(pol2, Policy)
    assert pol2.name == "rl_run1" and pol2.shares_decision_points is True
    assert not pol2.Q.flags.writeable and not pol2.greedy_table.flags.writeable
    for ctx in _random_contexts(rng, 200):
        assert pol2.decide(ctx) == pol.decide(ctx)


def test_runs_as_job_policy(tmp_path):
    exp = tmp_path / "short.toml"
    exp.write_text("[sim]\nwarmup_s = 60\nmeasured_s = 240\n", encoding="utf-8")
    cfg = ConfigLoader(CONFIGS, exp)("high_churn")
    rng = np.random.default_rng(5)
    pol = Learned(rng.normal(size=(N_STATES, N_ACTIONS)), name="rl_test")
    res = run_job(Job(pol, "high_churn", Split.CALIBRATION, 0, 0), cfg=cfg, keep_records=True)
    assert res.policy == "rl_test"
    decided = [d for d in res.decisions if d.action is not None]
    assert decided
    assert all(d.action == int(pol.greedy_table[d.state]) for d in decided)
