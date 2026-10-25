"""Tests for sim/churn.py (Requirement 5)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from churnmaster.config import ScenarioConfig
from churnmaster.seeding import Split, env_seed, make_streams
from churnmaster.sim.churn import (
    NodeType,
    analytic_online_fraction,
    draw_node_types,
    generate_trace,
    weibull_scale,
)

HIGH_CHURN = ScenarioConfig(name="high_churn", stable_share=0.30, stable_session_mean_s=3600.0,
                            flaky_session_mean_s=300.0, downtime_mean_s=600.0)
STABLE = ScenarioConfig(name="stable", stable_share=0.70, stable_session_mean_s=7200.0,
                        flaky_session_mean_s=900.0, downtime_mean_s=600.0)

N = 128
FOCAL = 0
HORIZON_MS = 3000 * 1000


def _trace(scn: ScenarioConfig, seed: int, n: int = N, horizon_ms: int = HORIZON_MS):
    streams = make_streams(env_seed(12345, Split.CALIBRATION, 0, seed))
    types = draw_node_types(streams.types_positions, n, scn.stable_share, focal=FOCAL)
    return types, generate_trace(scn, types, streams.churn, horizon_ms, FOCAL)


def test_weibull_scale_matches_mean():
    assert weibull_scale(300.0, 0.6) == pytest.approx(300.0 / math.gamma(1 + 1 / 0.6))
    rng = np.random.default_rng(0)
    samples = rng.weibull(0.6, 400_000) * weibull_scale(300.0, 0.6)
    assert samples.mean() == pytest.approx(300.0, rel=0.03)


def test_node_types_follow_stable_share():
    rng = np.random.default_rng(1)
    types = draw_node_types(rng, N, 0.30, focal=FOCAL)
    assert len(types) == N
    non_focal = [t for i, t in enumerate(types) if i != FOCAL]
    assert sum(t.stable for t in non_focal) == round(0.30 * (N - 1))
    assert all(isinstance(t, NodeType) for t in types)


def test_trace_deterministic_for_same_seed():
    t1, a = _trace(HIGH_CHURN, 3)
    t2, b = _trace(HIGH_CHURN, 3)
    assert t1 == t2
    assert a.all_sessions() == b.all_sessions()
    assert list(a.events()) == list(b.events())
    assert a.trace_hash() == b.trace_hash()
    _, c = _trace(HIGH_CHURN, 4)
    assert c.trace_hash() != a.trace_hash()


@pytest.mark.parametrize("scn", [HIGH_CHURN, STABLE])
def test_sessions_ordered_non_overlapping_and_clipped(scn):
    for seed in range(5):
        _, tr = _trace(scn, seed)
        for node in range(N):
            ss = tr.sessions(node)
            for s in ss:
                assert s.node == node
                assert 0 <= s.start_ms < s.end_ms <= HORIZON_MS
            for prev, nxt in zip(ss, ss[1:]):
                # strictly after: a gap of at least 1 ms separates sessions
                assert prev.end_ms < nxt.start_ms


def test_focal_always_online():
    _, tr = _trace(HIGH_CHURN, 7)
    assert tr.sessions(FOCAL) == (tr.sessions(FOCAL)[0],)
    for t in range(0, HORIZON_MS, 1000):
        assert tr.is_online(FOCAL, t)
    assert all(e.node != FOCAL for e in tr.events())


def test_events_sorted_and_consistent_with_is_online():
    _, tr = _trace(HIGH_CHURN, 2)
    evs = list(tr.events())
    assert evs == sorted(evs)
    assert evs, "high churn must produce events"
    online = set(tr.online_at(0))
    for e in evs:
        assert 0 < e.t_ms < HORIZON_MS
        if e.kind == "up":
            assert e.node not in online
            online.add(e.node)
        else:
            assert e.node in online
            online.remove(e.node)
        assert tr.is_online(e.node, e.t_ms) == (e.kind == "up")
        assert tr.is_online(e.node, e.t_ms - 1) == (e.kind == "down")
    assert online == set(tr.online_at(HORIZON_MS - 1))


@pytest.mark.parametrize("scn", [HIGH_CHURN, STABLE])
def test_online_fraction_within_3pp_of_analytic(scn):
    fracs = []
    for seed in range(20):
        _, tr = _trace(scn, seed)
        for t in range(0, HORIZON_MS, 30_000):
            on = tr.online_at(t)
            fracs.append((len(on) - 1) / (N - 1))  # exclude F
    assert abs(float(np.mean(fracs)) - analytic_online_fraction(scn)) <= 0.03


def test_analytic_online_fraction_formula():
    assert analytic_online_fraction(HIGH_CHURN) == pytest.approx(
        0.3 * 3600 / 4200 + 0.7 * 300 / 900)


def test_older_sessions_survive_next_10_min_more_often():
    """Decreasing Weibull hazard: older sessions outlive the next 10 min more often."""
    ten_min = 600_000
    ages, survived = [], []
    for seed in range(10):
        _, tr = _trace(HIGH_CHURN, seed)
        for t in range(1_200_000, HORIZON_MS - ten_min, 60_000):
            for node in range(N):
                if node == FOCAL:
                    continue
                s = tr.current_session(node, t)
                if s is None:
                    continue
                ages.append(t - s.start_ms)
                survived.append(s.end_ms >= t + ten_min)
    ages_a = np.array(ages)
    surv = np.array(survived)
    med = np.median(ages_a)
    young, old = surv[ages_a < med].mean(), surv[ages_a >= med].mean()
    assert old > young + 0.05
