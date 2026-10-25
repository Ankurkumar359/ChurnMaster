"""Tests for sim/network.py (Requirements 7.1-7.6, 9.3)."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, strategies as st

from churnmaster.config import ScenarioConfig, SimConfig
from churnmaster.seeding import Split, env_seed, make_streams
from churnmaster.sim.idspace import closest
from churnmaster.sim.keyed_rng import Purpose, keyed_normal, keyed_uniform
from churnmaster.sim.network import FOCAL_INDEX, Network, build_network

HORIZON_MS = 3_000_000
CFG = SimConfig()
SCN = ScenarioConfig(name="high_churn", stable_share=0.3, stable_session_mean_s=3600.0,
                     flaky_session_mean_s=600.0, downtime_mean_s=600.0)


def _net(episode: int = 0) -> Network:
    streams = make_streams(env_seed(1234, Split.TEST, 0, episode))
    return build_network(CFG, SCN, streams, HORIZON_MS)


@pytest.fixture(scope="module")
def net() -> Network:
    return _net()


def test_layout(net):
    assert net.focal == FOCAL_INDEX
    assert net.positions.shape == (CFG.n_nodes, 2)
    assert ((net.positions >= 0) & (net.positions < 1)).all()
    assert len(set(net.ids)) == CFG.n_nodes
    assert all(net.index_of(net.id_of(i)) == i for i in range(CFG.n_nodes))


def test_congestion_shape(net):
    prone = set(net.congestion)
    assert len(prone) == round(0.10 * (CFG.n_nodes - 1))
    assert FOCAL_INDEX not in prone
    for node, periods in net.congestion.items():
        prev_end = 0
        for s, e in periods:
            assert prev_end < s < e <= HORIZON_MS
            assert e - s == CFG.congestion_len_s * 1000 or e == HORIZON_MS
            assert net.is_congested(node, s) and not net.is_congested(node, e)
            prev_end = e
    # Mean gap close to 20 min across all prone nodes (loose bound).
    n_periods = sum(len(p) for p in net.congestion.values())
    expected = len(prone) * HORIZON_MS / ((CFG.congestion_gap_s + CFG.congestion_len_s) * 1000)
    assert 0.4 * expected < n_periods < 2.0 * expected


def test_loss_rate_close_to_two_percent(net):
    online = sorted(net.trace.online_at(0) - {FOCAL_INDEX})
    fails = 0
    for i in range(10_000):
        r = net.query(online[i % len(online)], 0, (i, online[i % len(online)]))
        fails += not r.success
    assert abs(fails / 10_000 - CFG.p_loss) <= 0.005


def test_offline_query_times_out(net):
    t = 600_000
    off = next(i for i in range(CFG.n_nodes) if not net.is_online(i, t))
    r = net.query(off, t, (7, off))
    assert not r.success and r.cost_ms == CFG.timeout_ms and r.rtt_ms is None


def test_same_key_same_outcome_across_networks():
    # Two independently built networks for the same episode see identical
    # luck for identical keys, regardless of interleaved calls.
    a, b = _net(), _net()
    t = 0
    node = sorted(a.trace.online_at(t) - {FOCAL_INDEX})[0]
    first = a.query(node, t, (3, node))
    for j in range(50):
        b.query(node, t, (j + 100, node))
    assert b.query(node, t, (3, node)) == first == a.query(node, t, (3, node))
    assert a.rtt_ms(node, t, (3, node)) == b.rtt_ms(node, t, (3, node))


@given(node=st.integers(1, CFG.n_nodes - 1), t=st.integers(0, HORIZON_MS - 1),
       key=st.tuples(st.integers(0, 10**6), st.integers(0, 255)))
def test_query_rtt_formula(net, node, t, key):
    """Property 9: query and RTT semantics. **Validates: Requirements 7.3, 7.4**"""
    r = net.query(node, t, key)
    lossless = keyed_uniform(net.keyed_root, Purpose.LOSS, *key) >= CFG.p_loss
    assert r.success == (net.is_online(node, t) and lossless)
    if r.success:
        d = math.dist(net.positions[0], net.positions[node]) / math.sqrt(2)
        z = keyed_normal(net.keyed_root, Purpose.JITTER, *key)
        exp = (10 + 240 * d) * math.exp(0.25 * z) * (3 if net.is_congested(node, t) else 1)
        assert r.cost_ms == pytest.approx(exp) and r.rtt_ms == r.cost_ms
    else:
        assert r.cost_ms == CFG.timeout_ms and r.rtt_ms is None


@given(key=st.integers(0, 255), t=st.integers(0, HORIZON_MS - 1))
def test_holders(net, key, t):
    """Property 10: replica holders. **Validates: Requirements 7.6**"""
    h = net.holders(key, t)
    online_ids = [net.id_of(i) for i in net.trace.online_at(t) if i != FOCAL_INDEX]
    assert FOCAL_INDEX not in h
    assert all(net.is_online(i, t) for i in h)
    assert [net.id_of(i) for i in h] == closest(key, online_ids, CFG.replicas)


def test_bootstrap_entries(net):
    entries = net.bootstrap_entries()
    assert entries == net.bootstrap_entries()
    per_bucket: dict[int, int] = {}
    for node_id, t_ms, rtt in entries:
        idx = net.index_of(node_id)
        assert idx != FOCAL_INDEX and net.is_online(idx, 0) and t_ms == 0 and rtt is None
        b = net.bucket_of(idx)
        per_bucket[b] = per_bucket.get(b, 0) + 1
    assert len({e.node for e in entries}) == len(entries)
    for b in range(CFG.id_bits):
        avail = sum(1 for i in net.trace.online_at(0)
                    if i != FOCAL_INDEX and net.bucket_of(i) == b)
        assert per_bucket.get(b, 0) == min(CFG.k, avail)
