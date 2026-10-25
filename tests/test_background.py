"""Tests for sim/background.py (Requirements 8.1-8.4)."""

from __future__ import annotations

import time

import pytest

from churnmaster.config import ScenarioConfig, SimConfig
from churnmaster.seeding import Split, env_seed, make_streams
from churnmaster.sim.background import BackgroundTables
from churnmaster.sim.idspace import bucket_index
from churnmaster.sim.keyed_rng import Purpose, keyed_uniform
from churnmaster.sim.network import FOCAL_INDEX, Network, build_network

HORIZON_MS = 3_000_000
CFG = SimConfig()
SCN = ScenarioConfig(name="high_churn", stable_share=0.3, stable_session_mean_s=3600.0,
                     flaky_session_mean_s=600.0, downtime_mean_s=600.0)
PERIOD_MS = CFG.rebuild_period_s * 1000


def _net(episode: int = 0) -> Network:
    return build_network(CFG, SCN, make_streams(env_seed(1234, Split.TEST, 0, episode)),
                         HORIZON_MS)


@pytest.fixture(scope="module")
def net() -> Network:
    return _net()


@pytest.fixture()
def bg(net) -> BackgroundTables:
    b = BackgroundTables(net)
    b.build_all(0)
    return b


# ------------------------------------------------------------------ schedule
def test_schedule_staggered_and_sorted(net):
    bg = BackgroundTables(net)
    sched = list(bg.rebuild_schedule(HORIZON_MS))
    assert sched == sorted(sched)
    n = net.n_nodes
    assert all(node != FOCAL_INDEX for _, node, _ in sched)
    # Initial build: every non-focal node exactly once at t=0 with idx 0.
    at_zero = [(node, idx) for t, node, idx in sched if t == 0]
    assert at_zero == [(u, 0) for u in range(n) if u != FOCAL_INDEX]
    for u in range(1, n):
        times = [(t, idx) for t, node, idx in sched if node == u]
        assert [idx for _, idx in times] == list(range(len(times)))
        offset = u * PERIOD_MS // n
        expected = [0] + [offset + PERIOD_MS * m for m in range(HORIZON_MS // PERIOD_MS + 1)
                          if 0 < offset + PERIOD_MS * m < HORIZON_MS]
        assert [t for t, _ in times] == expected
    # Staggering: later rebuilds of distinct nodes never share a timestamp.
    later = [t for t, _, _ in sched if t > 0]
    assert len(later) == len(set(later))


def test_schedule_respects_horizon(net):
    bg = BackgroundTables(net)
    assert list(bg.rebuild_schedule(0)) == []
    assert all(t < 1000 for t, _, _ in bg.rebuild_schedule(1000))


def test_focal_has_no_table(bg):
    with pytest.raises(ValueError):
        bg.rebuild(FOCAL_INDEX, 0, 0)
    with pytest.raises(ValueError):
        bg.closest_known(FOCAL_INDEX, 5)


# ------------------------------------------------------------------ contents
def test_buckets_valid_after_rebuild(net, bg):
    t = 900_000
    for u in range(1, net.n_nodes):
        bg.rebuild(u, 3, t)
        for b, bucket in enumerate(bg.table(u)):
            assert len(bucket) <= CFG.k
            assert len(set(bucket)) == len(bucket)
            for c in bucket:
                assert c != u
                assert net.is_online(c, t)
                assert bucket_index(net.id_of(u), net.id_of(c)) == b
            # Bucket is full whenever enough online candidates exist.
            n_online = sum(1 for v in range(net.n_nodes) if v != u and net.is_online(v, t)
                           and bucket_index(net.id_of(u), net.id_of(v)) == b)
            assert len(bucket) == min(CFG.k, n_online)


def test_rebuild_deterministic_and_keyed(net):
    a, b = BackgroundTables(_net()), BackgroundTables(_net())
    for u in range(1, net.n_nodes):
        a.rebuild(u, 2, 600_000)
        b.rebuild(u, 2, 600_000)
        assert a.table(u) == b.table(u)
    # A different rebuild index (same online set) reshuffles some picks.
    changed = 0
    for u in range(1, net.n_nodes):
        before = a.table(u)
        a.rebuild(u, 5, 600_000)
        changed += before != a.table(u)
    assert changed > 0


def test_selection_matches_keyed_order(net, bg):
    # Each over-full bucket keeps the k online candidates with the smallest
    # keyed_uniform(root, BG, node, rebuild_idx, candidate).
    t, idx = 1_200_000, 4
    for u in range(1, net.n_nodes):
        bg.rebuild(u, idx, t)
        for b, bucket in enumerate(bg.table(u)):
            online = [v for v in range(net.n_nodes) if v != u and net.is_online(v, t)
                      and bucket_index(net.id_of(u), net.id_of(v)) == b]
            online.sort(key=lambda c: (keyed_uniform(net.keyed_root, Purpose.BG, u, idx, c), c))
            assert sorted(bucket) == sorted(online[:CFG.k])


# ----------------------------------------------------------------- staleness
def test_table_unchanged_between_rebuilds_and_returns_dead(net, bg):
    # Snapshot after the t=0 build; no rebuild happens until each node's next
    # scheduled time, so queries at later times see the frozen table.
    snap = {u: bg.table(u) for u in range(1, net.n_nodes)}
    t_late = PERIOD_MS - 1  # churn has happened, but before any m=1 rebuild
    dead_seen = 0
    for u in range(1, net.n_nodes):
        assert bg.table(u) == snap[u]
        for target in range(0, 256, 17):
            got = bg.closest_known(u, target, 4)
            flat = [c for bucket in snap[u] for c in bucket]
            assert got == sorted(flat, key=lambda c: net.id_of(c) ^ target)[:4]
            dead_seen += sum(not net.is_online(c, t_late) for c in got)
    assert dead_seen > 0  # stale tables hand out offline contacts


def test_closest_known_order_and_n(net, bg):
    u, target = 5, 0xA7
    got = bg.closest_known(u, target)
    d = [net.id_of(c) ^ target for c in got]
    assert len(got) == 4 and d == sorted(d)
    assert bg.closest_known(u, target, 0) == []
    assert len(bg.closest_known(u, target, 100)) == sum(len(x) for x in bg.table(u))


def test_rebuild_cost_is_small(net):
    bg = BackgroundTables(net)
    sched = list(bg.rebuild_schedule(HORIZON_MS))
    start = time.perf_counter()
    for t, u, i in sched:
        bg.rebuild(u, i, t)
    assert time.perf_counter() - start < 2.0
