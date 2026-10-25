"""Tests for sim/lookup.py (Requirements 10.1-10.10)."""

from __future__ import annotations

import numpy as np
import pytest

from churnmaster.config import SimConfig
from churnmaster.sim.background import BackgroundTables
from churnmaster.sim.churn import ChurnTrace, NodeType, Session
from churnmaster.sim.idspace import bucket_index
from churnmaster.sim.lookup import LookupResult, run_lookup, run_lookup_traced
from churnmaster.sim.network import FOCAL_INDEX, Network
from churnmaster.sim.routing_table import LocalView
from churnmaster.sim.workload import LookupRequest

HORIZON_MS = 10_000_000
T0 = 5_000


def _make_net(cfg: SimConfig, ids: list[int], pos: list[float],
              sessions: dict[int, tuple[int, int]] | None = None) -> Network:
    """Hand-built network; node i at (pos[i], pos[i]) so with rtt_sigma=0 its
    RTT is exactly rtt_base + rtt_span * pos[i]. Nodes are always online
    unless ``sessions`` gives ``(start, end)`` (or None -> never online)."""
    sessions = sessions or {}
    n = len(ids)
    sess = []
    for i in range(n):
        s = sessions.get(i, (0, HORIZON_MS))
        sess.append([] if s is None else [Session(i, s[0], s[1])])
    trace = ChurnTrace(n, HORIZON_MS, sess)
    positions = np.array([[p, p] for p in pos], dtype=float)
    types = [NodeType(stable=True)] * n
    return Network(cfg, ids, positions, types, trace, {}, keyed_root=99, focal=FOCAL_INDEX)


def _setup(cfg, ids, pos, table_ids, sessions=None, view_k=4):
    net = _make_net(cfg, ids, pos, sessions)
    bg = BackgroundTables(net)
    bg.build_all(0)
    view = LocalView(net.focal_id, id_bits=cfg.id_bits, k=view_k)
    view.bootstrap([(nid, 0, None) for nid in table_ids])
    return net, bg, view


# Small layout: F=0, A=128 (RTT 70), B=64 (RTT 130), H=201 (holder for K=200,
# RTT 40). F knows only A and B; A and B know H via background tables.
CFG_SMALL = SimConfig(n_nodes=4, p_loss=0.0, rtt_sigma=0.0, replicas=1)
IDS = [0, 128, 64, 201]
POS = [0.0, 0.25, 0.5, 0.125]
A, B, H = 128, 64, 201
KEY = 200


def _req(key=KEY, j=7, t=T0):
    return LookupRequest(lookup_id=j, t_ms=t, key=key)


# ------------------------------------------------------- hand-computed cases
def test_hand_computed_latency_and_attribution():
    net, bg, view = _setup(CFG_SMALL, IDS, POS, [A, B])
    res, gt = run_lookup_traced(_req(), view, net, bg, CFG_SMALL)
    # Round 1 queries A and B (max 130 ms); round 2 hits H (40 ms).
    assert res == LookupResult(7, T0, KEY, 7, True, pytest.approx(170.0), 2, 3, None)
    assert gt.holders == (H,)
    assert gt.queried == (A, B, H)
    assert 0 not in gt.queried                          # F never queried
    assert res.bucket == bucket_index(0, KEY)


def test_run_lookup_matches_traced_and_is_deterministic():
    net, bg, view = _setup(CFG_SMALL, IDS, POS, [A, B])
    a = run_lookup(_req(), view, net, bg, CFG_SMALL)
    net2, bg2, view2 = _setup(CFG_SMALL, IDS, POS, [A, B])
    b, _ = run_lookup_traced(_req(), view2, net2, bg2, CFG_SMALL)
    assert a == b


def test_key_equal_focal_attributed_to_bucket_zero():
    net, bg, view = _setup(CFG_SMALL, IDS, POS, [A, B])
    res = run_lookup(_req(key=0), view, net, bg, CFG_SMALL)
    assert res.bucket == 0


def test_success_at_exact_deadline_and_deadline_failure():
    cfg = SimConfig(n_nodes=4, p_loss=0.0, rtt_sigma=0.0, replicas=1, deadline_ms=170)
    net, bg, view = _setup(cfg, IDS, POS, [A, B])
    res = run_lookup(_req(), view, net, bg, cfg)
    assert res.success and res.latency_ms == pytest.approx(170.0)

    cfg = SimConfig(n_nodes=4, p_loss=0.0, rtt_sigma=0.0, replicas=1, deadline_ms=169)
    net, bg, view = _setup(cfg, IDS, POS, [A, B])
    res = run_lookup(_req(), view, net, bg, cfg)
    assert not res.success and res.fail_reason == "deadline"
    assert res.latency_ms == 169.0 and res.rounds == 2


def test_round_limit():
    cfg = SimConfig(n_nodes=4, p_loss=0.0, rtt_sigma=0.0, replicas=1, max_rounds=1)
    net, bg, view = _setup(cfg, IDS, POS, [A, B])
    res = run_lookup(_req(), view, net, bg, cfg)
    assert (res.success, res.fail_reason, res.rounds, res.queries) == (False, "rounds", 1, 2)
    assert res.latency_ms == pytest.approx(130.0)


# Dead layout: F's table holds 4 offline nodes; holder H is online but unknown.
IDS_DEAD = [0, 192, 224, 160, 176, 201]
POS_DEAD = [0.0, 0.1, 0.1, 0.1, 0.1, 0.1]
DEAD = {1: None, 2: None, 3: None, 4: None}


def test_exhaustion_with_dead_table():
    net, bg, view = _setup(CFG_SMALL, IDS_DEAD, POS_DEAD, [192, 224, 160, 176], DEAD)
    res = run_lookup(_req(), view, net, bg, CFG_SMALL)
    # Two rounds of two timeouts each, then nothing left to query.
    assert (res.success, res.fail_reason, res.rounds, res.queries) == (False, "exhausted", 2, 4)
    assert res.latency_ms == 1000.0
    assert view.timeout_fraction(T0) == 1.0
    assert all(view.entry(n).consec_fail == 1 for n in [192, 224, 160, 176])


def test_deadline_with_timeouts():
    cfg = SimConfig(n_nodes=6, p_loss=0.0, rtt_sigma=0.0, replicas=1, deadline_ms=900)
    net, bg, view = _setup(cfg, IDS_DEAD, POS_DEAD, [192, 224, 160, 176], DEAD)
    res = run_lookup(_req(), view, net, bg, cfg)
    assert (res.success, res.fail_reason, res.rounds, res.latency_ms) == \
        (False, "deadline", 2, 900.0)


def test_empty_table_exhausts_immediately():
    net, bg, view = _setup(CFG_SMALL, IDS, POS, [])
    res = run_lookup(_req(), view, net, bg, CFG_SMALL)
    assert (res.success, res.fail_reason, res.rounds, res.queries, res.latency_ms) == \
        (False, "exhausted", 0, 0, 0.0)


# ------------------------------------------------------------- side effects
# F's table: D=192 (dead, closest to K) and A=128; H=201 holder; view k=2 so
# bucket 7 (D, A) is full and H must be cached.
IDS_SE = [0, 128, 192, 201]
POS_SE = [0.0, 0.25, 0.1, 0.125]


def test_side_effects_stats_timeouts_and_cache():
    net, bg, view = _setup(CFG_SMALL, IDS_SE, POS_SE, [192, 128], {2: None}, view_k=2)
    res = run_lookup(_req(), view, net, bg, CFG_SMALL)
    assert res.success and res.latency_ms == pytest.approx(540.0)   # 500 timeout + 40
    d, a = view.entry(192), view.entry(128)
    assert d.consec_fail == 1 and d.last_response_ms is None
    assert a.consec_fail == 0 and a.last_response_ms == T0 and a.srtt_ms == pytest.approx(70.0)
    assert view.timeout_fraction(T0) == pytest.approx(1 / 3)
    assert H not in view and view.in_cache(H)
    assert view.candidate(7, T0).last_rtt_ms == pytest.approx(40.0)


def test_responders_inserted_into_empty_slots():
    net, bg, view = _setup(CFG_SMALL, IDS, POS, [A, B])
    run_lookup(_req(), view, net, bg, CFG_SMALL)
    e = view.entry(H)
    assert (e.inserted_ms, e.last_response_ms) == (T0, T0)
    assert e.srtt_ms == pytest.approx(40.0)


def test_atomic_evaluation_at_start_time():
    # H goes offline 50 ms after t0, before the 170 ms latency elapses; the
    # lookup still sees it online because everything is evaluated at t0.
    net, bg, view = _setup(CFG_SMALL, IDS, POS, [A, B], {3: (0, T0 + 50)})
    res = run_lookup(_req(), view, net, bg, CFG_SMALL)
    assert res.success and res.latency_ms == pytest.approx(170.0)


# ----------------------------------------------------------- static network
def test_static_network_all_succeed_few_rounds():
    cfg = SimConfig(p_loss=0.0)
    rng = np.random.default_rng(2024)
    ids = [int(x) for x in rng.choice(256, size=cfg.n_nodes, replace=False)]
    trace = ChurnTrace(cfg.n_nodes, HORIZON_MS,
                       [[Session(i, 0, HORIZON_MS)] for i in range(cfg.n_nodes)])
    net = Network(cfg, ids, rng.random((cfg.n_nodes, 2)),
                  [NodeType(stable=True)] * cfg.n_nodes, trace, {}, keyed_root=12345)
    bg = BackgroundTables(net)
    bg.build_all(0)
    view = LocalView(net.focal_id)
    view.bootstrap(net.bootstrap_entries(0))
    keys = rng.integers(0, 256, size=1000)
    results = [run_lookup(LookupRequest(j, 1000 + j * 1000, int(k)), view, net, bg, cfg)
               for j, k in enumerate(keys)]
    assert sum(r.success for r in results) == 1000
    assert np.mean([r.rounds for r in results]) <= 3.0
    assert all(r.latency_ms <= cfg.deadline_ms for r in results)
