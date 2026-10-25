"""Tests for sim/maintenance.py (Requirements 14.3-14.9, 17.3)."""

from __future__ import annotations

import numpy as np
import pytest

from churnmaster.config import SimConfig
from churnmaster.sim.churn import ChurnTrace, NodeType, Session
from churnmaster.sim.keyed_rng import PingKind, Purpose, keyed_uniform
from churnmaster.sim.maintenance import (
    Action,
    ActionOutcome,
    IllegalActionError,
    apply_action,
    kademlia_std_on_offer,
)
from churnmaster.sim.network import FOCAL_INDEX, Network
from churnmaster.sim.routing_table import LocalView

HORIZON_MS = 10_000_000
T = 60_000
ROOT = 99
B = 7                      # bucket of IDs 128..255 relative to F=0
INC, OTHER, CAND = 128, 129, 130
IDS = [0, INC, OTHER, CAND]
POS = [0.0, 0.25, 0.5, 0.125]       # rtt = 10 + 240 * pos with rtt_sigma=0 (defaults)
CFG = SimConfig(n_nodes=4, p_loss=0.0, rtt_sigma=0.0)


def _rtt(cfg: SimConfig, i: int) -> float:
    return cfg.rtt_base_ms + cfg.rtt_span_ms * POS[i]


def _setup(offline: tuple[int, ...] = (), cand: bool = True, cfg: SimConfig = CFG):
    """F's bucket 7 is full (k=2) with INC (LRS) and OTHER; CAND optionally cached."""
    n = len(IDS)
    sess = [[] if i in offline else [Session(i, 0, HORIZON_MS)] for i in range(n)]
    trace = ChurnTrace(n, HORIZON_MS, sess)
    positions = np.array([[p, p] for p in POS], dtype=float)
    net = Network(cfg, IDS, positions, [NodeType(stable=True)] * n, trace, {},
                  keyed_root=ROOT, focal=FOCAL_INDEX)
    view = LocalView(0, id_bits=cfg.id_bits, k=2)
    view.bootstrap([(INC, 0, None), (OTHER, 1_000, None)])
    if cand:
        view.cache_put(B, CAND, T - 5_000, 55.0)
    return net, view


def _nodes(view: LocalView) -> set[int]:
    return {e.node for e in view.bucket(B)}


def _cached(view: LocalView) -> set[int]:
    return {c.node for c in view.cache(B)}


# ------------------------------------------------------------------ KEEP
def test_keep_sends_nothing_and_changes_nothing():
    net, view = _setup()
    out = apply_action(B, Action.KEEP, 3, T, view, net, CFG)
    assert (out.pings, out.evictions, out.inserted, out.evicted) == (0, 0, None, None)
    assert _nodes(view) == {INC, OTHER} and _cached(view) == {CAND}
    assert view.entry(INC).last_response_ms is None
    assert view.timeout_fraction(T) == 0.0 and not view._queries


# ----------------------------------------------------------------- PROBE
def test_probe_incumbent_answers_refreshes():
    net, view = _setup()
    out = apply_action(B, Action.PROBE, 3, T, view, net, CFG)
    assert (out.pings, out.evictions, out.inserted, out.evicted) == (1, 0, None, None)
    e = view.entry(INC)
    assert e.last_response_ms == T and e.srtt_ms == pytest.approx(_rtt(CFG, 1))
    assert _nodes(view) == {INC, OTHER} and _cached(view) == {CAND}
    assert view.incumbent(B).node == OTHER          # LRS moved
    assert view.timeout_fraction(T) == 0.0


def test_probe_fail_no_candidate_leaves_slot():
    net, view = _setup(offline=(1,), cand=False)
    out = apply_action(B, Action.PROBE, 3, T, view, net, CFG)
    assert (out.pings, out.evictions, out.inserted, out.evicted) == (1, 1, None, INC)
    assert _nodes(view) == {OTHER}
    assert view.timeout_fraction(T) == 1.0


def test_probe_fail_candidate_answers_inserted():
    net, view = _setup(offline=(1,))
    out = apply_action(B, Action.PROBE, 3, T, view, net, CFG)
    assert (out.pings, out.evictions, out.inserted, out.evicted) == (2, 1, CAND, INC)
    assert _nodes(view) == {OTHER, CAND} and _cached(view) == set()
    e = view.entry(CAND)
    assert e.inserted_ms == T and e.srtt_ms == pytest.approx(_rtt(CFG, 3))
    assert view.timeout_fraction(T) == pytest.approx(0.5)


def test_probe_fail_candidate_fails_dropped():
    net, view = _setup(offline=(1, 3))
    out = apply_action(B, Action.PROBE, 3, T, view, net, CFG)
    assert (out.pings, out.evictions, out.inserted, out.evicted) == (2, 1, None, INC)
    assert _nodes(view) == {OTHER} and _cached(view) == set()
    assert view.timeout_fraction(T) == 1.0


# --------------------------------------------------------------- REPLACE
def test_replace_candidate_answers_swaps():
    net, view = _setup()
    out = apply_action(B, Action.REPLACE, 3, T, view, net, CFG)
    assert (out.pings, out.evictions, out.inserted, out.evicted) == (1, 1, CAND, INC)
    assert out.inc_node == INC and out.cand_node == CAND
    assert _nodes(view) == {OTHER, CAND} and _cached(view) == set()
    assert view.timeout_fraction(T) == 0.0


def test_replace_candidate_fails_keeps_incumbent():
    net, view = _setup(offline=(3,))
    out = apply_action(B, Action.REPLACE, 3, T, view, net, CFG)
    assert (out.pings, out.evictions, out.inserted, out.evicted) == (1, 0, None, None)
    assert _nodes(view) == {INC, OTHER} and _cached(view) == set()
    assert view.entry(INC).consec_fail == 0          # incumbent untouched
    assert view.timeout_fraction(T) == 1.0


def test_replace_without_candidate_is_illegal():
    net, view = _setup(cand=False)
    with pytest.raises(IllegalActionError):
        apply_action(B, Action.REPLACE, 3, T, view, net, CFG)
    assert issubclass(IllegalActionError, ValueError)
    assert _nodes(view) == {INC, OTHER} and not view._queries


def test_expired_candidate_counts_as_absent():
    net, view = _setup()
    t_late = T + view.cache_ttl_ms           # cached at T-5000 -> expired
    with pytest.raises(IllegalActionError):
        apply_action(B, Action.REPLACE, 3, t_late, view, net, CFG)


def test_empty_bucket_raises_value_error():
    net, view = _setup()
    with pytest.raises(ValueError):
        apply_action(0, Action.KEEP, 3, T, view, net, CFG)


# ------------------------------------------------------------ key schema
def test_epoch_ping_keys(monkeypatch):
    net, view = _setup(offline=(1,))
    seen = []
    orig = net.query
    monkeypatch.setattr(net, "query", lambda i, t, key: seen.append((i, t, key)) or orig(i, t, key))
    apply_action(B, Action.PROBE, 11, T, view, net, CFG)
    assert seen == [(1, T, (11, B, INC, PingKind.INC)), (3, T, (11, B, CAND, PingKind.CAND))]


def test_loss_follows_keyed_draw_per_epoch_and_kind():
    cfg = SimConfig(n_nodes=4, p_loss=0.5, rtt_sigma=0.0)
    inc_ok, cand_ok = [], []
    for k in range(40):
        net, view = _setup(cfg=cfg)
        out = apply_action(B, Action.PROBE, k, T, view, net, cfg)
        exp = keyed_uniform(ROOT, Purpose.LOSS, k, B, INC, PingKind.INC) >= 0.5
        assert (out.evictions == 0) == exp
        inc_ok.append(exp)
        net, view = _setup(cfg=cfg)
        out = apply_action(B, Action.REPLACE, k, T, view, net, cfg)
        exp_c = keyed_uniform(ROOT, Purpose.LOSS, k, B, CAND, PingKind.CAND) >= 0.5
        assert (out.inserted == CAND) == exp_c
        cand_ok.append(exp_c)
    # Different epochs give different luck (not a constant outcome).
    assert 0 < sum(inc_ok) < 40 and 0 < sum(cand_ok) < 40


# ----------------------------------------------------------- kademlia_std
def test_kademlia_std_incumbent_fails_newcomer_inserted(monkeypatch):
    net, view = _setup(offline=(1,))
    seen = []
    orig = net.query
    monkeypatch.setattr(net, "query", lambda i, t, key: seen.append(key) or orig(i, t, key))
    out = kademlia_std_on_offer(B, CAND, 42, T, view, net, CFG)
    assert out == ActionOutcome(1, 1, CAND, INC, INC, CAND)
    assert seen == [(42, B, INC, PingKind.STD)]
    assert _nodes(view) == {OTHER, CAND} and _cached(view) == set()
    e = view.entry(CAND)
    assert e.inserted_ms == T and e.srtt_ms == pytest.approx(55.0)   # cached RTT
    assert view.timeout_fraction(T) == 1.0


def test_kademlia_std_incumbent_answers_newcomer_stays_cached():
    net, view = _setup()
    out = kademlia_std_on_offer(B, CAND, 42, T, view, net, CFG)
    assert (out.pings, out.evictions, out.inserted, out.evicted) == (1, 0, None, None)
    assert _nodes(view) == {INC, OTHER} and _cached(view) == {CAND}
    assert view.entry(INC).last_response_ms == T
    assert view.incumbent(B).node == OTHER
    assert view.timeout_fraction(T) == 0.0


def test_kademlia_std_preconditions():
    net, view = _setup(cand=False)
    with pytest.raises(ValueError):
        kademlia_std_on_offer(B, CAND, 1, T, view, net, CFG)     # not cached
    with pytest.raises(ValueError):
        kademlia_std_on_offer(B, OTHER, 1, T, view, net, CFG)    # already in table
    view.evict(OTHER)
    view.cache_put(B, CAND, T, 50.0)
    with pytest.raises(ValueError):
        kademlia_std_on_offer(B, CAND, 1, T, view, net, CFG)     # bucket not full
