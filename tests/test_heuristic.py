"""Tests for policies/heuristic.py (Requirement 18)."""

from __future__ import annotations

import numpy as np
import pytest

from churnmaster.actions import Action
from churnmaster.policies.heuristic import (
    DEFAULT_CONFIG,
    Heuristic,
    HeuristicConfig,
    all_configs,
    as_table,
    heuristic_action,
)
from churnmaster.rl.state import (
    AGE_MID,
    AGE_OLD,
    AGE_YOUNG,
    CAND_FRESH,
    CAND_NONE,
    CAND_STALE,
    N_STATES,
    TIER_OTHER,
    TIER_SECOND,
    TIER_TOP,
    Features,
    allowed_actions,
    discretize,
    encode,
    reachable_states,
)
from churnmaster.sim.observation import DecisionContext

K, P, R = Action.KEEP, Action.PROBE, Action.REPLACE


def cfg(f=2, po=False, rs=False, ps=False, tg="all") -> HeuristicConfig:
    return HeuristicConfig(f, po, rs, ps, tg)


def feat(tier=TIER_TOP, fail=0, stale=0, age=AGE_MID, cand=CAND_FRESH, faster=0, churn=0):
    return Features(tier, fail, stale, age, cand, faster, churn)


# ---------------------------------------------------------------- configs


def test_exactly_32_unique_configs_in_fixed_order():
    cs = all_configs()
    assert len(cs) == 32 and len(set(cs)) == 32
    assert cs[0] == cfg(1, False, False, False, "all")
    assert cs[1] == cfg(1, False, False, False, "top_two")
    assert cs[-1] == cfg(2, True, True, True, "top_two")
    assert cs == all_configs()
    assert DEFAULT_CONFIG in cs


def test_ids_unique_and_round_trip():
    ids = [c.id for c in all_configs()]
    assert len(set(ids)) == 32
    for c in all_configs():
        assert HeuristicConfig.from_id(c.id) == c
    assert DEFAULT_CONFIG.id == "f=2,po=1,rs=0,ps=1,tg=all"


@pytest.mark.parametrize("bad", ["", "f=3,po=1,rs=0,ps=1,tg=all", "f=2,po=2,rs=0,ps=1,tg=all",
                                 "f=2,po=1,rs=0,ps=1,tg=top", "f=2,po=1,rs=0,ps=1",
                                 "f=2,po=1,rs=0,ps=1,tg=all,x=1", "garbage"])
def test_from_id_rejects_malformed(bad):
    with pytest.raises(ValueError):
        HeuristicConfig.from_id(bad)


@pytest.mark.parametrize("kwargs", [dict(f=0), dict(f=3), dict(f=True), dict(po=1),
                                    dict(rs="yes"), dict(ps=None), dict(tg="top-two")])
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        cfg(**kwargs)


def test_policy_attributes():
    h = Heuristic()
    assert h.cfg == DEFAULT_CONFIG
    assert h.shares_decision_points is True
    assert h.name == "heuristic[f=2,po=1,rs=0,ps=1,tg=all]"


# ---------------------------------------------------------------- rule order


def test_rule1_threshold_f():
    assert heuristic_action(cfg(f=1), feat(fail=1)) == R
    assert heuristic_action(cfg(f=2), feat(fail=1)) == K
    assert heuristic_action(cfg(f=2), feat(fail=2)) == R
    assert heuristic_action(cfg(f=1), feat(fail=1, cand=CAND_NONE)) == P
    assert heuristic_action(cfg(f=1), feat(fail=2, cand=CAND_STALE)) == R


def test_rule1_protect_old_edge():
    # OLD incumbent with 1 failure is protected -> PROBE (even with a candidate)
    assert heuristic_action(cfg(f=1, po=True), feat(fail=1, age=AGE_OLD)) == P
    # 2 failures: not protected
    assert heuristic_action(cfg(f=1, po=True), feat(fail=2, age=AGE_OLD)) == R
    # MID age: not protected
    assert heuristic_action(cfg(f=1, po=True), feat(fail=1, age=AGE_MID)) == R
    # protect_old off
    assert heuristic_action(cfg(f=1, po=False), feat(fail=1, age=AGE_OLD)) == R
    # with f=2, fail>=2 always so protection never applies
    assert heuristic_action(cfg(f=2, po=True), feat(fail=2, age=AGE_OLD)) == R


def test_rule2_rtt_swap():
    base = dict(faster=1, cand=CAND_FRESH, age=AGE_YOUNG)
    assert heuristic_action(cfg(rs=True), feat(**base)) == R
    assert heuristic_action(cfg(rs=False), feat(**base)) == K
    assert heuristic_action(cfg(rs=True), feat(**{**base, "faster": 0})) == K
    assert heuristic_action(cfg(rs=True), feat(**{**base, "cand": CAND_STALE})) == K
    assert heuristic_action(cfg(rs=True), feat(**{**base, "age": AGE_MID})) == K


def test_rule3_probe_stale():
    assert heuristic_action(cfg(ps=True), feat(stale=1)) == P
    assert heuristic_action(cfg(ps=True), feat(stale=1, cand=CAND_STALE)) == P
    assert heuristic_action(cfg(ps=True), feat(stale=1, cand=CAND_NONE)) == K
    assert heuristic_action(cfg(ps=True), feat(stale=0)) == K
    assert heuristic_action(cfg(ps=False), feat(stale=1)) == K


def test_precedence():
    everything = feat(fail=2, stale=1, age=AGE_YOUNG, cand=CAND_FRESH, faster=1)
    c = cfg(f=2, po=True, rs=True, ps=True)
    # rule 1 beats rules 2 and 3
    assert heuristic_action(c, everything) == R
    assert heuristic_action(c, Features(**{**everything.__dict__, "cand": CAND_NONE,
                                           "cand_faster": 0})) == P
    # rule 2 beats rule 3 (rule 3 would PROBE)
    assert heuristic_action(c, feat(fail=1, stale=1, age=AGE_YOUNG, faster=1)) == R
    # rule 3 when rule 2 does not match
    assert heuristic_action(c, feat(fail=1, stale=1, age=AGE_MID, faster=1)) == P
    # rule 4
    assert heuristic_action(c, feat(fail=1)) == K


def test_tier_gate_applies_only_to_rules_2_and_3():
    c = cfg(f=1, rs=True, ps=True, tg="top_two")
    swap = dict(faster=1, cand=CAND_FRESH, age=AGE_YOUNG)
    for tier in (TIER_TOP, TIER_SECOND):
        assert heuristic_action(c, feat(tier=tier, **swap)) == R
        assert heuristic_action(c, feat(tier=tier, stale=1)) == P
    assert heuristic_action(c, feat(tier=TIER_OTHER, **swap)) == K
    assert heuristic_action(c, feat(tier=TIER_OTHER, stale=1)) == K
    # rule 1 is not gated
    assert heuristic_action(c, feat(tier=TIER_OTHER, fail=1)) == R
    # tier_gate="all" applies to OTHER tier too
    c_all = cfg(f=1, rs=True, ps=True, tg="all")
    assert heuristic_action(c_all, feat(tier=TIER_OTHER, **swap)) == R
    assert heuristic_action(c_all, feat(tier=TIER_OTHER, stale=1)) == P


def test_decide_on_hand_made_context():
    ctx = DecisionContext(bucket=7, epoch_idx=0, t_ms=0, inc_consec_fail=1,
                          inc_since_response_s=90.0, inc_age_s=60.0, inc_srtt_ms=100.0,
                          cand_since_seen_s=5.0, cand_last_rtt_ms=50.0, timeout_fraction=0.0)
    assert Heuristic(cfg(f=1)).decide(ctx) == R                    # rule 1
    assert Heuristic(cfg(f=2, rs=True, ps=True)).decide(ctx) == R  # rule 2
    assert Heuristic(cfg(f=2, ps=True)).decide(ctx) == P           # rule 3
    assert Heuristic(cfg(f=2)).decide(ctx) == K                    # rule 4
    # bucket 0 is OTHER tier: gated rules off
    other = DecisionContext(**{**ctx.__dict__, "bucket": 0})
    assert Heuristic(cfg(f=2, rs=True, ps=True, tg="top_two")).decide(other) == K


# ---------------------------------------------------------------- tables


@pytest.mark.parametrize("c", all_configs(), ids=lambda c: c.id)
def test_table_shape_and_all_actions_allowed(c):
    t = as_table(c)
    assert t.shape == (N_STATES,)
    for s in range(N_STATES):
        assert Action(int(t[s])) in allowed_actions(s)


def _edge_or_uniform(rng, edges, lo, hi):
    if rng.random() < 0.4:
        return float(rng.choice(edges))
    return float(rng.uniform(lo, hi))


def generate_contexts(n: int, seed: int = 12345) -> list[DecisionContext]:
    """Seeded contexts covering every bin, bin edges, and None values."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        has_cand = rng.random() < 0.7
        srtt = None if rng.random() < 0.15 else float(rng.uniform(1.0, 500.0))
        if has_cand:
            since_seen = _edge_or_uniform(rng, [0.0, 29.999, 30.0, 30.001, 300.0], 0.0, 300.0)
            if srtt is not None and rng.random() < 0.3:
                cand_rtt = float(0.7 * srtt)  # exactly on the faster edge (not faster)
            else:
                cand_rtt = float(rng.uniform(1.0, 500.0))
        else:
            since_seen = cand_rtt = None
        out.append(DecisionContext(
            # tiers roughly balanced: bucket 7 = TOP, 6 = SECOND, 0..5 = OTHER
            bucket=int(rng.choice([7, 6, int(rng.integers(0, 6))])),
            epoch_idx=i,
            t_ms=i * 1000,
            inc_consec_fail=int(rng.choice([0, 1, int(rng.integers(2, 5))])),
            inc_since_response_s=_edge_or_uniform(rng, [0.0, 59.999, 60.0, 60.001], 0.0, 200.0),
            inc_age_s=_edge_or_uniform(rng, [0.0, 119.999, 120.0, 1199.999, 1200.0], 0.0, 3000.0),
            inc_srtt_ms=srtt,
            cand_since_seen_s=since_seen,
            cand_last_rtt_ms=cand_rtt,
            timeout_fraction=_edge_or_uniform(rng, [0.0, 0.0999, 0.1, 1.0], 0.0, 1.0),
        ))
    return out


@pytest.fixture(scope="module")
def contexts():
    ctxs = generate_contexts(10_000)
    states = {encode(discretize(c)) for c in ctxs}
    # the generator covers (nearly) the whole reachable state space
    assert states <= set(reachable_states())
    assert len(states) >= 500
    return ctxs


@pytest.mark.parametrize("c", all_configs(), ids=lambda c: c.id)
def test_representability(c, contexts):
    """Table lookup equals the rule function on 10k contexts (Req 18.4)."""
    table = as_table(c)
    h = Heuristic(c)
    for ctx in contexts:
        assert table[encode(discretize(ctx))] == h.decide(ctx)
