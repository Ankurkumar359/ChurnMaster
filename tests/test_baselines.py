"""Tests for policies/base.py and policies/baselines.py (Requirement 17)."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from churnmaster.actions import Action
from churnmaster.policies.base import Policy, allowed, has_candidate
from churnmaster.policies.baselines import (
    BASELINE_NAMES, AlwaysReplace, KademliaEpoch, KademliaStd, MatchedRandom,
    NeverReplace, UniformRandom, make_baseline,
)
from churnmaster.policies.oracle import OracleFacts, OracleHeuristic
from churnmaster.rl.state import allowed_actions, discretize, encode
from churnmaster.seeding import baseline_rng
from churnmaster.sim.observation import DecisionContext

FREQS = {False: [0.7, 0.3, 0.0], True: [0.2, 0.3, 0.5]}


def ctx(cand: bool, **kw) -> DecisionContext:
    base = dict(
        bucket=7, epoch_idx=3, t_ms=90_000, inc_consec_fail=1,
        inc_since_response_s=70.0, inc_age_s=300.0, inc_srtt_ms=100.0,
        cand_since_seen_s=10.0 if cand else None,
        cand_last_rtt_ms=50.0 if cand else None, timeout_fraction=0.2,
    )
    base.update(kw)
    return DecisionContext(**base)


WITH, WITHOUT = ctx(True), ctx(False)


def test_helpers_agree_with_state_masking():
    assert has_candidate(WITH) and not has_candidate(WITHOUT)
    for c in (WITH, WITHOUT, ctx(True, cand_since_seen_s=200.0)):
        assert allowed(c) == allowed_actions(encode(discretize(c)))
    assert Action.REPLACE not in allowed(WITHOUT)
    assert Action.REPLACE in allowed(WITH)


@pytest.mark.parametrize("cls, with_c, without_c", [
    (NeverReplace, Action.KEEP, Action.KEEP),
    (AlwaysReplace, Action.REPLACE, Action.KEEP),
    (KademliaEpoch, Action.PROBE, Action.KEEP),
    (KademliaStd, Action.KEEP, Action.KEEP),
])
def test_deterministic_baselines(cls, with_c, without_c):
    p = cls()
    assert p.decide(WITH) is with_c
    assert p.decide(WITHOUT) is without_c


def test_kademlia_std_attributes():
    p = KademliaStd()
    assert p.name == "kademlia_std"
    assert p.shares_decision_points is False
    assert p.env_mode == "kademlia_std"
    for cls in (NeverReplace, AlwaysReplace, KademliaEpoch):
        assert cls().shares_decision_points is True


def test_protocol_conformance_and_names():
    rng = np.random.default_rng(0)
    for name in BASELINE_NAMES:
        p = make_baseline(name, rng=rng, freqs=FREQS)
        assert isinstance(p, Policy)
        assert p.name == name
    oracle = OracleHeuristic(lambda b, t: OracleFacts(True, False, None))
    assert isinstance(oracle, Policy)


def test_make_baseline_errors():
    with pytest.raises(ValueError):
        make_baseline("nope")
    with pytest.raises(ValueError):
        make_baseline("uniform_random")
    with pytest.raises(ValueError):
        make_baseline("matched_random", freqs=FREQS)
    with pytest.raises(ValueError):
        make_baseline("matched_random", rng=np.random.default_rng(0))


@pytest.mark.parametrize("make", [
    lambda r: UniformRandom(r),
    lambda r: MatchedRandom(FREQS, r),
])
def test_random_policies_masked_and_seeded(make):
    a = make(baseline_rng(1, 0, 2))
    b = make(baseline_rng(1, 0, 2))
    seq_a = [a.decide(c) for c in [WITH, WITHOUT] * 500]
    seq_b = [b.decide(c) for c in [WITH, WITHOUT] * 500]
    assert seq_a == seq_b
    assert all(isinstance(x, Action) for x in seq_a)
    assert all(x != Action.REPLACE for x in seq_a[1::2])


def test_uniform_random_frequencies():
    p = UniformRandom(np.random.default_rng(42))
    n = 6000
    cw = Counter(p.decide(WITH) for _ in range(n))
    assert set(cw) == {Action.KEEP, Action.PROBE, Action.REPLACE}
    assert all(abs(v / n - 1 / 3) < 0.03 for v in cw.values())
    co = Counter(p.decide(WITHOUT) for _ in range(n))
    assert set(co) == {Action.KEEP, Action.PROBE}
    assert all(abs(v / n - 1 / 2) < 0.03 for v in co.values())


def test_matched_random_reproduces_frequencies():
    p = MatchedRandom(FREQS, np.random.default_rng(7))
    n = 10_000
    for c, key in ((WITH, True), (WITHOUT, False)):
        counts = Counter(p.decide(c) for _ in range(n))
        for a in Action:
            assert abs(counts.get(a, 0) / n - FREQS[key][int(a)]) < 0.02


@pytest.mark.parametrize("bad", [
    {False: [0.5, 0.4, 0.1], True: [0.2, 0.3, 0.5]},   # REPLACE without candidate
    {False: [0.5, 0.4, 0.0], True: [0.2, 0.3, 0.5]},   # sum != 1
    {False: [1.0, 0.0, 0.0], True: [1.2, -0.2, 0.0]},  # negative
    {False: [1.0, 0.0], True: [0.2, 0.3, 0.5]},        # wrong length
    {True: [0.2, 0.3, 0.5]},                           # missing key
])
def test_matched_random_validation(bad):
    with pytest.raises(ValueError):
        MatchedRandom(bad, np.random.default_rng(0))


def test_random_policies_use_only_injected_rng():
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    p = UniformRandom(np.random.default_rng(0))
    m = MatchedRandom(FREQS, np.random.default_rng(0))
    for _ in range(100):
        p.decide(WITH)
        m.decide(WITH)
    assert np.array_equal(np.random.get_state()[1], before)
