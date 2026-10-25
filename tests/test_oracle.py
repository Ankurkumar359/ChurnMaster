"""Tests for policies/oracle.py (Requirements 19.1, 19.2)."""

from __future__ import annotations

import pytest

from churnmaster.actions import Action
from churnmaster.policies.oracle import OracleFacts, OracleHeuristic, oracle_action
from churnmaster.sim.observation import DecisionContext

# (inc_online, cand_present, cand_online) -> expected action; all GT combinations.
CASES = [
    (False, True, True, Action.REPLACE),
    (False, True, False, Action.PROBE),
    (False, False, None, Action.PROBE),
    (True, True, True, Action.KEEP),
    (True, True, False, Action.KEEP),
    (True, False, None, Action.KEEP),
]


def _ctx(bucket: int = 3, t_ms: int = 90_000) -> DecisionContext:
    return DecisionContext(
        bucket=bucket, epoch_idx=3, t_ms=t_ms, inc_consec_fail=0,
        inc_since_response_s=1.0, inc_age_s=5.0, inc_srtt_ms=None,
        cand_since_seen_s=None, cand_last_rtt_ms=None, timeout_fraction=0.0,
    )


@pytest.mark.parametrize("inc, present, cand, expected", CASES)
def test_oracle_action_all_combinations(inc, present, cand, expected):
    assert oracle_action(OracleFacts(inc, present, cand)) is expected


@pytest.mark.parametrize("inc, present, cand, expected", CASES)
def test_policy_decide_matches_rule(inc, present, cand, expected):
    pol = OracleHeuristic(lambda b, t: OracleFacts(inc, present, cand))
    assert pol.decide(_ctx()) is expected


def test_never_replace_without_candidate():
    for inc in (True, False):
        assert oracle_action(OracleFacts(inc, False, None)) is not Action.REPLACE


def test_provider_called_with_bucket_and_time():
    calls = []

    def provider(b: int, t: int) -> OracleFacts:
        calls.append((b, t))
        return OracleFacts(True, False, None)

    OracleHeuristic(provider).decide(_ctx(bucket=17, t_ms=123_000))
    assert calls == [(17, 123_000)]


def test_labels():
    pol = OracleHeuristic(lambda b, t: OracleFacts(True, False, None))
    assert pol.name == "oracle_heuristic"
    assert pol.shares_decision_points is True
    assert pol.is_upper_bound is False
    assert "NOT an upper bound" in (OracleHeuristic.__doc__ or "")
    assert "headroom reference" in pol.label


def test_facts_consistency_validation():
    with pytest.raises(ValueError):
        OracleFacts(False, False, True)
    with pytest.raises(ValueError):
        OracleFacts(False, True, None)
