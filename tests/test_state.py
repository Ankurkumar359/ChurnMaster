"""Tests for rl/state.py: bins, cand_faster, encode/decode, reachability, masking (Req 13)."""

from __future__ import annotations

import ast
import pathlib
import sys
from dataclasses import replace

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from churnmaster.actions import Action
from churnmaster.rl import state as S
from churnmaster.sim.observation import DecisionContext

BASE = DecisionContext(
    bucket=3, epoch_idx=0, t_ms=0, inc_consec_fail=0, inc_since_response_s=0.0,
    inc_age_s=0.0, inc_srtt_ms=None, cand_since_seen_s=None, cand_last_rtt_ms=None,
    timeout_fraction=0.0,
)


def d(**kw) -> S.Features:
    return S.discretize(replace(BASE, **kw))


# --------------------------------------------------------------------------- bins


@pytest.mark.parametrize("bucket, tier", [(7, S.TIER_TOP), (6, S.TIER_SECOND), (5, S.TIER_OTHER), (0, S.TIER_OTHER)])
def test_tier(bucket, tier):
    assert d(bucket=bucket).tier == tier


def test_tier_generalizes_to_id_bits():
    ctx = replace(BASE, bucket=15)
    assert S.discretize(ctx, id_bits=16).tier == S.TIER_TOP
    assert S.discretize(replace(BASE, bucket=14), id_bits=16).tier == S.TIER_SECOND
    assert S.discretize(replace(BASE, bucket=7), id_bits=16).tier == S.TIER_OTHER


@pytest.mark.parametrize("n, b", [(0, 0), (1, 1), (2, 2), (3, 2), (50, 2)])
def test_inc_fail(n, b):
    assert d(inc_consec_fail=n).inc_fail == b


@pytest.mark.parametrize("x, b", [(0.0, 0), (59.999, 0), (60.0, 1), (60.001, 1), (1e6, 1)])
def test_inc_stale(x, b):
    assert d(inc_since_response_s=x).inc_stale == b


@pytest.mark.parametrize(
    "x, b",
    [(0.0, S.AGE_YOUNG), (119.999, S.AGE_YOUNG), (120.0, S.AGE_MID), (1199.999, S.AGE_MID),
     (1200.0, S.AGE_OLD), (1e6, S.AGE_OLD)],
)
def test_inc_age(x, b):
    assert d(inc_age_s=x).inc_age == b


@pytest.mark.parametrize(
    "x, b",
    [(None, S.CAND_NONE), (0.0, S.CAND_FRESH), (29.999, S.CAND_FRESH), (30.0, S.CAND_STALE),
     (300.0, S.CAND_STALE)],
)
def test_cand(x, b):
    rtt = None if x is None else 50.0
    assert d(cand_since_seen_s=x, cand_last_rtt_ms=rtt).cand == b


@pytest.mark.parametrize("x, b", [(0.0, 0), (0.0999, 0), (0.10, 1), (0.5, 1), (1.0, 1)])
def test_churn(x, b):
    assert d(timeout_fraction=x).churn == b


# --------------------------------------------------------------------------- cand_faster


@pytest.mark.parametrize(
    "seen, cand_rtt, inc_srtt, faster",
    [
        (5.0, 69.0, 100.0, 1),     # strictly below 0.7x
        (5.0, 70.0, 100.0, 0),     # exactly 0.7x is not faster
        (5.0, 80.0, 100.0, 0),
        (5.0, 10.0, None, 0),      # unknown incumbent SRTT
        (None, None, 100.0, 0),    # no candidate
        (None, None, None, 0),
        (100.0, 10.0, 100.0, 1),   # stale candidate can still be faster
    ],
)
def test_cand_faster(seen, cand_rtt, inc_srtt, faster):
    f = d(cand_since_seen_s=seen, cand_last_rtt_ms=cand_rtt, inc_srtt_ms=inc_srtt)
    assert f.cand_faster == faster


# --------------------------------------------------------------------------- encoding


def test_counts():
    assert S.N_STATES == 648 == int(np.prod(S.RADICES))
    assert S.N_ACTIONS == 3
    reach = S.reachable_states()
    assert len(reach) == S.N_REACHABLE == 540
    assert len(set(reach)) == 540 and all(0 <= s < 648 for s in reach)


def test_encode_decode_roundtrip_all_states():
    seen = set()
    for s in range(S.N_STATES):
        f = S.decode(s)
        assert S.encode(f) == s
        seen.add(f)
    assert len(seen) == S.N_STATES


def test_encode_is_mixed_radix():
    assert S.encode(S.Features(0, 0, 0, 0, 0, 0, 0)) == 0
    assert S.encode(S.Features(0, 0, 0, 0, 0, 0, 1)) == 1
    assert S.encode(S.Features(2, 2, 1, 2, 2, 1, 1)) == 647


@pytest.mark.parametrize(
    "f",
    [S.Features(3, 0, 0, 0, 0, 0, 0), S.Features(0, 0, 2, 0, 0, 0, 0),
     S.Features(0, 0, 0, 0, 0, 0, -1), S.Features(0, 0, 0, 0, 0, 0, 0.5)],
)
def test_encode_rejects_out_of_range(f):
    with pytest.raises(ValueError):
        S.encode(f)


@pytest.mark.parametrize("s", [-1, 648, 10_000])
def test_decode_rejects_out_of_range(s):
    with pytest.raises(ValueError):
        S.decode(s)
    with pytest.raises(ValueError):
        S.allowed_actions(s)


# --------------------------------------------------------------------------- masking


def test_allowed_actions():
    for s in range(S.N_STATES):
        acts = S.allowed_actions(s)
        assert acts[:2] == (Action.KEEP, Action.PROBE)
        assert (Action.REPLACE in acts) == (S.decode(s).cand != S.CAND_NONE)


def test_allowed_mask_matches_allowed_actions():
    m = S.allowed_mask()
    assert m.shape == (648, 3) and m.dtype == bool
    assert m[:, Action.KEEP].all() and m[:, Action.PROBE].all()
    for s in range(S.N_STATES):
        assert set(np.flatnonzero(m[s])) == {int(a) for a in S.allowed_actions(s)}
    # REPLACE allowed in exactly the states with a candidate: 648 * 2/3
    assert m[:, Action.REPLACE].sum() == 432


def test_maintenance_reexports_single_action():
    from churnmaster.sim import maintenance
    assert maintenance.Action is Action


def test_actions_module_imports_only_stdlib():
    path = pathlib.Path(__file__).resolve().parents[1] / "src" / "churnmaster" / "actions.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            mods.add((node.module or "").split(".")[0])
    assert mods <= set(sys.stdlib_module_names) | {"__future__"}, mods


# --------------------------------------------------------------------------- property

_opt_pos = st.one_of(st.none(), st.floats(min_value=0.0, max_value=1e4, allow_nan=False))


@st.composite
def contexts(draw):
    has_cand = draw(st.booleans())
    return DecisionContext(
        bucket=draw(st.integers(0, 7)),
        epoch_idx=0,
        t_ms=0,
        inc_consec_fail=draw(st.integers(0, 10)),
        inc_since_response_s=draw(st.floats(0.0, 1e5, allow_nan=False)),
        inc_age_s=draw(st.floats(0.0, 1e5, allow_nan=False)),
        inc_srtt_ms=draw(_opt_pos),
        cand_since_seen_s=draw(st.floats(0.0, 300.0)) if has_cand else None,
        cand_last_rtt_ms=draw(st.floats(0.0, 1e4)) if has_cand else None,
        timeout_fraction=draw(st.floats(0.0, 1.0)),
    )


@given(contexts())
def test_discretize_always_reachable(ctx):
    """**Validates: Requirements 13.3, 13.4**"""
    s = S.encode(S.discretize(ctx))
    assert 0 <= s < S.N_STATES
    assert S.is_reachable(s)
    assert S.decode(s) == S.discretize(ctx)
