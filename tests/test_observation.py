"""Tests for DecisionContext / build_context (Requirement 12.1)."""

from __future__ import annotations

import dataclasses

import pytest

from churnmaster.sim.observation import DecisionContext, build_context
from churnmaster.sim.routing_table import LocalView

OBSERVABLE_FIELDS = [
    "bucket",
    "epoch_idx",
    "t_ms",
    "inc_consec_fail",
    "inc_since_response_s",
    "inc_age_s",
    "inc_srtt_ms",
    "cand_since_seen_s",
    "cand_last_rtt_ms",
    "timeout_fraction",
]


def _view() -> LocalView:
    # Focal 0: nodes 128..255 fall in bucket 7.
    return LocalView(focal_id=0, k=2, cache_size=4)


def test_field_set_is_exactly_the_observable_list():
    assert [f.name for f in dataclasses.fields(DecisionContext)] == OBSERVABLE_FIELDS


def test_scripted_view_gives_exact_values():
    v = _view()
    v.bootstrap([(200, 1_000, 50.0), (201, 2_000, 80.0)])
    v.record_response(201, 10_000, 60.0)  # 201 seen at 10 s; 200 last seen at 1 s
    v.record_failure(200, 20_000)
    v.record_failure(200, 25_000)
    v.cache_put(7, 150, 30_000, 20.0)
    v.cache_put(7, 151, 40_000, 35.0)  # most recent -> candidate
    v.record_query_outcome(41_000, True)
    v.record_query_outcome(42_000, False)
    v.record_query_outcome(43_000, False)
    v.record_query_outcome(44_000, False)

    ctx = build_context(v, 7, epoch_idx=3, t_ms=50_000)
    assert ctx == DecisionContext(
        bucket=7,
        epoch_idx=3,
        t_ms=50_000,
        inc_consec_fail=2,
        inc_since_response_s=49.0,
        inc_age_s=49.0,
        inc_srtt_ms=50.0,
        cand_since_seen_s=10.0,
        cand_last_rtt_ms=35.0,
        timeout_fraction=0.25,
    )


def test_srtt_smoothing_and_age_differ_from_staleness():
    v = _view()
    v.bootstrap([(200, 1_000, 100.0)])
    v.record_response(200, 5_000, 200.0)  # srtt = 0.7*100 + 0.3*200 = 130
    ctx = build_context(v, 7, epoch_idx=0, t_ms=9_000)
    assert ctx.inc_since_response_s == 4.0
    assert ctx.inc_age_s == 8.0
    assert ctx.inc_srtt_ms == pytest.approx(130.0)
    assert ctx.inc_consec_fail == 0


def test_never_contacted_incumbent_uses_insertion_time():
    v = _view()
    v.bootstrap([(200, 3_000, None)])
    ctx = build_context(v, 7, epoch_idx=1, t_ms=63_000)
    assert ctx.inc_since_response_s == 60.0
    assert ctx.inc_age_s == 60.0
    assert ctx.inc_srtt_ms is None


def test_no_candidate_gives_none_fields():
    v = _view()
    v.bootstrap([(200, 0, 10.0)])
    ctx = build_context(v, 7, epoch_idx=0, t_ms=1_000)
    assert ctx.cand_since_seen_s is None
    assert ctx.cand_last_rtt_ms is None
    assert ctx.timeout_fraction == 0.0

    # An expired cache entry (older than 300 s) is not a candidate either.
    v2 = LocalView(focal_id=0, k=1)
    v2.bootstrap([(200, 0, 10.0)])
    v2.cache_put(7, 150, 0, 20.0)
    ctx2 = build_context(v2, 7, epoch_idx=0, t_ms=300_001)
    assert ctx2.cand_since_seen_s is None and ctx2.cand_last_rtt_ms is None


def test_context_is_frozen():
    v = _view()
    v.bootstrap([(200, 0, 10.0)])
    ctx = build_context(v, 7, epoch_idx=0, t_ms=1_000)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.inc_consec_fail = 5  # type: ignore[misc]


def test_empty_bucket_raises():
    v = _view()
    v.bootstrap([(200, 0, 10.0)])
    with pytest.raises(ValueError):
        build_context(v, 3, epoch_idx=0, t_ms=1_000)
