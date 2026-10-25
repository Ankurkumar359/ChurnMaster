"""Unit tests for sim/routing_table.py (Requirements 9.1-9.8)."""

from __future__ import annotations

import ast
import pathlib

import pytest

from churnmaster.sim.idspace import bucket_index
from churnmaster.sim.routing_table import (
    BootstrapEntry,
    CacheEntry,
    LocalView,
    OfferOutcome,
)

# With focal_id = 0, bucket b holds node IDs in [2**b, 2**(b+1)); bucket 7 = 128..255.
B = 7


def full_view(t: int = 0) -> LocalView:
    v = LocalView(focal_id=0)
    for n in (128, 129, 130, 131):
        assert v.offer(n, t, 50.0) is OfferOutcome.INSERTED
    return v


def test_structure_and_bucket_placement():
    v = LocalView(focal_id=0x5A)
    v.bootstrap([(0x5B, 0, None), BootstrapEntry(0xDA, 0), (0x1A, 0, 20.0)])
    for e in v.all_entries():
        assert e in v.bucket(bucket_index(0x5A, e.node))
    assert 0x5B in v and 0xDA in v and 0x1A in v and 0x00 not in v
    assert len(v.all_entries()) == 3


def test_bootstrap_without_rtt_has_empty_stats():
    v = LocalView(focal_id=0)
    v.bootstrap([(128, 0, None), (129, 0, 30.0)])
    a, b = v.entry(128), v.entry(129)
    assert (a.inserted_ms, a.last_response_ms, a.srtt_ms, a.consec_fail) == (0, None, None, 0)
    assert (b.last_response_ms, b.srtt_ms) == (0, 30.0)


def test_bootstrap_rejects_overfull_and_duplicates():
    v = LocalView(focal_id=0)
    with pytest.raises(ValueError):
        v.bootstrap([(128, 0, None), (128, 0, None)])
    v = LocalView(focal_id=0)
    with pytest.raises(ValueError):
        v.bootstrap([(n, 0, None) for n in range(128, 133)])


def test_focal_node_rejected():
    v = LocalView(focal_id=9)
    with pytest.raises(ValueError):
        v.offer(9, 0, 10.0)


def test_record_response_ewma_and_failures():
    v = LocalView(focal_id=0)
    v.bootstrap([(128, 0, None)])
    v.record_failure(128, 10)
    v.record_failure(128, 20)
    assert v.entry(128).consec_fail == 2
    v.record_response(128, 30, 100.0)
    e = v.entry(128)
    assert (e.consec_fail, e.last_response_ms, e.srtt_ms) == (0, 30, 100.0)
    v.record_response(128, 40, 200.0)
    assert v.entry(128).srtt_ms == pytest.approx(0.7 * 100.0 + 0.3 * 200.0)
    with pytest.raises(KeyError):
        v.record_response(200, 50, 10.0)


def test_offer_outcomes():
    v = full_view()
    v.record_failure(128, 5)
    assert v.offer(128, 10, 999.0) is OfferOutcome.IN_TABLE
    assert v.entry(128).consec_fail == 1  # offer never touches stats
    assert v.entry(128).srtt_ms == 50.0
    assert v.offer(140, 10, 40.0) is OfferOutcome.CACHED
    assert [c.node for c in v.cache(B)] == [140]
    assert 140 not in v
    # A different (non-full) bucket inserts directly.
    assert v.offer(64, 10, 40.0) is OfferOutcome.INSERTED
    e = v.entry(64)
    assert (e.inserted_ms, e.last_response_ms, e.srtt_ms) == (10, 10, 40.0)


def test_cache_dedupe_and_size_limit():
    v = full_view()
    v.cache_put(B, 140, 10, 40.0)
    v.cache_put(B, 140, 20, 60.0)
    assert v.cache(B) == [CacheEntry(140, 20, 60.0)]
    for i, n in enumerate((141, 142, 143)):
        v.cache_put(B, n, 30 + i, 10.0)
    v.cache_put(B, 144, 100, 10.0)  # evicts oldest last_seen (140 @ 20)
    assert sorted(c.node for c in v.cache(B)) == [141, 142, 143, 144]
    with pytest.raises(ValueError):
        v.cache_put(B, 128, 0, 1.0)  # already in table
    with pytest.raises(ValueError):
        v.cache_put(3, 140, 0, 1.0)  # wrong bucket


def test_cache_ttl_expiry_boundary():
    v = full_view()
    v.cache_put(B, 140, 1_000, 10.0)
    v.cache_put(B, 141, 2_000, 10.0)
    v.cache_expire(301_000)  # 140 is exactly 300 s old: still valid
    assert sorted(c.node for c in v.cache(B)) == [140, 141]
    v.cache_expire(301_001)
    assert [c.node for c in v.cache(B)] == [141]
    assert v.candidate(B, 302_000).node == 141
    assert v.candidate(B, 302_001) is None  # expired entries are never candidates


def test_candidate_tie_break():
    v = full_view()
    assert v.candidate(B, 0) is None
    v.cache_put(B, 140, 10, 80.0)
    v.cache_put(B, 141, 20, 90.0)
    v.cache_put(B, 142, 20, 30.0)
    assert v.candidate(B, 20).node == 142  # most recent, tie lower RTT
    v.cache_put(B, 143, 21, 500.0)
    assert v.candidate(B, 21).node == 143


def test_incumbent_lrs_and_tie_break():
    v = LocalView(focal_id=0)
    v.bootstrap([(128, 0, None), (129, 0, None), (130, 5, None)])
    assert v.incumbent(B).node == 128  # all never contacted at 0; earlier insertion wins
    v.record_response(128, 10, 1.0)
    assert v.incumbent(B).node == 129
    v.record_response(129, 3, 1.0)
    # 129 -> 3, 130 -> inserted 5 (never contacted), 128 -> 10
    assert v.incumbent(B).node == 129
    v.record_response(129, 20, 1.0)
    assert v.incumbent(B).node == 130
    # Equal effective time, earlier insertion breaks the tie.
    v2 = LocalView(focal_id=0)
    v2.bootstrap([(128, 0, None), (129, 7, None)])
    v2.record_response(128, 7, 1.0)
    assert v2.incumbent(B).node == 128
    with pytest.raises(ValueError):
        v2.incumbent(3)


def test_autofill_from_cache_most_recent_first():
    v = full_view()
    v.cache_put(B, 140, 10, 40.0)
    v.cache_put(B, 141, 30, 70.0)
    v.cache_put(B, 142, 30, 20.0)
    v.evict(128)
    v.evict(129)
    assert len(v.bucket(B)) == 2
    assert v.autofill(B, 100) == 2
    assert {e.node for e in v.bucket(B)} == {130, 131, 142, 141}
    assert [c.node for c in v.cache(B)] == [140]
    e = v.entry(142)
    assert (e.inserted_ms, e.last_response_ms, e.srtt_ms, e.consec_fail) == (100, 30, 20.0, 0)
    assert v.autofill(B, 100) == 0  # full bucket: nothing moves


def test_autofill_skips_expired_and_partial_fill():
    v = full_view()
    v.cache_put(B, 140, 0, 40.0)
    v.cache_put(B, 141, 200_000, 40.0)
    for n in (128, 129, 130):
        v.evict(n)
    assert v.autofill(B, 400_000) == 1  # 140 expired and dropped
    assert {e.node for e in v.bucket(B)} == {131, 141}
    assert v.cache(B) == []


def test_offer_with_empty_slot_moves_node_out_of_cache():
    v = full_view()
    v.offer(140, 10, 40.0)
    v.evict(128)
    assert v.offer(140, 20, 35.0) is OfferOutcome.INSERTED
    assert 140 in v and not v.in_cache(140)


def test_insert_evict_cache_drop():
    v = full_view()
    v.cache_put(B, 140, 10, 40.0)
    with pytest.raises(ValueError):
        v.insert(B, 140, 20, 40.0)  # bucket full
    old = v.evict(128)
    assert old.node == 128 and 128 not in v
    with pytest.raises(KeyError):
        v.evict(128)
    e = v.insert(B, 140, 20, 40.0)
    assert (e.inserted_ms, e.last_response_ms, e.srtt_ms) == (20, 20, 40.0)
    assert not v.in_cache(140)
    with pytest.raises(ValueError):
        v.insert(B, 129, 20, 1.0)  # already present
    with pytest.raises(ValueError):
        v.insert(3, 200, 20, 1.0)  # wrong bucket
    v.cache_put(B, 141, 30, 1.0)
    assert v.cache_drop(B, 141) is True
    assert v.cache_drop(B, 141) is False


def test_closest_entries_by_xor():
    v = LocalView(focal_id=0)
    v.bootstrap([(n, 0, None) for n in (1, 2, 4, 8, 16, 200)])
    assert [e.node for e in v.closest_entries(5, 3)] == [4, 1, 2]
    assert [e.node for e in v.closest_entries(201, 1)] == [200]
    assert v.closest_entries(5, 0) == []
    assert len(v.closest_entries(5, 100)) == 6


def test_timeout_fraction_sliding_window():
    v = LocalView(focal_id=0)
    assert v.timeout_fraction(0) == 0.0
    v.record_query_outcome(0, True)
    v.record_query_outcome(10_000, False)
    v.record_query_outcome(60_000, False)
    v.record_query_outcome(100_000, True)
    assert v.timeout_fraction(100_000) == pytest.approx(0.5)
    assert v.timeout_fraction(119_999) == pytest.approx(0.5)
    assert v.timeout_fraction(120_000) == pytest.approx(1 / 3)  # t=0 left the window
    assert v.timeout_fraction(130_000) == pytest.approx(0.5)
    assert v.timeout_fraction(220_000) == 0.0


def test_routing_table_imports_are_observable_only():
    src = pathlib.Path(__file__).parents[1] / "src" / "churnmaster" / "sim" / "routing_table.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            mods.add(node.module or "")
    for m in mods:
        if m.startswith("churnmaster"):
            assert m == "churnmaster.sim.idspace", m
