"""Unit tests for sim/workload.py (Requirements 6.1, 6.2)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy import stats

from churnmaster.sim.workload import LookupRequest, generate_workload


def _gen(seed: int = 1, rate: float = 1.0, start: int = 0, end: int = 3_600_000, bits: int = 8):
    return generate_workload(np.random.default_rng(seed), rate, start, end, bits)


def test_rate_within_tolerance_long_horizon():
    # 10^5 s at 1/s: Poisson std ~ 316, so 2% (~6 sigma) is a safe bound.
    wl = _gen(seed=7, end=100_000_000)
    assert abs(len(wl) - 100_000) < 2_000
    # Inter-arrival gaps ~ Exp(mean 1000 ms) (ms flooring adds < 1 ms noise).
    gaps = np.diff([r.t_ms for r in wl])
    assert abs(gaps.mean() - 1000.0) < 20.0


def test_ids_sequential_from_zero():
    wl = _gen(seed=3)
    assert [r.lookup_id for r in wl] == list(range(len(wl)))


def test_timestamps_sorted_integer_in_window():
    start, end = 5_000, 1_805_000
    wl = _gen(seed=11, start=start, end=end)
    ts = [r.t_ms for r in wl]
    assert all(type(t) is int for t in ts)
    assert ts == sorted(ts)
    assert all(start <= t < end for t in ts)


def test_keys_in_range_and_roughly_uniform():
    wl = _gen(seed=5, end=60_000_000)  # ~60k lookups
    keys = np.array([r.key for r in wl])
    assert keys.min() >= 0 and keys.max() <= 255
    counts = np.bincount(keys, minlength=256)
    assert stats.chisquare(counts).pvalue > 0.001


def test_deterministic_given_seed():
    assert _gen(seed=42) == _gen(seed=42)
    assert _gen(seed=42) != _gen(seed=43)


def test_empty_window_and_zero_rate():
    assert _gen(start=1000, end=1000) == []
    assert _gen(rate=0.0) == []


def test_request_is_frozen():
    r = LookupRequest(lookup_id=0, t_ms=1, key=2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.key = 3  # type: ignore[misc]


def test_invalid_args():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        generate_workload(rng, -1.0, 0, 10, 8)
    with pytest.raises(ValueError):
        generate_workload(rng, 1.0, 10, 0, 8)
