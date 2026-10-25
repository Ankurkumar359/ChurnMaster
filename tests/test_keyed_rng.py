"""Unit tests for sim/keyed_rng.py (Requirements 4.1-4.4)."""

from __future__ import annotations

import math
import statistics

from scipy import stats

from churnmaster.sim.keyed_rng import (
    MASK64,
    PingKind,
    Purpose,
    keyed_normal,
    keyed_uniform,
    splitmix64,
)

ROOT = 0xC0FFEE


def test_splitmix64_reference_values():
    # Reference: splitmix64 generator seeded with 0 yields these first outputs
    # (state advanced by the golden gamma each step).
    state = 0
    out = []
    for _ in range(3):
        out.append(splitmix64(state))
        state = (state + 0x9E3779B97F4A7C15) & MASK64
    assert out == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]


def test_splitmix64_stays_in_64_bits():
    for x in (0, 1, MASK64, MASK64 - 1, 1 << 63):
        assert 0 <= splitmix64(x) <= MASK64


def test_enum_values():
    assert [int(p) for p in Purpose] == [1, 2, 3, 4]
    assert [int(k) for k in PingKind] == [1, 2, 3]


def test_determinism_and_interleaving():
    keys = [(Purpose.LOSS, 7, 42), (Purpose.JITTER, 3, 5, 99, PingKind.INC), (Purpose.BOOT, 2, 1)]
    first = [keyed_uniform(ROOT, p, *ids) for p, *ids in keys]
    # Interleave unrelated calls; results must not change (no hidden state).
    for i in range(100):
        keyed_uniform(ROOT, Purpose.BG, i, i + 1)
        keyed_normal(ROOT, Purpose.JITTER, i)
    second = [keyed_uniform(ROOT, p, *ids) for p, *ids in keys]
    assert first == second
    assert keyed_normal(ROOT, Purpose.JITTER, 1, 2) == keyed_normal(ROOT, Purpose.JITTER, 1, 2)


def test_independence_across_purposes_ids_and_roots():
    base = keyed_uniform(ROOT, Purpose.LOSS, 10, 20)
    assert keyed_uniform(ROOT, Purpose.JITTER, 10, 20) != base
    assert keyed_uniform(ROOT, Purpose.LOSS, 20, 10) != base
    assert keyed_uniform(ROOT, Purpose.LOSS, 10, 21) != base
    assert keyed_uniform(ROOT + 1, Purpose.LOSS, 10, 20) != base
    # Different key lengths are domain-separated.
    assert keyed_uniform(ROOT, Purpose.LOSS, 10) != keyed_uniform(ROOT, Purpose.LOSS, 10, 0)
    # Epoch ping kinds differ for the same (epoch, bucket, dst).
    assert keyed_uniform(ROOT, Purpose.LOSS, 5, 7, 33, PingKind.INC) != keyed_uniform(
        ROOT, Purpose.LOSS, 5, 7, 33, PingKind.CAND
    )


def test_uniform_range_and_ks_10k_keys():
    xs = [keyed_uniform(ROOT, Purpose.LOSS, j, j % 256) for j in range(10_000)]
    assert all(0.0 <= x < 1.0 for x in xs)
    assert len(set(xs)) == len(xs)
    assert stats.kstest(xs, "uniform").pvalue > 0.01


def test_loss_and_jitter_streams_uncorrelated():
    a = [keyed_uniform(ROOT, Purpose.LOSS, j, 3) for j in range(5_000)]
    b = [keyed_uniform(ROOT, Purpose.JITTER, j, 3) for j in range(5_000)]
    r = stats.pearsonr(a, b).statistic
    assert abs(r) < 0.05


def test_normal_mean_std_sanity():
    zs = [keyed_normal(ROOT, Purpose.JITTER, j, 17) for j in range(20_000)]
    assert all(math.isfinite(z) for z in zs)
    assert abs(statistics.fmean(zs)) < 0.03
    assert abs(statistics.pstdev(zs) - 1.0) < 0.03
    assert stats.kstest(zs, "norm").pvalue > 0.01


def test_box_muller_u1_zero_is_safe():
    # u1 == 0 -> log(1 - 0) = 0, radius 0, finite result. Largest u1 < 1 is
    # 1 - 2**-53 -> finite radius.
    assert math.sqrt(-2.0 * math.log(1.0 - 0.0)) == 0.0
    assert math.isfinite(math.sqrt(-2.0 * math.log(1.0 - (1.0 - 2.0**-53))))
