"""Unit tests for sim/idspace.py (Requirements 3.1-3.5)."""

from __future__ import annotations

import numpy as np
import pytest

from churnmaster.sim.idspace import bucket_index, closest, draw_ids, xor_dist


def test_xor_dist_examples():
    assert xor_dist(0b1010, 0b0110) == 0b1100
    assert xor_dist(0, 255) == 255
    assert xor_dist(37, 37) == 0
    assert xor_dist(5, 9) == xor_dist(9, 5)


@pytest.mark.parametrize(
    "f, x, expected",
    [
        (0, 1, 0),
        (0, 2, 1),
        (0, 3, 1),
        (0, 4, 2),
        (0, 127, 6),
        (0, 128, 7),
        (0, 255, 7),
        (0b10000000, 0b00000000, 7),
        (0b10101010, 0b10101011, 0),
        (0b10101010, 0b10101000, 1),
        (0b10101010, 0b11101010, 6),
    ],
)
def test_bucket_boundaries(f, x, expected):
    assert bucket_index(f, x) == expected


def test_bucket_range_exhaustive_for_one_focal():
    f = 0x5A
    for x in range(256):
        if x == f:
            continue
        b = bucket_index(f, x)
        assert 0 <= b <= 7
        assert 2**b <= (f ^ x) < 2 ** (b + 1)


def test_key_equal_to_focal_is_bucket_zero():
    for f in (0, 17, 128, 255):
        assert bucket_index(f, f) == 0


def test_closest_orders_by_xor():
    ids = [0b0000, 0b0001, 0b0100, 0b1000, 0b1111]
    assert closest(0b0101, ids, 3) == [0b0100, 0b0001, 0b0000]
    assert closest(0b0101, ids, 10) == sorted(ids, key=lambda i: i ^ 0b0101)
    assert closest(0b0101, ids, 0) == []
    assert closest(0b0101, iter(ids), 1) == [0b0100]


def test_draw_ids_unique_and_in_range():
    ids = draw_ids(np.random.default_rng(1))
    assert len(ids) == 128
    assert len(set(ids)) == 128
    assert all(isinstance(i, int) and 0 <= i <= 255 for i in ids)


def test_draw_ids_deterministic():
    a = draw_ids(np.random.default_rng(42))
    b = draw_ids(np.random.default_rng(42))
    assert a == b


def test_draw_ids_rejects_too_many():
    with pytest.raises(ValueError):
        draw_ids(np.random.default_rng(0), n=257)
