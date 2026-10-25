"""Basics for rl/reward.py lookup_loss / action_cost."""

from __future__ import annotations

import pytest

from churnmaster.rl.reward import action_cost, lookup_loss


@pytest.mark.parametrize("success, latency, expected", [
    (True, 0.0, 0.0),
    (True, 1000.0, 0.15),
    (True, 2000.0, 0.3),
    (True, 5000.0, 0.3),       # latency term capped at w_L
    (False, 2000.0, 1.3),
    (False, 0.0, 1.0),
    (True, -5.0, 0.0),         # clamped
])
def test_lookup_loss_values(success, latency, expected):
    assert lookup_loss(success, latency, 0.3, 2000) == pytest.approx(expected)


@pytest.mark.parametrize("w_L", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("deadline", [1000, 2000])
def test_lookup_loss_range(w_L, deadline):
    for s in (True, False):
        for lat in (0.0, 1.0, deadline / 2, deadline, 10 * deadline):
            assert 0.0 <= lookup_loss(s, lat, w_L, deadline) <= 1.0 + w_L


def test_lookup_loss_extremes():
    w_L = 0.3
    assert lookup_loss(True, 0.0, w_L) == 0.0                         # minimum
    assert lookup_loss(False, float("inf"), w_L) == 1.0 + w_L         # maximum
    assert lookup_loss(False, 1e12, w_L) == 1.0 + w_L


def test_lookup_loss_default_deadline():
    assert lookup_loss(True, 1000.0, 0.3) == pytest.approx(0.15)


def test_action_cost():
    assert action_cost(0, 0, 0.2, 0.3) == 0.0
    assert action_cost(6, 1, 0.2, 0.3) == pytest.approx(1.5)
