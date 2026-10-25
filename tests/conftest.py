"""Shared pytest configuration: `slow` marker and Hypothesis profiles."""

from __future__ import annotations

import os

from hypothesis import HealthCheck, settings

# Property tests run >= 100 examples by default. Episode-level properties
# override max_examples locally with a documented lower value.
settings.register_profile(
    "default",
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "ci",
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile("dev", max_examples=100, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: long-running tests (episode timing, multi-episode diagnostics)",
    )
