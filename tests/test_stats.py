"""Tests for experiments/stats.py paired bootstrap CI (Requirements 22.2, 25.4)."""

import numpy as np
import pandas as pd
import pytest

from churnmaster.experiments.stats import (
    PairedCI,
    paired_bootstrap_ci,
    paired_ci_from_frame,
    paired_differences,
)


def _shifted(shift: float, n: int = 30, seed: int = 123):
    rng = np.random.default_rng(seed)
    b = rng.normal(0.0, 1.0, n)
    a = b + shift + rng.normal(0.0, 0.1, n)
    return a, b


def test_known_shift_ci_contains_shift_and_excludes_zero():
    a, b = _shifted(0.5)
    ci = paired_bootstrap_ci(a, b)
    assert ci.n == 30 and ci.n_resamples == 10_000 and ci.level == 0.95
    assert ci.lo <= 0.5 <= ci.hi
    assert ci.lo <= ci.mean <= ci.hi
    assert ci.excludes_zero


def test_zero_shift_ci_contains_zero():
    a, b = _shifted(0.0)
    ci = paired_bootstrap_ci(a, b)
    assert ci.lo <= 0.0 <= ci.hi
    assert not ci.excludes_zero


def test_negative_shift_excludes_zero():
    a, b = _shifted(-0.5)
    assert paired_bootstrap_ci(a, b).hi < 0


def test_deterministic_for_seed_and_differs_across_seeds():
    a, b = _shifted(0.5)
    assert paired_bootstrap_ci(a, b, seed=7) == paired_bootstrap_ci(a, b, seed=7)
    c1, c2 = paired_bootstrap_ci(a, b, seed=7), paired_bootstrap_ci(a, b, seed=8)
    assert (c1.lo, c1.hi) != (c2.lo, c2.hi)
    assert c1.mean == c2.mean


def test_level_affects_width():
    a, b = _shifted(0.5)
    w90 = paired_bootstrap_ci(a, b, level=0.90)
    w99 = paired_bootstrap_ci(a, b, level=0.99)
    assert (w99.hi - w99.lo) > (w90.hi - w90.lo)


def test_single_pair_degenerate_interval():
    ci = paired_bootstrap_ci([3.0], [1.0])
    assert ci.n == 1 and ci.mean == ci.lo == ci.hi == 2.0
    assert ci.excludes_zero


def test_to_dict():
    d = paired_bootstrap_ci([1.0, 2.0], [0.0, 0.0]).to_dict()
    assert set(d) == {"mean", "lo", "hi", "n", "level", "n_resamples", "excludes_zero"}
    assert isinstance(paired_bootstrap_ci([1.0], [0.0]), PairedCI)


def test_paired_differences_values():
    np.testing.assert_array_equal(paired_differences([3, 5], [1, 1]), [2.0, 4.0])


@pytest.mark.parametrize(
    "a,b,msg",
    [
        ([1.0, 2.0], [1.0], "equal length"),
        ([], [], "at least one"),
        ([1.0, np.nan], [0.0, 0.0], "NaN"),
        ([1.0, 2.0], [0.0, np.inf], "NaN"),
    ],
)
def test_validation_errors(a, b, msg):
    with pytest.raises(ValueError, match=msg):
        paired_differences(a, b)
    with pytest.raises(ValueError, match=msg):
        paired_bootstrap_ci(a, b)


def test_invalid_level():
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1.0, 2.0], [0.0, 0.0], level=1.0)


def _frame(n: int = 10, shift: float = 0.5, seed: int = 1) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    a, b = _shifted(shift, n=n, seed=seed)
    rows = []
    for i in range(n):
        key = {"scenario": "high_churn", "split": "eval", "run": 0, "episode": i}
        rows.append({"policy": "A", **key, "J": a[i]})
        rows.append({"policy": "B", **key, "J": b[i]})
    return pd.DataFrame(rows), a, b


def test_frame_pairing_matches_arrays_and_ignores_row_order():
    df, a, b = _frame()
    ref = paired_bootstrap_ci(a, b, seed=3)
    shuffled = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    got = paired_ci_from_frame(shuffled, "J", "A", "B", seed=3)
    assert got.mean == pytest.approx(ref.mean)
    assert got.lo == pytest.approx(ref.lo) and got.hi == pytest.approx(ref.hi)
    assert got.n == 10


def test_frame_missing_keys_raises():
    df, _, _ = _frame()
    drop = df[(df["policy"] == "B") & (df["episode"] == 3)].index
    with pytest.raises(ValueError, match="same keys"):
        paired_ci_from_frame(df.drop(drop), "J", "A", "B")


def test_frame_unknown_policy_or_metric_raises():
    df, _, _ = _frame()
    with pytest.raises(ValueError, match="no rows"):
        paired_ci_from_frame(df, "J", "A", "Z")
    with pytest.raises(ValueError, match="missing columns"):
        paired_ci_from_frame(df, "nope", "A", "B")
