"""Paired statistics for policy comparisons (Requirements 22.2, 25.4).

All comparisons are *paired by seed*: for each evaluation key (scenario,
split, run, episode) the two policies saw the same churn trace and keyed
randomness, so we analyse the per-key differences ``a_i - b_i``.

Bootstrap randomness uses ``np.random.default_rng(seed)`` with its own
explicit seed. This generator is independent of every experiment stream
(the simulator uses keyed hashing and per-purpose seeded streams), so
computing a CI never perturbs, and is never perturbed by, simulation
outcomes. The same inputs and seed always give the same CI.

Later additions (Wilcoxon signed-rank, Holm correction) live in this module
too and reuse :func:`paired_differences`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

DEFAULT_KEYS: tuple[str, ...] = ("scenario", "split", "run", "episode")


def paired_differences(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return ``a_i - b_i`` as a float array.

    Raises ``ValueError`` if lengths differ, inputs are empty / not 1-D, or any
    value is non-finite. NaN handling is the caller's decision (e.g. a metric
    that is undefined for an episode) and is never silently dropped here.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.ndim != 1 or b_arr.ndim != 1:
        raise ValueError("paired_differences expects 1-D sequences")
    if a_arr.shape[0] != b_arr.shape[0]:
        raise ValueError(f"paired samples must have equal length (got {a_arr.shape[0]} and {b_arr.shape[0]})")
    if a_arr.shape[0] < 1:
        raise ValueError("paired samples must contain at least one pair")
    if not (np.all(np.isfinite(a_arr)) and np.all(np.isfinite(b_arr))):
        raise ValueError("paired samples contain NaN or infinite values; handle missing metrics before calling")
    return a_arr - b_arr


@dataclass(frozen=True)
class PairedCI:
    """Percentile bootstrap CI for the mean paired difference (a − b)."""

    mean: float
    lo: float
    hi: float
    n: int
    level: float
    n_resamples: int

    @property
    def excludes_zero(self) -> bool:
        return self.lo > 0.0 or self.hi < 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["excludes_zero"] = self.excludes_zero
        return d


def paired_bootstrap_ci(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    n_resamples: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> PairedCI:
    """Percentile bootstrap CI of the mean of ``a - b``.

    Resamples pair indices with replacement (vectorised, ``n_resamples × n``)
    using ``np.random.default_rng(seed)``, a generator independent of all
    experiment streams. With ``n == 1`` every resample equals the single
    difference, so ``lo == hi == mean`` (a degenerate, zero-width interval).
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    d = paired_differences(a, b)
    n = int(d.shape[0])
    mean = float(d.mean())
    if n == 1:
        return PairedCI(mean=mean, lo=mean, hi=mean, n=1, level=level, n_resamples=n_resamples)
    rng = np.random.default_rng(seed)
    idx = rng.integers(n, size=(n_resamples, n))
    boot_means = d[idx].mean(axis=1)
    alpha = 1.0 - level
    lo, hi = np.quantile(boot_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return PairedCI(mean=mean, lo=float(lo), hi=float(hi), n=n, level=level, n_resamples=n_resamples)


@dataclass(frozen=True)
class TwoSampleCI:
    """Percentile bootstrap CI for ``mean(a) − mean(b)`` of two independent samples."""

    mean: float
    lo: float
    hi: float
    n_a: int
    n_b: int
    level: float
    n_resamples: int

    @property
    def excludes_zero(self) -> bool:
        return self.lo > 0.0 or self.hi < 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["excludes_zero"] = self.excludes_zero
        return d


def _finite_1d(x: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1 or arr.shape[0] < 1:
        raise ValueError(f"sample {name} must be a non-empty 1-D sequence")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"sample {name} contains NaN or infinite values")
    return arr


def _boot_means(x: np.ndarray, n_resamples: int, rng: np.random.Generator,
                max_cells: int = 4_000_000) -> np.ndarray:
    """Means of ``n_resamples`` resamples of ``x`` (with replacement), chunked for memory."""
    n = x.shape[0]
    out = np.empty(n_resamples, dtype=float)
    step = max(1, max_cells // n)
    for start in range(0, n_resamples, step):
        stop = min(n_resamples, start + step)
        idx = rng.integers(n, size=(stop - start, n))
        out[start:stop] = x[idx].mean(axis=1)
    return out


def two_sample_bootstrap_ci(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    n_resamples: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> TwoSampleCI:
    """Percentile bootstrap CI of ``mean(a) − mean(b)`` for independent (unpaired) samples.

    Each sample is resampled with replacement within itself (stratified
    bootstrap), using ``np.random.default_rng(seed)``: ``a``'s resamples are
    drawn first, then ``b``'s. Resampling is chunked so large samples do not
    allocate ``n_resamples × n`` indices at once. Samples of size 1 contribute
    zero variance.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    xa, xb = _finite_1d(a, "a"), _finite_1d(b, "b")
    mean = float(xa.mean() - xb.mean())
    rng = np.random.default_rng(seed)
    diffs = _boot_means(xa, n_resamples, rng) - _boot_means(xb, n_resamples, rng)
    alpha = 1.0 - level
    lo, hi = np.quantile(diffs, [alpha / 2.0, 1.0 - alpha / 2.0])
    return TwoSampleCI(mean=mean, lo=float(lo), hi=float(hi), n_a=int(xa.shape[0]),
                       n_b=int(xb.shape[0]), level=level, n_resamples=n_resamples)


def paired_frame(
    df: pd.DataFrame,
    metric: str,
    policy_a: str,
    policy_b: str,
    keys: Sequence[str] = DEFAULT_KEYS,
) -> pd.DataFrame:
    """Align ``metric`` for two policies by ``keys``.

    Returns a DataFrame indexed by ``keys`` (sorted) with columns ``a`` and
    ``b``. Raises ``ValueError`` on missing columns, duplicate keys within a
    policy, or if the two policies do not cover exactly the same key set.
    """
    keys = list(keys)
    missing = [c for c in ["policy", metric, *keys] if c not in df.columns]
    if missing:
        raise ValueError(f"results frame is missing columns: {missing}")

    def _side(policy: str) -> pd.Series:
        sub = df.loc[df["policy"] == policy, [*keys, metric]]
        if sub.empty:
            raise ValueError(f"no rows for policy {policy!r}")
        if sub.duplicated(subset=keys).any():
            raise ValueError(f"duplicate keys {keys} for policy {policy!r}")
        return sub.set_index(keys)[metric]

    sa, sb = _side(policy_a), _side(policy_b)
    ka, kb = set(sa.index), set(sb.index)
    if ka != kb:
        only_a = sorted(ka - kb, key=repr)[:5]
        only_b = sorted(kb - ka, key=repr)[:5]
        raise ValueError(
            f"policies {policy_a!r} and {policy_b!r} do not cover the same keys; "
            f"only in {policy_a!r}: {only_a}, only in {policy_b!r}: {only_b}"
        )
    out = pd.DataFrame({"a": sa, "b": sb.reindex(sa.index)}).sort_index()
    return out


def paired_ci_from_frame(
    df: pd.DataFrame,
    metric: str,
    policy_a: str,
    policy_b: str,
    keys: Sequence[str] = DEFAULT_KEYS,
    **kw: Any,
) -> PairedCI:
    """Paired bootstrap CI of ``metric[policy_a] - metric[policy_b]`` from a results table."""
    pf = paired_frame(df, metric, policy_a, policy_b, keys)
    return paired_bootstrap_ci(pf["a"].to_numpy(), pf["b"].to_numpy(), **kw)
