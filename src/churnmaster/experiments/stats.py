"""Paired statistics for policy comparisons (Requirements 22.2, 25.4).

All comparisons are *paired by seed*: for each evaluation key (scenario,
split, run, episode) the two policies saw the same churn trace and keyed
randomness, so we analyse the per-key differences ``a_i - b_i``.

Bootstrap randomness uses ``np.random.default_rng(seed)`` with its own
explicit seed. This generator is independent of every experiment stream
(the simulator uses keyed hashing and per-purpose seeded streams), so
computing a CI never perturbs, and is never perturbed by, simulation
outcomes. The same inputs and seed always give the same CI.

Task 9.4 adds (Requirements 25.4–25.6): Wilcoxon signed-rank tests, Holm
correction over the 9 primary tests, RL averaging over training runs,
per-run results and the automatic H1–H5 hypothesis table.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats as _sps

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


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank and Holm correction (Requirements 25.4, 25.5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WilcoxonResult:
    """Wilcoxon signed-rank test on paired differences ``a − b``."""

    statistic: float
    p: float
    n: int
    n_nonzero: int
    alternative: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def wilcoxon_paired(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    alternative: str = "two-sided",
    zero_method: str = "wilcox",
) -> WilcoxonResult:
    """Wilcoxon signed-rank test of ``a − b`` (``scipy.stats.wilcoxon``, ``method="auto"``).

    ``method="auto"`` uses the exact null distribution for small samples
    without zeros/ties and the normal approximation otherwise.

    Degenerate cases (documented, not errors):
    * all differences zero (``n_nonzero == 0``): no evidence of any shift, so
      ``statistic = 0.0`` and ``p = 1.0``;
    * a non-finite p from scipy (e.g. pathological ties): mapped to ``p = 1.0``.
    """
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(f"alternative must be 'two-sided', 'greater' or 'less', got {alternative!r}")
    if zero_method not in ("wilcox", "pratt", "zsplit"):
        raise ValueError(f"zero_method must be 'wilcox', 'pratt' or 'zsplit', got {zero_method!r}")
    d = paired_differences(a, b)
    n = int(d.shape[0])
    n_nonzero = int(np.count_nonzero(d))
    if n_nonzero == 0:
        return WilcoxonResult(statistic=0.0, p=1.0, n=n, n_nonzero=0, alternative=alternative)
    res = _sps.wilcoxon(d, zero_method=zero_method, alternative=alternative, method="auto")
    stat, p = float(res.statistic), float(res.pvalue)
    if not math.isfinite(p):
        p = 1.0
    return WilcoxonResult(statistic=stat, p=min(1.0, max(0.0, p)), n=n, n_nonzero=n_nonzero,
                          alternative=alternative)


@dataclass(frozen=True)
class HolmResult:
    """One Holm–Bonferroni adjusted test."""

    name: str
    p: float
    p_adj: float
    reject: bool


def holm(
    pvalues: Mapping[str, float] | Sequence[float],
    alpha: float = 0.05,
) -> list[HolmResult]:
    """Holm step-down correction; output is in *input order*.

    With ``m`` tests sorted ascending ``p_(1) ≤ … ≤ p_(m)``, the adjusted value
    is ``p_adj_(i) = min(1, max_{j ≤ i} (m − j + 1) · p_(j))`` (cumulative max
    makes it monotone in the raw p). ``reject = p_adj ≤ alpha``, which equals
    the classical step-down rule. Ties are ordered stably. A sequence input
    gets names ``"0", "1", …``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if isinstance(pvalues, Mapping):
        names = [str(k) for k in pvalues.keys()]
        ps = np.asarray(list(pvalues.values()), dtype=float)
    else:
        ps = np.asarray(list(pvalues), dtype=float)
        names = [str(i) for i in range(ps.shape[0])]
    if ps.ndim != 1:
        raise ValueError("p-values must be 1-D")
    m = int(ps.shape[0])
    if m == 0:
        return []
    if not np.all(np.isfinite(ps)) or np.any(ps < 0.0) or np.any(ps > 1.0):
        raise ValueError("p-values must be finite and in [0, 1]")
    order = np.argsort(ps, kind="stable")
    scaled = ps[order] * (m - np.arange(m))
    adj_sorted = np.minimum(1.0, np.maximum.accumulate(scaled))
    adj = np.empty(m, dtype=float)
    adj[order] = adj_sorted
    return [HolmResult(name=names[i], p=float(ps[i]), p_adj=float(adj[i]), reject=bool(adj[i] <= alpha))
            for i in range(m)]


# ---------------------------------------------------------------------------
# Evaluation-table helpers (Requirements 25.4–25.6)
# ---------------------------------------------------------------------------

#: Pairing keys for evaluation tables. ``run`` is excluded: eval environments
#: always use run=0, while RL rows may carry their training run index.
EVAL_KEYS: tuple[str, ...] = ("scenario", "split", "episode")

#: +1 = higher is better, −1 = lower is better.
METRIC_DIRECTION: dict[str, int] = {
    "J": +1,
    "success_rate": +1,
    "p95_latency_all_ms": -1,
    "p95_latency_success_ms": -1,
    "pings_per_hour": -1,
    "evictions_per_hour": -1,
    "messages_per_lookup": -1,
}

PRIMARY_METRICS: tuple[str, ...] = ("success_rate", "p95_latency_all_ms", "pings_per_hour")
DEFAULT_SCENARIOS: tuple[str, ...] = ("stable", "high_churn", "heldout_medium")
DEFAULT_COMPARATORS: tuple[str, ...] = ("kademlia_epoch", "tuned_heuristic", "matched_random")


def average_over_runs(
    df: pd.DataFrame,
    rl_policies: Sequence[str],
    label: str = "rl_mean",
    keys: Sequence[str] = EVAL_KEYS,
    metrics: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Per-seed mean of ``metrics`` across the RL run policies.

    Each policy in ``rl_policies`` is one training run evaluated on the same
    seeds. Returns new rows with ``policy = label`` (and ``run = 0`` if the
    input has a ``run`` column, matching the eval environment). Raises
    ``ValueError`` if a run is missing, has duplicate keys, or does not cover
    exactly the same keys as the other runs. ``metrics`` defaults to every
    numeric column other than the keys and ``run``.
    """
    keys = list(keys)
    rl_policies = list(rl_policies)
    if not rl_policies:
        raise ValueError("rl_policies must be non-empty")
    if len(set(rl_policies)) != len(rl_policies):
        raise ValueError(f"duplicate RL policies: {rl_policies}")
    if metrics is None:
        skip = set(keys) | {"run", "policy"}
        metrics = [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
    metrics = list(metrics)
    missing = [c for c in ["policy", *keys, *metrics] if c not in df.columns]
    if missing:
        raise ValueError(f"results frame is missing columns: {missing}")

    parts: list[pd.DataFrame] = []
    ref_index: pd.Index | None = None
    for p in rl_policies:
        sub = df.loc[df["policy"] == p, [*keys, *metrics]]
        if sub.empty:
            raise ValueError(f"missing run: no rows for RL policy {p!r}")
        if sub.duplicated(subset=keys).any():
            raise ValueError(f"duplicate keys {keys} for RL policy {p!r}")
        s = sub.set_index(keys).sort_index()
        if ref_index is None:
            ref_index = s.index
        elif set(s.index) != set(ref_index):
            raise ValueError(f"missing run: RL policy {p!r} does not cover the same keys as {rl_policies[0]!r}")
        parts.append(s.reindex(ref_index).astype(float))
    mean = sum(parts) / len(parts)
    out = mean.reset_index()
    out.insert(0, "policy", label)
    if "run" in df.columns:
        out.insert(1 + len(keys), "run", 0)
    return out


def _with_rl_mean(df: pd.DataFrame, rl_policies: Sequence[str], rl_label: str,
                  keys: Sequence[str], metrics: Sequence[str]) -> pd.DataFrame:
    if (df["policy"] == rl_label).any():
        return df
    avg = average_over_runs(df, rl_policies, rl_label, keys, [m for m in metrics if m in df.columns])
    return pd.concat([df, avg], ignore_index=True)


def _paired_in_scenario(df: pd.DataFrame, scenario: str, metric: str, a: str, b: str,
                        keys: Sequence[str]) -> pd.DataFrame:
    sub = df[df["scenario"] == scenario]
    if sub.empty:
        raise ValueError(f"no rows for scenario {scenario!r}")
    return paired_frame(sub, metric, a, b, keys)


def _compare(df: pd.DataFrame, scenario: str, metric: str, a: str, b: str, keys: Sequence[str],
             n_resamples: int, seed: int, level: float = 0.95) -> dict[str, Any]:
    pf = _paired_in_scenario(df, scenario, metric, a, b, keys)
    xa, xb = pf["a"].to_numpy(), pf["b"].to_numpy()
    ci = paired_bootstrap_ci(xa, xb, n_resamples=n_resamples, level=level, seed=seed)
    w = wilcoxon_paired(xa, xb)
    return {"n": ci.n, "mean_diff": ci.mean, "ci_lo": ci.lo, "ci_hi": ci.hi,
            "wilcoxon_stat": w.statistic, "p": w.p}


def primary_tests(
    df: pd.DataFrame,
    rl: str = "rl_mean",
    comparators: Sequence[str] = DEFAULT_COMPARATORS,
    scenarios: Sequence[str] = DEFAULT_SCENARIOS,
    metric: str = "J",
    n_resamples: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
    keys: Sequence[str] = EVAL_KEYS,
) -> pd.DataFrame:
    """The primary family: ``rl`` vs each comparator × scenario on ``metric``.

    ``df`` must already contain rows for ``rl`` (e.g. from
    :func:`average_over_runs`). Differences are ``rl − comparator``. Each row
    has the paired bootstrap CI, two-sided Wilcoxon p, and the Holm-adjusted p
    over *all* rows of the family (9 with the defaults). ``significant`` is the
    Holm reject flag; ``direction`` is +1/−1 (favourable/unfavourable sign of
    the mean diff relative to ``METRIC_DIRECTION``) or 0.
    """
    sign = METRIC_DIRECTION.get(metric, +1)
    rows = []
    for sc in scenarios:
        for comp in comparators:
            r = _compare(df, sc, metric, rl, comp, keys, n_resamples, seed)
            rows.append({"scenario": sc, "comparator": comp, "metric": metric, **r})
    hr = holm({f"{r['scenario']}|{r['comparator']}": r["p"] for r in rows}, alpha=alpha)
    for r, h in zip(rows, hr):
        r["p_holm"] = h.p_adj
        r["significant"] = h.reject
        r["direction"] = int(np.sign(sign * r["mean_diff"]))
    cols = ["scenario", "comparator", "metric", "n", "mean_diff", "ci_lo", "ci_hi",
            "wilcoxon_stat", "p", "p_holm", "significant", "direction"]
    return pd.DataFrame(rows, columns=cols)


def per_run_results(
    df: pd.DataFrame,
    rl_policies: Sequence[str],
    comparators: Sequence[str] = DEFAULT_COMPARATORS,
    scenarios: Sequence[str] = DEFAULT_SCENARIOS,
    metric: str = "J",
    n_resamples: int = 10_000,
    seed: int = 0,
    keys: Sequence[str] = EVAL_KEYS,
) -> pd.DataFrame:
    """Per-run paired diffs (``run policy − comparator``) with CI and *uncorrected* Wilcoxon p.

    Informational only (not part of the Holm family). ``run`` is the index of
    the policy in ``rl_policies``.
    """
    rows = []
    for i, rp in enumerate(rl_policies):
        for sc in scenarios:
            for comp in comparators:
                r = _compare(df, sc, metric, rp, comp, keys, n_resamples, seed)
                rows.append({"rl_policy": rp, "run": i, "scenario": sc, "comparator": comp,
                             "metric": metric, **r})
    cols = ["rl_policy", "run", "scenario", "comparator", "metric", "n", "mean_diff",
            "ci_lo", "ci_hi", "wilcoxon_stat", "p"]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# H1–H5 hypothesis table (Requirement 25.6)
# ---------------------------------------------------------------------------

PASS, FAIL, INCONCLUSIVE = "pass", "fail", "inconclusive"


def _combine(verdicts: Sequence[str]) -> str:
    """All pass → pass; any fail → fail; otherwise inconclusive."""
    if verdicts and all(v == PASS for v in verdicts):
        return PASS
    if any(v == FAIL for v in verdicts):
        return FAIL
    return INCONCLUSIVE


def _fmt(x: float) -> str:
    return f"{x:+.4g}"


def _better(sign: int, lo: float, hi: float) -> bool:
    """CI entirely on the favourable side of 0."""
    return lo > 0.0 if sign > 0 else hi < 0.0


def _cannot_be_better(sign: int, lo: float, hi: float) -> bool:
    """CI excludes every favourable value (lies at or on the unfavourable side of 0)."""
    return hi <= 0.0 if sign > 0 else lo >= 0.0


def _h2_check(df: pd.DataFrame, scenario: str, rl: str, comparators: Sequence[str],
              metrics: Sequence[str], margin: float, keys: Sequence[str],
              n_resamples: int, seed: int) -> tuple[str, str]:
    """H2-style check in one scenario; returns (verdict, evidence).

    Per comparator:
    * non-inferior on success: bootstrap CI lo of (RL − comp) success > −margin;
    * clearly inferior: mean < −margin and CI lo < −margin;
    * better on metric m: CI on the favourable side of 0 (higher J/success,
      lower p95/pings).
    pass = non-inferior and better on ≥1 of ``metrics``; fail = clearly
    inferior, or every metric's CI excludes all favourable values; else
    inconclusive. Overall: combined with :func:`_combine`.
    """
    verdicts, ev = [], []
    for comp in comparators:
        s = _compare(df, scenario, "success_rate", rl, comp, keys, n_resamples, seed)
        noninf = s["ci_lo"] > -margin
        inferior = s["mean_diff"] < -margin and s["ci_lo"] < -margin
        better, hopeless = [], True
        for m in metrics:
            sign = METRIC_DIRECTION.get(m, +1)
            r = s if m == "success_rate" else _compare(df, scenario, m, rl, comp, keys, n_resamples, seed)
            if _better(sign, r["ci_lo"], r["ci_hi"]):
                better.append(m)
            if not _cannot_be_better(sign, r["ci_lo"], r["ci_hi"]):
                hopeless = False
        if noninf and better:
            v = PASS
        elif inferior or hopeless:
            v = FAIL
        else:
            v = INCONCLUSIVE
        verdicts.append(v)
        ev.append(f"{comp}: dsucc={_fmt(s['mean_diff'])} [{_fmt(s['ci_lo'])},{_fmt(s['ci_hi'])}], "
                  f"better on {','.join(better) if better else 'none'} ({v})")
    return _combine(verdicts), f"{scenario}: " + "; ".join(ev)


def hypothesis_table(
    df: pd.DataFrame,
    rl_policies: Sequence[str],
    rl_label: str = "rl_mean",
    kademlia: str = "kademlia_epoch",
    tuned: str = "tuned_heuristic",
    matched: str = "matched_random",
    scenarios: Sequence[str] = DEFAULT_SCENARIOS,
    h2_scenario: str = "high_churn",
    h4_scenario: str = "heldout_medium",
    h3_baselines: Sequence[str] | None = None,
    exclude_from_best: Sequence[str] = ("oracle_heuristic",),
    primary_metrics: Sequence[str] = PRIMARY_METRICS,
    margin: float = 0.01,
    p95_tolerance: float = 0.05,
    alpha: float = 0.05,
    n_resamples: int = 10_000,
    seed: int = 0,
    keys: Sequence[str] = EVAL_KEYS,
) -> pd.DataFrame:
    """Automatic H1–H5 table: columns ``id, statement, verdict, evidence``.

    RL is the per-seed mean over ``rl_policies`` (added as ``rl_label`` if not
    already present). All diffs are RL − baseline. Verdicts ∈ {pass, fail,
    inconclusive}:

    * H1 (J, every scenario, Holm over the 9 primary tests): pass = in every
      scenario RL vs matched is Holm-significant with mean > 0; fail = any
      scenario Holm-significant with mean < 0, or CI hi < 0; else inconclusive.
    * H2 (``h2_scenario``): :func:`_h2_check` vs kademlia and tuned on
      ``primary_metrics`` with non-inferiority ``margin`` (1 pp).
    * H3 (each scenario): pings — pass if mean diff vs kademlia ≤ 0 and CI hi
      ≤ 0, fail if CI lo > 0; p95_all — RL mean ≤ (1 + ``p95_tolerance``) ×
      lowest baseline mean (``h3_baselines`` or every non-RL policy present in
      the scenario except ``exclude_from_best``), else fail. Scenario pass needs
      both; overall combined over scenarios.
    * H4: the H2 check on ``h4_scenario``.
    * H5 (``h2_scenario``): for every run and each of kademlia/tuned, the run's
      mean J diff has the same sign as the pooled (RL-mean) diff and the run is
      non-inferior on success (CI lo > −margin). fail if some run has the
      opposite sign with a CI excluding 0, or is clearly inferior on success
      (mean < −margin and CI lo < −margin); a zero pooled diff, or otherwise
      unmet criteria, give inconclusive.
    """
    rl_policies = list(rl_policies)
    needed = ["J", *primary_metrics]
    d = _with_rl_mean(df, rl_policies, rl_label, keys, needed)
    kw = dict(keys=keys, n_resamples=n_resamples, seed=seed)
    rows = []

    # H1
    pt = primary_tests(d, rl=rl_label, comparators=(kademlia, tuned, matched), scenarios=scenarios,
                       metric="J", alpha=alpha, **kw)
    h1 = pt[pt["comparator"] == matched]
    v1 = []
    for _, r in h1.iterrows():
        if r["significant"] and r["mean_diff"] > 0:
            v1.append(PASS)
        elif (r["significant"] and r["mean_diff"] < 0) or r["ci_hi"] < 0:
            v1.append(FAIL)
        else:
            v1.append(INCONCLUSIVE)
    ev1 = "; ".join(f"{r['scenario']}: dJ={_fmt(r['mean_diff'])} [{_fmt(r['ci_lo'])},{_fmt(r['ci_hi'])}], "
                    f"p_holm={r['p_holm']:.3g}" for _, r in h1.iterrows())
    rows.append({"id": "H1", "statement": f"RL > {matched} on J in every scenario (Holm-corrected)",
                 "verdict": _combine(v1), "evidence": ev1})

    # H2
    v2, ev2 = _h2_check(d, h2_scenario, rl_label, (kademlia, tuned), primary_metrics, margin, **kw)
    rows.append({"id": "H2", "statement": f"{h2_scenario}: RL non-inferior (within {margin * 100:g} pp success) "
                 f"to {kademlia} and {tuned}, and significantly better on >=1 primary metric",
                 "verdict": v2, "evidence": ev2})

    # H3
    rl_set = set(rl_policies) | {rl_label}
    v3, ev3 = [], []
    for sc in scenarios:
        p = _compare(d, sc, "pings_per_hour", rl_label, kademlia, n_resamples=n_resamples, seed=seed, keys=keys)
        if p["mean_diff"] <= 0 and p["ci_hi"] <= 0:
            vp = PASS
        elif p["ci_lo"] > 0:
            vp = FAIL
        else:
            vp = INCONCLUSIVE
        sub = d[d["scenario"] == sc]
        if h3_baselines is None:
            base = [q for q in pd.unique(sub["policy"]) if q not in rl_set and q not in set(exclude_from_best)]
        else:
            base = list(h3_baselines)
        means = sub[sub["policy"].isin(base)].groupby("policy")["p95_latency_all_ms"].mean()
        if means.empty:
            raise ValueError(f"no baseline p95 values in scenario {sc!r}")
        best_pol, best = str(means.idxmin()), float(means.min())
        rl_p95 = float(sub.loc[sub["policy"] == rl_label, "p95_latency_all_ms"].mean())
        vl = PASS if rl_p95 <= (1.0 + p95_tolerance) * best else FAIL
        v = _combine([vp, vl])
        v3.append(v)
        ev3.append(f"{sc}: dpings={_fmt(p['mean_diff'])} [{_fmt(p['ci_lo'])},{_fmt(p['ci_hi'])}], "
                   f"p95 RL={rl_p95:.4g} vs best {best_pol}={best:.4g} (ratio {rl_p95 / best if best else float('inf'):.3f}) ({v})")
    rows.append({"id": "H3", "statement": f"RL pings/h <= {kademlia} and p95_all within "
                 f"{p95_tolerance * 100:g}% of the best baseline, in every scenario",
                 "verdict": _combine(v3), "evidence": "; ".join(ev3)})

    # H4
    v4, ev4 = _h2_check(d, h4_scenario, rl_label, (kademlia, tuned), primary_metrics, margin, **kw)
    rows.append({"id": "H4", "statement": f"H2 holds on {h4_scenario}", "verdict": v4, "evidence": ev4})

    # H5
    v5, ev5 = [], []
    for comp in (kademlia, tuned):
        pooled = _compare(d, h2_scenario, "J", rl_label, comp, **kw)
        ps = int(np.sign(pooled["mean_diff"]))
        signs = []
        for rp in rl_policies:
            j = _compare(d, h2_scenario, "J", rp, comp, **kw)
            s = _compare(d, h2_scenario, "success_rate", rp, comp, **kw)
            rs = int(np.sign(j["mean_diff"]))
            noninf = s["ci_lo"] > -margin
            inferior = s["mean_diff"] < -margin and s["ci_lo"] < -margin
            opposite_sig = ps != 0 and rs == -ps and (j["ci_lo"] > 0 or j["ci_hi"] < 0)
            if ps != 0 and rs == ps and noninf:
                v5.append(PASS)
            elif opposite_sig or inferior:
                v5.append(FAIL)
            else:
                v5.append(INCONCLUSIVE)
            signs.append(f"{rp}={_fmt(j['mean_diff'])}")
        ev5.append(f"vs {comp}: pooled dJ={_fmt(pooled['mean_diff'])}, runs " + ",".join(signs))
    rows.append({"id": "H5", "statement": f"all {len(rl_policies)} RL runs individually in H2's direction "
                 f"({h2_scenario})", "verdict": _combine(v5), "evidence": "; ".join(ev5)})

    return pd.DataFrame(rows, columns=["id", "statement", "verdict", "evidence"])


def to_markdown(table: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-style Markdown table (no extra dependency)."""
    def cell(x: Any) -> str:
        if isinstance(x, float):
            s = f"{x:.4g}"
        else:
            s = str(x)
        return s.replace("|", "\\|").replace("\n", " ")

    cols = [str(c) for c in table.columns]
    lines = ["| " + " | ".join(cell(c) for c in cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for row in table.itertuples(index=False):
        lines.append("| " + " | ".join(cell(x) for x in row) + " |")
    return "\n".join(lines) + "\n"


def write_hypothesis_table(table: pd.DataFrame, csv_path: str | Path, md_path: str | Path) -> None:
    """Write the H1–H5 table as CSV and as a Markdown table (parents created)."""
    csv_path, md_path = Path(csv_path), Path(md_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    md_path.write_text("# Hypotheses H1–H5\n\n" + to_markdown(table), encoding="utf-8")
