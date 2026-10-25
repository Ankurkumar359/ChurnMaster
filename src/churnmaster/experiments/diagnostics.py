"""Reward and state-coverage diagnostics (Requirement 23, design "Reward diagnostics").

Run before training, on calibration seeds, with the ``uniform_random`` policy:
its actions are independent of the state and of ground truth, so comparing
returns of different actions taken in the same (ground-truth) situation is an
unbiased Monte-Carlo comparison under a random continuation policy.

1. Reward-total identity (23.1). Per episode: ``|metrics.J − ledger.score()|``
   and ``|Σ r[b,k] over measured bucket-epochs − ledger.total_reward()|``
   (a measured bucket-epoch is one holding at least one measured record).
   Both must be ≤ 1e-9.
2. Reward sanity (23.2). Every non-null epoch decision ``(b, k)`` is joined
   with its ground-truth ``"decision"`` record (same bucket and ``epoch_idx``)
   and given the discounted return ``G = Σ_{i<H} γ^i r[b, k+i]`` from the
   ledger's per-bucket-epoch rewards (all records, warm-up included, as in
   learning). Decisions whose horizon runs past the episode's last epoch are
   skipped (no truncated returns). Groups and comparisons (difference of
   means, better − worse, two-sample percentile bootstrap CI; pass iff
   CI lo > 0):

   * dead incumbent (``inc_online`` False) in tiers TOP/SECOND:
     PROBE − KEEP (primary); PROBE − KEEP restricted to ``cand_present``
     (info); REPLACE − KEEP restricted to ``cand_present`` (primary -- REPLACE
     is only allowed with a candidate, so KEEP is restricted to the same
     situations for a fair comparison);
   * live incumbent (any tier): KEEP − REPLACE restricted to ``cand_present``
     (primary); the same restricted to tiers TOP/SECOND (info).
3. Coverage (23.3). Visits per state over the non-null decisions of
   ``n_episodes`` episodes; the fraction of decisions in states visited at
   least ``min_visits`` times must be ≥ 0.80.
4. J vs success (23.4). Spearman rank correlation between J and success
   rate, across per-(scenario, policy) means and across per-seed rows
   (pooled and per scenario), from the headroom ``per_seed.csv``.

:func:`run_diagnostics` refuses (``FrozenEnvError``) unless every scenario's
env hash matches the frozen manifest.
"""

from __future__ import annotations

import json
import math
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from churnmaster.actions import Action
from churnmaster.experiments.runner import (ConfigLoader, EpisodeResult, Job, check_frozen,
                                            read_manifest, results_table, run_job, run_jobs)
from churnmaster.experiments.stats import TwoSampleCI, two_sample_bootstrap_ci
from churnmaster.rl.ledger import Ledger
from churnmaster.rl.state import TIER_SECOND, TIER_TOP, decode
from churnmaster.seeding import Split
from churnmaster.sim.observation import DecisionRecord, GTRecord, ObservedLookup

__all__ = [
    "DIAG_POLICY", "DEFAULT_SCENARIOS", "GAMMA", "HORIZON", "MIN_VISITS", "COVERAGE_TARGET",
    "N_EPISODES", "IDENTITY_TOL", "DecisionSample", "Comparison", "COMPARISONS",
    "IdentityReport", "ComparisonResult", "SanityReport", "CoverageReport", "SpearmanResult",
    "DiagnosticsReport", "identity_errors", "reward_identity", "discounted_return",
    "decision_samples", "samples_from_results", "reward_sanity", "reward_sanity_from_samples",
    "state_visits", "coverage_from_visits", "coverage", "j_success_spearman",
    "run_diagnostics", "write_outputs", "DIAG_FAILED_EXIT",
]

DIAG_POLICY = "uniform_random"
DEFAULT_SCENARIOS: tuple[str, ...] = ("high_churn", "stable")
GAMMA = 0.9
HORIZON = 10
MIN_VISITS = 50
COVERAGE_TARGET = 0.80
N_EPISODES = 20
IDENTITY_TOL = 1e-9
DIAG_FAILED_EXIT = 4
TOP2 = (TIER_TOP, TIER_SECOND)


def _clean(x: Any) -> Any:
    """JSON-safe copy: NaN -> None, numpy scalars -> Python, tuple keys -> str."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return None if math.isnan(v) else ("inf" if math.isinf(v) else v)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


# --------------------------------------------------------------------------- #
# 1. Reward-total identity
# --------------------------------------------------------------------------- #

def _abs_err(x: float, y: float) -> float:
    if math.isnan(x) and math.isnan(y):
        return 0.0
    if math.isnan(x) or math.isnan(y):
        return math.inf
    return abs(x - y)


def identity_errors(J: float, ledger: Ledger, lookups: Iterable[ObservedLookup],
                    decisions: Iterable[DecisionRecord]) -> tuple[float, float]:
    """``(|J − ledger.score()|, |Σ measured r[b,k] − ledger.total_reward()|)``."""
    keys = {(lk.bucket, int(lk.t_ms) // ledger.epoch_ms) for lk in lookups if lk.measured}
    keys |= {(d.bucket, d.epoch_idx) for d in decisions if d.measured}
    total = math.fsum(ledger.reward(b, k) for b, k in keys)
    return _abs_err(float(J), ledger.score()), abs(total - ledger.total_reward())


@dataclass
class IdentityReport:
    n_episodes: int
    n_passed: int
    max_abs_err_J: float
    max_abs_err_total: float
    tol: float = IDENTITY_TOL

    @property
    def passed(self) -> bool:
        return self.n_episodes > 0 and self.n_passed == self.n_episodes

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "passed": self.passed}


def _identity_report(errs: Sequence[tuple[float, float]], tol: float = IDENTITY_TOL) -> IdentityReport:
    n_ok = sum(1 for eJ, eT in errs if eJ <= tol and eT <= tol)
    return IdentityReport(len(errs), n_ok, max((e[0] for e in errs), default=math.nan),
                          max((e[1] for e in errs), default=math.nan), tol)


def _need_records(r: EpisodeResult) -> None:
    if r.ledger is None or r.decisions is None or r.lookups is None or r.gts is None:
        raise ValueError("diagnostics need results run with keep_records=True")


def reward_identity(results: Iterable[EpisodeResult], tol: float = IDENTITY_TOL) -> IdentityReport:
    """Identity check over every episode (results must be run with ``keep_records``)."""
    errs = []
    for r in results:
        _need_records(r)
        errs.append(identity_errors(r.metrics.J, r.ledger, r.lookups, r.decisions))
    return _identity_report(errs, tol)


# --------------------------------------------------------------------------- #
# 2. Reward sanity
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DecisionSample:
    """One non-null epoch decision with its GT facts and discounted return ``G``."""

    scenario: str
    episode: int
    bucket: int
    epoch_idx: int
    state: int
    action: int
    tier: int
    inc_online: bool
    cand_present: bool
    G: float


def discounted_return(reward: Callable[[int, int], float], bucket: int, epoch_idx: int,
                      gamma: float = GAMMA, horizon: int = HORIZON) -> float:
    """``Σ_{i=0}^{H-1} γ^i · reward(bucket, epoch_idx + i)``."""
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    return float(sum((gamma ** i) * reward(bucket, epoch_idx + i) for i in range(horizon)))


def decision_samples(decisions: Iterable[DecisionRecord], gts: Iterable[GTRecord],
                     reward: Callable[[int, int], float], gamma: float = GAMMA,
                     horizon: int = HORIZON, last_epoch: int | None = None,
                     scenario: str = "", episode: int = 0) -> list[DecisionSample]:
    """Join non-null decisions with their GT ``"decision"`` records and compute ``G``.

    ``last_epoch`` (default: the largest ``epoch_idx`` among ``decisions``) is
    the last epoch of the episode; decisions with ``epoch_idx + horizon − 1 >
    last_epoch`` are skipped. Raises ValueError when a decision has no GT
    record or the GT action disagrees.
    """
    decisions = list(decisions)
    gt_by_key: dict[tuple[int, int], GTRecord] = {}
    for g in gts:
        if g.kind == "decision":
            gt_by_key[(int(g.bucket), int(g.get("epoch_idx")))] = g
    if last_epoch is None:
        last_epoch = max((d.epoch_idx for d in decisions), default=-1)
    out: list[DecisionSample] = []
    for d in decisions:
        if d.state is None:
            continue
        g = gt_by_key.get((d.bucket, d.epoch_idx))
        if g is None:
            raise ValueError(f"no GT decision record for bucket {d.bucket} epoch {d.epoch_idx}")
        if int(g.get("action")) != int(d.action):
            raise ValueError(f"GT action {g.get('action')} != decision action {d.action} "
                             f"at bucket {d.bucket} epoch {d.epoch_idx}")
        if d.epoch_idx + horizon - 1 > last_epoch:
            continue
        out.append(DecisionSample(
            scenario=scenario, episode=int(episode), bucket=int(d.bucket),
            epoch_idx=int(d.epoch_idx), state=int(d.state), action=int(d.action),
            tier=decode(int(d.state)).tier, inc_online=bool(g.get("inc_online")),
            cand_present=bool(g.get("cand_present")),
            G=discounted_return(reward, d.bucket, d.epoch_idx, gamma, horizon)))
    return out


def samples_from_results(results: Iterable[EpisodeResult], gamma: float = GAMMA,
                         horizon: int = HORIZON) -> list[DecisionSample]:
    out: list[DecisionSample] = []
    for r in results:
        _need_records(r)
        out += decision_samples(r.decisions, r.gts, r.ledger.reward, gamma, horizon,
                                scenario=r.scenario, episode=r.episode)
    return out


_GROUPS: dict[str, Callable[[DecisionSample], bool]] = {
    "dead_top2": lambda s: (not s.inc_online) and s.tier in TOP2,
    "live": lambda s: s.inc_online,
    "live_top2": lambda s: s.inc_online and s.tier in TOP2,
}


@dataclass(frozen=True)
class Comparison:
    """``better − worse`` mean discounted return within ``group``."""

    name: str
    group: str
    better: Action
    worse: Action
    cand_only: bool      # restrict both sides to decisions with a candidate present
    primary: bool        # counts toward the overall pass flag


COMPARISONS: tuple[Comparison, ...] = (
    Comparison("dead_top2_probe_vs_keep", "dead_top2", Action.PROBE, Action.KEEP, False, True),
    Comparison("dead_top2_probe_vs_keep_cand", "dead_top2", Action.PROBE, Action.KEEP, True, False),
    Comparison("dead_top2_replace_vs_keep_cand", "dead_top2", Action.REPLACE, Action.KEEP, True, True),
    Comparison("live_keep_vs_replace_cand", "live", Action.KEEP, Action.REPLACE, True, True),
    Comparison("live_top2_keep_vs_replace_cand", "live_top2", Action.KEEP, Action.REPLACE, True, False),
)


@dataclass
class ComparisonResult:
    name: str
    group: str
    better: str
    worse: str
    cand_only: bool
    primary: bool
    n_better: int
    n_worse: int
    mean_better: float
    mean_worse: float
    ci: TwoSampleCI | None   # None when a side is empty

    @property
    def passed(self) -> bool:
        return self.ci is not None and self.ci.lo > 0.0

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in asdict(self).items() if k != "ci"}
        d["ci"] = None if self.ci is None else self.ci.to_dict()
        d["passed"] = self.passed
        return d


@dataclass
class SanityReport:
    gamma: float
    horizon: int
    n_samples: int
    comparisons: list[ComparisonResult]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.comparisons if c.primary)

    def comparison(self, name: str) -> ComparisonResult:
        for c in self.comparisons:
            if c.name == name:
                return c
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        return {"gamma": self.gamma, "horizon": self.horizon, "n_samples": self.n_samples,
                "passed": self.passed, "comparisons": [c.to_dict() for c in self.comparisons]}


def reward_sanity_from_samples(samples: Sequence[DecisionSample], gamma: float = GAMMA,
                               horizon: int = HORIZON,
                               comparisons: Sequence[Comparison] = COMPARISONS,
                               n_resamples: int = 10_000, level: float = 0.95,
                               seed: int = 0) -> SanityReport:
    out = []
    for c in comparisons:
        sel = [s for s in samples if _GROUPS[c.group](s) and (s.cand_present or not c.cand_only)]
        gb = np.array([s.G for s in sel if s.action == int(c.better)], dtype=float)
        gw = np.array([s.G for s in sel if s.action == int(c.worse)], dtype=float)
        ci = (two_sample_bootstrap_ci(gb, gw, n_resamples=n_resamples, level=level, seed=seed)
              if gb.size and gw.size else None)
        out.append(ComparisonResult(
            name=c.name, group=c.group, better=c.better.name, worse=c.worse.name,
            cand_only=c.cand_only, primary=c.primary, n_better=int(gb.size), n_worse=int(gw.size),
            mean_better=float(gb.mean()) if gb.size else math.nan,
            mean_worse=float(gw.mean()) if gw.size else math.nan, ci=ci))
    return SanityReport(gamma, horizon, len(samples), out)


def reward_sanity(results: Iterable[EpisodeResult], gamma: float = GAMMA, horizon: int = HORIZON,
                  **kw: Any) -> SanityReport:
    """Reward sanity over ``results`` (run with ``keep_records``); see module docstring."""
    return reward_sanity_from_samples(samples_from_results(results, gamma, horizon),
                                      gamma, horizon, **kw)


# --------------------------------------------------------------------------- #
# 3. Coverage
# --------------------------------------------------------------------------- #

def state_visits(decisions: Iterable[DecisionRecord]) -> Counter:
    """Visits per state over non-null decisions."""
    return Counter(int(d.state) for d in decisions if d.state is not None)


@dataclass
class CoverageReport:
    n_episodes: int
    n_decisions: int
    n_states_visited: int
    n_states_covered: int     # states with >= min_visits
    fraction: float           # decisions in covered states / all decisions
    min_visits: int
    target: float
    visits: dict[int, int] = field(repr=False, default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.n_decisions > 0 and self.fraction >= self.target

    def to_dict(self, with_visits: bool = False) -> dict[str, Any]:
        d = {k: v for k, v in asdict(self).items() if k != "visits"}
        d["passed"] = self.passed
        if with_visits:
            d["visits"] = {str(s): n for s, n in sorted(self.visits.items())}
        return d


def coverage_from_visits(visits: Mapping[int, int], min_visits: int = MIN_VISITS,
                         target: float = COVERAGE_TARGET, n_episodes: int = 0) -> CoverageReport:
    total = int(sum(visits.values()))
    covered = {s: n for s, n in visits.items() if n >= min_visits}
    frac = (sum(covered.values()) / total) if total else math.nan
    return CoverageReport(n_episodes=n_episodes, n_decisions=total,
                          n_states_visited=sum(1 for n in visits.values() if n > 0),
                          n_states_covered=len(covered), fraction=float(frac),
                          min_visits=min_visits, target=target,
                          visits={int(s): int(n) for s, n in visits.items()})


def coverage(results: Sequence[EpisodeResult], min_visits: int = MIN_VISITS,
             n_episodes: int | None = N_EPISODES,
             target: float = COVERAGE_TARGET) -> CoverageReport:
    """Coverage over the first ``n_episodes`` results (all if None); ValueError if fewer."""
    results = list(results)
    if n_episodes is not None:
        if len(results) < n_episodes:
            raise ValueError(f"coverage needs {n_episodes} episodes, got {len(results)}")
        results = results[:n_episodes]
    visits: Counter = Counter()
    for r in results:
        if r.decisions is None:
            raise ValueError("coverage needs results run with keep_records=True")
        visits.update(state_visits(r.decisions))
    return coverage_from_visits(visits, min_visits, target, len(results))


# --------------------------------------------------------------------------- #
# 4. J vs success-rate Spearman
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SpearmanResult:
    level: str      # "policy_means" or "per_seed"
    scope: str      # "pooled" or a scenario name
    n: int
    rho: float
    p: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    from scipy.stats import spearmanr

    if x.size < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return math.nan, math.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = spearmanr(x, y)
    return float(res.statistic), float(res.pvalue)


def j_success_spearman(df: pd.DataFrame, x: str = "J", y: str = "success_rate") -> list[SpearmanResult]:
    """Spearman ρ(J, success) over per-(scenario, policy) means and per-seed rows.

    Rows with NaN in either column are dropped. Scopes: pooled and each
    scenario. Fewer than 3 points or a constant column give NaN.
    """
    missing = [c for c in ("policy", "scenario", x, y) if c not in df.columns]
    if missing:
        raise ValueError(f"frame is missing columns {missing}")
    d = df[["policy", "scenario", x, y]].dropna()
    means = d.groupby(["scenario", "policy"], sort=False)[[x, y]].mean().reset_index()
    out = []
    for level, frame in (("policy_means", means), ("per_seed", d)):
        scopes = [("pooled", frame)] + [(s, g) for s, g in frame.groupby("scenario", sort=False)]
        for scope, g in scopes:
            rho, p = _spearman(g[x].to_numpy(float), g[y].to_numpy(float))
            out.append(SpearmanResult(level, str(scope), int(len(g)), rho, p))
    return out


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

@dataclass
class DiagnosticsReport:
    policy: str
    split: Split
    n_episodes: int
    scenarios: list[str]
    gamma: float
    horizon: int
    identity: dict[str, IdentityReport]          # per scenario + "pooled"
    sanity: dict[str, SanityReport]              # per scenario + "pooled"
    coverage: dict[str, CoverageReport]          # per scenario + "pooled"
    spearman: list[SpearmanResult]
    spearman_source: str
    env_hashes: dict[str, str]

    @property
    def passed(self) -> dict[str, bool]:
        """Per-diagnostic verdicts. Coverage is judged per scenario (each over
        ``n_episodes`` episodes); the pooled coverage is informational."""
        return {
            "identity": self.identity["pooled"].passed,
            "reward_sanity": all(self.sanity[s].passed for s in self.scenarios),
            "coverage": all(self.coverage[s].passed for s in self.scenarios),
        }

    @property
    def all_passed(self) -> bool:
        return all(self.passed.values())

    def to_dict(self) -> dict[str, Any]:
        return _clean({
            "policy": self.policy, "split": self.split.name.lower(),
            "n_episodes": self.n_episodes, "scenarios": list(self.scenarios),
            "gamma": self.gamma, "horizon": self.horizon,
            "passed": self.passed, "all_passed": self.all_passed,
            "identity": {k: v.to_dict() for k, v in self.identity.items()},
            "reward_sanity": {k: v.to_dict() for k, v in self.sanity.items()},
            "coverage": {k: v.to_dict() for k, v in self.coverage.items()},
            "spearman": [s.to_dict() for s in self.spearman],
            "spearman_source": self.spearman_source,
            "env_hashes": dict(self.env_hashes),
        })


def _frozen_loader(configs_dir: Path, experiment_path: Path | None,
                   manifest_path: Path) -> Callable[[str], Any]:
    """ConfigLoader with the manifest's tuned env overrides applied (if any)."""
    from churnmaster.experiments.headroom import TunedLoader

    loader = ConfigLoader(configs_dir, experiment_path)
    if manifest_path.is_file():
        overrides = read_manifest(manifest_path).get("env_overrides") or {}
        if overrides:
            return TunedLoader(loader, tuple(sorted(overrides.items())))
    return loader


def run_diagnostics(configs_dir: Path | str, manifest_path: Path | str, *,
                    experiment_path: Path | str | None = None,
                    n_episodes: int = N_EPISODES, split: Split = Split.CALIBRATION,
                    scenarios: Sequence[str] = DEFAULT_SCENARIOS, run: int = 0,
                    gamma: float = GAMMA, horizon: int = HORIZON,
                    min_visits: int = MIN_VISITS, target: float = COVERAGE_TARGET,
                    per_seed_csv: Path | str | None = None,
                    spearman_policies: Sequence[str] | None = None, spearman_seeds: int = 10,
                    n_resamples: int = 10_000, level: float = 0.95, seed: int = 0,
                    progress: Callable[[str], None] | None = None) -> DiagnosticsReport:
    """Run all four diagnostics on ``n_episodes`` uniform_random episodes per scenario.

    Every scenario config is checked against the frozen manifest first
    (``FrozenEnvError`` on mismatch; nothing is run). Spearman uses
    ``per_seed_csv`` (headroom per-seed table) when it exists, else runs
    ``spearman_policies`` (default: the headroom policies) on
    ``spearman_seeds`` episodes of ``split``.
    """
    configs_dir, manifest_path = Path(configs_dir), Path(manifest_path)
    exp = None if experiment_path is None else Path(experiment_path)
    loader = _frozen_loader(configs_dir, exp, manifest_path)
    cfgs = {s: loader(s) for s in scenarios}
    env_hashes = {s: check_frozen(manifest_path, c) for s, c in cfgs.items()}

    errs: dict[str, list[tuple[float, float]]] = {}
    samples: dict[str, list[DecisionSample]] = {}
    visits: dict[str, Counter] = {}
    for s in scenarios:
        errs[s], samples[s], visits[s] = [], [], Counter()
        for ep in range(n_episodes):
            r = run_job(Job(DIAG_POLICY, s, split, run, ep), cfg=cfgs[s], keep_records=True)
            errs[s].append(identity_errors(r.metrics.J, r.ledger, r.lookups, r.decisions))
            samples[s] += decision_samples(r.decisions, r.gts, r.ledger.reward, gamma, horizon,
                                           scenario=s, episode=ep)
            visits[s].update(state_visits(r.decisions))
            if progress:
                progress(f"diagnostics: {s} episode {ep + 1}/{n_episodes} J={r.metrics.J:.4f}")
            del r

    kw = dict(n_resamples=n_resamples, level=level, seed=seed)
    identity = {s: _identity_report(errs[s]) for s in scenarios}
    identity["pooled"] = _identity_report([e for s in scenarios for e in errs[s]])
    sanity = {s: reward_sanity_from_samples(samples[s], gamma, horizon, **kw) for s in scenarios}
    sanity["pooled"] = reward_sanity_from_samples(
        [x for s in scenarios for x in samples[s]], gamma, horizon, **kw)
    cov = {s: coverage_from_visits(visits[s], min_visits, target, n_episodes) for s in scenarios}
    pooled_visits: Counter = Counter()
    for s in scenarios:
        pooled_visits.update(visits[s])
    cov["pooled"] = coverage_from_visits(pooled_visits, min_visits, target,
                                         n_episodes * len(scenarios))

    if per_seed_csv is not None and Path(per_seed_csv).is_file():
        df = pd.read_csv(per_seed_csv)
        source = str(per_seed_csv)
    else:
        from churnmaster.experiments.headroom import DEFAULT_POLICIES

        pols = list(spearman_policies or DEFAULT_POLICIES)
        if progress:
            progress(f"diagnostics: running {len(pols)} policies x {spearman_seeds} seeds "
                     f"for the J/success Spearman")
        jobs = [Job(p, s, split, run, ep) for s in scenarios for p in pols
                for ep in range(spearman_seeds)]
        df = results_table(run_jobs(jobs, cfg_loader=lambda s: cfgs[s]))
        source = f"ran {pols} x {spearman_seeds} {split.name.lower()} seeds"

    return DiagnosticsReport(
        policy=DIAG_POLICY, split=split, n_episodes=n_episodes, scenarios=list(scenarios),
        gamma=gamma, horizon=horizon, identity=identity, sanity=sanity, coverage=cov,
        spearman=j_success_spearman(df), spearman_source=source, env_hashes=env_hashes)


def write_outputs(report: DiagnosticsReport, out_dir: Path | str) -> list[Path]:
    """Write diagnostics.json, coverage.csv, reward_sanity.csv and state_visits.png."""
    from churnmaster.plots import plot_state_visits

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "diagnostics.json").write_text(json.dumps(report.to_dict(), indent=2) + "\n",
                                          encoding="utf-8")
    rows = []
    for scope, c in report.coverage.items():
        for s, n in sorted(c.visits.items()):
            f = decode(s)
            rows.append({"scope": scope, "state": s, "visits": n, "covered": n >= c.min_visits,
                         **asdict(f)})
    pd.DataFrame(rows).to_csv(out / "coverage.csv", index=False)
    srows = []
    for scope, rep in report.sanity.items():
        for c in rep.comparisons:
            d = c.to_dict()
            ci = d.pop("ci") or {}
            srows.append({"scope": scope, **d, "diff": ci.get("mean"), "ci_lo": ci.get("lo"),
                          "ci_hi": ci.get("hi")})
    pd.DataFrame(srows).to_csv(out / "reward_sanity.csv", index=False)
    fig = plot_state_visits(
        [(s, list(report.coverage[s].visits.values())) for s in report.scenarios], out /
        "state_visits.png", min_visits=report.coverage[report.scenarios[0]].min_visits
        if report.scenarios else None,
        title=f"State visits ({report.policy}, {report.n_episodes} episodes per scenario)")
    return [out / "diagnostics.json", out / "coverage.csv", out / "reward_sanity.csv", fig]


if __name__ == "__main__":  # pragma: no cover
    from churnmaster.cli import main

    raise SystemExit(main(["diagnostics"]))
