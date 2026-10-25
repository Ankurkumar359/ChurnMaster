"""Headroom gate and env freeze (Requirement 22, task 6.2).

Revised gates (user decision after checkpoint 5.17): never_replace already
reaches ~99% success in high_churn, so the gates are defined on J and p95
latency instead of success-rate pp. All comparisons are paired by seed
(``stats.paired_ci_from_frame``; percentile bootstrap, 10k resamples, 95%).

G1 -- maintenance matters (scenario high_churn)
    "Best" challenger = the policy among kademlia_epoch, always_replace,
    oracle_heuristic with the largest mean paired ΔJ = J(p) − J(never_replace)
    (ties: that order). G1 passes if the best challenger satisfies

    * route ``J``:   mean ΔJ ≥ ``g1_min_dJ`` and CI(ΔJ).lo > 0, or
    * route ``p95``: mean per-seed relative reduction
      (p95_nr − p95_p) / p95_nr ≥ ``g1_min_p95_rel`` and the paired CI of the
      ms difference (p95_nr − p95_p) has lo > 0.

    All three challengers are reported with their own route flags.

G2 -- room above Kademlia (scenario high_churn)
    ΔJ = J(oracle_heuristic) − J(kademlia_epoch);
    ping_ratio = mean pings/h (oracle) / mean pings/h (kademlia_epoch).

    * route ``J``:     mean ΔJ ≥ ``g2_min_dJ`` and CI(ΔJ).lo > 0, or
    * route ``pings``: ping_ratio ≤ ``g2_max_ping_ratio`` and mean ΔJ ≥ −``g2_J_tol``.

    A G2 failure only sets ``limited_headroom``; the study proceeds. The same
    G2 numbers for stable are reported as information.

G1-failure fallback
    The sim grid deadline_ms × p_loss × replicas (``experiment.tuning``) is
    searched on high_churn *calibration* seeds with the four gate policies,
    in order of closeness to the default: fewest changed parameters, then the
    smallest total distance in grid steps, then grid order. The first setting
    that passes G1 is reported as ``env_overrides`` (gates.json / manifest) and
    is used for the freeze; base config files are never edited. If none
    passes, the manifest is written with ``frozen = false`` (the CLI exits 3).

Freeze
    ``runner.freeze_env`` records the env hash of every scenario file in
    ``configs/scenarios`` (with overrides applied) so later train / eval runs
    can be refused on mismatch (``runner.check_frozen``).
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from churnmaster.config import Config, GatesConfig, StatsConfig, TuningConfig, validate
from churnmaster.experiments.runner import (MANIFEST_NAME, ConfigLoader, EpisodeResult, Job,
                                            env_config_hash, freeze_env, results_table,
                                            run_jobs, summary_by_policy)
from churnmaster.experiments.stats import PairedCI, paired_ci_from_frame, paired_frame
from churnmaster.seeding import Split

__all__ = [
    "BASELINE", "CHALLENGERS", "GATE_POLICIES", "INFO_POLICIES", "DEFAULT_POLICIES",
    "GATE_SCENARIO", "G1_FAILED_EXIT", "TunedLoader", "CandidateStats", "GateResult",
    "TuningResult", "HeadroomReport", "evaluate_gates", "tuning_candidates", "tune_env",
    "run_headroom", "write_outputs",
]

BASELINE = "never_replace"
CHALLENGERS: tuple[str, ...] = ("kademlia_epoch", "always_replace", "oracle_heuristic")
GATE_POLICIES: tuple[str, ...] = (BASELINE,) + CHALLENGERS
INFO_POLICIES: tuple[str, ...] = ("heuristic", "kademlia_std")   # reported, not gated
DEFAULT_POLICIES: tuple[str, ...] = GATE_POLICIES + INFO_POLICIES
DEFAULT_SCENARIOS: tuple[str, ...] = ("stable", "high_churn")
GATE_SCENARIO = "high_churn"
TUNED_FIELDS: tuple[str, ...] = ("deadline_ms", "p_loss", "replicas")
G1_FAILED_EXIT = 3


# --------------------------------------------------------------------------- #
# Gate evaluation (pure)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CandidateStats:
    """One challenger vs never_replace in the gate scenario."""

    policy: str
    dJ: PairedCI                 # J(policy) − J(never_replace)
    p95_diff_ms: PairedCI        # p95(never_replace) − p95(policy)
    p95_rel: float               # mean per-seed (p95_nr − p95_p) / p95_nr
    j_route: bool
    p95_route: bool

    @property
    def passes(self) -> bool:
        return self.j_route or self.p95_route

    def to_dict(self) -> dict[str, Any]:
        return {"policy": self.policy, "dJ": self.dJ.to_dict(),
                "p95_diff_ms": self.p95_diff_ms.to_dict(), "p95_rel": self.p95_rel,
                "j_route": self.j_route, "p95_route": self.p95_route, "passes": self.passes}


def _g2_stats(sub: pd.DataFrame, gates: GatesConfig, ci_kw: Mapping[str, Any]) -> dict[str, Any]:
    dJ = paired_ci_from_frame(sub, "J", "oracle_heuristic", "kademlia_epoch", **ci_kw)
    p_or = float(sub.loc[sub["policy"] == "oracle_heuristic", "pings_per_hour"].mean())
    p_kd = float(sub.loc[sub["policy"] == "kademlia_epoch", "pings_per_hour"].mean())
    ratio = p_or / p_kd if p_kd > 0 else (math.nan if p_or == 0 else math.inf)
    j_route = dJ.mean >= gates.g2_min_dJ and dJ.lo > 0
    ping_route = (not math.isnan(ratio)) and ratio <= gates.g2_max_ping_ratio \
        and dJ.mean >= -gates.g2_J_tol
    route = "J" if j_route else ("pings" if ping_route else None)
    return {"dJ": dJ.to_dict(), "ping_ratio": ratio, "pings_oracle": p_or,
            "pings_kademlia": p_kd, "j_route": j_route, "ping_route": ping_route,
            "passed": bool(j_route or ping_route), "route": route}


@dataclass(frozen=True)
class GateResult:
    scenario: str
    candidates: tuple[CandidateStats, ...]
    best: str
    g1_passed: bool
    g1_route: str | None              # "J", "p95" (J preferred when both hold) or None
    g2: dict[str, Any]
    g2_info: dict[str, dict[str, Any]] = field(default_factory=dict)  # other scenarios
    thresholds: dict[str, float] = field(default_factory=dict)

    @property
    def g2_passed(self) -> bool:
        return bool(self.g2["passed"])

    @property
    def limited_headroom(self) -> bool:
        return not self.g2_passed

    def candidate(self, policy: str) -> CandidateStats:
        for c in self.candidates:
            if c.policy == policy:
                return c
        raise KeyError(policy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario, "thresholds": dict(self.thresholds),
            "G1": {"passed": self.g1_passed, "route": self.g1_route, "best": self.best,
                   "best_rule": "largest mean paired dJ vs never_replace",
                   "candidates": {c.policy: c.to_dict() for c in self.candidates}},
            "G2": {**self.g2, "limited_headroom": self.limited_headroom},
            "G2_info": self.g2_info,
        }


def evaluate_gates(df: pd.DataFrame, gates: GatesConfig | None = None, *,
                   scenario: str = GATE_SCENARIO, n_resamples: int = 10_000,
                   level: float = 0.95, seed: int = 0) -> GateResult:
    """Evaluate G1 / G2 on a per-seed results table (see module docstring).

    ``df`` needs ``policy, scenario, split, run, episode, J, p95_latency_all_ms,
    pings_per_hour`` rows for the four gate policies in ``scenario``. Other
    scenarios present in ``df`` with all gate policies get informational G2
    numbers in ``g2_info``.
    """
    gates = gates or GatesConfig()
    ci_kw = {"n_resamples": n_resamples, "level": level, "seed": seed}
    sub = df[df["scenario"] == scenario]
    if sub.empty:
        raise ValueError(f"no rows for gate scenario {scenario!r}")

    cands = []
    for pol in CHALLENGERS:
        dJ = paired_ci_from_frame(sub, "J", pol, BASELINE, **ci_kw)
        p95 = paired_ci_from_frame(sub, "p95_latency_all_ms", BASELINE, pol, **ci_kw)
        pf = paired_frame(sub, "p95_latency_all_ms", BASELINE, pol)
        base_v, pol_v = pf["a"].to_numpy(float), pf["b"].to_numpy(float)
        # Per-seed relative reduction; a seed with p95_nr == 0 contributes 0.
        safe = np.where(base_v > 0, base_v, 1.0)
        p95_rel = float(np.where(base_v > 0, (base_v - pol_v) / safe, 0.0).mean())
        cands.append(CandidateStats(
            policy=pol, dJ=dJ, p95_diff_ms=p95, p95_rel=p95_rel,
            j_route=bool(dJ.mean >= gates.g1_min_dJ and dJ.lo > 0),
            p95_route=bool(p95_rel >= gates.g1_min_p95_rel and p95.lo > 0)))
    best = max(cands, key=lambda c: (c.dJ.mean, -CHALLENGERS.index(c.policy)))
    route = "J" if best.j_route else ("p95" if best.p95_route else None)

    info = {}
    for scn in dict.fromkeys(df["scenario"]):
        if scn == scenario:
            continue
        s2 = df[df["scenario"] == scn]
        if {"oracle_heuristic", "kademlia_epoch"} <= set(s2["policy"]):
            info[scn] = _g2_stats(s2, gates, ci_kw)

    return GateResult(scenario=scenario, candidates=tuple(cands), best=best.policy,
                      g1_passed=route is not None, g1_route=route,
                      g2=_g2_stats(sub, gates, ci_kw), g2_info=info,
                      thresholds=dataclasses.asdict(gates))


# --------------------------------------------------------------------------- #
# Tuning fallback
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TunedLoader:
    """Picklable loader applying ``sim`` overrides on top of a ``ConfigLoader``."""

    base: Callable[[str], Config]
    overrides: tuple[tuple[str, Any], ...] = ()

    def __call__(self, scenario: str) -> Config:
        cfg = self.base(scenario)
        if not self.overrides:
            return cfg
        cfg = dataclasses.replace(cfg, sim=dataclasses.replace(cfg.sim, **dict(self.overrides)))
        validate(cfg)
        return cfg


def tuning_candidates(tuning: TuningConfig, default: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Grid settings ordered by closeness to ``default`` (default itself excluded).

    Key: (number of changed parameters, total |grid-index distance|, grid order).
    A default value missing from its grid counts as index distance from the
    nearest grid value.
    """
    grids = {f: tuple(getattr(tuning, f)) for f in TUNED_FIELDS}

    def idx(f: str, v: Any) -> float:
        g = grids[f]
        if default[f] in g:
            return float(g.index(v) - g.index(default[f]))
        nearest = min(range(len(g)), key=lambda i: abs(g[i] - default[f]))
        return float(g.index(v) - nearest) + 0.5

    combos = []
    for order, values in enumerate(itertools.product(*(grids[f] for f in TUNED_FIELDS))):
        setting = dict(zip(TUNED_FIELDS, values))
        changed = sum(setting[f] != default[f] for f in TUNED_FIELDS)
        if changed == 0:
            continue
        dist = sum(abs(idx(f, setting[f])) for f in TUNED_FIELDS if setting[f] != default[f])
        combos.append(((changed, dist, order), setting))
    return [s for _, s in sorted(combos, key=lambda c: c[0])]


@dataclass
class TuningResult:
    tried: list[dict[str, Any]]                  # per setting: overrides + G1 numbers
    chosen: dict[str, Any] | None                # first passing overrides, or None

    @property
    def passed(self) -> bool:
        return self.chosen is not None

    def to_dict(self) -> dict[str, Any]:
        return {"ran": True, "passed": self.passed, "chosen_overrides": self.chosen,
                "order": "fewest changed params, then grid-step distance, then grid order",
                "tried": self.tried}


def tune_env(loader: Callable[[str], Config], tuning: TuningConfig, gates: GatesConfig, *,
             n_seeds: int, split: Split, run: int = 0, stats: StatsConfig | None = None,
             progress: Callable[[str], None] | None = None) -> TuningResult:
    """G1-failure fallback: search the sim grid on high_churn calibration seeds."""
    stats = stats or StatsConfig()
    cfg0 = loader(GATE_SCENARIO)
    default = {f: getattr(cfg0.sim, f) for f in TUNED_FIELDS}
    tried: list[dict[str, Any]] = []
    for setting in tuning_candidates(tuning, default):
        tl = TunedLoader(loader, tuple(sorted(setting.items())))
        jobs = [Job(p, GATE_SCENARIO, split, run, ep)
                for p in GATE_POLICIES for ep in range(n_seeds)]
        df = results_table(run_jobs(jobs, cfg_loader=tl))
        res = evaluate_gates(df, gates, n_resamples=stats.bootstrap_resamples,
                             level=stats.ci_level)
        best = res.candidate(res.best)
        tried.append({"overrides": setting, "g1_passed": res.g1_passed, "route": res.g1_route,
                      "best": res.best, "dJ": best.dJ.mean, "dJ_lo": best.dJ.lo,
                      "p95_rel": best.p95_rel, "p95_diff_lo": best.p95_diff_ms.lo})
        if progress:
            progress(f"tuning {setting}: G1 {'PASS' if res.g1_passed else 'fail'} "
                     f"(best={res.best}, dJ={best.dJ.mean:+.4f}, p95_rel={best.p95_rel:+.1%})")
        if res.g1_passed:
            return TuningResult(tried, setting)
    return TuningResult(tried, None)


# --------------------------------------------------------------------------- #
# Study
# --------------------------------------------------------------------------- #

@dataclass
class HeadroomReport:
    results: list[EpisodeResult]
    df: pd.DataFrame
    summary: pd.DataFrame
    gates: GateResult
    tuning: TuningResult | None
    env_overrides: dict[str, Any] | None
    manifest: dict[str, Any]
    n_seeds: int
    split: Split

    @property
    def g1_passed(self) -> bool:
        """G1 passed on the default env, or the tuning fallback found a passing setting."""
        return self.gates.g1_passed or (self.tuning is not None and self.tuning.passed)

    @property
    def limited_headroom(self) -> bool:
        return self.gates.limited_headroom

    def gates_dict(self) -> dict[str, Any]:
        return {
            "split": self.split.name.lower(), "n_seeds": self.n_seeds,
            "gates": self.gates.to_dict(),
            "g1_passed_final": self.g1_passed,
            "limited_headroom": self.limited_headroom,
            "tuning": self.tuning.to_dict() if self.tuning is not None else {"ran": False},
            "env_overrides": self.env_overrides,
            "env_hashes": self.manifest.get("env_hashes", {}),
            "frozen": self.manifest.get("frozen", False),
            "informational_policies": list(INFO_POLICIES),
        }


def scenario_names(configs_dir: Path) -> list[str]:
    return sorted(p.stem for p in (Path(configs_dir) / "scenarios").glob("*.toml"))


def run_headroom(configs_dir: Path | str, experiment_path: Path | str | None = None, *,
                 n_seeds: int = 10, scenarios: Sequence[str] = DEFAULT_SCENARIOS,
                 policies: Sequence[str] = DEFAULT_POLICIES,
                 seed_split: Split = Split.CALIBRATION, run: int = 0,
                 manifest_path: Path | str | None = None,
                 progress: Callable[[str], None] | None = None) -> HeadroomReport:
    """Run the headroom study, the G1 fallback if needed, and freeze the env.

    ``manifest_path`` (default: none written) receives the frozen-env manifest.
    """
    configs_dir = Path(configs_dir)
    if GATE_SCENARIO not in scenarios:
        raise ValueError(f"scenarios must include {GATE_SCENARIO!r}")
    missing = [p for p in GATE_POLICIES if p not in policies]
    if missing:
        raise ValueError(f"policies must include the gate policies {missing}")
    loader = ConfigLoader(configs_dir, None if experiment_path is None else Path(experiment_path))
    cfg0 = loader(GATE_SCENARIO)
    exp = cfg0.experiment
    gates_cfg = exp.gates or GatesConfig()
    stats_cfg = exp.stats or StatsConfig()
    tuning_cfg = exp.tuning or TuningConfig()

    jobs = [Job(p, s, seed_split, run, ep)
            for s in scenarios for p in policies for ep in range(n_seeds)]
    if progress:
        progress(f"headroom: {len(jobs)} episodes ({len(policies)} policies x "
                 f"{len(scenarios)} scenarios x {n_seeds} seeds)")
    results = run_jobs(jobs, cfg_loader=loader)
    df = results_table(results)
    gate = evaluate_gates(df, gates_cfg, n_resamples=stats_cfg.bootstrap_resamples,
                          level=stats_cfg.ci_level)

    tuning: TuningResult | None = None
    overrides: dict[str, Any] | None = None
    if not gate.g1_passed:
        if progress:
            progress("G1 failed on the default env; running the tuning fallback")
        tuning = tune_env(loader, tuning_cfg, gates_cfg, n_seeds=n_seeds, split=seed_split,
                          run=run, stats=stats_cfg, progress=progress)
        overrides = tuning.chosen

    freeze_loader = TunedLoader(loader, tuple(sorted((overrides or {}).items())))
    all_scn = list(dict.fromkeys([*scenario_names(configs_dir), *scenarios]))
    cfgs = {s: freeze_loader(s) for s in all_scn}
    g1_final = gate.g1_passed or (tuning is not None and tuning.passed)
    extra = {
        "experiment": "headroom", "split": seed_split.name.lower(), "run": run,
        "n_seeds": n_seeds, "episodes": list(range(n_seeds)),
        "scenarios": list(scenarios), "policies": list(policies),
        "env_hash_scope": "sim + scenario sections + root_seed",
        "env_overrides": overrides,
        "g1_passed": g1_final, "g1_route": gate.g1_route, "g1_best": gate.best,
        "g2_passed": gate.g2_passed, "limited_headroom": gate.limited_headroom,
        "tuning_ran": tuning is not None,
    }
    if manifest_path is not None:
        manifest = freeze_env(manifest_path, cfgs, extra=extra, frozen=g1_final)
    else:
        manifest = {**extra, "frozen": g1_final,
                    "env_hashes": {s: env_config_hash(c) for s, c in sorted(cfgs.items())}}
    return HeadroomReport(results=results, df=df, summary=summary_by_policy(df), gates=gate,
                          tuning=tuning, env_overrides=overrides, manifest=manifest,
                          n_seeds=n_seeds, split=seed_split)


def _json_safe(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return None if math.isnan(v) else ("inf" if math.isinf(v) else v)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def write_outputs(report: HeadroomReport, out_dir: Path | str) -> list[Path]:
    """Write per_seed.csv, summary.csv, gates.json (+ tuning.csv) and figures."""
    from churnmaster.plots import plot_dead_fraction, plot_headroom_bars, plot_success_vs_pings

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    report.df.to_csv(out / "per_seed.csv", index=False)
    report.summary.to_csv(out / "summary.csv", index=False)
    (out / "gates.json").write_text(json.dumps(_json_safe(report.gates_dict()), indent=2) + "\n",
                                    encoding="utf-8")
    written += [out / "per_seed.csv", out / "summary.csv", out / "gates.json"]
    if report.tuning is not None:
        rows = [{**t["overrides"], **{k: v for k, v in t.items() if k != "overrides"}}
                for t in report.tuning.tried]
        pd.DataFrame(rows).to_csv(out / "tuning.csv", index=False)
        written.append(out / "tuning.csv")
    written.append(plot_headroom_bars(report.df, out / "headroom_J_p95.png"))
    written.append(plot_success_vs_pings(report.df, out / "success_vs_pings.png"))
    for scn in dict.fromkeys(report.df["scenario"]):
        series = []
        for pol in dict.fromkeys(report.df["policy"]):
            eps = [r.metrics.dead_entry_fraction for r in report.results
                   if r.scenario == scn and r.policy == pol]
            if eps:
                series.append((pol, eps))
        written.append(plot_dead_fraction(series, out / f"dead_fraction_{scn}.png",
                                          title=f"Dead-entry fraction over time ({scn}, GT)"))
    return written


def manifest_path_for(out_dir: Path | str) -> Path:
    return Path(out_dir) / MANIFEST_NAME
