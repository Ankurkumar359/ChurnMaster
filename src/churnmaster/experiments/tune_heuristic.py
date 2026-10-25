"""Heuristic tuning (Task 9.1, Requirement 25.1).

All 32 heuristic configurations (``policies.heuristic.all_configs()``) are run
on ``n_seeds`` (default 10) TRAIN-split episodes per tuning scenario (default
stable + high_churn).

Seeds
-----
Tuning uses the TRAIN split (disjoint from VALIDATION and EVAL by
construction of ``seeding.env_seed``), with run index :data:`TUNING_RUN`
(1000) and episodes ``0..n_seeds-1``. RL training uses TRAIN runs 0..2, so
the run index 1000 keeps tuning episodes distinct from every training
episode while staying on the training split ("tuned on the training split").

Selection (:func:`select`)
--------------------------
* Per scenario: mean J over that scenario's seeds.
* Overall J of a config: the unweighted mean of its per-scenario mean J
  (equal weight per scenario, independent of the number of rows per
  scenario).
* Global best ("tuned_heuristic"): highest overall J.
* Per-scenario best ("tuned_heuristic_privileged"): highest mean J within
  that scenario. Labeled privileged: it is selected on the scenario it is
  evaluated on, so it is a reference, not a fair competitor.
* Ties (|ΔJ| <= :data:`TIE_TOL`): ``DEFAULT_CONFIG`` first, then
  ``all_configs()`` order.

Outputs (:func:`write_outputs`): ``per_seed.csv``, ``summary.csv`` and
``selected.json``; :func:`load_selected` / :func:`tuned_policy_names` read
the selection back for evaluation (task 9.3).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from churnmaster.experiments.runner import (Job, check_frozen, results_table, run_jobs)
from churnmaster.policies.heuristic import DEFAULT_CONFIG, HeuristicConfig, all_configs
from churnmaster.seeding import Split

__all__ = [
    "TUNING_RUN", "TIE_TOL", "DEFAULT_SCENARIOS", "SUMMARY_METRICS", "GLOBAL_LABEL",
    "PRIVILEGED_LABEL", "Selection", "TuningReport", "config_priority", "select",
    "run_tuning", "write_outputs", "load_selected", "tuned_policy_names",
]

TUNING_RUN = 1000
TIE_TOL = 1e-12
DEFAULT_SCENARIOS: tuple[str, ...] = ("stable", "high_churn")
DEFAULT_N_SEEDS = 10
SUMMARY_METRICS: tuple[str, ...] = ("success_rate", "p95_latency_all_ms", "pings_per_hour")
GLOBAL_LABEL = "tuned_heuristic"
PRIVILEGED_LABEL = "tuned_heuristic_privileged"
_PREFIX = "heuristic:"


def _config_id(policy: str) -> str:
    if policy == "heuristic":
        return DEFAULT_CONFIG.id
    return policy[len(_PREFIX):] if policy.startswith(_PREFIX) else policy


def config_priority() -> list[str]:
    """Tie-break order: DEFAULT_CONFIG first, then ``all_configs()`` order."""
    ids = [c.id for c in all_configs()]
    return [DEFAULT_CONFIG.id] + [i for i in ids if i != DEFAULT_CONFIG.id]


def _pick(values: Mapping[str, float], order: Sequence[str]) -> str:
    """Highest value; ties within TIE_TOL resolved by ``order``."""
    best = max(values.values())
    for cid in order:
        if cid in values and values[cid] >= best - TIE_TOL:
            return cid
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class Selection:
    global_id: str
    per_scenario: dict[str, str]
    scenarios: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "global": self.global_id,
            "global_policy": _PREFIX + self.global_id,
            "global_label": GLOBAL_LABEL,
            "per_scenario": dict(self.per_scenario),
            "privileged": [{"scenario": s, "config": c, "policy": _PREFIX + c,
                            "label": PRIVILEGED_LABEL, "privileged": True}
                           for s, c in self.per_scenario.items()],
            "scenarios": list(self.scenarios),
        }


def select(df: pd.DataFrame) -> tuple[pd.DataFrame, Selection]:
    """Selection on a per-seed table (columns ``policy, scenario, J`` + metrics).

    Returns the ranked summary table (one row per config) and the selection.
    Every config must have rows in every scenario present in ``df``.
    """
    if df.empty:
        raise ValueError("select needs a non-empty per-seed table")
    d = df.assign(config=df["policy"].map(_config_id))
    scenarios = tuple(dict.fromkeys(d["scenario"]))
    known = config_priority()
    present = list(dict.fromkeys(d["config"]))
    order = [c for c in known if c in present] + [c for c in present if c not in known]
    metrics = [m for m in SUMMARY_METRICS if m in d.columns]

    g = d.groupby(["config", "scenario"])
    counts = g.size()
    for cid in order:
        missing = [s for s in scenarios if (cid, s) not in counts.index]
        if missing:
            raise ValueError(f"config {cid!r} has no rows for scenario(s) {missing}")
    mean_j = g["J"].mean()
    std_j = g["J"].std(ddof=1)
    means = {m: g[m].mean() for m in metrics}

    rows = []
    for cid in order:
        row: dict[str, Any] = {"config": cid, "policy": _PREFIX + cid,
                               "is_default": cid == DEFAULT_CONFIG.id}
        per = [float(mean_j[(cid, s)]) for s in scenarios]
        row["J_overall"] = float(sum(per) / len(per))
        for m in metrics:
            row[f"{m}_overall"] = float(sum(float(means[m][(cid, s)]) for s in scenarios)
                                        / len(scenarios))
        for s in scenarios:
            row[f"J_mean_{s}"] = float(mean_j[(cid, s)])
            sd = std_j[(cid, s)]
            row[f"J_std_{s}"] = float(sd) if not pd.isna(sd) else math.nan
            row[f"n_{s}"] = int(counts[(cid, s)])
            for m in metrics:
                row[f"{m}_mean_{s}"] = float(means[m][(cid, s)])
        rows.append(row)
    summary = pd.DataFrame(rows)

    overall = dict(zip(summary["config"], summary["J_overall"]))
    global_id = _pick(overall, order)
    per_scenario = {s: _pick(dict(zip(summary["config"], summary[f"J_mean_{s}"])), order)
                    for s in scenarios}

    # Rank: overall J descending; ties (within TIE_TOL) by priority order.
    remaining = dict(overall)
    ranked: list[str] = []
    while remaining:
        cid = _pick(remaining, order)
        ranked.append(cid)
        del remaining[cid]
    rank = {cid: i + 1 for i, cid in enumerate(ranked)}
    summary.insert(0, "rank", summary["config"].map(rank))
    summary["global_best"] = summary["config"] == global_id
    for s in scenarios:
        summary[f"privileged_best_{s}"] = summary["config"] == per_scenario[s]
    summary = summary.sort_values("rank", kind="stable").reset_index(drop=True)
    return summary, Selection(global_id, per_scenario, scenarios)


@dataclass
class TuningReport:
    per_seed: pd.DataFrame
    summary: pd.DataFrame
    selection: Selection
    seeds: dict[str, Any]
    env_hashes: dict[str, str]
    wall_s: float = 0.0
    n_jobs: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def selected_json(self) -> dict[str, Any]:
        return {**self.selection.to_dict(), "seeds": self.seeds, "env_hashes": self.env_hashes,
                "n_configs": int(len(self.summary)), "n_jobs": self.n_jobs,
                "selection_rule": ("global: max of the unweighted mean over scenarios of "
                                   "per-scenario mean J; per-scenario: max mean J in that "
                                   f"scenario (privileged); ties |dJ|<={TIE_TOL}: "
                                   "DEFAULT_CONFIG first, then all_configs() order"),
                "default_config": DEFAULT_CONFIG.id,
                "default_rank": int(self.summary.loc[self.summary["config"] == DEFAULT_CONFIG.id,
                                                     "rank"].iloc[0])
                if (self.summary["config"] == DEFAULT_CONFIG.id).any() else None,
                "wall_s": round(self.wall_s, 1)}


def run_tuning(configs_dir: Path | str, manifest_path: Path | str,
               experiment_path: Path | str | None = None, n_seeds: int | None = None,
               scenarios: Sequence[str] | None = None, split: Split = Split.TRAIN,
               run: int = TUNING_RUN, workers: int | None = None,
               configs: Sequence[HeuristicConfig] | None = None,
               progress: Callable[[str], None] | None = None) -> TuningReport:
    """Evaluate every heuristic config and select (see module docstring).

    ``n_seeds`` / ``scenarios`` default to the experiment file's values
    (``configs/experiments/tune.toml``), else 10 / (stable, high_churn).
    Raises ``FrozenEnvError`` before running anything if any scenario's env
    differs from the frozen manifest.
    """
    from churnmaster.experiments.diagnostics import _frozen_loader

    configs_dir = Path(configs_dir)
    manifest_path = Path(manifest_path)
    if experiment_path is None:
        experiment_path = configs_dir / "experiments" / "tune.toml"
    loader = _frozen_loader(configs_dir, Path(experiment_path), manifest_path)
    first = loader((tuple(scenarios) if scenarios else DEFAULT_SCENARIOS)[0])
    exp = first.experiment
    if scenarios is None:
        scenarios = tuple(exp.scenarios) or DEFAULT_SCENARIOS
    if n_seeds is None:
        n_seeds = exp.n_seeds or DEFAULT_N_SEEDS
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be >= 1, got {n_seeds}")
    scenarios = tuple(scenarios)
    env_hashes = {s: check_frozen(manifest_path, loader(s)) for s in scenarios}

    cfgs = list(configs) if configs is not None else all_configs()
    jobs = [Job(_PREFIX + c.id, s, split, run, ep)
            for s in scenarios for c in cfgs for ep in range(n_seeds)]
    if progress:
        progress(f"tune: {len(cfgs)} configs x {len(scenarios)} scenarios x {n_seeds} seeds "
                 f"= {len(jobs)} episodes (workers={workers})")
    t0 = time.perf_counter()
    df = results_table(run_jobs(jobs, cfg_loader=loader, workers=workers))
    wall = time.perf_counter() - t0
    summary, selection = select(df)
    seeds = {"split": split.name.lower(), "run": int(run), "episodes": list(range(n_seeds)),
             "n_seeds": int(n_seeds), "scenarios": list(scenarios),
             "note": ("TRAIN split (disjoint from VALIDATION and EVAL); run index "
                      f"{run} keeps tuning episodes distinct from RL training runs 0..2")}
    return TuningReport(df, summary, selection, seeds, env_hashes, wall_s=wall, n_jobs=len(jobs))


def write_outputs(rep: TuningReport, out_dir: Path | str) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = [out / "per_seed.csv", out / "summary.csv", out / "selected.json"]
    rep.per_seed.to_csv(files[0], index=False)
    rep.summary.to_csv(files[1], index=False)
    files[2].write_text(json.dumps(rep.selected_json(), indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    return files


def load_selected(path: Path | str) -> dict[str, Any]:
    """Read ``selected.json``; validates that every selected id parses."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "global" not in data or "per_scenario" not in data:
        raise ValueError(f"{path}: not a tuning selection (needs 'global' and 'per_scenario')")
    HeuristicConfig.from_id(data["global"])
    for cid in data["per_scenario"].values():
        HeuristicConfig.from_id(cid)
    return data


def tuned_policy_names(selected: Mapping[str, Any]) -> dict[str, Any]:
    """``{"tuned_heuristic": "heuristic:<global>", "tuned_heuristic_privileged": {scn: name}}``."""
    return {GLOBAL_LABEL: _PREFIX + selected["global"],
            PRIVILEGED_LABEL: {s: _PREFIX + c for s, c in selected["per_scenario"].items()}}
