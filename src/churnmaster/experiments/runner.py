"""Job runner and policy factory (Requirements 11.5, 21.1, 26.1).

A *job* is one ``(policy, scenario, split, run, episode)`` episode. Jobs are
plain picklable dataclasses (the policy is normally given by its name), so
the sequential :func:`run_jobs` can later be swapped for a process pool
(task 8.4) with identical results: every job builds its own config, env and
policy from scratch and depends on nothing but its own fields.

Policy names (:func:`make_policy`)
----------------------------------
* baselines: ``never_replace``, ``always_replace``, ``uniform_random``,
  ``matched_random`` (needs ``freqs``), ``kademlia_epoch``, ``kademlia_std``
  (runs the env in ``mode="kademlia_std"`` and is labeled
  ``shares_decision_points=False``);
* ``heuristic`` (``policies.heuristic.DEFAULT_CONFIG``) and
  ``heuristic:<config id>`` (``HeuristicConfig.from_id``);
* ``oracle_heuristic`` (needs the running ``Env``; ground truth is read via
  ``Env.oracle_facts``) -- headroom reference only, NOT an upper bound;
* ``none``: maintenance off (no decision points at all).

Extension point: a ready ``Policy`` object may be passed instead of a name
(learned policies, task 8.2). Optional attributes honored on any policy:
``env_mode`` (default ``"epoch"``) and ``maintenance`` (default True).

Random baseline streams
-----------------------
Random baselines draw from the AGENT split only. ``POLICY_IDS`` fixes the
policy id per name (uniform_random=0, matched_random=1). ``make_policy``
without an explicit ``rng`` uses ``seeding.baseline_rng(root, run, id)``.
``run_job`` instead gives every job its own child stream,
``SeedSequence(root, spawn_key=(AGENT, run, 1000 + id, split, episode))``,
i.e. the ``baseline_rng`` seed extended by ``(split, episode)``, so episodes do
not reuse the same random draws and results do not depend on job order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from churnmaster.config import Config, config_hash, load_config
from churnmaster.metrics import SCALAR_FIELDS, EpisodeMetrics, summarize, summary_hash
from churnmaster.policies.base import Policy
from churnmaster.policies.baselines import BASELINE_NAMES, RANDOM_BASELINES, make_baseline
from churnmaster.policies.heuristic import DEFAULT_CONFIG, Heuristic, HeuristicConfig
from churnmaster.policies.oracle import OracleFacts, OracleHeuristic
from churnmaster.rl.state import discretize, encode
from churnmaster.seeding import BASELINE_EPISODE_OFFSET, Split, baseline_rng, env_seed, \
    episode_seed, make_streams
from churnmaster.sim.env import EpisodeTrace, Env
from churnmaster.sim.observation import DecisionContext, DecisionRecord, GTRecord, ObservedLookup

__all__ = [
    "ENV_SPLITS", "POLICY_IDS", "POLICY_NAMES", "DEFAULT_COMPARE_POLICIES",
    "NoMaintenance", "ConfigLoader", "Job", "EpisodeResult",
    "parse_split", "policy_env_mode", "make_policy", "job_baseline_rng",
    "run_job", "run_jobs", "run_episode", "results_table", "summary_by_policy",
]

ENV_SPLITS = ("calibration", "train", "validation", "eval", "test")

#: Fixed AGENT-split policy ids of the random baselines (never reorder).
POLICY_IDS: dict[str, int] = {"uniform_random": 0, "matched_random": 1}

POLICY_NAMES: tuple[str, ...] = BASELINE_NAMES + ("heuristic", "oracle_heuristic", "none")

DEFAULT_COMPARE_POLICIES: tuple[str, ...] = (
    "never_replace", "always_replace", "kademlia_epoch", "kademlia_std",
    "heuristic", "oracle_heuristic", "uniform_random",
)

_HEURISTIC_PREFIX = "heuristic:"


def parse_split(name: str) -> Split:
    """Map an environment split name (case-insensitive) to ``Split``.

    ``agent`` is rejected: it seeds exploration, never environments.
    """
    key = name.strip().lower()
    if key not in ENV_SPLITS:
        raise ValueError(f"unknown environment split {name!r} (expected one of {', '.join(ENV_SPLITS)})")
    return Split[key.upper()]


# --------------------------------------------------------------------------- #
# Policies
# --------------------------------------------------------------------------- #

class NoMaintenance:
    """Maintenance off: the env runs no decision points (policy name ``none``)."""

    name = "none"
    shares_decision_points = True
    maintenance = False

    def decide(self, ctx: DecisionContext):
        raise RuntimeError("NoMaintenance never decides")


def _check_name(name: str) -> None:
    if name in POLICY_NAMES:
        return
    if name.startswith(_HEURISTIC_PREFIX):
        HeuristicConfig.from_id(name[len(_HEURISTIC_PREFIX):])  # ValueError if malformed
        return
    raise ValueError(f"unknown policy {name!r}; expected one of {POLICY_NAMES} "
                     f"or 'heuristic:<config id>'")


def policy_env_mode(policy: str | Policy) -> str:
    """Env mode a policy needs: ``"kademlia_std"`` for that baseline, else ``"epoch"``."""
    if isinstance(policy, str):
        _check_name(policy)
        return "kademlia_std" if policy == "kademlia_std" else "epoch"
    return getattr(policy, "env_mode", "epoch")


def make_policy(name: str, cfg: Config, run: int = 0, env: Env | None = None,
                freqs: Mapping[bool, Sequence[float]] | None = None,
                rng: np.random.Generator | None = None) -> Policy:
    """Build a policy by name (see module docstring).

    ``rng`` overrides the random-baseline generator (default
    ``baseline_rng(cfg.root_seed, run, POLICY_IDS[name])``). ``env`` is
    required for ``oracle_heuristic``.
    """
    _check_name(name)
    if name == "none":
        return NoMaintenance()
    if name in BASELINE_NAMES:
        if name in RANDOM_BASELINES and rng is None:
            rng = baseline_rng(cfg.root_seed, run, POLICY_IDS[name])
        return make_baseline(name, rng=rng, freqs=freqs)
    if name == "heuristic":
        return Heuristic(DEFAULT_CONFIG, id_bits=cfg.sim.id_bits)
    if name.startswith(_HEURISTIC_PREFIX):
        hc = HeuristicConfig.from_id(name[len(_HEURISTIC_PREFIX):])
        return Heuristic(hc, id_bits=cfg.sim.id_bits)
    # oracle_heuristic
    if env is None:
        raise ValueError("policy 'oracle_heuristic' requires the env")
    return OracleHeuristic(lambda b, t: OracleFacts(*env.oracle_facts(b, t)))


def job_baseline_rng(root: int, run: int, policy_id: int, split: Split,
                     episode: int) -> np.random.Generator:
    """Per-job random-baseline stream: the ``baseline_rng`` seed + ``(split, episode)``."""
    ss = episode_seed(root, Split.AGENT, run, BASELINE_EPISODE_OFFSET + int(policy_id))
    return np.random.default_rng(
        np.random.SeedSequence(ss.entropy, spawn_key=tuple(ss.spawn_key) + (int(split), int(episode))))


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ConfigLoader:
    """Picklable ``scenario -> Config`` loader: base -> scenario -> experiment."""

    configs_dir: Path
    experiment: Path | None = None

    def __call__(self, scenario: str) -> Config:
        root = Path(self.configs_dir)
        return load_config(root / "base.toml", root / "scenarios" / f"{scenario}.toml",
                           None if self.experiment is None else Path(self.experiment))


@dataclass(frozen=True)
class Job:
    policy: str | Policy
    scenario: str
    split: Split
    run: int = 0
    episode: int = 0
    policy_kw: Mapping[str, Any] = field(default_factory=dict)  # e.g. freqs for matched_random

    @property
    def policy_name(self) -> str:
        return self.policy if isinstance(self.policy, str) else self.policy.name


@dataclass
class EpisodeResult:
    policy: str
    scenario: str
    split: Split
    run: int
    episode: int
    shares_decision_points: bool
    metrics: EpisodeMetrics
    trace: EpisodeTrace
    config_hash: str
    lookups: list[ObservedLookup] | None = None    # only with keep_records
    decisions: list[DecisionRecord] | None = None
    gts: list[GTRecord] | None = None

    def row(self) -> dict[str, Any]:
        """Flat scalar row for :func:`results_table`."""
        return {
            "policy": self.policy, "scenario": self.scenario, "split": self.split.name.lower(),
            "run": self.run, "episode": self.episode,
            "shares_decision_points": self.shares_decision_points,
            **self.metrics.to_dict(diagnostics=False),
            "trace_hash": self.trace.trace_hash, "workload_hash": self.trace.workload_hash,
            "summary_hash": summary_hash(self.metrics), "config_hash": self.config_hash,
        }


def _job_config(job: Job, cfg: Config | None, cfg_loader: Callable[[str], Config] | None,
                configs_dir: Path | None) -> Config:
    if cfg is not None:
        if cfg.scenario.name != job.scenario:
            raise ValueError(f"config scenario {cfg.scenario.name!r} != job scenario {job.scenario!r}")
        return cfg
    if cfg_loader is None:
        if configs_dir is None:
            raise ValueError("run_job needs cfg, cfg_loader or configs_dir")
        cfg_loader = ConfigLoader(Path(configs_dir))
    return cfg_loader(job.scenario)


def run_job(job: Job, configs_dir: Path | None = None, *,
            cfg_loader: Callable[[str], Config] | None = None, cfg: Config | None = None,
            keep_records: bool = False) -> EpisodeResult:
    """Run one job: build config, env and policy, run the episode, summarize.

    The config comes from ``cfg`` (explicit override, must match the job's
    scenario), else ``cfg_loader(job.scenario)``, else ``configs_dir``.
    """
    cfg = _job_config(job, cfg, cfg_loader, configs_dir)
    mode = policy_env_mode(job.policy)
    if isinstance(job.policy, str):
        maintenance = job.policy != "none"
    else:
        maintenance = bool(getattr(job.policy, "maintenance", True))
    env = Env(cfg, make_streams(env_seed(cfg.root_seed, job.split, job.run, job.episode)),
              mode=mode, maintenance=maintenance)
    if isinstance(job.policy, str):
        kw = dict(job.policy_kw)
        if job.policy in RANDOM_BASELINES and "rng" not in kw:
            kw["rng"] = job_baseline_rng(cfg.root_seed, job.run, POLICY_IDS[job.policy],
                                         job.split, job.episode)
        policy = make_policy(job.policy, cfg, run=job.run, env=env, **kw)
    else:
        policy = job.policy

    id_bits = cfg.sim.id_bits
    lookups: list[ObservedLookup] = []
    decisions: list[DecisionRecord] = []
    gts: list[GTRecord] = []
    trace = env.run(policy.decide if maintenance else None, decisions.append, lookups.append,
                    gt_annex=gts.append,
                    encode_state=(lambda ctx: encode(discretize(ctx, id_bits))) if maintenance else None)
    metrics = summarize(lookups, decisions, gts, cfg)
    return EpisodeResult(
        policy=job.policy_name, scenario=job.scenario, split=Split(job.split), run=job.run,
        episode=job.episode, shares_decision_points=bool(policy.shares_decision_points),
        metrics=metrics, trace=trace, config_hash=config_hash(cfg),
        lookups=lookups if keep_records else None,
        decisions=decisions if keep_records else None,
        gts=gts if keep_records else None)


def run_jobs(jobs: Iterable[Job], configs_dir: Path | None = None, *,
             cfg_loader: Callable[[str], Config] | None = None,
             keep_records: bool = False) -> list[EpisodeResult]:
    """Run jobs sequentially; results are in job order."""
    if cfg_loader is None and configs_dir is not None:
        cfg_loader = ConfigLoader(Path(configs_dir))
    cache: dict[str, Config] = {}

    def load(scenario: str) -> Config:
        if scenario not in cache:
            if cfg_loader is None:
                raise ValueError("run_jobs needs cfg_loader or configs_dir")
            cache[scenario] = cfg_loader(scenario)
        return cache[scenario]

    return [run_job(j, cfg=load(j.scenario), keep_records=keep_records) for j in jobs]


def run_episode(cfg: Config, split: Split, run: int = 0, episode: int = 0,
                maintenance: bool = False,
                policy: str | Policy | None = None) -> tuple[EpisodeMetrics, EpisodeTrace]:
    """Run one episode under ``cfg`` and return its metrics and trace summary.

    ``maintenance=False`` runs policy ``none``; ``maintenance=True`` needs a
    ``policy`` (name or object).
    """
    if not maintenance:
        policy = "none"
    elif policy is None or policy == "none":
        raise ValueError("maintenance-on episodes need a policy")
    res = run_job(Job(policy, cfg.scenario.name, split, run, episode), cfg=cfg)
    return res.metrics, res.trace


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #

KEY_COLUMNS = ("policy", "scenario", "split", "run", "episode")
METRIC_COLUMNS = tuple(f for f in SCALAR_FIELDS if f not in ("n_lookups", "n_decisions"))


def results_table(results: Iterable[EpisodeResult]) -> pd.DataFrame:
    """One row per (policy, scenario, split, run, episode) with scalar metrics."""
    rows = [r.row() for r in results]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if df.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("duplicate (policy, scenario, split, run, episode) rows")
    return df


def summary_by_policy(df: pd.DataFrame,
                      metrics: Sequence[str] = METRIC_COLUMNS) -> pd.DataFrame:
    """Mean / std (ddof=1) / n per (policy, scenario), in first-appearance order.

    Columns: ``policy, scenario, shares_decision_points, n, <m>_mean, <m>_std``.
    """
    out = []
    for (pol, scn), g in df.groupby(["policy", "scenario"], sort=False):
        row: dict[str, Any] = {"policy": pol, "scenario": scn,
                               "shares_decision_points": bool(g["shares_decision_points"].all()),
                               "n": len(g)}
        for m in metrics:
            vals = g[m].astype(float)
            row[f"{m}_mean"] = float(vals.mean())
            row[f"{m}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else math.nan
        out.append(row)
    return pd.DataFrame(out)
