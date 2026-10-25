"""Job runner and policy factory (Requirements 11.5, 21.1, 26.1).

A *job* is one ``(policy, scenario, split, run, episode)`` episode. Jobs are
plain picklable dataclasses (the policy is normally given by its name), so
:func:`run_jobs` can run them in a process pool (``workers > 1``, task 8.4)
with results identical to sequential execution (Requirement 21.3): every job
builds its own config, env and policy from scratch and depends on nothing
but its own fields.

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
import multiprocessing
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
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
from churnmaster.rl.ledger import Ledger, Transition
from churnmaster.rl.state import discretize, encode
from churnmaster.seeding import BASELINE_EPISODE_OFFSET, Split, baseline_rng, env_seed, \
    episode_seed, make_streams
from churnmaster.sim.env import EpisodeTrace, Env
from churnmaster.sim.observation import DecisionContext, DecisionRecord, GTRecord, ObservedLookup

__all__ = [
    "ENV_SPLITS", "POLICY_IDS", "POLICY_NAMES", "DEFAULT_COMPARE_POLICIES",
    "NoMaintenance", "ConfigLoader", "Job", "EpisodeResult",
    "parse_split", "policy_env_mode", "make_policy", "job_baseline_rng",
    "run_job", "run_jobs", "default_workers", "run_episode", "results_table",
    "summary_by_policy",
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
    ledger_score: float = math.nan                 # Ledger.score() (== metrics.J, Req. 15.5)
    n_transitions: int = 0                         # τ=1 transitions emitted by the Ledger
    lookups: list[ObservedLookup] | None = None    # only with keep_records
    decisions: list[DecisionRecord] | None = None
    gts: list[GTRecord] | None = None
    transitions: list[Transition] | None = None
    ledger: Ledger | None = None                   # only with keep_records (per-(b,k) rewards)

    def row(self) -> dict[str, Any]:
        """Flat scalar row for :func:`results_table`."""
        return {
            "policy": self.policy, "scenario": self.scenario, "split": self.split.name.lower(),
            "run": self.run, "episode": self.episode,
            "shares_decision_points": self.shares_decision_points,
            **self.metrics.to_dict(diagnostics=False),
            "trace_hash": self.trace.trace_hash, "workload_hash": self.trace.workload_hash,
            "summary_hash": summary_hash(self.metrics), "config_hash": self.config_hash,
            "ledger_score": self.ledger_score, "n_transitions": self.n_transitions,
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
            keep_records: bool = False,
            on_transition: Callable[[Transition], None] | None = None) -> EpisodeResult:
    """Run one job: build config, env and policy, run the episode, summarize.

    The config comes from ``cfg`` (explicit override, must match the job's
    scenario), else ``cfg_loader(job.scenario)``, else ``configs_dir``.

    Every observable record (lookups and decision/cost records, never GT
    records) is fed to a :class:`~churnmaster.rl.ledger.Ledger` in env order;
    ``on_transition`` (if given) receives each τ=1 transition as soon as the
    ledger emits it. After summarizing, ``metrics.J == ledger.score()`` is
    checked (Requirement 15.5); a mismatch raises ``RuntimeError``.
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
    ledger = Ledger.from_config(cfg)
    n_tr = 0

    def feed(rec: ObservedLookup | DecisionRecord) -> None:
        nonlocal n_tr
        closed = ledger.add(rec)
        n_tr += len(closed)
        if on_transition is not None:
            for tr in closed:
                on_transition(tr)

    def on_decision(rec: DecisionRecord) -> None:
        decisions.append(rec)
        feed(rec)

    def on_lookup(rec: ObservedLookup) -> None:
        lookups.append(rec)
        feed(rec)

    trace = env.run(policy.decide if maintenance else None, on_decision, on_lookup,
                    gt_annex=gts.append,
                    encode_state=(lambda ctx: encode(discretize(ctx, id_bits))) if maintenance else None)
    ledger.finish()
    metrics = summarize(lookups, decisions, gts, cfg)
    score = ledger.score()
    _check_identity(metrics.J, score, job)
    return EpisodeResult(
        policy=job.policy_name, scenario=job.scenario, split=Split(job.split), run=job.run,
        episode=job.episode, shares_decision_points=bool(policy.shares_decision_points),
        metrics=metrics, trace=trace, config_hash=config_hash(cfg),
        ledger_score=score, n_transitions=n_tr,
        lookups=lookups if keep_records else None,
        decisions=decisions if keep_records else None,
        gts=gts if keep_records else None,
        transitions=list(ledger.transitions) if keep_records else None,
        ledger=ledger if keep_records else None)


_IDENTITY_TOL = 1e-9


def _check_identity(J: float, score: float, job: Job) -> None:
    """Requirement 15.5: ``Metrics.J == Ledger.score()`` (both NaN allowed)."""
    if math.isnan(J) and math.isnan(score):
        return
    if math.isnan(J) or math.isnan(score) or abs(J - score) > _IDENTITY_TOL:
        raise RuntimeError(
            f"J identity violated for {job.policy_name}/{job.scenario}/{Split(job.split).name.lower()}"
            f"/run={job.run}/ep={job.episode}: metrics.J={J!r} != ledger.score()={score!r}")


def default_workers() -> int:
    """Default pool size: ``min(cpu_count − 1, 8)``, at least 1."""
    return max(1, min((os.cpu_count() or 1) - 1, 8))


#: Per-process config cache of the pool workers, keyed by ``(loader, scenario)``.
_WORKER_CFG_CACHE: dict[tuple[Any, str], Config] = {}


def _worker_config(cfg_loader: Callable[[str], Config], scenario: str) -> Config:
    key = (cfg_loader, scenario)
    try:
        hash(key)
    except TypeError:          # unhashable loader: no cache
        return cfg_loader(scenario)
    cfg = _WORKER_CFG_CACHE.get(key)
    if cfg is None:
        cfg = _WORKER_CFG_CACHE[key] = cfg_loader(scenario)
    return cfg


def _run_job_worker(args: tuple[Job, Callable[[str], Config], bool]) -> EpisodeResult:
    """Pool entry point (module level, so it pickles under ``spawn``)."""
    job, cfg_loader, keep_records = args
    return run_job(job, cfg=_worker_config(cfg_loader, job.scenario), keep_records=keep_records)


def _check_picklable(cfg_loader: Callable[[str], Config], jobs: Sequence[Job]) -> None:
    """Fail early with a clear message instead of a pool-side pickling error."""
    try:
        pickle.dumps(cfg_loader)
    except Exception as e:  # noqa: BLE001 -- PicklingError, AttributeError, TypeError, ...
        raise TypeError(f"run_jobs(workers>1) needs a picklable cfg_loader (e.g. ConfigLoader "
                        f"or TunedLoader, not a lambda/closure): {e}") from e
    seen: set[int] = set()
    for j in jobs:
        if isinstance(j.policy, str) or id(j.policy) in seen:
            continue
        seen.add(id(j.policy))
        try:
            pickle.dumps(j)
        except Exception as e:  # noqa: BLE001
            raise TypeError(f"run_jobs(workers>1): job with policy {j.policy_name!r} is not "
                            f"picklable; pass the policy by name or use a picklable policy "
                            f"object (no lambdas/closures): {e}") from e


def run_jobs(jobs: Iterable[Job], configs_dir: Path | None = None, *,
             cfg_loader: Callable[[str], Config] | None = None,
             keep_records: bool = False, workers: int | None = None,
             chunksize: int = 1) -> list[EpisodeResult]:
    """Run jobs; results are in job order (Requirement 21.3).

    ``workers`` None/0/1 runs sequentially in this process. ``workers > 1``
    uses a ``ProcessPoolExecutor`` with the ``spawn`` start method (the
    Windows default; also used elsewhere for identical behavior); results
    come back in job order (``executor.map``) and are identical to the
    sequential ones, because every job depends only on its own fields (env
    and random-baseline streams are seeded per job). Each worker process
    loads configs via ``cfg_loader`` and caches them per ``(loader,
    scenario)``. With ``workers > 1`` the loader and any ``Policy`` objects
    in ``jobs`` must be picklable (``TypeError`` otherwise; policy names are
    preferred), and per-transition hooks (``run_job(on_transition=...)``)
    are not available. Callers using workers must guard their entry point
    with ``if __name__ == "__main__":`` (spawn re-imports ``__main__``).
    """
    if cfg_loader is None and configs_dir is not None:
        cfg_loader = ConfigLoader(Path(configs_dir))
    jobs = list(jobs)
    if workers is not None and workers < 0:
        raise ValueError(f"workers must be >= 0, got {workers}")
    if chunksize < 1:
        raise ValueError(f"chunksize must be >= 1, got {chunksize}")

    if workers is not None and workers > 1 and len(jobs) > 1:
        if cfg_loader is None:
            raise ValueError("run_jobs needs cfg_loader or configs_dir")
        _check_picklable(cfg_loader, jobs)
        n = min(int(workers), len(jobs))
        with ProcessPoolExecutor(max_workers=n,
                                 mp_context=multiprocessing.get_context("spawn")) as ex:
            return list(ex.map(_run_job_worker,
                               [(j, cfg_loader, keep_records) for j in jobs],
                               chunksize=chunksize))

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


# --------------------------------------------------------------------------- #
# Manifest and env freeze (Requirements 22.6, 22.7)
# --------------------------------------------------------------------------- #
#
# The *env hash* identifies everything that defines the environment a policy
# is evaluated in: the ``sim`` and ``scenario`` sections plus ``root_seed``
# (every churn trace, workload and keyed draw derives from it). Reward,
# agent and experiment sections are excluded, so tuning the reward or the
# learner never invalidates the freeze. ``freeze_env`` records one env hash
# per scenario -- the headroom command freezes every scenario found in
# ``configs/scenarios`` (including heldout_medium and bursty), so later train
# / eval runs on any scenario are checked. ``check_frozen`` refuses
# (``FrozenEnvError``) a config whose scenario is unknown to the manifest or
# whose env hash differs, and refuses when no manifest exists.

import dataclasses as _dataclasses
import datetime as _datetime
import hashlib as _hashlib
import json as _json
import subprocess as _subprocess

MANIFEST_NAME = "manifest.json"
_MANIFEST_PACKAGES = ("numpy", "pandas", "matplotlib", "churnmaster")

__all__ += ["FrozenEnvError", "MANIFEST_NAME", "env_config_hash", "write_manifest",
            "read_manifest", "git_hash", "package_versions", "freeze_env", "check_frozen"]


class FrozenEnvError(RuntimeError):
    """A run's env config does not match the frozen env hash (or nothing is frozen)."""


def env_config_hash(cfg: Config) -> str:
    """SHA-256 over the env-defining part of ``cfg`` (``sim``, ``scenario``, ``root_seed``)."""
    payload = {"sim": _dataclasses.asdict(cfg.sim),
               "scenario": _dataclasses.asdict(cfg.scenario),
               "root_seed": int(cfg.root_seed)}
    text = _json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                       allow_nan=False)
    return _hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_manifest(path: Path | str, data: Mapping[str, Any]) -> Path:
    """Write ``data`` as pretty, key-sorted JSON (parent dirs created)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_json.dumps(dict(data), indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    return out


def read_manifest(path: Path | str) -> dict[str, Any]:
    return _json.loads(Path(path).read_text(encoding="utf-8"))


def git_hash(cwd: Path | str | None = None) -> str | None:
    """``git rev-parse HEAD`` or None when git / the repository is unavailable."""
    try:
        proc = _subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                               timeout=10, cwd=None if cwd is None else str(cwd))
    except (OSError, _subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def package_versions(names: Sequence[str] = _MANIFEST_PACKAGES) -> dict[str, str | None]:
    import sys
    from importlib import metadata

    out: dict[str, str | None] = {"python": sys.version.split()[0]}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


def freeze_env(manifest_path: Path | str, cfgs_by_scenario: Mapping[str, Config],
               extra: Mapping[str, Any] | None = None, frozen: bool = True) -> dict[str, Any]:
    """Record ``{scenario: env_hash}`` (+ git hash, package versions) in the manifest.

    The manifest is rewritten; ``extra`` entries (seeds, gate verdicts, tuned
    overrides, ...) are stored alongside. ``frozen=False`` records the hashes
    but marks the env as NOT frozen (e.g. G1 failed even after tuning);
    ``check_frozen`` then refuses every run. Returns the written dict.
    """
    if not cfgs_by_scenario:
        raise ValueError("freeze_env needs at least one scenario config")
    for scn, cfg in cfgs_by_scenario.items():
        if cfg.scenario.name != scn:
            raise ValueError(f"config for {scn!r} has scenario name {cfg.scenario.name!r}")
    data: dict[str, Any] = dict(extra or {})
    data.update({
        "frozen": bool(frozen),
        "env_hashes": {s: env_config_hash(c) for s, c in sorted(cfgs_by_scenario.items())},
        "config_hashes": {s: config_hash(c) for s, c in sorted(cfgs_by_scenario.items())},
        "git_hash": git_hash(),
        "packages": package_versions(),
        "frozen_at_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(timespec="seconds"),
    })
    write_manifest(manifest_path, data)
    return data


def check_frozen(manifest_path: Path | str, cfg: Config) -> str:
    """Return the env hash of ``cfg`` if it matches the frozen one; else raise ``FrozenEnvError``."""
    path = Path(manifest_path)
    if not path.is_file():
        raise FrozenEnvError(f"no frozen env manifest at {path}; run `churnmaster headroom` first")
    manifest = read_manifest(path)
    if manifest.get("frozen") is not True:
        raise FrozenEnvError(f"manifest {path} does not hold a frozen env "
                             f"(headroom gate G1 did not pass)")
    hashes = manifest.get("env_hashes") or {}
    scn = cfg.scenario.name
    if scn not in hashes:
        raise FrozenEnvError(f"scenario {scn!r} is not in the frozen manifest {path} "
                             f"(frozen: {sorted(hashes)})")
    got = env_config_hash(cfg)
    if got != hashes[scn]:
        raise FrozenEnvError(f"env hash for scenario {scn!r} differs from the frozen one "
                             f"({got[:12]} != {hashes[scn][:12]}); refusing to run")
    return got
