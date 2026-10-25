"""Locked evaluation (Task 9.3, Requirements 2.3, 2.4, 25.2, 25.3, 25.6).

Protocol
--------
Every policy label (see :data:`DEFAULT_POLICIES`) and every RL run is run on
the EVAL split, episodes ``0..n_seeds-1`` (default 20), for each scenario
(default stable, high_churn, heldout_medium). Eval environments use
``seeding.eval_seed(root, episode)`` (run index 0), so all policies and all RL
runs see identical environments (Requirement 2.4). Random baselines draw
from their own AGENT-split stream per job (``runner.job_baseline_rng`` with
run 0), never from an env stream.

Labels (``policy`` column) vs implementations (``policy_impl`` column):

* ``never_replace``, ``always_replace``, ``uniform_random``,
  ``kademlia_epoch``, ``kademlia_std`` (``shares_decision_points=False``),
  ``heuristic`` (``DEFAULT_CONFIG``): the runner policy of the same name;
* ``matched_random``: per-scenario frequencies from ``results/matched/freqs.json``;
* ``tuned_heuristic``: the global tuning selection (``results/tune/selected.json``);
* ``tuned_heuristic_privileged`` (``privileged=True``): the per-scenario
  selection. Scenarios without a per-scenario selection (heldout_medium was
  never tuned on) use the global config and are flagged
  ``privileged_fallback=True``;
* ``oracle_heuristic`` (``reference=True``): ground-truth reference, not a
  competitor and not an upper bound;
* ``rl_run0..rl_run{R-1}``: greedy :class:`~churnmaster.policies.learned.Learned`
  over each final Q-table (token ``learned`` in the experiment's policy list);
  ``rl_mean`` is their per-seed mean (statistics only).

Eval-split lock (Requirement 25.3)
----------------------------------
The lock is stored in the frozen env manifest (default
``results/headroom/manifest.json``) under ``"eval_lock"``: split, run,
episodes, scenarios, root seed, the ``seed_state`` fingerprint of every eval
episode seed, and a ``seed_list_hash`` over all of it. Creating the lock is
an explicit, separate step (``churnmaster evaluate --create-lock``);
:func:`create_eval_lock` is idempotent for an identical lock and refuses to
overwrite a different one. :func:`check_eval_lock` raises
:class:`SplitLockError` when the manifest or lock is missing or any field
differs (episodes, scenarios, root seed, fingerprints); evaluation then
writes no results. Both also check that the eval seed fingerprints are
disjoint from every seed used elsewhere in the project
(:data:`USED_SPLITS`: calibration run 0 episodes 0..19 -- headroom used
0..9, diagnostics 0..19 --, train runs 0..2 episodes 0..199, tuning TRAIN
run 1000 episodes 0..9, validation run 0 episodes 0..4).

Single use
----------
Completed evaluations are appended to the manifest's ``"eval_runs"`` list
(timestamp, git hash, output dir, per-seed hash). The eval split is meant
to be used once: if a run is already recorded, :func:`evaluate` raises
:class:`EvalRerunError` unless ``allow_rerun=True`` (``--allow-rerun``).
Reruns give identical results (everything is seeded per job).

Outputs (``results/eval``): ``per_seed.csv``, ``per_run.csv``,
``primary_tests.csv``, ``summary.csv``, ``hypotheses.csv`` / ``.md``,
``eval_manifest.json`` and figures (``J.png``, ``p95_latency_all_ms.png``,
``pings_per_hour.png``, ``success_vs_pings_<scenario>.png``).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from churnmaster.config import Config, config_hash
from churnmaster.experiments import stats as st
from churnmaster.experiments.diagnostics import _frozen_loader
from churnmaster.experiments.matched import load_matched_freqs
from churnmaster.experiments.runner import (KEY_COLUMNS, Job, check_frozen, git_hash,
                                            package_versions, read_manifest, run_jobs,
                                            write_manifest)
from churnmaster.experiments.tune_heuristic import (GLOBAL_LABEL, PRIVILEGED_LABEL, TUNING_RUN,
                                                    load_selected, tuned_policy_names)
from churnmaster.policies.learned import Learned
from churnmaster.rl.qlearning import _paths as _q_paths
from churnmaster.seeding import Split, episode_seed, eval_seed, seed_state

__all__ = [
    "SplitLockError", "EvalRerunError", "LOCK_KEY", "RUNS_KEY", "DEFAULT_SCENARIOS",
    "DEFAULT_N_SEEDS", "DEFAULT_POLICIES", "RL_TOKEN", "USED_SPLITS", "STAT_METRICS",
    "COMPARATORS", "EvalReport", "used_seed_states", "check_eval_disjoint", "build_eval_lock",
    "create_eval_lock", "check_eval_lock", "resolve_eval", "create_lock_from_configs",
    "evaluate", "default_q_paths",
]


class SplitLockError(RuntimeError):
    """The eval split is not locked, does not match the lock, or overlaps another split."""


class EvalRerunError(RuntimeError):
    """An evaluation run is already recorded and ``allow_rerun`` was not given."""


LOCK_KEY = "eval_lock"
RUNS_KEY = "eval_runs"
DEFAULT_SCENARIOS: tuple[str, ...] = ("stable", "high_churn", "heldout_medium")
DEFAULT_N_SEEDS = 20
RL_TOKEN = "learned"
RL_MEAN = "rl_mean"
DEFAULT_POLICIES: tuple[str, ...] = (
    "never_replace", "always_replace", "uniform_random", "matched_random", "kademlia_epoch",
    "kademlia_std", "heuristic", GLOBAL_LABEL, PRIVILEGED_LABEL, "oracle_heuristic", RL_TOKEN,
)
_PLAIN = frozenset({"never_replace", "always_replace", "uniform_random", "kademlia_epoch",
                    "kademlia_std", "heuristic", "oracle_heuristic"})
STAT_METRICS: tuple[str, ...] = ("J", "success_rate", "p95_latency_all_ms", "pings_per_hour",
                                 "evictions_per_hour")
COMPARATORS: tuple[str, ...] = ("kademlia_epoch", GLOBAL_LABEL, "matched_random")
#: Labels never counted as "best baseline" in H3 (reference / privileged).
H3_EXCLUDE: tuple[str, ...] = ("oracle_heuristic", PRIVILEGED_LABEL)
PLOT_METRICS: tuple[str, ...] = ("J", "p95_latency_all_ms", "pings_per_hour")

#: Seeds used elsewhere: (name, split, runs, episodes).
USED_SPLITS: tuple[tuple[str, Split, tuple[int, ...], range], ...] = (
    ("calibration", Split.CALIBRATION, (0,), range(20)),
    ("train", Split.TRAIN, (0, 1, 2), range(200)),
    ("tuning", Split.TRAIN, (TUNING_RUN,), range(10)),
    ("validation", Split.VALIDATION, (0,), range(5)),
)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Disjointness and lock
# --------------------------------------------------------------------------- #

def used_seed_states(root: int, used: Sequence[tuple[str, Split, Sequence[int], Sequence[int]]]
                     = USED_SPLITS) -> dict[tuple[int, ...], str]:
    """``seed_state`` fingerprint -> ``"name/run/episode"`` for every seed used outside eval."""
    out: dict[tuple[int, ...], str] = {}
    for name, split, runs, episodes in used:
        for r in runs:
            for e in episodes:
                out[seed_state(episode_seed(root, split, r, e))] = f"{name}/run{r}/ep{e}"
    return out


def check_eval_disjoint(root: int, episodes: Sequence[int],
                        used: Sequence[tuple[str, Split, Sequence[int], Sequence[int]]]
                        = USED_SPLITS) -> dict[str, Any]:
    """Raise ``SplitLockError`` if any eval seed fingerprint repeats or is used elsewhere."""
    other = used_seed_states(root, used)
    seen: dict[tuple[int, ...], int] = {}
    for e in episodes:
        fp = seed_state(eval_seed(root, int(e)))
        if fp in seen:
            raise SplitLockError(f"eval episodes {seen[fp]} and {e} share a seed state")
        if fp in other:
            raise SplitLockError(f"eval episode {e} seed state overlaps {other[fp]}")
        seen[fp] = int(e)
    return {"checked_against": [{"name": n, "split": s.name.lower(), "runs": list(r),
                                 "episodes": [min(ep), max(ep)] if len(ep) else []}
                                for n, s, r, ep in used],
            "n_other_seed_states": len(other), "overlap": False}


def _check_episodes(episodes: Sequence[int]) -> list[int]:
    eps = [int(e) for e in episodes]
    if not eps:
        raise ValueError("the eval lock needs at least one episode")
    if any(e < 0 for e in eps) or len(set(eps)) != len(eps):
        raise ValueError(f"eval episodes must be unique non-negative ints, got {eps}")
    return eps


_LOCK_FIELDS = ("split", "run", "episodes", "scenarios", "root_seed", "seed_states",
                "seed_list_hash")


def build_eval_lock(root: int, episodes: Sequence[int], scenarios: Sequence[str]) -> dict[str, Any]:
    """The lock payload (without timestamps); deterministic in its inputs."""
    eps = _check_episodes(episodes)
    if not scenarios:
        raise ValueError("the eval lock needs at least one scenario")
    payload: dict[str, Any] = {
        "split": "eval", "run": 0, "episodes": eps, "scenarios": sorted(set(scenarios)),
        "root_seed": int(root),
        "seed_states": {str(e): list(seed_state(eval_seed(root, e))) for e in eps},
    }
    payload["seed_list_hash"] = _sha256_text(json.dumps(payload, sort_keys=True,
                                                        separators=(",", ":")))
    return payload


def _mismatch(stored: Mapping[str, Any], expected: Mapping[str, Any]) -> list[str]:
    return [k for k in _LOCK_FIELDS if stored.get(k) != expected.get(k)]


def _read(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise SplitLockError(f"no manifest at {manifest_path}; run `churnmaster headroom` and "
                             f"`churnmaster evaluate --create-lock` first")
    return read_manifest(manifest_path)


def create_eval_lock(manifest_path: Path | str, cfg: Config, episodes: Sequence[int],
                     scenarios: Sequence[str]) -> dict[str, Any]:
    """Write the eval lock into the manifest (idempotent; never overwrites a different lock)."""
    path = Path(manifest_path)
    manifest = _read(path)
    expected = build_eval_lock(cfg.root_seed, episodes, scenarios)
    disjoint = check_eval_disjoint(cfg.root_seed, expected["episodes"])
    existing = manifest.get(LOCK_KEY)
    if existing is not None:
        bad = _mismatch(existing, expected)
        if bad:
            raise SplitLockError(f"manifest {path} already holds a different eval lock "
                                 f"(differs in {bad}); refusing to overwrite")
        return existing
    lock = {**expected, "disjoint": disjoint, "created_utc": _now()}
    manifest[LOCK_KEY] = lock
    write_manifest(path, manifest)
    return lock


def check_eval_lock(manifest_path: Path | str, cfg: Config, episodes: Sequence[int],
                    scenarios: Sequence[str]) -> dict[str, Any]:
    """Return the stored lock if it matches; else raise ``SplitLockError``."""
    path = Path(manifest_path)
    lock = _read(path).get(LOCK_KEY)
    if not lock:
        raise SplitLockError(f"the eval split is not locked in {path}; run "
                             f"`churnmaster evaluate --create-lock` first")
    try:
        expected = build_eval_lock(cfg.root_seed, episodes, scenarios)
    except ValueError as e:
        raise SplitLockError(str(e)) from e
    bad = _mismatch(lock, expected)
    if bad:
        raise SplitLockError(f"eval split does not match the lock in {path} (differs in {bad})")
    check_eval_disjoint(cfg.root_seed, expected["episodes"])
    return lock


# --------------------------------------------------------------------------- #
# Configuration resolution
# --------------------------------------------------------------------------- #

def default_q_paths(n_runs: int = 3) -> list[Path]:
    return [Path(f"results/train/final/run{i}/q_final.npy") for i in range(n_runs)]


@dataclass
class _Resolved:
    loader: Callable[[str], Config]
    cfgs: dict[str, Config]
    scenarios: tuple[str, ...]
    episodes: list[int]
    env_hashes: dict[str, str]
    root: int
    policies: tuple[str, ...]
    n_resamples: int
    ci_level: float


def resolve_eval(configs_dir: Path | str, manifest_path: Path | str,
                 experiment_path: Path | str | None = None, n_seeds: int | None = None,
                 scenarios: Sequence[str] | None = None,
                 policies: Sequence[str] | None = None) -> _Resolved:
    """Load configs (eval experiment layer) and check every scenario against the frozen env."""
    configs_dir, manifest_path = Path(configs_dir), Path(manifest_path)
    if experiment_path is None:
        experiment_path = configs_dir / "experiments" / "eval.toml"
    loader = _frozen_loader(configs_dir, Path(experiment_path), manifest_path)
    first = loader((tuple(scenarios) if scenarios else DEFAULT_SCENARIOS)[0])
    exp = first.experiment
    scns = tuple(scenarios) if scenarios else (tuple(exp.scenarios) or DEFAULT_SCENARIOS)
    n = n_seeds if n_seeds is not None else (exp.n_seeds or DEFAULT_N_SEEDS)
    if n < 1:
        raise ValueError(f"n_seeds must be >= 1, got {n}")
    pols = tuple(policies) if policies else (tuple(exp.policies) or DEFAULT_POLICIES)
    unknown = [p for p in pols if p not in DEFAULT_POLICIES]
    if unknown:
        raise ValueError(f"unknown eval policies {unknown}; expected a subset of {DEFAULT_POLICIES}")
    missing = [p for p in ("kademlia_epoch", GLOBAL_LABEL, "matched_random", RL_TOKEN)
               if p not in pols]
    if missing:
        raise ValueError(f"eval policies must include {missing} (primary tests)")
    cfgs = {s: loader(s) for s in scns}
    env_hashes = {s: check_frozen(manifest_path, c) for s, c in cfgs.items()}
    roots = {int(c.root_seed) for c in cfgs.values()}
    if len(roots) != 1:
        raise ValueError(f"scenarios disagree on root_seed: {sorted(roots)}")
    stats_cfg = exp.stats
    return _Resolved(loader, cfgs, scns, list(range(n)), env_hashes, roots.pop(), pols,
                     stats_cfg.bootstrap_resamples if stats_cfg else 10_000,
                     stats_cfg.ci_level if stats_cfg else 0.95)


def create_lock_from_configs(configs_dir: Path | str, manifest_path: Path | str,
                             experiment_path: Path | str | None = None,
                             n_seeds: int | None = None,
                             scenarios: Sequence[str] | None = None) -> dict[str, Any]:
    """``evaluate --create-lock``: frozen-env check, then :func:`create_eval_lock`."""
    r = resolve_eval(configs_dir, manifest_path, experiment_path, n_seeds, scenarios)
    return create_eval_lock(manifest_path, r.cfgs[r.scenarios[0]], r.episodes, r.scenarios)


# --------------------------------------------------------------------------- #
# Policy labels
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class _Spec:
    label: str
    impl: str
    policy: Any                      # policy name or Learned
    policy_kw: Mapping[str, Any] = field(default_factory=dict)
    privileged: bool = False
    reference: bool = False
    privileged_fallback: bool = False
    rl_run: int = -1


def _specs(scenario: str, policies: Sequence[str], tuned: Mapping[str, Any],
           freqs: Mapping[str, Any], rl: Sequence[Learned]) -> list[_Spec]:
    out: list[_Spec] = []
    for p in policies:
        if p in _PLAIN:
            out.append(_Spec(p, p, p, reference=p == "oracle_heuristic"))
        elif p == "matched_random":
            out.append(_Spec(p, p, p, {"freqs": freqs[scenario]}))
        elif p == GLOBAL_LABEL:
            out.append(_Spec(p, tuned[GLOBAL_LABEL], tuned[GLOBAL_LABEL]))
        elif p == PRIVILEGED_LABEL:
            per = tuned[PRIVILEGED_LABEL]
            impl = per.get(scenario, tuned[GLOBAL_LABEL])
            out.append(_Spec(p, impl, impl, privileged=True,
                             privileged_fallback=scenario not in per))
        elif p == RL_TOKEN:
            out.extend(_Spec(pol.name, RL_TOKEN, pol, rl_run=i) for i, pol in enumerate(rl))
        else:  # pragma: no cover - validated in resolve_eval
            raise ValueError(p)
    return out


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

@dataclass
class EvalReport:
    per_seed: pd.DataFrame
    summary: pd.DataFrame
    primary: pd.DataFrame
    per_run: pd.DataFrame
    hypotheses: pd.DataFrame
    manifest: dict[str, Any]
    lock: dict[str, Any]
    files: list[Path]
    wall_s: float
    n_jobs: int


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pol, scn), g in df.groupby(["policy", "scenario"], sort=False):
        row: dict[str, Any] = {"policy": pol, "scenario": scn,
                               "policy_impl": g["policy_impl"].iloc[0],
                               "shares_decision_points": bool(g["shares_decision_points"].all()),
                               "privileged": bool(g["privileged"].any()),
                               "privileged_fallback": bool(g["privileged_fallback"].any()),
                               "reference": bool(g["reference"].any()), "n": len(g)}
        for m in STAT_METRICS:
            v = g[m].astype(float)
            row[f"{m}_mean"] = float(v.mean())
            row[f"{m}_std"] = float(v.std(ddof=1)) if len(v) > 1 else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate(configs_dir: Path | str, manifest_path: Path | str, out_dir: Path | str, *,
             q_paths: Sequence[Path | str] | None = None,
             tune_selected: Path | str = "results/tune/selected.json",
             matched_freqs: Path | str = "results/matched/freqs.json",
             experiment_path: Path | str | None = None, n_seeds: int | None = None,
             scenarios: Sequence[str] | None = None, policies: Sequence[str] | None = None,
             workers: int | None = None, allow_rerun: bool = False,
             n_resamples: int | None = None,
             progress: Callable[[str], None] | None = None) -> EvalReport:
    """Run the locked evaluation and write all outputs (see module docstring).

    Raises ``FrozenEnvError`` / ``SplitLockError`` / ``EvalRerunError`` before
    running anything or writing any file.
    """
    say = progress or (lambda m: None)
    manifest_path, out = Path(manifest_path), Path(out_dir)
    r = resolve_eval(configs_dir, manifest_path, experiment_path, n_seeds, scenarios, policies)
    lock = check_eval_lock(manifest_path, r.cfgs[r.scenarios[0]], r.episodes, r.scenarios)
    prior = read_manifest(manifest_path).get(RUNS_KEY) or []
    if prior and not allow_rerun:
        raise EvalRerunError(f"{len(prior)} evaluation run(s) already recorded in {manifest_path} "
                             f"(last: {prior[-1].get('out_dir')}); the eval split is single-use, "
                             f"pass --allow-rerun to run again")

    q_paths = [Path(p) for p in (q_paths if q_paths is not None else default_q_paths())]
    if not q_paths:
        raise ValueError("evaluate needs at least one RL Q-table")
    selected = load_selected(tune_selected)
    tuned = tuned_policy_names(selected)
    freqs = load_matched_freqs(matched_freqs)
    missing = [s for s in r.scenarios if s not in freqs]
    if missing and "matched_random" in r.policies:
        raise ValueError(f"{matched_freqs} has no matched_random frequencies for {missing}")
    id_bits = r.cfgs[r.scenarios[0]].sim.id_bits
    rl = [Learned.load(p, name=f"rl_run{i}", id_bits=id_bits) for i, p in enumerate(q_paths)]
    rl_labels = [p.name for p in rl]
    n_res = int(n_resamples if n_resamples is not None else r.n_resamples)

    specs: list[_Spec] = []
    jobs: list[Job] = []
    for s in r.scenarios:
        for sp in _specs(s, r.policies, tuned, freqs, rl):
            for ep in r.episodes:
                specs.append(sp)
                jobs.append(Job(sp.policy, s, Split.EVAL, 0, ep, dict(sp.policy_kw)))
    say(f"evaluate: {len(set(sp.label for sp in specs))} labels x {len(r.scenarios)} scenarios x "
        f"{len(r.episodes)} eval seeds = {len(jobs)} episodes (workers={workers})")
    t0 = time.perf_counter()
    results = run_jobs(jobs, cfg_loader=r.loader, workers=workers)
    sim_wall = time.perf_counter() - t0
    say(f"evaluate: episodes done in {sim_wall:.1f}s; computing statistics (bootstrap {n_res})")

    rows = []
    for sp, res in zip(specs, results):
        row = res.row()
        row.pop("policy")
        rows.append({"policy": sp.label, "policy_impl": sp.impl,
                     "privileged": sp.privileged, "privileged_fallback": sp.privileged_fallback,
                     "reference": sp.reference, "rl_run": sp.rl_run, **row})
    df = pd.DataFrame(rows)
    if df.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("duplicate (label, scenario, split, run, episode) rows")
    for m in STAT_METRICS:
        bad = df[~df[m].astype(float).map(math.isfinite)]
        if not bad.empty:
            raise ValueError(f"non-finite {m} in {len(bad)} eval rows, e.g. "
                             f"{bad[['policy', 'scenario', 'episode']].iloc[0].to_dict()}")

    avg = st.average_over_runs(df, rl_labels, RL_MEAN, metrics=list(STAT_METRICS))
    avg = avg.assign(policy_impl=f"mean({','.join(rl_labels)})", shares_decision_points=True,
                     privileged=False, privileged_fallback=False, reference=False, rl_run=-1)
    full = pd.concat([df, avg], ignore_index=True)
    primary = st.primary_tests(full, rl=RL_MEAN, comparators=COMPARATORS, scenarios=r.scenarios,
                               metric="J", n_resamples=n_res, seed=0)
    per_run = pd.concat([st.per_run_results(full, rl_labels, COMPARATORS, r.scenarios, metric=m,
                                            n_resamples=n_res, seed=0)
                         for m in ("J", *st.PRIMARY_METRICS)], ignore_index=True)
    if {"high_churn", "heldout_medium"} <= set(r.scenarios):
        hyp = st.hypothesis_table(full, rl_labels, RL_MEAN, kademlia="kademlia_epoch",
                                  tuned=GLOBAL_LABEL, matched="matched_random",
                                  scenarios=r.scenarios, exclude_from_best=H3_EXCLUDE,
                                  n_resamples=n_res, seed=0)
    else:
        hyp = pd.DataFrame(columns=["id", "statement", "verdict", "evidence"])
    summary = _summary(full)
    wall = time.perf_counter() - t0

    # ---- write outputs (only after every check and computation succeeded) ----
    from churnmaster.plots import plot_policy_bars, plot_success_vs_pings

    out.mkdir(parents=True, exist_ok=True)
    files = [out / "per_seed.csv", out / "per_run.csv", out / "primary_tests.csv",
             out / "summary.csv", out / "hypotheses.csv", out / "hypotheses.md"]
    df.to_csv(files[0], index=False)
    per_run.to_csv(files[1], index=False)
    primary.to_csv(files[2], index=False)
    summary.to_csv(files[3], index=False)
    st.write_hypothesis_table(hyp, files[4], files[5])
    for m in PLOT_METRICS:
        files.append(plot_policy_bars(full, m, out / f"{m}.png",
                                      title=f"{m} on eval seeds (mean ± std; rl_mean = mean of runs)"))
    for s in r.scenarios:
        files.append(plot_success_vs_pings(df[df["scenario"] == s], out / f"success_vs_pings_{s}.png",
                                           title=f"Success vs pings/h, {s} (eval seeds)"))

    q_info = []
    for p in q_paths:
        qp, _, mp, _ = _q_paths(p)
        q_info.append({"path": str(p), "sha256": _sha256_file(qp),
                       "mask_sha256": _sha256_file(mp) if mp.is_file() else None})
    per_seed_sha = _sha256_file(files[0])
    labels = {}
    for sp in specs:
        labels.setdefault(sp.label, {}).setdefault(sp.impl, None)
    eval_manifest = {
        "git_hash": git_hash(), "packages": package_versions(), "created_utc": _now(),
        "frozen_manifest": str(manifest_path), "eval_lock": lock,
        "seeds": {"split": "eval", "run": 0, "episodes": r.episodes, "scenarios": list(r.scenarios)},
        "env_hashes": r.env_hashes,
        "config_hashes": {s: config_hash(c) for s, c in r.cfgs.items()},
        "experiment": str(experiment_path) if experiment_path else "configs/experiments/eval.toml",
        "policies": list(r.policies),
        "labels": {k: sorted(v) for k, v in labels.items()},
        "privileged_fallback": sorted({sp_s for sp_s in r.scenarios
                                       if sp_s not in tuned[PRIVILEGED_LABEL]}),
        "rl_q_tables": q_info, "tune_selected": {"path": str(tune_selected), **selected},
        "matched_freqs": {"path": str(matched_freqs),
                          **json.loads(Path(matched_freqs).read_text(encoding="utf-8"))},
        "stats": {"bootstrap_resamples": n_res, "ci_level": 0.95, "bootstrap_seed": 0,
                  "comparators": list(COMPARATORS), "rl_label": RL_MEAN,
                  "rl_runs": rl_labels, "h3_exclude_from_best": list(H3_EXCLUDE),
                  "holm_family": "RL mean vs comparators x scenarios on J"},
        "n_jobs": len(jobs), "workers": workers, "sim_wall_s": round(sim_wall, 1),
        "wall_s": round(wall, 1), "per_seed_sha256": per_seed_sha,
        "allow_rerun": bool(allow_rerun), "prior_eval_runs": len(prior),
    }
    files.append(write_manifest(out / "eval_manifest.json", eval_manifest))

    # Record the completed run in the frozen manifest (re-read: keep other keys).
    m = read_manifest(manifest_path)
    m.setdefault(RUNS_KEY, []).append({
        "completed_utc": _now(), "git_hash": eval_manifest["git_hash"], "out_dir": str(out),
        "n_jobs": len(jobs), "seed_list_hash": lock["seed_list_hash"],
        "per_seed_sha256": per_seed_sha, "allow_rerun": bool(allow_rerun)})
    write_manifest(manifest_path, m)
    say(f"evaluate: wrote {len(files)} files to {out} ({wall:.1f}s)")
    return EvalReport(df, summary, primary, per_run, hyp, eval_manifest, lock, files, wall, len(jobs))
