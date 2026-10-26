"""Agent ablations (Task 12.1, Requirement 28.1) -- EXPLORATORY.

Pre-registered analysis plan (fixed before any ablation result existed; the
same text, :data:`ANALYSIS_PLAN`, is stored in ``<out>/manifest.json``)
--------------------------------------------------------------------------
1. Scope. Ablations are exploratory and strictly separate from the locked
   primary evaluation. They never use the EVAL split for episodes and never
   read or write the eval lock. They never write into canonical results
   directories (headroom, diagnostics, diagnostics_60, train, tune, matched,
   eval, checkpoint_5_17, figures, reproduce_*); the frozen env manifest
   (``results/headroom/manifest.json``) is only READ (``check_frozen``). All
   outputs go to ``<out>/<suite>/`` and ``<out>/manifest.json`` (default
   ``<out> = results/ablations``).
2. Evaluation seeds. Split TEST, run 0, episodes 0..19, scenarios stable,
   high_churn, heldout_medium, full-length episodes. Before anything runs,
   the ``seed_state`` fingerprints of these seeds are checked to be disjoint
   from calibration (run 0, ep 0..19), train (runs 0..2, ep 0..199), tuning
   (TRAIN run 1000, ep 0..9), validation (run 0, ep 0..4) and eval (ep 0..19)
   seeds; the run aborts otherwise (:class:`AblationSeedError`).
3. Reference ("canonical RL"). The 3 canonical final Q-tables
   ``results/train/final/run{0,1,2}/q_final.npy`` (γ=0.8, α_min=0.02),
   re-evaluated greedily on the same TEST seeds. Context policies on the same
   seeds: tuned_heuristic (global selection of ``results/tune/selected.json``),
   kademlia_epoch, never_replace.
4. Variants. Each variant trains 3 independent runs (runs 0, 1, 2: the same
   agent exploration streams and the same TRAIN env seeds as canonical, so the
   only difference is the ablated component): 200 episodes alternating
   stable / high_churn, γ=0.8, α_min=0.02, all other agent settings and the
   reward weights (w_L 0.3, w_M 0.1, w_R 0.15) canonical; no grid search, no
   validation checkpoints. The final greedy policies are evaluated on the
   TEST seeds.

   * ``gamma0``: γ=0.
   * ``no_probe``: PROBE masked out in training and evaluation (allowed: KEEP,
     REPLACE iff a candidate exists) via the agent's action mask; updates
     bootstrap over the masked next-state actions. ``rl/state.py`` unchanged.
   * ``shuffled_reward``: transitions of each training episode are buffered;
     at episode end their rewards are randomly permuted across the episode's
     transitions and the Q updates are applied in emission order. The
     permutation generator is ``SeedSequence(root, spawn_key=(AGENT, run, 0,
     1))`` (the run's exploration seed extended by stream id 1; one generator
     per run, consumed across its episodes).
   * ``buffered_control``: the same buffering and episode-end updates, rewards
     NOT permuted. shuffled_reward vs buffered_control isolates the reward
     signal; buffered_control vs canonical isolates the update-timing change.
   * ``drop_<feature>`` for tier, inc_fail, inc_stale, inc_age, cand_faster,
     churn: a discretizer wrapper (``rl.variants.FeatureDrop``) collapses the
     feature to the single bin 0. ``cand`` is special: collapsing it to NONE
     would mask REPLACE everywhere, so ``drop_cand_freshness`` merges FRESH
     and STALE into one "present" bin and keeps NONE (presence and masking
     stay correct). cand_faster stays 0 whenever cand is NONE. The same
     wrapper is used in training and evaluation (the env records the wrapped
     state); the table shape stays 648 x 3.
5. Statistics (exploratory, NO multiple-comparison correction; p-values are
   uncorrected). Per variant and scenario: the paired per-seed difference
   ``rl_mean(variant) − rl_mean(canonical)`` (``rl_mean`` = per-seed mean over
   the 3 runs) on J, success_rate, p95_latency_all_ms and pings_per_hour, with
   a paired bootstrap 95% CI (10 000 resamples, seed 0) and a two-sided
   Wilcoxon signed-rank p; shuffled_reward vs buffered_control on the same
   metrics; each variant (and canonical) vs tuned_heuristic on J; per-run
   values of every variant. Functions: ``stats.average_over_runs``,
   ``stats.paired_bootstrap_ci``, ``stats.wilcoxon_paired``.
6. Also saved per variant: ``curves.csv`` and ``q_final.npy`` per run (with
   the |ΔQ| first/last-window trend) and greedy-policy statistics
   (``train.greedy_stats``).

Outputs (``<out>/agent/``)::

    per_seed.csv        one row per (label, scenario, TEST episode)
    comparison.csv      variant x reference x scenario x metric: diff, CI, p (uncorrected)
    per_run.csv         per (group, run, scenario): metric means + ΔJ vs canonical rl_mean
    summary.csv         per (label, scenario): mean / std
    greedy_stats.json   greedy-policy statistics and |ΔQ| trend per group and run
    variants/<name>/run<r>/{q_final.npy (+ sidecars), curves.csv}

plus ``<out>/figures/agent_ablation_J.png`` (forest plot of ΔJ vs canonical),
``<out>/figures/agent_ablation_metrics.png`` (Δp95, Δpings) and
``<out>/manifest.json`` (git hash, config / env hashes, package versions, the
plan, settings, timings). An existing suite output is never overwritten
unless ``overwrite=True``.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from churnmaster.config import Config, config_hash
from churnmaster.experiments import stats as st
from churnmaster.experiments.evaluate import USED_SPLITS, default_q_paths, used_seed_states
from churnmaster.experiments.runner import (KEY_COLUMNS, Job, check_frozen, env_config_hash,
                                            git_hash, package_versions, run_jobs, write_manifest)
from churnmaster.experiments.train import (DQ_WINDOW, TrainHooks, TrainOutput, TrainSettings,
                                           TrainSpec, greedy_stats, run_trainings)
from churnmaster.experiments.tune_heuristic import GLOBAL_LABEL, load_selected, tuned_policy_names
from churnmaster.policies.ablated import EncodedLearned
from churnmaster.rl.qlearning import QAgent
from churnmaster.rl.qlearning import _paths as _q_paths
from churnmaster.rl.variants import FEATURE_NAMES, FeatureDrop, no_probe_mask
from churnmaster.seeding import Split, env_seed, seed_state

__all__ = [
    "ANALYSIS_PLAN", "SUITES", "CANONICAL_LABEL", "CANONICAL_GAMMA", "CANONICAL_ALPHA_MIN",
    "TEST_RUN", "DEFAULT_N_SEEDS", "EVAL_SCENARIOS", "TRAIN_SCENARIOS", "RUNS", "METRICS",
    "CONTEXT_POLICIES", "PROTECTED_NAMES", "AblationSeedError", "TrainVariant",
    "AGENT_VARIANTS", "variant_name_for_drop", "check_test_seeds_disjoint",
    "guard_ablation_out", "AblationReport", "run_agent_suite",
]

ANALYSIS_PLAN = (__doc__ or "Pre-registered analysis plan Outputs (").split("Pre-registered analysis plan", 1)[1].split("Outputs (", 1)[0]
ANALYSIS_PLAN = "Pre-registered analysis plan" + ANALYSIS_PLAN.rstrip()

SUITES = ("agent",)
CANONICAL_LABEL = "canonical"
CANONICAL_GAMMA = 0.8
CANONICAL_ALPHA_MIN = 0.02
TEST_SPLIT = Split.TEST
TEST_RUN = 0
DEFAULT_N_SEEDS = 20
EVAL_SCENARIOS: tuple[str, ...] = ("stable", "high_churn", "heldout_medium")
TRAIN_SCENARIOS: tuple[str, ...] = ("stable", "high_churn")
RUNS: tuple[int, ...] = (0, 1, 2)
METRICS: tuple[str, ...] = ("J", "success_rate", "p95_latency_all_ms", "pings_per_hour")
SUMMARY_METRICS: tuple[str, ...] = METRICS + ("evictions_per_hour",)
CONTEXT_POLICIES: tuple[str, ...] = (GLOBAL_LABEL, "kademlia_epoch", "never_replace")
#: Eval seeds checked for disjointness (the locked eval uses episodes 0..19).
EVAL_EPISODES_CHECKED = range(20)
#: Results sub-directories ablations never write into (``reproduce_*`` too).
PROTECTED_NAMES: tuple[str, ...] = ("headroom", "diagnostics", "diagnostics_60", "train", "tune",
                                    "matched", "eval", "checkpoint_5_17", "figures")
_PROTECTED_PREFIX = "reproduce"
_REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_NAME = "manifest.json"


class AblationSeedError(RuntimeError):
    """The ablation TEST seeds overlap a seed used elsewhere (or repeat)."""


# --------------------------------------------------------------------------- #
# Variant registry
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TrainVariant:
    """One agent ablation: what differs from canonical training (everything else equal)."""

    name: str
    description: str
    gamma: float | None = None          # None: canonical gamma
    no_probe: bool = False
    drop: str | None = None             # feature collapsed by FeatureDrop
    update_mode: str = "online"

    def encoder(self) -> FeatureDrop | None:
        return None if self.drop is None else FeatureDrop(self.drop)

    def mask(self) -> np.ndarray | None:
        return no_probe_mask() if self.no_probe else None

    def hooks(self) -> TrainHooks:
        return TrainHooks(encode=self.encoder(), mask=self.mask(), update_mode=self.update_mode)

    def settings(self, base: TrainSettings) -> TrainSettings:
        return base if self.gamma is None else dataclasses.replace(base, gamma=float(self.gamma))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def variant_name_for_drop(feature: str) -> str:
    return "drop_cand_freshness" if feature == "cand" else f"drop_{feature}"


def _agent_variants() -> dict[str, TrainVariant]:
    vs = [
        TrainVariant("gamma0", "gamma = 0 (myopic: Q estimates the immediate epoch reward)",
                     gamma=0.0),
        TrainVariant("no_probe", "PROBE masked out in training and evaluation", no_probe=True),
        TrainVariant("shuffled_reward", "episode-end updates with rewards permuted within the "
                     "episode", update_mode="shuffled"),
        TrainVariant("buffered_control", "episode-end updates, rewards not permuted (control "
                     "for shuffled_reward)", update_mode="buffered"),
    ]
    for f in FEATURE_NAMES:
        desc = ("cand FRESH/STALE merged into one 'present' bin (presence kept)" if f == "cand"
                else f"feature {f} collapsed to bin 0")
        vs.append(TrainVariant(variant_name_for_drop(f), desc, drop=f))
    return {v.name: v for v in vs}


AGENT_VARIANTS: dict[str, TrainVariant] = _agent_variants()


# --------------------------------------------------------------------------- #
# Seeds and output guard
# --------------------------------------------------------------------------- #

def check_test_seeds_disjoint(root: int, episodes: Sequence[int], run: int = TEST_RUN
                              ) -> dict[str, Any]:
    """Raise :class:`AblationSeedError` unless the TEST seeds are unique and unused elsewhere."""
    used = list(USED_SPLITS) + [("eval", Split.EVAL, (0,), EVAL_EPISODES_CHECKED)]
    other = used_seed_states(root, used)
    seen: dict[tuple[int, ...], int] = {}
    for e in episodes:
        fp = seed_state(env_seed(root, TEST_SPLIT, run, int(e)))
        if fp in seen:
            raise AblationSeedError(f"TEST episodes {seen[fp]} and {e} share a seed state")
        if fp in other:
            raise AblationSeedError(f"TEST run {run} episode {e} overlaps {other[fp]}")
        seen[fp] = int(e)
    return {"split": "test", "run": int(run), "episodes": [int(e) for e in episodes],
            "checked_against": [{"name": n, "split": s.name.lower(), "runs": list(r),
                                 "episodes": [min(ep), max(ep)]} for n, s, r, ep in used],
            "n_other_seed_states": len(other), "overlap": False}


def _is_relative_to(p: Path, other: Path) -> bool:
    try:
        p.relative_to(other)
        return True
    except ValueError:
        return False


def guard_ablation_out(out: Path | str) -> Path:
    """Resolve ``out``; ``ValueError`` if it is (inside / contains) a protected results dir.

    Protected: the results root itself, :data:`PROTECTED_NAMES` and every
    ``reproduce*`` directory, under both ``./results`` and the repository's
    ``results``.
    """
    p = Path(out).resolve()
    for root in {(Path.cwd() / "results").resolve(), (_REPO_ROOT / "results").resolve()}:
        if p == root or _is_relative_to(root, p):
            raise ValueError(f"ablation output dir {out!s} is or contains the results root {root}")
        if not _is_relative_to(p, root):
            continue
        first = p.relative_to(root).parts[0]
        if first in PROTECTED_NAMES or first.startswith(_PROTECTED_PREFIX):
            raise ValueError(f"ablation output dir {out!s} is inside canonical results dir "
                             f"{root / first}; use e.g. results/ablations")
    return p


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return None if math.isnan(v) else v
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _run_label(group: str, run: int) -> str:
    return f"{group}:run{run}"


def dq_summary(curves: pd.DataFrame, window: int = DQ_WINDOW, step: int = 20) -> dict[str, Any]:
    """Mean |ΔQ| over the first / last ``window`` episodes and per ``step``-episode block."""
    dq = curves.sort_values("episode")["mean_abs_dq"].to_numpy(float)
    if dq.size == 0:
        return {"window": 0, "first_mean": math.nan, "last_mean": math.nan, "shrinks": False,
                "trend": []}
    w = max(1, min(int(window), dq.size // 2)) if dq.size >= 2 else 1
    first, last = float(np.nanmean(dq[:w])), float(np.nanmean(dq[-w:]))
    trend = [{"from_episode": i, "to_episode": min(i + step, dq.size),
              "mean_abs_dq": float(np.nanmean(dq[i:i + step]))}
             for i in range(0, dq.size, step)]
    return {"window": w, "first_mean": first, "last_mean": last,
            "shrinks": bool(dq.size >= 2 and last < first), "trend": trend}


def _compare_row(df: pd.DataFrame, scenario: str, metric: str, a: str, b: str,
                 n_resamples: int) -> dict[str, Any]:
    pf = st.paired_frame(df[df["scenario"] == scenario], metric, a, b, st.EVAL_KEYS)
    xa, xb = pf["a"].to_numpy(float), pf["b"].to_numpy(float)
    ci = st.paired_bootstrap_ci(xa, xb, n_resamples=n_resamples, level=0.95, seed=0)
    w = st.wilcoxon_paired(xa, xb)
    sign = st.METRIC_DIRECTION.get(metric, +1)
    return {"variant": a, "reference": b, "scenario": scenario, "metric": metric, "n": ci.n,
            "mean_variant": float(xa.mean()), "mean_reference": float(xb.mean()),
            "mean_diff": ci.mean, "ci_lo": ci.lo, "ci_hi": ci.hi,
            "excludes_zero": ci.excludes_zero, "wilcoxon_stat": w.statistic,
            "p_uncorrected": w.p, "favourable": int(np.sign(sign * ci.mean))}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

@dataclass
class AblationReport:
    suite: str
    out_dir: Path
    per_seed: pd.DataFrame
    comparison: pd.DataFrame
    per_run: pd.DataFrame
    summary: pd.DataFrame
    greedy: dict[str, Any]
    manifest: dict[str, Any]
    files: list[Path] = field(default_factory=list)
    wall_s: float = 0.0
    train_wall_s: float = 0.0
    eval_wall_s: float = 0.0


def _check_canonical_settings(q_paths: Sequence[Path], base: TrainSettings) -> dict[str, Any]:
    """Compare the canonical Q-table params / train manifest with the ablation settings."""
    info: dict[str, Any] = {"q_tables": []}
    for p in q_paths:
        qp, np_, mp, jp = _q_paths(p)
        meta = json.loads(jp.read_text(encoding="utf-8")) if jp.is_file() else {}
        info["q_tables"].append({"path": str(p), "sha256": _sha256_file(qp),
                                 "mask_sha256": _sha256_file(mp) if mp.is_file() else None,
                                 "gamma": meta.get("gamma"), "alpha_min": meta.get("alpha_min")})
        for k in ("gamma", "alpha_min", "alpha_exp"):
            if k in meta and not math.isclose(float(meta[k]), float(getattr(base, k))):
                raise ValueError(f"canonical Q-table {p} has {k}={meta[k]} but the ablation "
                                 f"settings use {getattr(base, k)}")
    man = Path(q_paths[0]).resolve().parent.parent / "train_manifest.json" if q_paths else None
    if man is not None and man.is_file():
        sets = json.loads(man.read_text(encoding="utf-8")).get("settings") or []
        if sets:
            canon = dict(sets[0])
            mine = base.to_dict()
            diff = [k for k in canon if k != "validate_every" and canon.get(k) != mine.get(k)]
            if diff:
                raise ValueError(f"ablation base settings differ from the canonical training "
                                 f"({man}) in {diff}: {[(k, canon[k], mine.get(k)) for k in diff]}")
        info["train_manifest"] = str(man)
        info["settings_match"] = True
    else:
        info["train_manifest"] = None
        info["settings_match"] = None
    return info


def run_agent_suite(configs_dir: Path | str, manifest_path: Path | str = "results/headroom/manifest.json",
                    out_dir: Path | str = "results/ablations", *,
                    experiment_path: Path | str | None = None,
                    variants: Sequence[str] | None = None,
                    runs: Sequence[int] = RUNS, episodes: int | None = None,
                    n_seeds: int = DEFAULT_N_SEEDS,
                    scenarios: Sequence[str] = EVAL_SCENARIOS,
                    canonical_q: Sequence[Path | str] | None = None,
                    tune_selected: Path | str = "results/tune/selected.json",
                    workers: int | None = None, n_resamples: int = 10_000,
                    overwrite: bool = False, log_every: int = 0,
                    progress: Callable[[str], None] | None = None) -> AblationReport:
    """Train, evaluate and compare the agent ablations (see module docstring / plan)."""
    from churnmaster.experiments.diagnostics import _frozen_loader
    from churnmaster.plots import plot_ablation_forest

    t0 = time.perf_counter()
    say = progress or (lambda m: None)
    configs_dir, manifest_path = Path(configs_dir), Path(manifest_path)
    out = guard_ablation_out(out_dir)
    suite_dir = out / "agent"
    if (suite_dir / "comparison.csv").exists() and not overwrite:
        raise ValueError(f"{suite_dir} already holds ablation results; pass overwrite=True "
                         f"(--overwrite) to replace them")
    names = list(variants) if variants is not None else list(AGENT_VARIANTS)
    unknown = [n for n in names if n not in AGENT_VARIANTS]
    if unknown or not names or len(set(names)) != len(names):
        raise ValueError(f"bad variant list {names}; known: {list(AGENT_VARIANTS)}")
    runs = [int(r) for r in runs]
    if not runs or len(set(runs)) != len(runs):
        raise ValueError(f"runs must be unique, got {runs}")
    if n_seeds < 1:
        raise ValueError("n_seeds must be >= 1")
    exp = Path(experiment_path) if experiment_path is not None else \
        configs_dir / "experiments" / "train.toml"
    q_paths = [Path(p) for p in (canonical_q if canonical_q is not None else default_q_paths(len(runs)))]
    if len(q_paths) != len(runs):
        raise ValueError(f"need one canonical Q-table per run ({len(runs)}), got {len(q_paths)}")

    # ---- configs, frozen env (read-only), seeds --------------------------------
    loader = _frozen_loader(configs_dir, exp, manifest_path)
    all_scn = list(dict.fromkeys([*TRAIN_SCENARIOS, *scenarios]))
    cfgs: dict[str, Config] = {s: loader(s) for s in all_scn}
    env_hashes = {s: check_frozen(manifest_path, c) for s, c in cfgs.items()}
    roots = {int(c.root_seed) for c in cfgs.values()}
    if len(roots) != 1:
        raise ValueError(f"scenarios disagree on root_seed: {sorted(roots)}")
    root = roots.pop()
    eps_list = list(range(int(n_seeds)))
    disjoint = check_test_seeds_disjoint(root, eps_list)
    id_bits = cfgs[TRAIN_SCENARIOS[0]].sim.id_bits

    base = TrainSettings.from_config(cfgs[TRAIN_SCENARIOS[0]], TRAIN_SCENARIOS,
                                     gamma=CANONICAL_GAMMA, alpha_min=CANONICAL_ALPHA_MIN,
                                     episodes=episodes)
    base = dataclasses.replace(base, validate_every=base.episodes)   # no validation checkpoints
    # Full-length runs must match the canonical training settings (raises otherwise);
    # shortened runs (tests) only record the canonical Q-table hashes.
    canon_info = _check_canonical_settings(q_paths, base) if episodes is None else \
        _q_info_only(q_paths)
    selected = load_selected(tune_selected)
    tuned_impl = tuned_policy_names(selected)[GLOBAL_LABEL]
    rw = cfgs[TRAIN_SCENARIOS[0]].reward

    # ---- training ---------------------------------------------------------------
    vs = [AGENT_VARIANTS[n] for n in names]
    specs = [TrainSpec(loader, v.settings(base), r, log_every, v.hooks()) for v in vs for r in runs]
    say(f"ablate[agent]: training {len(specs)} runs ({len(vs)} variants x {len(runs)} runs x "
        f"{base.episodes} episodes, workers={workers})")
    t_tr = time.perf_counter()
    outputs: list[TrainOutput] = run_trainings(specs, workers=workers)
    train_wall = time.perf_counter() - t_tr
    say(f"ablate[agent]: training done in {train_wall:.1f}s")

    files: list[Path] = []
    trained: dict[str, dict[int, TrainOutput]] = {}
    k = 0
    for v in vs:
        for r in runs:
            o = outputs[k]
            k += 1
            trained.setdefault(v.name, {})[r] = o
            d = suite_dir / "variants" / v.name / f"run{r}"
            d.mkdir(parents=True, exist_ok=True)
            files.append(o.final.save(d / "q_final.npy"))
            o.curves.to_csv(d / "curves.csv", index=False)
            files.append(d / "curves.csv")

    # ---- evaluation on TEST seeds ------------------------------------------------
    labels: list[tuple[str, str, int, str, Any]] = []     # (label, group, run, impl, policy)
    for r, p in zip(runs, q_paths):
        lab = _run_label(CANONICAL_LABEL, r)
        labels.append((lab, CANONICAL_LABEL, r, str(p), EncodedLearned(QAgent.load(p), name=lab,
                                                                         id_bits=id_bits)))
    for v in vs:
        for r in runs:
            lab = _run_label(v.name, r)
            labels.append((lab, v.name, r, f"ablation:{v.name}",
                           EncodedLearned(trained[v.name][r].final, name=lab, id_bits=id_bits,
                                          encode=v.encoder())))
    for c in CONTEXT_POLICIES:
        impl = tuned_impl if c == GLOBAL_LABEL else c
        labels.append((c, c, -1, impl, impl))
    jobs, meta = [], []
    for s in scenarios:
        for lab, grp, r, impl, pol in labels:
            for ep in eps_list:
                jobs.append(Job(pol, s, TEST_SPLIT, TEST_RUN, ep))
                meta.append((lab, grp, r, impl))
    say(f"ablate[agent]: evaluating {len(labels)} labels x {len(scenarios)} scenarios x "
        f"{len(eps_list)} TEST seeds = {len(jobs)} episodes")
    t_ev = time.perf_counter()
    results = run_jobs(jobs, cfg_loader=loader, workers=workers)
    eval_wall = time.perf_counter() - t_ev
    say(f"ablate[agent]: evaluation done in {eval_wall:.1f}s")

    rows = []
    for (lab, grp, r, impl), res in zip(meta, results):
        row = res.row()
        row.pop("policy")
        rows.append({"policy": lab, "group": grp, "rl_run": r, "policy_impl": impl, **row})
    df = pd.DataFrame(rows)
    if df.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("duplicate (label, scenario, split, run, episode) rows")
    for m in SUMMARY_METRICS:
        bad = df[~df[m].astype(float).map(math.isfinite)]
        if not bad.empty:
            raise ValueError(f"non-finite {m} in {len(bad)} ablation rows, e.g. "
                             f"{bad[['policy', 'scenario', 'episode']].iloc[0].to_dict()}")

    # ---- statistics (exploratory; uncorrected) ------------------------------------
    groups = [CANONICAL_LABEL, *[v.name for v in vs]]
    means = []
    for g in groups:
        avg = st.average_over_runs(df, [_run_label(g, r) for r in runs], g,
                                   metrics=list(SUMMARY_METRICS))
        means.append(avg.assign(group=g, rl_run=-1, policy_impl=f"rl_mean({g})"))
    full = pd.concat([df, *means], ignore_index=True)

    comp = []
    for v in vs:
        for s in scenarios:
            for m in METRICS:
                comp.append({"comparison": "vs_canonical",
                             **_compare_row(full, s, m, v.name, CANONICAL_LABEL, n_resamples)})
    if "shuffled_reward" in names and "buffered_control" in names:
        for s in scenarios:
            for m in METRICS:
                comp.append({"comparison": "shuffled_vs_buffered",
                             **_compare_row(full, s, m, "shuffled_reward", "buffered_control",
                                            n_resamples)})
    for g in groups:
        for s in scenarios:
            comp.append({"comparison": "vs_tuned_heuristic",
                         **_compare_row(full, s, "J", g, GLOBAL_LABEL, n_resamples)})
    comparison = pd.DataFrame(comp)

    pr = []
    for g in groups:
        for r in runs:
            lab = _run_label(g, r)
            for s in scenarios:
                sub = df[(df["policy"] == lab) & (df["scenario"] == s)]
                row: dict[str, Any] = {"group": g, "run": r, "label": lab, "scenario": s,
                                       "n": len(sub)}
                for m in SUMMARY_METRICS:
                    row[f"{m}_mean"] = float(sub[m].astype(float).mean())
                c = _compare_row(full, s, "J", lab, CANONICAL_LABEL, n_resamples)
                row.update({"dJ_vs_canonical_mean": c["mean_diff"], "dJ_ci_lo": c["ci_lo"],
                            "dJ_ci_hi": c["ci_hi"], "dJ_p_uncorrected": c["p_uncorrected"]})
                pr.append(row)
    per_run = pd.DataFrame(pr)

    srows = []
    for (pol, s), g in full.groupby(["policy", "scenario"], sort=False):
        row = {"policy": pol, "group": g["group"].iloc[0], "scenario": s, "n": len(g)}
        for m in SUMMARY_METRICS:
            x = g[m].astype(float)
            row[f"{m}_mean"] = float(x.mean())
            row[f"{m}_std"] = float(x.std(ddof=1)) if len(x) > 1 else math.nan
        srows.append(row)
    summary = pd.DataFrame(srows)

    greedy: dict[str, Any] = {}
    for r, p in zip(runs, q_paths):
        ent: dict[str, Any] = {"greedy": greedy_stats(QAgent.load(p))}
        cpath = Path(p).parent / "curves.csv"
        if cpath.is_file():
            ent["dq"] = dq_summary(pd.read_csv(cpath))
        greedy.setdefault(CANONICAL_LABEL, {})[f"run{r}"] = ent
    for v in vs:
        for r in runs:
            o = trained[v.name][r]
            greedy.setdefault(v.name, {})[f"run{r}"] = {
                "greedy": greedy_stats(o.final), "dq": dq_summary(o.curves),
                "train_wall_s": o.wall_s,
                "n_updates_total": int(o.curves["n_updates"].sum()),
                "n_transitions_total": int(o.curves["n_transitions"].sum())}

    # ---- write --------------------------------------------------------------------
    suite_dir.mkdir(parents=True, exist_ok=True)
    tables = {"per_seed.csv": df, "comparison.csv": comparison, "per_run.csv": per_run,
              "summary.csv": summary}
    for fname, t in tables.items():
        t.to_csv(suite_dir / fname, index=False)
        files.append(suite_dir / fname)
    files.append(_write_json(suite_dir / "greedy_stats.json", greedy))
    fig_dir = out / "figures"
    vc = comparison[comparison["comparison"] == "vs_canonical"]
    files.append(plot_ablation_forest(
        vc, ["J"], fig_dir / "agent_ablation_J.png",
        title="Agent ablations: ΔJ = rl_mean(variant) − rl_mean(canonical), TEST seeds, 95% CI"))
    files.append(plot_ablation_forest(
        vc, ["p95_latency_all_ms", "pings_per_hour"], fig_dir / "agent_ablation_metrics.png",
        title="Agent ablations: Δ vs canonical (TEST seeds, 95% paired bootstrap CI)"))

    wall = time.perf_counter() - t0
    suite_manifest = {
        "suite": "agent", "created_utc": _now(), "git_hash": git_hash(),
        "packages": package_versions(("numpy", "scipy", "pandas", "matplotlib", "churnmaster")),
        "analysis_plan": ANALYSIS_PLAN, "exploratory": True,
        "multiple_comparison_correction": "none (exploratory; p-values uncorrected)",
        "frozen_manifest": {"path": str(manifest_path), "sha256": _sha256_file(manifest_path),
                            "access": "read-only (check_frozen)"},
        "env_hashes": env_hashes, "config_hashes": {s: config_hash(c) for s, c in cfgs.items()},
        "experiment": str(exp), "root_seed": root,
        "reward": {"w_L": rw.w_L, "w_M": rw.w_M, "w_R": rw.w_R},
        "train_settings": base.to_dict(), "train_split": "train", "runs": runs,
        "variants": {v.name: {**v.to_dict(), "settings": v.settings(base).to_dict()} for v in vs},
        "shuffle_stream": "SeedSequence(root, spawn_key=(AGENT, run, 0, 1))",
        "eval_seeds": disjoint, "eval_scenarios": list(scenarios),
        "canonical": canon_info,
        "context_policies": {c: (tuned_impl if c == GLOBAL_LABEL else c) for c in CONTEXT_POLICIES},
        "tune_selected": {"path": str(tune_selected), "global": selected.get("global")},
        "stats": {"bootstrap_resamples": n_resamples, "ci_level": 0.95, "bootstrap_seed": 0,
                  "wilcoxon": "two-sided, zero_method=wilcox", "pairing": list(st.EVAL_KEYS),
                  "rl_mean": "per-seed mean over the runs of a group"},
        "n_train_runs": len(specs), "n_eval_jobs": len(jobs), "workers": workers,
        "timings_s": {"train": round(train_wall, 1), "eval": round(eval_wall, 1),
                      "total": round(wall, 1),
                      "per_train_run": {_run_label(v.name, r): round(trained[v.name][r].wall_s, 1)
                                        for v in vs for r in runs}},
        "outputs": sorted(str(Path(f).relative_to(out)) for f in files),
        "per_seed_sha256": _sha256_file(suite_dir / "per_seed.csv"),
    }
    mpath = out / MANIFEST_NAME
    top: dict[str, Any] = {}
    if mpath.is_file():
        top = json.loads(mpath.read_text(encoding="utf-8"))
    top.update({"kind": "churnmaster-ablations", "analysis_plan": ANALYSIS_PLAN,
                "exploratory": True, "updated_utc": _now()})
    top.setdefault("suites", {})["agent"] = suite_manifest
    write_manifest(mpath, _clean(top))
    files.append(mpath)
    say(f"ablate[agent]: wrote {len(files)} files to {out} ({wall:.1f}s)")
    return AblationReport("agent", out, df, comparison, per_run, summary, greedy, suite_manifest,
                          files, wall, train_wall, eval_wall)


def _q_info_only(q_paths: Sequence[Path]) -> dict[str, Any]:
    """Canonical Q-table hashes without the settings check (shortened test runs)."""
    out = []
    for p in q_paths:
        qp, _, mp, _ = _q_paths(p)
        out.append({"path": str(p), "sha256": _sha256_file(qp),
                    "mask_sha256": _sha256_file(mp) if mp.is_file() else None})
    return {"q_tables": out, "train_manifest": None, "settings_match": None,
            "note": "episodes overridden: canonical settings not compared"}
