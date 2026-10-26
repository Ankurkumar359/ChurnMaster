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

Reward-weight sweep (Task 12.2, Requirement 28.2) -- EXPLORATORY
================================================================

Reward-weight sweep: pre-registered analysis plan (fixed before any sweep
result existed; the same text, :data:`REWARD_ANALYSIS_PLAN`, is stored in
``<out>/manifest.json`` under ``suites.reward``)
--------------------------------------------------------------------------
1. Question. The report hypothesizes that RL trails the tuned heuristic
   partly because the latency weight w_L=0.3 is too light (RL p95 is 18-38%
   above the best baseline). The sweep tests whether a higher w_L lowers RL's
   p95 and narrows the gap.
2. Cells. The task grid w_M in {0.05, 0.2, 0.5} x w_L in {0.1, 0.3, 0.6}
   (9 cells), plus a w_L line at the canonical w_M=0.1: w_L in {0.1, 0.6,
   1.0} (3 cells). Together 12 trained cells. The point w_L=0.3, w_M=0.1 is
   canonical and reuses the 3 canonical Q-tables (no retraining). In every
   cell w_R = 1.5 * w_M (the canonical ratio 0.15 / 0.1). Cell names
   ``wL<w_L>_wM<w_M>``, e.g. ``wL0.6_wM0.2``.
3. Training. As in the agent suite: 3 runs (runs 0..2), gamma=0.8,
   alpha_min=0.02, 200 episodes alternating stable / high_churn, the same
   TRAIN seeds and agent exploration streams; only the reward weights used
   by the Ledger during training change. Reward is not part of
   env_config_hash, so the frozen env is unaffected (``check_frozen`` is
   still called for every cell config).
4. Evaluation. TEST split, run 0, episodes 0..19 x {stable, high_churn,
   heldout_medium} (the agent-suite seeds; ``check_test_seeds_disjoint``).
   Every policy is SCORED UNDER THE CANONICAL WEIGHTS (w_L 0.3, w_M 0.1,
   w_R 0.15), so J is comparable to the primary study. Additionally
   ``J_train`` = J under the cell's own training weights, recomputed from the
   per-lookup / per-decision records with ``rl.reward.lookup_loss`` /
   ``action_cost``; the same recomputation at the canonical weights is checked
   to equal the ledger J for every episode. Raw metrics: success_rate,
   p95_latency_all_ms, mean_latency_ms, pings_per_hour, evictions_per_hour.
5. References on the same TEST seeds: canonical RL (3 Q-tables),
   tuned_heuristic, kademlia_epoch, never_replace. Their rows are reused from
   ``<out>/agent/per_seed.csv`` when the labels / seeds / config hashes /
   canonical Q-table hashes match and a re-run of the first TEST episode of
   every reference x scenario reproduces the stored trace and summary hashes;
   otherwise they are re-run.
6. Statistics (exploratory, NO multiple-comparison correction; p-values are
   uncorrected). Per cell x scenario: the paired per-seed difference
   ``rl_mean(cell) - rl_mean(canonical)`` on canonical-weight J,
   p95_latency_all_ms, pings_per_hour and success_rate (paired bootstrap 95%
   CI, 10 000 resamples, seed 0; two-sided Wilcoxon signed-rank p);
   ``rl_mean(cell) - tuned_heuristic`` on canonical-weight J and on
   p95_latency_all_ms (the "gap" quantities); per-run values. Monotonic-trend
   summary: Spearman rho between w_L and the RL mean p95 across the w_L line
   at w_M=0.1 (w_L 0.1, 0.3, 0.6, 1.0), per scenario -- n=4 points,
   descriptive only.
7. Caveat. The tuned heuristic was tuned under the canonical weights; a
   heuristic re-tuned for other weights is not part of this sweep.

Reward outputs (``<out>/reward/``)::

    per_seed.csv        one row per (label, scenario, TEST episode); J (canonical weights), J_train
    comparison.csv      cell x reference x scenario x metric: diff, CI, p (uncorrected)
    per_run.csv         per (cell, run, scenario): metric means + ΔJ / Δp95 vs canonical rl_mean
    summary.csv         per (label, scenario): mean / std
    trend.csv           Spearman rho(w_L, RL p95) on the w_M=0.1 line, per scenario
    greedy_stats.json   greedy-policy statistics and |ΔQ| trend per cell and run
    cells/<cell>/run<r>/{q_final.npy (+ sidecars), curves.csv}

plus ``<out>/figures/reward_sweep_p95.png``, ``reward_sweep_J.png`` and
``reward_sweep_tradeoff.png``.

Memoryless held-out scenario (Task 12.3, Requirement 28.3) -- EXPLORATORY
=========================================================================

Memoryless held-out scenario: pre-registered analysis plan (fixed before any
memoryless result existed; the same text, :data:`MEMORYLESS_ANALYSIS_PLAN`,
is stored in ``<out>/manifest.json`` under ``suites.memoryless``)
--------------------------------------------------------------------------
1. Question. Every frozen scenario draws sessions from a Weibull with shape
   0.6 (decreasing hazard: a contact that has been online long is likely to
   stay online, which the inc_age feature and the heuristics can exploit).
   Does the canonical RL policy's standing relative to the baselines change
   when sessions are memoryless (exponential, shape 1.0), i.e. when a
   contact's age carries no information about its remaining lifetime?
2. Scenario. ``heldout_memoryless`` = ``heldout_medium`` with
   ``weibull_shape = 1.0``. The churn model sets the Weibull scale to
   mean / Gamma(1 + 1/shape), so the mean session lengths, the online
   fraction and the mean churn rate are unchanged; only the hazard shape
   changes. The scenario is NOT in the frozen manifest. Its env must differ
   from the frozen heldout_medium env only in ``scenario.name`` and
   ``scenario.weibull_shape``: substituting heldout_medium's values for those
   two fields must give exactly the frozen heldout_medium env hash
   (:func:`check_derived_scenario`; abort otherwise). Its env hash, config
   hash and TOML sha256 are recorded in ``<out>/manifest.json``
   (``suites.memoryless.scenario`` and ``extra_scenarios``); the frozen
   manifest is only read.
3. Seeds. TEST split, run 0, episodes 0..19 (the agent-suite seeds,
   ``check_test_seeds_disjoint``), on heldout_memoryless AND on
   heldout_medium (the paired shape-0.6 reference: same seeds, same
   policies), full-length episodes.
4. Policies. Canonical RL (the 3 final Q-tables, greedy; ``rl_mean`` =
   per-seed mean over the 3 runs), tuned_heuristic (global selection of
   ``results/tune/selected.json``), kademlia_epoch, matched_random,
   never_replace, uniform_random and oracle_heuristic (ground-truth
   reference, not a competitor). No retraining and no re-tuning on the new
   scenario. matched_random frequencies for heldout_memoryless are measured
   as in task 9.2: the canonical Q-tables run greedily on VALIDATION run 0,
   episodes 0..4 of heldout_memoryless, decisions pooled; heldout_medium uses
   ``results/matched/freqs.json`` (read-only) when its env hash matches,
   else it is measured the same way.
5. Statistics (exploratory, NO multiple-comparison correction; p-values are
   uncorrected; paired bootstrap 95% CI, 10 000 resamples, seed 0; two-sided
   Wilcoxon signed-rank p).
   (a) ``rl_vs_reference``: per scenario, rl_mean − reference on J,
       success_rate, p95_latency_all_ms and pings_per_hour.
   (b) ``shape_effect``: per policy (and rl_mean), metric(heldout_memoryless)
       − metric(heldout_medium), paired by TEST episode (same env seed; the
       churn draws differ), same metrics.
   (c) ``gap_shift``: [rl_mean − ref]_memoryless − [rl_mean − ref]_medium on
       J and p95_latency_all_ms, ref in tuned_heuristic, kademlia_epoch,
       matched_random, paired by episode.
   Cross-check (recorded, not a gate): heldout_medium rows of labels also in
   ``<out>/agent/per_seed.csv`` must reproduce its trace / summary hashes.
6. Caveat. One scenario, 20 seeds, no correction: descriptive evidence about
   the hazard shape only.

Memoryless outputs (``<out>/memoryless/``)::

    per_seed.csv        one row per (label, scenario, TEST episode)
    summary.csv         per (label, scenario): mean / std
    deltas.csv          rl_vs_reference / shape_effect / gap_shift: diff, CI, p (uncorrected)
    summary.md          markdown tables of the headline numbers
    matched_freqs.json  matched_random frequencies used per scenario

plus ``<out>/figures/memoryless_J_p95.png``.
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
    "REWARD_ANALYSIS_PLAN", "CANONICAL_WEIGHTS", "RewardCell", "RewardLoader", "reward_cells",
    "j_from_records", "run_reward_suite",
    "MEMORYLESS_ANALYSIS_PLAN", "MEMORYLESS_SCENARIO", "MEMORYLESS_PARENT",
    "MEMORYLESS_POLICIES", "check_derived_scenario", "run_memoryless_suite",
]

ANALYSIS_PLAN = (__doc__ or "Pre-registered analysis plan Outputs (").split("Pre-registered analysis plan", 1)[1].split("Outputs (", 1)[0]
ANALYSIS_PLAN = "Pre-registered analysis plan" + ANALYSIS_PLAN.rstrip()
_RW_MARK = "Reward-weight sweep: pre-registered analysis plan"
REWARD_ANALYSIS_PLAN = _RW_MARK + (__doc__ or _RW_MARK + "Reward outputs (").split(
    _RW_MARK, 1)[1].split("Reward outputs (", 1)[0].rstrip()
_ML_MARK = "Memoryless held-out scenario: pre-registered analysis plan"
MEMORYLESS_ANALYSIS_PLAN = _ML_MARK + (__doc__ or _ML_MARK + "Memoryless outputs (").split(
    _ML_MARK, 1)[1].split("Memoryless outputs (", 1)[0].rstrip()

SUITES = ("agent", "reward", "memoryless")
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
    trend: pd.DataFrame | None = None      # reward suite only


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


# =========================================================================== #
# Reward-weight sweep (Task 12.2, Requirement 28.2)
# =========================================================================== #

#: Canonical scoring weights ``(w_L, w_M, w_R)``; every sweep policy is scored under them.
CANONICAL_WEIGHTS: tuple[float, float, float] = (0.3, 0.1, 0.15)
REWARD_GRID_W_M: tuple[float, ...] = (0.05, 0.2, 0.5)
REWARD_GRID_W_L: tuple[float, ...] = (0.1, 0.3, 0.6)
REWARD_LINE_W_M = 0.1
REWARD_LINE_W_L: tuple[float, ...] = (0.1, 0.6, 1.0)
W_R_PER_W_M = 1.5
REWARD_METRICS: tuple[str, ...] = ("J", "p95_latency_all_ms", "pings_per_hour", "success_rate")
GAP_METRICS: tuple[str, ...] = ("J", "p95_latency_all_ms")
REWARD_SUMMARY_METRICS: tuple[str, ...] = ("J", "J_train", "success_rate", "p95_latency_all_ms",
                                           "mean_latency_ms", "pings_per_hour",
                                           "evictions_per_hour")
REWARD_CAVEAT = ("The tuned heuristic was tuned under the canonical weights (w_L 0.3, w_M 0.1, "
                 "w_R 0.15); a heuristic re-tuned for other weights is not part of this sweep.")
TREND_NOTE = "Spearman rho over n=4 w_L points at w_M=0.1 (descriptive only; no inference)"
_J_TOL = 1e-9


def _fmt_w(x: float) -> str:
    s = f"{float(x):g}"
    return s if ("." in s or "e" in s) else s + ".0"


@dataclass(frozen=True)
class RewardCell:
    """One reward-weight cell; ``w_R = 1.5 * w_M`` (the canonical ratio)."""

    w_L: float
    w_M: float
    origin: str = "grid"        # grid | line | canonical

    @property
    def w_R(self) -> float:
        return round(W_R_PER_W_M * float(self.w_M), 12)

    @property
    def weights(self) -> tuple[float, float, float]:
        return (float(self.w_L), float(self.w_M), self.w_R)

    @property
    def name(self) -> str:
        return f"wL{_fmt_w(self.w_L)}_wM{_fmt_w(self.w_M)}"

    @property
    def is_canonical(self) -> bool:
        return all(math.isclose(a, b) for a, b in zip(self.weights, CANONICAL_WEIGHTS))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "w_L": float(self.w_L), "w_M": float(self.w_M),
                "w_R": self.w_R, "origin": self.origin, "reuses_canonical": self.is_canonical}


CANONICAL_CELL = RewardCell(CANONICAL_WEIGHTS[0], CANONICAL_WEIGHTS[1], "canonical")


def reward_cells(include_canonical: bool = False) -> list[RewardCell]:
    """The 12 trained cells (9 grid + 3 line), optionally followed by the canonical cell."""
    cells = [RewardCell(wl, wm, "grid") for wm in REWARD_GRID_W_M for wl in REWARD_GRID_W_L]
    cells += [RewardCell(wl, REWARD_LINE_W_M, "line") for wl in REWARD_LINE_W_L]
    if include_canonical:
        cells.append(CANONICAL_CELL)
    return cells


REWARD_CELLS: dict[str, RewardCell] = {c.name: c for c in reward_cells()}


@dataclass(frozen=True)
class RewardLoader:
    """Picklable loader replacing the ``reward`` section of ``base``'s configs.

    The env hash (``sim``, ``scenario``, ``root_seed``) is unchanged; only the
    Ledger / metrics weights differ.
    """

    base: Callable[[str], Config]
    w_L: float
    w_M: float
    w_R: float

    def __call__(self, scenario: str) -> Config:
        from churnmaster.config import validate

        cfg = self.base(scenario)
        cfg = dataclasses.replace(cfg, reward=dataclasses.replace(
            cfg.reward, w_L=float(self.w_L), w_M=float(self.w_M), w_R=float(self.w_R)))
        validate(cfg)
        return cfg


def j_from_records(lookups: Sequence[Any], decisions: Sequence[Any], w_L: float, w_M: float,
                   w_R: float, deadline_ms: float = 2000) -> float:
    """J under the given weights from an episode's measured lookup / decision records."""
    from churnmaster.rl.reward import action_cost, lookup_loss

    lks = [lk for lk in lookups if lk.measured]
    if not lks:
        return math.nan
    loss = sum(lookup_loss(lk.success, lk.latency_ms, w_L, deadline_ms) for lk in lks)
    cost = sum(action_cost(int(d.pings), int(d.evictions), w_M, w_R)
               for d in decisions if d.measured)
    return 1.0 - loss / len(lks) - cost / len(lks)


def _same_float(a: float, b: float, tol: float = _J_TOL) -> bool:
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= tol


@dataclass(frozen=True)
class _RewardEvalTask:
    job: Job
    loader: Callable[[str], Config]                  # canonical-weight (scoring) loader
    train_weights: tuple[float, float, float]


def _reward_eval(task: _RewardEvalTask) -> dict[str, Any]:
    """Pool entry point: one TEST episode -> row with J (canonical weights) and J_train."""
    from churnmaster.experiments.runner import _worker_config, run_job

    cfg = _worker_config(task.loader, task.job.scenario)
    res = run_job(task.job, cfg=cfg, keep_records=True)
    rw, dl = cfg.reward, cfg.sim.deadline_ms
    j_rec = j_from_records(res.lookups, res.decisions, rw.w_L, rw.w_M, rw.w_R, dl)
    if not _same_float(j_rec, res.ledger_score):
        raise RuntimeError(f"record-based J {j_rec!r} != ledger J {res.ledger_score!r} for "
                           f"{task.job.policy_name}/{task.job.scenario}/ep{task.job.episode}")
    row = res.row()
    row["J_recomputed"] = j_rec
    row["J_train"] = j_from_records(res.lookups, res.decisions, *task.train_weights, dl)
    return row


def _run_reward_eval(tasks: Sequence[_RewardEvalTask], workers: int | None) -> list[dict[str, Any]]:
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    tasks = list(tasks)
    if workers is not None and workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=min(int(workers), len(tasks)),
                                 mp_context=multiprocessing.get_context("spawn")) as ex:
            return list(ex.map(_reward_eval, tasks, chunksize=2))
    return [_reward_eval(t) for t in tasks]


def _cell_cols(cell: RewardCell | None) -> dict[str, Any]:
    if cell is None:
        return {"cell": "", "w_L": math.nan, "w_M": math.nan, "w_R": math.nan}
    return {"cell": cell.name, "w_L": float(cell.w_L), "w_M": float(cell.w_M), "w_R": cell.w_R}


def _reuse_references(out: Path, labels: Sequence[tuple[str, str, int, str, Any]],
                      scenarios: Sequence[str], eps: Sequence[int],
                      cfg_hashes: Mapping[str, str], canon_q_sha: Sequence[str],
                      tuned_impl: str) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Reference rows from ``<out>/agent/per_seed.csv`` if they match exactly (else None)."""
    path, man = out / "agent" / "per_seed.csv", out / MANIFEST_NAME
    info: dict[str, Any] = {"candidate": str(path)}

    def no(reason: str) -> tuple[None, dict[str, Any]]:
        info.update(reused=False, reason=reason)
        return None, info

    if not path.is_file() or not man.is_file():
        return no("agent per_seed.csv / manifest not found")
    try:
        agent = json.loads(man.read_text(encoding="utf-8"))["suites"]["agent"]
        old_sha = [q["sha256"] for q in agent["canonical"]["q_tables"]]
        old_eval = agent["eval_seeds"]
    except (KeyError, TypeError, json.JSONDecodeError) as e:
        return no(f"agent manifest unusable: {e!r}")
    if list(old_sha) != list(canon_q_sha):
        return no("canonical Q-table hashes differ from the agent suite")
    if old_eval.get("split") != "test" or int(old_eval.get("run", -1)) != TEST_RUN or \
            not set(eps) <= set(old_eval.get("episodes", [])):
        return no("agent suite used different TEST seeds")
    old = pd.read_csv(path, float_precision="round_trip")
    names = [lab for lab, *_ in labels]
    sub = old[old["policy"].isin(names) & (old["split"] == "test") & (old["run"] == TEST_RUN)
              & old["scenario"].isin(list(scenarios)) & old["episode"].isin(list(eps))].copy()
    if len(sub) != len(names) * len(scenarios) * len(eps) or \
            sub.duplicated(list(KEY_COLUMNS)).any():
        return no(f"expected {len(names) * len(scenarios) * len(eps)} unique reference rows, "
                  f"found {len(sub)}")
    impl = dict(zip(sub["policy"], sub["policy_impl"]))
    if impl.get(GLOBAL_LABEL) != tuned_impl:
        return no(f"tuned_heuristic impl {impl.get(GLOBAL_LABEL)!r} != {tuned_impl!r}")
    for s in scenarios:
        hs = set(sub.loc[sub["scenario"] == s, "config_hash"])
        if hs != {cfg_hashes[s]}:
            return no(f"config hash mismatch for {s}: {sorted(hs)} vs {cfg_hashes[s]}")
    sub["J_train"] = sub["J"].astype(float)
    sub["J_recomputed"] = sub["ledger_score"].astype(float)
    sub["source"] = "reused:agent/per_seed.csv"
    info.update(reused=True, sha256=_sha256_file(path), n_rows=len(sub))
    return sub, info


def _ci_halfwidth(std: float, n: int) -> float:
    return 1.96 * std / math.sqrt(n) if n > 1 and math.isfinite(std) else math.nan


def run_reward_suite(configs_dir: Path | str,
                     manifest_path: Path | str = "results/headroom/manifest.json",
                     out_dir: Path | str = "results/ablations", *,
                     experiment_path: Path | str | None = None,
                     cells: Sequence[str] | None = None,
                     runs: Sequence[int] = RUNS, episodes: int | None = None,
                     n_seeds: int = DEFAULT_N_SEEDS,
                     scenarios: Sequence[str] = EVAL_SCENARIOS,
                     canonical_q: Sequence[Path | str] | None = None,
                     tune_selected: Path | str = "results/tune/selected.json",
                     workers: int | None = None, n_resamples: int = 10_000,
                     overwrite: bool = False, log_every: int = 0, reuse_references: bool = True,
                     progress: Callable[[str], None] | None = None) -> AblationReport:
    """Train, evaluate and compare the reward-weight cells (see :data:`REWARD_ANALYSIS_PLAN`)."""
    from scipy.stats import spearmanr

    from churnmaster.experiments.diagnostics import _frozen_loader
    from churnmaster.plots import plot_reward_sweep, plot_reward_tradeoff

    t0 = time.perf_counter()
    say = progress or (lambda m: None)
    configs_dir, manifest_path = Path(configs_dir), Path(manifest_path)
    out = guard_ablation_out(out_dir)
    suite_dir = out / "reward"
    if (suite_dir / "comparison.csv").exists() and not overwrite:
        raise ValueError(f"{suite_dir} already holds reward-sweep results; pass overwrite=True "
                         f"(--overwrite) to replace them")
    names = list(cells) if cells is not None else list(REWARD_CELLS)
    unknown = [n for n in names if n not in REWARD_CELLS]
    if unknown or not names or len(set(names)) != len(names):
        raise ValueError(f"bad cell list {names}; known: {list(REWARD_CELLS)} (the canonical "
                         f"cell {CANONICAL_CELL.name} reuses the canonical Q-tables)")
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
    cs = [REWARD_CELLS[n] for n in names]

    # ---- configs, frozen env (read-only), seeds --------------------------------
    loader = _frozen_loader(configs_dir, exp, manifest_path)
    all_scn = list(dict.fromkeys([*TRAIN_SCENARIOS, *scenarios]))
    cfgs: dict[str, Config] = {s: loader(s) for s in all_scn}
    for s, c in cfgs.items():
        w = (c.reward.w_L, c.reward.w_M, c.reward.w_R)
        if not all(math.isclose(a, b) for a, b in zip(w, CANONICAL_WEIGHTS)):
            raise ValueError(f"config reward weights {w} for {s} are not the canonical "
                             f"{CANONICAL_WEIGHTS}; the sweep scores under the canonical weights")
    env_hashes = {s: check_frozen(manifest_path, c) for s, c in cfgs.items()}
    cfg_hashes = {s: config_hash(c) for s, c in cfgs.items()}
    cell_loaders: dict[str, RewardLoader] = {}
    cell_cfg_hashes: dict[str, dict[str, str]] = {}
    for c in cs:
        ld = RewardLoader(loader, *c.weights)
        cell_loaders[c.name] = ld
        cell_cfg_hashes[c.name] = {}
        for s in all_scn:
            cc = ld(s)
            if check_frozen(manifest_path, cc) != env_hashes[s]:
                raise ValueError(f"cell {c.name} changes the env hash of {s}")
            cell_cfg_hashes[c.name][s] = config_hash(cc)
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
    base = dataclasses.replace(base, validate_every=base.episodes)
    canon_info = _check_canonical_settings(q_paths, base) if episodes is None else \
        _q_info_only(q_paths)
    canon_sha = [q["sha256"] for q in canon_info["q_tables"]]
    selected = load_selected(tune_selected)
    tuned_impl = tuned_policy_names(selected)[GLOBAL_LABEL]

    # ---- training (canonical path; only the Ledger weights differ) ----------------
    specs = [TrainSpec(cell_loaders[c.name], base, r, log_every) for c in cs for r in runs]
    say(f"ablate[reward]: training {len(specs)} runs ({len(cs)} cells x {len(runs)} runs x "
        f"{base.episodes} episodes, workers={workers}); {CANONICAL_CELL.name} reuses the "
        f"canonical Q-tables")
    t_tr = time.perf_counter()
    outputs: list[TrainOutput] = run_trainings(specs, workers=workers)
    train_wall = time.perf_counter() - t_tr
    say(f"ablate[reward]: training done in {train_wall:.1f}s")

    files: list[Path] = []
    trained: dict[str, dict[int, TrainOutput]] = {}
    it = iter(outputs)
    for c in cs:
        for r in runs:
            o = next(it)
            trained.setdefault(c.name, {})[r] = o
            d = suite_dir / "cells" / c.name / f"run{r}"
            d.mkdir(parents=True, exist_ok=True)
            files.append(o.final.save(d / "q_final.npy"))
            o.curves.to_csv(d / "curves.csv", index=False)
            files.append(d / "curves.csv")

    # ---- evaluation on TEST seeds (scored under the canonical weights) ------------
    ref_labels: list[tuple[str, str, int, str, Any]] = []
    for r, p in zip(runs, q_paths):
        lab = _run_label(CANONICAL_LABEL, r)
        ref_labels.append((lab, CANONICAL_LABEL, r, str(p),
                           EncodedLearned(QAgent.load(p), name=lab, id_bits=id_bits)))
    for c in CONTEXT_POLICIES:
        impl = tuned_impl if c == GLOBAL_LABEL else c
        ref_labels.append((c, c, -1, impl, impl))
    cell_of = {CANONICAL_LABEL: CANONICAL_CELL, **{c.name: c for c in cs}}

    tasks: list[_RewardEvalTask] = []
    meta: list[tuple[str, str, int, str, str]] = []      # (label, group, run, impl, role)
    for c in cs:
        for r in runs:
            lab = _run_label(c.name, r)
            pol = EncodedLearned(trained[c.name][r].final, name=lab, id_bits=id_bits)
            for s in scenarios:
                for ep in eps_list:
                    tasks.append(_RewardEvalTask(Job(pol, s, TEST_SPLIT, TEST_RUN, ep), loader,
                                                 c.weights))
                    meta.append((lab, c.name, r, f"reward:{c.name}", "cell"))
    reused, ref_info = (None, {"reused": False, "reason": "reuse disabled"})
    if reuse_references:
        reused, ref_info = _reuse_references(out, ref_labels, scenarios, eps_list, cfg_hashes,
                                             canon_sha, tuned_impl)
    ref_eps = eps_list[:1] if reused is not None else eps_list
    role = "spot_check" if reused is not None else "reference"
    for lab, grp, r, impl, pol in ref_labels:
        for s in scenarios:
            for ep in ref_eps:
                tasks.append(_RewardEvalTask(Job(pol, s, TEST_SPLIT, TEST_RUN, ep), loader,
                                             CANONICAL_WEIGHTS))
                meta.append((lab, grp, r, impl, role))
    say(f"ablate[reward]: evaluating {len(tasks)} TEST episodes (references "
        f"{'reused + ' + str(len(ref_labels) * len(scenarios)) + ' spot checks' if reused is not None else 're-run'})")
    t_ev = time.perf_counter()
    results = _run_reward_eval(tasks, workers)
    eval_wall = time.perf_counter() - t_ev
    say(f"ablate[reward]: evaluation done in {eval_wall:.1f}s")

    rows, spot = [], []
    for (lab, grp, r, impl, rl), row in zip(meta, results):
        row = dict(row)
        row.pop("policy")
        full_row = {"policy": lab, "group": grp, "rl_run": r, "policy_impl": impl,
                    **_cell_cols(cell_of.get(grp)), **row, "source": "evaluated"}
        (spot if rl == "spot_check" else rows).append(full_row)
    if reused is not None:
        ref_info["spot_checks"] = []
        old = reused.set_index(["policy", "scenario", "episode"])
        ok = True
        for sr in spot:
            o = old.loc[(sr["policy"], sr["scenario"], sr["episode"])]
            match = (o["trace_hash"] == sr["trace_hash"] and o["summary_hash"] == sr["summary_hash"]
                     and _same_float(float(o["J"]), float(sr["J"])))
            ok &= bool(match)
            ref_info["spot_checks"].append({"policy": sr["policy"], "scenario": sr["scenario"],
                                            "episode": sr["episode"], "match": bool(match)})
        if ok:
            ref_rows = reused.copy()
            for k, v in _cell_cols(None).items():
                ref_rows[k] = v
            for k, v in _cell_cols(CANONICAL_CELL).items():
                ref_rows.loc[ref_rows["group"] == CANONICAL_LABEL, k] = v
            rows_df = pd.concat([pd.DataFrame(rows), ref_rows[list(pd.DataFrame(rows).columns)]],
                                ignore_index=True)
        else:
            say("ablate[reward]: reference spot check failed; re-running the references")
            ref_info.update(reused=False, reason="spot check mismatch (re-run)")
            rt, rm = [], []
            for lab, grp, r, impl, pol in ref_labels:
                for s in scenarios:
                    for ep in eps_list:
                        rt.append(_RewardEvalTask(Job(pol, s, TEST_SPLIT, TEST_RUN, ep), loader,
                                                  CANONICAL_WEIGHTS))
                        rm.append((lab, grp, r, impl))
            t_re = time.perf_counter()
            for (lab, grp, r, impl), row in zip(rm, _run_reward_eval(rt, workers)):
                row = dict(row)
                row.pop("policy")
                rows.append({"policy": lab, "group": grp, "rl_run": r, "policy_impl": impl,
                             **_cell_cols(cell_of.get(grp)), **row, "source": "evaluated"})
            eval_wall += time.perf_counter() - t_re
            rows_df = pd.DataFrame(rows)
    else:
        rows_df = pd.DataFrame(rows)
    df = rows_df
    if df.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("duplicate (label, scenario, split, run, episode) rows")
    for m in REWARD_SUMMARY_METRICS:
        bad = df[~df[m].astype(float).map(math.isfinite)]
        if not bad.empty:
            raise ValueError(f"non-finite {m} in {len(bad)} reward-sweep rows, e.g. "
                             f"{bad[['policy', 'scenario', 'episode']].iloc[0].to_dict()}")
    j_dev = float((df["J_recomputed"].astype(float) - df["ledger_score"].astype(float)).abs().max())
    refs_mask = df["group"].isin([CANONICAL_LABEL, *CONTEXT_POLICIES])
    if not np.allclose(df.loc[refs_mask, "J_train"], df.loc[refs_mask, "J"], rtol=0, atol=_J_TOL):
        raise RuntimeError("J_train != J for a reference scored under the canonical weights")

    # ---- statistics (exploratory; uncorrected) ------------------------------------
    groups = [CANONICAL_LABEL, *[c.name for c in cs]]
    means = []
    for g in groups:
        avg = st.average_over_runs(df, [_run_label(g, r) for r in runs], g,
                                   metrics=list(REWARD_SUMMARY_METRICS))
        means.append(avg.assign(group=g, rl_run=-1, policy_impl=f"rl_mean({g})",
                                **_cell_cols(cell_of[g])))
    full = pd.concat([df, *means], ignore_index=True)

    def crow(comparison: str, s: str, m: str, a: str, b: str) -> dict[str, Any]:
        cc = cell_of.get(a)
        return {"comparison": comparison, **_cell_cols(cc),
                **_compare_row(full, s, m, a, b, n_resamples)}

    comp = []
    for c in cs:
        for s in scenarios:
            for m in REWARD_METRICS:
                comp.append(crow("vs_canonical", s, m, c.name, CANONICAL_LABEL))
    for g in groups:
        for s in scenarios:
            for m in GAP_METRICS:
                comp.append(crow("vs_tuned_heuristic", s, m, g, GLOBAL_LABEL))
    comparison = pd.DataFrame(comp)

    pr = []
    for g in groups:
        for r in runs:
            lab = _run_label(g, r)
            for s in scenarios:
                sub = df[(df["policy"] == lab) & (df["scenario"] == s)]
                row: dict[str, Any] = {"group": g, **_cell_cols(cell_of[g]), "run": r,
                                       "label": lab, "scenario": s, "n": len(sub)}
                for m in REWARD_SUMMARY_METRICS:
                    row[f"{m}_mean"] = float(sub[m].astype(float).mean())
                for m, tag in (("J", "dJ"), ("p95_latency_all_ms", "dp95")):
                    c = _compare_row(full, s, m, lab, CANONICAL_LABEL, n_resamples)
                    row.update({f"{tag}_vs_canonical_mean": c["mean_diff"],
                                f"{tag}_ci_lo": c["ci_lo"], f"{tag}_ci_hi": c["ci_hi"],
                                f"{tag}_p_uncorrected": c["p_uncorrected"]})
                pr.append(row)
    per_run = pd.DataFrame(pr)

    srows = []
    for (pol, s), g in full.groupby(["policy", "scenario"], sort=False):
        grp = g["group"].iloc[0]
        kind = "reference" if grp in CONTEXT_POLICIES else ("rl_mean" if pol == grp else "run")
        row = {"policy": pol, "group": grp, "kind": kind, **_cell_cols(cell_of.get(grp)),
               "scenario": s, "n": len(g)}
        for m in REWARD_SUMMARY_METRICS:
            x = g[m].astype(float)
            row[f"{m}_mean"] = float(x.mean())
            row[f"{m}_std"] = float(x.std(ddof=1)) if len(x) > 1 else math.nan
        srows.append(row)
    summary = pd.DataFrame(srows)

    trows = []
    line = summary[(summary["kind"] == "rl_mean") &
                   np.isclose(summary["w_M"].astype(float), REWARD_LINE_W_M)]
    for s in scenarios:
        pts = line[line["scenario"] == s].sort_values("w_L")
        x = pts["w_L"].astype(float).to_numpy()
        y = pts["p95_latency_all_ms_mean"].astype(float).to_numpy()
        rho = float(spearmanr(x, y)[0]) if len(x) >= 2 and np.ptp(y) > 0 else math.nan
        trows.append({"scenario": s, "metric": "p95_latency_all_ms", "w_M": REWARD_LINE_W_M,
                      "n_points": len(x), "w_L_values": json.dumps([float(v) for v in x]),
                      "rl_mean_values": json.dumps([float(v) for v in y]),
                      "cells": json.dumps(list(pts["cell"])), "spearman_rho": rho,
                      "note": TREND_NOTE})
    trend = pd.DataFrame(trows)

    greedy: dict[str, Any] = {}
    for r, p in zip(runs, q_paths):
        ent: dict[str, Any] = {"greedy": greedy_stats(QAgent.load(p))}
        cpath = Path(p).parent / "curves.csv"
        if cpath.is_file():
            ent["dq"] = dq_summary(pd.read_csv(cpath))
        greedy.setdefault(CANONICAL_LABEL, {})[f"run{r}"] = ent
    for c in cs:
        for r in runs:
            o = trained[c.name][r]
            greedy.setdefault(c.name, {})[f"run{r}"] = {
                "weights": c.to_dict(), "greedy": greedy_stats(o.final), "dq": dq_summary(o.curves),
                "train_wall_s": o.wall_s,
                "n_updates_total": int(o.curves["n_updates"].sum()),
                "n_transitions_total": int(o.curves["n_transitions"].sum())}

    # ---- write --------------------------------------------------------------------
    suite_dir.mkdir(parents=True, exist_ok=True)
    tables = {"per_seed.csv": df, "comparison.csv": comparison, "per_run.csv": per_run,
              "summary.csv": summary, "trend.csv": trend}
    for fname, t in tables.items():
        t.to_csv(suite_dir / fname, index=False)
        files.append(suite_dir / fname)
    files.append(_write_json(suite_dir / "greedy_stats.json", greedy))
    fig_dir = out / "figures"
    refs_plot = (GLOBAL_LABEL, "kademlia_epoch")
    files.append(plot_reward_sweep(
        summary, "p95_latency_all_ms", fig_dir / "reward_sweep_p95.png", references=refs_plot,
        ylabel="RL p95 latency, all lookups (ms)",
        title="Reward sweep: RL p95 vs w_L per w_M (TEST seeds; mean ± 95% CI over seeds)"))
    files.append(plot_reward_sweep(
        summary, "J", fig_dir / "reward_sweep_J.png", references=refs_plot,
        ylabel="J under the canonical weights",
        title="Reward sweep: canonical-weight J vs w_L per w_M (TEST seeds; mean ± 95% CI)"))
    files.append(plot_reward_tradeoff(
        summary, fig_dir / "reward_sweep_tradeoff.png",
        references=(GLOBAL_LABEL, "kademlia_epoch", "never_replace"),
        title="Reward sweep: p95 vs maintenance pings/h (cells = RL mean over runs)"))

    wall = time.perf_counter() - t0
    suite_manifest = {
        "suite": "reward", "created_utc": _now(), "git_hash": git_hash(),
        "packages": package_versions(("numpy", "scipy", "pandas", "matplotlib", "churnmaster")),
        "analysis_plan": REWARD_ANALYSIS_PLAN, "exploratory": True,
        "multiple_comparison_correction": "none (exploratory; p-values uncorrected)",
        "caveat": REWARD_CAVEAT, "trend_note": TREND_NOTE,
        "frozen_manifest": {"path": str(manifest_path), "sha256": _sha256_file(manifest_path),
                            "access": "read-only (check_frozen)"},
        "env_hashes": env_hashes, "config_hashes": cfg_hashes,
        "cell_config_hashes": cell_cfg_hashes,
        "experiment": str(exp), "root_seed": root,
        "scoring_weights": dict(zip(("w_L", "w_M", "w_R"), CANONICAL_WEIGHTS)),
        "cells": {c.name: c.to_dict() for c in [*cs, CANONICAL_CELL]},
        "w_R_rule": f"w_R = {W_R_PER_W_M} * w_M",
        "train_settings": base.to_dict(), "train_split": "train", "runs": runs,
        "eval_seeds": disjoint, "eval_scenarios": list(scenarios),
        "canonical": canon_info, "references": ref_info,
        "context_policies": {c: (tuned_impl if c == GLOBAL_LABEL else c) for c in CONTEXT_POLICIES},
        "tune_selected": {"path": str(tune_selected), "global": selected.get("global")},
        "j_recomputation": {"method": "rl.reward.lookup_loss / action_cost over measured "
                                      "records", "max_abs_dev_vs_ledger": j_dev,
                            "tolerance": _J_TOL},
        "stats": {"bootstrap_resamples": n_resamples, "ci_level": 0.95, "bootstrap_seed": 0,
                  "wilcoxon": "two-sided, zero_method=wilcox", "pairing": list(st.EVAL_KEYS),
                  "rl_mean": "per-seed mean over the runs of a cell"},
        "n_train_runs": len(specs), "n_eval_jobs": len(tasks), "workers": workers,
        "timings_s": {"train": round(train_wall, 1), "eval": round(eval_wall, 1),
                      "total": round(wall, 1),
                      "per_train_run": {_run_label(c.name, r): round(trained[c.name][r].wall_s, 1)
                                        for c in cs for r in runs}},
        "outputs": sorted(str(Path(f).relative_to(out)) for f in files),
        "per_seed_sha256": _sha256_file(suite_dir / "per_seed.csv"),
    }
    mpath = out / MANIFEST_NAME
    top: dict[str, Any] = {}
    if mpath.is_file():
        top = json.loads(mpath.read_text(encoding="utf-8"))
    top.setdefault("kind", "churnmaster-ablations")
    top["exploratory"] = True
    top["updated_utc"] = _now()
    top["reward_analysis_plan"] = REWARD_ANALYSIS_PLAN
    top.setdefault("suites", {})["reward"] = suite_manifest
    write_manifest(mpath, _clean(top))
    files.append(mpath)
    say(f"ablate[reward]: wrote {len(files)} files to {out} ({wall:.1f}s)")
    return AblationReport("reward", out, df, comparison, per_run, summary, greedy, suite_manifest,
                          files, wall, train_wall, eval_wall, trend=trend)


# =========================================================================== #
# Memoryless held-out scenario (Task 12.3, Requirement 28.3)
# =========================================================================== #

MEMORYLESS_SCENARIO = "heldout_memoryless"
MEMORYLESS_PARENT = "heldout_medium"
MEMORYLESS_SHAPE = 1.0
#: Subdirectory of ``configs/`` holding exploratory scenarios. Deliberately NOT
#: ``configs/scenarios``: headroom / reproduce freeze every file there, so an
#: ablation scenario in that directory would leak into canonical manifests.
ABLATION_CONFIG_DIR = "ablations"
#: Fields in which the memoryless scenario may differ from its frozen parent.
MEMORYLESS_DIFF_FIELDS: tuple[str, ...] = ("name", "weibull_shape")
MEMORYLESS_SCENARIOS: tuple[str, ...] = (MEMORYLESS_SCENARIO, MEMORYLESS_PARENT)
MEMORYLESS_POLICIES: tuple[str, ...] = (GLOBAL_LABEL, "kademlia_epoch", "matched_random",
                                        "never_replace", "uniform_random", "oracle_heuristic")
GAP_REFERENCES: tuple[str, ...] = (GLOBAL_LABEL, "kademlia_epoch", "matched_random")
RL_MEAN_LABEL = "rl_mean"
MEMORYLESS_SUMMARY_METRICS: tuple[str, ...] = SUMMARY_METRICS + ("mean_latency_ms",)
FREQ_SEEDS = 5
SHAPE_PAIR = f"{MEMORYLESS_SCENARIO}-{MEMORYLESS_PARENT}"
MEMORYLESS_CAVEAT = ("One held-out scenario, 20 TEST seeds, no multiple-comparison correction; "
                     "the canonical policies were neither retrained nor re-tuned on it.")


def check_derived_scenario(manifest_path: Path | str, cfg: Config, parent_cfg: Config,
                           fields: Sequence[str] = MEMORYLESS_DIFF_FIELDS) -> dict[str, Any]:
    """Check that ``cfg``'s env equals the frozen ``parent_cfg`` env except for ``fields``.

    ``parent_cfg`` must pass :func:`check_frozen`; substituting the parent's
    ``scenario.<field>`` values into ``cfg`` must reproduce the frozen parent
    env hash (``FrozenEnvError`` otherwise). Returns the derived scenario's
    hashes and the differing field values. The manifest is only read.
    """
    from churnmaster.experiments.runner import FrozenEnvError, read_manifest

    parent_hash = check_frozen(manifest_path, parent_cfg)
    sub = dataclasses.replace(cfg, scenario=dataclasses.replace(
        cfg.scenario, **{f: getattr(parent_cfg.scenario, f) for f in fields}))
    if env_config_hash(sub) != parent_hash:
        raise FrozenEnvError(
            f"scenario {cfg.scenario.name!r} differs from the frozen {parent_cfg.scenario.name!r} "
            f"env in more than {list(fields)}; refusing to run")
    frozen = read_manifest(manifest_path).get("env_hashes") or {}
    in_manifest = cfg.scenario.name in frozen
    if in_manifest:
        check_frozen(manifest_path, cfg)
    diffs = {f: [getattr(parent_cfg.scenario, f), getattr(cfg.scenario, f)] for f in fields
             if getattr(parent_cfg.scenario, f) != getattr(cfg.scenario, f)}
    return {"name": cfg.scenario.name, "parent": parent_cfg.scenario.name,
            "parent_env_hash": parent_hash, "env_hash": env_config_hash(cfg),
            "config_hash": config_hash(cfg), "allowed_diff_fields": list(fields),
            "differs": diffs, "in_frozen_manifest": in_manifest,
            "scenario": dataclasses.asdict(cfg.scenario)}


def _stat_row(xa: np.ndarray, xb: np.ndarray, metric: str, n_resamples: int) -> dict[str, Any]:
    ci = st.paired_bootstrap_ci(xa, xb, n_resamples=n_resamples, level=0.95, seed=0)
    w = st.wilcoxon_paired(xa, xb)
    sign = st.METRIC_DIRECTION.get(metric, +1)
    return {"metric": metric, "n": ci.n, "mean_variant": float(np.mean(xa)),
            "mean_reference": float(np.mean(xb)), "mean_diff": ci.mean, "ci_lo": ci.lo,
            "ci_hi": ci.hi, "excludes_zero": ci.excludes_zero, "wilcoxon_stat": w.statistic,
            "p_uncorrected": w.p, "favourable": int(np.sign(sign * ci.mean))}


def _by_episode(full: pd.DataFrame, policy: str, scenario: str, metric: str) -> pd.Series:
    sub = full[(full["policy"] == policy) & (full["scenario"] == scenario)]
    if sub.empty or sub["episode"].duplicated().any():
        raise ValueError(f"need unique rows for {policy}/{scenario}, got {len(sub)}")
    return sub.set_index("episode")[metric].astype(float).sort_index()


def _measure_freqs(q_paths: Sequence[Path], loader: Callable[[str], Config], scenario: str,
                   n_seeds: int, id_bits: int, workers: int | None) -> dict[str, Any]:
    """Task 9.2 procedure: canonical greedy RL on VALIDATION run 0, episodes 0..n-1, pooled."""
    from churnmaster.policies.baselines import action_counts, action_frequencies
    from churnmaster.policies.learned import Learned

    pols = [Learned.load(p, name=f"learned_run{i}", id_bits=id_bits) for i, p in enumerate(q_paths)]
    jobs = [Job(pol, scenario, Split.VALIDATION, 0, ep) for pol in pols for ep in range(n_seeds)]
    results = run_jobs(jobs, cfg_loader=loader, keep_records=True, workers=workers)
    counts = action_counts([d for r in results for d in (r.decisions or [])])
    freqs = action_frequencies(counts)
    return {"freqs": freqs,
            "counts": {n: [int(x) for x in counts[k]] for k, n in ((False, "no_cand"),
                                                                   (True, "cand"))},
            "n_episodes": len(jobs), "split": "validation", "run": 0,
            "episodes": [0, n_seeds - 1]}


def _fmt_ci(r: Mapping[str, Any] | None, d: int) -> str:
    if r is None:
        return "-"
    return f"{r['mean_diff']:+.{d}f} [{r['ci_lo']:+.{d}f}, {r['ci_hi']:+.{d}f}]"


def _memoryless_markdown(summary: pd.DataFrame, deltas: pd.DataFrame, policies: Sequence[str],
                         scenario_info: Mapping[str, Any]) -> str:
    def s_get(pol: str, scn: str, m: str) -> float:
        r = summary[(summary["policy"] == pol) & (summary["scenario"] == scn)]
        return float(r[f"{m}_mean"].iloc[0]) if not r.empty else math.nan

    def d_get(comp: str, a: str, b: str, scn: str, m: str) -> Mapping[str, Any] | None:
        r = deltas[(deltas["comparison"] == comp) & (deltas["variant"] == a)
                   & (deltas["reference"] == b) & (deltas["scenario"] == scn)
                   & (deltas["metric"] == m)]
        return None if r.empty else r.iloc[0].to_dict()

    M, P = MEMORYLESS_SCENARIO, MEMORYLESS_PARENT
    sh = scenario_info.get("differs", {}).get("weibull_shape", ["?", "?"])
    lines = [
        "# Memoryless held-out scenario (Task 12.3, exploratory)",
        "",
        f"`{M}` = `{P}` with Weibull shape {sh[1]} (frozen: {sh[0]}); mean session "
        "lengths unchanged. TEST split, run 0, episodes 0..19. RL = rl_mean over the 3 "
        "canonical runs. 95% paired bootstrap CIs; p-values uncorrected (no Holm).",
        "",
        "## Per-policy means and shape effect",
        "",
        f"| policy | J ({M}) | J ({P}) | ΔJ shape [95% CI] | p95 ms ({M}) | p95 ms ({P}) | "
        f"Δp95 shape [95% CI] | success ({M}) | pings/h ({M}) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for pol in policies:
        lines.append(
            f"| {pol} | {s_get(pol, M, 'J'):.4f} | {s_get(pol, P, 'J'):.4f} | "
            f"{_fmt_ci(d_get('shape_effect', pol, pol, SHAPE_PAIR, 'J'), 4)} | "
            f"{s_get(pol, M, 'p95_latency_all_ms'):.1f} | {s_get(pol, P, 'p95_latency_all_ms'):.1f} | "
            f"{_fmt_ci(d_get('shape_effect', pol, pol, SHAPE_PAIR, 'p95_latency_all_ms'), 1)} | "
            f"{s_get(pol, M, 'success_rate'):.4f} | {s_get(pol, M, 'pings_per_hour'):.1f} |")
    lines += ["", "## RL (rl_mean) vs references", "",
              "| reference | scenario | ΔJ [95% CI] | p (J) | Δp95 ms [95% CI] | Δpings/h [95% CI] |",
              "|---|---|---|---|---|---|"]
    for ref in [p for p in policies if p != RL_MEAN_LABEL]:
        for scn in (M, P):
            dj = d_get("rl_vs_reference", RL_MEAN_LABEL, ref, scn, "J")
            lines.append(
                f"| {ref} | {scn} | {_fmt_ci(dj, 4)} | "
                f"{'-' if dj is None else format(dj['p_uncorrected'], '.3g')} | "
                f"{_fmt_ci(d_get('rl_vs_reference', RL_MEAN_LABEL, ref, scn, 'p95_latency_all_ms'), 1)} | "
                f"{_fmt_ci(d_get('rl_vs_reference', RL_MEAN_LABEL, ref, scn, 'pings_per_hour'), 1)} |")
    lines += ["", f"## Gap shift: [RL − ref]_{M} − [RL − ref]_{P}", "",
              "| reference | ΔΔJ [95% CI] | p (J) | ΔΔp95 ms [95% CI] |", "|---|---|---|---|"]
    for ref in GAP_REFERENCES:
        gj = d_get("gap_shift", RL_MEAN_LABEL, ref, SHAPE_PAIR, "J")
        lines.append(f"| {ref} | {_fmt_ci(gj, 4)} | "
                     f"{'-' if gj is None else format(gj['p_uncorrected'], '.3g')} | "
                     f"{_fmt_ci(d_get('gap_shift', RL_MEAN_LABEL, ref, SHAPE_PAIR, 'p95_latency_all_ms'), 1)} |")
    lines += ["", f"Caveat: {MEMORYLESS_CAVEAT}", ""]
    return "\n".join(lines)


def _cross_check_agent(out: Path, df: pd.DataFrame, scenario: str) -> dict[str, Any]:
    """Compare heldout_medium rows with ``<out>/agent/per_seed.csv`` (same labels / seeds)."""
    path = out / "agent" / "per_seed.csv"
    info: dict[str, Any] = {"candidate": str(path), "scenario": scenario}
    if not path.is_file():
        info.update(checked=False, reason="agent per_seed.csv not found")
        return info
    old = pd.read_csv(path, float_precision="round_trip")
    old = old[(old["scenario"] == scenario) & (old["split"] == "test") & (old["run"] == TEST_RUN)]
    key = ["policy", "scenario", "episode"]
    m = df[df["scenario"] == scenario].merge(old, on=key, suffixes=("", "_agent"))
    if m.empty:
        info.update(checked=False, reason="no common (label, episode) rows")
        return info
    ok = (m["trace_hash"] == m["trace_hash_agent"]) & (m["summary_hash"] == m["summary_hash_agent"])
    info.update(checked=True, n_rows=int(len(m)), labels=sorted(set(m["policy"])),
                all_match=bool(ok.all()), n_mismatch=int((~ok).sum()),
                sha256=_sha256_file(path))
    return info


def run_memoryless_suite(configs_dir: Path | str,
                         manifest_path: Path | str = "results/headroom/manifest.json",
                         out_dir: Path | str = "results/ablations", *,
                         experiment_path: Path | str | None = None,
                         policies: Sequence[str] | None = None,
                         runs: Sequence[int] = RUNS, n_seeds: int = DEFAULT_N_SEEDS,
                         canonical_q: Sequence[Path | str] | None = None,
                         tune_selected: Path | str = "results/tune/selected.json",
                         matched_freqs: Path | str = "results/matched/freqs.json",
                         freq_seeds: int = FREQ_SEEDS,
                         workers: int | None = None, n_resamples: int = 10_000,
                         overwrite: bool = False,
                         progress: Callable[[str], None] | None = None) -> AblationReport:
    """Evaluate the canonical policies on the memoryless held-out scenario (see plan)."""
    from churnmaster.experiments.diagnostics import _frozen_loader
    from churnmaster.experiments.matched import load_matched_freqs
    from churnmaster.plots import plot_headroom_bars

    t0 = time.perf_counter()
    say = progress or (lambda m: None)
    configs_dir, manifest_path = Path(configs_dir), Path(manifest_path)
    out = guard_ablation_out(out_dir)
    suite_dir = out / "memoryless"
    if (suite_dir / "summary.csv").exists() and not overwrite:
        raise ValueError(f"{suite_dir} already holds memoryless results; pass overwrite=True "
                         f"(--overwrite) to replace them")
    pols = list(policies) if policies is not None else list(MEMORYLESS_POLICIES)
    unknown = [p for p in pols if p not in MEMORYLESS_POLICIES]
    if unknown or not pols or len(set(pols)) != len(pols):
        raise ValueError(f"bad policy list {pols}; known: {list(MEMORYLESS_POLICIES)}")
    runs = [int(r) for r in runs]
    if not runs or len(set(runs)) != len(runs):
        raise ValueError(f"runs must be unique, got {runs}")
    if n_seeds < 1 or freq_seeds < 1:
        raise ValueError("n_seeds and freq_seeds must be >= 1")
    exp = Path(experiment_path) if experiment_path is not None else \
        configs_dir / "experiments" / "train.toml"
    q_paths = [Path(p) for p in (canonical_q if canonical_q is not None else default_q_paths(len(runs)))]
    if len(q_paths) != len(runs):
        raise ValueError(f"need one canonical Q-table per run ({len(runs)}), got {len(q_paths)}")
    toml_path = configs_dir / ABLATION_CONFIG_DIR / f"{MEMORYLESS_SCENARIO}.toml"
    if not toml_path.is_file():
        raise FileNotFoundError(f"memoryless scenario config not found: {toml_path}")

    # ---- configs: parent frozen (read-only), memoryless derived ------------------
    loader = _frozen_loader(configs_dir, exp, manifest_path, (ABLATION_CONFIG_DIR,))
    cfgs: dict[str, Config] = {s: loader(s) for s in MEMORYLESS_SCENARIOS}
    mcfg, pcfg = cfgs[MEMORYLESS_SCENARIO], cfgs[MEMORYLESS_PARENT]
    if not math.isclose(float(mcfg.scenario.weibull_shape), MEMORYLESS_SHAPE):
        raise ValueError(f"{toml_path} has weibull_shape={mcfg.scenario.weibull_shape}, "
                         f"expected {MEMORYLESS_SHAPE}")
    scn_info = check_derived_scenario(manifest_path, mcfg, pcfg)
    scn_info.update({"toml": toml_path.as_posix(), "toml_sha256": _sha256_file(toml_path)})
    env_hashes = {MEMORYLESS_PARENT: scn_info["parent_env_hash"],
                  MEMORYLESS_SCENARIO: scn_info["env_hash"]}
    cfg_hashes = {s: config_hash(c) for s, c in cfgs.items()}
    roots = {int(c.root_seed) for c in cfgs.values()}
    if len(roots) != 1:
        raise ValueError(f"scenarios disagree on root_seed: {sorted(roots)}")
    root = roots.pop()
    eps_list = list(range(int(n_seeds)))
    disjoint = check_test_seeds_disjoint(root, eps_list)
    id_bits = pcfg.sim.id_bits
    canon_info = _q_info_only(q_paths)
    canon_info.pop("note", None)
    selected = load_selected(tune_selected)
    tuned_impl = tuned_policy_names(selected)[GLOBAL_LABEL]
    rw = pcfg.reward

    # ---- matched_random frequencies ----------------------------------------------
    freqs: dict[str, dict[bool, tuple[float, float, float]]] = {}
    freq_info: dict[str, Any] = {}
    t_fq = time.perf_counter()
    if "matched_random" in pols:
        canon_freqs, canon_raw = {}, {}
        mf = Path(matched_freqs)
        if mf.is_file():
            canon_freqs = load_matched_freqs(mf)
            canon_raw = json.loads(mf.read_text(encoding="utf-8"))
        for s in MEMORYLESS_SCENARIOS:
            raw = canon_raw.get(s) or {}
            if s in canon_freqs and raw.get("env_hash") == env_hashes[s]:
                freqs[s] = canon_freqs[s]
                freq_info[s] = {"source": f"canonical:{mf.as_posix()} (read-only)",
                                "counts": raw.get("counts"), "env_hash": raw.get("env_hash"),
                                "n_episodes": raw.get("n_episodes"), "split": raw.get("split")}
            else:
                say(f"ablate[memoryless]: measuring matched_random frequencies for {s} "
                    f"(VALIDATION run 0, {freq_seeds} episodes x {len(q_paths)} Q-tables)")
                m = _measure_freqs(q_paths, loader, s, freq_seeds, id_bits, workers)
                freqs[s] = m.pop("freqs")
                freq_info[s] = {"source": "measured (task 9.2 procedure)",
                                "env_hash": env_hashes[s], **m}
            freq_info[s]["no_cand"] = [float(x) for x in freqs[s][False]]
            freq_info[s]["cand"] = [float(x) for x in freqs[s][True]]
    freq_wall = time.perf_counter() - t_fq

    # ---- evaluation on TEST seeds ------------------------------------------------
    labels: list[tuple[str, str, int, str, Any]] = []
    for r, p in zip(runs, q_paths):
        lab = _run_label(CANONICAL_LABEL, r)
        labels.append((lab, CANONICAL_LABEL, r, str(p),
                       EncodedLearned(QAgent.load(p), name=lab, id_bits=id_bits)))
    for c in pols:
        impl = tuned_impl if c == GLOBAL_LABEL else c
        labels.append((c, c, -1, impl, impl))
    jobs, meta = [], []
    for s in MEMORYLESS_SCENARIOS:
        for lab, grp, r, impl, pol in labels:
            kw = {"freqs": freqs[s]} if grp == "matched_random" else {}
            for ep in eps_list:
                jobs.append(Job(pol, s, TEST_SPLIT, TEST_RUN, ep, kw))
                meta.append((lab, grp, r, impl))
    say(f"ablate[memoryless]: evaluating {len(labels)} labels x {len(MEMORYLESS_SCENARIOS)} "
        f"scenarios x {len(eps_list)} TEST seeds = {len(jobs)} episodes (workers={workers})")
    t_ev = time.perf_counter()
    results = run_jobs(jobs, cfg_loader=loader, workers=workers)
    eval_wall = time.perf_counter() - t_ev
    say(f"ablate[memoryless]: evaluation done in {eval_wall:.1f}s")

    rows = []
    for (lab, grp, r, impl), res in zip(meta, results):
        row = res.row()
        row.pop("policy")
        rows.append({"policy": lab, "group": grp, "rl_run": r, "policy_impl": impl, **row})
    df = pd.DataFrame(rows)
    if df.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("duplicate (label, scenario, split, run, episode) rows")
    for s in MEMORYLESS_SCENARIOS:
        hs = set(df.loc[df["scenario"] == s, "config_hash"])
        if hs != {cfg_hashes[s]}:
            raise RuntimeError(f"config hash mismatch for {s}: {sorted(hs)}")
    for m in MEMORYLESS_SUMMARY_METRICS:
        bad = df[~df[m].astype(float).map(math.isfinite)]
        if not bad.empty:
            raise ValueError(f"non-finite {m} in {len(bad)} memoryless rows, e.g. "
                             f"{bad[['policy', 'scenario', 'episode']].iloc[0].to_dict()}")

    # ---- statistics (exploratory; uncorrected) ------------------------------------
    avg = st.average_over_runs(df, [_run_label(CANONICAL_LABEL, r) for r in runs], RL_MEAN_LABEL,
                               metrics=list(MEMORYLESS_SUMMARY_METRICS))
    avg = avg.assign(group=CANONICAL_LABEL, rl_run=-1, policy_impl=f"rl_mean({CANONICAL_LABEL})")
    full = pd.concat([df, avg], ignore_index=True)
    shown = [RL_MEAN_LABEL, *pols]

    comp: list[dict[str, Any]] = []
    for s in MEMORYLESS_SCENARIOS:
        for ref in pols:
            for m in METRICS:
                a, b = _by_episode(full, RL_MEAN_LABEL, s, m), _by_episode(full, ref, s, m)
                comp.append({"comparison": "rl_vs_reference", "variant": RL_MEAN_LABEL,
                             "reference": ref, "scenario": s,
                             **_stat_row(a.to_numpy(), b.loc[a.index].to_numpy(), m, n_resamples)})
    for pol in [*shown, *[_run_label(CANONICAL_LABEL, r) for r in runs]]:
        for m in METRICS:
            a = _by_episode(full, pol, MEMORYLESS_SCENARIO, m)
            b = _by_episode(full, pol, MEMORYLESS_PARENT, m)
            comp.append({"comparison": "shape_effect", "variant": pol, "reference": pol,
                         "scenario": SHAPE_PAIR,
                         **_stat_row(a.to_numpy(), b.loc[a.index].to_numpy(), m, n_resamples)})
    for ref in [g for g in GAP_REFERENCES if g in pols]:
        for m in ("J", "p95_latency_all_ms"):
            gm = _by_episode(full, RL_MEAN_LABEL, MEMORYLESS_SCENARIO, m) - \
                _by_episode(full, ref, MEMORYLESS_SCENARIO, m)
            gp = _by_episode(full, RL_MEAN_LABEL, MEMORYLESS_PARENT, m) - \
                _by_episode(full, ref, MEMORYLESS_PARENT, m)
            comp.append({"comparison": "gap_shift", "variant": RL_MEAN_LABEL, "reference": ref,
                         "scenario": SHAPE_PAIR,
                         **_stat_row(gm.to_numpy(), gp.loc[gm.index].to_numpy(), m, n_resamples)})
    deltas = pd.DataFrame(comp)

    srows = []
    for (pol, s), g in full.groupby(["policy", "scenario"], sort=False):
        grp = g["group"].iloc[0]
        kind = "rl_mean" if pol == RL_MEAN_LABEL else ("run" if grp == CANONICAL_LABEL
                                                        else "reference")
        row = {"policy": pol, "group": grp, "kind": kind, "policy_impl": g["policy_impl"].iloc[0],
               "scenario": s, "n": len(g)}
        for m in MEMORYLESS_SUMMARY_METRICS:
            x = g[m].astype(float)
            row[f"{m}_mean"] = float(x.mean())
            row[f"{m}_std"] = float(x.std(ddof=1)) if len(x) > 1 else math.nan
        srows.append(row)
    summary = pd.DataFrame(srows)
    xcheck = _cross_check_agent(out, df, MEMORYLESS_PARENT)
    if xcheck.get("checked") and not xcheck.get("all_match"):
        say(f"ablate[memoryless]: WARNING {xcheck['n_mismatch']} heldout_medium rows differ "
            f"from {xcheck['candidate']}")

    # ---- write --------------------------------------------------------------------
    files: list[Path] = []
    suite_dir.mkdir(parents=True, exist_ok=True)
    for fname, t in {"per_seed.csv": df, "summary.csv": summary, "deltas.csv": deltas}.items():
        t.to_csv(suite_dir / fname, index=False)
        files.append(suite_dir / fname)
    if freq_info:
        files.append(_write_json(suite_dir / "matched_freqs.json", freq_info))
    md = suite_dir / "summary.md"
    md.write_text(_memoryless_markdown(summary, deltas, shown, scn_info), encoding="utf-8")
    files.append(md)
    plot_df = pd.concat([full[(full["policy"] == p) & (full["scenario"] == s)]
                         for p in shown for s in MEMORYLESS_SCENARIOS], ignore_index=True)
    files.append(plot_headroom_bars(
        plot_df, out / "figures" / "memoryless_J_p95.png",
        title=f"Memoryless (shape 1.0) vs {MEMORYLESS_PARENT} (shape 0.6): mean ± std over "
              f"TEST seeds (RL = rl_mean of {len(runs)} runs)"))

    wall = time.perf_counter() - t0
    suite_manifest = {
        "suite": "memoryless", "created_utc": _now(), "git_hash": git_hash(),
        "packages": package_versions(("numpy", "scipy", "pandas", "matplotlib", "churnmaster")),
        "analysis_plan": MEMORYLESS_ANALYSIS_PLAN, "exploratory": True,
        "multiple_comparison_correction": "none (exploratory; p-values uncorrected)",
        "caveat": MEMORYLESS_CAVEAT,
        "frozen_manifest": {"path": str(manifest_path), "sha256": _sha256_file(manifest_path),
                            "access": "read-only (check_frozen / check_derived_scenario)"},
        "scenario": scn_info, "env_hashes": env_hashes, "config_hashes": cfg_hashes,
        "experiment": str(exp), "root_seed": root,
        "reward": {"w_L": rw.w_L, "w_M": rw.w_M, "w_R": rw.w_R},
        "eval_seeds": disjoint, "eval_scenarios": list(MEMORYLESS_SCENARIOS),
        "canonical": canon_info, "runs": runs,
        "policies": {c: (tuned_impl if c == GLOBAL_LABEL else c) for c in pols},
        "tune_selected": {"path": str(tune_selected), "global": selected.get("global")},
        "matched_freqs": freq_info, "agent_cross_check": xcheck,
        "stats": {"bootstrap_resamples": n_resamples, "ci_level": 0.95, "bootstrap_seed": 0,
                  "wilcoxon": "two-sided, zero_method=wilcox", "pairing": "TEST episode",
                  "rl_mean": "per-seed mean over the canonical runs"},
        "n_eval_jobs": len(jobs), "workers": workers,
        "timings_s": {"matched_freqs": round(freq_wall, 1), "eval": round(eval_wall, 1),
                      "total": round(wall, 1)},
        "outputs": sorted(str(Path(f).relative_to(out)) for f in files),
        "per_seed_sha256": _sha256_file(suite_dir / "per_seed.csv"),
    }
    mpath = out / MANIFEST_NAME
    top: dict[str, Any] = {}
    if mpath.is_file():
        top = json.loads(mpath.read_text(encoding="utf-8"))
    top.setdefault("kind", "churnmaster-ablations")
    top["exploratory"] = True
    top["updated_utc"] = _now()
    top["memoryless_analysis_plan"] = MEMORYLESS_ANALYSIS_PLAN
    top.setdefault("extra_scenarios", {})[MEMORYLESS_SCENARIO] = {
        k: scn_info[k] for k in ("parent", "env_hash", "config_hash", "toml", "toml_sha256",
                                 "differs", "in_frozen_manifest")}
    top.setdefault("suites", {})["memoryless"] = suite_manifest
    write_manifest(mpath, _clean(top))
    files.append(mpath)
    say(f"ablate[memoryless]: wrote {len(files)} files to {out} ({wall:.1f}s)")
    return AblationReport("memoryless", out, df, deltas, pd.DataFrame(), summary, {},
                          suite_manifest, files, wall, 0.0, eval_wall)
