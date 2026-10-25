"""Training protocol (Requirements 16.6, 24; design "Q update", "Training (Task 8)").

Training run
------------
One run trains one :class:`~churnmaster.rl.qlearning.QAgent` for
``episodes`` episodes, **sequentially** (the agent carries over between
episodes). Episode ``e`` uses scenario ``scenarios[e % len(scenarios)]``
(default ``("stable", "high_churn")``: even -> stable, odd -> high_churn; the
debug mode trains on high_churn only) and the environment seed
``env_seed(root, TRAIN, run, e)``. The agent acts through
:class:`TrainingPolicy` (ε-greedy, ``ε = epsilon(e, settings)``), and every
τ=1 transition the Ledger emits -- warm-up and measured periods alike
(Requirement 16.6) -- is applied online via
``run_job(on_transition=agent.update_transition)``.

Exploration randomness: ONE generator per run, ``seeding.agent_rng(root,
run)`` (AGENT split, disjoint from every env stream), consumed across all of
the run's episodes in order. Runs never share a generator, so a run's
result does not depend on which other runs are trained or in which process.

Checkpoints and validation
--------------------------
Checkpoint ``c`` is the agent after ``c`` training episodes, for ``c`` in
``{0, validate_every, 2·validate_every, ..., episodes}`` (``episodes`` is
always included). A frozen copy of the agent is kept per checkpoint; after
training, every checkpoint is validated greedily (ε=0, :class:`Learned`) on
``validation_seeds`` VALIDATION episodes (``run=0``, episodes ``0..n-1``, so
identical across runs and settings) per validation scenario. Validation does
not influence training, so all validation episodes (every checkpoint of
every run plus the reference policies) run as one batch through
:func:`run_jobs` -- in parallel with ``workers > 1`` and identical to the
sequential result. Training runs (grid settings / final runs) are mutually
independent and may also run in a process pool (:func:`run_trainings`);
each stays sequential internally, so results equal sequential execution.

Episode-0 checkpoint (I4 reference) -- documented decision
----------------------------------------------------------
The plan defines the I4 reference as "the untrained agent (the episode-0
checkpoint, which acts uniformly at random)". Greedily, an untrained Q-table
(all zeros) picks KEEP everywhere (tie order KEEP < PROBE < REPLACE), i.e.
it equals ``never_replace``. We therefore report both:

* checkpoint 0 = the untrained *greedy* policy (== never_replace J, tested);
* ``uniform_random`` on the same validation seeds = the untrained agent as
  it actually behaves at ε=1 in episode 0.

:func:`i4_check` compares the final checkpoint with both (paired bootstrap
CI, 10k resamples, paired by (scenario, seed)); the PRIMARY verdict uses
``uniform_random`` (the plan's definition). The second I4 condition is that
the mean |ΔQ| per episode shrinks: mean over the last 20 training episodes
< mean over the first 20 (window ``min(20, episodes // 2)``); the
per-checkpoint-interval trend is reported too.

Modes (Requirement 24.2-24.4)
-----------------------------
* ``debug`` (8a): ``experiment.debug`` -- high_churn only, 1 run, 100 episodes.
* ``grid`` (8b): γ × α_min from ``experiment.grid``, run 0 each, full
  protocol; :func:`select_grid` picks the highest mean validation J over the
  last ``experiment.last_checkpoints`` (3) checkpoints, averaged over both
  scenarios and all seeds. Ties (|Δ| <= 1e-12) prefer the config default
  (``agent.gamma``, ``agent.alpha_min`` = 0.9, 0.05), then grid order
  (γ outer, α_min inner, as listed in the config).
* ``final`` (8c): ``agent.runs`` (3) runs with the selected setting
  (``selected.json`` of the grid output, unless γ / α_min are given).

Outputs (``out_dir``; per run ``run<r>/``, in grid mode ``<tag>/run0/``)::

    references.csv           validation rows of the reference policies
    train_manifest.json      settings, env hashes, selection, wall time
    grid.csv, selected.json  grid mode only
    run<r>/q_ep<c>.npy       Q snapshot per checkpoint (+ QAgent sidecars)
    run<r>/q_final.npy       final Q-table (used for evaluation)
    run<r>/curves.csv        one row per training episode
    run<r>/validation.csv    one row per (checkpoint, scenario, seed)
    run<r>/greedy_table.csv  decoded state features -> greedy action, Q, N
    run<r>/i4.json           I4 check (Requirement 24.6)
    run<r>/summary.json      settings, greedy-policy statistics, I4 verdict
"""

from __future__ import annotations

import copy
import dataclasses
import itertools
import json
import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from churnmaster.actions import Action
from churnmaster.config import Config
from churnmaster.experiments.runner import (EpisodeResult, Job, check_frozen, env_config_hash,
                                            run_job, run_jobs)
from churnmaster.experiments.stats import paired_bootstrap_ci
from churnmaster.policies.learned import Learned
from churnmaster.rl.qlearning import QAgent, epsilon
from churnmaster.rl.state import (N_STATES, decode, discretize, encode, is_reachable)
from churnmaster.seeding import Split, agent_rng
from churnmaster.sim.observation import DecisionContext

__all__ = [
    "MODES", "DEFAULT_SCENARIOS", "REFERENCE_POLICIES", "PRIMARY_REFERENCE",
    "GREEDY_UNTRAINED_REFERENCE", "DQ_WINDOW", "TrainSettings", "TrainingPolicy", "TrainSpec",
    "TrainOutput", "train_agent", "run_trainings", "run_validation", "validation_row",
    "select_grid", "i4_from_frames", "i4_check", "greedy_table_frame", "greedy_stats",
    "RunReport", "TrainReport", "run_training",
]

MODES = ("debug", "grid", "final")
DEFAULT_SCENARIOS: tuple[str, ...] = ("stable", "high_churn")
REFERENCE_POLICIES: tuple[str, ...] = ("never_replace", "uniform_random", "kademlia_epoch",
                                       "heuristic", "oracle_heuristic")
PRIMARY_REFERENCE = "uniform_random"
GREEDY_UNTRAINED_REFERENCE = "never_replace"
DQ_WINDOW = 20
TIE_TOL = 1e-12
VALIDATION_METRICS = ("J", "success_rate", "p95_latency_all_ms", "pings_per_hour",
                      "evictions_per_hour", "n_decisions")


def _clean(x: Any) -> Any:
    """JSON-safe copy: NaN -> None, numpy scalars -> Python."""
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


# --------------------------------------------------------------------------- #
# Settings and the training policy
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TrainSettings:
    """Hyper-parameters of one training run (picklable; ``epsilon()`` reads it)."""

    gamma: float = 0.9
    alpha_min: float = 0.05
    alpha_exp: float = 0.7
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.6
    episodes: int = 200
    validate_every: int = 20
    scenarios: tuple[str, ...] = DEFAULT_SCENARIOS

    def __post_init__(self) -> None:
        if self.episodes < 1 or self.validate_every < 1:
            raise ValueError("episodes and validate_every must be >= 1")
        if not self.scenarios:
            raise ValueError("at least one training scenario is required")

    @classmethod
    def from_config(cls, cfg: Config, scenarios: Sequence[str], **overrides: Any) -> "TrainSettings":
        a = cfg.agent
        kw: dict[str, Any] = dict(gamma=a.gamma, alpha_min=a.alpha_min, alpha_exp=a.alpha_exp,
                                  eps_start=a.eps_start, eps_end=a.eps_end,
                                  eps_decay_frac=a.eps_decay_frac, episodes=a.episodes,
                                  validate_every=a.validate_every, scenarios=tuple(scenarios))
        kw.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**kw)

    @property
    def tag(self) -> str:
        return f"g{self.gamma:g}_a{self.alpha_min:g}"

    def scenario_for(self, episode: int) -> str:
        return self.scenarios[episode % len(self.scenarios)]

    def checkpoints(self) -> list[int]:
        return sorted(set(range(0, self.episodes + 1, self.validate_every)) | {self.episodes})

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["scenarios"] = list(self.scenarios)
        return d


class TrainingPolicy:
    """ε-greedy in-process policy over a *live* agent (training only, not picklable-safe)."""

    name = "training"
    shares_decision_points = True

    def __init__(self, agent: QAgent, eps: float, rng: np.random.Generator, id_bits: int):
        self.agent, self.eps, self.rng, self.id_bits = agent, float(eps), rng, int(id_bits)

    def decide(self, ctx: DecisionContext) -> Action:
        return self.agent.act(encode(discretize(ctx, self.id_bits)), self.eps, self.rng)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TrainSpec:
    """One training run: picklable (pool entry point argument)."""

    loader: Callable[[str], Config]
    settings: TrainSettings
    run: int = 0
    log_every: int = 0      # >0: print a progress line every N episodes


@dataclass
class TrainOutput:
    settings: TrainSettings
    run: int
    curves: pd.DataFrame
    snapshots: dict[int, QAgent]     # checkpoint -> frozen copy of the agent
    final: QAgent
    wall_s: float
    id_bits: int
    root_seed: int


def train_agent(spec: TrainSpec) -> TrainOutput:
    """Train one run sequentially (see module docstring). Module level: pool-safe."""
    st, run = spec.settings, int(spec.run)
    t0 = time.perf_counter()
    cfgs = {s: spec.loader(s) for s in dict.fromkeys(st.scenarios)}
    roots = {c.root_seed for c in cfgs.values()}
    bits = {c.sim.id_bits for c in cfgs.values()}
    if len(roots) != 1 or len(bits) != 1:
        raise ValueError("training scenarios must share root_seed and sim.id_bits")
    root, id_bits = roots.pop(), bits.pop()

    agent = QAgent(st.gamma, st.alpha_min, st.alpha_exp)
    rng = agent_rng(root, run)                       # one exploration stream per run
    ckpts = set(st.checkpoints())
    snapshots: dict[int, QAgent] = {}
    if 0 in ckpts:
        snapshots[0] = copy.deepcopy(agent)
    rows = []
    for e in range(st.episodes):
        scn = st.scenario_for(e)
        cfg = cfgs[scn]
        eps = epsilon(e, st)
        agent.reset_stats()
        pol = TrainingPolicy(agent, eps, rng, id_bits)
        res = run_job(Job(pol, scn, Split.TRAIN, run, e), cfg=cfg,
                      on_transition=agent.update_transition)
        m = res.metrics
        rows.append({
            "episode": e, "scenario": scn, "epsilon": eps, "J": m.J,
            "success_rate": m.success_rate, "p95_latency_all_ms": m.p95_latency_all_ms,
            "pings_per_hour": m.pings_per_hour, "evictions_per_hour": m.evictions_per_hour,
            "mean_abs_dq": agent.mean_abs_delta(), "n_updates": agent.n_updates,
            "n_transitions": res.n_transitions, "n_decisions": m.n_decisions,
            "states_visited": int((agent.N.sum(axis=1) > 0).sum()),
        })
        if e + 1 in ckpts:
            snapshots[e + 1] = copy.deepcopy(agent)
        if spec.log_every and (e + 1) % spec.log_every == 0:
            print(f"train {st.tag} run{run}: episode {e + 1}/{st.episodes} {scn} "
                  f"eps={eps:.3f} J={m.J:.4f} |dQ|={agent.mean_abs_delta():.4g} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    agent.reset_stats()
    return TrainOutput(settings=st, run=run, curves=pd.DataFrame(rows), snapshots=snapshots,
                       final=agent, wall_s=time.perf_counter() - t0, id_bits=id_bits,
                       root_seed=root)


def run_trainings(specs: Sequence[TrainSpec], workers: int | None = None) -> list[TrainOutput]:
    """Train independent runs, in spec order; a spawn pool when ``workers > 1``.

    Each run is sequential and seeded only by its own fields, so the pool
    gives the same outputs as sequential execution.
    """
    specs = list(specs)
    if workers is not None and workers > 1 and len(specs) > 1:
        with ProcessPoolExecutor(max_workers=min(int(workers), len(specs)),
                                 mp_context=multiprocessing.get_context("spawn")) as ex:
            return list(ex.map(train_agent, specs))
    return [train_agent(s) for s in specs]


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def validation_row(r: EpisodeResult) -> dict[str, Any]:
    """Scalar validation metrics plus on-policy action counts by true incumbent state."""
    m = r.metrics
    row: dict[str, Any] = {"policy": r.policy, "scenario": r.scenario, "episode": r.episode,
                           **{k: getattr(m, k) for k in VALIDATION_METRICS}}
    counts = m.actions_by_true_state_dict()
    for state in ("inc_online", "inc_offline"):
        for a in Action:
            row[f"act_{state}_{a.name}"] = int(counts.get((state, int(a)), 0))
    return row


def run_validation(policies: Mapping[str, str | Learned], loader: Callable[[str], Config],
                   scenarios: Sequence[str], n_seeds: int,
                   workers: int | None = None) -> pd.DataFrame:
    """Validate ``{label: policy}`` on VALIDATION episodes ``0..n_seeds-1`` (run 0) per scenario.

    Every label must be unique (``Learned`` objects are named by their label).
    Returns one row per (label, scenario, seed) with column ``policy = label``.
    """
    jobs: list[Job] = []
    for label, pol in policies.items():
        if not isinstance(pol, str) and pol.name != label:
            raise ValueError(f"policy object name {pol.name!r} != label {label!r}")
        jobs += [Job(pol, s, Split.VALIDATION, 0, ep) for s in scenarios for ep in range(n_seeds)]
    res = run_jobs(jobs, cfg_loader=loader, workers=workers)
    return pd.DataFrame([validation_row(r) for r in res])


def _snapshot_label(tag: str, run: int, ckpt: int) -> str:
    return f"learned:{tag}:run{run}:ep{ckpt}"


# --------------------------------------------------------------------------- #
# Grid selection (Requirement 24.3)
# --------------------------------------------------------------------------- #

def select_grid(table: pd.DataFrame, last_k: int,
                default: tuple[float, float] | None = None,
                order: Sequence[tuple[float, float]] | None = None
                ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Pick the (gamma, alpha_min) with the highest mean validation J over its last ``last_k``
    checkpoints (mean over all scenario × seed rows of those checkpoints).

    ``table`` needs columns ``gamma, alpha_min, checkpoint, J``. Ties within
    ``TIE_TOL`` prefer ``default``, then ``order`` (default: first appearance).
    Returns ``(summary, selected)``; summary has one row per setting in order.
    """
    if last_k < 1:
        raise ValueError("last_k must be >= 1")
    missing = [c for c in ("gamma", "alpha_min", "checkpoint", "J") if c not in table.columns]
    if missing:
        raise ValueError(f"grid table is missing columns {missing}")
    if order is None:
        order = list(dict.fromkeys(zip(table["gamma"].astype(float), table["alpha_min"].astype(float))))
    rows = []
    for g, a in order:
        sub = table[(table["gamma"] == g) & (table["alpha_min"] == a)]
        if sub.empty:
            raise ValueError(f"no validation rows for gamma={g}, alpha_min={a}")
        ckpts = sorted(sub["checkpoint"].unique())
        last = ckpts[-last_k:]
        sel = sub[sub["checkpoint"].isin(last)]
        row = {"gamma": float(g), "alpha_min": float(a), "score": float(sel["J"].mean()),
               "last_checkpoints": ",".join(str(int(c)) for c in last),
               "final_J": float(sub[sub["checkpoint"] == ckpts[-1]]["J"].mean())}
        if "scenario" in sub.columns:
            for scn, g2 in sel.groupby("scenario", sort=False):
                row[f"score_{scn}"] = float(g2["J"].mean())
        rows.append(row)
    summary = pd.DataFrame(rows)
    top = float(summary["score"].max())
    tied = [i for i, s in enumerate(summary["score"]) if s >= top - TIE_TOL]
    pick = tied[0]
    reason = "highest score" if len(tied) == 1 else "tie: grid order"
    if len(tied) > 1 and default is not None:
        for i in tied:
            if (summary.at[i, "gamma"], summary.at[i, "alpha_min"]) == (float(default[0]), float(default[1])):
                pick, reason = i, "tie: default setting"
                break
    summary["selected"] = [i == pick for i in range(len(summary))]
    r = summary.iloc[pick]
    selected = {"gamma": float(r["gamma"]), "alpha_min": float(r["alpha_min"]),
                "score": float(r["score"]), "last_k": int(last_k), "reason": reason,
                "n_tied": len(tied)}
    return summary, selected


# --------------------------------------------------------------------------- #
# I4 check (Requirement 24.6)
# --------------------------------------------------------------------------- #

def _paired_J(a: pd.DataFrame, b: pd.DataFrame, **ci_kw: Any) -> dict[str, Any]:
    sa = a.set_index(["scenario", "episode"])["J"].sort_index()
    sb = b.set_index(["scenario", "episode"])["J"]
    if set(sa.index) != set(sb.index) or sa.index.has_duplicates or sb.index.has_duplicates:
        raise ValueError("I4 comparison needs the same (scenario, seed) keys on both sides")
    ci = paired_bootstrap_ci(sa.to_numpy(float), sb.reindex(sa.index).to_numpy(float), **ci_kw)
    return {**ci.to_dict(), "beats": bool(ci.lo > 0.0)}


def i4_from_frames(validation: pd.DataFrame, references: pd.DataFrame | None,
                   curves: pd.DataFrame, *, n_resamples: int = 10_000, level: float = 0.95,
                   seed: int = 0, window: int = DQ_WINDOW) -> dict[str, Any]:
    """I4 verdict from a run's validation / curves frames and the reference rows.

    ``validation``: ``checkpoint, scenario, episode, J``; ``references``:
    ``policy, scenario, episode, J`` (may be None: only the checkpoint-0
    comparison is made and the primary verdict falls back to it);
    ``curves``: ``episode, mean_abs_dq``.
    """
    ci_kw = dict(n_resamples=n_resamples, level=level, seed=seed)
    ckpts = sorted(validation["checkpoint"].unique())
    first_c, final_c = int(ckpts[0]), int(ckpts[-1])
    final = validation[validation["checkpoint"] == final_c]
    comps: dict[str, Any] = {}
    comps["checkpoint0"] = _paired_J(final, validation[validation["checkpoint"] == first_c], **ci_kw)
    if references is not None:
        for pol in (PRIMARY_REFERENCE, GREEDY_UNTRAINED_REFERENCE):
            ref = references[references["policy"] == pol]
            if not ref.empty:
                keys = set(zip(final["scenario"], final["episode"]))
                ref = ref[[k in keys for k in zip(ref["scenario"], ref["episode"])]]
                comps[pol] = _paired_J(final, ref, **ci_kw)
    primary = PRIMARY_REFERENCE if PRIMARY_REFERENCE in comps else "checkpoint0"

    c = curves.sort_values("episode")
    dq = c["mean_abs_dq"].to_numpy(float)
    w = max(1, min(int(window), len(dq) // 2)) if len(dq) >= 2 else 1
    first = float(np.nanmean(dq[:w])) if len(dq) else math.nan
    last = float(np.nanmean(dq[-w:])) if len(dq) else math.nan
    shrinks = bool(len(dq) >= 2 and not math.isnan(first) and not math.isnan(last) and last < first)
    trend = []
    for lo, hi in zip(ckpts[:-1], ckpts[1:]):
        seg = c[(c["episode"] >= lo) & (c["episode"] < hi)]["mean_abs_dq"].to_numpy(float)
        trend.append({"from_episode": int(lo), "to_episode": int(hi),
                      "mean_abs_dq": float(np.nanmean(seg)) if seg.size and
                      not np.all(np.isnan(seg)) else math.nan})
    beats = bool(comps[primary]["beats"])
    return {
        "final_checkpoint": final_c, "first_checkpoint": first_c, "primary_reference": primary,
        "comparisons": comps, "beats_primary": beats,
        "dq": {"window": w, "first_mean": first, "last_mean": last, "shrinks": shrinks,
               "trend": trend},
        "passed": bool(beats and shrinks),
        "note": ("checkpoint0 is the untrained greedy policy (all KEEP == never_replace); "
                 "uniform_random is the untrained agent's actual episode-0 behaviour (primary)"),
    }


def i4_check(run_dir: Path | str, references: pd.DataFrame | None = None,
             write: bool = True, **kw: Any) -> dict[str, Any]:
    """I4 check of one run directory (``validation.csv``, ``curves.csv``); writes ``i4.json``.

    ``references`` defaults to ``references.csv`` in the run's mode directory
    (``run_dir.parent``, or ``run_dir.parent.parent`` in grid mode) if present.
    """
    d = Path(run_dir)
    val = pd.read_csv(d / "validation.csv")
    curves = pd.read_csv(d / "curves.csv")
    if references is None:
        for cand in (d.parent / "references.csv", d.parent.parent / "references.csv"):
            if cand.is_file():
                references = pd.read_csv(cand)
                break
    out = i4_from_frames(val, references, curves, **kw)
    if write:
        _write_json(d / "i4.json", out)
    return out


# --------------------------------------------------------------------------- #
# Greedy policy tables
# --------------------------------------------------------------------------- #

def greedy_table_frame(agent: QAgent) -> pd.DataFrame:
    """One row per state: decoded features, reachability, Q, N, greedy action."""
    greedy = agent.greedy_table()
    rows = []
    for s in range(N_STATES):
        f = decode(s)
        row: dict[str, Any] = {"state": s, **dataclasses.asdict(f), "reachable": is_reachable(s)}
        for a in Action:
            row[f"q_{a.name}"] = float(agent.Q[s, int(a)]) if agent.mask[s, int(a)] else math.nan
            row[f"n_{a.name}"] = int(agent.N[s, int(a)])
        row["visits"] = int(agent.N[s].sum())
        row["greedy"] = Action(int(greedy[s])).name if greedy[s] >= 0 else ""
        rows.append(row)
    return pd.DataFrame(rows)


def greedy_stats(agent: QAgent) -> dict[str, Any]:
    """Greedy-action frequencies and the inc_fail diagnostics.

    Three weightings over reachable states: ``all`` (unweighted), ``visited``
    (states with at least one update) and ``weighted`` (by update count
    ``N[s].sum()``, a proxy for visits). Groups: all states,
    ``inc_fail >= 2`` (fraction not KEEP) and ``inc_fail == 0 & inc_stale == 0``
    (fraction KEEP).
    """
    df = greedy_table_frame(agent)
    df = df[df["reachable"]]
    out: dict[str, Any] = {}

    def freqs(sub: pd.DataFrame) -> dict[str, Any]:
        res: dict[str, Any] = {}
        for wname in ("all", "visited", "weighted"):
            if wname == "all":
                w = np.ones(len(sub))
            elif wname == "visited":
                w = (sub["visits"] > 0).to_numpy(float)
            else:
                w = sub["visits"].to_numpy(float)
            tot = float(w.sum())
            res[wname] = {"n": int((w > 0).sum()), **{
                a.name: (float(w[(sub["greedy"] == a.name).to_numpy()].sum() / tot) if tot else math.nan)
                for a in Action}}
        return res

    out["actions"] = freqs(df)
    fail2 = freqs(df[df["inc_fail"] >= 2])
    healthy = freqs(df[(df["inc_fail"] == 0) & (df["inc_stale"] == 0)])
    out["inc_fail_ge2"] = {k: {"n": v["n"], "frac_not_keep": (1.0 - v["KEEP"]) if not
                               math.isnan(v["KEEP"]) else math.nan, **v} for k, v in fail2.items()}
    out["healthy"] = {k: {"n": v["n"], "frac_keep": v["KEEP"], **v} for k, v in healthy.items()}
    out["states_visited"] = int((df["visits"] > 0).sum())
    return out


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

@dataclass
class RunReport:
    settings: TrainSettings
    run: int
    out_dir: Path
    curves: pd.DataFrame
    validation: pd.DataFrame
    i4: dict[str, Any]
    greedy: dict[str, Any]
    wall_s: float


@dataclass
class TrainReport:
    mode: str
    out_dir: Path
    scenarios: tuple[str, ...]
    validation_seeds: int
    references: pd.DataFrame
    runs: list[RunReport]
    env_hashes: dict[str, str]
    wall_s: float
    grid: pd.DataFrame | None = None
    selected: dict[str, Any] | None = None
    selection_source: str | None = None
    frozen_checked: bool = True
    files: list[Path] = field(default_factory=list)


def _write_run(out: Path, o: TrainOutput, val: pd.DataFrame, refs: pd.DataFrame,
               **i4_kw: Any) -> RunReport:
    out.mkdir(parents=True, exist_ok=True)
    for c, ag in sorted(o.snapshots.items()):
        ag.save(out / f"q_ep{c}.npy")
    o.final.save(out / "q_final.npy")
    o.curves.to_csv(out / "curves.csv", index=False)
    val.to_csv(out / "validation.csv", index=False)
    greedy_table_frame(o.final).to_csv(out / "greedy_table.csv", index=False)
    i4 = i4_from_frames(val, refs, o.curves, **i4_kw)
    _write_json(out / "i4.json", i4)
    gs = greedy_stats(o.final)
    means = (val.groupby(["checkpoint", "scenario"])["J"].mean().reset_index()
             .to_dict(orient="records"))
    _write_json(out / "summary.json", {"settings": o.settings.to_dict(), "run": o.run,
                                       "wall_s": o.wall_s, "greedy": gs, "i4_passed": i4["passed"],
                                       "validation_J_means": means})
    return RunReport(o.settings, o.run, out, o.curves, val, i4, gs, o.wall_s)


def _load_selected(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_training(mode: str, configs_dir: Path | str, *,
                 experiment_path: Path | str | None = None,
                 manifest_path: Path | str = Path("results/headroom/manifest.json"),
                 out_dir: Path | str | None = None,
                 episodes: int | None = None, runs: int | None = None,
                 gamma: float | None = None, alpha_min: float | None = None,
                 validate_every: int | None = None, validation_seeds: int | None = None,
                 grid_dir: Path | str | None = None, allow_unfrozen: bool = False,
                 reference_policies: Sequence[str] = REFERENCE_POLICIES,
                 workers: int | None = None, log_every: int = 0,
                 n_resamples: int = 10_000,
                 progress: Callable[[str], None] | None = None) -> TrainReport:
    """Run one training mode end to end (see module docstring) and write its outputs.

    Refuses (``FrozenEnvError``) unless every scenario's env hash matches the
    frozen manifest; ``allow_unfrozen`` skips the check (tests / debugging).
    """
    from churnmaster.experiments.diagnostics import _frozen_loader

    if mode not in MODES:
        raise ValueError(f"unknown training mode {mode!r}; expected one of {MODES}")
    t0 = time.perf_counter()
    say = progress or (lambda m: None)
    configs_dir = Path(configs_dir)
    exp = Path(experiment_path) if experiment_path is not None else \
        configs_dir / "experiments" / "train.toml"
    manifest_path = Path(manifest_path)
    out = Path(out_dir) if out_dir is not None else Path("results/train") / mode
    loader = _frozen_loader(configs_dir, exp, manifest_path)

    probe = loader("high_churn" if mode == "debug" else DEFAULT_SCENARIOS[0])
    ex = probe.experiment
    if mode == "debug":
        dbg = ex.debug
        scenarios: tuple[str, ...] = (dbg.scenario if dbg else "high_churn",)
        n_runs = runs or (dbg.runs if dbg else 1)
        n_eps = episodes or (dbg.episodes if dbg else 100)
    else:
        scenarios = tuple(ex.scenarios) or DEFAULT_SCENARIOS
        n_runs = runs or (1 if mode == "grid" else probe.agent.runs)
        n_eps = episodes or probe.agent.episodes
    n_val = validation_seeds or ex.validation_seeds or 5
    last_k = ex.last_checkpoints or 3

    cfgs = {s: loader(s) for s in scenarios}
    if allow_unfrozen:
        env_hashes = {s: env_config_hash(c) for s, c in cfgs.items()}
    else:
        env_hashes = {s: check_frozen(manifest_path, c) for s, c in cfgs.items()}

    base = cfgs[scenarios[0]]
    common = dict(episodes=n_eps, validate_every=validate_every)
    selection_source = None
    if mode == "grid":
        grid = ex.grid
        gammas = tuple(grid.gamma) if grid else (0.8, 0.9)
        alphas = tuple(grid.alpha_min) if grid else (0.02, 0.05)
        if gamma is not None:
            gammas = (gamma,)
        if alpha_min is not None:
            alphas = (alpha_min,)
        settings = [TrainSettings.from_config(base, scenarios, gamma=g, alpha_min=a, **common)
                    for g, a in itertools.product(gammas, alphas)]
        run_ids = list(range(n_runs))
    else:
        g, a = gamma, alpha_min
        if mode == "final" and (g is None or a is None):
            sel_path = (Path(grid_dir) if grid_dir is not None else out.parent / "grid") / "selected.json"
            if sel_path.is_file():
                sel = _load_selected(sel_path)
                g = sel["gamma"] if g is None else g
                a = sel["alpha_min"] if a is None else a
                selection_source = str(sel_path)
            else:
                selection_source = "config defaults (no selected.json found)"
        else:
            selection_source = "cli" if (g is not None or a is not None) else "config"
        settings = [TrainSettings.from_config(base, scenarios, gamma=g, alpha_min=a, **common)]
        run_ids = list(range(n_runs))

    specs = [TrainSpec(loader, st, r, log_every) for st in settings for r in run_ids]
    say(f"train[{mode}]: {len(specs)} run(s) x {n_eps} episodes on {list(scenarios)}; "
        f"settings {[s.tag for s in settings]}")
    outputs = run_trainings(specs, workers=workers)
    t_train = time.perf_counter() - t0
    say(f"train[{mode}]: training done in {t_train:.1f}s; validating")

    id_bits = base.sim.id_bits
    pols: dict[str, str | Learned] = {p: p for p in reference_policies}
    for o in outputs:
        for c, ag in o.snapshots.items():
            lab = _snapshot_label(o.settings.tag, o.run, c)
            pols[lab] = Learned(ag, name=lab, id_bits=id_bits)
    val_all = run_validation(pols, loader, scenarios, n_val, workers=workers)
    refs = val_all[val_all["policy"].isin(list(reference_policies))].reset_index(drop=True)
    say(f"train[{mode}]: validation done ({len(val_all)} episodes) at "
        f"{time.perf_counter() - t0:.1f}s")

    out.mkdir(parents=True, exist_ok=True)
    refs.to_csv(out / "references.csv", index=False)
    reports: list[RunReport] = []
    grid_rows = []
    for o in outputs:
        parts = []
        for c in sorted(o.snapshots):
            v = val_all[val_all["policy"] == _snapshot_label(o.settings.tag, o.run, c)].copy()
            v.insert(0, "checkpoint", c)
            parts.append(v.drop(columns=["policy"]))
        val = pd.concat(parts, ignore_index=True)
        run_dir = (out / o.settings.tag if mode == "grid" else out) / f"run{o.run}"
        reports.append(_write_run(run_dir, o, val, refs, n_resamples=n_resamples))
        if mode == "grid":
            grid_rows.append(val.assign(gamma=o.settings.gamma, alpha_min=o.settings.alpha_min,
                                        run=o.run))

    grid_summary = selected = None
    if mode == "grid":
        table = pd.concat(grid_rows, ignore_index=True)
        order = [(s.gamma, s.alpha_min) for s in settings]
        grid_summary, selected = select_grid(table, last_k, (base.agent.gamma, base.agent.alpha_min),
                                             order)
        grid_summary.to_csv(out / "grid.csv", index=False)
        _write_json(out / "selected.json", selected)

    wall = time.perf_counter() - t0
    _write_json(out / "train_manifest.json", {
        "mode": mode, "scenarios": list(scenarios), "validation_seeds": n_val,
        "runs": run_ids, "settings": [s.to_dict() for s in settings],
        "env_hashes": env_hashes, "frozen_checked": not allow_unfrozen,
        "manifest": str(manifest_path), "experiment": str(exp),
        "selection_source": selection_source, "selected": selected,
        "reference_policies": list(reference_policies), "workers": workers,
        "wall_s": wall, "train_wall_s": t_train,
    })
    return TrainReport(mode=mode, out_dir=out, scenarios=scenarios, validation_seeds=n_val,
                       references=refs, runs=reports, env_hashes=env_hashes, wall_s=wall,
                       grid=grid_summary, selected=selected, selection_source=selection_source,
                       frozen_checked=not allow_unfrozen)
