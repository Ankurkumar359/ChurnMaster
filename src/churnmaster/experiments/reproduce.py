"""Reproduction driver (Requirements 26.2, 26.3, 26.4).

* :func:`figures_only` regenerates the report figures and tables read-only from
  an existing results tree (no simulation).
* :func:`run_pipeline` (``reproduce --quick`` / ``--full``) re-runs the whole
  study into its own output directory ``OUT`` (default
  ``results/reproduce_quick`` / ``results/reproduce_full``)::

      headroom    -> OUT/headroom     (gate study; freezes OUT/headroom/manifest.json)
      diagnostics -> OUT/diagnostics  (uses OUT's manifest and OUT/headroom/per_seed.csv)
      train       -> OUT/train/grid + OUT/train/final  (quick: final only)
      tune        -> OUT/tune
      matched     -> OUT/matched/freqs.json  (over OUT/train/final/run*/q_final.npy)
      evaluate    -> eval lock in OUT's manifest, then OUT/eval (statistics + H1-H5)
      figures     -> OUT/figures      (figures_only(OUT, OUT/figures))
      verify      -> full mode only: OUT vs the canonical results/ (per_seed.csv, selections)

  Every stage uses OUT's own frozen manifest; the canonical manifest (eval lock,
  recorded eval runs) is never read by a stage nor written. ``OUT`` must pass
  :func:`guard_out_dir`. Stage timings, settings and the verify result are in
  ``OUT/reproduce_manifest.json`` (rewritten after every stage). The pipeline
  stops at the first failing stage, records it and returns a non-zero exit.

Settings (:data:`QUICK_SETTINGS` / :data:`FULL_SETTINGS`): full mode uses the
same values as the canonical commands (headroom 10 calibration seeds,
diagnostics 20 episodes, grid then final training from ``train.toml``, tuning
10 seeds, matched 5 validation seeds, evaluation 20 eval seeds). Quick mode keeps
full-length episodes and the same headroom / diagnostics / matched seeds but
skips the grid and trains one final run of 40 episodes with the canonical grid
selection fixed (gamma=0.8, alpha_min=0.02), validating every 10 episodes on 2
seeds; tuning uses 3 seeds and evaluation 5 eval seeds. Quick results are a
smoke reproduction, not the reported numbers.

``experiment_overrides`` (tests only) is a TOML-shaped dict deep-merged into
every stage's experiment file; the merged files are written to
``OUT/_experiments/<stage>.toml`` and used instead (e.g.
``{"sim": {"warmup_s": 60, "measured_s": 240}}`` for short episodes). With no
overrides the stages read ``configs/experiments/*.toml`` directly.

The canonical result directories (frozen env manifest, eval lock, final
Q-tables, ...) are never written here: :func:`guard_out_dir` refuses any output
directory that is, is inside, or contains one of them.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import hashlib
import json
import math
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

__all__ = [
    "CANONICAL_NAMES", "CANONICAL_DIRS", "RESULTS_ROOT", "REFERENCE_POLICIES",
    "SCATTER_POLICIES", "MANIFEST_PACKAGES", "guard_out_dir", "figures_only",
    "write_reproduce_manifest", "timed_stage", "sha256_file", "markdown_table",
    "QUICK_SETTINGS", "FULL_SETTINGS", "STAGE_NAMES", "PipelineResult", "run_pipeline",
    "compare_per_seed", "compare_selections", "verify_against", "EXIT_STAGE_FAILED",
    "EXIT_VERIFY_MISMATCH",
]

RESULTS_ROOT = Path("results")
CANONICAL_NAMES = ("headroom", "diagnostics", "diagnostics_60", "train", "tune", "matched",
                   "eval", "checkpoint_5_17", "ablations")
CANONICAL_DIRS = tuple(RESULTS_ROOT / n for n in CANONICAL_NAMES)

# Reference lines on the validation learning curves (references.csv policies).
REFERENCE_POLICIES = ("never_replace", "uniform_random", "kademlia_epoch", "heuristic",
                      "oracle_heuristic")
# Success-vs-pings scatter: at most 8 policies so every color is distinct.
SCATTER_POLICIES = ("never_replace", "uniform_random", "matched_random", "kademlia_epoch",
                    "kademlia_std", "tuned_heuristic", "oracle_heuristic", "rl_mean")
COMPARISON_METRICS = ("J", "p95_latency_all_ms", "pings_per_hour")
MANIFEST_PACKAGES = ("numpy", "scipy", "pandas", "matplotlib", "pytest", "hypothesis",
                     "churnmaster")

_REPO_ROOT = Path(__file__).resolve().parents[3]


# --------------------------------------------------------------------------- #
# Output guard
# --------------------------------------------------------------------------- #

def _protected(extra_roots: Sequence[Path | str] = ()) -> list[Path]:
    """Resolved results roots and canonical dirs (cwd- and repo-relative, + extra roots)."""
    roots = {Path.cwd() / RESULTS_ROOT, _REPO_ROOT / RESULTS_ROOT}
    roots.update(Path(r) for r in extra_roots)
    out: list[Path] = []
    for r in roots:
        r = r.resolve()
        out.append(r)
        out.extend(r / n for n in CANONICAL_NAMES)
    return out


def _is_relative_to(p: Path, other: Path) -> bool:
    try:
        p.relative_to(other)
        return True
    except ValueError:
        return False


def guard_out_dir(out: Path | str, extra_roots: Sequence[Path | str] = ()) -> Path:
    """Return ``out`` resolved, or raise ``ValueError`` if it would touch canonical results.

    Rejected: the results root itself (``results/``), a canonical dir, anything
    inside one, and anything containing one (e.g. the repository root).
    ``extra_roots`` adds further results roots (e.g. ``--results-dir``) whose
    canonical sub-directories are protected the same way.
    """
    p = Path(out).resolve()
    results_roots = {(Path.cwd() / RESULTS_ROOT).resolve(), (_REPO_ROOT / RESULTS_ROOT).resolve()}
    results_roots.update(Path(r).resolve() for r in extra_roots)
    for c in _protected(extra_roots):
        if c in results_roots:
            if p == c:
                raise ValueError(f"output dir {out!s} is the results root {c}; "
                                 "choose a sub-directory such as results/figures")
            if _is_relative_to(c, p):
                raise ValueError(f"output dir {out!s} contains the results root {c}")
            continue
        if p == c or _is_relative_to(p, c):
            raise ValueError(f"output dir {out!s} is inside canonical results dir {c}")
        if _is_relative_to(c, p):
            raise ValueError(f"output dir {out!s} contains canonical results dir {c}")
    return p


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cell(v: Any, digits: int = 4) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e6):
            return f"{v:.3g}"
        return f"{v:.{digits}f}" if abs(v) < 10 else f"{v:.1f}"
    return str(v).replace("|", "\\|").replace("\n", " ")


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """GitHub-flavoured markdown table (cells escaped, floats compact)."""
    lines = ["| " + " | ".join(_cell(h) for h in headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(_cell(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def _pm(mean: float, std: float, digits: int = 4) -> str:
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "nan"
    d = digits if abs(mean) < 10 else 1
    if std is None or (isinstance(std, float) and math.isnan(std)):
        return f"{mean:.{d}f}"
    return f"{mean:.{d}f} ± {std:.{d}f}"


def _as_bool(series):
    return series.astype(str).str.lower().eq("true")


@contextlib.contextmanager
def timed_stage(stages: list[dict[str, Any]], name: str, **info: Any) -> Iterator[dict[str, Any]]:
    """Append ``{name, status, wall_s, **info}`` to ``stages`` around a block.

    ``status`` is ``"ok"`` or ``"failed"`` (the exception is re-raised).
    """
    rec: dict[str, Any] = {"name": name, **info}
    t0 = time.perf_counter()
    try:
        yield rec
        rec.setdefault("status", "ok")
    except BaseException:
        rec["status"] = "failed"
        raise
    finally:
        rec["wall_s"] = round(time.perf_counter() - t0, 3)
        stages.append(rec)


# --------------------------------------------------------------------------- #
# Reproduce manifest
# --------------------------------------------------------------------------- #

def _config_hashes(configs_dir: Path, experiment_path: Path | None) -> dict[str, Any]:
    from churnmaster.config import config_hash, load_config
    from churnmaster.experiments.runner import env_config_hash

    cfg_h: dict[str, str] = {}
    env_h: dict[str, str] = {}
    base = configs_dir / "base.toml"
    for scn in sorted((configs_dir / "scenarios").glob("*.toml")):
        cfg = load_config(base, scn, experiment_path)
        cfg_h[cfg.scenario.name] = config_hash(cfg)
        env_h[cfg.scenario.name] = env_config_hash(cfg)
    files = {}
    for f in sorted(configs_dir.rglob("*.toml")):
        files[f.relative_to(configs_dir).as_posix()] = sha256_file(f)
    return {"config_hashes": cfg_h, "env_hashes": env_h, "config_files": files}


def write_reproduce_manifest(path: Path | str, mode: str, args: Any = None,
                             stages: Sequence[Mapping[str, Any]] | None = None,
                             extra: Mapping[str, Any] | None = None, *,
                             configs_dir: Path | str | None = None,
                             experiment_path: Path | str | None = None) -> dict[str, Any]:
    """Write the reproduce manifest (Requirement 26.4) and return it.

    Records the git hash, per-scenario config and env hashes (all scenario
    TOMLs under ``configs_dir`` merged with ``experiment_path``), sha256 of
    every config file, package and Python versions, stage timings, the CLI
    arguments (``argparse.Namespace`` or mapping) and ``extra`` entries.
    """
    from churnmaster.cli import default_configs_dir
    from churnmaster.experiments.runner import git_hash, package_versions, write_manifest

    if args is None:
        arg_d: dict[str, Any] = {}
    elif isinstance(args, Mapping):
        arg_d = dict(args)
    else:
        arg_d = {k: v for k, v in vars(args).items() if not callable(v)}
    root = Path(configs_dir) if configs_dir else default_configs_dir()
    exp = Path(experiment_path) if experiment_path else None
    stage_list = [dict(s) for s in (stages or [])]
    data: dict[str, Any] = dict(extra or {})
    data.update({
        "mode": mode,
        "args": arg_d,
        "git_hash": git_hash(_REPO_ROOT),
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "packages": package_versions(MANIFEST_PACKAGES),
        "configs_dir": str(root),
        "experiment_path": None if exp is None else str(exp),
        **_config_hashes(root, exp),
        "stages": stage_list,
        "total_wall_s": round(sum(float(s.get("wall_s", 0.0)) for s in stage_list), 3),
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    })
    write_manifest(path, data)
    return data


# --------------------------------------------------------------------------- #
# Figures-only regeneration
# --------------------------------------------------------------------------- #

class _MissingInput(Exception):
    """A (non-fatal) input file of one figure / table is absent."""


class _Ctx:
    def __init__(self, results_dir: Path, out_dir: Path):
        self.results_dir = results_dir
        self.out_dir = out_dir
        self.inputs: dict[str, str] = {}
        self.outputs: list[dict[str, Any]] = []
        self.skipped: list[dict[str, str]] = []

    def path(self, rel: str) -> Path:
        p = self.results_dir / rel
        if not p.is_file():
            raise _MissingInput(f"missing input {rel}")
        self.inputs.setdefault(rel, sha256_file(p))
        return p

    def csv(self, rel: str):
        import pandas as pd

        return pd.read_csv(self.path(rel))

    def json(self, rel: str) -> Any:
        return json.loads(self.path(rel).read_text(encoding="utf-8"))

    def runs(self, name: str) -> list[tuple[str, Any]]:
        """``(runK, df)`` for every ``train/final/runK/<name>``; at least one required."""
        base = self.results_dir / "train" / "final"
        dirs = sorted((d for d in base.glob("run*") if d.is_dir() and d.name[3:].isdigit()),
                      key=lambda d: int(d.name[3:])) if base.is_dir() else []
        items = [(d.name, self.csv(f"train/final/{d.name}/{name}"))
                 for d in dirs if (d / name).is_file()]
        if not items:
            raise _MissingInput(f"missing input train/final/run*/{name}")
        return items


def _eval_per_seed_with_rl_mean(ctx: _Ctx):
    """Eval per-seed table plus derived ``rl_mean`` rows (mean of rl_run* per seed)."""
    import pandas as pd

    df = ctx.csv("eval/per_seed.csv")
    with contextlib.suppress(_MissingInput):
        summary = ctx.csv("eval/summary.csv")
        rl = df[df["policy"].astype(str).str.startswith("rl_run")]
        if "rl_mean" in set(summary["policy"]) and not rl.empty and "rl_mean" not in set(df["policy"]):
            metrics = [c for c in ("success_rate", "p95_latency_success_ms", "p95_latency_all_ms",
                                   "pings_per_hour", "mean_latency_ms", "evictions_per_hour",
                                   "messages_per_lookup", "J") if c in rl.columns]
            m = rl.groupby(["scenario", "episode"], sort=False)[metrics].mean().reset_index()
            m.insert(0, "policy", "rl_mean")
            df = pd.concat([df, m], ignore_index=True)
    return df


def _policy_order(ctx: _Ctx, df) -> list[str]:
    order = list(dict.fromkeys(df["policy"]))
    with contextlib.suppress(_MissingInput):
        s = list(dict.fromkeys(ctx.csv("eval/summary.csv")["policy"]))
        order = [p for p in s if p in set(order)] + [p for p in order if p not in s]
    return order


# --- figures ---------------------------------------------------------------- #

def _fig_validation(ctx: _Ctx, out: Path) -> Path:
    from churnmaster.plots import plot_validation_curves

    runs = ctx.runs("validation.csv")
    refs = None
    with contextlib.suppress(_MissingInput):
        refs = ctx.csv("train/final/references.csv")
        refs = refs[refs["policy"].isin(REFERENCE_POLICIES)]
    return plot_validation_curves(runs, out, references=refs)


def _fig_training(ctx: _Ctx, out: Path) -> Path:
    from churnmaster.plots import plot_training_curves

    return plot_training_curves(ctx.runs("curves.csv"), out)


def _fig_comparison(ctx: _Ctx, out: Path) -> Path:
    from churnmaster.plots import plot_comparison_bars

    df = _eval_per_seed_with_rl_mean(ctx)
    return plot_comparison_bars(df, out, metrics=COMPARISON_METRICS,
                                policies=_policy_order(ctx, df), error="std",
                                title="Locked evaluation: mean ± std over eval seeds")


def _fig_success_vs_pings(ctx: _Ctx, out_dir: Path) -> list[Path]:
    from churnmaster.plots import plot_success_vs_pings

    df = _eval_per_seed_with_rl_mean(ctx)
    paths = []
    for scn in dict.fromkeys(df["scenario"]):
        g = df[(df["scenario"] == scn) & df["policy"].isin(SCATTER_POLICIES)]
        if g.empty:
            continue
        g = g.assign(_o=g["policy"].map({p: i for i, p in enumerate(SCATTER_POLICIES)}))
        g = g.sort_values("_o", kind="stable").drop(columns="_o")
        paths.append(plot_success_vs_pings(
            g, out_dir / f"eval_success_vs_pings_{scn}.png",
            title=f"{scn}: success vs maintenance cost (eval seeds; large = mean)"))
    return paths


def _fig_greedy(ctx: _Ctx, out: Path) -> Path:
    from churnmaster.plots import plot_greedy_heatmap

    tables = [df for _, df in ctx.runs("greedy_table.csv")]
    return plot_greedy_heatmap(tables, out)


def _fig_headroom(ctx: _Ctx, out: Path) -> Path:
    from churnmaster.plots import plot_headroom_bars

    return plot_headroom_bars(ctx.csv("headroom/per_seed.csv"), out)


def _fig_state_visits(ctx: _Ctx, out: Path) -> Path:
    from churnmaster.plots import plot_state_visits

    cov = ctx.csv("diagnostics/coverage.csv")
    min_visits = None
    with contextlib.suppress(_MissingInput, KeyError, TypeError):
        d = ctx.json("diagnostics/diagnostics.json")
        vals = {int(v["min_visits"]) for v in d["coverage"].values()}
        min_visits = vals.pop() if len(vals) == 1 else None
    series = [(str(scope), g.loc[g["visits"] > 0, "visits"].astype(int).tolist())
              for scope, g in cov.groupby("scope", sort=False)]
    return plot_state_visits(series, out, min_visits=min_visits,
                             title="Diagnostics: visits per state (uniform_random)")


def _fig_tuning(ctx: _Ctx, out: Path) -> Path:
    from churnmaster.plots import plot_tuning_scatter

    return plot_tuning_scatter(ctx.csv("tune/summary.csv"), out)


# --- tables ----------------------------------------------------------------- #

def _tab_eval_summary(ctx: _Ctx, out: Path) -> Path:
    s = ctx.csv("eval/summary.csv")
    cols = [c for c in ("J", "success_rate", "p95_latency_all_ms", "pings_per_hour",
                        "evictions_per_hour") if f"{c}_mean" in s.columns]

    def tag(r) -> str:
        for col, t in (("privileged_fallback", "priv*"), ("privileged", "priv"),
                       ("reference", "ref")):
            if col in r and str(r[col]).lower() == "true":
                return t
        if "shares_decision_points" in r and str(r["shares_decision_points"]).lower() == "false":
            return "no-dp"
        return ""

    rows = [[r["scenario"], r["policy"], tag(r), int(r["n"]) if "n" in r else "",
             *(_pm(float(r[f"{c}_mean"]), float(r.get(f"{c}_std", float("nan")))) for c in cols)]
            for _, r in s.iterrows()]
    text = ("# Locked evaluation summary (mean ± std over eval seeds)\n\n"
            + markdown_table(["scenario", "policy", "tag", "n", *cols], rows))
    out.write_text(text, encoding="utf-8")
    return out


def _tab_primary(ctx: _Ctx, out: Path) -> Path:
    p = ctx.csv("eval/primary_tests.csv")
    cols = [c for c in ("scenario", "comparator", "metric", "n", "mean_diff", "ci_lo", "ci_hi",
                        "p", "p_holm", "significant") if c in p.columns]
    rows = [[r[c] for c in cols] for _, r in p.iterrows()]
    out.write_text("# Primary tests (rl_mean − comparator, paired; Holm-corrected)\n\n"
                   + markdown_table(cols, rows), encoding="utf-8")
    return out


def _tab_hypotheses(ctx: _Ctx, out: Path) -> Path:
    h = ctx.csv("eval/hypotheses.csv")
    cols = [c for c in ("id", "statement", "verdict", "evidence") if c in h.columns]
    rows = [[r[c] for c in cols] for _, r in h.iterrows()]
    out.write_text("# Hypotheses H1–H5\n\n" + markdown_table(cols, rows), encoding="utf-8")
    return out


def _ci_txt(ci: Mapping[str, Any] | None, digits: int = 4) -> str:
    if not ci:
        return ""
    return f"{ci['mean']:+.{digits}f} [{ci['lo']:+.{digits}f}, {ci['hi']:+.{digits}f}]"


def _tab_gates(ctx: _Ctx, out: Path) -> Path:
    data = ctx.json("headroom/gates.json")
    g = data.get("gates", data)
    g1, g2 = g.get("G1", {}), g.get("G2", {})
    lines = [f"# Headroom gates ({g.get('scenario', '?')}, split={data.get('split', '?')}, "
             f"n_seeds={data.get('n_seeds', '?')})\n",
             f"G1: {'PASS' if g1.get('passed') else 'FAIL'} (route={g1.get('route')}, "
             f"best={g1.get('best')})\n"]
    rows = [[pol, _ci_txt(c.get("dJ")), _ci_txt(c.get("p95_diff_ms"), 1),
             c.get("p95_rel"), c.get("j_route"), c.get("p95_route"), c.get("passes")]
            for pol, c in g1.get("candidates", {}).items()]
    lines.append(markdown_table(["candidate", "dJ vs never_replace [95% CI]",
                                 "p95 never_replace − candidate (ms)", "p95_rel", "J route",
                                 "p95 route", "passes"], rows))
    lines.append(f"\nG2: {'PASS' if g2.get('passed') else 'FAIL'} (route={g2.get('route')}, "
                 f"limited_headroom={g2.get('limited_headroom', data.get('limited_headroom'))})\n")
    rows2 = [[g.get("scenario", "?"), "gate", _ci_txt(g2.get("dJ")), g2.get("ping_ratio"),
              g2.get("passed")]]
    rows2 += [[scn, "info", _ci_txt(i.get("dJ")), i.get("ping_ratio"), i.get("passed")]
              for scn, i in g.get("G2_info", {}).items()]
    lines.append(markdown_table(["scenario", "kind", "dJ oracle − kademlia_epoch [95% CI]",
                                 "ping ratio", "passed"], rows2))
    if "thresholds" in g:
        lines.append("\nThresholds: " + ", ".join(f"{k}={v}" for k, v in g["thresholds"].items())
                     + "\n")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _tab_tuning(ctx: _Ctx, out: Path, top: int = 5) -> Path:
    s = ctx.csv("tune/summary.csv")
    scns = [c[len("J_mean_"):] for c in s.columns if c.startswith("J_mean_")]
    if "rank" in s.columns:
        s = s.sort_values("rank", kind="stable")
    import pandas as pd

    show = s.head(top)
    # Always list the default config (even outside the top N) for comparison.
    default = s[_as_bool(s["is_default"])] if "is_default" in s.columns else s.iloc[0:0]
    extra = default[~default.index.isin(show.index)]
    if not extra.empty:
        show = pd.concat([show, extra])
    rows = []
    for _, r in show.iterrows():
        flags = [n for n, c in (("default", "is_default"), ("global best", "global_best"),
                                *((f"privileged {x}", f"privileged_best_{x}") for x in scns))
                 if c in r and str(r[c]).lower() == "true"]
        rows.append([r.get("rank", ""), r["config"], r["J_overall"],
                     *(_pm(float(r[f"J_mean_{x}"]), float(r.get(f"J_std_{x}", float("nan"))))
                       for x in scns),
                     r.get("success_rate_overall", ""), r.get("p95_latency_all_ms_overall", ""),
                     r.get("pings_per_hour_overall", ""), ", ".join(flags)])
    headers = ["rank", "config", "J_overall", *(f"J {x}" for x in scns), "success",
               "p95_all_ms", "pings/h", "flags"]
    out.write_text(f"# Heuristic tuning: top {top} of {len(s)} configs (train seeds)\n\n"
                   + markdown_table(headers, rows), encoding="utf-8")
    return out


_FIGURES: tuple[tuple[str, Callable[[_Ctx, Path], Any]], ...] = (
    ("validation_curves.png", _fig_validation),
    ("training_curves.png", _fig_training),
    ("eval_comparison.png", _fig_comparison),
    ("eval_success_vs_pings_*.png", _fig_success_vs_pings),
    ("greedy_policy_heatmap.png", _fig_greedy),
    ("headroom_J_p95.png", _fig_headroom),
    ("diagnostics_state_visits.png", _fig_state_visits),
    ("tuning_scatter.png", _fig_tuning),
)
_TABLES: tuple[tuple[str, Callable[[_Ctx, Path], Path]], ...] = (
    ("eval_summary.md", _tab_eval_summary),
    ("primary_tests.md", _tab_primary),
    ("hypotheses.md", _tab_hypotheses),
    ("headroom_gates.md", _tab_gates),
    ("tuning_top5.md", _tab_tuning),
)
FIGURE_NAMES = tuple(n for n, _ in _FIGURES)
TABLE_NAMES = tuple(n for n, _ in _TABLES)
__all__ += ["FIGURE_NAMES", "TABLE_NAMES"]


def figures_only(results_dir: Path | str = "results", out_dir: Path | str = "results/figures",
                 progress: Callable[[str], None] | None = None) -> list[Path]:
    """Regenerate report figures (PNG) and markdown tables from existing results.

    Read-only on ``results_dir`` (no simulation). Writes figures to
    ``out_dir``, tables to ``out_dir/tables`` and ``out_dir/figures_manifest.json``
    (sha256 of every input read and every output written, plus skipped items).
    A figure / table whose input is missing is skipped with a warning recorded in
    the manifest. Returns the written paths (manifest last).
    """
    from churnmaster.experiments.runner import git_hash, package_versions, write_manifest

    res = Path(results_dir)
    out = guard_out_dir(out_dir, extra_roots=[res])
    if not res.is_dir():
        raise FileNotFoundError(f"results dir {res} does not exist")
    ctx = _Ctx(res, out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    say = progress or (lambda m: None)

    def run(kind: str, name: str, fn: Callable[[_Ctx, Path], Any], dest: Path) -> None:
        try:
            produced = fn(ctx, dest if "*" not in name else dest.parent)
        except _MissingInput as e:
            msg = f"skipped {kind} {name}: {e}"
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
            ctx.skipped.append({"kind": kind, "output": name, "reason": str(e)})
            say(msg)
            return
        for p in produced if isinstance(produced, list) else [produced]:
            ctx.outputs.append({"kind": kind, "path": Path(p).relative_to(out).as_posix(),
                                "sha256": sha256_file(p)})
            say(f"wrote {p}")
        if isinstance(produced, list) and not produced:
            ctx.skipped.append({"kind": kind, "output": name, "reason": "no data"})

    for name, fn in _FIGURES:
        run("figure", name, fn, out / name)
    for name, fn in _TABLES:
        run("table", name, fn, out / "tables" / name)

    manifest = {
        "mode": "figures-only",
        "results_dir": str(res),
        "out_dir": str(out),
        "inputs": dict(sorted(ctx.inputs.items())),
        "outputs": ctx.outputs,
        "skipped": ctx.skipped,
        "git_hash": git_hash(_REPO_ROOT),
        "packages": package_versions(MANIFEST_PACKAGES),
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    mpath = write_manifest(out / "figures_manifest.json", manifest)
    return [out / o["path"] for o in ctx.outputs] + [mpath]


# --------------------------------------------------------------------------- #
# Quick / full pipelines
# --------------------------------------------------------------------------- #

EXIT_VERIFY_MISMATCH = 3   # full mode: OUT differs from the canonical results
EXIT_STAGE_FAILED = 5      # a stage raised (recorded in the manifest)

STAGE_NAMES = ("headroom", "diagnostics", "train", "tune", "matched", "evaluate", "figures",
               "verify")
# Sub-directories (and files) a pipeline creates inside OUT; cleared before a re-run.
_OUT_ENTRIES = ("headroom", "diagnostics", "train", "tune", "matched", "eval", "figures",
                "_experiments", "reproduce_manifest.json")
DIAG_SCENARIOS = ("high_churn", "stable")  # same as `churnmaster diagnostics`

# ``None`` = the experiment file's / command's default (the canonical value).
FULL_SETTINGS: dict[str, Any] = {
    "headroom_seeds": 10, "diag_episodes": 20, "train_grid": True, "train_runs": None,
    "train_episodes": None, "train_gamma": None, "train_alpha_min": None,
    "train_validate_every": None, "train_validation_seeds": None, "tune_seeds": None,
    "matched_seeds": 5, "eval_seeds": None,
}
QUICK_SETTINGS: dict[str, Any] = {
    **FULL_SETTINGS, "train_grid": False, "train_runs": 1, "train_episodes": 40,
    # Grid skipped: the canonical grid selection (results/train/grid/selected.json) is fixed.
    "train_gamma": 0.8, "train_alpha_min": 0.02, "train_validate_every": 10,
    "train_validation_seeds": 2, "tune_seeds": 3, "eval_seeds": 5,
}
_SETTINGS = {"quick": QUICK_SETTINGS, "full": FULL_SETTINGS}


class StageError(RuntimeError):
    """A stage finished but its result makes the rest of the pipeline meaningless."""


@dataclass
class PipelineResult:
    mode: str
    out_dir: Path
    exit_code: int
    manifest: dict[str, Any]

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


@dataclass
class _Pipe:
    mode: str
    out: Path
    configs_dir: Path
    workers: int
    settings: dict[str, Any]
    experiments: dict[str, Path | None]
    say: Callable[[str], None]
    canonical_root: Path

    @property
    def manifest(self) -> Path:
        return self.out / "headroom" / "manifest.json"


# --- experiment-file overrides ---------------------------------------------- #

def _toml_key(k: str) -> str:
    import re

    return k if re.fullmatch(r"[A-Za-z0-9_-]+", k) else json.dumps(k)


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError(f"cannot write {type(v).__name__} to TOML")


def toml_dumps(data: Mapping[str, Any], _prefix: tuple[str, ...] = ()) -> str:
    """Minimal TOML writer for nested tables of scalars / scalar lists."""
    lines = [f"{_toml_key(k)} = {_toml_value(v)}" for k, v in data.items()
             if not isinstance(v, Mapping)]
    text = "\n".join(lines) + ("\n" if lines else "")
    for k, v in data.items():
        if isinstance(v, Mapping):
            path = (*_prefix, k)
            text += f"\n[{'.'.join(_toml_key(p) for p in path)}]\n" + toml_dumps(v, path)
    return text


def _experiment_paths(configs_dir: Path, out: Path,
                      overrides: Mapping[str, Any] | None) -> dict[str, Path | None]:
    """Per-stage experiment file (stage -> path or None = no experiment layer)."""
    from churnmaster.config import merge, read_toml

    files = {"headroom": "headroom", "diagnostics": None, "train": "train", "tune": "tune",
             "matched": None, "evaluate": "eval"}
    if not overrides:
        return {s: None if f is None else configs_dir / "experiments" / f"{f}.toml"
                for s, f in files.items()}
    d = out / "_experiments"
    d.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path | None] = {}
    for stage, f in files.items():
        base = read_toml(configs_dir / "experiments" / f"{f}.toml") if f else {}
        p = d / f"{stage}.toml"
        p.write_text(f"# generated by reproduce: {f or 'no'} experiment file + overrides\n"
                     + toml_dumps(merge(base, dict(overrides))), encoding="utf-8")
        paths[stage] = p
    return paths


# --- stages ------------------------------------------------------------------ #
# Each takes the pipeline context and returns extra info for its stage record.
# They are looked up by name at run time (tests monkeypatch them).

def _stage_headroom(p: _Pipe) -> dict[str, Any]:
    from churnmaster.experiments.headroom import run_headroom, write_outputs

    rep = run_headroom(p.configs_dir, p.experiments["headroom"],
                       n_seeds=p.settings["headroom_seeds"], manifest_path=p.manifest,
                       progress=p.say, workers=p.workers)
    write_outputs(rep, p.out / "headroom")
    g = rep.gates.to_dict()
    info = {"g1_passed": bool(rep.g1_passed), "g1_route": g["G1"]["route"],
            "g2_passed": bool(g["G2"]["passed"]), "frozen": bool(rep.manifest["frozen"]),
            "env_hashes": rep.manifest["env_hashes"]}
    if not rep.manifest["frozen"]:
        raise StageError("headroom gate G1 failed (env not frozen); later stages cannot run")
    return info


def _stage_diagnostics(p: _Pipe) -> dict[str, Any]:
    from churnmaster.experiments.diagnostics import run_diagnostics, write_outputs

    rep = run_diagnostics(p.configs_dir, p.manifest, experiment_path=p.experiments["diagnostics"],
                          n_episodes=p.settings["diag_episodes"], scenarios=DIAG_SCENARIOS,
                          per_seed_csv=p.out / "headroom" / "per_seed.csv", workers=p.workers)
    write_outputs(rep, p.out / "diagnostics")
    # Diagnostic verdicts are findings (reported), not pipeline failures.
    return {"identity": all(r.passed for r in rep.identity.values()),
            "reward_sanity": all(r.passed for r in rep.sanity.values()),
            "coverage": all(r.passed for r in rep.coverage.values()),
            "all_passed": bool(rep.all_passed)}


def _stage_train(p: _Pipe) -> dict[str, Any]:
    from churnmaster.experiments.train import run_training

    s = p.settings
    common = dict(experiment_path=p.experiments["train"], manifest_path=p.manifest,
                  episodes=s["train_episodes"], validate_every=s["train_validate_every"],
                  validation_seeds=s["train_validation_seeds"], workers=p.workers,
                  progress=p.say)
    info: dict[str, Any] = {}
    grid_dir = p.out / "train" / "grid"
    if s["train_grid"]:
        g = run_training("grid", p.configs_dir, out_dir=grid_dir, **common)
        info["grid_selected"] = g.selected
        info["grid_wall_s"] = round(g.wall_s, 3)
    f = run_training("final", p.configs_dir, out_dir=p.out / "train" / "final",
                     runs=s["train_runs"], gamma=s["train_gamma"], alpha_min=s["train_alpha_min"],
                     grid_dir=grid_dir, **common)
    st = f.runs[0].settings
    info.update({"final_runs": [r.run for r in f.runs], "gamma": st.gamma,
                 "alpha_min": st.alpha_min, "episodes": st.episodes,
                 "validation_seeds": f.validation_seeds, "final_wall_s": round(f.wall_s, 3)})
    return info


def _stage_tune(p: _Pipe) -> dict[str, Any]:
    from churnmaster.experiments.tune_heuristic import run_tuning, write_outputs

    rep = run_tuning(p.configs_dir, p.manifest, experiment_path=p.experiments["tune"],
                     n_seeds=p.settings["tune_seeds"], workers=p.workers, progress=p.say)
    write_outputs(rep, p.out / "tune")
    return {"n_seeds": rep.seeds["n_seeds"], "global": rep.selection.global_id,
            "per_scenario": dict(rep.selection.per_scenario)}


def _q_paths(p: _Pipe) -> list[Path]:
    base = p.out / "train" / "final"
    paths = sorted((d / "q_final.npy" for d in base.glob("run*")
                    if d.name[3:].isdigit() and (d / "q_final.npy").is_file()),
                   key=lambda q: int(q.parent.name[3:]))
    if not paths:
        raise StageError(f"no Q-tables under {base}")
    return paths


def _stage_matched(p: _Pipe) -> dict[str, Any]:
    from churnmaster.experiments.matched import FREQS_NAME, measure_matched_freqs

    q = _q_paths(p)
    measure_matched_freqs(q, p.configs_dir, p.manifest, n_seeds=p.settings["matched_seeds"],
                          workers=p.workers, experiment_path=p.experiments["matched"],
                          out_path=p.out / "matched" / FREQS_NAME)
    return {"q_paths": [str(x) for x in q], "n_seeds": p.settings["matched_seeds"]}


def _stage_evaluate(p: _Pipe) -> dict[str, Any]:
    from churnmaster.experiments.evaluate import create_lock_from_configs, evaluate
    from churnmaster.experiments.matched import FREQS_NAME

    exp, n = p.experiments["evaluate"], p.settings["eval_seeds"]
    lock = create_lock_from_configs(p.configs_dir, p.manifest, exp, n_seeds=n)
    rep = evaluate(p.configs_dir, p.manifest, p.out / "eval", q_paths=_q_paths(p),
                   tune_selected=p.out / "tune" / "selected.json",
                   matched_freqs=p.out / "matched" / FREQS_NAME, experiment_path=exp,
                   n_seeds=n, workers=p.workers, progress=p.say)
    return {"n_seeds": len(lock["episodes"]), "n_jobs": rep.n_jobs,
            "seed_list_hash": lock["seed_list_hash"],
            "hypotheses": {r["id"]: r["verdict"] for _, r in rep.hypotheses.iterrows()}}


def _stage_figures(p: _Pipe) -> dict[str, Any]:
    files = figures_only(p.out, p.out / "figures")
    man = json.loads(files[-1].read_text(encoding="utf-8"))
    return {"n_outputs": len(man["outputs"]), "skipped": man["skipped"]}


def _stage_verify(p: _Pipe) -> dict[str, Any]:
    v = verify_against(p.out, p.canonical_root)
    if v["status"] == "mismatch":
        raise _VerifyMismatch(v)
    return {"verify": v}


class _VerifyMismatch(Exception):
    def __init__(self, result: dict[str, Any]):
        super().__init__("reproduction differs from the canonical results")
        self.result = result


# --- verify ------------------------------------------------------------------ #

_KEY_COLS = ("policy", "scenario", "split", "run", "episode")


def _volatile(col: str) -> bool:
    c = col.lower()
    return any(t in c for t in ("wall", "path", "time", "utc"))


def compare_per_seed(a: Path | str, b: Path | str, atol: float = 0.0) -> dict[str, Any]:
    """Compare two per-seed tables (reproduced ``a`` vs canonical ``b``).

    Rows are matched on the key columns (policy, scenario, split, run, episode);
    numeric columns report the max absolute difference (NaN == NaN), other
    columns the number of differing cells. Wall-time / path columns are ignored.
    ``ok`` iff same keys, same columns and every difference <= ``atol`` / zero.
    """
    import numpy as np
    import pandas as pd

    a, b = Path(a), Path(b)
    da, db = pd.read_csv(a), pd.read_csv(b)
    res: dict[str, Any] = {"sha256_equal": sha256_file(a) == sha256_file(b),
                           "n_rows": [len(da), len(db)]}
    cols_a = [c for c in da.columns if not _volatile(c)]
    cols_b = [c for c in db.columns if not _volatile(c)]
    res["columns_equal"] = cols_a == cols_b
    res["columns_only_reproduced"] = [c for c in cols_a if c not in cols_b]
    res["columns_only_canonical"] = [c for c in cols_b if c not in cols_a]
    keys = [k for k in _KEY_COLS if k in da.columns and k in db.columns]
    ka = da[keys].astype(str).agg("|".join, axis=1)
    kb = db[keys].astype(str).agg("|".join, axis=1)
    res["duplicate_keys"] = bool(ka.duplicated().any() or kb.duplicated().any())
    res["keys_equal"] = set(ka) == set(kb) and not res["duplicate_keys"]
    res["rows_only_reproduced"] = sorted(set(ka) - set(kb))[:20]
    res["rows_only_canonical"] = sorted(set(kb) - set(ka))[:20]
    max_diff: dict[str, float] = {}
    text_diff: dict[str, int] = {}
    if not res["duplicate_keys"]:
        common = sorted(set(ka) & set(kb))
        xa = da.assign(_k=ka.values).set_index("_k").loc[common]
        xb = db.assign(_k=kb.values).set_index("_k").loc[common]
        for c in (c for c in cols_a if c in cols_b and c not in keys):
            sa, sb = xa[c], xb[c]
            if pd.api.types.is_numeric_dtype(sa) and pd.api.types.is_numeric_dtype(sb) \
                    and not pd.api.types.is_bool_dtype(sa):
                va, vb = sa.to_numpy(float), sb.to_numpy(float)
                both_nan = np.isnan(va) & np.isnan(vb)
                d = np.where(both_nan, 0.0, np.abs(va - vb))
                d = np.where(np.isnan(d), np.inf, d)
                max_diff[c] = float(d.max()) if len(d) else 0.0
            else:
                text_diff[c] = int((sa.astype(str).to_numpy() != sb.astype(str).to_numpy()).sum())
    res["max_abs_diff"] = max_diff
    res["n_text_diff"] = text_diff
    res["differing_columns"] = sorted([c for c, v in max_diff.items() if v > atol]
                                      + [c for c, v in text_diff.items() if v])
    res["ok"] = bool(res["columns_equal"] and res["keys_equal"] and not res["differing_columns"])
    return res


_SELECTION_KEYS = {"train/grid/selected.json": ("gamma", "alpha_min", "score"),
                   "tune/selected.json": ("global", "per_scenario", "default_config",
                                          "default_rank")}


def compare_selections(out_root: Path | str, canonical_root: Path | str) -> dict[str, Any]:
    """Compare the grid / tuning selections (non-volatile keys only); missing files skipped."""
    res: dict[str, Any] = {}
    for rel, keys in _SELECTION_KEYS.items():
        a, b = Path(out_root) / rel, Path(canonical_root) / rel
        if not (a.is_file() and b.is_file()):
            res[rel] = {"status": "skipped", "reason": f"missing {a if not a.is_file() else b}"}
            continue
        ja, jb = (json.loads(x.read_text(encoding="utf-8")) for x in (a, b))
        diff = {k: {"reproduced": ja.get(k), "canonical": jb.get(k)}
                for k in keys if ja.get(k) != jb.get(k)}
        res[rel] = {"status": "ok" if not diff else "mismatch", "differences": diff}
    return res


def verify_against(out_root: Path | str, canonical_root: Path | str = RESULTS_ROOT
                   ) -> dict[str, Any]:
    """``status``: ``ok`` / ``mismatch`` / ``skipped`` (no canonical eval/per_seed.csv)."""
    out_root, canonical_root = Path(out_root), Path(canonical_root)
    canon = canonical_root / "eval" / "per_seed.csv"
    if not canon.is_file():
        return {"status": "skipped", "reason": f"no canonical {canon}"}
    mine = out_root / "eval" / "per_seed.csv"
    if not mine.is_file():
        return {"status": "mismatch", "reason": f"missing {mine}"}
    ps = compare_per_seed(mine, canon)
    sel = compare_selections(out_root, canonical_root)
    ok = ps["ok"] and all(v["status"] != "mismatch" for v in sel.values())
    return {"status": "ok" if ok else "mismatch", "canonical_root": str(canonical_root),
            "per_seed": ps, "selections": sel}


# --- driver ------------------------------------------------------------------ #

def _prepare_out(out: Path) -> None:
    """Create OUT; a previous reproduce output is cleared, any other non-empty dir refused."""
    import shutil

    if out.exists() and not out.is_dir():
        raise ValueError(f"output path {out} is not a directory")
    if out.is_dir() and any(out.iterdir()):
        if not (out / "reproduce_manifest.json").is_file():
            raise ValueError(f"output dir {out} is not empty and holds no reproduce_manifest.json; "
                             "choose an empty or new directory")
        for name in _OUT_ENTRIES:
            q = out / name
            if q.is_dir():
                shutil.rmtree(q)
            elif q.exists():
                q.unlink()
    out.mkdir(parents=True, exist_ok=True)


def _canonical_root() -> Path:
    cwd = Path.cwd() / RESULTS_ROOT
    return cwd if (cwd / "eval").is_dir() else _REPO_ROOT / RESULTS_ROOT


def run_pipeline(mode: str, out_dir: Path | str, configs_dir: Path | str | None = None,
                 workers: int | None = None, experiment_overrides: Mapping[str, Any] | None = None,
                 progress: Callable[[str], None] | None = None, *,
                 settings_overrides: Mapping[str, Any] | None = None, args: Any = None,
                 canonical_root: Path | str | None = None,
                 stages: Sequence[str] | None = None) -> PipelineResult:
    """Run the ``quick`` / ``full`` reproduction into ``out_dir`` (see module docstring).

    ``settings_overrides`` replaces entries of :data:`QUICK_SETTINGS` /
    :data:`FULL_SETTINGS` (tests); ``stages`` restricts the stage list (default:
    all; ``verify`` runs in full mode only). Exit codes: 0 ok;
    :data:`EXIT_STAGE_FAILED` when a stage raised; :data:`EXIT_VERIFY_MISMATCH`
    when full-mode verification found differences. Raises ``ValueError`` before
    running anything for a bad mode, a guarded / non-empty output dir or
    unknown settings.
    """
    from churnmaster.cli import default_configs_dir
    from churnmaster.experiments.runner import default_workers

    if mode not in _SETTINGS:
        raise ValueError(f"mode must be 'quick' or 'full', got {mode!r}")
    settings = dict(_SETTINGS[mode])
    unknown = sorted(set(settings_overrides or {}) - set(settings))
    if unknown:
        raise ValueError(f"unknown settings {unknown}")
    settings.update(settings_overrides or {})
    out = guard_out_dir(out_dir)
    root = Path(configs_dir) if configs_dir else default_configs_dir()
    n_workers = default_workers() if workers is None else int(workers)
    if n_workers < 1:
        raise ValueError("workers must be >= 1")
    names = [s for s in (stages or STAGE_NAMES) if mode == "full" or s != "verify"]
    bad = [s for s in names if s not in STAGE_NAMES]
    if bad:
        raise ValueError(f"unknown stages {bad}")
    _prepare_out(out)
    say = progress or (lambda m: None)
    exps = _experiment_paths(root, out, experiment_overrides)
    pipe = _Pipe(mode, out, root, n_workers, settings, exps, say,
                 Path(canonical_root) if canonical_root else _canonical_root())

    recs: list[dict[str, Any]] = []
    extra: dict[str, Any] = {
        "out_dir": str(out), "workers": n_workers, "settings": settings,
        "experiment_overrides": dict(experiment_overrides or {}) or None,
        "experiment_files": {s: None if p is None else {"path": str(p), "sha256": sha256_file(p)}
                             for s, p in exps.items()},
        "stage_plan": names, "frozen_manifest": str(pipe.manifest),
    }
    exit_code = 0
    t0 = time.perf_counter()

    def write(status: str) -> dict[str, Any]:
        extra.update({"status": status, "exit_code": exit_code,
                      "pipeline_wall_s": round(time.perf_counter() - t0, 3)})
        return write_reproduce_manifest(out / "reproduce_manifest.json", mode, args, recs, extra,
                                        configs_dir=root, experiment_path=exps["evaluate"])

    write("running")
    for i, name in enumerate(names):
        fn = globals()[f"_stage_{name}"]
        say(f"reproduce[{mode}]: stage {i + 1}/{len(names)} {name}")
        try:
            with timed_stage(recs, name) as rec:
                rec.update(fn(pipe) or {})
        except _VerifyMismatch as e:
            recs[-1]["status"] = "mismatch"
            recs[-1]["verify"] = e.result
            extra["verify"] = e.result
            exit_code = EXIT_VERIFY_MISMATCH
            say(f"reproduce[{mode}]: verify found differences vs {pipe.canonical_root}")
            break
        except Exception as e:  # noqa: BLE001 - recorded, pipeline stops
            import traceback

            recs[-1]["error"] = f"{type(e).__name__}: {e}"
            recs[-1]["traceback"] = traceback.format_exc(limit=8)
            extra["failed_stage"] = name
            extra["not_run"] = names[i + 1:]
            exit_code = EXIT_STAGE_FAILED
            say(f"reproduce[{mode}]: stage {name} FAILED: {type(e).__name__}: {e}")
            break
        if name == "verify":
            extra["verify"] = recs[-1]["verify"]
        say(f"reproduce[{mode}]: stage {name} ok ({recs[-1]['wall_s']:.1f}s)")
        write("running")
    status = {0: "ok", EXIT_VERIFY_MISMATCH: "verify_mismatch"}.get(exit_code, "failed")
    manifest = write(status)
    return PipelineResult(mode, out, exit_code, manifest)
