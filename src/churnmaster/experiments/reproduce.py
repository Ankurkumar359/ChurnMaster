"""Reproduction driver (Requirements 26.2, 26.3, 26.4).

Part A: the output guard, read-only regeneration of the report figures and
tables from the canonical results (:func:`figures_only`), and the reproduce
manifest writer (:func:`write_reproduce_manifest`). The ``--quick`` and
``--full`` pipelines build on these.

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
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

__all__ = [
    "CANONICAL_NAMES", "CANONICAL_DIRS", "RESULTS_ROOT", "REFERENCE_POLICIES",
    "SCATTER_POLICIES", "MANIFEST_PACKAGES", "guard_out_dir", "figures_only",
    "write_reproduce_manifest", "timed_stage", "sha256_file", "markdown_table",
]

RESULTS_ROOT = Path("results")
CANONICAL_NAMES = ("headroom", "diagnostics", "diagnostics_60", "train", "tune", "matched",
                   "eval", "checkpoint_5_17")
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
