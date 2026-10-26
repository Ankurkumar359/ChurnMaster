"""Figures for traces and experiment results (Requirement 26.3).

Figures are built with ``matplotlib.figure.Figure`` and rendered through the
non-interactive Agg canvas directly. This never imports ``pyplot`` and never
changes the global matplotlib backend, so it is safe to use alongside code
that has its own backend configured.

Each plotting function takes data and an output path, writes a PNG, releases
the figure, and returns the written path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from churnmaster.sim.churn import ChurnTrace

__all__ = [
    "PALETTE",
    "online_count_series",
    "session_lengths_s",
    "plot_online_count",
    "plot_session_histogram",
    "plot_policy_bars",
    "plot_headroom_bars",
    "plot_success_vs_pings",
    "dead_fraction_series",
    "plot_dead_fraction",
    "plot_state_visits",
    "plot_validation_curves",
    "plot_training_curves",
    "plot_comparison_bars",
    "greedy_share_matrix",
    "plot_greedy_heatmap",
    "plot_tuning_scatter",
]

# Okabe–Ito colorblind-friendly palette.
PALETTE = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00",
    "#CC79A7", "#56B4E9", "#F0E442", "#000000",
)
_LINESTYLES = ("-", "--", "-.", ":")
_DPI = 120

PathLike = Union[str, Path]
TraceArg = Union[ChurnTrace, Sequence[tuple[str, ChurnTrace]]]


def online_count_series(trace: ChurnTrace, step_ms: int = 30_000,
                        focal: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Number of online nodes sampled every ``step_ms`` on ``[0, horizon]``.

    Returns ``(times_ms, counts)``. The focal node, if given, is excluded.
    Computed with a single sweep over ``trace.events()``; the sample at time
    ``t`` reflects every event with ``t_ms <= t`` (sessions are half-open).
    """
    if step_ms <= 0:
        raise ValueError("step_ms must be > 0")
    times = np.arange(0, trace.horizon_ms + 1, step_ms, dtype=np.int64)
    online0 = trace.online_at(0)
    count = len(online0) - (1 if focal is not None and focal in online0 else 0)
    events = [e for e in trace.events() if e.node != focal]
    counts = np.empty(len(times), dtype=np.int64)
    j = 0
    for i, t in enumerate(times):
        while j < len(events) and events[j].t_ms <= t:
            count += 1 if events[j].kind == "up" else -1
            j += 1
        counts[i] = count
    return times, counts


def session_lengths_s(trace: ChurnTrace, focal: int | None = None) -> np.ndarray:
    """Clipped session lengths in seconds, excluding the focal node's session."""
    return np.array(
        [(s.end_ms - s.start_ms) / 1000.0 for s in trace.all_sessions() if s.node != focal],
        dtype=float,
    )


def _save(fig: Figure, path: PathLike) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    FigureCanvasAgg(fig)
    fig.savefig(out, format="png", dpi=_DPI)
    fig.clear()
    return out


def _as_labeled(traces: TraceArg) -> list[tuple[str, ChurnTrace]]:
    if isinstance(traces, ChurnTrace):
        return [("online nodes", traces)]
    items = list(traces)
    if not items:
        raise ValueError("at least one trace is required")
    return items


def plot_online_count(traces: TraceArg, path: PathLike, step_ms: int = 30_000,
                      focal: int | None = None, title: str | None = None) -> Path:
    """Online node count over time for one trace or a list of ``(label, trace)``.

    ``focal`` (a node index) is excluded from every trace's count.
    """
    items = _as_labeled(traces)
    fig = Figure(figsize=(7.0, 4.0), layout="constrained")
    ax = fig.add_subplot()
    for k, (label, tr) in enumerate(items):
        t_ms, counts = online_count_series(tr, step_ms=step_ms, focal=focal)
        ax.step(t_ms / 60_000.0, counts, where="post", label=label,
                color=PALETTE[k % len(PALETTE)],
                linestyle=_LINESTYLES[k % len(_LINESTYLES)], linewidth=1.6)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Online nodes (count)")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)
    ax.set_title(title or "Online nodes over time")
    return _save(fig, path)


def plot_policy_bars(df, metric: str, path: PathLike, title: str | None = None,
                     ylabel: str | None = None) -> Path:
    """Bar chart of ``metric`` per policy: mean over rows, error bar = std (ddof=1).

    ``df`` is a per-seed results table (``experiments.runner.results_table``)
    with ``policy``, ``scenario`` and ``metric`` columns. Several scenarios give
    grouped bars (one color per scenario). Policy / scenario order is the order
    of first appearance.
    """
    if df is None or len(df) == 0:
        raise ValueError("df must contain at least one row")
    if metric not in df.columns:
        raise ValueError(f"unknown metric column {metric!r}")
    policies = list(dict.fromkeys(df["policy"]))
    scenarios = list(dict.fromkeys(df["scenario"]))
    width = 0.8 / len(scenarios)
    x = np.arange(len(policies), dtype=float)
    fig = Figure(figsize=(max(6.0, 1.0 * len(policies) + 2.0), 4.2), layout="constrained")
    ax = fig.add_subplot()
    for j, scn in enumerate(scenarios):
        means, stds = [], []
        for pol in policies:
            vals = df.loc[(df["policy"] == pol) & (df["scenario"] == scn), metric].astype(float)
            means.append(float(vals.mean()) if len(vals) else np.nan)
            stds.append(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)
        ax.bar(x + (j - (len(scenarios) - 1) / 2) * width, means, width, yerr=stds,
               capsize=3, label=scn, color=PALETTE[j % len(PALETTE)], edgecolor="white")
    ax.set_xticks(x, policies, rotation=30, ha="right")
    ax.set_ylabel(ylabel or metric)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", frameon=True, title="scenario")
    ax.set_title(title or f"{metric} by policy (mean ± std over seeds)")
    return _save(fig, path)


def plot_session_histogram(trace: ChurnTrace, path: PathLike, focal: int | None = None,
                           log: bool = True, bins: int = 40,
                           title: str | None = None) -> Path:
    """Histogram of (clipped) session lengths in seconds.

    With ``log=True`` the x-axis uses log-spaced bins on a log scale, which
    suits the heavy-tailed Weibull sessions. The focal session is excluded.
    """
    lengths = session_lengths_s(trace, focal=focal)
    fig = Figure(figsize=(7.0, 4.0), layout="constrained")
    ax = fig.add_subplot()
    if lengths.size:
        if log:
            lo, hi = max(lengths.min(), 1e-3), lengths.max()
            edges = np.geomspace(lo, hi if hi > lo else lo * 10.0, bins + 1)
            ax.set_xscale("log")
        else:
            edges = bins
        ax.hist(lengths, bins=edges, color=PALETTE[0], edgecolor="white",
                linewidth=0.5, label=f"sessions (n={lengths.size})")
        ax.legend(loc="best", frameon=True)
    else:
        ax.text(0.5, 0.5, "no sessions", transform=ax.transAxes,
                ha="center", va="center")
    ax.set_xlabel("Session length (s)" + (", log scale" if log else ""))
    ax.set_ylabel("Sessions (count)")
    ax.grid(True, alpha=0.3)
    ax.set_title(title or "Session length distribution")
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# Headroom figures (Task 6.2)
# --------------------------------------------------------------------------- #

def _grouped_bars(ax, df, metric: str, policies: list[str], scenarios: list[str]) -> None:
    width = 0.8 / len(scenarios)
    x = np.arange(len(policies), dtype=float)
    for j, scn in enumerate(scenarios):
        means, stds = [], []
        for pol in policies:
            vals = df.loc[(df["policy"] == pol) & (df["scenario"] == scn), metric].astype(float)
            means.append(float(vals.mean()) if len(vals) else np.nan)
            stds.append(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)
        ax.bar(x + (j - (len(scenarios) - 1) / 2) * width, means, width, yerr=stds,
               capsize=3, label=scn, color=PALETTE[j % len(PALETTE)], edgecolor="white")
    ax.set_xticks(x, policies, rotation=30, ha="right")
    ax.grid(True, axis="y", alpha=0.3)


def plot_headroom_bars(df, path: PathLike, metrics: Sequence[str] = ("J", "p95_latency_all_ms"),
                       ylabels: Sequence[str] | None = None, title: str | None = None) -> Path:
    """One panel per metric (default J and p95 latency): grouped bars per policy and scenario.

    Bars are means over seeds, error bars std (ddof=1). Order = first appearance.
    """
    if df is None or len(df) == 0:
        raise ValueError("df must contain at least one row")
    for m in metrics:
        if m not in df.columns:
            raise ValueError(f"unknown metric column {m!r}")
    policies = list(dict.fromkeys(df["policy"]))
    scenarios = list(dict.fromkeys(df["scenario"]))
    labels = list(ylabels) if ylabels is not None else [
        {"J": "J (score)", "p95_latency_all_ms": "p95 latency, all lookups (ms)"}.get(m, m)
        for m in metrics]
    fig = Figure(figsize=(max(6.0, 1.0 * len(policies) + 2.0) * len(metrics) * 0.75, 4.4),
                 layout="constrained")
    axes = fig.subplots(1, len(metrics), squeeze=False)[0]
    for ax, m, lab in zip(axes, metrics, labels):
        _grouped_bars(ax, df, m, policies, scenarios)
        ax.set_ylabel(lab)
        ax.set_title(m)
    axes[0].legend(loc="best", frameon=True, title="scenario")
    fig.suptitle(title or "Headroom: mean ± std over calibration seeds")
    return _save(fig, path)


_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")


def plot_success_vs_pings(df, path: PathLike, title: str | None = None) -> Path:
    """Success rate vs pings per hour: one point per seed (small) and the mean (large).

    Color = policy, marker = scenario.
    """
    if df is None or len(df) == 0:
        raise ValueError("df must contain at least one row")
    policies = list(dict.fromkeys(df["policy"]))
    scenarios = list(dict.fromkeys(df["scenario"]))
    fig = Figure(figsize=(7.5, 4.8), layout="constrained")
    ax = fig.add_subplot()
    for i, pol in enumerate(policies):
        color = PALETTE[i % len(PALETTE)]
        for j, scn in enumerate(scenarios):
            g = df[(df["policy"] == pol) & (df["scenario"] == scn)]
            if g.empty:
                continue
            x = g["pings_per_hour"].astype(float).to_numpy()
            y = g["success_rate"].astype(float).to_numpy()
            mk = _MARKERS[j % len(_MARKERS)]
            ax.scatter(x, y, s=12, alpha=0.35, color=color, marker=mk)
            ax.scatter([x.mean()], [y.mean()], s=80, color=color, marker=mk,
                       edgecolor="black", linewidth=0.8, label=f"{pol} / {scn}")
    ax.set_xlabel("Maintenance pings per hour")
    ax.set_ylabel("Lookup success rate")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True, fontsize="small", ncol=2)
    ax.set_title(title or "Success vs maintenance cost (per seed; large = mean)")
    return _save(fig, path)


def dead_fraction_series(samples: Sequence[Sequence[tuple[int, float]]]) -> tuple[np.ndarray, np.ndarray]:
    """Mean dead-entry fraction per sample time over several episodes.

    ``samples`` holds one ``EpisodeMetrics.dead_entry_fraction`` sequence of
    ``(t_ms, fraction)`` per episode; values sharing a ``t_ms`` are averaged.
    Returns ``(times_ms, mean_fraction)`` sorted by time.
    """
    acc: dict[int, list[float]] = {}
    for ep in samples:
        for t, f in ep:
            acc.setdefault(int(t), []).append(float(f))
    times = np.array(sorted(acc), dtype=np.int64)
    means = np.array([float(np.mean(acc[t])) for t in times], dtype=float)
    return times, means


def plot_dead_fraction(series: Sequence[tuple[str, Sequence[Sequence[tuple[int, float]]]]],
                       path: PathLike, title: str | None = None) -> Path:
    """Ground-truth dead-entry fraction of F's table over time, one line per label.

    ``series`` is a list of ``(label, [episode dead_entry_fraction, ...])``; each
    line is the mean over that label's episodes (see :func:`dead_fraction_series`).
    """
    items = list(series)
    if not items:
        raise ValueError("at least one series is required")
    fig = Figure(figsize=(7.5, 4.2), layout="constrained")
    ax = fig.add_subplot()
    for k, (label, eps) in enumerate(items):
        t, y = dead_fraction_series(eps)
        if t.size == 0:
            continue
        ax.plot(t / 60_000.0, y, label=label, color=PALETTE[k % len(PALETTE)],
                linestyle=_LINESTYLES[k % len(_LINESTYLES)], linewidth=1.5)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Dead routing-table entries (fraction)")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True, fontsize="small")
    ax.set_title(title or "Dead-entry fraction over time (ground truth, mean over seeds)")
    return _save(fig, path)


def plot_state_visits(visits: Sequence[tuple[str, Sequence[int]]], path: PathLike,
                      min_visits: int | None = None, bins: int = 30,
                      title: str | None = None) -> Path:
    """Histogram of per-state visit counts (visited states only), one series per label.

    ``visits`` is a list of ``(label, [visit count of each visited state, ...])``.
    Bins are log-spaced on a log x-axis; ``min_visits`` draws the coverage
    threshold as a dashed vertical line.
    """
    items = [(lab, np.asarray(v, dtype=float)) for lab, v in visits]
    if not items:
        raise ValueError("at least one series is required")
    allv = np.concatenate([v for _, v in items]) if items else np.array([])
    allv = allv[allv > 0]
    fig = Figure(figsize=(7.5, 4.2), layout="constrained")
    ax = fig.add_subplot()
    if allv.size:
        hi = max(float(allv.max()), 2.0)
        edges = np.geomspace(1.0, hi * 1.0001, bins + 1)
        for k, (label, v) in enumerate(items):
            v = v[v > 0]
            ax.hist(v, bins=edges, histtype="step", linewidth=1.6,
                    color=PALETTE[k % len(PALETTE)],
                    linestyle=_LINESTYLES[k % len(_LINESTYLES)],
                    label=f"{label} ({v.size} states)")
        ax.set_xscale("log")
    else:
        ax.text(0.5, 0.5, "no visits", transform=ax.transAxes, ha="center", va="center")
    if min_visits is not None:
        ax.axvline(min_visits, color="black", linestyle="--", linewidth=1.0,
                   label=f"threshold = {min_visits} visits")
    ax.set_xlabel("Visits per state (decisions), log scale")
    ax.set_ylabel("States (count)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True, fontsize="small")
    ax.set_title(title or "State visit counts")
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# Report figures (Task 10.1): learning curves, comparisons, greedy policy, tuning
# --------------------------------------------------------------------------- #

_METRIC_LABELS = {
    "J": "J (score, higher is better)",
    "success_rate": "Lookup success rate",
    "p95_latency_all_ms": "p95 latency, all lookups (ms)",
    "pings_per_hour": "Maintenance pings per hour",
    "evictions_per_hour": "Evictions per hour",
    "mean_abs_dq": "Mean |ΔQ| per update",
}


# Reference-line colors: high-contrast on white (no yellow), distinct from run lines 0-2.
_REF_COLORS = ("#D55E00", "#CC79A7", "#56B4E9", "#000000", "#7F7F7F")


def _metric_label(metric: str) -> str:
    return _METRIC_LABELS.get(metric, metric)


def _ordered(values) -> list:
    return list(dict.fromkeys(values))


def plot_validation_curves(runs: Sequence[tuple[str, "object"]], path: PathLike,
                           references=None, metric: str = "J",
                           title: str | None = None) -> Path:
    """Validation ``metric`` per checkpoint, one panel per scenario, one line per run.

    ``runs`` is a list of ``(label, validation_df)`` where each frame has
    ``checkpoint, scenario, episode, <metric>`` columns (``validation.csv``).
    Lines are means over validation seeds with a shaded ± 1 standard error band.
    ``references`` (``references.csv``: ``policy, scenario, <metric>``) adds one
    dotted / dashed horizontal line per reference policy (mean over seeds).
    """
    items = list(runs)
    if not items:
        raise ValueError("at least one run is required")
    scenarios = _ordered(s for _, df in items for s in df["scenario"])
    fig = Figure(figsize=(5.2 * len(scenarios) + 2.6, 4.4), layout="constrained")
    axes = fig.subplots(1, len(scenarios), squeeze=False)[0]
    ref_pols = _ordered(references["policy"]) if references is not None and len(references) else []
    ref_styles = (":", "--", "-.", (0, (1, 3)), (0, (5, 2, 1, 2)))
    for ax, scn in zip(axes, scenarios):
        for k, (label, df) in enumerate(items):
            g = df[df["scenario"] == scn].groupby("checkpoint")[metric]
            if g.ngroups == 0:
                continue
            mean = g.mean()
            sem = g.std(ddof=1).fillna(0.0) / np.sqrt(g.count().clip(lower=1))
            x = mean.index.to_numpy(dtype=float)
            color = PALETTE[k % len(PALETTE)]
            ax.plot(x, mean.to_numpy(), color=color, linestyle=_LINESTYLES[k % len(_LINESTYLES)],
                    marker=_MARKERS[k % len(_MARKERS)], markersize=4, linewidth=1.6, label=label)
            ax.fill_between(x, (mean - sem).to_numpy(), (mean + sem).to_numpy(), color=color,
                            alpha=0.15, linewidth=0)
        for j, pol in enumerate(ref_pols):
            vals = references.loc[(references["policy"] == pol) & (references["scenario"] == scn),
                                  metric].astype(float)
            if vals.empty:
                continue
            ax.axhline(float(vals.mean()), color=_REF_COLORS[j % len(_REF_COLORS)],
                       linestyle=ref_styles[j % len(ref_styles)], linewidth=1.3,
                       label=f"{pol} (reference)")
        ax.set_xlabel("Training episode (checkpoint)")
        ax.set_ylabel(_metric_label(metric))
        ax.set_title(scn)
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right upper", frameon=True, fontsize="small")
    fig.suptitle(title or f"Validation {metric} per checkpoint (mean ± 1 s.e. over validation seeds)")
    return _save(fig, path)


def plot_training_curves(runs: Sequence[tuple[str, "object"]], path: PathLike,
                         window: int = 10, title: str | None = None) -> Path:
    """Training J (per scenario) and mean |ΔQ| per episode, rolling mean over ``window``.

    ``runs`` is a list of ``(label, curves_df)`` (``curves.csv``: ``episode,
    scenario, J, mean_abs_dq``). Color = run, line style = scenario.
    """
    items = list(runs)
    if not items:
        raise ValueError("at least one run is required")
    if window < 1:
        raise ValueError("window must be >= 1")
    scenarios = _ordered(s for _, df in items for s in df["scenario"])
    fig = Figure(figsize=(12.0, 4.4), layout="constrained")
    ax_j, ax_q = fig.subplots(1, 2)
    for k, (label, df) in enumerate(items):
        color = PALETTE[k % len(PALETTE)]
        for j, scn in enumerate(scenarios):
            g = df[df["scenario"] == scn].sort_values("episode")
            if g.empty:
                continue
            y = g["J"].astype(float).rolling(window, min_periods=1).mean()
            ax_j.plot(g["episode"], y, color=color, linestyle=_LINESTYLES[j % len(_LINESTYLES)],
                      linewidth=1.4, label=f"{label} / {scn}")
        g = df.sort_values("episode")
        y = g["mean_abs_dq"].astype(float).rolling(window, min_periods=1).mean()
        ax_q.plot(g["episode"], y, color=color, linestyle=_LINESTYLES[k % len(_LINESTYLES)],
                  linewidth=1.5, label=label)
    ax_j.set_xlabel("Training episode")
    ax_j.set_ylabel(f"Training J, rolling mean over {window} episodes")
    ax_j.set_title("Training-episode J (ε-greedy behaviour)")
    ax_j.grid(True, alpha=0.3)
    ax_j.legend(loc="best", frameon=True, fontsize="small")
    ax_q.set_xlabel("Training episode")
    ax_q.set_ylabel(f"Mean |ΔQ| per update, rolling mean over {window} (log scale)")
    ax_q.set_yscale("log")
    ax_q.set_title("Q-value change per episode")
    ax_q.grid(True, alpha=0.3, which="both")
    ax_q.legend(loc="best", frameon=True, fontsize="small")
    fig.suptitle(title or "Training curves")
    return _save(fig, path)


def plot_comparison_bars(df, path: PathLike,
                         metrics: Sequence[str] = ("J", "p95_latency_all_ms", "pings_per_hour"),
                         policies: Sequence[str] | None = None, error: str = "ci95",
                         title: str | None = None) -> Path:
    """Grouped bars (policy × scenario), one stacked panel per metric.

    ``error``: ``"ci95"`` (default) = normal-approximation 95% CI of the mean
    (1.96 · s / √n), ``"std"`` = standard deviation (ddof=1). Policy order is
    ``policies`` or first appearance.
    """
    if df is None or len(df) == 0:
        raise ValueError("df must contain at least one row")
    if error not in ("ci95", "std"):
        raise ValueError("error must be 'ci95' or 'std'")
    for m in metrics:
        if m not in df.columns:
            raise ValueError(f"unknown metric column {m!r}")
    pols = list(policies) if policies is not None else _ordered(df["policy"])
    scenarios = _ordered(df["scenario"])
    width = 0.8 / len(scenarios)
    x = np.arange(len(pols), dtype=float)
    fig = Figure(figsize=(max(8.0, 0.75 * len(pols) + 3.0), 3.3 * len(metrics) + 0.8),
                 layout="constrained")
    axes = fig.subplots(len(metrics), 1, squeeze=False)[:, 0]
    hatches = ("", "//", "..", "xx", "\\\\")
    for ax, m in zip(axes, metrics):
        for j, scn in enumerate(scenarios):
            means, errs = [], []
            for pol in pols:
                v = df.loc[(df["policy"] == pol) & (df["scenario"] == scn), m].astype(float)
                means.append(float(v.mean()) if len(v) else np.nan)
                sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
                errs.append(1.96 * sd / np.sqrt(len(v)) if error == "ci95" and len(v) else sd)
            ax.bar(x + (j - (len(scenarios) - 1) / 2) * width, means, width, yerr=errs, capsize=2,
                   label=scn, color=PALETTE[j % len(PALETTE)], edgecolor="white",
                   hatch=hatches[j % len(hatches)])
        ax.set_ylabel(_metric_label(m))
        ax.grid(True, axis="y", alpha=0.3)
        if m == "J":
            vals = df[m].astype(float)
            lo, hi = float(vals.min()), float(vals.max())
            pad = max(0.1 * (hi - lo), 1e-3)
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_xticks(x, pols if ax is axes[-1] else [""] * len(pols),
                      rotation=35 if ax is axes[-1] else 0, ha="right")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(scenarios), frameon=True,
               title="scenario", fontsize="small")
    err_txt = "95% CI of the mean" if error == "ci95" else "std"
    fig.suptitle(title or f"Policy comparison (mean ± {err_txt} over seeds)")
    return _save(fig, path)


_TIER_NAMES = ("TOP bucket", "SECOND bucket", "OTHER buckets")


def greedy_share_matrix(table, action: str, tier: int) -> tuple[np.ndarray, np.ndarray]:
    """Visit-weighted share of decisions whose greedy action is ``action``.

    ``table`` is a (possibly pooled over runs) ``greedy_table.csv`` frame. Only
    reachable states with ``visits > 0`` in ``tier`` count. Returns
    ``(share, visits)`` arrays of shape ``(3 inc_fail, 2 inc_stale)``; cells
    without visits are NaN in ``share``.
    """
    share = np.full((3, 2), np.nan)
    visits = np.zeros((3, 2), dtype=np.int64)
    t = table[(table["tier"] == tier) & (table["visits"] > 0)]
    if "reachable" in t.columns:
        t = t[t["reachable"].astype(str).str.lower() == "true"]
    for f in range(3):
        for s in range(2):
            c = t[(t["inc_fail"] == f) & (t["inc_stale"] == s)]
            n = int(c["visits"].sum())
            visits[f, s] = n
            if n > 0:
                share[f, s] = float(c.loc[c["greedy"] == action, "visits"].sum()) / n
    return share, visits


def plot_greedy_heatmap(tables: Sequence["object"], path: PathLike,
                        actions: Sequence[str] = ("PROBE", "REPLACE"),
                        title: str | None = None) -> Path:
    """Greedy policy heatmaps: rows = action, columns = tier; cells = inc_fail × inc_stale.

    Each cell is the visit-weighted share of decisions (pooled over the given
    ``greedy_table.csv`` frames, i.e. over runs) whose greedy action equals the
    row's action; the remaining share is KEEP. Cells are annotated with the
    share and the visit count; unvisited cells are marked "n/a".
    """
    import pandas as pd

    frames = list(tables)
    if not frames:
        raise ValueError("at least one greedy table is required")
    pooled = pd.concat(frames, ignore_index=True)
    fig = Figure(figsize=(3.2 * 3 + 1.4, 2.9 * len(actions) + 0.9), layout="constrained")
    axes = fig.subplots(len(actions), 3, squeeze=False)
    im = None
    for i, act in enumerate(actions):
        for tier in range(3):
            ax = axes[i, tier]
            share, visits = greedy_share_matrix(pooled, act, tier)
            im = ax.imshow(np.ma.masked_invalid(share), cmap="cividis", vmin=0.0, vmax=1.0,
                           aspect="auto")
            for f in range(3):
                for s in range(2):
                    v = share[f, s]
                    txt = "n/a" if np.isnan(v) else f"{v:.2f}\n(n={visits[f, s]})"
                    dark = np.isnan(v) or v < 0.6
                    ax.text(s, f, txt, ha="center", va="center", fontsize="small",
                            color="white" if dark and not np.isnan(v) else "black")
            ax.set_xticks([0, 1], ["fresh (<60 s)", "stale (≥60 s)"])
            ax.set_yticks([0, 1, 2], ["0", "1", "≥2"])
            ax.set_xlabel("Incumbent last response")
            ax.set_ylabel("Incumbent consecutive failures")
            ax.set_title(f"{act}: {_TIER_NAMES[tier]}", fontsize="medium")
            ax.set_facecolor("#DDDDDD")
    fig.colorbar(im, ax=axes, shrink=0.8,
                 label="Share of visited decisions with this greedy action")
    fig.suptitle(title or f"Greedy policy of the final Q-tables (pooled over {len(frames)} run(s), "
                          "visit-weighted)")
    return _save(fig, path)


def plot_tuning_scatter(summary, path: PathLike, scenarios: Sequence[str] | None = None,
                        title: str | None = None) -> Path:
    """Mean J vs mean pings/h of every heuristic config, one panel per scenario + overall.

    ``summary`` is the tuning ``summary.csv`` (``J_mean_<s>``,
    ``pings_per_hour_mean_<s>``, ``J_overall``, ``pings_per_hour_overall``,
    ``is_default``, ``global_best``, ``privileged_best_<s>``).
    """
    if summary is None or len(summary) == 0:
        raise ValueError("summary must contain at least one row")
    scns = list(scenarios) if scenarios is not None else [
        c[len("J_mean_"):] for c in summary.columns if c.startswith("J_mean_")]
    panels = [(s, f"pings_per_hour_mean_{s}", f"J_mean_{s}", f"privileged_best_{s}") for s in scns]
    panels.append(("overall (mean over scenarios)", "pings_per_hour_overall", "J_overall", None))
    fig = Figure(figsize=(4.6 * len(panels) + 0.6, 4.4), layout="constrained")
    axes = fig.subplots(1, len(panels), squeeze=False)[0]

    def flag(col: str | None):
        if col is None or col not in summary.columns:
            return np.zeros(len(summary), dtype=bool)
        return summary[col].astype(str).str.lower().eq("true").to_numpy()

    for ax, (name, xc, yc, priv) in zip(axes, panels):
        x, y = summary[xc].astype(float).to_numpy(), summary[yc].astype(float).to_numpy()
        ax.scatter(x, y, s=22, color=PALETTE[0], alpha=0.6, marker="o",
                   label=f"configs (n={len(summary)})")
        # Privileged best is a large hollow diamond so a coinciding global best stays visible.
        m = flag(priv)
        if m.any():
            ax.scatter(x[m], y[m], s=220, facecolors="none", edgecolors=PALETTE[3], marker="D",
                       linewidth=2.0, label="per-scenario best (privileged)", zorder=4)
        for col, color, mk, lab in (("is_default", PALETTE[1], "s", "default config"),
                                    ("global_best", PALETTE[2], "^", "global best (tuned)")):
            m = flag(col)
            if m.any():
                ax.scatter(x[m], y[m], s=90, color=color, marker=mk, edgecolor="black",
                           linewidth=0.8, label=lab, zorder=3)
        ax.set_xlabel("Maintenance pings per hour")
        ax.set_ylabel("Mean J on tuning seeds")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    handles, labels = axes[-1].get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    for ax in axes[:-1]:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            seen.setdefault(ll, hh)
    fig.legend(list(seen.values()), list(seen.keys()), loc="outside lower center",
               ncol=len(seen), frameon=True, fontsize="small")
    fig.suptitle(title or "Heuristic tuning: J vs maintenance cost per config (train seeds)")
    return _save(fig, path)


__all__ += ["plot_ablation_forest"]


def plot_ablation_forest(comparison, metrics: Sequence[str], path: PathLike,
                         title: str | None = None) -> Path:
    """Forest plot of paired differences vs a reference, one panel per metric.

    ``comparison`` needs columns ``variant, scenario, metric, mean_diff,
    ci_lo, ci_hi``. Rows: variants (input order); one marker + CI bar per
    scenario (offset vertically); a dashed line at 0.
    """
    variants = list(dict.fromkeys(comparison["variant"]))
    scenarios = list(dict.fromkeys(comparison["scenario"]))
    metrics = list(metrics)
    if not variants or not metrics:
        raise ValueError("plot_ablation_forest needs at least one variant and one metric")
    height = max(3.0, 0.45 * len(variants) * max(1, len(scenarios)) / 2 + 1.5)
    fig = Figure(figsize=(5.2 * len(metrics) + 1.5, height))
    axes = fig.subplots(1, len(metrics), sharey=True, squeeze=False)[0]
    n_s = max(1, len(scenarios))
    offs = np.linspace(-0.25, 0.25, n_s) if n_s > 1 else np.zeros(1)
    y_base = np.arange(len(variants))[::-1].astype(float)
    for ax, m in zip(axes, metrics):
        sub = comparison[comparison["metric"] == m]
        for j, s in enumerate(scenarios):
            ss = sub[sub["scenario"] == s].set_index("variant")
            ys, xs, lo, hi = [], [], [], []
            for i, v in enumerate(variants):
                if v not in ss.index:
                    continue
                r = ss.loc[v]
                ys.append(y_base[i] + offs[j])
                xs.append(float(r["mean_diff"]))
                lo.append(float(r["mean_diff"]) - float(r["ci_lo"]))
                hi.append(float(r["ci_hi"]) - float(r["mean_diff"]))
            ax.errorbar(xs, ys, xerr=[lo, hi], fmt="o", ms=4, capsize=2, lw=1,
                        color=PALETTE[j % len(PALETTE)], label=s)
        ax.axvline(0.0, color="0.4", ls="--", lw=0.8)
        ax.set_xlabel(f"Δ {m}")
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(y_base)
    axes[0].set_yticklabels(variants)
    axes[-1].legend(loc="best", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    return _save(fig, path)
