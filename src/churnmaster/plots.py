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
