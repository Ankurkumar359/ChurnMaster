"""Per-episode metrics (Requirements 20.1-20.3, 11.3).

``summarize`` reduces an episode's records to an ``EpisodeMetrics``. Only
records with ``measured=True`` are used; warm-up records are ignored
everywhere (Requirement 11.3).

Definitions (N = number of measured lookups, H = ``cfg.sim.measured_s / 3600``)
-----------------------------------------------------------------------------
* ``success_rate`` = successes / N.
* ``p95_latency_success_ms``: 95th percentile of latency over successful
  measured lookups. ``p95_latency_all_ms``: over all measured lookups, where a
  failed lookup counts at its give-up latency (``ObservedLookup.latency_ms``
  already holds the give-up time). Percentiles use ``numpy.percentile`` with
  ``method="linear"`` (``PERCENTILE_METHOD``).
* ``mean_latency_ms``: mean latency of successful measured lookups.
* ``pings_per_hour`` / ``evictions_per_hour``: totals over measured
  ``DecisionRecord``s (including null records such as ``kademlia_std`` ping
  costs) divided by H.
* ``messages_per_lookup`` = (sum of lookup queries + maintenance pings) / N.
* ``J = 1 - mean(loss) - (w_M * pings + w_R * evictions) / N`` with
  ``loss = rl.reward.lookup_loss(S, L, w_L, cfg.sim.deadline_ms)``.
* ``dead_entry_fraction``: ``(t_ms, fraction)`` from measured
  ``dead_fraction`` GT records, in input order.
* ``actions_by_true_state``: counts of measured ``decision`` GT records by
  ``("inc_online" | "inc_offline", action)``; records lacking ``action`` or
  ``inc_online`` are skipped.

Edge cases: with N = 0, every lookup-derived metric (success rate, latencies,
messages per lookup, J) is NaN; latency metrics over successful lookups are
NaN when there are none.

``EpisodeMetrics`` is frozen and hashable, so the two GT diagnostics are
stored as tuples (``actions_by_true_state`` as sorted ``(key, count)`` pairs;
``actions_by_true_state_dict()`` gives the dict view).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from churnmaster.config import Config
from churnmaster.rl.reward import action_cost, lookup_loss
from churnmaster.sim.observation import DecisionRecord, GTRecord, ObservedLookup

__all__ = ["PERCENTILE_METHOD", "EpisodeMetrics", "summarize", "summary_hash"]

PERCENTILE_METHOD = "linear"
NAN = float("nan")

SCALAR_FIELDS = (
    "success_rate", "p95_latency_success_ms", "p95_latency_all_ms", "pings_per_hour",
    "mean_latency_ms", "evictions_per_hour", "messages_per_lookup", "J",
    "n_lookups", "n_decisions",
)


@dataclass(frozen=True)
class EpisodeMetrics:
    success_rate: float
    p95_latency_success_ms: float
    p95_latency_all_ms: float
    pings_per_hour: float
    mean_latency_ms: float
    evictions_per_hour: float
    messages_per_lookup: float
    J: float
    dead_entry_fraction: tuple[tuple[int, float], ...]                 # GT diagnostic
    actions_by_true_state: tuple[tuple[tuple[str, int], int], ...]     # GT diagnostic
    n_lookups: int = 0
    n_decisions: int = 0

    def actions_by_true_state_dict(self) -> dict[tuple[str, int], int]:
        return dict(self.actions_by_true_state)

    def to_dict(self, diagnostics: bool = True) -> dict[str, Any]:
        """Plain dict. With ``diagnostics=False`` only scalar fields (CSV row)."""
        out: dict[str, Any] = {f: getattr(self, f) for f in SCALAR_FIELDS}
        if diagnostics:
            out["dead_entry_fraction"] = [list(p) for p in self.dead_entry_fraction]
            out["actions_by_true_state"] = {
                f"{s}:{a}": c for (s, a), c in self.actions_by_true_state}
        return out


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return NAN
    return float(np.percentile(np.asarray(xs, dtype=float), q, method=PERCENTILE_METHOD))


def summarize(lookups: Iterable[ObservedLookup], decisions: Iterable[DecisionRecord],
              gt_records: Iterable[GTRecord], cfg: Config) -> EpisodeMetrics:
    """Reduce one episode's records (measured window only) to metrics."""
    sim, rw = cfg.sim, cfg.reward
    lks = [lk for lk in lookups if lk.measured]
    decs = [d for d in decisions if d.measured]
    gts = [g for g in gt_records if g.measured]

    n = len(lks)
    succ_lat = [float(lk.latency_ms) for lk in lks if lk.success]
    all_lat = [float(lk.latency_ms) for lk in lks]
    pings = sum(int(d.pings) for d in decs)
    evictions = sum(int(d.evictions) for d in decs)
    queries = sum(int(lk.queries) for lk in lks)
    hours = sim.measured_s / 3600.0

    if n:
        success_rate = len(succ_lat) / n
        loss = sum(lookup_loss(lk.success, lk.latency_ms, rw.w_L, sim.deadline_ms) for lk in lks)
        J = 1.0 - loss / n - action_cost(pings, evictions, rw.w_M, rw.w_R) / n
        msgs = (queries + pings) / n
    else:
        success_rate = J = msgs = NAN

    dead = tuple((int(g.t_ms), float(g.get("fraction"))) for g in gts
                 if g.kind == "dead_fraction")
    counts: dict[tuple[str, int], int] = {}
    for g in gts:
        if g.kind != "decision":
            continue
        a, on = g.get("action"), g.get("inc_online")
        if a is None or on is None:
            continue
        key = ("inc_online" if on else "inc_offline", int(a))
        counts[key] = counts.get(key, 0) + 1

    return EpisodeMetrics(
        success_rate=success_rate,
        p95_latency_success_ms=_percentile(succ_lat, 95),
        p95_latency_all_ms=_percentile(all_lat, 95),
        pings_per_hour=pings / hours,
        mean_latency_ms=sum(succ_lat) / len(succ_lat) if succ_lat else NAN,
        evictions_per_hour=evictions / hours,
        messages_per_lookup=msgs,
        J=J,
        dead_entry_fraction=dead,
        actions_by_true_state=tuple(sorted(counts.items())),
        n_lookups=n,
        n_decisions=len(decs),
    )


def _canon(v: Any) -> Any:
    """JSON-safe canonical form: floats as ``repr`` strings (NaN -> "nan")."""
    if isinstance(v, bool) or v is None or isinstance(v, (int, str)):
        return v
    if isinstance(v, float):
        return "nan" if math.isnan(v) else repr(v)
    if isinstance(v, (tuple, list)):
        return [_canon(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _canon(x) for k, x in v.items()}
    return repr(v)


def summary_hash(m: EpisodeMetrics) -> str:
    """SHA-256 of the canonical JSON of all fields (exact float reprs)."""
    data = {f.name: _canon(getattr(m, f.name)) for f in dataclasses.fields(m)}
    blob = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
