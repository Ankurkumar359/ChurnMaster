"""Maintenance actions KEEP / PROBE / REPLACE and the kademlia_std mode
(Requirements 14.3-14.9, 17.3).

Ground-truth side: this module evaluates pings via ``sim.network`` and must
not import ``rl`` or ``policies`` (candidate presence is checked directly on
the ``LocalView``).

Pings are atomic: every ping is evaluated against network state at the given
time ``t_ms`` and the clock is never advanced.

Key schema (``sim.keyed_rng``), ``dst`` is the pinged node's **ID** (the
``Entry.node`` / ``CacheEntry.node``), consistent with ``sim.lookup``:

* epoch ping:        ``(epoch_idx, bucket, dst, PingKind.INC | PingKind.CAND)``
* kademlia_std ping: ``(lookup_id, bucket, dst, PingKind.STD)``

Lookup queries use ``(lookup_id, dst)``. Keyed draws absorb the key length
first, so the 4-tuples used here are domain-separated from the 2-tuple lookup
keys even when ``epoch_idx == lookup_id``; the trailing kind separates epoch
pings from kademlia_std pings.

Every ping also feeds ``view.record_query_outcome(t_ms, timed_out)``.

Documented choices
------------------
* ``apply_action`` on an empty bucket raises ``ValueError`` (decisions are only
  requested for non-empty buckets).
* PROBE failure evicts the incumbent; the freed slot makes inserting the
  candidate valid. Without a candidate the slot is left for the next autofill.
* ``kademlia_std_on_offer`` precondition: bucket ``b`` is full and
  ``newcomer`` is not in the table but is in bucket ``b``'s cache (the lookup's
  ``offer`` just cached it). On incumbent failure the newcomer is inserted with
  its cached ``last_rtt_ms`` (``insert`` removes it from the cache); on
  success the incumbent is refreshed and the newcomer stays cached.
"""

from __future__ import annotations

from dataclasses import dataclass

from churnmaster.actions import Action  # single definition, re-exported here
from churnmaster.config import SimConfig
from churnmaster.sim.keyed_rng import PingKind
from churnmaster.sim.network import Network, QueryResult
from churnmaster.sim.routing_table import LocalView

__all__ = [
    "Action",
    "IllegalActionError",
    "ActionOutcome",
    "apply_action",
    "kademlia_std_on_offer",
]



class IllegalActionError(ValueError):
    """An action not allowed in the current state (REPLACE without candidate)."""


@dataclass(frozen=True)
class ActionOutcome:
    """Cost and effect of one maintenance step. Node fields are node IDs.

    ``inc_node`` / ``cand_node``: the incumbent and candidate (if any) the
    step was decided on, for the env's GT annex.
    """

    pings: int
    evictions: int
    inserted: int | None = None
    evicted: int | None = None
    inc_node: int | None = None
    cand_node: int | None = None


def _ping(view: LocalView, net: Network, node: int, t_ms: int,
          key: tuple[int, ...]) -> QueryResult:
    r = net.query(net.index_of(node), t_ms, key)
    view.record_query_outcome(t_ms, not r.success)
    return r


def apply_action(b: int, action: Action | int, epoch_idx: int, t_ms: int,
                 view: LocalView, net: Network, cfg: SimConfig | None = None
                 ) -> ActionOutcome:
    """Apply one epoch action to non-empty bucket ``b`` at ``t_ms``."""
    action = Action(action)
    inc = view.incumbent(b).node          # ValueError on an empty bucket
    c = view.candidate(b, t_ms)
    cand = None if c is None else c.node
    k = int(epoch_idx)

    if action == Action.KEEP:
        return ActionOutcome(0, 0, inc_node=inc, cand_node=cand)

    if action == Action.PROBE:
        r = _ping(view, net, inc, t_ms, (k, b, inc, PingKind.INC))
        if r.success:
            view.record_response(inc, t_ms, r.rtt_ms)
            return ActionOutcome(1, 0, inc_node=inc, cand_node=cand)
        view.record_failure(inc, t_ms)
        view.evict(inc)
        if cand is None:
            return ActionOutcome(1, 1, None, inc, inc, None)   # slot left for autofill
        rc = _ping(view, net, cand, t_ms, (k, b, cand, PingKind.CAND))
        if rc.success:
            view.insert(b, cand, t_ms, rc.rtt_ms)
            return ActionOutcome(2, 1, cand, inc, inc, cand)
        view.cache_drop(b, cand)
        return ActionOutcome(2, 1, None, inc, inc, cand)

    # REPLACE
    if cand is None:
        raise IllegalActionError(f"REPLACE without a candidate in bucket {b}")
    rc = _ping(view, net, cand, t_ms, (k, b, cand, PingKind.CAND))
    if rc.success:
        view.evict(inc)
        view.insert(b, cand, t_ms, rc.rtt_ms)
        return ActionOutcome(1, 1, cand, inc, inc, cand)
    view.cache_drop(b, cand)
    return ActionOutcome(1, 0, None, None, inc, cand)


def kademlia_std_on_offer(b: int, newcomer: int, lookup_id: int, t_ms: int,
                          view: LocalView, net: Network, cfg: SimConfig | None = None
                          ) -> ActionOutcome:
    """Classic Kademlia reaction to a responder cached by full bucket ``b``."""
    if len(view.bucket(b)) < view.k:
        raise ValueError(f"bucket {b} is not full")
    if newcomer in view:
        raise ValueError(f"newcomer {newcomer} is already in the table")
    cached = {ce.node: ce for ce in view.cache(b)}
    if newcomer not in cached:
        raise ValueError(f"newcomer {newcomer} is not in bucket {b}'s cache")
    inc = view.incumbent(b).node
    r = _ping(view, net, inc, t_ms, (int(lookup_id), b, inc, PingKind.STD))
    if r.success:
        view.record_response(inc, t_ms, r.rtt_ms)
        return ActionOutcome(1, 0, inc_node=inc, cand_node=newcomer)
    view.record_failure(inc, t_ms)
    view.evict(inc)
    view.insert(b, newcomer, t_ms, cached[newcomer].last_rtt_ms)   # also drops it from cache
    return ActionOutcome(1, 1, newcomer, inc, inc, newcomer)
