"""Atomic iterative lookup from the focal node F (Requirement 10).

A lookup for key ``K`` issued at ``t0`` is evaluated entirely against network
state at ``t0``; the simulation clock is never advanced. Round durations only
accumulate into the reported latency.

Algorithm (design "Lookup (atomic at t0)"):

1. The shortlist starts with F's 4 XOR-closest table entries to ``K`` (no
   liveness filter: dead entries are queried and time out).
2. Each round queries up to ``alpha`` unqueried shortlist nodes closest to
   ``K``. A query costs its RTT on success and ``timeout_ms`` otherwise; the
   round lasts as long as its slowest query.
3. If a replica holder responded and ``elapsed + min(holder RTT)`` is within
   the deadline, the lookup succeeds with that latency.
4. Otherwise the round's duration is added; if ``elapsed > deadline_ms`` the
   lookup fails with ``"deadline"`` (latency = ``deadline_ms``). Responders
   return their 4 closest known contacts (stale background tables), which are
   merged into the shortlist; the shortlist keeps the 4 closest unqueried
   nodes, never F.
5. Before a new round: no unqueried shortlist node -> ``"exhausted"``;
   ``max_rounds`` rounds already used -> ``"rounds"``. Both fail with
   latency = elapsed time at give-up.

F-side side effects, per round, in batch order (closest to ``K`` first):
every query result feeds ``view.record_query_outcome(t0, timed_out)``; a
queried node that is in F's table gets ``record_response`` /
``record_failure``. After those statistic updates, every responder of the
round is offered to F's view (``view.offer(id, t0, rtt)``: no-op if already
in the table, insert into an empty slot, else cache). Offers happen for every
round, including the final one.

Identifier conventions (documented choice):

* The shortlist, XOR distances, F's view, and the query key all use node
  **IDs**. ``Network.query`` and ``BackgroundTables.closest_known`` use node
  **indices**; conversion goes through ``Network.index_of`` / ``id_of``.
* The keyed query key is ``(lookup_id, dst)`` where ``dst`` is the
  destination node's **ID** (consistent with maintenance, which keys pings by
  the ``Entry.node`` ID).
* The result is attributed to ``bucket_index(F_id, K)`` (``K == F`` -> 0).

Ground truth: observable-only modules must not import this module. The
``LookupGT`` annex returned by ``run_lookup_traced`` contains hidden facts
(replica holders) and is for diagnostics/tests only; it must never reach a
policy or the ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from churnmaster.config import SimConfig
from churnmaster.sim.background import BackgroundTables
from churnmaster.sim.idspace import bucket_index, closest
from churnmaster.sim.network import Network, QueryResult
from churnmaster.sim.routing_table import LocalView
from churnmaster.sim.workload import LookupRequest

__all__ = ["FailReason", "LookupResult", "LookupGT", "run_lookup", "run_lookup_traced"]

FailReason = Literal["deadline", "rounds", "exhausted"]
SHORTLIST_SIZE = 4


@dataclass(frozen=True)
class LookupResult:
    """Observable outcome of one lookup (what F itself can know)."""

    lookup_id: int
    t_ms: int
    key: int
    bucket: int
    success: bool
    latency_ms: float
    rounds: int
    queries: int
    fail_reason: FailReason | None


@dataclass(frozen=True)
class LookupGT:
    """GROUND-TRUTH-ONLY annex for diagnostics and tests.

    ``holders``: node IDs of the replica holders at ``t0``.
    ``queried``: node IDs in query order. ``responders``: IDs that answered,
    in query order.
    """

    holders: tuple[int, ...]
    queried: tuple[int, ...]
    responders: tuple[int, ...]


def run_lookup(req: LookupRequest, view: LocalView, net: Network, bg: BackgroundTables,
               cfg: SimConfig) -> LookupResult:
    """Run one atomic lookup and return its observable result."""
    return run_lookup_traced(req, view, net, bg, cfg)[0]


def run_lookup_traced(req: LookupRequest, view: LocalView, net: Network,
                      bg: BackgroundTables, cfg: SimConfig
                      ) -> tuple[LookupResult, LookupGT]:
    """Run one atomic lookup; also return the GT-only annex."""
    t0, key, j = int(req.t_ms), int(req.key), int(req.lookup_id)
    focal_id = net.focal_id
    bucket = bucket_index(focal_id, key)
    holders = tuple(net.id_of(i) for i in net.holders(key, t0))
    holder_set = set(holders)
    deadline = float(cfg.deadline_ms)

    shortlist = {e.node for e in view.closest_entries(key, SHORTLIST_SIZE)}
    shortlist.discard(focal_id)
    queried: set[int] = set()
    queried_order: list[int] = []
    responders: list[int] = []
    elapsed = 0.0
    rounds = 0

    def finish(success: bool, latency: float, reason: FailReason | None):
        res = LookupResult(j, t0, key, bucket, success, float(latency), rounds,
                           len(queried_order), reason)
        return res, LookupGT(holders, tuple(queried_order), tuple(responders))

    while True:
        batch = closest(key, shortlist - queried, cfg.alpha)
        if not batch:
            return finish(False, elapsed, "exhausted")
        if rounds == cfg.max_rounds:
            return finish(False, elapsed, "rounds")
        rounds += 1

        # Atomic: every query is evaluated at t0; dst in the key is the node ID.
        results: list[tuple[int, QueryResult]] = [
            (n, net.query(net.index_of(n), t0, (j, n))) for n in batch]
        queried.update(batch)
        queried_order.extend(batch)

        # F-side statistics.
        for n, r in results:
            view.record_query_outcome(t0, not r.success)
            if n in view:
                if r.success:
                    view.record_response(n, t0, r.rtt_ms)
                else:
                    view.record_failure(n, t0)
        ok = [(n, r) for n, r in results if r.success]
        responders.extend(n for n, _ in ok)

        def offer_responders() -> None:
            for n, r in ok:
                view.offer(n, t0, r.rtt_ms)

        hit = [r.rtt_ms for n, r in ok if n in holder_set]
        if hit and elapsed + min(hit) <= deadline:
            offer_responders()
            return finish(True, elapsed + min(hit), None)

        elapsed += max(r.cost_ms for _, r in results)
        if elapsed > deadline:
            offer_responders()
            return finish(False, deadline, "deadline")

        learned = {net.id_of(c) for n, _ in ok
                   for c in bg.closest_known(net.index_of(n), key, SHORTLIST_SIZE)}
        pool = (shortlist | learned) - queried
        pool.discard(focal_id)
        shortlist = set(closest(key, pool, SHORTLIST_SIZE))
        offer_responders()
