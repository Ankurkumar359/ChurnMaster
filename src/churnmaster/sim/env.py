"""Deterministic event-driven episode (Requirements 11.1-11.4, 12.6).

Time model
----------
* Integer-ms clock; the episode covers ``[0, horizon_ms)`` with
  ``horizon_ms = (warmup_s + measured_s) * 1000``. Events at or after the
  horizon are never generated, so the episode ends on the time limit.
* A record is *measured* iff its timestamp is ``>= warmup_ms``.

Event order
-----------
All events are merged once into a list sorted by ``(t_ms, priority, tie)``:

====================  ========  ==============================================
event                 priority  tie
====================  ========  ==============================================
churn (up/down)       0         node index
background rebuild    1         node index
GT dead-entry sample  2         0 (observes F's table before maintenance)
maintenance epoch     2         1
lookup                3         lookup_id
====================  ========  ==============================================

Documented choices
------------------
* Bootstrap (before any event): every background table gets its initial build
  at t=0 (``rebuild_idx`` 0, ``bg.build_all(0)``) and F's view is filled from
  ``net.bootstrap_entries(0)``. The t=0 builds are therefore not queued again.
* Churn events are informational: ground truth is read from the trace
  directly, but the events are still processed in order as the hook for
  future logic and GT snapshots.
* Epochs are at ``t_k = k * epoch_ms`` for ``k = 0, 1, ...`` with
  ``t_k < horizon_ms``; ``epoch_idx = k``.
* GT dead-entry samples every 60 s (``t = 0, 60 s, ...``): the fraction of
  F's table entries whose node is offline at t (0.0 for an empty table).
* ``run`` builds fresh background tables and a fresh LocalView each call;
  network and workload are fixed at construction, so repeated runs of the
  same Env are identical.

Maintenance epochs (Requirements 14.1, 14.2, 14.8, 21.4)
--------------------------------------------------------
At epoch ``k`` (time ``t_k``), when maintenance is on:

1. ``view.cache_expire(t_k)``;
2. for ``b`` in ``range(id_bits)``: ``view.autofill(b, t_k)`` (0 messages);
   an empty bucket emits a null ``DecisionRecord(b, k, t_k, None, None, 0, 0)``
   (it drops the bucket's open transition in the Ledger); otherwise exactly
   one ``decide(build_context(view, b, k, t_k))`` call, the action is applied
   via ``maintenance.apply_action`` and a ``DecisionRecord`` with its ping /
   eviction cost is emitted. With ``gt_annex`` a ``"decision"`` GTRecord is
   also emitted (``epoch_idx``, ``action``, ``inc_online``, ``cand_present``,
   ``cand_online``, ``inserted``, ``evicted``); online facts are read
   *before* the action is applied.

With maintenance off, epochs do nothing and no DecisionRecords are emitted.

Layering: ``sim/`` must not import ``rl/`` or ``policies/``. Hence

* ``decide`` receives only the observable ``DecisionContext`` and returns an
  ``Action`` (or int). The discrete state stored in ``DecisionRecord.state``
  comes from the optional ``encode_state(ctx) -> int`` argument of ``run``
  (experiments wire ``rl.state.encode(discretize(ctx))``); without it
  ``state`` is None.
* Illegal actions: an int that is not an ``Action`` and REPLACE without a
  candidate raise ``maintenance.IllegalActionError`` (Requirement 14.8; the
  latter is raised by ``apply_action`` and propagates).
* The oracle hook is provided *by* the env: ``Env.oracle_facts(bucket, t_ms)``
  returns ground-truth ``(inc_online, cand_present, cand_online)`` for the
  current incumbent / candidate of the bucket; experiments wrap it into
  ``policies.oracle.OracleFacts``. ``run`` therefore takes no oracle argument.

kademlia_std mode (Requirements 17.3, 17.4)
-------------------------------------------
* Epochs run exactly as above (same decision points and records); ``decide``
  must return KEEP, otherwise ``ValueError``.
* After each lookup (same ``t``), every responder of that lookup, in query
  order, that is not in F's table, is in its bucket's cache, and whose bucket
  is full triggers ``maintenance.kademlia_std_on_offer``. Its cost is emitted
  as a null-state ``DecisionRecord(b, t // epoch_ms, t, None, None, pings,
  evictions, measured)``; no GT decision record is emitted for it. The plain
  rule is followed: several newcomers for the same bucket in one lookup each
  ping the bucket's then-current LRS incumbent (possibly the same one twice).
  The responder list comes from the lookup's GT annex, but which nodes
  responded is observable to F.

Ground truth: observable-only modules must not import this module. GT
records are delivered only through ``gt_annex`` (Requirement 12.6);
``on_lookup`` / ``on_decision`` receive only observable records.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal

from churnmaster.config import Config
from churnmaster.seeding import EpisodeStreams
from churnmaster.sim.background import BackgroundTables
from churnmaster.sim.lookup import run_lookup_traced
from churnmaster.sim.maintenance import (Action, IllegalActionError, apply_action,
                                         kademlia_std_on_offer)
from churnmaster.sim.network import Network, build_network
from churnmaster.sim.observation import (DecisionContext, DecisionRecord, GTRecord,
                                         ObservedLookup, build_context)
from churnmaster.sim.routing_table import LocalView
from churnmaster.sim.workload import LookupRequest, generate_workload

__all__ = [
    "P_CHURN", "P_REBUILD", "P_EPOCH", "P_LOOKUP",
    "GT_SAMPLE_PERIOD_MS",
    "Event", "EpisodeTrace", "Env", "merge_events", "workload_hash",
]

P_CHURN, P_REBUILD, P_EPOCH, P_LOOKUP = 0, 1, 2, 3
_TIE_GT_SAMPLE, _TIE_EPOCH = 0, 1
GT_SAMPLE_PERIOD_MS = 60_000

# Event kinds (payload type in brackets).
K_CHURN = "churn"          # ("up" | "down")
K_REBUILD = "rebuild"      # rebuild_idx
K_GT_SAMPLE = "gt_sample"  # None
K_EPOCH = "epoch"          # epoch_idx
K_LOOKUP = "lookup"        # LookupRequest

Mode = Literal["epoch", "kademlia_std"]


@dataclass(frozen=True)
class Event:
    t_ms: int
    priority: int
    tie: int
    kind: str
    payload: Any

    @property
    def sort_key(self) -> tuple[int, int, int]:
        return (self.t_ms, self.priority, self.tie)


def merge_events(churn: Iterable[tuple[int, int, str]],
                 rebuilds: Iterable[tuple[int, int, int]],
                 epochs: Iterable[tuple[int, int]],
                 gt_samples: Iterable[int],
                 lookups: Iterable[LookupRequest],
                 horizon_ms: int) -> list[Event]:
    """Merge all event sources into one list sorted by ``(t, priority, tie)``.

    ``churn``: ``(t, node, kind)``; ``rebuilds``: ``(t, node, idx)``;
    ``epochs``: ``(t, epoch_idx)``; ``gt_samples``: ``t``. Events outside
    ``[0, horizon_ms)`` are dropped.
    """
    evs: list[Event] = []
    evs.extend(Event(int(t), P_CHURN, int(n), K_CHURN, kind) for t, n, kind in churn)
    evs.extend(Event(int(t), P_REBUILD, int(n), K_REBUILD, int(i)) for t, n, i in rebuilds)
    evs.extend(Event(int(t), P_EPOCH, _TIE_GT_SAMPLE, K_GT_SAMPLE, None) for t in gt_samples)
    evs.extend(Event(int(t), P_EPOCH, _TIE_EPOCH, K_EPOCH, int(k)) for t, k in epochs)
    evs.extend(Event(int(r.t_ms), P_LOOKUP, int(r.lookup_id), K_LOOKUP, r) for r in lookups)
    evs = [e for e in evs if 0 <= e.t_ms < horizon_ms]
    evs.sort(key=lambda e: (e.t_ms, e.priority, e.tie))
    return evs


def workload_hash(workload: Iterable[LookupRequest]) -> str:
    h = hashlib.sha256()
    for r in workload:
        h.update(f"{r.lookup_id},{r.t_ms},{r.key}\n".encode("ascii"))
    return h.hexdigest()


@dataclass(frozen=True)
class EpisodeTrace:
    """Summary of one processed episode (no per-record data)."""

    horizon_ms: int
    warmup_ms: int
    n_churn: int
    n_rebuild: int
    n_epoch: int
    n_gt_sample: int
    n_lookup: int
    n_measured_lookup: int
    trace_hash: str
    workload_hash: str
    events: tuple[tuple[int, int, int, str], ...] | None = None  # only if record_events


class Env:
    """One episode: ground truth, F's view, and the event loop."""

    def __init__(self, cfg: Config, streams: EpisodeStreams, mode: Mode = "epoch",
                 maintenance: bool = True, record_events: bool = False):
        if mode not in ("epoch", "kademlia_std"):
            raise ValueError(f"unknown mode {mode!r}")
        sim = cfg.sim
        self.cfg = cfg
        self.mode = mode
        self.maintenance = bool(maintenance)
        self.record_events = bool(record_events)
        self.warmup_ms = int(sim.warmup_s) * 1000
        self.horizon_ms = (int(sim.warmup_s) + int(sim.measured_s)) * 1000
        self.epoch_ms = int(sim.epoch_s) * 1000
        self.net: Network = build_network(sim, cfg.scenario, streams, self.horizon_ms)
        self.workload: list[LookupRequest] = generate_workload(
            streams.workload, sim.lookup_rate_hz, 0, self.horizon_ms, sim.id_bits)
        self.bg: BackgroundTables | None = None
        self.view: LocalView | None = None

    # ------------------------------------------------------------ schedule
    def events(self) -> list[Event]:
        """The episode's full event list in processing order."""
        bg = self.bg if self.bg is not None else BackgroundTables(self.net)
        rebuilds = [(t, n, i) for t, n, i in bg.rebuild_schedule(self.horizon_ms) if i > 0]
        churn = [(e.t_ms, e.node, e.kind) for e in self.net.trace.events()]
        epochs = [(t, k) for k, t in enumerate(range(0, self.horizon_ms, self.epoch_ms))]
        samples = range(0, self.horizon_ms, GT_SAMPLE_PERIOD_MS)
        return merge_events(churn, rebuilds, epochs, samples, self.workload, self.horizon_ms)

    def is_measured(self, t_ms: int) -> bool:
        return t_ms >= self.warmup_ms

    # ----------------------------------------------------------------- run
    def run(self, decide: Callable[[DecisionContext], Action | int] | None,
            on_decision: Callable[[DecisionRecord], None],
            on_lookup: Callable[[ObservedLookup], None],
            gt_annex: Callable[[GTRecord], None] | None = None,
            encode_state: Callable[[DecisionContext], int] | None = None) -> EpisodeTrace:
        """Process the episode; observable records go to ``on_lookup`` /
        ``on_decision``, GT records only to ``gt_annex``.

        ``decide`` may be None only with maintenance off. ``encode_state``
        fills ``DecisionRecord.state`` (None when not given).
        """
        if self.maintenance and decide is None:
            raise ValueError("decide is required when maintenance is on")
        sim = self.cfg.sim
        net = self.net
        self._decide, self._on_decision = decide, on_decision
        self._gt_annex, self._encode_state = gt_annex, encode_state
        std = self.mode == "kademlia_std" and self.maintenance

        # Bootstrap at t=0.
        self.bg = bg = BackgroundTables(net)
        bg.build_all(0)
        self.view = view = LocalView(
            net.focal_id, id_bits=sim.id_bits, k=sim.k, cache_size=sim.cache_size,
            cache_ttl_ms=int(sim.cache_ttl_s) * 1000,
            window_ms=int(sim.window_timeout_s) * 1000, rtt_beta=sim.rtt_beta)
        view.bootstrap(net.bootstrap_entries(0))

        counts = {K_CHURN: 0, K_REBUILD: 0, K_EPOCH: 0, K_GT_SAMPLE: 0, K_LOOKUP: 0}
        n_measured = 0
        log: list[tuple[int, int, int, str]] | None = [] if self.record_events else None
        warmup = self.warmup_ms

        for ev in self.events():
            t, kind = ev.t_ms, ev.kind
            counts[kind] += 1
            if log is not None:
                log.append((t, ev.priority, ev.tie, kind))
            if kind == K_LOOKUP:
                res, gt = run_lookup_traced(ev.payload, view, net, bg, sim)
                measured = t >= warmup
                n_measured += measured
                on_lookup(ObservedLookup(res.lookup_id, res.t_ms, res.bucket, res.success,
                                         res.latency_ms, measured, res.queries))
                if gt_annex is not None:
                    gt_annex(GTRecord.make(t, "lookup", res.bucket, measured,
                                           lookup_id=res.lookup_id, holders=gt.holders,
                                           queried=gt.queried, responders=gt.responders))
                if std:
                    self._kademlia_std_after_lookup(res.lookup_id, t, gt.responders, measured)
            elif kind == K_REBUILD:
                bg.rebuild(ev.tie, ev.payload, t)
            elif kind == K_CHURN:
                self._on_churn(t, ev.tie, ev.payload)
            elif kind == K_EPOCH:
                self._on_epoch(ev.payload, t)
            elif kind == K_GT_SAMPLE:
                self._on_gt_sample(t)

        return EpisodeTrace(
            horizon_ms=self.horizon_ms, warmup_ms=warmup,
            n_churn=counts[K_CHURN], n_rebuild=counts[K_REBUILD], n_epoch=counts[K_EPOCH],
            n_gt_sample=counts[K_GT_SAMPLE], n_lookup=counts[K_LOOKUP],
            n_measured_lookup=n_measured, trace_hash=net.trace.trace_hash(),
            workload_hash=workload_hash(self.workload),
            events=None if log is None else tuple(log))

    # ------------------------------------------------------------ handlers
    def _on_churn(self, t_ms: int, node: int, kind: str) -> None:
        """Informational: GT is read from the trace directly."""

    def _on_epoch(self, epoch_idx: int, t_ms: int) -> None:
        """Maintenance epoch (see module docstring); no-op with maintenance off."""
        if not self.maintenance:
            return
        view, net, sim = self.view, self.net, self.cfg.sim
        decide, emit, gt_annex = self._decide, self._on_decision, self._gt_annex
        encode = self._encode_state
        std = self.mode == "kademlia_std"
        k, measured = int(epoch_idx), t_ms >= self.warmup_ms
        view.cache_expire(t_ms)
        for b in range(view.id_bits):
            view.autofill(b, t_ms)
            if not view.bucket(b):
                emit(DecisionRecord(b, k, t_ms, None, None, 0, 0, measured))
                continue
            ctx = build_context(view, b, k, t_ms)
            state = None if encode is None else int(encode(ctx))
            facts = self.oracle_facts(b, t_ms) if gt_annex is not None else None
            raw = decide(ctx)
            try:
                action = Action(raw)
            except ValueError:
                raise IllegalActionError(f"unknown action {raw!r} for bucket {b}") from None
            if std and action != Action.KEEP:
                raise ValueError(f"kademlia_std mode requires KEEP at epochs, got {action!r}")
            out = apply_action(b, action, k, t_ms, view, net, sim)
            emit(DecisionRecord(b, k, t_ms, state, int(action), out.pings, out.evictions,
                                measured))
            if gt_annex is not None:
                inc_online, cand_present, cand_online = facts
                gt_annex(GTRecord.make(t_ms, "decision", b, measured, epoch_idx=k,
                                       action=int(action), inc_online=inc_online,
                                       cand_present=cand_present, cand_online=cand_online,
                                       inserted=out.inserted, evicted=out.evicted))

    def _kademlia_std_after_lookup(self, lookup_id: int, t_ms: int,
                                   responders: Iterable[int], measured: bool) -> None:
        view, net, sim = self.view, self.net, self.cfg.sim
        epoch_idx = t_ms // self.epoch_ms
        for n in responders:
            if n in view:
                continue
            b = view.bucket_of(n)
            if len(view.bucket(b)) < view.k or not view.in_cache(n):
                continue
            out = kademlia_std_on_offer(b, n, lookup_id, t_ms, view, net, sim)
            self._on_decision(DecisionRecord(b, epoch_idx, t_ms, None, None, out.pings,
                                             out.evictions, measured))

    # ------------------------------------------------------------- oracle
    def oracle_facts(self, bucket: int, t_ms: int) -> tuple[bool, bool, bool | None]:
        """GROUND TRUTH for ``bucket``'s current incumbent / candidate at ``t_ms``:
        ``(inc_online, cand_present, cand_online)``; ``cand_online`` is None
        without a candidate. Valid only during ``run`` for a non-empty bucket
        (ValueError otherwise). Experiments wrap it into ``OracleFacts``."""
        view, net = self.view, self.net
        if view is None:
            raise ValueError("oracle_facts is only available during run")
        inc = view.incumbent(bucket).node
        cand = view.candidate(bucket, t_ms)
        inc_online = net.is_online(net.index_of(inc), t_ms)
        if cand is None:
            return inc_online, False, None
        return inc_online, True, net.is_online(net.index_of(cand.node), t_ms)

    def _on_gt_sample(self, t_ms: int) -> None:
        if self._gt_annex is None:
            return
        net = self.net
        entries = self.view.all_entries()
        n_dead = sum(1 for e in entries if not net.is_online(net.index_of(e.node), t_ms))
        frac = n_dead / len(entries) if entries else 0.0
        self._gt_annex(GTRecord.make(t_ms, "dead_fraction", None, self.is_measured(t_ms),
                                     fraction=frac, n_entries=len(entries), n_dead=n_dead))
