"""Record types crossing the env boundary (Requirements 11, 12).

Observable-only module: it imports nothing but the standard library, so the
env can emit these records without importing ``rl/`` and ``rl/ledger.py`` can
import them without touching ground truth (enforced by the import guard).

* ``ObservedLookup`` and ``DecisionRecord`` carry only what F itself can
  observe; they go to the Ledger and Metrics.
* ``GTRecord`` carries hidden ground-truth facts (dead-entry fractions, true
  online state of incumbent/candidate, replica holders). It is delivered only
  through the env's ``gt_annex`` callback, i.e. to Metrics, never to a policy
  or the Ledger.

``DecisionRecord.action`` is typed as a plain ``int``: the ``Action`` IntEnum
lives in ``sim/maintenance.py`` (task 5.2) and, being an IntEnum, is itself an
``int``, so ``Action`` members can be stored here without an import cycle.

``DecisionContext`` / ``build_context`` (Requirement 12.1) build the observable
feature vector for one bucket at one epoch from F's ``LocalView`` only. The
context deliberately carries no node IDs: a policy must not see identities;
maintenance re-derives incumbent/candidate from the view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # observable module; type-only import avoids a runtime dependency
    from churnmaster.sim.routing_table import LocalView

__all__ = [
    "DecisionContext",
    "build_context",
    "ObservedLookup",
    "DecisionRecord",
    "GTKind",
    "GTRecord",
]


@dataclass(frozen=True)
class DecisionContext:
    """Observable features of one non-empty bucket at one decision epoch.

    Incumbent = least-recently-seen entry of the bucket; candidate = best
    valid cache entry (None when the cache has no valid entry, in which case
    both ``cand_*`` fields are None). Durations are in seconds.
    """

    bucket: int
    epoch_idx: int
    t_ms: int
    inc_consec_fail: int
    inc_since_response_s: float
    inc_age_s: float
    inc_srtt_ms: float | None
    cand_since_seen_s: float | None
    cand_last_rtt_ms: float | None
    timeout_fraction: float


def build_context(view: "LocalView", b: int, epoch_idx: int, t_ms: int) -> DecisionContext:
    """Build the DecisionContext for bucket ``b`` at ``t_ms``.

    ``inc_since_response_s`` falls back to insertion time when the incumbent
    has never responded. Raises ValueError for an empty bucket.
    """
    inc = view.incumbent(b)  # raises ValueError on an empty bucket
    last = inc.last_response_ms if inc.last_response_ms is not None else inc.inserted_ms
    cand = view.candidate(b, t_ms)
    return DecisionContext(
        bucket=int(b),
        epoch_idx=int(epoch_idx),
        t_ms=int(t_ms),
        inc_consec_fail=int(inc.consec_fail),
        inc_since_response_s=(t_ms - last) / 1000.0,
        inc_age_s=(t_ms - inc.inserted_ms) / 1000.0,
        inc_srtt_ms=None if inc.srtt_ms is None else float(inc.srtt_ms),
        cand_since_seen_s=None if cand is None else (t_ms - cand.last_seen_ms) / 1000.0,
        cand_last_rtt_ms=None if cand is None else float(cand.last_rtt_ms),
        timeout_fraction=float(view.timeout_fraction(t_ms)),
    )


@dataclass(frozen=True)
class ObservedLookup:
    """Observable outcome of one lookup.

    ``queries`` is the number of FIND_NODE messages F sent for the lookup
    (observable; used by Metrics for messages per lookup). It defaults to 0
    so hand-made records may omit it.
    """

    lookup_id: int
    t_ms: int
    bucket: int
    success: bool
    latency_ms: float
    measured: bool
    queries: int = 0


@dataclass(frozen=True)
class DecisionRecord:
    """One bucket-epoch decision (or a null record).

    ``state`` / ``action`` are None for null records: empty buckets at an
    epoch, and ``kademlia_std`` ping costs (which still enter J).
    ``action`` is an ``int`` (``maintenance.Action`` is an IntEnum).
    """

    bucket: int
    epoch_idx: int
    t_ms: int
    state: int | None
    action: int | None
    pings: int
    evictions: int
    measured: bool


GTKind = Literal["dead_fraction", "decision", "lookup"]


@dataclass(frozen=True)
class GTRecord:
    """GROUND-TRUTH-ONLY diagnostic record (delivered via ``gt_annex`` only).

    ``values`` is an immutable tuple of ``(name, value)`` pairs so new kinds
    and fields can be added without changing the type. Current payloads:

    * ``dead_fraction``: ``fraction``, ``n_entries``, ``n_dead`` (bucket None)
    * ``lookup``: ``lookup_id``, ``holders``, ``queried``, ``responders``
    * ``decision`` (task 5.7): e.g. ``epoch_idx``, ``action``,
      ``inc_online``, ``cand_online``
    """

    t_ms: int
    kind: GTKind
    bucket: int | None
    measured: bool
    values: tuple[tuple[str, Any], ...] = field(default=())

    @classmethod
    def make(cls, t_ms: int, kind: GTKind, bucket: int | None, measured: bool,
             **values: Any) -> "GTRecord":
        return cls(int(t_ms), kind, bucket, bool(measured), tuple(values.items()))

    def get(self, name: str, default: Any = None) -> Any:
        for k, v in self.values:
            if k == name:
                return v
        return default

    def as_dict(self) -> dict[str, Any]:
        return dict(self.values)
