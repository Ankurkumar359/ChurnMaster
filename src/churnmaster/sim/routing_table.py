"""F's observable local view: routing table buckets plus per-bucket candidate cache.

Observable-only module (Requirement 9). It must not import ground-truth modules
(``sim.network``, ``sim.churn``, ``sim.background``, ``sim.lookup``, ``sim.env``,
``sim.keyed_rng``) nor ``rl``/``policies``. It imports only stdlib and
``churnmaster.sim.idspace``.

Identification: every entry is keyed by **node ID** (the 8-bit Kademlia ID of
the peer, not a population index). The bucket of a node is
``bucket_index(focal_id, node_id)``.

Documented semantic choices
---------------------------
* Cache TTL: a cache entry is *expired* at time ``t`` iff
  ``t - last_seen_ms > cache_ttl_ms`` (strictly older than 300 s). An entry seen
  exactly 300 s ago is still valid.
* Timeout window: a query recorded at ``tq`` counts at time ``t`` iff
  ``t - window_ms < tq <= t`` (a query exactly 120 s old has left the window).
  Query times are assumed non-decreasing.
* ``bootstrap``: an entry with ``rtt_ms is None`` is inserted *without* a
  response (``last_response_ms=None``, ``srtt_ms=None``); if an RTT is given it
  is treated as a response at ``t_ms``.
* ``offer`` / ``insert``: the node has just responded, so the new entry gets
  ``last_response_ms=t_ms`` and ``srtt_ms=rtt_ms``. The node is removed from the
  cache if present (no duplicates across table and cache).
* ``autofill``: cache entries were observed responding, so a promoted entry gets
  ``inserted_ms=t_ms``, ``last_response_ms=last_seen_ms``, ``srtt_ms=last_rtt_ms``.
  Expired cache entries of the bucket are dropped first; no messages are sent.
* ``offer`` on a node already in the table is a no-op: response statistics are
  maintained by the lookup via ``record_response`` / ``record_failure``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, NamedTuple

from churnmaster.sim.idspace import bucket_index


@dataclass
class Entry:
    """A routing-table entry (observable statistics only)."""

    node: int
    inserted_ms: int
    last_response_ms: int | None
    consec_fail: int = 0
    srtt_ms: float | None = None


@dataclass
class CacheEntry:
    """A candidate-cache entry: a responder seen while its bucket was full."""

    node: int
    last_seen_ms: int
    last_rtt_ms: float


class BootstrapEntry(NamedTuple):
    """One bootstrap insertion, computed GT-side (``Network.bootstrap_entries``)."""

    node: int
    t_ms: int
    rtt_ms: float | None = None


class OfferOutcome(Enum):
    IN_TABLE = "in_table"
    INSERTED = "inserted"
    CACHED = "cached"


class LocalView:
    """F's routing table (``id_bits`` buckets of ``k`` entries) plus candidate caches."""

    def __init__(
        self,
        focal_id: int,
        id_bits: int = 8,
        k: int = 4,
        cache_size: int = 4,
        cache_ttl_ms: int = 300_000,
        window_ms: int = 120_000,
        rtt_beta: float = 0.3,
    ) -> None:
        if not 0 <= focal_id < (1 << id_bits):
            raise ValueError(f"focal_id {focal_id} outside {id_bits}-bit space")
        self.focal_id = focal_id
        self.id_bits = id_bits
        self.k = k
        self.cache_size = cache_size
        self.cache_ttl_ms = cache_ttl_ms
        self.window_ms = window_ms
        self.rtt_beta = rtt_beta
        # Buckets keep insertion order; caches are keyed by node ID.
        self._buckets: list[list[Entry]] = [[] for _ in range(id_bits)]
        self._caches: list[dict[int, CacheEntry]] = [{} for _ in range(id_bits)]
        self._index: dict[int, Entry] = {}
        self._queries: deque[tuple[int, bool]] = deque()
        self._n_timeouts = 0

    # ------------------------------------------------------------------ access
    def bucket_of(self, node: int) -> int:
        if node == self.focal_id:
            raise ValueError("focal node cannot be stored in its own view")
        if not 0 <= node < (1 << self.id_bits):
            raise ValueError(f"node {node} outside {self.id_bits}-bit space")
        return bucket_index(self.focal_id, node)

    def bucket(self, b: int) -> list[Entry]:
        return list(self._buckets[b])

    def cache(self, b: int) -> list[CacheEntry]:
        return list(self._caches[b].values())

    def all_entries(self) -> list[Entry]:
        return [e for bucket in self._buckets for e in bucket]

    def closest_entries(self, target: int, n: int) -> list[Entry]:
        if n <= 0:
            return []
        return sorted(self._index.values(), key=lambda e: e.node ^ target)[:n]

    def entry(self, node: int) -> Entry:
        return self._index[node]

    def __contains__(self, node: object) -> bool:
        return node in self._index

    def __len__(self) -> int:
        return len(self._index)

    def in_cache(self, node: int) -> bool:
        return node in self._caches[self.bucket_of(node)]

    # ------------------------------------------------------------- statistics
    def record_response(self, node: int, t_ms: int, rtt_ms: float) -> None:
        """Successful response from a table member (KeyError if not a member)."""
        e = self._index[node]
        e.consec_fail = 0
        e.last_response_ms = t_ms
        if e.srtt_ms is None:
            e.srtt_ms = float(rtt_ms)
        else:
            e.srtt_ms = (1.0 - self.rtt_beta) * e.srtt_ms + self.rtt_beta * rtt_ms

    def record_failure(self, node: int, t_ms: int) -> None:
        """Timeout from a table member (KeyError if not a member)."""
        self._index[node].consec_fail += 1

    # ------------------------------------------------------------ table edits
    def _add(self, b: int, entry: Entry) -> None:
        self._buckets[b].append(entry)
        self._index[entry.node] = entry
        self._caches[b].pop(entry.node, None)

    def insert(self, b: int, node: int, t_ms: int, rtt_ms: float | None) -> Entry:
        """Insert a node that just responded (``rtt_ms``) into bucket ``b``.

        Raises ValueError if ``b`` is not the node's bucket, the node is already
        in the table, or the bucket is full.
        """
        if self.bucket_of(node) != b:
            raise ValueError(f"node {node} does not belong to bucket {b}")
        if node in self._index:
            raise ValueError(f"node {node} already in table")
        if len(self._buckets[b]) >= self.k:
            raise ValueError(f"bucket {b} is full")
        entry = Entry(
            node=node,
            inserted_ms=t_ms,
            last_response_ms=None if rtt_ms is None else t_ms,
            srtt_ms=None if rtt_ms is None else float(rtt_ms),
        )
        self._add(b, entry)
        return entry

    def evict(self, node: int) -> Entry:
        """Remove a node from the table (KeyError if absent)."""
        entry = self._index.pop(node)
        self._buckets[self.bucket_of(node)].remove(entry)
        return entry

    def bootstrap(self, entries: Iterable[BootstrapEntry | tuple]) -> None:
        """Insert ``(node, t_ms, rtt_ms_or_None)`` items in the given order."""
        for item in entries:
            be = BootstrapEntry(*item)
            self.insert(self.bucket_of(be.node), be.node, be.t_ms, be.rtt_ms)

    def offer(self, node: int, t_ms: int, rtt_ms: float) -> OfferOutcome:
        """A responder observed during a lookup is offered to the table."""
        if node in self._index:
            return OfferOutcome.IN_TABLE
        b = self.bucket_of(node)
        if len(self._buckets[b]) < self.k:
            self.insert(b, node, t_ms, rtt_ms)
            return OfferOutcome.INSERTED
        self.cache_put(b, node, t_ms, rtt_ms)
        return OfferOutcome.CACHED

    # ------------------------------------------------------------------ cache
    def _expired(self, c: CacheEntry, t_ms: int) -> bool:
        return t_ms - c.last_seen_ms > self.cache_ttl_ms

    def cache_put(self, b: int, node: int, t_ms: int, rtt_ms: float) -> None:
        """Add or refresh ``node`` in bucket ``b``'s cache; evict oldest when full."""
        if self.bucket_of(node) != b:
            raise ValueError(f"node {node} does not belong to bucket {b}")
        if node in self._index:
            raise ValueError(f"node {node} is in the table; cannot cache it")
        cache = self._caches[b]
        if node in cache:
            c = cache[node]
            c.last_seen_ms = t_ms
            c.last_rtt_ms = float(rtt_ms)
            return
        cache[node] = CacheEntry(node, t_ms, float(rtt_ms))
        while len(cache) > self.cache_size:
            # Oldest last_seen; ties: higher RTT first, then node ID (deterministic).
            victim = min(cache.values(), key=lambda c: (c.last_seen_ms, -c.last_rtt_ms, c.node))
            del cache[victim.node]

    def cache_drop(self, b: int, node: int) -> bool:
        """Remove ``node`` from bucket ``b``'s cache; returns whether it was present."""
        return self._caches[b].pop(node, None) is not None

    def _expire_bucket(self, b: int, t_ms: int) -> None:
        cache = self._caches[b]
        for node in [n for n, c in cache.items() if self._expired(c, t_ms)]:
            del cache[node]

    def cache_expire(self, t_ms: int) -> None:
        for b in range(self.id_bits):
            self._expire_bucket(b, t_ms)

    def _ranked_cache(self, b: int, t_ms: int) -> list[CacheEntry]:
        valid = [c for c in self._caches[b].values() if not self._expired(c, t_ms)]
        # Most recent first, ties by lower RTT, then node ID for determinism.
        return sorted(valid, key=lambda c: (-c.last_seen_ms, c.last_rtt_ms, c.node))

    def autofill(self, b: int, t_ms: int) -> int:
        """Fill empty slots of bucket ``b`` from its cache (0 messages). Returns count."""
        self._expire_bucket(b, t_ms)
        filled = 0
        for c in self._ranked_cache(b, t_ms):
            if len(self._buckets[b]) >= self.k:
                break
            self._add(
                b,
                Entry(
                    node=c.node,
                    inserted_ms=t_ms,
                    last_response_ms=c.last_seen_ms,
                    srtt_ms=c.last_rtt_ms,
                ),
            )
            filled += 1
        return filled

    # -------------------------------------------------------------- selection
    def incumbent(self, b: int) -> Entry:
        """Least-recently-seen entry (ValueError on an empty bucket)."""
        bucket = self._buckets[b]
        if not bucket:
            raise ValueError(f"bucket {b} is empty")
        return min(
            bucket,
            key=lambda e: (
                e.last_response_ms if e.last_response_ms is not None else e.inserted_ms,
                e.inserted_ms,
            ),
        )  # min() keeps the first (earliest-inserted) element on full ties.

    def candidate(self, b: int, t_ms: int) -> CacheEntry | None:
        ranked = self._ranked_cache(b, t_ms)
        return ranked[0] if ranked else None

    # -------------------------------------------------------- timeout window
    def _prune(self, t_ms: int) -> None:
        q = self._queries
        while q and q[0][0] <= t_ms - self.window_ms:
            _, timed_out = q.popleft()
            self._n_timeouts -= timed_out

    def record_query_outcome(self, t_ms: int, timed_out: bool) -> None:
        self._queries.append((t_ms, bool(timed_out)))
        self._n_timeouts += bool(timed_out)
        self._prune(t_ms)

    def timeout_fraction(self, t_ms: int) -> float:
        self._prune(t_ms)
        if not self._queries:
            return 0.0
        return self._n_timeouts / len(self._queries)
