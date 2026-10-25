"""Stale background routing tables of the non-focal nodes (Requirement 8).

Every non-focal node keeps a Kademlia table (``id_bits`` buckets of up to
``k`` contacts, buckets relative to the node's **own** ID). Tables are rebuilt
from ground truth on a fixed, policy-independent schedule and are frozen in
between, so a responder can hand out contacts that have since gone offline,
as in real DHTs.

Conventions (documented choices):

* Contacts are node **indices** (``0..N-1``), matching ``Network.query``.
  ``closest_known`` orders them by the XOR distance of their IDs.
* F (the focal node) is a real online node, so it may appear in other nodes'
  tables. Lookups remove F from the shortlist themselves (``- {F}``).
  F itself owns no background table.
* Keyed order: in bucket ``b`` of node ``u`` at rebuild ``r``, online
  candidates ``c`` are sorted by ``keyed_uniform(root, BG, u, r, c)`` (the
  design key ``(BG, node, rebuild_idx)`` extended with the candidate index,
  ties broken by index) and the first ``k`` are kept. The shared hash prefix
  is computed once per rebuild (bit-identical to ``keyed_uniform``).
* Schedule: every non-focal node is built at ``t = 0`` (``rebuild_idx`` 0),
  then rebuilt at ``node_idx*P//N + P*m`` ms for ``m >= 0`` with the time
  ``> 0`` (``P = rebuild_period_s * 1000``), so no node is built twice at
  ``t = 0``. Times are half-open ``[0, horizon_ms)``; ``rebuild_idx`` counts
  builds of that node starting at 0.

Ground truth: observable-only modules must not import this module.
"""

from __future__ import annotations

import heapq
from typing import Iterator

from churnmaster.sim.idspace import bucket_index
from churnmaster.sim.keyed_rng import MASK64, Purpose, splitmix64
from churnmaster.sim.network import Network

__all__ = ["BackgroundTables"]


def _bg_prefix(root: int, node: int, rebuild_idx: int) -> int:
    # Mirrors keyed_rng._hash for a 3-id key, stopping before the last id.
    # splitmix64(prefix ^ c) >> 11 is keyed_uniform(root, BG, node,
    # rebuild_idx, c) scaled by 2**53, so the sort order is identical.
    h = splitmix64(int(root) & MASK64)
    h = splitmix64(h ^ int(Purpose.BG))
    h = splitmix64(h ^ 3)
    h = splitmix64(h ^ (int(node) & MASK64))
    return splitmix64(h ^ (int(rebuild_idx) & MASK64))


class BackgroundTables:
    """Per-node stale routing tables, rebuilt only on schedule."""

    def __init__(self, net: Network):
        self.net = net
        self.n_nodes = net.n_nodes
        self.focal = net.focal
        self.k = int(net.cfg.k)
        self.id_bits = int(net.cfg.id_bits)
        self.period_ms = int(net.cfg.rebuild_period_s) * 1000
        ids = net.ids
        # members[u][b]: indices of all other nodes in bucket b of node u,
        # ascending by index. Precomputed once so a rebuild is O(N).
        self._members: list[list[list[int]]] = []
        for u in range(self.n_nodes):
            buckets: list[list[int]] = [[] for _ in range(self.id_bits)]
            for v in range(self.n_nodes):
                if v != u:
                    buckets[bucket_index(ids[u], ids[v])].append(v)
            self._members.append(buckets)
        # _tables[u][b]: current contacts of bucket b; _flat[u]: all contacts.
        self._tables: list[list[tuple[int, ...]]] = [
            [() for _ in range(self.id_bits)] for _ in range(self.n_nodes)]
        self._flat: list[tuple[int, ...]] = [() for _ in range(self.n_nodes)]
        self._last_build: list[tuple[int, int] | None] = [None] * self.n_nodes

    # ------------------------------------------------------------ schedule
    def node_rebuild_times(self, node: int, horizon_ms: int) -> list[int]:
        """Build times of ``node`` in ``[0, horizon_ms)``; index = rebuild_idx."""
        self._check_node(node)
        if horizon_ms <= 0:
            return []
        times = [0]
        t = node * self.period_ms // self.n_nodes
        if t == 0:
            t += self.period_ms
        while t < horizon_ms:
            times.append(t)
            t += self.period_ms
        return times

    def rebuild_schedule(self, horizon_ms: int) -> Iterator[tuple[int, int, int]]:
        """Yield ``(t_ms, node, rebuild_idx)`` for all non-focal nodes, sorted
        by ``(t_ms, node)``."""
        events = [(t, u, i)
                  for u in range(self.n_nodes) if u != self.focal
                  for i, t in enumerate(self.node_rebuild_times(u, horizon_ms))]
        events.sort()
        yield from events

    # ------------------------------------------------------------- rebuild
    def rebuild(self, node: int, rebuild_idx: int, t_ms: int) -> None:
        """Refill every bucket of ``node`` from nodes online at ``t_ms``."""
        self._check_node(node)
        trace = self.net.trace
        root = self.net.keyed_root
        k = self.k
        # Bit-identical to keyed_uniform(root, BG, node, rebuild_idx, c): the
        # hash chain over (root, BG, len=3, node, rebuild_idx) is shared by all
        # candidates, so absorb it once and only mix in c per draw.
        prefix = _bg_prefix(root, node, rebuild_idx)
        table: list[tuple[int, ...]] = []
        for members in self._members[node]:
            online = [c for c in members if trace.is_online(c, t_ms)]
            if len(online) > k:
                online.sort(key=lambda c: (splitmix64(prefix ^ c) >> 11, c))
                online = online[:k]
            table.append(tuple(online))
        self._tables[node] = table
        self._flat[node] = tuple(c for bucket in table for c in bucket)
        self._last_build[node] = (rebuild_idx, t_ms)

    def build_all(self, t_ms: int = 0) -> None:
        """Initial build (rebuild_idx 0) of every non-focal node at ``t_ms``."""
        for u in range(self.n_nodes):
            if u != self.focal:
                self.rebuild(u, 0, t_ms)

    # --------------------------------------------------------------- query
    def closest_known(self, node: int, target: int, n: int = 4) -> list[int]:
        """The ``n`` contacts of ``node`` XOR-closest (by ID) to ``target`` ID,
        in increasing distance, regardless of their current online status."""
        self._check_node(node)
        if n <= 0:
            return []
        ids = self.net.ids
        return heapq.nsmallest(n, self._flat[node], key=lambda c: ids[c] ^ target)

    def table(self, node: int) -> list[tuple[int, ...]]:
        """Current contacts of ``node`` per bucket (index b = bucket b)."""
        self._check_node(node)
        return list(self._tables[node])

    def last_build(self, node: int) -> tuple[int, int] | None:
        """``(rebuild_idx, t_ms)`` of the node's latest build, or None."""
        return self._last_build[node]

    def _check_node(self, node: int) -> None:
        if not 0 <= node < self.n_nodes:
            raise ValueError(f"invalid node index {node}")
        if node == self.focal:
            raise ValueError("F has no background table")
