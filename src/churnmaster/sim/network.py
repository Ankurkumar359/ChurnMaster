"""Hidden network ground truth (Requirement 7, bootstrap for 9.3).

Nodes are addressed by **index** ``0..N-1``; ``ids[i]`` is node i's 8-bit
Kademlia ID. ``id_of`` / ``index_of`` convert between the two. The focal node
F is index ``FOCAL_INDEX`` (0) by convention.

All per-message randomness (loss, jitter) and F's bootstrap selection are
keyed draws (``sim.keyed_rng``), so they depend only on the key tuple and
never on the policy or on call order. Positions, types, churn, and congestion
are drawn once per episode from the dedicated ``EpisodeStreams`` generators.

Ground truth: observable-only modules must not import this module.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from churnmaster.config import ScenarioConfig, SimConfig
from churnmaster.seeding import EpisodeStreams
from churnmaster.sim.churn import ChurnTrace, NodeType, draw_node_types, generate_trace
from churnmaster.sim.idspace import bucket_index, draw_ids
from churnmaster.sim.keyed_rng import Purpose, keyed_normal, keyed_uniform

__all__ = [
    "FOCAL_INDEX",
    "QueryResult",
    "Network",
    "draw_congestion",
    "build_network",
]

FOCAL_INDEX = 0
_SQRT2 = math.sqrt(2.0)


@dataclass(frozen=True)
class QueryResult:
    success: bool
    cost_ms: float
    rtt_ms: float | None


class _BootEntry(NamedTuple):
    # Field order matches routing_table.BootstrapEntry(node, t_ms, rtt_ms).
    node: int
    t_ms: int
    rtt_ms: float | None


def draw_congestion(rng: np.random.Generator, n_nodes: int, focal: int, share: float,
                    len_s: int, gap_s: int, horizon_ms: int) -> dict[int, list[tuple[int, int]]]:
    """Congestion-prone nodes and their half-open ``[start_ms, end_ms)`` periods.

    Exactly ``round(share * (N-1))`` non-focal nodes are chosen uniformly. For
    each chosen node (in increasing index order) the clock starts at 0 and
    alternates an Exp(gap_s) quiet gap with a fixed ``len_s`` congested period,
    clipped to ``[0, horizon]``. Nodes that are not congestion-prone are absent.
    """
    others = [i for i in range(n_nodes) if i != focal]
    n_prone = int(round(share * len(others)))
    perm = rng.permutation(len(others))
    prone = sorted(others[int(j)] for j in perm[:n_prone])
    len_ms = int(len_s) * 1000
    out: dict[int, list[tuple[int, int]]] = {}
    for node in prone:
        periods: list[tuple[int, int]] = []
        clock = 0
        while True:
            clock += max(1, int(round(float(rng.exponential(gap_s)) * 1000.0)))
            if clock >= horizon_ms:
                break
            end = min(clock + len_ms, horizon_ms)
            periods.append((clock, end))
            clock = end
        out[node] = periods
    return out


class Network:
    """Ground-truth view of the simulated DHT for one episode."""

    def __init__(self, cfg: SimConfig, ids: list[int], positions: np.ndarray,
                 types: list[NodeType], trace: ChurnTrace,
                 congestion: dict[int, list[tuple[int, int]]], keyed_root: int,
                 focal: int = FOCAL_INDEX):
        n = len(ids)
        if len(set(ids)) != n:
            raise ValueError("ids must be unique")
        if positions.shape != (n, 2) or len(types) != n or trace.n_nodes != n:
            raise ValueError("ids, positions, types, and trace must cover the same nodes")
        if not 0 <= focal < n:
            raise ValueError("focal must be a valid node index")
        self.cfg = cfg
        self.ids = list(ids)
        self.positions = positions
        self.types = types
        self.trace = trace
        self.congestion = congestion
        self.keyed_root = int(keyed_root)
        self.focal = focal
        self.n_nodes = n
        self._index = {nid: i for i, nid in enumerate(self.ids)}
        self._cong_starts = {k: [s for s, _ in v] for k, v in congestion.items()}
        # Base RTT (no jitter, no congestion) per node, from distance to F.
        d = np.linalg.norm(positions - positions[focal], axis=1) / _SQRT2
        self._base_rtt = [float(cfg.rtt_base_ms + cfg.rtt_span_ms * x) for x in d]

    # ----------------------------------------------------------------- ids
    @property
    def focal_id(self) -> int:
        return self.ids[self.focal]

    def id_of(self, node: int) -> int:
        return self.ids[node]

    def index_of(self, node_id: int) -> int:
        return self._index[node_id]

    def bucket_of(self, node: int) -> int:
        """Bucket of node index ``node`` relative to F (by ID)."""
        if node == self.focal:
            raise ValueError("F has no bucket relative to itself")
        return bucket_index(self.focal_id, self.ids[node])

    # --------------------------------------------------------------- state
    def is_online(self, node: int, t_ms: int) -> bool:
        return self.trace.is_online(node, t_ms)

    def is_congested(self, node: int, t_ms: int) -> bool:
        starts = self._cong_starts.get(node)
        if not starts:
            return False
        i = bisect_right(starts, t_ms) - 1
        return i >= 0 and t_ms < self.congestion[node][i][1]

    # ------------------------------------------------------------ messages
    def rtt_ms(self, node: int, t_ms: int, key: tuple[int, ...]) -> float:
        """(base + span·dist/√2) · exp(σ·z_JITTER(key)) · (mult if congested)."""
        z = keyed_normal(self.keyed_root, Purpose.JITTER, *key)
        rtt = self._base_rtt[node] * math.exp(self.cfg.rtt_sigma * z)
        if self.is_congested(node, t_ms):
            rtt *= self.cfg.congestion_mult
        return rtt

    def query(self, node: int, t_ms: int, key: tuple[int, ...]) -> QueryResult:
        """Success iff online and keyed_uniform(LOSS, *key) >= p_loss.

        Cost is the RTT on success, otherwise ``timeout_ms``.
        """
        if self.is_online(node, t_ms) and \
                keyed_uniform(self.keyed_root, Purpose.LOSS, *key) >= self.cfg.p_loss:
            rtt = self.rtt_ms(node, t_ms, key)
            return QueryResult(True, rtt, rtt)
        return QueryResult(False, float(self.cfg.timeout_ms), None)

    def holders(self, key: int, t_ms: int) -> list[int]:
        """Indices of the ``replicas`` online non-focal nodes XOR-closest to ``key``
        (by ID), in increasing distance."""
        online = [i for i in range(self.n_nodes)
                  if i != self.focal and self.trace.is_online(i, t_ms)]
        online.sort(key=lambda i: self.ids[i] ^ key)
        return online[: self.cfg.replicas]

    # ----------------------------------------------------------- bootstrap
    def bootstrap_entries(self, t_ms: int = 0) -> list[_BootEntry]:
        """F's initial table as ``(node_id, t_ms, None)`` tuples for
        ``LocalView.bootstrap``.

        Note ``node_id`` is the node's **ID** (LocalView works in ID space), not
        its index. For each bucket b (ascending), the online non-focal nodes in
        b are sorted by ID; slot s = 0..k-1 picks, without replacement, the
        remaining node at position ``floor(keyed_uniform(BOOT, b, s) * len)``.
        No message is exchanged, so ``rtt_ms`` is None.
        """
        by_bucket: list[list[int]] = [[] for _ in range(self.cfg.id_bits)]
        for i in range(self.n_nodes):
            if i != self.focal and self.trace.is_online(i, t_ms):
                by_bucket[self.bucket_of(i)].append(self.ids[i])
        out: list[_BootEntry] = []
        for b, pool in enumerate(by_bucket):
            pool.sort()
            for slot in range(min(self.cfg.k, len(pool))):
                u = keyed_uniform(self.keyed_root, Purpose.BOOT, b, slot)
                out.append(_BootEntry(pool.pop(int(u * len(pool))), t_ms, None))
        return out


def build_network(cfg: SimConfig, scn: ScenarioConfig, streams: EpisodeStreams,
                  horizon_ms: int, focal: int = FOCAL_INDEX) -> Network:
    """Draw one episode's ground truth from its streams.

    ids <- streams.ids; types then positions <- streams.types_positions;
    trace <- streams.churn; congestion <- streams.congestion.
    """
    ids = draw_ids(streams.ids, cfg.n_nodes, cfg.id_bits)
    types = draw_node_types(streams.types_positions, cfg.n_nodes, scn.stable_share, focal)
    positions = streams.types_positions.random((cfg.n_nodes, 2))
    trace = generate_trace(scn, types, streams.churn, horizon_ms, focal)
    congestion = draw_congestion(streams.congestion, cfg.n_nodes, focal, cfg.congestion_share,
                                 cfg.congestion_len_s, cfg.congestion_gap_s, horizon_ms)
    return Network(cfg, ids, positions, types, trace, congestion, streams.keyed_root, focal)
