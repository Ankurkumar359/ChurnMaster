"""Hidden churn trace (Requirement 5).

Each non-focal node alternates Weibull(shape) sessions and exponential
downtimes, starting at ``t = -burn_in`` in a random state (online with the
type's stationary probability). The trace is clipped to ``[0, horizon]``.
The focal node F has a single session covering the whole episode.

Nodes are indexed ``0..N-1`` (indices, not IDs). Sessions are half-open
``[start_ms, end_ms)``; all times are integer milliseconds.

This module depends only on the standard library, numpy, and
``churnmaster.config``.
"""

from __future__ import annotations

import hashlib
import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import Iterator, Literal, NamedTuple

import numpy as np

from churnmaster.config import ScenarioConfig

__all__ = [
    "NodeType",
    "Session",
    "ChurnEvent",
    "ChurnTrace",
    "draw_node_types",
    "generate_trace",
    "analytic_online_fraction",
    "weibull_scale",
]


@dataclass(frozen=True)
class NodeType:
    stable: bool


@dataclass(frozen=True)
class Session:
    node: int
    start_ms: int
    end_ms: int


class ChurnEvent(NamedTuple):
    t_ms: int
    node: int
    kind: Literal["down", "up"]


def draw_node_types(rng: np.random.Generator, n_nodes: int, stable_share: float,
                    focal: int | None = None) -> list[NodeType]:
    """Assign hidden types: exactly ``round(share * M)`` of the M non-focal nodes
    are stable, chosen uniformly at random. F (if given) is marked stable; its
    type does not affect churn."""
    if n_nodes < 1:
        raise ValueError("n_nodes must be >= 1")
    others = [i for i in range(n_nodes) if i != focal]
    n_stable = int(round(stable_share * len(others)))
    perm = rng.permutation(len(others))
    stable = {others[int(j)] for j in perm[:n_stable]}
    if focal is not None:
        stable.add(focal)
    return [NodeType(stable=i in stable) for i in range(n_nodes)]


def weibull_scale(mean_s: float, shape: float) -> float:
    """Weibull scale giving the requested mean: mean / Γ(1 + 1/shape)."""
    return mean_s / math.gamma(1.0 + 1.0 / shape)


def _session_mean(scn: ScenarioConfig, t: NodeType) -> float:
    return scn.stable_session_mean_s if t.stable else scn.flaky_session_mean_s


def analytic_online_fraction(scn: ScenarioConfig) -> float:
    """share·m_s/(m_s+d) + (1−share)·m_f/(m_f+d)."""
    d = scn.downtime_mean_s
    ms, mf = scn.stable_session_mean_s, scn.flaky_session_mean_s
    return scn.stable_share * ms / (ms + d) + (1.0 - scn.stable_share) * mf / (mf + d)


def _to_ms(x_s: float) -> int:
    # Durations are at least 1 ms so sessions and gaps are never empty.
    return max(1, int(round(x_s * 1000.0)))


class ChurnTrace:
    """Per-node sorted, non-overlapping sessions clipped to ``[0, horizon]``."""

    def __init__(self, n_nodes: int, horizon_ms: int, sessions: list[list[Session]]):
        if len(sessions) != n_nodes:
            raise ValueError("sessions must have one list per node")
        self.n_nodes = n_nodes
        self.horizon_ms = horizon_ms
        self._sessions = [tuple(s) for s in sessions]
        self._starts = [[s.start_ms for s in ss] for ss in self._sessions]
        self._ends = [[s.end_ms for s in ss] for ss in self._sessions]

    def sessions(self, node: int) -> tuple[Session, ...]:
        return self._sessions[node]

    def all_sessions(self) -> list[Session]:
        return [s for ss in self._sessions for s in ss]

    def current_session(self, node: int, t_ms: int) -> Session | None:
        i = bisect_right(self._starts[node], t_ms) - 1
        if i >= 0 and t_ms < self._ends[node][i]:
            return self._sessions[node][i]
        return None

    def is_online(self, node: int, t_ms: int) -> bool:
        i = bisect_right(self._starts[node], t_ms) - 1
        return i >= 0 and t_ms < self._ends[node][i]

    def online_at(self, t_ms: int) -> frozenset[int]:
        return frozenset(n for n in range(self.n_nodes) if self.is_online(n, t_ms))

    def events(self) -> Iterator[ChurnEvent]:
        """State transitions strictly inside ``(0, horizon)``, sorted by
        (t_ms, node, kind). The state at t=0 is given by ``online_at(0)``."""
        evs: list[ChurnEvent] = []
        for ss in self._sessions:
            for s in ss:
                if s.start_ms > 0:
                    evs.append(ChurnEvent(s.start_ms, s.node, "up"))
                if s.end_ms < self.horizon_ms:
                    evs.append(ChurnEvent(s.end_ms, s.node, "down"))
        evs.sort()
        return iter(evs)

    def trace_hash(self) -> str:
        h = hashlib.sha256()
        h.update(f"N={self.n_nodes};H={self.horizon_ms}\n".encode("ascii"))
        for ss in self._sessions:
            for s in ss:
                h.update(f"{s.node},{s.start_ms},{s.end_ms}\n".encode("ascii"))
        return h.hexdigest()


def generate_trace(scn: ScenarioConfig, types: list[NodeType], rng: np.random.Generator,
                   horizon_ms: int, focal: int) -> ChurnTrace:
    """Generate the hidden churn trace for one episode.

    Nodes are processed in index order and draws are consumed sequentially, so
    the trace is a pure function of the generator state.
    """
    if horizon_ms <= 0:
        raise ValueError("horizon_ms must be > 0")
    n = len(types)
    if not 0 <= focal < n:
        raise ValueError("focal must be a valid node index")
    shape = scn.weibull_shape
    burn_in_ms = int(scn.burn_in_s) * 1000
    d_mean = scn.downtime_mean_s
    out: list[list[Session]] = []
    for node, t in enumerate(types):
        if node == focal:
            out.append([Session(node, 0, horizon_ms)])
            continue
        m = _session_mean(scn, t)
        scale = weibull_scale(m, shape)
        online = bool(rng.random() < m / (m + d_mean))
        clock = -burn_in_ms
        sessions: list[Session] = []
        while clock < horizon_ms:
            if online:
                dur = _to_ms(float(rng.weibull(shape)) * scale)
                start, end = max(clock, 0), min(clock + dur, horizon_ms)
                if end > start:
                    sessions.append(Session(node, start, end))
            else:
                dur = _to_ms(float(rng.exponential(d_mean)))
            clock += dur
            online = not online
        out.append(sessions)
    return ChurnTrace(n, horizon_ms, out)
