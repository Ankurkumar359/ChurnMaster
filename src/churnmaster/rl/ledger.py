"""Reward ledger: bucket-epoch rewards, τ=1 transitions, and J (Requirement 15).

Observable-only: accepts exactly ``ObservedLookup`` and ``DecisionRecord``
(Requirement 12.5) and imports nothing that carries hidden state.

Records must be fed in env order. Attribution:

* A lookup at ``t`` belongs to bucket-epoch ``(lookup.bucket, t // epoch_ms)``
  (a lookup at exactly ``t_k`` is in epoch k: epochs precede lookups at equal
  timestamps).
* A ``DecisionRecord`` belongs to ``(rec.bucket, rec.epoch_idx)``. Its cost
  ``w_M*pings + w_R*evictions`` is subtracted from that bucket-epoch reward.

    r[b, k] = -sum(lookup_loss over lookups of (b, k)) - sum(cost over records of (b, k))

Epoch records vs cost records. A ``DecisionRecord`` is an *epoch record*
iff ``t_ms == epoch_idx * epoch_ms`` and it is the first record seen for
``(bucket, epoch_idx)``. The env emits exactly one epoch record per bucket per
epoch (a decision, or a null record ``state=None`` for an empty bucket) before
any lookup at ``t_k``. Every other record is a *cost record*; these are the
``kademlia_std`` ping-on-offer costs (``state=None``, emitted at lookup times).
Cost records only add cost; they never open, close, or drop transitions.
A cost record with a non-null state is rejected (ValueError).

Transitions (τ=1). An epoch decision for ``(b, k)`` opens a transition. When
the next epoch record for ``b`` arrives:

* at epoch ``k+1`` with a non-null state ``s'``: the transition
  ``(s, a, r[b, k], s')`` is emitted (its reward is complete, because all
  lookups and cost records of epoch k arrive before ``t_{k+1}``);
* a null record (empty bucket) or a later epoch (gap): the transition is
  dropped (Requirement 15.7).

``finish()`` drops every open transition (episode end). Transition rewards
use all records (warm-up and measured, so learning happens in warm-up);
``total_reward()`` / ``score()`` use measured records only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from churnmaster.rl.reward import action_cost, lookup_loss
from churnmaster.rl.state import allowed_actions
from churnmaster.sim.observation import DecisionRecord, ObservedLookup

__all__ = ["Transition", "Ledger"]


@dataclass(frozen=True)
class Transition:
    """``(s_{b,k}, a_{b,k}, r_{b,k}, s_{b,k+1})``; ``allowed_next = allowed_actions(s_next)``."""

    bucket: int
    epoch_idx: int
    s: int
    a: int
    r: float
    s_next: int
    allowed_next: tuple[int, ...]


class Ledger:
    """Accumulates rewards per bucket-epoch and emits τ=1 transitions."""

    def __init__(self, w_L: float, w_M: float, w_R: float, deadline_ms: float = 2000,
                 epoch_ms: int = 30_000, id_bits: int = 8):
        if epoch_ms <= 0:
            raise ValueError(f"epoch_ms must be > 0, got {epoch_ms}")
        self.w_L, self.w_M, self.w_R = float(w_L), float(w_M), float(w_R)
        self.deadline_ms = deadline_ms
        self.epoch_ms = int(epoch_ms)
        self.id_bits = int(id_bits)
        self._rewards: dict[tuple[int, int], float] = {}     # all records (learning)
        self._epoch_seen: set[tuple[int, int]] = set()
        self._open: dict[int, tuple[int, int, int]] = {}     # bucket -> (k, s, a)
        self._transitions: list[Transition] = []
        self._meas_reward = 0.0
        self._n_meas_lookups = 0

    @classmethod
    def from_config(cls, cfg: Any) -> "Ledger":
        """Build from a ``churnmaster.config.Config``."""
        rw, sim = cfg.reward, cfg.sim
        return cls(rw.w_L, rw.w_M, rw.w_R, sim.deadline_ms, int(sim.epoch_s) * 1000, sim.id_bits)

    # ------------------------------------------------------------------ #

    def add(self, rec: ObservedLookup | DecisionRecord) -> list[Transition]:
        """Ingest one record; return the transitions it closed (usually empty).

        Raises TypeError for any type other than exactly ObservedLookup or
        DecisionRecord, ValueError for an inconsistent decision record.
        """
        t = type(rec)
        if t is ObservedLookup:
            self._add_lookup(rec)
            return []
        if t is DecisionRecord:
            return self._add_decision(rec)
        raise TypeError(f"Ledger accepts only ObservedLookup or DecisionRecord, got {t.__name__}")

    def _check_bucket(self, b: int) -> None:
        if not 0 <= b < self.id_bits:
            raise ValueError(f"bucket {b} out of range [0, {self.id_bits})")

    def _add_lookup(self, lk: ObservedLookup) -> None:
        self._check_bucket(lk.bucket)
        loss = lookup_loss(lk.success, lk.latency_ms, self.w_L, self.deadline_ms)
        key = (lk.bucket, int(lk.t_ms) // self.epoch_ms)
        self._rewards[key] = self._rewards.get(key, 0.0) - loss
        if lk.measured:
            self._meas_reward -= loss
            self._n_meas_lookups += 1

    def _add_decision(self, d: DecisionRecord) -> list[Transition]:
        b, k = d.bucket, d.epoch_idx
        self._check_bucket(b)
        if (d.state is None) != (d.action is None):
            raise ValueError("state and action must both be None or both be set")
        key = (b, k)
        is_epoch = d.t_ms == k * self.epoch_ms and key not in self._epoch_seen
        if d.state is not None:
            if not is_epoch:
                raise ValueError(f"non-null decision record for {key} at t={d.t_ms} "
                                 "is not the epoch record")
            if int(d.action) not in allowed_actions(d.state):
                raise ValueError(f"action {d.action!r} not allowed in state {d.state}")

        cost = action_cost(d.pings, d.evictions, self.w_M, self.w_R)
        self._rewards[key] = self._rewards.get(key, 0.0) - cost
        if d.measured:
            self._meas_reward -= cost
        if not is_epoch:
            return []  # cost record: never opens/closes/drops transitions

        self._epoch_seen.add(key)
        out: list[Transition] = []
        prev = self._open.pop(b, None)
        if prev is not None and d.state is not None and prev[0] == k - 1:
            pk, s, a = prev
            tr = Transition(b, pk, s, a, self._rewards.get((b, pk), 0.0), int(d.state),
                            tuple(int(x) for x in allowed_actions(d.state)))
            out.append(tr)
            self._transitions.append(tr)
        # else: no open transition, empty bucket, or gap -> dropped
        if d.state is not None:
            self._open[b] = (k, int(d.state), int(d.action))
        return out

    def finish(self) -> list[Transition]:
        """Episode end: drop all open transitions. Always returns []."""
        self._open.clear()
        return []

    # ------------------------------------------------------------------ #

    def close_epoch(self, epoch_idx: int) -> list[Transition]:
        """Transitions ending at ``epoch_idx`` (i.e. ``s`` at ``epoch_idx - 1``)
        emitted so far; transitions are emitted by ``add`` as epoch records arrive."""
        return [tr for tr in self._transitions if tr.epoch_idx == epoch_idx - 1]

    @property
    def transitions(self) -> tuple[Transition, ...]:
        return tuple(self._transitions)

    def reward(self, bucket: int, epoch_idx: int) -> float:
        """``r[b, k]`` from all records (0.0 if nothing was attributed)."""
        return self._rewards.get((bucket, epoch_idx), 0.0)

    @property
    def n_measured_lookups(self) -> int:
        return self._n_meas_lookups

    def total_reward(self) -> float:
        """Sum of rewards over measured records (incl. bucket-epochs without a decision)."""
        return self._meas_reward

    def score(self) -> float:
        """``J = 1 + total_reward / N_measured_lookups``; NaN when there are none
        (matching ``metrics.summarize``)."""
        n = self._n_meas_lookups
        if n == 0:
            return math.nan
        return 1.0 + self._meas_reward / n
