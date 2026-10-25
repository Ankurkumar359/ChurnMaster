"""Greedy evaluation policy over a frozen Q-table (Requirement 16.4).

Observable-only: imports stdlib, numpy, ``churnmaster.actions``,
``churnmaster.rl.state``, ``churnmaster.rl.qlearning`` and
``churnmaster.sim.observation`` only.

``Learned`` copies the Q-table (and mask) at construction and marks the copies
read-only, so later training updates to the source agent never change its
decisions. ``decide(ctx)`` returns the greedy allowed action of
``encode(discretize(ctx, id_bits))`` with the same tie order as
:meth:`QAgent.greedy` (KEEP < PROBE < REPLACE: lowest index among the maxima).
The greedy action per state is precomputed once.

Instances hold only numpy arrays, ints and strings, so they pickle cleanly
for multiprocessing jobs (read-only flags are restored on unpickle).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from churnmaster.actions import Action
from churnmaster.rl.qlearning import QAgent
from churnmaster.rl.state import DEFAULT_ID_BITS, N_ACTIONS, N_STATES, allowed_mask, discretize, encode
from churnmaster.sim.observation import DecisionContext

__all__ = ["Learned"]


def _greedy_table(Q: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Same rule as ``QAgent.greedy_table``: masked argmax, first max wins; -1 if none allowed."""
    out = np.argmax(np.where(mask, Q, -np.inf), axis=1).astype(np.int64)
    out[~mask.any(axis=1)] = -1
    return out


def _frozen(a: np.ndarray) -> np.ndarray:
    a.flags.writeable = False
    return a


class Learned:
    """Epoch-mode greedy policy over a frozen copy of a Q-table."""

    shares_decision_points = True

    def __init__(self, source: QAgent | np.ndarray, mask: np.ndarray | None = None,
                 name: str = "learned", id_bits: int = DEFAULT_ID_BITS) -> None:
        if isinstance(source, QAgent):
            Q = source.Q
            if mask is None:
                mask = source.mask
        else:
            Q = source
        Q = np.array(Q, dtype=np.float64, copy=True)
        if Q.shape != (N_STATES, N_ACTIONS):
            raise ValueError(f"Q shape {Q.shape} != {(N_STATES, N_ACTIONS)}")
        if not np.isfinite(Q).all():
            raise ValueError("Q table contains non-finite values")
        mask = allowed_mask() if mask is None else np.array(mask, dtype=bool, copy=True)
        if mask.shape != Q.shape:
            raise ValueError(f"mask shape {mask.shape} != {Q.shape}")
        self.name = str(name)
        self.id_bits = int(id_bits)
        self.Q = _frozen(Q)
        self.mask = _frozen(mask)
        self._greedy = _frozen(_greedy_table(Q, mask))

    @classmethod
    def from_agent(cls, agent: QAgent, name: str = "learned",
                   id_bits: int = DEFAULT_ID_BITS) -> "Learned":
        return cls(agent, name=name, id_bits=id_bits)

    @classmethod
    def load(cls, path: str | Path, name: str = "learned",
             id_bits: int = DEFAULT_ID_BITS) -> "Learned":
        """Load an agent saved by :meth:`QAgent.save` and freeze it."""
        return cls(QAgent.load(path), name=name, id_bits=id_bits)

    @property
    def greedy_table(self) -> np.ndarray:
        """Greedy action per state, shape ``(N_STATES,)`` int64 (read-only)."""
        return self._greedy

    def state(self, ctx: DecisionContext) -> int:
        return encode(discretize(ctx, self.id_bits))

    def decide(self, ctx: DecisionContext) -> Action:
        s = self.state(ctx)
        a = int(self._greedy[s])
        if a < 0:
            raise ValueError(f"state {s} has no allowed actions")
        return Action(a)

    # Pickle support: arrays come back writable, so re-freeze them.
    def __getstate__(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        for k in ("Q", "mask", "_greedy"):
            arr = np.array(self.__dict__[k], copy=True)
            self.__dict__[k] = _frozen(arr)

    def __repr__(self) -> str:
        return f"Learned(name={self.name!r}, id_bits={self.id_bits})"
