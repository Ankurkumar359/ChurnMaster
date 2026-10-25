"""Policy protocol and shared helpers (Requirement 17).

Observable-only: imports stdlib, ``churnmaster.actions``,
``churnmaster.sim.observation`` and ``churnmaster.rl.state`` only.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from churnmaster.actions import Action
from churnmaster.rl.state import allowed_actions, discretize, encode
from churnmaster.sim.observation import DecisionContext

__all__ = ["Policy", "has_candidate", "allowed"]


@runtime_checkable
class Policy(Protocol):
    """A decision function from an observable DecisionContext to an Action.

    ``shares_decision_points`` is True for every epoch-mode policy; only the
    labeled ``kademlia_std`` mode sets it to False.
    """

    name: str
    shares_decision_points: bool

    def decide(self, ctx: DecisionContext) -> Action: ...


def has_candidate(ctx: DecisionContext) -> bool:
    """A candidate is present iff ``cand_since_seen_s`` is not None."""
    return ctx.cand_since_seen_s is not None


def allowed(ctx: DecisionContext) -> tuple[Action, ...]:
    """Allowed actions for ``ctx``, via the same masking as ``rl.state``.

    Masking depends only on candidate presence, so ``id_bits`` does not matter.
    """
    return allowed_actions(encode(discretize(ctx)))
