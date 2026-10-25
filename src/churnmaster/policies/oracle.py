"""Oracle heuristic: ground-truth reference policy (Requirement 19).

HEADROOM REFERENCE ONLY; NOT AN UPPER BOUND. The oracle sees the true online
state of the incumbent and candidate, but it still acts with the same
action set and decision points as every other epoch-mode policy, and its
simple rule is not optimal (e.g. it never replaces a live-but-slow incumbent).
It estimates available headroom; it does not bound achievable performance.

This is the only module under ``policies/`` allowed ground-truth access, and
only through ``OracleFacts`` delivered by a facts provider. It does not import
``sim.network`` or any other hidden-state module: experiments wrap the env's
``Env.oracle_facts(bucket, t_ms) -> (inc_online, cand_present, cand_online)``
into a ``Callable[[int, int], OracleFacts]`` and pass it here as the provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from churnmaster.actions import Action
from churnmaster.sim.observation import DecisionContext

__all__ = ["OracleFacts", "OracleFactsProvider", "oracle_action", "OracleHeuristic"]


@dataclass(frozen=True)
class OracleFacts:
    """Ground-truth facts for one bucket at one decision time.

    ``cand_online`` is None exactly when there is no candidate
    (``cand_present`` is False).
    """

    inc_online: bool
    cand_present: bool
    cand_online: bool | None

    def __post_init__(self) -> None:
        if not self.cand_present and self.cand_online is not None:
            raise ValueError("cand_online must be None when no candidate is present")
        if self.cand_present and self.cand_online is None:
            raise ValueError("cand_online must be a bool when a candidate is present")


#: Provider signature (bucket, t_ms) -> facts; wraps ``Env.oracle_facts``.
OracleFactsProvider = Callable[[int, int], OracleFacts]


def oracle_action(facts: OracleFacts) -> Action:
    """Requirement 19.1 rule.

    Incumbent offline and an online candidate exists -> REPLACE;
    incumbent offline otherwise -> PROBE; incumbent online -> KEEP.
    """
    if not facts.inc_online:
        if facts.cand_present and facts.cand_online:
            return Action.REPLACE
        return Action.PROBE
    return Action.KEEP


class OracleHeuristic:
    """Ground-truth reference policy: headroom reference only; NOT an upper bound.

    ``facts`` is called as ``facts(ctx.bucket, ctx.t_ms)`` (same signature as
    ``Env.oracle_facts``) and must return the ``OracleFacts`` for
    that bucket at that decision time.
    """

    name = "oracle_heuristic"
    shares_decision_points = True
    is_upper_bound = False
    label = "headroom reference only; NOT an upper bound"

    def __init__(self, facts: OracleFactsProvider) -> None:
        self._facts = facts

    def decide(self, ctx: DecisionContext) -> Action:
        return oracle_action(self._facts(ctx.bucket, ctx.t_ms))
