"""Heuristic family: tuned rule-based competitor in the RL state space (Requirement 18).

Observable-only: imports stdlib, numpy, ``churnmaster.actions``,
``churnmaster.rl.state`` and ``churnmaster.sim.observation`` only.

The rule function is defined on the discretized ``Features`` (not on the raw
context), so every configuration is exactly representable as a 648-entry
state -> action table (540 reachable states); see :func:`as_table`.

Rules, applied in order (first match wins):

1. ``inc_fail >= f``: REPLACE if a candidate exists else PROBE; with
   ``protect_old`` an OLD incumbent (>= 20 min) with fewer than 2 failures
   gets PROBE instead.
2. ``rtt_swap`` (gated): REPLACE if the candidate is faster, FRESH, and the
   incumbent is YOUNG (< 2 min).
3. ``probe_stale`` (gated): PROBE if the incumbent is stale (>= 60 s since
   last response) and a candidate exists.
4. KEEP.

"Gated" means: with ``tier_gate == "top_two"`` rules 2 and 3 apply only to
TOP and SECOND tier buckets; rule 1 is never gated.

Configuration order of :func:`all_configs` is lexicographic (itertools.product)
over ``f in (1, 2)``, ``protect_old in (False, True)``,
``rtt_swap in (False, True)``, ``probe_stale in (False, True)``,
``tier_gate in ("all", "top_two")`` -- last field varies fastest.
Each config has a stable id, e.g. ``"f=2,po=1,rs=0,ps=1,tg=all"``, that
round-trips through :meth:`HeuristicConfig.from_id`.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from churnmaster.actions import Action
from churnmaster.rl.state import (
    AGE_OLD,
    AGE_YOUNG,
    CAND_FRESH,
    CAND_NONE,
    DEFAULT_ID_BITS,
    N_STATES,
    TIER_SECOND,
    TIER_TOP,
    Features,
    decode,
    discretize,
)
from churnmaster.sim.observation import DecisionContext

__all__ = [
    "F_VALUES", "TIER_GATES", "HeuristicConfig", "DEFAULT_CONFIG", "all_configs",
    "heuristic_action", "as_table", "Heuristic",
]

F_VALUES: tuple[int, ...] = (1, 2)
TIER_GATES: tuple[str, ...] = ("all", "top_two")


@dataclass(frozen=True)
class HeuristicConfig:
    f: int
    protect_old: bool
    rtt_swap: bool
    probe_stale: bool
    tier_gate: str

    def __post_init__(self) -> None:
        if isinstance(self.f, bool) or self.f not in F_VALUES:
            raise ValueError(f"f must be one of {F_VALUES}, got {self.f!r}")
        for name in ("protect_old", "rtt_swap", "probe_stale"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool, got {getattr(self, name)!r}")
        if self.tier_gate not in TIER_GATES:
            raise ValueError(f"tier_gate must be one of {TIER_GATES}, got {self.tier_gate!r}")

    @property
    def id(self) -> str:
        """Stable string id, e.g. ``"f=2,po=1,rs=0,ps=1,tg=all"``."""
        return (f"f={self.f},po={int(self.protect_old)},rs={int(self.rtt_swap)},"
                f"ps={int(self.probe_stale)},tg={self.tier_gate}")

    @classmethod
    def from_id(cls, s: str) -> "HeuristicConfig":
        """Parse an id produced by :attr:`id`; ValueError on malformed input."""
        try:
            parts = dict(p.split("=", 1) for p in s.split(","))
        except ValueError:
            raise ValueError(f"malformed heuristic config id {s!r}") from None
        if set(parts) != {"f", "po", "rs", "ps", "tg"} or len(s.split(",")) != 5:
            raise ValueError(f"malformed heuristic config id {s!r}")

        def flag(k: str) -> bool:
            if parts[k] not in ("0", "1"):
                raise ValueError(f"malformed heuristic config id {s!r}")
            return parts[k] == "1"

        if parts["f"] not in ("1", "2"):
            raise ValueError(f"malformed heuristic config id {s!r}")
        cfg = cls(int(parts["f"]), flag("po"), flag("rs"), flag("ps"), parts["tg"])
        return cfg


#: Plan's Task 6 default heuristic.
DEFAULT_CONFIG = HeuristicConfig(f=2, protect_old=True, rtt_swap=False, probe_stale=True,
                                 tier_gate="all")


def all_configs() -> list[HeuristicConfig]:
    """The 32 configurations in the fixed order documented in the module docstring."""
    return [
        HeuristicConfig(f, po, rs, ps, tg)
        for f, po, rs, ps, tg in itertools.product(
            F_VALUES, (False, True), (False, True), (False, True), TIER_GATES)
    ]


def heuristic_action(cfg: HeuristicConfig, f: Features) -> Action:
    """Requirement 18.1/18.2 rule function on discretized features."""
    has_c = f.cand != CAND_NONE
    if f.inc_fail >= cfg.f:                                                    # rule 1
        if cfg.protect_old and f.inc_age == AGE_OLD and f.inc_fail < 2:
            return Action.PROBE
        return Action.REPLACE if has_c else Action.PROBE
    gated = cfg.tier_gate == "all" or f.tier in (TIER_TOP, TIER_SECOND)
    if (gated and cfg.rtt_swap and f.cand_faster and f.cand == CAND_FRESH
            and f.inc_age == AGE_YOUNG):
        return Action.REPLACE                                                  # rule 2
    if gated and cfg.probe_stale and f.inc_stale == 1 and has_c:
        return Action.PROBE                                                    # rule 3
    return Action.KEEP                                                         # rule 4


def as_table(cfg: HeuristicConfig) -> np.ndarray:
    """Action for every state index, shape ``(N_STATES,)`` int8.

    Unreachable states are filled too (computed from :func:`decode`).
    """
    return np.array([int(heuristic_action(cfg, decode(s))) for s in range(N_STATES)],
                    dtype=np.int8)


class Heuristic:
    """Epoch-mode policy applying :func:`heuristic_action` to ``discretize(ctx)``."""

    shares_decision_points = True

    def __init__(self, cfg: HeuristicConfig = DEFAULT_CONFIG,
                 id_bits: int = DEFAULT_ID_BITS) -> None:
        self.cfg = cfg
        self.id_bits = id_bits
        self.name = f"heuristic[{cfg.id}]"

    def decide(self, ctx: DecisionContext) -> Action:
        return heuristic_action(self.cfg, discretize(ctx, self.id_bits))
