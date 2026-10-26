"""Observable-only state / action variants for agent ablations (Task 12.1, Requirement 28.1).

Observable-only: imports stdlib, numpy, ``churnmaster.actions``,
``churnmaster.rl.state`` and ``churnmaster.sim.observation`` only.

Nothing here changes :mod:`churnmaster.rl.state`; the variants wrap it.

* :class:`FeatureDrop` -- a discretizer wrapper that collapses ONE feature of
  :class:`~churnmaster.rl.state.Features` before encoding, so the agent cannot
  distinguish states that differ only in that feature. The table keeps its
  shape (648 x 3); only the collapsed sub-space is ever visited.

  - every feature except ``cand`` is set to the constant bin 0;
  - ``cand`` is special: collapsing it to NONE (0) would mask REPLACE in every
    state. Instead FRESH and STALE are merged into one "present" bin
    (``CAND_FRESH``); NONE stays NONE, so candidate presence and therefore
    the action mask stay exactly correct (variant ``drop_cand_freshness``).
  - ``cand_faster`` is never set to 1 by the wrapper, so ``cand == NONE``
    still implies ``cand_faster == 0`` (collapsed states stay reachable).

  Instances are frozen dataclasses (picklable) and callable as
  ``drop(ctx, id_bits) -> state index``.

* :func:`no_probe_mask` -- the default action mask with PROBE removed
  everywhere (allowed: KEEP, and REPLACE iff a candidate exists).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from churnmaster.actions import Action
from churnmaster.rl.state import (CAND_FRESH, CAND_NONE, DEFAULT_ID_BITS, Features, allowed_mask,
                                  decode, discretize, encode)
from churnmaster.sim.observation import DecisionContext

__all__ = ["FEATURE_NAMES", "FeatureDrop", "no_probe_mask"]

FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in dataclasses.fields(Features))


@dataclass(frozen=True)
class FeatureDrop:
    """Collapse ``feature`` to a single bin (``cand``: merge FRESH/STALE, keep NONE)."""

    feature: str

    def __post_init__(self) -> None:
        if self.feature not in FEATURE_NAMES:
            raise ValueError(f"unknown feature {self.feature!r}; expected one of {FEATURE_NAMES}")

    def apply(self, f: Features) -> Features:
        if self.feature == "cand":
            return dataclasses.replace(f, cand=CAND_NONE if f.cand == CAND_NONE else CAND_FRESH)
        return dataclasses.replace(f, **{self.feature: 0})

    def apply_state(self, s: int) -> int:
        """Collapse an already encoded state index."""
        return encode(self.apply(decode(s)))

    def __call__(self, ctx: DecisionContext, id_bits: int = DEFAULT_ID_BITS) -> int:
        return encode(self.apply(discretize(ctx, id_bits)))


def no_probe_mask() -> np.ndarray:
    """Default mask with the PROBE column set to False (shape ``(N_STATES, N_ACTIONS)``)."""
    m = allowed_mask()
    m[:, int(Action.PROBE)] = False
    return m
