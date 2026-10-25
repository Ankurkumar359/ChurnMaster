"""State discretization and action masking (Requirement 13).

Observable-only: imports stdlib, numpy, ``churnmaster.actions`` and
``churnmaster.sim.observation`` only.

Features and bins (all thresholds are module constants):

* ``tier``: bucket ``id_bits-1`` -> TOP (0), ``id_bits-2`` -> SECOND (1), else OTHER (2)
* ``inc_fail``: consecutive failures 0 / 1 / >=2
* ``inc_stale``: ``inc_since_response_s`` < 60 -> 0, else 1
* ``inc_age``: < 120 s YOUNG (0), [120, 1200) MID (1), >= 1200 OLD (2)
* ``cand``: NONE (0) if no candidate, FRESH (1) if seen < 30 s ago, else STALE (2)
  (the cache TTL guarantees STALE candidates are at most 300 s old)
* ``cand_faster``: 1 iff a candidate exists, the incumbent SRTT is known, and
  ``cand_last_rtt_ms < 0.7 * inc_srtt_ms`` (strict)
* ``churn``: ``timeout_fraction`` < 0.10 -> 0, else 1

Boundary semantics: a value exactly on an edge falls in the higher bin
(``x >= edge``), e.g. ``inc_age_s == 120`` is MID.

State index: mixed radix over ``RADICES = (3, 3, 2, 3, 3, 2, 2)`` in field
order, first field most significant -> ``[0, 648)``. States with
``cand == NONE and cand_faster == 1`` are unreachable, leaving 540.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from churnmaster.actions import Action
from churnmaster.sim.observation import DecisionContext

__all__ = [
    "N_STATES", "N_REACHABLE", "N_ACTIONS", "RADICES", "DEFAULT_ID_BITS",
    "TIER_TOP", "TIER_SECOND", "TIER_OTHER",
    "INC_STALE_EDGE_S", "INC_AGE_MID_EDGE_S", "INC_AGE_OLD_EDGE_S",
    "AGE_YOUNG", "AGE_MID", "AGE_OLD",
    "CAND_NONE", "CAND_FRESH", "CAND_STALE", "CAND_FRESH_EDGE_S",
    "CAND_FASTER_RATIO", "CHURN_EDGE",
    "Features", "discretize", "encode", "decode",
    "allowed_actions", "allowed_mask", "reachable_states", "is_reachable",
]

RADICES: tuple[int, ...] = (3, 3, 2, 3, 3, 2, 2)
N_STATES = 648
N_REACHABLE = 540
N_ACTIONS = len(Action)
DEFAULT_ID_BITS = 8

TIER_TOP, TIER_SECOND, TIER_OTHER = 0, 1, 2
INC_STALE_EDGE_S = 60.0
INC_AGE_MID_EDGE_S = 120.0
INC_AGE_OLD_EDGE_S = 1200.0
AGE_YOUNG, AGE_MID, AGE_OLD = 0, 1, 2
CAND_NONE, CAND_FRESH, CAND_STALE = 0, 1, 2
CAND_FRESH_EDGE_S = 30.0
CAND_FASTER_RATIO = 0.7
CHURN_EDGE = 0.10


@dataclass(frozen=True)
class Features:
    tier: int
    inc_fail: int
    inc_stale: int
    inc_age: int
    cand: int
    cand_faster: int
    churn: int


def discretize(ctx: DecisionContext, id_bits: int = DEFAULT_ID_BITS) -> Features:
    """Map an observable DecisionContext to its binned Features."""
    if ctx.bucket == id_bits - 1:
        tier = TIER_TOP
    elif ctx.bucket == id_bits - 2:
        tier = TIER_SECOND
    else:
        tier = TIER_OTHER

    inc_fail = min(max(int(ctx.inc_consec_fail), 0), 2)
    inc_stale = 0 if ctx.inc_since_response_s < INC_STALE_EDGE_S else 1

    if ctx.inc_age_s < INC_AGE_MID_EDGE_S:
        inc_age = AGE_YOUNG
    elif ctx.inc_age_s < INC_AGE_OLD_EDGE_S:
        inc_age = AGE_MID
    else:
        inc_age = AGE_OLD

    if ctx.cand_since_seen_s is None:
        cand = CAND_NONE
    elif ctx.cand_since_seen_s < CAND_FRESH_EDGE_S:
        cand = CAND_FRESH
    else:
        cand = CAND_STALE

    cand_faster = int(
        cand != CAND_NONE
        and ctx.inc_srtt_ms is not None
        and ctx.cand_last_rtt_ms is not None
        and ctx.cand_last_rtt_ms < CAND_FASTER_RATIO * ctx.inc_srtt_ms
    )
    churn = 0 if ctx.timeout_fraction < CHURN_EDGE else 1
    return Features(tier, inc_fail, inc_stale, inc_age, cand, cand_faster, churn)


def encode(f: Features) -> int:
    """Mixed-radix encoding of ``f`` into ``[0, N_STATES)``; ValueError if out of range."""
    s = 0
    for fld, radix in zip(fields(Features), RADICES):
        v = getattr(f, fld.name)
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or not 0 <= v < radix:
            raise ValueError(f"feature {fld.name}={v!r} not in [0, {radix})")
        s = s * radix + int(v)
    return s


def decode(s: int) -> Features:
    """Inverse of :func:`encode`; ValueError if ``s`` is not in ``[0, N_STATES)``."""
    if isinstance(s, bool) or not isinstance(s, (int, np.integer)) or not 0 <= s < N_STATES:
        raise ValueError(f"state {s!r} not in [0, {N_STATES})")
    s = int(s)
    digits = []
    for radix in reversed(RADICES):
        s, d = divmod(s, radix)
        digits.append(d)
    return Features(*reversed(digits))


def is_reachable(s: int) -> bool:
    f = decode(s)
    return not (f.cand == CAND_NONE and f.cand_faster == 1)


def reachable_states() -> list[int]:
    """All state indices that :func:`discretize` can produce (540)."""
    return [s for s in range(N_STATES) if is_reachable(s)]


def allowed_actions(s: int) -> tuple[Action, ...]:
    """KEEP and PROBE always; REPLACE only when a candidate exists."""
    if decode(s).cand == CAND_NONE:
        return (Action.KEEP, Action.PROBE)
    return (Action.KEEP, Action.PROBE, Action.REPLACE)


def allowed_mask() -> np.ndarray:
    """Boolean array of shape ``(N_STATES, N_ACTIONS)``; True where the action is allowed."""
    mask = np.zeros((N_STATES, N_ACTIONS), dtype=bool)
    for s in range(N_STATES):
        mask[s, [int(a) for a in allowed_actions(s)]] = True
    return mask


# Sanity: constants agree with the radix layout.
assert int(np.prod(RADICES)) == N_STATES and len(RADICES) == len(fields(Features))
