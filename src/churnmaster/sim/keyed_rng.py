"""Keyed (counter-based) randomness for policy-independent network outcomes.

Every random network outcome (loss, jitter, background rebuild picks, F's
bootstrap) is a pure function of ``(root, purpose, *ids)``. Changing the
policy therefore cannot change the luck seen for a given key tuple.

Key schema (fixed, see design):

=====================  ==================================================
use                    key
=====================  ==================================================
lookup query           (LOSS | JITTER, lookup_id, dst)
epoch ping             (LOSS | JITTER, epoch_idx, bucket, dst, INC | CAND)
kademlia_std ping      (LOSS | JITTER, lookup_id, bucket, dst, STD)
background rebuild     (BG, node, rebuild_idx)
F bootstrap            (BOOT, bucket, slot)
=====================  ==================================================

Pure Python, stdlib only. All arguments must be non-negative ints < 2**64.
"""

from __future__ import annotations

import math
from enum import IntEnum

__all__ = [
    "Purpose",
    "PingKind",
    "splitmix64",
    "keyed_uniform",
    "keyed_normal",
]

MASK64 = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15
_INV_2_53 = 1.0 / (1 << 53)
_TWO_PI = 2.0 * math.pi


class Purpose(IntEnum):
    LOSS = 1
    JITTER = 2
    BG = 3
    BOOT = 4


class PingKind(IntEnum):
    INC = 1
    CAND = 2
    STD = 3


def splitmix64(x: int) -> int:
    """One splitmix64 step: add the golden gamma, then finalize (mix64)."""
    z = (x + _GOLDEN) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def _hash(root: int, purpose: int, ids: tuple[int, ...]) -> int:
    # Seed the chain with root, purpose, and the key length so tuples of
    # different lengths are domain-separated; then absorb each id in order.
    h = splitmix64(int(root) & MASK64)
    h = splitmix64(h ^ (int(purpose) & MASK64))
    h = splitmix64(h ^ len(ids))
    for x in ids:
        h = splitmix64(h ^ (int(x) & MASK64))
    return h


def keyed_uniform(root: int, purpose: int, *ids: int) -> float:
    """Uniform draw in [0, 1) that is a pure function of its arguments.

    Uses the top 53 bits of the hash, so the result is exactly representable
    and strictly less than 1.0.
    """
    return (_hash(root, purpose, ids) >> 11) * _INV_2_53


def keyed_normal(root: int, purpose: int, *ids: int) -> float:
    """Standard normal draw via Box-Muller on sub-keys (*ids, 0) and (*ids, 1).

    ``u1`` lies in [0, 1), so ``1 - u1`` lies in (0, 1] and the logarithm is
    always finite (u1 == 0 maps to radius 0).
    """
    u1 = keyed_uniform(root, purpose, *ids, 0)
    u2 = keyed_uniform(root, purpose, *ids, 1)
    r = math.sqrt(-2.0 * math.log(1.0 - u1))
    return r * math.cos(_TWO_PI * u2)
