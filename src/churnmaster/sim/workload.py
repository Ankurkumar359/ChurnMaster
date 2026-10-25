"""Pre-generated lookup workload (Requirement 6).

Lookups arrive as a homogeneous Poisson process. The whole workload for an
episode is generated up front from a dedicated ``numpy.random.Generator`` so
every policy faces the identical request sequence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["LookupRequest", "generate_workload"]


@dataclass(frozen=True)
class LookupRequest:
    lookup_id: int
    t_ms: int
    key: int


def generate_workload(
    rng: np.random.Generator,
    rate_hz: float,
    start_ms: int,
    end_ms: int,
    bits: int,
) -> list[LookupRequest]:
    """Generate Poisson lookup arrivals in ``[start_ms, end_ms)``.

    A homogeneous Poisson process on a window is sampled exactly by drawing
    the count ``N ~ Poisson(rate * duration)`` and then ``N`` i.i.d. uniform
    arrival times in the window. Continuous times are floored to integer ms,
    so several lookups may share a millisecond; ties are ordered by
    ``lookup_id``. Keys are uniform in ``[0, 2**bits)``. ``lookup_id`` runs
    0..N-1 in timestamp order. Deterministic given ``rng``'s state.
    """
    if rate_hz < 0:
        raise ValueError(f"rate_hz must be >= 0, got {rate_hz}")
    if end_ms < start_ms:
        raise ValueError(f"end_ms ({end_ms}) < start_ms ({start_ms})")
    if bits <= 0:
        raise ValueError(f"bits must be > 0, got {bits}")

    duration_ms = end_ms - start_ms
    n = int(rng.poisson(rate_hz * duration_ms / 1000.0)) if duration_ms > 0 else 0
    if n == 0:
        return []

    # Uniform in [start_ms, end_ms); floor keeps timestamps strictly < end_ms.
    times = np.sort(rng.uniform(float(start_ms), float(end_ms), size=n))
    t_ms = np.floor(times).astype(np.int64)
    # Guard against float rounding landing exactly on end_ms.
    np.clip(t_ms, start_ms, end_ms - 1, out=t_ms)
    keys = rng.integers(0, 1 << bits, size=n, dtype=np.int64)

    return [
        LookupRequest(lookup_id=i, t_ms=int(t), key=int(k))
        for i, (t, k) in enumerate(zip(t_ms, keys))
    ]
