"""Per-lookup loss and per-decision action cost (design "Reward and score").

Single source of truth for the terms of the reward and of J, shared by the
Ledger (task 7.1) and ``metrics.py``. Pure functions; standard library only.
"""

from __future__ import annotations

__all__ = ["lookup_loss", "action_cost"]


def lookup_loss(success: bool, latency_ms: float, w_L: float,
                deadline_ms: float = 2000) -> float:
    """``(1 - S) + w_L * min(L / deadline, 1)``, in ``[0, 1 + w_L]``.

    Negative latencies are clamped to 0 so the range holds for any input.
    """
    frac = min(max(float(latency_ms), 0.0) / float(deadline_ms), 1.0)
    return (0.0 if success else 1.0) + w_L * frac


def action_cost(pings: int, evictions: int, w_M: float, w_R: float) -> float:
    """``w_M * pings + w_R * evictions``."""
    return w_M * pings + w_R * evictions
