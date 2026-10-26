"""Greedy policy over a frozen Q-table with a custom state encoder (Task 12.1 ablations).

Observable-only: imports stdlib, numpy, ``churnmaster.actions``,
``churnmaster.policies.learned``, ``churnmaster.rl`` and
``churnmaster.sim.observation`` only.

:class:`EncodedLearned` is :class:`~churnmaster.policies.learned.Learned`
with an optional picklable encoder ``encode(ctx, id_bits) -> state`` (e.g.
:class:`churnmaster.rl.variants.FeatureDrop`). It exposes ``encode_state(ctx)``,
which :func:`churnmaster.experiments.runner.run_job` uses to fill
``DecisionRecord.state``, so the recorded states, the ledger transitions and
the greedy lookups all use the same (wrapped) state space as in training.
With ``encode=None`` it behaves exactly like ``Learned``. Masks (e.g. the
no-PROBE mask) come from the source agent as in ``Learned``.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from churnmaster.policies.learned import Learned
from churnmaster.rl.qlearning import QAgent
from churnmaster.rl.state import DEFAULT_ID_BITS
from churnmaster.sim.observation import DecisionContext

__all__ = ["EncodedLearned"]


class EncodedLearned(Learned):
    """``Learned`` with an optional state encoder wrapper (picklable if the encoder is)."""

    def __init__(self, source: QAgent | np.ndarray, mask: np.ndarray | None = None,
                 name: str = "learned", id_bits: int = DEFAULT_ID_BITS,
                 encode: Callable[[DecisionContext, int], int] | None = None) -> None:
        super().__init__(source, mask=mask, name=name, id_bits=id_bits)
        self.encode = encode

    def state(self, ctx: DecisionContext) -> int:
        if self.encode is None:
            return super().state(ctx)
        return int(self.encode(ctx, self.id_bits))

    def encode_state(self, ctx: DecisionContext) -> int:
        return self.state(ctx)

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)

    def __repr__(self) -> str:
        return f"EncodedLearned(name={self.name!r}, id_bits={self.id_bits}, encode={self.encode!r})"
