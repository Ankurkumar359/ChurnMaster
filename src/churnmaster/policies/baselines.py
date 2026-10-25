"""Baseline policies (Requirement 17).

Observable-only. Random baselines draw only from their injected
``np.random.Generator`` (built by experiments from
``seeding.baseline_rng``), separate from all environment streams (17.5).
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

from churnmaster.actions import Action
from churnmaster.policies.base import allowed, has_candidate
from churnmaster.rl.state import CAND_NONE, decode
from churnmaster.sim.observation import DecisionContext, DecisionRecord

__all__ = [
    "NeverReplace", "AlwaysReplace", "UniformRandom", "MatchedRandom",
    "KademliaEpoch", "KademliaStd", "BASELINE_NAMES", "RANDOM_BASELINES",
    "make_baseline", "action_counts", "action_frequencies", "FALLBACK_FREQS",
]

#: Fixed action order used for sampling in MatchedRandom.
_ORDER: tuple[Action, ...] = (Action.KEEP, Action.PROBE, Action.REPLACE)
_FREQ_TOL = 1e-9


class NeverReplace:
    name = "never_replace"
    shares_decision_points = True

    def decide(self, ctx: DecisionContext) -> Action:
        return Action.KEEP


class AlwaysReplace:
    name = "always_replace"
    shares_decision_points = True

    def decide(self, ctx: DecisionContext) -> Action:
        return Action.REPLACE if has_candidate(ctx) else Action.KEEP


class KademliaEpoch:
    name = "kademlia_epoch"
    shares_decision_points = True

    def decide(self, ctx: DecisionContext) -> Action:
        return Action.PROBE if has_candidate(ctx) else Action.KEEP


class KademliaStd:
    """Standard Kademlia ping-on-offer; epoch decisions are always KEEP.

    Results are labeled as not sharing decision points (17.4); the env runs
    in ``mode=env_mode`` (wired in task 5.7).
    """

    name = "kademlia_std"
    shares_decision_points = False
    env_mode = "kademlia_std"

    def decide(self, ctx: DecisionContext) -> Action:
        return Action.KEEP


class UniformRandom:
    """Uniform over the allowed actions of the context."""

    name = "uniform_random"
    shares_decision_points = True

    def __init__(self, rng: np.random.Generator) -> None:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy Generator")
        self._rng = rng

    def decide(self, ctx: DecisionContext) -> Action:
        acts = allowed(ctx)
        return acts[int(self._rng.integers(len(acts)))]


def _validate_freqs(freqs: Mapping[bool, Sequence[float]]) -> dict[bool, np.ndarray]:
    out: dict[bool, np.ndarray] = {}
    for key in (False, True):
        if key not in freqs:
            raise ValueError(f"freqs missing entry for has_candidate={key}")
        p = np.asarray(freqs[key], dtype=float)
        if p.shape != (len(_ORDER),):
            raise ValueError(f"freqs[{key}] must have {len(_ORDER)} entries (KEEP, PROBE, REPLACE)")
        if not np.all(np.isfinite(p)) or np.any(p < 0):
            raise ValueError(f"freqs[{key}] must be finite and non-negative")
        if abs(p.sum() - 1.0) > _FREQ_TOL:
            raise ValueError(f"freqs[{key}] must sum to 1 (got {p.sum()!r})")
        if not key and p[int(Action.REPLACE)] > 0:
            raise ValueError("REPLACE frequency must be 0 when no candidate is present")
        out[key] = p / p.sum()
    return out


class MatchedRandom:
    """Samples from supplied action frequencies split by candidate presence.

    ``freqs[has_cand]`` gives probabilities over (KEEP, PROBE, REPLACE).
    Frequencies come from :func:`action_frequencies` over the RL policy's
    validation decisions per scenario (``experiments/matched.py``).
    """

    name = "matched_random"
    shares_decision_points = True

    def __init__(self, freqs: Mapping[bool, Sequence[float]], rng: np.random.Generator) -> None:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy Generator")
        self._p = _validate_freqs(freqs)
        self._rng = rng

    @property
    def freqs(self) -> dict[bool, np.ndarray]:
        return {k: v.copy() for k, v in self._p.items()}

    def decide(self, ctx: DecisionContext) -> Action:
        i = int(self._rng.choice(len(_ORDER), p=self._p[has_candidate(ctx)]))
        return _ORDER[i]


# --------------------------------------------------------------------------- #
# Frequency extraction for matched_random (Requirement 17.2, task 9.2)
# --------------------------------------------------------------------------- #

#: Fallback when a candidate-presence group has no decisions: uniform over
#: the allowed actions of that group (no candidate: KEEP/PROBE; candidate:
#: KEEP/PROBE/REPLACE), i.e. matched_random degrades to uniform_random there.
FALLBACK_FREQS: dict[bool, tuple[float, float, float]] = {
    False: (0.5, 0.5, 0.0),
    True: (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
}


def action_counts(decisions: Iterable[DecisionRecord]) -> dict[bool, tuple[int, int, int]]:
    """Counts of (KEEP, PROBE, REPLACE) per candidate presence.

    Candidate presence is read from the encoded state
    (``decode(d.state).cand != CAND_NONE``). Null records (empty buckets and
    ``kademlia_std`` ping-cost records, ``state``/``action`` None) are skipped.
    Raises ValueError on an unknown action or on REPLACE without a candidate
    (impossible for a masked policy, so it signals corrupted input).
    """
    c = {False: [0, 0, 0], True: [0, 0, 0]}
    for d in decisions:
        if d.state is None or d.action is None:
            continue
        a = int(d.action)
        if not 0 <= a < len(_ORDER):
            raise ValueError(f"unknown action {d.action!r} in decision record")
        cand = decode(int(d.state)).cand != CAND_NONE
        if not cand and a == int(Action.REPLACE):
            raise ValueError("decision record has REPLACE without a candidate")
        c[cand][a] += 1
    return {k: (v[0], v[1], v[2]) for k, v in c.items()}


def _normalize(counts: Sequence[int], key: bool) -> tuple[float, float, float]:
    total = int(sum(counts))
    if total == 0:
        return FALLBACK_FREQS[key]
    p = [n / total for n in counts]
    # Put rounding residue on the largest entry so the sum is 1 to float precision.
    i = max(range(len(p)), key=lambda j: p[j])
    p[i] = 1.0 - sum(q for j, q in enumerate(p) if j != i)
    return (p[0], p[1], p[2])


def action_frequencies(
    decisions: Iterable[DecisionRecord] | Mapping[bool, Sequence[int]],
) -> dict[bool, tuple[float, float, float]]:
    """Normalized :func:`action_counts` per candidate presence (matched_random input).

    Also accepts a counts mapping as returned by :func:`action_counts`. An
    empty group falls back to :data:`FALLBACK_FREQS`. The no-candidate
    REPLACE frequency is exactly 0.0; the result passes MatchedRandom
    validation.
    """
    counts = decisions if isinstance(decisions, Mapping) else action_counts(decisions)
    out = {k: _normalize(counts[k], k) for k in (False, True)}
    assert out[False][int(Action.REPLACE)] == 0.0
    _validate_freqs(out)
    return out


BASELINE_NAMES: tuple[str, ...] = (
    "never_replace", "always_replace", "uniform_random",
    "matched_random", "kademlia_epoch", "kademlia_std",
)
RANDOM_BASELINES = frozenset({"uniform_random", "matched_random"})


def make_baseline(name: str, rng: np.random.Generator | None = None,
                  freqs: Mapping[bool, Sequence[float]] | None = None):
    """Construct a baseline by name; random baselines require ``rng``."""
    if name not in BASELINE_NAMES:
        raise ValueError(f"unknown baseline {name!r}; expected one of {BASELINE_NAMES}")
    if name in RANDOM_BASELINES and rng is None:
        raise ValueError(f"baseline {name!r} requires an rng")
    if name == "never_replace":
        return NeverReplace()
    if name == "always_replace":
        return AlwaysReplace()
    if name == "kademlia_epoch":
        return KademliaEpoch()
    if name == "kademlia_std":
        return KademliaStd()
    if name == "uniform_random":
        return UniformRandom(rng)
    if freqs is None:
        raise ValueError("baseline 'matched_random' requires freqs")
    return MatchedRandom(freqs, rng)
