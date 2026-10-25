"""Seed derivation and data splits (Requirement 2).

Every episode seed is ``SeedSequence(root, spawn_key=(split, run, episode))``.
Distinct (split, run, episode) triples give distinct SeedSequence states, so
calibration, train, validation, eval, agent, and test streams never overlap.

Eval environments always use ``run=0`` so every policy and every RL run sees
identical eval episodes. Agent exploration and random baselines draw from the
AGENT split, which is disjoint from every environment stream.

This module depends only on numpy and the standard library.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

__all__ = [
    "Split",
    "EpisodeStreams",
    "episode_seed",
    "make_streams",
    "env_seed",
    "eval_seed",
    "agent_rng",
    "baseline_rng",
    "seed_state",
    "BASELINE_EPISODE_OFFSET",
]

# Random baselines use episode index 1000 + policy_id within the AGENT split,
# keeping them apart from agent exploration (episode 0).
BASELINE_EPISODE_OFFSET = 1000

_MASK64 = (1 << 64) - 1


class Split(IntEnum):
    CALIBRATION = 0
    TRAIN = 1
    VALIDATION = 2
    EVAL = 3
    AGENT = 4
    TEST = 5


def _check_non_negative(**kwargs: int) -> None:
    for name, value in kwargs.items():
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative int, got {value!r}")


def episode_seed(root: int, split: Split, run: int, episode: int) -> np.random.SeedSequence:
    """Seed for one (split, run, episode) triple under a root seed."""
    _check_non_negative(root=root, run=run, episode=episode)
    return np.random.SeedSequence(int(root), spawn_key=(int(Split(split)), int(run), int(episode)))


def seed_state(ss: np.random.SeedSequence, words: int = 4) -> tuple[int, ...]:
    """Hashable fingerprint of a SeedSequence's generated state."""
    return tuple(int(x) for x in ss.generate_state(words, dtype=np.uint32))


@dataclass(frozen=True)
class EpisodeStreams:
    """Independent environment streams for one episode."""

    ids: np.random.Generator
    types_positions: np.random.Generator
    churn: np.random.Generator
    congestion: np.random.Generator
    workload: np.random.Generator
    keyed_root: int  # 64-bit root for keyed_uniform / keyed_normal


def make_streams(ss: np.random.SeedSequence) -> EpisodeStreams:
    """Spawn the per-episode environment streams from an episode seed.

    Child order is fixed; adding a stream must append, never reorder.
    """
    children = ss.spawn(6)
    keyed_words = children[5].generate_state(2, dtype=np.uint32)
    keyed_root = ((int(keyed_words[0]) << 32) | int(keyed_words[1])) & _MASK64
    return EpisodeStreams(
        ids=np.random.default_rng(children[0]),
        types_positions=np.random.default_rng(children[1]),
        churn=np.random.default_rng(children[2]),
        congestion=np.random.default_rng(children[3]),
        workload=np.random.default_rng(children[4]),
        keyed_root=keyed_root,
    )


_ENV_SPLITS = frozenset({Split.CALIBRATION, Split.TRAIN, Split.VALIDATION, Split.EVAL, Split.TEST})


def env_seed(root: int, split: Split, run: int, episode: int) -> np.random.SeedSequence:
    """Environment seed for a split. EVAL ignores ``run`` (always run=0)."""
    split = Split(split)
    if split not in _ENV_SPLITS:
        raise ValueError(f"{split.name} is not an environment split")
    if split == Split.EVAL:
        return eval_seed(root, episode)
    return episode_seed(root, split, run, episode)


def eval_seed(root: int, episode: int) -> np.random.SeedSequence:
    """Eval environment seed; independent of the RL run index (Requirement 2.4)."""
    return episode_seed(root, Split.EVAL, 0, episode)


def agent_rng(root: int, run: int) -> np.random.Generator:
    """Exploration generator for RL run ``run`` (Requirement 2.5)."""
    return np.random.default_rng(episode_seed(root, Split.AGENT, run, 0))


def baseline_rng(root: int, run: int, policy_id: int) -> np.random.Generator:
    """Generator for a random baseline (uniform_random, matched_random)."""
    _check_non_negative(policy_id=policy_id)
    return np.random.default_rng(
        episode_seed(root, Split.AGENT, run, BASELINE_EPISODE_OFFSET + int(policy_id))
    )
