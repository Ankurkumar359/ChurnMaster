"""Unit tests for churnmaster.seeding (Requirements 2.1-2.5)."""

from __future__ import annotations

import numpy as np
import pytest

from churnmaster.seeding import (
    BASELINE_EPISODE_OFFSET,
    EpisodeStreams,
    Split,
    agent_rng,
    baseline_rng,
    env_seed,
    episode_seed,
    eval_seed,
    make_streams,
    seed_state,
)

ROOT = 20240601


def test_split_ids_distinct():
    assert len({int(s) for s in Split}) == len(Split) == 6
    assert {s.name for s in Split} == {
        "CALIBRATION", "TRAIN", "VALIDATION", "EVAL", "AGENT", "TEST"
    }


def test_episode_seed_uses_spawn_key():
    ss = episode_seed(ROOT, Split.TRAIN, 2, 17)
    assert ss.entropy == ROOT
    assert tuple(ss.spawn_key) == (int(Split.TRAIN), 2, 17)
    ref = np.random.SeedSequence(ROOT, spawn_key=(1, 2, 17))
    assert seed_state(ss) == seed_state(ref)


def _experiment_seeds():
    """All (label, SeedSequence) pairs the experiment protocols use."""
    seeds = []
    for e in range(10):  # headroom calibration
        seeds.append((("cal", 0, e), episode_seed(ROOT, Split.CALIBRATION, 0, e)))
    for run in range(4):  # 3 final runs + spare index for grid/debug
        for e in range(200):
            seeds.append((("train", run, e), episode_seed(ROOT, Split.TRAIN, run, e)))
    for v in range(10):  # 5 validation seeds per training scenario
        seeds.append((("val", 0, v), episode_seed(ROOT, Split.VALIDATION, 0, v)))
    for e in range(20):
        seeds.append((("eval", 0, e), eval_seed(ROOT, e)))
    for run in range(4):
        seeds.append((("agent", run, 0), episode_seed(ROOT, Split.AGENT, run, 0)))
        for pid in range(4):
            ep = BASELINE_EPISODE_OFFSET + pid
            seeds.append((("agent", run, ep), episode_seed(ROOT, Split.AGENT, run, ep)))
    for e in range(5):
        seeds.append((("test", 0, e), episode_seed(ROOT, Split.TEST, 0, e)))
    return seeds


def test_split_disjointness_over_experiment_seeds():
    seeds = _experiment_seeds()
    labels = [lab for lab, _ in seeds]
    assert len(set(labels)) == len(labels)
    states = [seed_state(ss, 8) for _, ss in seeds]
    assert len(set(states)) == len(states)


def test_streams_disjoint_across_splits_and_within_episode():
    roots = set()
    first_draws = set()
    for split in (Split.CALIBRATION, Split.TRAIN, Split.VALIDATION, Split.EVAL, Split.TEST):
        for e in range(5):
            st = make_streams(episode_seed(ROOT, split, 0, e))
            roots.add(st.keyed_root)
            for g in (st.ids, st.types_positions, st.churn, st.congestion, st.workload):
                first_draws.add(int(g.integers(0, 2**63)))
    assert len(roots) == 25
    assert len(first_draws) == 25 * 5


def test_eval_seeds_independent_of_run():
    for e in range(20):
        ref = seed_state(eval_seed(ROOT, e))
        for run in range(5):
            assert seed_state(env_seed(ROOT, Split.EVAL, run, e)) == ref
    # Non-eval splits do depend on run.
    assert seed_state(env_seed(ROOT, Split.TRAIN, 0, 3)) != seed_state(
        env_seed(ROOT, Split.TRAIN, 1, 3)
    )


def test_env_seed_rejects_agent_split():
    with pytest.raises(ValueError):
        env_seed(ROOT, Split.AGENT, 0, 0)


def test_make_streams_deterministic():
    a = make_streams(episode_seed(ROOT, Split.EVAL, 0, 7))
    b = make_streams(episode_seed(ROOT, Split.EVAL, 0, 7))
    assert isinstance(a, EpisodeStreams)
    assert a.keyed_root == b.keyed_root
    assert 0 <= a.keyed_root < 2**64
    for name in ("ids", "types_positions", "churn", "congestion", "workload"):
        np.testing.assert_array_equal(
            getattr(a, name).random(16), getattr(b, name).random(16)
        )


def test_agent_and_baseline_rngs_separate_from_env():
    env_draws = set()
    for split in (Split.CALIBRATION, Split.TRAIN, Split.VALIDATION, Split.EVAL, Split.TEST):
        st = make_streams(episode_seed(ROOT, split, 0, 0))
        for g in (st.ids, st.types_positions, st.churn, st.congestion, st.workload):
            env_draws.add(int(g.integers(0, 2**63)))
    a0 = int(agent_rng(ROOT, 0).integers(0, 2**63))
    a1 = int(agent_rng(ROOT, 1).integers(0, 2**63))
    b0 = int(baseline_rng(ROOT, 0, 0).integers(0, 2**63))
    b1 = int(baseline_rng(ROOT, 0, 1).integers(0, 2**63))
    others = [a0, a1, b0, b1]
    assert len(set(others)) == 4
    assert env_draws.isdisjoint(others)
    # Reproducible.
    assert int(agent_rng(ROOT, 0).integers(0, 2**63)) == a0
    assert int(baseline_rng(ROOT, 0, 1).integers(0, 2**63)) == b1


@pytest.mark.parametrize("bad", [-1, 1.5, True])
def test_invalid_indices_rejected(bad):
    with pytest.raises(ValueError):
        episode_seed(ROOT, Split.TRAIN, bad, 0)
