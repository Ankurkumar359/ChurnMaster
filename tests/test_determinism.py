"""Determinism and performance of maintenance-off episodes (Requirements 11.5, 11.6)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from churnmaster.config import load_config
from churnmaster.experiments.runner import parse_split, run_episode
from churnmaster.metrics import summary_hash
from churnmaster.seeding import Split

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _cfg(scenario: str = "high_churn"):
    return load_config(CONFIGS / "base.toml", CONFIGS / "scenarios" / f"{scenario}.toml")


@pytest.fixture(scope="module")
def cfg():
    return _cfg()


@pytest.fixture(scope="module")
def first(cfg):
    return run_episode(cfg, Split.TEST, 0, 0)


def test_same_seed_identical_summary_hash(cfg, first):
    m1, t1 = first
    m2, t2 = run_episode(cfg, Split.TEST, 0, 0)
    assert summary_hash(m1) == summary_hash(m2)
    assert t1.trace_hash == t2.trace_hash and t1.workload_hash == t2.workload_hash
    assert m1.n_lookups > 0


def test_fresh_config_load_gives_same_hash(first):
    m, _ = run_episode(_cfg(), Split.TEST, 0, 0)
    assert summary_hash(m) == summary_hash(first[0])


@pytest.mark.parametrize("split, run, episode", [
    (Split.TEST, 0, 1), (Split.TEST, 1, 0), (Split.TRAIN, 0, 0)])
def test_different_seed_different_hashes(cfg, first, split, run, episode):
    m, t = run_episode(cfg, split, run, episode)
    assert summary_hash(m) != summary_hash(first[0])
    assert t.trace_hash != first[1].trace_hash
    assert t.workload_hash != first[1].workload_hash


def test_parse_split():
    assert parse_split("test") is Split.TEST and parse_split("EVAL") is Split.EVAL
    with pytest.raises(ValueError):
        parse_split("agent")
    with pytest.raises(ValueError):
        parse_split("bogus")


def test_maintenance_on_needs_policy(cfg):
    with pytest.raises(ValueError):
        run_episode(cfg, Split.TEST, 0, 0, maintenance=True)


def test_maintenance_on_with_policy_is_deterministic(cfg, first):
    m1, t1 = run_episode(cfg, Split.TEST, 0, 0, maintenance=True, policy="kademlia_epoch")
    m2, _ = run_episode(cfg, Split.TEST, 0, 0, maintenance=True, policy="kademlia_epoch")
    assert summary_hash(m1) == summary_hash(m2)
    assert t1.trace_hash == first[1].trace_hash  # same churn as maintenance off
    assert m1.n_decisions > 0 and first[0].n_decisions == 0


@pytest.mark.slow
def test_maintenance_off_episode_cpu_budget(cfg):
    """Full default 3000 s high_churn episode (incl. setup and metrics) in <= 2 s CPU."""
    assert (cfg.sim.warmup_s + cfg.sim.measured_s) == 3000
    run_episode(cfg, Split.TEST, 0, 5)  # warm imports/caches
    c0, w0 = time.process_time(), time.perf_counter()
    _, trace = run_episode(cfg, Split.TEST, 0, 6)
    cpu, wall = time.process_time() - c0, time.perf_counter() - w0
    print(f"maintenance-off 3000 s episode: cpu={cpu:.3f}s wall={wall:.3f}s "
          f"lookups={trace.n_lookup}")
    assert trace.horizon_ms == 3_000_000
    assert cpu <= 2.0, f"episode took {cpu:.3f}s CPU (> 2 s)"
