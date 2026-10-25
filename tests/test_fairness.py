"""Fairness across policies (Requirements 21.1, 21.2, 21.4).

All policies on the same episode seed face the same churn trace, workload and keyed
network luck; every epoch-mode policy gets decision points (including null records for
empty buckets) at exactly the same epoch times for every bucket.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from churnmaster.experiments.runner import (POLICY_NAMES, ConfigLoader, Job, make_policy,
                                            policy_env_mode, run_job)
from churnmaster.policies.heuristic import all_configs
from churnmaster.seeding import Split, env_seed, make_streams
from churnmaster.sim.env import Env
from churnmaster.sim.keyed_rng import PingKind
from churnmaster.sim.network import build_network

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SHORT_SIM = "[sim]\nwarmup_s = 60\nmeasured_s = 240\n"
SCENARIO = "high_churn"
FREQS = {False: (0.5, 0.5, 0.0), True: (0.2, 0.3, 0.5)}

EPOCH_POLICIES = tuple(p for p in POLICY_NAMES
                       if p not in ("kademlia_std", "none") and policy_env_mode(p) == "epoch") + (
    f"heuristic:{all_configs()[0].id}", f"heuristic:{all_configs()[-1].id}")
ALL_POLICIES = EPOCH_POLICIES + ("kademlia_std", "none")


def _job(policy: str, episode: int = 0) -> Job:
    kw = {"freqs": FREQS} if policy == "matched_random" else {}
    return Job(policy, SCENARIO, Split.CALIBRATION, 0, episode, policy_kw=kw)


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    exp = tmp_path_factory.mktemp("exp") / "short.toml"
    exp.write_text(SHORT_SIM, encoding="utf-8")
    return ConfigLoader(CONFIGS, exp)(SCENARIO)


@pytest.fixture(scope="module")
def results(cfg):
    return {p: run_job(_job(p), cfg=cfg, keep_records=True) for p in ALL_POLICIES}


def _epoch_records(decisions):
    """Epoch-loop records: decisions, or null records with no cost (empty buckets).
    kademlia_std offer records are null-state but always cost one ping."""
    return [d for d in decisions if d.action is not None or (d.pings == 0 and d.evictions == 0)]


def test_epoch_policy_set():
    assert {"never_replace", "always_replace", "uniform_random", "matched_random",
            "kademlia_epoch", "heuristic", "oracle_heuristic"} <= set(EPOCH_POLICIES)


# --------------------------------------------------------------------- 21.1 trace/workload

def test_same_seed_identical_trace_and_workload(results):
    assert len({r.trace.trace_hash for r in results.values()}) == 1
    assert len({r.trace.workload_hash for r in results.values()}) == 1
    assert len({(r.trace.n_churn, r.trace.n_lookup, r.trace.n_epoch)
                for r in results.values()}) == 1


def test_different_seed_differs(cfg, results):
    for p in ("never_replace", "kademlia_std"):
        other = run_job(_job(p, episode=1), cfg=cfg)
        assert other.trace.trace_hash != results[p].trace.trace_hash
        assert other.trace.workload_hash != results[p].trace.workload_hash


def test_every_policy_sees_the_same_lookups(results):
    seen = {p: [(l.lookup_id, l.t_ms) for l in r.lookups] for p, r in results.items()}
    ref = seen["none"]
    assert [lid for lid, _ in ref] == list(range(len(ref))) and ref
    for p, s in seen.items():
        assert s == ref, p


def test_policies_actually_differ(results):
    """Guard against a vacuous fairness check: decisions do differ across policies."""
    acts = {p: Counter(d.action for d in results[p].decisions) for p in EPOCH_POLICIES}
    assert len({tuple(sorted(c.items(), key=str)) for c in acts.values()}) > 3


# --------------------------------------------------------------- 21.4 decision points

def test_epoch_mode_policies_share_decision_times(cfg, results):
    id_bits, epoch_ms = cfg.sim.id_bits, cfg.sim.epoch_s * 1000
    n_epoch = results["never_replace"].trace.n_epoch
    expected = [(k, k * epoch_ms, b) for k in range(n_epoch) for b in range(id_bits)]
    for p in EPOCH_POLICIES:
        r = results[p]
        assert r.shares_decision_points is True, p
        got = [(d.epoch_idx, d.t_ms, d.bucket) for d in r.decisions]
        assert got == expected, p                       # one record per bucket per epoch
        per_epoch = Counter((d.epoch_idx, d.t_ms) for d in r.decisions)
        assert set(per_epoch.values()) == {id_bits}, p


def test_kademlia_std_epoch_records_match_plus_offer_records(cfg, results):
    id_bits, epoch_ms = cfg.sim.id_bits, cfg.sim.epoch_s * 1000
    ref = [(d.epoch_idx, d.t_ms, d.bucket) for d in results["never_replace"].decisions]
    r = results["kademlia_std"]
    assert r.shares_decision_points is False
    epoch = _epoch_records(r.decisions)
    assert [(d.epoch_idx, d.t_ms, d.bucket) for d in epoch] == ref
    assert all(d.action in (None, 0) for d in epoch)    # KEEP or null at every epoch
    extra = [d for d in r.decisions if d not in epoch]
    assert extra, "high_churn should trigger ping-on-offer"
    lookup_times = {l.t_ms for l in r.lookups}
    for d in extra:
        assert d.state is None and d.action is None and d.pings == 1
        assert d.t_ms in lookup_times and d.epoch_idx == d.t_ms // epoch_ms
        assert 0 <= d.bucket < id_bits


def test_no_maintenance_has_no_decision_points(results):
    assert results["none"].decisions == []


# --------------------------------------------------------------- 21.2 keyed outcomes

def _probe_keys(net, horizon_ms: int):
    """(node index, t, key) triples in the lookup and epoch-ping key schemas."""
    out = []
    for j, t in enumerate(range(0, horizon_ms, 7_919)):
        idx = 1 + (j * 37) % (net.n_nodes - 1)
        dst = net.id_of(idx)
        out.append((idx, t, (j, dst)))
        out.append((idx, t, (j, net.bucket_of(idx), dst, int(PingKind.INC))))
    return out


def _probe(net, keys):
    return [(net.query(i, t, key), net.rtt_ms(i, t, key)) for i, t, key in keys]


def _fresh_env(cfg, mode="epoch"):
    return Env(cfg, make_streams(env_seed(cfg.root_seed, Split.CALIBRATION, 0, 0)), mode=mode)


def test_same_key_same_outcome_across_networks_and_call_order(cfg):
    horizon = (cfg.sim.warmup_s + cfg.sim.measured_s) * 1000
    nets = [build_network(cfg.sim, cfg.scenario,
                          make_streams(env_seed(cfg.root_seed, Split.CALIBRATION, 0, 0)), horizon)
            for _ in range(2)]
    keys = _probe_keys(nets[0], horizon)
    a = _probe(nets[0], keys)
    b = _probe(nets[1], keys[::-1])[::-1]
    assert a == b
    succ = [q.success for q, _ in a]
    assert any(succ) and not all(succ)                  # non-trivial mix of outcomes
    assert len({round(rtt, 9) for _, rtt in a}) > len(a) // 2  # jitter really varies


def test_same_key_same_outcome_interleaved_with_policies(cfg):
    env_a = _fresh_env(cfg)
    keys = _probe_keys(env_a.net, env_a.horizon_ms)
    before = _probe(env_a.net, keys)

    pol_a = make_policy("always_replace", cfg)
    env_a.run(pol_a.decide, lambda d: None, lambda l: None)
    after_a = _probe(env_a.net, keys)

    env_b = _fresh_env(cfg, mode="kademlia_std")
    pol_b = make_policy("kademlia_std", cfg)
    during_b: list = []

    def decide_b(ctx):
        if ctx.epoch_idx == 5 and not during_b:          # probe mid-episode
            during_b.extend(_probe(env_b.net, keys[::-1])[::-1])
        return pol_b.decide(ctx)

    env_b.run(decide_b, lambda d: None, lambda l: None)
    after_b = _probe(env_b.net, keys)
    assert during_b
    assert before == after_a == during_b == after_b
