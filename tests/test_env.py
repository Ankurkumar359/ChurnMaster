"""Tests for sim/env.py and sim/observation.py (Requirements 11.1-11.4, 12.6,
14.1, 14.2, 14.8, 14.10, 17.3, 17.4, 21.4)."""

from __future__ import annotations

import dataclasses

import pytest

from churnmaster.config import (AgentConfig, Config, ExperimentConfig, RewardConfig,
                                ScenarioConfig, SimConfig)
from churnmaster.seeding import Split, env_seed, make_streams
from churnmaster.sim.env import (P_CHURN, P_EPOCH, P_LOOKUP, P_REBUILD, Env, merge_events)
from churnmaster.sim.maintenance import Action, IllegalActionError
from churnmaster.sim.observation import (DecisionContext, DecisionRecord, GTRecord,
                                         ObservedLookup)
from churnmaster.sim.workload import LookupRequest

SCN = ScenarioConfig(name="high_churn", stable_share=0.3, stable_session_mean_s=3600.0,
                     flaky_session_mean_s=300.0, downtime_mean_s=600.0)


def _cfg(**sim) -> Config:
    return Config(sim=SimConfig(**sim), scenario=SCN, reward=RewardConfig(),
                  agent=AgentConfig(), experiment=ExperimentConfig(), root_seed=7)


def _env(cfg: Config | None = None, episode: int = 0, **kw) -> Env:
    cfg = cfg or _cfg()
    return Env(cfg, make_streams(env_seed(cfg.root_seed, Split.TEST, 0, episode)), **kw)


def _run(env: Env, gt: bool = True):
    lookups, decisions, gts = [], [], []
    trace = env.run(None, decisions.append, lookups.append,
                    gt_annex=gts.append if gt else None)
    return trace, lookups, decisions, gts


@pytest.fixture(scope="module")
def episode():
    env = _env(maintenance=False, record_events=True)
    return (env, *_run(env))


# ------------------------------------------------------------------ ordering
def test_merge_orders_coincident_events_by_priority_then_tie():
    t = 30_000
    evs = merge_events(
        churn=[(t, 5, "down"), (t, 2, "up")],
        rebuilds=[(t, 9, 1), (t, 3, 1)],
        epochs=[(t, 1)],
        gt_samples=[],
        lookups=[LookupRequest(4, t, 10), LookupRequest(3, t, 11)],
        horizon_ms=100_000)
    got = [(e.priority, e.tie) for e in evs]
    assert got == [(P_CHURN, 2), (P_CHURN, 5), (P_REBUILD, 3), (P_REBUILD, 9),
                   (P_EPOCH, 1), (P_LOOKUP, 3), (P_LOOKUP, 4)]


def test_merge_drops_events_at_or_after_horizon():
    evs = merge_events([(99, 1, "up"), (100, 1, "down")], [], [(0, 0), (100, 1)], [100],
                       [LookupRequest(0, 100, 1)], horizon_ms=100)
    assert [e.t_ms for e in evs] == [0, 99]


def test_episode_processes_events_in_order(episode):
    env, trace, *_ = episode
    keys = [(t, p, tie) for t, p, tie, _ in trace.events]
    assert keys == sorted(keys)
    # The real schedule has coincident rebuild/epoch timestamps.
    by_t: dict[int, set[int]] = {}
    for t, p, _ in keys:
        by_t.setdefault(t, set()).add(p)
    assert any({P_REBUILD, P_EPOCH} <= ps for ps in by_t.values())


# --------------------------------------------------------------- time model
def test_episode_ends_on_time_limit(episode):
    env, trace, lookups, _, gts = episode
    assert trace.horizon_ms == 3_000_000 and trace.warmup_ms == 600_000
    assert all(0 <= t < trace.horizon_ms for t, *_ in trace.events)
    assert all(lk.t_ms < trace.horizon_ms for lk in lookups)
    assert trace.n_epoch == 100 and trace.n_gt_sample == 50
    assert trace.n_lookup == len(lookups) == len(env.workload)


def test_warmup_lookups_unmeasured(episode):
    env, trace, lookups, _, gts = episode
    assert lookups and any(lk.t_ms < 600_000 for lk in lookups)
    for lk in lookups:
        assert lk.measured == (lk.t_ms >= 600_000)
    for g in gts:
        assert g.measured == (g.t_ms >= 600_000)
    assert trace.n_measured_lookup == sum(lk.measured for lk in lookups)


def test_maintenance_off_runs_without_decisions(episode):
    _, _, lookups, decisions, _ = episode
    assert decisions == []
    assert sum(lk.success for lk in lookups) > 0


# ------------------------------------------------------- observation boundary
def test_gt_records_only_via_gt_annex(episode):
    _, _, lookups, decisions, gts = episode
    assert all(type(lk) is ObservedLookup for lk in lookups)
    assert all(type(d) is DecisionRecord for d in decisions)
    assert gts and all(type(g) is GTRecord for g in gts)
    kinds = {g.kind for g in gts}
    assert kinds == {"lookup", "dead_fraction"}
    for g in gts:
        if g.kind == "dead_fraction":
            assert 0.0 <= g.get("fraction") <= 1.0


def test_gt_annex_optional_does_not_change_observations(episode):
    env, _, lookups, *_ = episode
    _, lookups2, _, gts2 = _run(env, gt=False)
    assert gts2 == [] and lookups2 == lookups


def test_same_seed_same_observations(episode):
    _, trace, lookups, *_ = episode
    trace2, lookups2, *_ = _run(_env(maintenance=False, record_events=True))
    assert lookups2 == lookups
    assert trace2 == trace
    trace3, lookups3, *_ = _run(_env(maintenance=False, episode=1))
    assert trace3.trace_hash != trace.trace_hash and lookups3 != lookups


def test_maintenance_on_requires_decide():
    with pytest.raises(ValueError):
        _run(_env(_cfg(warmup_s=60, measured_s=240)))


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        _env(mode="bogus")


# --------------------------------------------------------------- record types
def test_records_are_frozen():
    lk = ObservedLookup(0, 1, 2, True, 3.0, False)
    d = DecisionRecord(1, 2, 3, None, None, 1, 0, True)
    g = GTRecord.make(5, "dead_fraction", None, True, fraction=0.5)
    for rec in (lk, d, g):
        with pytest.raises(dataclasses.FrozenInstanceError):
            rec.t_ms = 9  # type: ignore[misc]
    assert g.get("fraction") == 0.5 and g.get("missing") is None
    assert g.as_dict() == {"fraction": 0.5}


# ------------------------------------------------------- maintenance epochs
# Requirements 14.1, 14.2, 14.8, 14.10, 17.3, 17.4, 21.4.
SHORT = dict(warmup_s=60, measured_s=540)   # 600 s -> 20 epochs


def keep(ctx: DecisionContext) -> Action:
    return Action.KEEP


def kademlia_epoch(ctx: DecisionContext) -> Action:
    return Action.PROBE if ctx.cand_since_seen_s is not None else Action.KEEP


def _run_policy(env: Env, decide, gt: bool = True, **kw):
    lookups, decisions, gts = [], [], []
    trace = env.run(decide, decisions.append, lookups.append,
                    gt_annex=gts.append if gt else None, **kw)
    return trace, lookups, decisions, gts


def test_one_decision_per_nonempty_bucket_and_null_records_for_empty():
    env = _env(_cfg(**SHORT))
    calls: list[tuple[int, int]] = []
    decisions: list[DecisionRecord] = []

    def decide(ctx):
        assert type(ctx) is DecisionContext
        calls.append((ctx.epoch_idx, ctx.bucket))
        return Action.KEEP

    def on_decision(d: DecisionRecord):
        # Records are emitted right after autofill (+ KEEP): null <=> empty bucket.
        assert (len(env.view.bucket(d.bucket)) == 0) == (d.action is None)
        decisions.append(d)

    trace = env.run(decide, on_decision, lambda lk: None)
    assert trace.n_epoch == 20 and len(decisions) == 20 * env.cfg.sim.id_bits
    by_epoch: dict[int, list[DecisionRecord]] = {}
    for d in decisions:
        by_epoch.setdefault(d.epoch_idx, []).append(d)
        assert d.t_ms == d.epoch_idx * env.epoch_ms
        assert d.measured == (d.t_ms >= env.warmup_ms)
    for k, recs in by_epoch.items():
        assert [d.bucket for d in recs] == list(range(env.cfg.sim.id_bits))
        non_null = [d.bucket for d in recs if d.action is not None]
        assert non_null == [b for kk, b in calls if kk == k]
        assert all(d.pings == d.evictions == 0 for d in recs)
        assert all(d.state is None for d in recs)
    assert any(d.action is None for d in decisions) and any(d.action is not None for d in decisions)


def test_encode_state_fills_state():
    enc = lambda ctx: ctx.bucket * 10 + min(ctx.inc_consec_fail, 9)  # noqa: E731
    _, _, decisions, _ = _run_policy(_env(_cfg(**SHORT)), kademlia_epoch, encode_state=enc)
    for d in decisions:
        assert (d.state is None) == (d.action is None)
        if d.state is not None:
            assert d.state // 10 == d.bucket


def test_replace_without_candidate_raises():
    with pytest.raises(IllegalActionError):
        _run_policy(_env(_cfg(**SHORT)), lambda ctx: Action.REPLACE)
    with pytest.raises(IllegalActionError):
        _run_policy(_env(_cfg(**SHORT)), lambda ctx: 7)


def test_candidates_only_from_lookup_responders():
    env = _env(_cfg(**SHORT))
    seen = {e.node for e in env.net.bootstrap_entries(0)}
    snapshots: list[set[int]] = []

    def gt_annex(g: GTRecord):
        if g.kind == "lookup":
            seen.update(g.get("responders"))
        elif g.kind == "decision" and g.get("inserted") is not None:
            assert g.get("inserted") in seen

    def decide(ctx):
        v = env.view
        snapshots.append({e.node for e in v.all_entries()}
                         | {c.node for b in range(v.id_bits) for c in v.cache(b)})
        return kademlia_epoch(ctx)

    env.run(decide, lambda d: None, lambda lk: None, gt_annex=gt_annex)
    assert snapshots and all(s <= seen for s in snapshots)
    v = env.view
    assert {c.node for b in range(v.id_bits) for c in v.cache(b)} <= seen
    assert {e.node for e in v.all_entries()} <= seen


def test_gt_decision_records_and_oracle_facts():
    env = _env(_cfg(**SHORT))
    facts: dict[tuple[int, int], tuple] = {}

    def decide(ctx):
        f = env.oracle_facts(ctx.bucket, ctx.t_ms)
        facts[(ctx.epoch_idx, ctx.bucket)] = f
        inc_online, cand_present, cand_online = f
        assert cand_present == (ctx.cand_since_seen_s is not None)
        assert (cand_online is None) == (not cand_present)
        return Action.PROBE   # probes reveal the incumbent's true state

    _, _, decisions, gts = _run_policy(env, decide)
    dec_gt = [g for g in gts if g.kind == "decision"]
    non_null = [d for d in decisions if d.action is not None]
    assert len(dec_gt) == len(non_null) == len(facts)
    for g, d in zip(dec_gt, non_null):
        assert (g.get("epoch_idx"), g.bucket, g.get("action")) == (d.epoch_idx, d.bucket, d.action)
        assert facts[(d.epoch_idx, d.bucket)] == (g.get("inc_online"), g.get("cand_present"),
                                                  g.get("cand_online"))
        if not g.get("inc_online"):           # an offline incumbent never answers
            assert d.evictions == 1 and g.get("evicted") is not None
    assert any(not g.get("inc_online") for g in dec_gt)
    # GT only through gt_annex: observable records carry no GT fields.
    assert all(type(d) is DecisionRecord for d in decisions)


def test_kademlia_std_mode_costs_and_keep_epochs():
    env = _env(_cfg(**SHORT), mode="kademlia_std")
    _, _, decisions, _ = _run_policy(env, keep)
    std = [d for d in decisions if d.state is None and d.action is None and d.pings > 0]
    epochs = [d for d in decisions if d.action is not None]
    assert epochs and all(d.action == Action.KEEP and d.pings == 0 for d in epochs)
    assert std and all(d.pings == 1 and d.evictions in (0, 1) for d in std)
    assert any(d.evictions == 1 for d in std)
    assert all(d.epoch_idx == d.t_ms // env.epoch_ms for d in std)
    with pytest.raises(ValueError):
        _run_policy(_env(_cfg(**SHORT), mode="kademlia_std"), lambda ctx: Action.PROBE)


def test_epoch_mode_has_no_std_records():
    _, _, decisions, _ = _run_policy(_env(_cfg(**SHORT)), keep)
    assert all(d.action is not None or (d.pings == 0 and d.evictions == 0) for d in decisions)


def test_decision_points_identical_across_policies():
    pts = []
    for pol in (keep, kademlia_epoch):
        calls: list[tuple[int, int]] = []

        def decide(ctx, pol=pol, calls=calls):
            calls.append((ctx.epoch_idx, ctx.bucket))
            return pol(ctx)
        _run_policy(_env(_cfg(**SHORT)), decide, gt=False)
        pts.append({k for k, _ in calls})
    # Every epoch offers decisions under both policies (same epoch times).
    assert pts[0] == pts[1] == set(range(20))


def test_maintenance_deterministic_with_deterministic_policy():
    a = _run_policy(_env(_cfg(**SHORT)), kademlia_epoch)
    b = _run_policy(_env(_cfg(**SHORT)), kademlia_epoch)
    assert a == b
    assert any(d.pings > 0 for d in a[2])


def test_full_episode_runtime():
    import time
    env = _env()
    t0 = time.perf_counter()
    trace, lookups, decisions, _ = _run_policy(env, kademlia_epoch, gt=False)
    dt = time.perf_counter() - t0
    print(f"full 3000 s episode (kademlia_epoch): {dt:.3f} s")
    assert trace.n_epoch == 100 and len(decisions) >= 100 * env.cfg.sim.id_bits
    assert dt < 5.0
