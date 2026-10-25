"""Tests for metrics.py (Requirements 20.1-20.3, 11.3)."""

from __future__ import annotations

import dataclasses
import math

import pytest

from churnmaster.config import (AgentConfig, Config, ExperimentConfig, RewardConfig,
                                ScenarioConfig, SimConfig)
from churnmaster.metrics import EpisodeMetrics, summarize, summary_hash
from churnmaster.seeding import Split, env_seed, make_streams
from churnmaster.sim.env import Env
from churnmaster.sim.observation import DecisionRecord, GTRecord, ObservedLookup

SCN = ScenarioConfig(name="high_churn", stable_share=0.3, stable_session_mean_s=3600.0,
                     flaky_session_mean_s=300.0, downtime_mean_s=600.0)


def _cfg(**sim) -> Config:
    return Config(sim=SimConfig(**sim), scenario=SCN, reward=RewardConfig(w_L=0.3, w_M=0.2, w_R=0.3),
                  agent=AgentConfig(), experiment=ExperimentConfig(), root_seed=7)


CFG = _cfg(measured_s=3600)  # 1 measured hour; w_L=0.3, w_M=0.2, w_R=0.3, deadline 2000


def _records():
    lookups = [
        ObservedLookup(0, 10, 7, False, 50.0, False, 99),      # warm-up: ignored
        ObservedLookup(1, 700_000, 7, True, 100.0, True, 3),
        ObservedLookup(2, 701_000, 6, True, 300.0, True, 2),
        ObservedLookup(3, 702_000, 7, False, 2000.0, True, 5),  # give-up latency
        ObservedLookup(4, 703_000, 5, True, 1000.0, True, 4),
    ]
    decisions = [
        DecisionRecord(7, 0, 0, 1, 2, 10, 5, False),           # warm-up: ignored
        DecisionRecord(7, 20, 600_000, 3, 2, 2, 1, True),
        DecisionRecord(6, 20, 600_000, 4, 1, 1, 0, True),
        DecisionRecord(5, 21, 630_000, None, None, 3, 0, True),  # null record, still costs
    ]
    gts = [
        GTRecord.make(0, "dead_fraction", None, False, fraction=0.1, n_entries=10, n_dead=1),
        GTRecord.make(700_000, "dead_fraction", None, True, fraction=0.25, n_entries=8, n_dead=2),
        GTRecord.make(760_000, "dead_fraction", None, True, fraction=0.5, n_entries=8, n_dead=4),
        GTRecord.make(0, "decision", 7, False, epoch_idx=0, action=1, inc_online=True),
        GTRecord.make(600_000, "decision", 7, True, epoch_idx=20, action=0, inc_online=True),
        GTRecord.make(600_000, "decision", 6, True, epoch_idx=20, action=0, inc_online=True),
        GTRecord.make(630_000, "decision", 5, True, epoch_idx=21, action=2, inc_online=False),
        GTRecord.make(700_000, "lookup", 7, True, lookup_id=1, holders=(), queried=(),
                      responders=()),
    ]
    return lookups, decisions, gts


def test_hand_computed_metrics():
    m = summarize(*_records(), CFG)
    assert m.n_lookups == 4 and m.n_decisions == 3
    assert m.success_rate == pytest.approx(0.75)
    # successful [100, 300, 1000]: rank 0.95*2 = 1.9 -> 300 + 0.9*700
    assert m.p95_latency_success_ms == pytest.approx(930.0)
    # all [100, 300, 1000, 2000]: rank 0.95*3 = 2.85 -> 1000 + 0.85*1000
    assert m.p95_latency_all_ms == pytest.approx(1850.0)
    assert m.mean_latency_ms == pytest.approx(1400.0 / 3)
    assert m.pings_per_hour == pytest.approx(6.0)
    assert m.evictions_per_hour == pytest.approx(1.0)
    assert m.messages_per_lookup == pytest.approx((14 + 6) / 4)
    # losses 0.015 + 0.045 + 1.3 + 0.15 = 1.51; cost 0.2*6 + 0.3*1 = 1.5
    assert m.J == pytest.approx(1 - 1.51 / 4 - 1.5 / 4, abs=1e-12)
    assert m.dead_entry_fraction == ((700_000, 0.25), (760_000, 0.5))
    assert m.actions_by_true_state_dict() == {("inc_online", 0): 2, ("inc_offline", 2): 1}


def test_rates_scale_with_measured_window():
    m = summarize(*_records(), _cfg(measured_s=1800))
    assert m.pings_per_hour == pytest.approx(12.0)
    assert m.evictions_per_hour == pytest.approx(2.0)


def test_warmup_records_excluded():
    lookups, decisions, gts = _records()
    base = summarize(lookups, decisions, gts, CFG)
    extra_l = [ObservedLookup(9, 5, 7, False, 2000.0, False, 50)]
    extra_d = [DecisionRecord(1, 1, 30_000, 0, 2, 7, 3, False)]
    extra_g = [GTRecord.make(60_000, "dead_fraction", None, False, fraction=1.0),
               GTRecord.make(30_000, "decision", 1, False, action=2, inc_online=False)]
    m = summarize(lookups + extra_l, decisions + extra_d, gts + extra_g, CFG)
    assert m == base and summary_hash(m) == summary_hash(base)


def test_only_warmup_gives_nan():
    lookups, decisions, gts = _records()
    m = summarize(lookups[:1], decisions[:1], gts[:1], CFG)
    assert m.n_lookups == 0 and m.n_decisions == 0
    for f in ("success_rate", "p95_latency_success_ms", "p95_latency_all_ms",
              "mean_latency_ms", "messages_per_lookup", "J"):
        assert math.isnan(getattr(m, f)), f
    assert m.pings_per_hour == 0.0 and m.dead_entry_fraction == ()
    assert m.actions_by_true_state == ()


def test_no_successes_gives_nan_success_latency():
    lks = [ObservedLookup(1, 0, 7, False, 2000.0, True, 6),
           ObservedLookup(2, 0, 7, False, 1500.0, True, 4)]
    m = summarize(lks, [], [], CFG)
    assert m.success_rate == 0.0
    assert math.isnan(m.p95_latency_success_ms) and math.isnan(m.mean_latency_ms)
    assert m.p95_latency_all_ms == pytest.approx(1500 + 0.95 * 500)
    # losses 1.3 + (1 + 0.3*0.75) = 2.525
    assert m.J == pytest.approx(1 - 2.525 / 2)
    assert m.messages_per_lookup == pytest.approx(5.0)


def test_metrics_frozen_and_hashable():
    m = summarize(*_records(), CFG)
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.J = 0.0  # type: ignore[misc]
    hash(m)
    row = m.to_dict(diagnostics=False)
    assert "dead_entry_fraction" not in row and row["J"] == m.J
    full = m.to_dict()
    assert full["actions_by_true_state"] == {"inc_online:0": 2, "inc_offline:2": 1}


def test_summary_hash_stable_and_sensitive():
    m = summarize(*_records(), CFG)
    m2 = summarize(*_records(), CFG)
    h = summary_hash(m)
    assert h == summary_hash(m2) and len(h) == 64
    assert summary_hash(dataclasses.replace(m, J=m.J + 1e-15)) != h
    assert summary_hash(dataclasses.replace(m, dead_entry_fraction=())) != h
    nan_m = summarize([], [], [], CFG)
    assert summary_hash(nan_m) == summary_hash(summarize([], [], [], CFG))


def test_env_smoke_maintenance_off():
    cfg = _cfg(warmup_s=60, measured_s=240)

    def run():
        env = Env(cfg, make_streams(env_seed(cfg.root_seed, Split.TEST, 0, 0)),
                  maintenance=False)
        lookups, decisions, gts = [], [], []
        trace = env.run(None, decisions.append, lookups.append, gt_annex=gts.append)
        return trace, summarize(lookups, decisions, gts, cfg), lookups

    trace, m, lookups = run()
    assert isinstance(m, EpisodeMetrics)
    assert m.n_lookups == trace.n_measured_lookup > 0 and m.n_decisions == 0
    assert 0.0 <= m.success_rate <= 1.0
    assert m.pings_per_hour == 0.0 and m.evictions_per_hour == 0.0
    assert sum(lk.queries for lk in lookups) > 0 and m.messages_per_lookup > 0
    assert -cfg.reward.w_L <= m.J <= 1.0
    assert [t for t, _ in m.dead_entry_fraction] == [60_000, 120_000, 180_000, 240_000]
    assert summary_hash(run()[1]) == summary_hash(m)
