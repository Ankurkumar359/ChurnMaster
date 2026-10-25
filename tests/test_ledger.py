"""Tests for rl/ledger.py (Requirements 15.1-15.4, 15.6, 15.7, 12.5)."""

from __future__ import annotations

import math

import pytest

from churnmaster.actions import Action
from churnmaster.config import (AgentConfig, Config, ExperimentConfig, RewardConfig,
                                ScenarioConfig, SimConfig)
from churnmaster.metrics import summarize
from churnmaster.rl.ledger import Ledger, Transition
from churnmaster.rl.state import CAND_FRESH, Features, allowed_actions, encode
from churnmaster.sim.observation import DecisionRecord, GTRecord, ObservedLookup

E = 30_000
S0 = 0                                                 # no candidate: KEEP, PROBE
SC = encode(Features(0, 0, 0, 0, CAND_FRESH, 0, 0))    # candidate: KEEP, PROBE, REPLACE
K, P, R = int(Action.KEEP), int(Action.PROBE), int(Action.REPLACE)

CFG = Config(sim=SimConfig(epoch_s=30, warmup_s=30, measured_s=90),
             scenario=ScenarioConfig(name="x", stable_share=0.3, stable_session_mean_s=3600.0,
                                     flaky_session_mean_s=300.0, downtime_mean_s=600.0),
             reward=RewardConfig(w_L=0.3, w_M=0.2, w_R=0.3), agent=AgentConfig(),
             experiment=ExperimentConfig(), root_seed=7)


def dec(b, k, s, a, pings=0, ev=0, t=None, measured=None):
    t = k * E if t is None else t
    return DecisionRecord(b, k, t, s, a, pings, ev, t >= E if measured is None else measured)


def lk(i, t, b, ok, lat):
    return ObservedLookup(i, t, b, ok, lat, t >= E)


def script():
    """Env-ordered records; epoch 0 is warm-up, epochs 1-3 measured."""
    return [
        # epoch 0
        dec(0, 0, S0, K), dec(1, 0, SC, R, 1, 1), dec(2, 0, None, None),
        lk(1, 0, 0, True, 1000.0),                          # loss 0.15 -> (0,0)
        lk(2, 10_000, 1, False, 2000.0),                    # loss 1.3  -> (1,0)
        dec(0, 0, None, None, 1, 0, t=10_000),              # kademlia_std cost 0.2 -> (0,0)
        lk(3, 20_000, 2, True, 0.0),                        # loss 0 -> (2,0), no decision
        # epoch 1
        dec(0, 1, S0, P, 1),                                # closes (0,0)
        dec(1, 1, None, None),                              # empty bucket: drops (1,0)
        dec(2, 1, SC, K),
        dec(0, 1, None, None, 1, 0, t=E),                   # cost record at t_k (not epoch rec)
        lk(4, E, 2, True, 500.0),                           # loss 0.075 -> (2,1)
        # epoch 2: bucket 0 has no record (gap)
        dec(1, 2, S0, K), dec(2, 2, S0, K),                 # closes (2,1)
        # epoch 3
        dec(0, 3, S0, K),                                   # gap: drops (0,1)
        dec(1, 3, SC, K),                                   # closes (1,2)
    ]


def run(records):
    led = Ledger(0.3, 0.2, 0.3, 2000, E, 8)
    emitted = []
    for rec in records:
        emitted.extend(led.add(rec))
    return led, emitted


def test_scripted_rewards_and_transitions():
    led, emitted = run(script())
    assert led.reward(0, 0) == pytest.approx(-0.35)
    assert led.reward(1, 0) == pytest.approx(-1.8)
    assert led.reward(2, 0) == 0.0
    assert led.reward(0, 1) == pytest.approx(-0.4)
    assert led.reward(2, 1) == pytest.approx(-0.075)
    got = [(t.bucket, t.epoch_idx, t.s, t.a, t.s_next) for t in emitted]
    assert got == [(0, 0, S0, K, S0), (2, 1, SC, K, S0), (1, 2, S0, K, SC)]
    assert [t.r for t in emitted] == pytest.approx([-0.35, -0.075, 0.0])
    for t in emitted:
        assert t.allowed_next == tuple(int(a) for a in allowed_actions(t.s_next))
    assert emitted[2].allowed_next == (K, P, R)
    assert led.transitions == tuple(emitted)
    assert led.close_epoch(1) == [emitted[0]] and led.close_epoch(3) == [emitted[2]]


def test_finish_drops_open_transitions():
    led, _ = run(script())
    assert led.finish() == []
    # bucket 1 had an open transition at epoch 3; it is gone after finish
    assert led.add(dec(1, 4, S0, K)) == []
    assert len(led.transitions) == 3


def test_cost_records_neither_close_nor_drop():
    led = Ledger(0.3, 0.2, 0.3, 2000, E)
    led.add(dec(5, 0, S0, K))
    assert led.add(dec(5, 0, None, None, 1, 1, t=100)) == []
    assert led.add(dec(5, 0, None, None, 1, 0, t=0)) == []    # at t_0 but after the epoch record
    out = led.add(dec(5, 1, S0, K))
    assert len(out) == 1 and out[0].r == pytest.approx(-(0.5 + 0.2))


def test_score_matches_metrics_and_excludes_warmup():
    records = script()
    led, emitted = run(records)
    lookups = [r for r in records if isinstance(r, ObservedLookup)]
    decisions = [r for r in records if isinstance(r, DecisionRecord)]
    m = summarize(lookups, decisions, [], CFG)
    assert led.n_measured_lookups == 1 == m.n_lookups
    assert led.total_reward() == pytest.approx(-0.475)
    assert led.score() == pytest.approx(0.525)
    assert abs(led.score() - m.J) < 1e-9
    # warm-up epoch 0 still yields a learning transition
    assert emitted[0].epoch_idx == 0


def test_total_reward_equals_measured_bucket_epoch_sum():
    led, _ = run(script())
    measured_sum = sum(led.reward(b, k) for b in range(8) for k in range(1, 4))
    assert led.total_reward() == pytest.approx(measured_sum)


def test_score_nan_without_measured_lookups():
    led = Ledger(0.3, 0.2, 0.3)
    led.add(dec(0, 0, S0, K, measured=False))
    assert math.isnan(led.score())


@pytest.mark.parametrize("bad", [
    GTRecord.make(0, "dead_fraction", None, True, fraction=0.0),
    {"bucket": 0, "t_ms": 0},
    None,
    (0, 0, 0),
])
def test_type_error_for_other_types(bad):
    with pytest.raises(TypeError):
        Ledger(0.3, 0.2, 0.3).add(bad)


def test_type_error_for_subclass():
    class Sub(ObservedLookup):
        pass
    with pytest.raises(TypeError):
        Ledger(0.3, 0.2, 0.3).add(Sub(1, 0, 0, True, 0.0, True))


def test_disallowed_action_rejected():
    with pytest.raises(ValueError):
        Ledger(0.3, 0.2, 0.3).add(dec(0, 0, S0, R))     # REPLACE needs a candidate


def test_non_epoch_decision_with_state_rejected():
    led = Ledger(0.3, 0.2, 0.3)
    led.add(dec(0, 0, S0, K))
    with pytest.raises(ValueError):
        led.add(dec(0, 0, S0, K))                       # duplicate epoch record
    with pytest.raises(ValueError):
        led.add(dec(1, 0, S0, K, t=5))                  # not at t_k


def test_from_config():
    led = Ledger.from_config(CFG)
    assert (led.w_L, led.w_M, led.w_R, led.epoch_ms, led.deadline_ms) == (0.3, 0.2, 0.3, E, 2000)
    assert isinstance(Transition(0, 0, 0, 0, 0.0, 0, (0, 1)), Transition)
