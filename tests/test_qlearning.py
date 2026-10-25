"""Tests for rl/qlearning.py (Requirement 16)."""

from __future__ import annotations

import numpy as np
import pytest

from churnmaster.actions import Action
from churnmaster.config import AgentConfig
from churnmaster.rl.ledger import Transition
from churnmaster.rl.qlearning import QAgent, epsilon
from churnmaster.rl.state import (
    CAND_FRESH, CAND_NONE, N_ACTIONS, N_STATES, Features, allowed_actions, decode, encode,
)

S_NOCAND = encode(Features(0, 0, 0, 0, CAND_NONE, 0, 0))
S_CAND = encode(Features(0, 0, 0, 0, CAND_FRESH, 1, 0))


def test_init_shapes_and_from_config():
    ag = QAgent.from_config(AgentConfig())
    assert ag.Q.shape == (N_STATES, N_ACTIONS) == (648, 3)
    assert ag.Q.dtype == np.float64 and ag.N.dtype == np.int64
    assert not ag.Q.any() and not ag.N.any()
    assert (ag.gamma, ag.alpha_min, ag.alpha_exp) == (0.9, 0.05, 0.7)


def test_alpha_schedule():
    ag = QAgent()
    s, a = S_CAND, 2
    assert ag.alpha(s, a) == 1.0
    ag.N[s, a] = 1
    assert ag.alpha(s, a) == pytest.approx(2 ** -0.7)
    ag.N[s, a] = 10
    assert ag.alpha(s, a) == pytest.approx(11 ** -0.7)
    ag.N[s, a] = 10**9
    assert ag.alpha(s, a) == 0.05


def test_hand_computed_update():
    ag = QAgent()
    s, a, s2 = S_CAND, int(Action.PROBE), S_NOCAND
    ag.Q[s2] = [-1.0, -0.5, -0.1]  # REPLACE not allowed in s2 -> max over KEEP/PROBE = -0.5
    ag.Q[s, a] = -2.0
    ag.N[s, a] = 1
    before_Q, before_N = ag.Q.copy(), ag.N.copy()
    d = ag.update(s, a, -0.3, s2)
    lr = 2 ** -0.7
    target = -0.3 + 0.9 * -0.5
    expected = -2.0 + lr * (target + 2.0)
    assert ag.Q[s, a] == pytest.approx(expected)
    assert d == pytest.approx(abs(expected + 2.0))
    assert ag.N[s, a] == 2
    diff_Q = ag.Q != before_Q
    diff_N = ag.N != before_N
    assert diff_Q.sum() == 1 and diff_Q[s, a]
    assert diff_N.sum() == 1 and diff_N[s, a]
    assert ag.n_updates == 1 and ag.mean_abs_delta() == pytest.approx(d)
    ag.reset_stats()
    assert ag.n_updates == 0 and np.isnan(ag.mean_abs_delta())


def test_update_transition_and_allowed_next():
    ag = QAgent()
    ag.Q[S_CAND] = [-1.0, -1.0, -0.2]
    tr = Transition(0, 3, S_NOCAND, 0, -1.0, S_CAND,
                    tuple(int(x) for x in allowed_actions(S_CAND)))
    ag.update_transition(tr)
    assert ag.Q[S_NOCAND, 0] == pytest.approx(-1.0 + 0.9 * -0.2)


def test_update_rejects_disallowed_action():
    ag = QAgent()
    with pytest.raises(ValueError):
        ag.update(S_NOCAND, int(Action.REPLACE), -1.0, S_CAND)
    with pytest.raises(ValueError):
        ag.update(N_STATES, 0, -1.0, 0)


def test_q_stays_nonpositive_with_nonpositive_rewards():
    rng = np.random.default_rng(0)
    ag = QAgent()
    for _ in range(3000):
        s = int(rng.integers(N_STATES))
        acts = allowed_actions(s)
        a = int(acts[rng.integers(len(acts))])
        s2 = int(rng.integers(N_STATES))
        ag.update(s, a, -float(rng.random()) * 3, s2)
    assert (ag.Q <= 0).all()
    assert (ag.Q[~ag.mask] == 0).all() and (ag.N[~ag.mask] == 0).all()


def test_greedy_ties_and_masking():
    ag = QAgent()
    assert ag.greedy(S_CAND) == Action.KEEP  # all-equal -> KEEP
    ag.Q[S_CAND] = [-1.0, -0.5, -0.5]
    assert ag.greedy(S_CAND) == Action.PROBE  # tie PROBE/REPLACE -> PROBE
    ag.Q[S_CAND] = [-1.0, -0.5, -0.1]
    assert ag.greedy(S_CAND) == Action.REPLACE
    ag.Q[S_NOCAND] = [-1.0, -0.5, 0.0]  # REPLACE has the max but is disallowed
    assert ag.greedy(S_NOCAND) == Action.PROBE
    table = ag.greedy_table()
    assert table.shape == (N_STATES,)
    assert table[S_CAND] == 2 and table[S_NOCAND] == 1
    for s in range(N_STATES):
        if decode(s).cand == CAND_NONE:
            assert table[s] != int(Action.REPLACE)


def test_act_never_disallowed_and_uniform_at_eps1():
    ag = QAgent()
    rng = np.random.default_rng(1)
    counts = np.zeros(3, dtype=int)
    for _ in range(6000):
        counts[ag.act(S_CAND, 1.0, rng)] += 1
    assert np.all(np.abs(counts / 6000 - 1 / 3) < 0.03)
    seen = {ag.act(S_NOCAND, 1.0, rng) for _ in range(500)}
    assert seen == {Action.KEEP, Action.PROBE}


def test_act_eps0_is_greedy_and_deterministic():
    ag = QAgent()
    rng = np.random.default_rng(2)
    ag.Q[:, 1] = -0.1
    ag.Q[:, 0] = -0.2
    for s in range(0, N_STATES, 7):
        assert ag.act(s, 0.0, rng) == ag.greedy(s)
    a1 = [QAgent().act(S_CAND, 0.5, r) for r in [np.random.default_rng(3)] for _ in range(50)]
    a2 = [QAgent().act(S_CAND, 0.5, r) for r in [np.random.default_rng(3)] for _ in range(50)]
    assert a1 == a2


def test_act_rejects_bad_eps_and_empty_state():
    ag = QAgent(n_states=2, n_actions=2, mask=np.array([[True, True], [False, False]]))
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        ag.act(0, 1.5, rng)
    with pytest.raises(ValueError):
        ag.act(1, 0.0, rng)


def test_epsilon_schedule():
    cfg = AgentConfig()  # 200 episodes, E = 120
    assert epsilon(0, cfg) == 1.0
    assert epsilon(60, cfg) == pytest.approx(1.0 + (0.05 - 1.0) * 0.5)
    assert epsilon(119, cfg) > 0.05
    assert epsilon(120, cfg) == 0.05
    assert epsilon(199, cfg) == 0.05
    one = AgentConfig(episodes=1)  # E = 0.6
    assert epsilon(0, one) == 1.0
    assert epsilon(1, one) == 0.05
    with pytest.raises(ValueError):
        epsilon(-1, cfg)
    eps = [epsilon(e, cfg) for e in range(200)]
    assert all(x >= y for x, y in zip(eps, eps[1:]))


def test_save_load_round_trip(tmp_path):
    rng = np.random.default_rng(4)
    ag = QAgent(gamma=0.8, alpha_min=0.1, alpha_exp=0.6)
    ag.Q[ag.mask] = -rng.random(int(ag.mask.sum()))
    ag.N[ag.mask] = rng.integers(0, 100, int(ag.mask.sum()))
    p = ag.save(tmp_path / "sub" / "q")
    assert p.suffix == ".npy" and p.exists()
    assert np.array_equal(np.load(p), ag.Q)
    back = QAgent.load(p)
    assert np.array_equal(back.Q, ag.Q) and np.array_equal(back.N, ag.N)
    assert np.array_equal(back.mask, ag.mask)
    assert (back.gamma, back.alpha_min, back.alpha_exp) == (0.8, 0.1, 0.6)


def test_toy_two_state_mdp_converges():
    # P[s, a] = probability of moving to state 1; R[s, a] = deterministic reward.
    P = np.array([[0.2, 0.9], [0.5, 0.1]])
    R = np.array([[-1.0, -2.0], [0.0, -0.5]])
    gamma = 0.9
    Qs = np.zeros((2, 2))
    for _ in range(2000):  # value iteration for analytic Q*
        V = Qs.max(axis=1)
        Qs = R + gamma * ((1 - P) * V[0] + P * V[1])

    ag = QAgent(gamma=gamma, alpha_min=0.001, n_states=2, n_actions=2,
                mask=np.ones((2, 2), dtype=bool))
    rng = np.random.default_rng(5)
    s = 0
    for _ in range(40_000):
        a = int(ag.act(s, 1.0, rng))
        s2 = int(rng.random() < P[s, a])
        ag.update(s, a, R[s, a], s2)
        s = s2
    assert np.max(np.abs(ag.Q - Qs)) <= 0.05
