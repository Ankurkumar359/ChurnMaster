"""Tabular Q-learning agent (Requirement 16).

Observable-only: imports stdlib, numpy, ``churnmaster.actions`` and
``churnmaster.rl.state`` (plus ``rl.ledger`` for the Transition type).

Update (Requirement 16.2)::

    target   = r + gamma * max_{a' in A(s')} Q[s', a']     (r if terminal)
    lr       = max(alpha_min, (1 + N[s, a]) ** -alpha_exp)  (count BEFORE increment)
    Q[s, a] += lr * (target - Q[s, a]);  N[s, a] += 1

Training only feeds τ=1 transitions (the ledger drops transitions open at
episode end), so ``terminal`` is never used in production; it exists for toy
MDP tests.

Greedy ties break by the fixed order KEEP < PROBE < REPLACE: the lowest
action index among the allowed maxima wins. Since Q is initialised to 0 and
rewards are <= 0, an unvisited state yields KEEP.

``act`` draw order (deterministic for a seeded ``np.random.Generator``):
exactly one ``rng.random()`` is always drawn; if it is ``< eps`` a second draw
``rng.integers(len(allowed))`` picks uniformly among the allowed actions
(in KEEP, PROBE, REPLACE order); otherwise the greedy action is returned.

File layout of ``save(path)`` (``.npy`` suffix added if missing)::

    <stem>.npy          Q table, float64 (n_states, n_actions)
    <stem>_counts.npy   N counts, int64 (n_states, n_actions)
    <stem>_mask.npy     allowed-action mask, bool (n_states, n_actions)
    <stem>.json         {"gamma", "alpha_min", "alpha_exp", "n_states", "n_actions"}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from churnmaster.actions import Action
from churnmaster.rl.ledger import Transition
from churnmaster.rl.state import N_ACTIONS, N_STATES, allowed_mask

__all__ = ["QAgent", "epsilon"]


def _paths(path: str | Path) -> tuple[Path, Path, Path, Path]:
    p = Path(path)
    if p.suffix != ".npy":
        p = p.with_name(p.name + ".npy")
    stem = p.with_suffix("")
    return (p, stem.with_name(stem.name + "_counts.npy"),
            stem.with_name(stem.name + "_mask.npy"), stem.with_suffix(".json"))


class QAgent:
    """One shared Q-table with per-(s, a) update counts."""

    def __init__(self, gamma: float = 0.9, alpha_min: float = 0.05, alpha_exp: float = 0.7,
                 n_states: int = N_STATES, n_actions: int = N_ACTIONS,
                 mask: np.ndarray | None = None):
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        if not 0.0 < alpha_min <= 1.0:
            raise ValueError(f"alpha_min must be in (0, 1], got {alpha_min}")
        if alpha_exp <= 0:
            raise ValueError(f"alpha_exp must be > 0, got {alpha_exp}")
        self.gamma = float(gamma)
        self.alpha_min = float(alpha_min)
        self.alpha_exp = float(alpha_exp)
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        if mask is None:
            if (self.n_states, self.n_actions) != (N_STATES, N_ACTIONS):
                raise ValueError("mask is required for non-default table shapes")
            mask = allowed_mask()
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.n_states, self.n_actions):
            raise ValueError(f"mask shape {mask.shape} != {(self.n_states, self.n_actions)}")
        self.mask = mask.copy()
        self.Q = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        self.N = np.zeros((self.n_states, self.n_actions), dtype=np.int64)
        self.reset_stats()

    @classmethod
    def from_config(cls, agent_cfg: Any) -> "QAgent":
        """Build from a ``churnmaster.config.AgentConfig``."""
        return cls(agent_cfg.gamma, agent_cfg.alpha_min, agent_cfg.alpha_exp)

    # ------------------------------------------------------------------ #

    def _check_state(self, s: int) -> int:
        if isinstance(s, bool) or not isinstance(s, (int, np.integer)) or not 0 <= s < self.n_states:
            raise ValueError(f"state {s!r} not in [0, {self.n_states})")
        return int(s)

    def _allowed(self, s: int) -> np.ndarray:
        idx = np.flatnonzero(self.mask[s])
        if idx.size == 0:
            raise ValueError(f"state {s} has no allowed actions")
        return idx

    def alpha(self, s: int, a: int) -> float:
        """``max(alpha_min, (1 + N[s, a]) ** -alpha_exp)`` using the current count."""
        return max(self.alpha_min, float((1.0 + self.N[s, a]) ** -self.alpha_exp))

    def greedy(self, s: int) -> Action:
        """Argmax over allowed actions; ties -> lowest index (KEEP < PROBE < REPLACE)."""
        s = self._check_state(s)
        idx = self._allowed(s)
        return Action(int(idx[int(np.argmax(self.Q[s, idx]))]))  # argmax returns first max

    def act(self, s: int, eps: float, rng: np.random.Generator) -> Action:
        """ε-greedy over allowed actions (see module docstring for draw order)."""
        if not 0.0 <= eps <= 1.0:
            raise ValueError(f"eps must be in [0, 1], got {eps}")
        s = self._check_state(s)
        idx = self._allowed(s)
        if rng.random() < eps:
            return Action(int(idx[int(rng.integers(idx.size))]))
        return Action(int(idx[int(np.argmax(self.Q[s, idx]))]))

    def greedy_table(self) -> np.ndarray:
        """Greedy action per state, shape ``(n_states,)`` int64 (-1 if no action allowed)."""
        masked = np.where(self.mask, self.Q, -np.inf)
        out = np.argmax(masked, axis=1).astype(np.int64)
        out[~self.mask.any(axis=1)] = -1
        return out

    # ------------------------------------------------------------------ #

    def update(self, s: int, a: int, r: float, s_next: int,
               allowed_next: tuple[int, ...] | None = None, terminal: bool = False) -> float:
        """Apply one Q-learning update; return ``|ΔQ[s, a]|``.

        ``allowed_next`` defaults to the mask row of ``s_next``.
        """
        s = self._check_state(s)
        a = int(a)
        if not 0 <= a < self.n_actions or not self.mask[s, a]:
            raise ValueError(f"action {a} not allowed in state {s}")
        if terminal:
            target = float(r)
        else:
            s_next = self._check_state(s_next)
            nxt = self._allowed(s_next) if allowed_next is None else np.asarray(allowed_next, dtype=np.int64)
            if nxt.size == 0:
                raise ValueError(f"no allowed actions for next state {s_next}")
            target = float(r) + self.gamma * float(np.max(self.Q[s_next, nxt]))
        lr = self.alpha(s, a)
        delta = lr * (target - self.Q[s, a])
        self.Q[s, a] += delta
        self.N[s, a] += 1
        d = abs(float(delta))
        self._delta_sum += d
        self._n_updates += 1
        return d

    def update_transition(self, tr: Transition) -> float:
        return self.update(tr.s, tr.a, tr.r, tr.s_next, tr.allowed_next)

    def reset_stats(self) -> None:
        self._delta_sum = 0.0
        self._n_updates = 0

    @property
    def n_updates(self) -> int:
        """Updates since the last ``reset_stats``."""
        return self._n_updates

    def mean_abs_delta(self) -> float:
        """Mean ``|ΔQ|`` since the last ``reset_stats`` (NaN if no updates)."""
        return self._delta_sum / self._n_updates if self._n_updates else float("nan")

    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> Path:
        """Write Q, N, mask and params (see module docstring); return the Q path."""
        q_p, n_p, m_p, j_p = _paths(path)
        q_p.parent.mkdir(parents=True, exist_ok=True)
        np.save(q_p, self.Q)
        np.save(n_p, self.N)
        np.save(m_p, self.mask)
        j_p.write_text(json.dumps({
            "gamma": self.gamma, "alpha_min": self.alpha_min, "alpha_exp": self.alpha_exp,
            "n_states": self.n_states, "n_actions": self.n_actions,
        }, indent=2), encoding="utf-8")
        return q_p

    @classmethod
    def load(cls, path: str | Path) -> "QAgent":
        """Restore an agent saved by :meth:`save`. Missing sidecars fall back to
        defaults (zero counts, default params, default mask)."""
        q_p, n_p, m_p, j_p = _paths(path)
        Q = np.load(q_p)
        meta = json.loads(j_p.read_text(encoding="utf-8")) if j_p.exists() else {}
        mask = np.load(m_p) if m_p.exists() else None
        agent = cls(meta.get("gamma", 0.9), meta.get("alpha_min", 0.05),
                    meta.get("alpha_exp", 0.7), Q.shape[0], Q.shape[1], mask)
        agent.Q[...] = Q
        if n_p.exists():
            agent.N[...] = np.load(n_p)
        return agent


def epsilon(episode: int, cfg: Any) -> float:
    """Linear decay ``eps_start -> eps_end`` over ``E = eps_decay_frac * episodes``
    episodes, then ``eps_end`` (Requirement 16.3)."""
    if isinstance(episode, bool) or not isinstance(episode, (int, np.integer)) or episode < 0:
        raise ValueError(f"episode must be an int >= 0, got {episode!r}")
    E = cfg.eps_decay_frac * cfg.episodes
    if E <= 0 or episode >= E:
        return float(cfg.eps_end)
    return float(cfg.eps_start + (cfg.eps_end - cfg.eps_start) * episode / E)
