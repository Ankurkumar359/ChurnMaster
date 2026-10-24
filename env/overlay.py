"""Discrete-time P2P overlay with node churn. Node 0 is the learning agent (never churns)."""
import numpy as np
from config import EnvConfig


def ring_dist(a, b):
    d = np.abs(a - b)
    return np.minimum(d, 1.0 - d)


class Overlay:
    def __init__(self, cfg: EnvConfig, rng: np.random.Generator):
        self.cfg, self.rng = cfg, rng
        n = cfg.n_nodes
        self.coords = rng.random((n, 2))          # latency embedding
        self.ids = rng.random(n)                  # position on the [0,1) ID ring
        self.stable = rng.random(n) < cfg.stable_frac
        self.alive = np.ones(n, dtype=bool)
        self.session_start = np.zeros(n)
        self.last_end = np.zeros(n)
        self.next_change = np.full(n, np.inf)
        self.t = 0
        self.churn_scale = cfg.churn_scale

    # ---- churn ----
    def _session(self, idx):
        c = self.cfg
        base = np.where(self.stable[idx], c.stable_session, c.flaky_session) / self.churn_scale
        return np.maximum(1.0, self.rng.weibull(c.weibull_shape, size=len(idx)) * base)

    def _down(self, idx):
        return np.maximum(1.0, self.rng.exponential(self.cfg.down_mean, size=len(idx)))

    def reset(self, churn_scale: float):
        c, n = self.cfg, self.cfg.n_nodes
        self.churn_scale = churn_scale
        idx = np.arange(n)
        self.alive = self.rng.random(n) < 0.7
        self.alive[0] = True
        self.session_start = -self.rng.random(n) * 300.0
        self.last_end = self.session_start.copy()
        self.next_change = np.where(self.alive, self._session(idx), self._down(idx))
        self.next_change[0] = np.inf             # the agent node never leaves
        self.t = 0
        for _ in range(c.warmup):
            self.advance()

    def advance(self):
        self.t += 1
        due = np.nonzero(self.next_change <= self.t)[0]
        if len(due) == 0:
            return
        was_alive = self.alive[due]
        dying, rising = due[was_alive], due[~was_alive]
        if len(dying):
            self.alive[dying] = False
            self.last_end[dying] = self.t
            self.next_change[dying] = self.t + self._down(dying)
        if len(rising):
            self.alive[rising] = True
            self.session_start[rising] = self.t
            self.next_change[rising] = self.t + self._session(rising)

    # ---- helpers ----
    def rtt(self, nodes, noisy=True):
        nodes = np.asarray(nodes)
        d = np.linalg.norm(self.coords[nodes] - self.coords[0], axis=-1)
        r = self.cfg.base_rtt_ms + self.cfg.rtt_per_unit_ms * d
        if noisy:
            r = r + self.rng.normal(0.0, self.cfg.rtt_noise_ms, size=np.shape(r))
        return np.maximum(1.0, r)

    def gossip_age(self, nodes):
        """Uptime a peer advertises: current session age if alive, stale last-session length if dead."""
        nodes = np.asarray(nodes)
        return np.where(self.alive[nodes],
                        self.t - self.session_start[nodes],
                        self.last_end[nodes] - self.session_start[nodes])
