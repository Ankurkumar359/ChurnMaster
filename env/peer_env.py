"""Gymnasium environment: one node learns how to manage its routing table under churn.

State  : (K + M) * 6 per-peer features + 3 global features   (default 147-dim, in [0, 1])
Action : Discrete(1 + K*M) -> 0 = no-op, 1 + i*M + j = evict slot i, insert candidate j
Reward : alpha*success - beta*latency - gamma*replacement - delta*dead_fraction
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import EnvConfig
from env.overlay import Overlay, ring_dist

N_FEATS = 6
AGE, RTT, SUCC, SINCE, FAIL, OFFS = range(N_FEATS)
N_GLOBAL = 3
RTT_NORM = 300.0


def split_obs(obs, k, m):
    obs = np.asarray(obs, dtype=np.float32)
    peers = obs[: (k + m) * N_FEATS].reshape(k + m, N_FEATS)
    return peers[:k], peers[k:], obs[(k + m) * N_FEATS:]


def encode_action(slot, cand, m):
    return 1 + slot * m + cand


class PeerSelectionEnv(gym.Env):
    def __init__(self, cfg: EnvConfig = None, seed=None):
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self.K, self.M = self.cfg.k, self.cfg.m
        self.obs_dim = (self.K + self.M) * N_FEATS + N_GLOBAL
        self.observation_space = spaces.Box(0.0, 1.0, (self.obs_dim,), np.float32)
        self.action_space = spaces.Discrete(1 + self.K * self.M)
        self._rng = np.random.default_rng(seed)
        self.net = None

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        churn = (options or {}).get("churn_scale", self.cfg.churn_scale)
        self.net = Overlay(self.cfg, self._rng)
        self.net.reset(churn)
        net, K = self.net, self.K
        alive_idx = np.nonzero(net.alive)[0]
        alive_idx = alive_idx[alive_idx != 0]
        self.nbr = self._rng.choice(alive_idx, K, replace=False)
        t = net.t
        self.adv_age = net.gossip_age(self.nbr).astype(float)
        self.adv_time = np.full(K, float(t))
        self.last_seen = np.full(K, float(t))
        self.fail = np.zeros(K)
        self.succ = np.full(K, 0.8)
        self.rtt_est = net.rtt(self.nbr)
        self.probe_ptr = 0
        self.step_count = 0
        self.g_succ, self.g_fail, self.g_repl = 1.0, 0.0, 0.0
        self._ep = dict(ret=0.0, succ=[], lat=[], repl=0, dead=[])
        self._sample_candidates()
        return self._obs(), {}

    # ------------------------------------------------------------- internals
    def _sample_candidates(self):
        mask = np.ones(self.cfg.n_nodes, dtype=bool)
        mask[0] = False
        mask[self.nbr] = False
        self.cand = self._rng.choice(np.nonzero(mask)[0], self.M, replace=False)
        self.cand_rtt = self.net.rtt(self.cand)

    def _install(self, slot, node):
        net = self.net
        self.nbr[slot] = node
        self.adv_age[slot] = net.gossip_age(node)
        self.adv_time[slot] = net.t
        self.last_seen[slot] = net.t
        self.fail[slot] = 0.0
        self.succ[slot] = 0.5                      # unknown reliability
        self.rtt_est[slot] = net.rtt(node)

    def _contact(self, slot, ok):
        net, node = self.net, self.nbr[slot]
        if ok:
            self.last_seen[slot] = net.t
            self.fail[slot] = 0.0
            self.succ[slot] += 0.2 * (1.0 - self.succ[slot])
            self.adv_age[slot] = net.gossip_age(node)
            self.adv_time[slot] = net.t
            self.rtt_est[slot] = 0.7 * self.rtt_est[slot] + 0.3 * float(net.rtt(node))
        else:
            self.fail[slot] += 1.0
            self.succ[slot] += 0.2 * (0.0 - self.succ[slot])

    def _lookup(self):
        """Greedy ring routing from the agent through its own table. Returns (ok, latency_ms, failures)."""
        cfg, net = self.cfg, self.net
        key = self._rng.random()
        d_self = ring_dist(net.ids[0], key)
        d = ring_dist(net.ids[self.nbr], key)
        prog = np.nonzero(d < d_self)[0]
        tail = cfg.tail_hops * cfg.tail_hop_ms
        if len(prog) == 0:                         # agent itself is closest to key
            return True, tail, 0
        order = prog[np.argsort(d[prog])][: cfg.max_tries]
        lat, fails = 0.0, 0
        for s in order:
            node = self.nbr[s]
            if net.alive[node] and self._rng.random() >= cfg.drop_prob:
                lat += float(net.rtt(node)) + tail
                self._contact(s, True)
                return True, lat, fails
            lat += cfg.timeout_ms
            self._contact(s, False)
            fails += 1
        return False, lat, fails

    def _obs(self):
        cfg, net, t = self.cfg, self.net, self.net.t
        age_n = self.adv_age + (t - self.adv_time)
        nb = np.stack([
            np.clip(np.log1p(age_n) / np.log1p(cfg.age_scale), 0, 1),
            np.clip(self.rtt_est / RTT_NORM, 0, 1),
            self.succ,
            np.clip((t - self.last_seen) / cfg.since_norm, 0, 1),
            np.clip(self.fail / cfg.fail_norm, 0, 1),
            (net.ids[self.nbr] - net.ids[0]) % 1.0,
        ], axis=1)
        c_age = net.gossip_age(self.cand)
        cd = np.stack([
            np.clip(np.log1p(c_age) / np.log1p(cfg.age_scale), 0, 1),
            np.clip(self.cand_rtt / RTT_NORM, 0, 1),
            np.full(self.M, 0.5),
            np.zeros(self.M),
            np.zeros(self.M),
            (net.ids[self.cand] - net.ids[0]) % 1.0,
        ], axis=1)
        peers = np.concatenate([nb, cd], axis=0)
        if cfg.ablate_latency:
            peers[:, RTT] = 0.0
        glob = np.array([self.g_succ, np.clip(self.g_fail, 0, 1), np.clip(self.g_repl, 0, 1)])
        return np.concatenate([peers.ravel(), glob]).astype(np.float32)

    # ------------------------------------------------------------------- step
    def step(self, action):
        cfg, net = self.cfg, self.net
        replaced = 0
        if action > 0:
            slot, cj = divmod(int(action) - 1, self.M)
            self._install(slot, self.cand[cj])
            replaced = 1
        net.advance()
        for _ in range(cfg.probes_per_step):
            s = self.probe_ptr
            self.probe_ptr = (s + 1) % self.K
            self._contact(s, bool(net.alive[self.nbr[s]]))

        n_ok, lat_sum, n_fail = 0, 0.0, 0
        for _ in range(cfg.lookups_per_step):
            ok, lat, f = self._lookup()
            n_ok += ok
            lat_sum += lat
            n_fail += f
        L = cfg.lookups_per_step
        success, latency = n_ok / L, lat_sum / L
        dead = float(np.mean(~net.alive[self.nbr]))
        reward = (cfg.alpha * success - cfg.beta * latency / cfg.lat_norm_ms
                  - cfg.gamma * replaced - cfg.delta * dead)

        self.g_succ += 0.1 * (success - self.g_succ)
        self.g_fail += 0.1 * (n_fail / L - self.g_fail)
        self.g_repl += 0.1 * (replaced - self.g_repl)
        self.step_count += 1
        self._sample_candidates()

        e = self._ep
        e["ret"] += reward
        e["succ"].append(success)
        e["lat"].append(latency)
        e["dead"].append(dead)
        e["repl"] += replaced
        truncated = self.step_count >= cfg.episode_len
        info = {"success": success, "latency_ms": latency, "dead_frac": dead}
        if truncated:
            info["episode"] = dict(
                ret=e["ret"], success=float(np.mean(e["succ"])), latency_ms=float(np.mean(e["lat"])),
                repl_per_step=e["repl"] / self.step_count, dead_frac=float(np.mean(e["dead"])))
        return self._obs(), float(reward), False, truncated, info
