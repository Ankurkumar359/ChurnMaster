"""Non-learning baselines. They see the same observation as the agent (fair comparison)."""
import numpy as np
from config import EnvConfig
from env.peer_env import split_obs, encode_action, AGE, RTT, FAIL


class Policy:
    name = "base"

    def __init__(self, cfg: EnvConfig = None, seed: int = 0):
        self.cfg = cfg or EnvConfig()
        self.K, self.M = self.cfg.k, self.cfg.m
        self.rng = np.random.default_rng(seed)

    def act(self, obs) -> int:
        raise NotImplementedError

    def _fails(self, nb):
        return nb[:, FAIL] * self.cfg.fail_norm


class StaticTable(Policy):
    """Never maintains the table."""
    name = "Static (no-op)"

    def act(self, obs):
        return 0


class RandomReplace(Policy):
    """With prob p replace a random slot by a random candidate."""
    name = "Random replace"

    def __init__(self, cfg=None, seed=0, p=0.1):
        super().__init__(cfg, seed)
        self.p = p

    def act(self, obs):
        if self.rng.random() > self.p:
            return 0
        return encode_action(self.rng.integers(self.K), self.rng.integers(self.M), self.M)


class KademliaStyle(Policy):
    """Keep-oldest heuristic: evict a failed peer, replace it with the oldest-uptime candidate."""
    name = "Kademlia-style (oldest)"

    def __init__(self, cfg=None, seed=0, thr=2):
        super().__init__(cfg, seed)
        self.thr = thr

    def act(self, obs):
        nb, cd, _ = split_obs(obs, self.K, self.M)
        f = self._fails(nb)
        if f.max() < self.thr:
            return 0
        return encode_action(int(f.argmax()), int(cd[:, AGE].argmax()), self.M)


class FailureThreshold(Policy):
    """Evict peers after `thr` failures, replace with the lowest-RTT candidate."""
    name = "Failure threshold"

    def __init__(self, cfg=None, seed=0, thr=2):
        super().__init__(cfg, seed)
        self.thr = thr

    def act(self, obs):
        nb, cd, _ = split_obs(obs, self.K, self.M)
        f = self._fails(nb)
        if f.max() < self.thr:
            return 0
        return encode_action(int(f.argmax()), int(cd[:, RTT].argmin()), self.M)


class LowestLatency(Policy):
    """Evict dead peers first; otherwise swap the slowest neighbour for a clearly faster candidate."""
    name = "Lowest latency"

    def __init__(self, cfg=None, seed=0, thr=2, margin=0.08):
        super().__init__(cfg, seed)
        self.thr, self.margin = thr, margin

    def act(self, obs):
        nb, cd, _ = split_obs(obs, self.K, self.M)
        f = self._fails(nb)
        best = int(cd[:, RTT].argmin())
        if f.max() >= self.thr:
            return encode_action(int(f.argmax()), best, self.M)
        worst = int(nb[:, RTT].argmax())
        if cd[best, RTT] + self.margin < nb[worst, RTT]:
            return encode_action(worst, best, self.M)
        return 0


ALL_BASELINES = [StaticTable, RandomReplace, KademliaStyle, FailureThreshold, LowestLatency]
