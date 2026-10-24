"""Quick sanity check:  python -m tests.test_env"""
import numpy as np
from env.peer_env import PeerSelectionEnv


def test_env_runs():
    env = PeerSelectionEnv(seed=0)
    obs, _ = env.reset(seed=0, options={"churn_scale": 2.0})
    assert obs.shape == (env.obs_dim,) and obs.min() >= 0 and obs.max() <= 1
    for _ in range(env.cfg.episode_len):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert np.isfinite(r) and obs.min() >= 0 and obs.max() <= 1
    assert trunc and "episode" in info
    print("env OK", info["episode"])


if __name__ == "__main__":
    test_env_runs()
