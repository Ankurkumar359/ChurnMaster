"""Evaluate trained agent(s) and baselines across churn levels (paired seeds)."""
import argparse, csv, os
import numpy as np

from config import EnvConfig
from env.peer_env import PeerSelectionEnv
from agents.dqn import DQNAgent
from baselines.policies import ALL_BASELINES


class AgentPolicy:
    def __init__(self, path):
        self.agent, ck = DQNAgent.from_checkpoint(path)
        self.cfg = EnvConfig(**ck["cfg"])
        self.name = ck["tag"]

    def act(self, obs):
        return self.agent.act(obs, eps=0.0)


def run(policy, cfg, churn, episodes, seed0):
    env = PeerSelectionEnv(cfg)
    rows = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep, options={"churn_scale": churn})
        while True:
            obs, r, term, trunc, info = env.step(policy.act(obs))
            if term or trunc:
                break
        rows.append(info["episode"])
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", action="append", default=["models/ddqn.pt"],
                   help="checkpoint(s); repeat flag to add the ablation model")
    p.add_argument("--churn", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0, 8.0])
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--out", type=str, default="results/eval_results.csv")
    args = p.parse_args()

    ckpts = [c for c in dict.fromkeys(args.ckpt) if os.path.exists(c)]
    policies = [(AgentPolicy(c), AgentPolicy(c).cfg) for c in ckpts]
    policies += [(B(EnvConfig(), seed=1), EnvConfig()) for B in ALL_BASELINES]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "churn", "return_mean", "return_std", "success_mean", "success_std",
                    "latency_ms", "repl_per_step", "dead_frac"])
        for pol, cfg in policies:
            for ch in args.churn:
                rows = run(pol, cfg, ch, args.episodes, args.seed)
                g = lambda k: np.array([r[k] for r in rows])
                w.writerow([pol.name, ch, f"{g('ret').mean():.4f}", f"{g('ret').std():.4f}",
                            f"{g('success').mean():.4f}", f"{g('success').std():.4f}",
                            f"{g('latency_ms').mean():.2f}", f"{g('repl_per_step').mean():.4f}",
                            f"{g('dead_frac').mean():.4f}"])
                print(f"{pol.name:28s} churn {ch:>4}: success {g('success').mean():.3f} "
                      f"ret {g('ret').mean():7.2f} repl/step {g('repl_per_step').mean():.3f}", flush=True)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
