"""Train the Double-DQN agent. Churn level is randomised per episode (domain randomisation)."""
import argparse, csv, os, time
from dataclasses import asdict
import numpy as np

from config import EnvConfig
from env.peer_env import PeerSelectionEnv
from agents.dqn import DQNAgent
from utils import set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", type=str, default="DDQN (ours)")
    p.add_argument("--out", type=str, default="models/ddqn.pt")
    p.add_argument("--log", type=str, default="results/train_log.csv")
    p.add_argument("--train-churn", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    p.add_argument("--ablate-latency", action="store_true", help="hide RTT features (ablation)")
    p.add_argument("--learn-start", type=int, default=2000)
    p.add_argument("--update-every", type=int, default=2)
    p.add_argument("--eps-frac", type=float, default=0.2, help="fraction of steps over which eps decays")
    args = p.parse_args()

    set_seed(args.seed)
    cfg = EnvConfig(ablate_latency=args.ablate_latency)
    env = PeerSelectionEnv(cfg)
    agent = DQNAgent(env.obs_dim, env.action_space.n)
    rng = np.random.default_rng(args.seed)
    total = args.episodes * cfg.episode_len
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)

    steps, t0 = 0, time.time()
    with open(args.log, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "churn", "return", "success", "latency_ms", "repl_per_step", "dead_frac", "eps"])
        for ep in range(args.episodes):
            churn = float(rng.choice(args.train_churn))
            obs, _ = env.reset(seed=args.seed * 100_000 + ep, options={"churn_scale": churn})
            while True:
                eps = max(0.05, 1.0 - steps / (args.eps_frac * total))
                a = agent.act(obs, eps)
                nobs, r, term, trunc, info = env.step(a)
                agent.buffer.add(obs, a, r, nobs, float(term))   # time-limit truncation is not terminal
                obs = nobs
                steps += 1
                if steps > args.learn_start and steps % args.update_every == 0:
                    agent.update()
                if term or trunc:
                    break
            e = info["episode"]
            w.writerow([ep, churn, f"{e['ret']:.4f}", f"{e['success']:.4f}", f"{e['latency_ms']:.2f}",
                        f"{e['repl_per_step']:.4f}", f"{e['dead_frac']:.4f}", f"{eps:.3f}"])
            f.flush()
            if ep % 10 == 0:
                print(f"ep {ep:4d} churn {churn:.1f} ret {e['ret']:7.2f} succ {e['success']:.3f} "
                      f"repl/step {e['repl_per_step']:.3f} eps {eps:.2f} [{time.time() - t0:.0f}s]", flush=True)
    agent.save(args.out, asdict(cfg), args.tag)
    print("saved", args.out)


if __name__ == "__main__":
    main()
