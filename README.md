# ChurnMaster — RL for Peer Selection under Node Churn

A reinforcement-learning agent (Double DQN, written from scratch) learns **which peers to keep or replace in its
routing table** in a P2P/DHT overlay with unpredictable node churn. Goal: maximise lookup success while minimising
latency and maintenance overhead.


## RL formulation
| | |
|---|---|
| **State** | 147-dim, in [0,1]. For each of K=16 neighbours and M=8 gossip candidates: age (log-uptime), RTT, lookup-success EWMA, time since last seen, failure count, ID-ring offset (6 features) + 3 globals (recent success, failure rate, replacement rate) |
| **Action** | Discrete(1 + K·M = 129): `0` = no-op, `1 + i·M + j` = evict slot *i*, insert candidate *j* |
| **Reward** | `r = α·success − β·latency/1000ms − γ·1[replace] − δ·dead_fraction` |
| **Environment** | Custom simulator (`env/`): 500 nodes, Weibull(0.6) heavy-tailed sessions, 30% stable / 70% flaky nodes, 2-D latency embedding, greedy ring routing with timeouts, tunable `churn_scale` |
| **Algorithm** | Double DQN, replay buffer, soft target update, biased ε-greedy; churn level randomised per episode |
| **Baselines** | Static, Random replace, Kademlia-style (keep oldest), Failure-threshold, Lowest-latency |

## Structure
```
churnmaster/
├── config.py            # all simulator / reward parameters
├── env/overlay.py       # churn + latency + ID-ring simulator
├── env/peer_env.py      # Gymnasium env (state, action, reward)
├── agents/dqn.py        # Double DQN
├── baselines/policies.py
├── train.py  eval.py  plot_results.py  utils.py
├── tests/test_env.py
├── models/              # trained checkpoints
└── results/             # logs, CSVs, figures
```

## Install
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m tests.test_env                              # sanity check
```

## Reproduce (seed 0)
```bash
python train.py --episodes 400 --seed 0 --out models/ddqn.pt --log results/train_log.csv
python train.py --episodes 400 --seed 0 --ablate-latency --tag "DDQN no-latency (ablation)" \
       --out models/ddqn_nolat.pt --log results/train_log_nolat.csv
python eval.py --ckpt models/ddqn.pt --ckpt models/ddqn_nolat.pt --episodes 20
python plot_results.py
```
Training uses churn ∈ {1, 2, 4}; evaluation also tests unseen churn 0.5 and 8 (generalisation).
Figures: `results/fig_training.png`, `fig_success_vs_churn.png`, `fig_overhead.png`.
