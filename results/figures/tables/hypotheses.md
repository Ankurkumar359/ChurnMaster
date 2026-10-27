# Hypotheses H1–H5

| id | statement | verdict | evidence |
| --- | --- | --- | --- |
| H1 | RL > matched_random on J in every scenario (Holm-corrected) | pass | stable: dJ=+0.01495 [+0.006086,+0.02952], p_holm=3.81e-05; high_churn: dJ=+0.01272 [+0.01005,+0.01601], p_holm=1.72e-05; heldout_medium: dJ=+0.01258 [+0.005293,+0.02352], p_holm=1.91e-05 |
| H2 | high_churn: RL non-inferior (within 1 pp success) to kademlia_epoch and tuned_heuristic, and significantly better on >=1 primary metric | inconclusive | high_churn: kademlia_epoch: dsucc=+0.00127 [-0.0003695,+0.003451], better on pings_per_hour (pass); tuned_heuristic: dsucc=-0.0005323 [-0.001628,+0.0004703], better on none (inconclusive) |
| H3 | RL pings/h <= kademlia_epoch and p95_all within 5% of the best baseline, in every scenario | fail | stable: dpings=-376.8 [-385.2,-368.9], p95 RL=456.6 vs best heuristic=362 (ratio 1.261) (fail); high_churn: dpings=-294.2 [-308.7,-279.4], p95 RL=553.8 vs best tuned_heuristic=468.4 (ratio 1.182) (fail); heldout_medium: dpings=-346.1 [-362.6,-330.4], p95 RL=521.3 vs best tuned_heuristic=378.7 (ratio 1.377) (fail) |
| H4 | H2 holds on heldout_medium | inconclusive | heldout_medium: kademlia_epoch: dsucc=-9.707e-06 [-0.0003081,+0.0002564], better on pings_per_hour (pass); tuned_heuristic: dsucc=-1.294e-05 [-0.000211,+0.0001559], better on none (inconclusive) |
| H5 | all 3 RL runs individually in H2's direction (high_churn) | pass | vs kademlia_epoch: pooled dJ=+0.008532, runs rl_run0=+0.008234,rl_run1=+0.006856,rl_run2=+0.01051; vs tuned_heuristic: pooled dJ=-0.003022, runs rl_run0=-0.003321,rl_run1=-0.004699,rl_run2=-0.001048 |
