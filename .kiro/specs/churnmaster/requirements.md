# Requirements Document

## Introduction

ChurnMaster is a research codebase that evaluates tabular Q-learning for adaptive peer replacement in the routing table of one focal node F of a simulated Kademlia-like DHT under node churn. These requirements are derived from the approved design (ChurnMaster Revised Implementation Plan v2). They cover the deterministic simulator, the observable/hidden boundary, the state/action/reward formulation, the Q-learning agent, baselines, experiment protocols, statistics, and reproducibility. Implementation acceptance criteria (I1–I6) are mandatory. Experimental hypotheses (H1–H5) are tested and reported but need not hold.

## Glossary

- **ChurnMaster**: The complete system described by this document.
- **F**: The focal node; always online; the only node that learns and is evaluated.
- **Config_Loader**: Component (`config.py`) that loads, merges, validates, and hashes TOML configuration.
- **Seeder**: Component (`seeding.py`) that derives per-split, per-run, per-episode seeds.
- **IdSpace**: Component (`sim/idspace.py`) providing XOR distance, bucket index, and closest-node selection.
- **Keyed_RNG**: Component (`sim/keyed_rng.py`) providing `keyed_uniform` and `keyed_normal` based on splitmix64.
- **Churn_Generator**: Component (`sim/churn.py`) producing the session/downtime trace.
- **Workload_Generator**: Component (`sim/workload.py`) producing lookup requests.
- **Network**: Ground-truth component (`sim/network.py`) holding hidden node state, RTT, loss, and replica holders.
- **Background_Tables**: Component (`sim/background.py`) holding the stale routing tables of non-focal nodes.
- **Local_View**: F's observable routing table and per-bucket candidate cache (`sim/routing_table.py`).
- **Lookup_Engine**: Component (`sim/lookup.py`) executing atomic iterative lookups.
- **Env**: Event-driven episode loop (`sim/env.py`).
- **Maintenance**: Component (`sim/maintenance.py`) applying KEEP/PROBE/REPLACE and the kademlia_std mode.
- **Observation_Builder**: Component (`sim/observation.py`) producing a DecisionContext.
- **DecisionContext**: Frozen dataclass of observable features for one bucket at one epoch.
- **Discretizer**: Component (`rl/state.py`) mapping a DecisionContext to one of 648 state indices and the allowed actions.
- **Ledger**: Component (`rl/ledger.py`) that accepts only ObservedLookup and DecisionRecord and produces rewards, transitions, and J.
- **Q_Agent**: Tabular Q-learning agent (`rl/qlearning.py`).
- **Policy**: Any decision function from DecisionContext to action (`policies/`).
- **Heuristic**: Bin-aligned rule policy family of 32 configurations (`policies/heuristic.py`).
- **Oracle_Heuristic**: Ground-truth-using reference policy (`policies/oracle.py`).
- **Metrics**: Component (`metrics.py`) computing per-episode metrics and ground-truth diagnostics.
- **Runner**: Experiment job executor (`experiments/runner.py`).
- **Experiments**: Modules under `experiments/` implementing headroom, diagnostics, training, tuning, evaluation, statistics, and reproduction.
- **CLI**: Command-line entry point (`cli.py`).
- **Import_Guard**: Test that statically checks module imports.
- **Bucket epoch**: The interval [t_k, t_{k+1}) of 30 s for one bucket b.
- **Incumbent**: The least-recently-seen (LRS) entry of a bucket.
- **Candidate**: The most recently seen candidate-cache entry of a bucket.
- **J**: Score `1 − mean(ℓ) − (w_M·pings + w_R·evictions)/N_lookups` over the measured window.
- **Measured window**: Episode time [600 s, 3000 s).
- **pp**: Percentage points.

## Requirements

### Requirement 1: Configuration

**User Story:** As a researcher, I want layered, validated, hashable configs, so that every run is fully specified and traceable.

#### Acceptance Criteria

1. THE Config_Loader SHALL parse TOML files with `tomllib` and deep-merge them in the order base → scenario → experiment, with later values overriding earlier values.
2. THE Config_Loader SHALL represent the merged configuration as frozen dataclasses.
3. IF a configuration contains an unknown key, a value of the wrong type, or an out-of-range value, THEN THE Config_Loader SHALL raise a ConfigError that names the offending key.
4. THE Config_Loader SHALL compute `config_hash` as the SHA-256 of a canonical JSON serialization of the merged configuration.
5. THE Config_Loader SHALL provide defaults of 128 nodes, 8-bit IDs, k=4, α=2, 3 replicas, cache size 4, cache TTL 300 s, rebuild period 300 s, epoch 30 s, warm-up 600 s, measured 2400 s, lookup rate 1/s, p_loss 0.02, timeout 500 ms, deadline 2000 ms, 8 max rounds, w_L=0.3, w_M=0.2, w_R=0.3, and γ=0.9.
6. THE ChurnMaster SHALL ship scenario configs for stable (0.7, 4 h, 30 min, 20 min), high_churn (0.3, 1 h, 5 min, 10 min), heldout_medium (0.45, 2 h, 10 min, 12 min), and bursty (0.5, 2 h, 15 min, 15 min), given as stable share, stable mean session, flaky mean session, and mean downtime.
7. WHEN the CLI `show-config` command is invoked, THE CLI SHALL print the merged configuration and its config_hash.

### Requirement 2: Seeding and Data Splits

**User Story:** As a researcher, I want disjoint, reproducible seed splits, so that calibration, training, validation, and evaluation never share environments.

#### Acceptance Criteria

1. THE Seeder SHALL derive each episode seed as `SeedSequence(root, spawn_key=(split_id, run, episode))`.
2. THE Seeder SHALL define distinct split identifiers for calibration, train, validation, eval, agent, and test.
3. THE Seeder SHALL produce pairwise-disjoint seed sets for distinct (split_id, run, episode) triples.
4. THE Seeder SHALL derive eval environment seeds independently of the RL run index, so that all policies and runs see identical eval environments.
5. THE Seeder SHALL give agent exploration and random baselines numpy generators that are separate from all environment streams.

### Requirement 3: ID Space

**User Story:** As a simulator developer, I want correct XOR geometry, so that routing behaves like Kademlia.

#### Acceptance Criteria

1. THE IdSpace SHALL draw 128 unique node IDs from the 256 8-bit values.
2. THE IdSpace SHALL compute distance as `a XOR b`.
3. THE IdSpace SHALL compute the bucket of node x relative to F as `floor(log2(F XOR x))`, for x ≠ F, yielding a value in [0, 7].
4. WHEN a lookup key equals F, THE IdSpace SHALL attribute the key to bucket 0.
5. THE IdSpace SHALL return the n closest IDs to a target ordered by increasing XOR distance.

### Requirement 4: Keyed Randomness

**User Story:** As a researcher, I want policy-independent randomness for network outcomes, so that policy comparisons share identical luck.

#### Acceptance Criteria

1. THE Keyed_RNG SHALL implement `keyed_uniform(root, purpose, *ids)` in pure Python using splitmix64, returning a value in [0, 1).
2. THE Keyed_RNG SHALL implement `keyed_normal` from two keyed uniforms derived from the same key.
3. THE Keyed_RNG SHALL return identical outputs for identical (root, purpose, ids) arguments.
4. THE Keyed_RNG SHALL produce outputs for 10,000 distinct keys that pass a Kolmogorov–Smirnov uniformity test at significance level 0.01.
5. THE ChurnMaster SHALL use the key schemas (LOSS|JITTER, lookup_id, dst) for lookup queries, (LOSS|JITTER, epoch_idx, bucket, dst, kind) for epoch pings, (LOSS|JITTER, lookup_id, bucket, dst, STD) for kademlia_std pings, (BG, node, rebuild_idx) for background rebuilds, and (BOOT, bucket, slot) for F's bootstrap.

### Requirement 5: Churn

**User Story:** As a researcher, I want realistic heavy-tailed churn, so that peer replacement matters.

#### Acceptance Criteria

1. THE Churn_Generator SHALL assign each non-focal node a hidden type stable or flaky according to the scenario stable share.
2. THE Churn_Generator SHALL generate alternating Weibull(shape 0.6) sessions with the type's mean session length and exponential downtimes with the scenario mean downtime, starting from a −2 h burn-in.
3. THE Churn_Generator SHALL produce non-overlapping, time-ordered sessions per node.
4. THE Churn_Generator SHALL keep F online for the entire episode.
5. THE Churn_Generator SHALL produce a mean online fraction within 3 pp of the analytic stationary fraction `share·m_s/(m_s+d) + (1−share)·m_f/(m_f+d)`.
6. THE Churn_Generator SHALL produce traces in which nodes with longer elapsed session age remain online through the next 10 min more often than nodes with shorter elapsed session age.
7. THE Churn_Generator SHALL produce an identical trace and trace hash for an identical episode seed.

### Requirement 6: Workload

**User Story:** As a researcher, I want a fixed pre-generated workload, so that all policies face the same lookups.

#### Acceptance Criteria

1. THE Workload_Generator SHALL pre-generate Poisson lookup arrivals at 1 lookup/s with integer ms timestamps and uniform 8-bit keys.
2. THE Workload_Generator SHALL assign sequential lookup_id values starting at 0.

### Requirement 7: Network Ground Truth

**User Story:** As a simulator developer, I want hidden, keyed network behavior, so that outcomes are realistic and policy-independent.

#### Acceptance Criteria

1. THE Network SHALL assign each node a hidden position in the unit square.
2. THE Network SHALL mark 10% of nodes as congestion-prone, with congestion periods of about 2 min occurring about every 20 min, during which RTT is multiplied by 3.
3. THE Network SHALL compute RTT as `(10 + 240·dist/√2)·exp(0.25·z)·(3 if congested)` ms, where dist is the Euclidean distance between F and the node and z is a keyed standard normal.
4. THE Network SHALL report a query as successful if and only if the destination is online at the query time and the keyed loss draw is at least p_loss.
5. THE Network SHALL produce an empirical loss rate within 0.5 pp of 2% for p_loss=0.02 over 10,000 queries to online nodes.
6. THE Network SHALL define the replica holders of key K at time t as the 3 online nodes closest to K, excluding F.

### Requirement 8: Background Routing Tables

**User Story:** As a simulator developer, I want stale background tables, so that responders can return dead contacts as in real DHTs.

#### Acceptance Criteria

1. THE Background_Tables SHALL rebuild each non-focal node's table every 300 s, staggered at `node_idx·300/N` s.
2. WHEN a node's table is rebuilt, THE Background_Tables SHALL fill each bucket with up to 4 online nodes chosen by the keyed order (BG, node, rebuild_idx).
3. WHILE between rebuilds, THE Background_Tables SHALL keep each table unchanged.
4. WHEN asked for contacts, THE Background_Tables SHALL return the 4 closest known contacts to the target, regardless of their online status.

### Requirement 9: F's Local View

**User Story:** As a learner, I want an observable routing table and candidate cache, so that decisions use only realistic information.

#### Acceptance Criteria

1. THE Local_View SHALL hold 8 buckets of at most k=4 entries, each with consecutive failures, last successful response time, insertion time, and smoothed RTT (EWMA β=0.3).
2. THE Local_View SHALL hold a candidate cache of at most 4 entries per bucket, each with last-seen time and last RTT, and SHALL expire entries older than 300 s.
3. WHEN F bootstraps at t=0, THE Local_View SHALL fill each bucket with online nodes selected by the keyed order (BOOT, bucket, slot).
4. WHEN a responder not in the table is offered and its bucket has an empty slot, THE Local_View SHALL insert the responder.
5. WHEN a responder not in the table is offered and its bucket is full, THE Local_View SHALL add or refresh the responder in that bucket's candidate cache.
6. THE Local_View SHALL select the incumbent as the entry with the oldest last successful response, using insertion time for never-contacted entries and breaking ties by earlier insertion.
7. THE Local_View SHALL select the candidate as the cache entry with the most recent last-seen time, breaking ties by lower last RTT, or none if the cache is empty.
8. THE Local_View SHALL track the fraction of F's queries that timed out in the last 120 s.

### Requirement 10: Atomic Lookup

**User Story:** As a researcher, I want a well-defined iterative lookup, so that success and latency reflect routing-table quality.

#### Acceptance Criteria

1. WHEN a lookup for key K starts, THE Lookup_Engine SHALL initialize the shortlist with F's 4 closest entries to K without filtering by liveness.
2. WHILE a lookup is in progress, THE Lookup_Engine SHALL query up to 2 unqueried shortlist nodes per round, with each query costing the sampled RTT on success and 500 ms on timeout, and the round lasting as long as its slowest query.
3. WHEN a queried node responds, THE Lookup_Engine SHALL merge the responder's 4 closest known contacts into the shortlist and keep the 4 closest unqueried nodes.
4. WHEN a replica holder responds, THE Lookup_Engine SHALL end the lookup as successful with latency equal to the sum of prior round durations plus the holder's RTT.
5. IF elapsed time exceeds 2000 ms, more than 8 rounds would be needed, or the shortlist has no unqueried nodes, THEN THE Lookup_Engine SHALL end the lookup as failed with latency equal to the give-up time.
6. THE Lookup_Engine SHALL evaluate all queries against network state at the lookup start time without advancing the simulation clock.
7. WHEN a lookup queries an entry of F's table, THE Lookup_Engine SHALL update that entry's statistics with the query outcome.
8. WHEN a lookup completes, THE Lookup_Engine SHALL offer every responder to F's Local_View.
9. THE Lookup_Engine SHALL attribute each lookup to bucket `b(K)`.
10. WHILE no node churns, THE Lookup_Engine SHALL succeed on all 1000 of 1000 lookups with a mean of at most 3 rounds when loss is disabled.

### Requirement 11: Event Loop and Time Model

**User Story:** As a researcher, I want a deterministic event-driven episode, so that results are exactly reproducible.

#### Acceptance Criteria

1. THE Env SHALL process events in order of integer ms timestamp and, at equal timestamps, in the order churn → rebuild → maintenance → lookup.
2. THE Env SHALL run each episode as a 600 s warm-up followed by a 2400 s measured window and end on the time limit.
3. THE Env SHALL exclude warm-up lookups and decisions from all metrics and from J.
4. THE Env SHALL support a maintenance-off mode in which no epoch actions are taken.
5. WHEN two episodes run with the same config and seed, THE Env SHALL produce identical metric summary hashes.
6. WHEN the maintenance-off mode is run, THE Env SHALL complete a 3000 s episode in at most 2 s of CPU time on the reference machine.

### Requirement 12: Observation Boundary and Leakage Prevention

**User Story:** As a researcher, I want hidden state isolated from learners, so that results reflect a realistic POMDP.

#### Acceptance Criteria

1. THE Observation_Builder SHALL produce a frozen DecisionContext containing only the incumbent's consecutive failures, time since last response, time since insertion, smoothed RTT, the candidate's time since seen and last RTT, F's 120 s timeout fraction, the bucket index, epoch index, and time.
2. THE Import_Guard SHALL verify that `rl/*`, `policies/*` except `oracle.py`, `sim/routing_table.py`, and `sim/observation.py` import none of `sim.network`, `sim.churn`, `sim.background`, `sim.lookup`, `sim.env`, `sim.keyed_rng`.
3. THE Import_Guard SHALL verify that `sim/*` imports nothing from `rl/` or `policies/`.
4. WHEN hidden state changes while the DecisionContext stays identical, THE ChurnMaster SHALL produce the same action from every non-oracle Policy.
5. IF the Ledger receives an object other than ObservedLookup or DecisionRecord, THEN THE Ledger SHALL raise a TypeError.
6. THE Env SHALL deliver ground-truth records only to Metrics and, through a dedicated oracle interface, to Oracle_Heuristic.

### Requirement 13: State Discretization and Action Masking

**User Story:** As a learner, I want a compact discrete state, so that tabular Q-learning is feasible.

#### Acceptance Criteria

1. THE Discretizer SHALL map a DecisionContext to 7 features: tier (top b=7, second b=6, other), inc_fail (0, 1, ≥2), inc_stale (<60 s, ≥60 s), inc_age (<2 min, 2–20 min, ≥20 min), cand (none, fresh <30 s, stale 30–300 s), cand_faster, and churn (<10%, ≥10% timeout fraction).
2. THE Discretizer SHALL set cand_faster to yes only when a candidate exists, the incumbent smoothed RTT is known, and the candidate last RTT is less than 0.7 times the incumbent smoothed RTT.
3. THE Discretizer SHALL encode features into a state index in [0, 648) and decode the index back to the same features.
4. THE Discretizer SHALL produce only the 540 reachable states.
5. THE Discretizer SHALL allow KEEP and PROBE in every state and allow REPLACE only in states with a candidate.

### Requirement 14: Maintenance Actions

**User Story:** As a learner, I want well-defined maintenance actions with costs, so that the trade-off between freshness and messages is explicit.

#### Acceptance Criteria

1. WHEN an epoch starts, THE Maintenance SHALL fill empty slots of each bucket from that bucket's candidate cache before decisions, at a cost of 0 messages.
2. WHEN an epoch starts, THE Maintenance SHALL request exactly one action per non-empty bucket from the active Policy.
3. WHEN KEEP is applied, THE Maintenance SHALL send 0 pings and leave the bucket unchanged.
4. WHEN PROBE is applied and the incumbent answers, THE Maintenance SHALL refresh the incumbent at a cost of 1 ping.
5. WHEN PROBE is applied and the incumbent does not answer, THE Maintenance SHALL evict the incumbent and, if a candidate exists, ping the candidate and insert the candidate if it answers or drop the candidate if it does not.
6. WHEN REPLACE is applied and the candidate answers, THE Maintenance SHALL evict the incumbent and insert the candidate at a cost of 1 ping and 1 eviction.
7. WHEN REPLACE is applied and the candidate does not answer, THE Maintenance SHALL drop the candidate and keep the incumbent at a cost of 1 ping.
8. IF a Policy returns an action not allowed in the current state, THEN THE Env SHALL raise an IllegalActionError.
9. THE Maintenance SHALL evaluate pings atomically against network state at the epoch time using the epoch ping key schema.
10. THE ChurnMaster SHALL add candidates to the cache only from responders of F's lookups.

### Requirement 15: Reward, Ledger, and Score

**User Story:** As a researcher, I want a reward whose sum equals the evaluation score, so that the agent optimizes exactly what is reported.

#### Acceptance Criteria

1. THE Ledger SHALL compute per-lookup loss `ℓ_j = (1 − S_j) + w_L·min(L_j/2000, 1)`, a value in [0, 1 + w_L].
2. THE Ledger SHALL compute action cost `c(a) = w_M·pings + w_R·evictions`.
3. THE Ledger SHALL compute bucket epoch reward `r_{b,k} = −Σ ℓ_j − c(a_{b,k})`, summing over lookups attributed to bucket b in [t_k, t_{k+1}), with a lookup term of 0 when no lookup occurred.
4. THE Ledger SHALL compute J such that `J = 1 − mean(ℓ) − (w_M·pings + w_R·evictions)/N_lookups = 1 + Σ r/N_lookups` over the measured window, with agreement within 1e-9.
5. THE Ledger SHALL include kademlia_std ping and eviction costs in J.
6. THE Ledger SHALL emit transitions (s_{b,k}, a_{b,k}, r_{b,k}, s_{b,k+1}) with τ=1 for each bucket that has a decision at both epoch k and epoch k+1.
7. WHEN an episode ends or a bucket becomes empty, THE Ledger SHALL drop the open transition for that bucket.

### Requirement 16: Q-Learning Agent

**User Story:** As a researcher, I want a standard tabular Q-learner, so that results are interpretable.

#### Acceptance Criteria

1. THE Q_Agent SHALL maintain one shared Q-table of shape 648×3 initialized to 0.
2. WHEN a transition is received, THE Q_Agent SHALL update `Q(s,a) ← Q(s,a) + α_n[r + γ·max_{a'∈A(s')} Q(s',a') − Q(s,a)]` with `α_n = max(α_min, (1+n(s,a))^−0.7)` and α_min defaulting to 0.05.
3. WHILE training, THE Q_Agent SHALL select actions ε-greedily over allowed actions only, with ε decaying linearly from 1.0 to 0.05 over the first 60% of episodes and staying at 0.05 afterwards.
4. WHILE evaluating, THE Q_Agent SHALL select the greedy allowed action with ε=0, breaking ties in the fixed order KEEP, PROBE, REPLACE.
5. THE Q_Agent SHALL save and load Q-tables as `.npy` files, with a loaded table equal to the saved table.
6. THE Q_Agent SHALL learn during warm-up and measured periods of training episodes.
7. WHEN trained on a toy 2-state MDP, THE Q_Agent SHALL converge to within 0.05 of the analytic Q-values.

### Requirement 17: Baseline Policies

**User Story:** As a researcher, I want strong and fair baselines, so that RL gains are meaningful.

#### Acceptance Criteria

1. THE ChurnMaster SHALL provide never_replace (always KEEP), always_replace (REPLACE if candidate else KEEP), uniform_random (uniform over allowed actions), and kademlia_epoch (PROBE if candidate else KEEP).
2. THE ChurnMaster SHALL provide matched_random, which samples actions from the RL policy's per-scenario action frequencies split by candidate presence, measured on validation seeds.
3. WHERE kademlia_std mode is selected, THE Maintenance SHALL, after each lookup, ping the LRS incumbent of each full bucket to which a responder was offered, evicting the incumbent and inserting the newcomer on failure, or refreshing the incumbent and caching the newcomer on success.
4. WHERE kademlia_std mode is selected, THE Env SHALL apply KEEP at every epoch decision and THE ChurnMaster SHALL label kademlia_std results as not sharing decision points.
5. THE ChurnMaster SHALL give uniform_random and matched_random seeded numpy generators separate from environment streams.

### Requirement 18: Heuristic Family

**User Story:** As a researcher, I want a tuned rule-based competitor in the same state space, so that the comparison isolates learning.

#### Acceptance Criteria

1. THE Heuristic SHALL apply rules in order: (1) if inc_fail ≥ f, REPLACE if candidate else PROBE, except that with protect_old an incumbent aged ≥20 min with fewer than 2 failures gets PROBE; (2) with rtt_swap, REPLACE if the candidate is faster, fresh, and the incumbent is aged <2 min; (3) with probe_stale, PROBE if inc_stale ≥60 s and a candidate exists; (4) otherwise KEEP.
2. WHERE tier_gate is top-two, THE Heuristic SHALL apply rules 2 and 3 only to top and second tier buckets.
3. THE Heuristic SHALL define exactly 32 configurations from f ∈ {1, 2}, protect_old, rtt_swap, probe_stale, and tier_gate ∈ {all, top-two}.
4. THE Heuristic SHALL, for every configuration, expose a 540-state action table that equals the rule function on 10,000 sampled DecisionContexts.

### Requirement 19: Oracle Heuristic

**User Story:** As a researcher, I want a ground-truth reference, so that I can estimate available headroom.

#### Acceptance Criteria

1. THE Oracle_Heuristic SHALL choose REPLACE when the incumbent is offline and the candidate is online, PROBE when the incumbent is offline and no online candidate exists, and KEEP otherwise.
2. THE ChurnMaster SHALL label Oracle_Heuristic as a headroom reference and not as an upper bound.

### Requirement 20: Metrics

**User Story:** As a researcher, I want consistent primary and secondary metrics, so that policies are compared on the same scale.

#### Acceptance Criteria

1. THE Metrics SHALL compute per episode success rate, p95 latency of successful lookups, p95 latency of all lookups with failures at give-up time, and maintenance pings per hour.
2. THE Metrics SHALL compute per episode mean latency, evictions per hour, messages per lookup, and J.
3. THE Metrics SHALL compute ground-truth diagnostics of dead-entry fraction over time and action counts by true incumbent state.
4. THE Metrics SHALL compute J identical to the Ledger score within 1e-9.

### Requirement 21: Determinism and Fairness

**User Story:** As a researcher, I want all policies evaluated under identical conditions, so that differences come only from decisions.

#### Acceptance Criteria

1. WHEN different policies run on the same episode seed, THE Env SHALL produce identical churn trace hashes and identical workloads.
2. WHEN different policies issue a query with the same key tuple to the same destination, THE Network SHALL return the same loss and jitter outcome.
3. THE Runner SHALL produce identical results with multiprocessing as with sequential execution.
4. THE Env SHALL present every epoch-mode policy with decision points at the same epoch times for each non-empty bucket.

### Requirement 22: Headroom Gate and Environment Freeze

**User Story:** As a researcher, I want to verify that the environment rewards good maintenance before training, so that experiments are worth running.

#### Acceptance Criteria

1. THE Experiments SHALL run never_replace, kademlia_epoch, always_replace, and oracle_heuristic on 10 calibration seeds for stable and high_churn.
2. THE Experiments SHALL evaluate gate G1 as passed when the best of kademlia_epoch, always_replace, and oracle_heuristic beats never_replace by at least 5 pp success in high_churn with a paired CI excluding 0.
3. THE Experiments SHALL evaluate gate G2 as passed when oracle_heuristic exceeds kademlia_epoch by at least 1.5 pp success, or uses at most 60% of kademlia_epoch's pings with success within 0.5 pp.
4. IF G1 fails, THEN THE Experiments SHALL tune deadline in {1000, 1500, 2000} ms, loss in {1, 2, 5}%, and replicas in {1, 2, 3} on calibration seeds only.
5. IF G2 fails, THEN THE Experiments SHALL proceed and report limited headroom.
6. WHEN the gate completes, THE Experiments SHALL record the frozen env config hash in the manifest.
7. IF a later training or evaluation config hash differs from the frozen env hash, THEN THE Experiments SHALL refuse to run.

### Requirement 23: Reward Diagnostics

**User Story:** As a researcher, I want to sanity-check the reward and state coverage, so that learning failures are diagnosable.

#### Acceptance Criteria

1. THE Experiments SHALL verify that the reward-total identity holds exactly on recorded episodes.
2. THE Experiments SHALL report whether, for truly dead incumbents in top and second tiers, PROBE and REPLACE discounted returns exceed KEEP, and whether, for live incumbents, KEEP exceeds REPLACE, with CIs.
3. THE Experiments SHALL report whether at least 80% of decisions fall in states visited at least 50 times over 20 episodes.
4. THE Experiments SHALL report the Spearman correlation between J and success rate.

### Requirement 24: Training Protocol

**User Story:** As a researcher, I want a fixed training protocol, so that learning results are comparable and reproducible.

#### Acceptance Criteria

1. THE Experiments SHALL train for 200 episodes per run, alternating stable and high_churn scenarios by episode, and use the final Q-table for evaluation.
2. THE Experiments SHALL run a high_churn debug training of 1 run × 100 episodes.
3. THE Experiments SHALL run a grid of γ ∈ {0.8, 0.9} × α_min ∈ {0.02, 0.05} and select the setting with the highest mean validation J over the last 3 checkpoints.
4. THE Experiments SHALL train 3 independent runs with the selected setting.
5. THE Experiments SHALL validate every 20 episodes, including episode 0, on 5 validation seeds per training scenario, and save `.npy` Q snapshots and `.csv` learning curves.
6. THE Experiments SHALL report, for each of the 3 runs, whether the final policy beats its episode-0 checkpoint on validation J with a paired CI excluding 0, and whether the mean |ΔQ| per episode shrinks.

### Requirement 25: Heuristic Tuning, Evaluation, and Statistics

**User Story:** As a researcher, I want a locked evaluation with corrected statistics, so that conclusions are trustworthy.

#### Acceptance Criteria

1. THE Experiments SHALL evaluate all 32 heuristic configurations on 10 train seeds per scenario and select a global best by mean J and a per-scenario best labeled privileged.
2. THE Experiments SHALL evaluate all policies and all 3 RL runs on 20 eval seeds for each of stable, high_churn, and heldout_medium.
3. IF the eval seed set is not locked in the manifest or does not match the lock, THEN THE Experiments SHALL raise a SplitLockError and write no results.
4. THE Experiments SHALL compute paired per-seed differences with RL averaged over runs, 10,000-resample bootstrap 95% CIs, and Wilcoxon signed-rank p-values.
5. THE Experiments SHALL apply Holm correction over the 9 primary tests of RL vs kademlia_epoch, tuned heuristic, and matched_random in each of 3 scenarios on J.
6. THE Experiments SHALL report per-run results and automatically generate an H1–H5 table.

### Requirement 26: Reproducibility, Reporting, and CLI

**User Story:** As a reviewer, I want one command to regenerate all results, so that I can verify the study.

#### Acceptance Criteria

1. THE CLI SHALL provide the commands show-config, simulate, compare, headroom, train, evaluate, and reproduce.
2. WHEN `reproduce --quick` is invoked, THE Experiments SHALL run 1 run of 40 episodes and 5 eval seeds and finish in under 10 min.
3. WHEN `reproduce --full` is invoked, THE Experiments SHALL regenerate all tables and figures of the study.
4. THE Experiments SHALL write a manifest containing the git hash, config hashes, and package versions.
5. THE ChurnMaster SHALL pin exact versions of numpy, scipy, pandas, matplotlib, pytest, and hypothesis in pyproject.toml and SHALL use no Gymnasium dependency.
6. THE ChurnMaster SHALL provide a README and a 4–6 page `docs/report.md` documenting the atomic-lookup limitation, the POMDP framing, and results including negative results.

### Requirement 27: Bursty Scenario Extension (optional, post-MVP)

**User Story:** As a researcher, I want correlated failures, so that I can test robustness beyond independent churn.

#### Acceptance Criteria

1. WHERE the bursty scenario is enabled, THE Churn_Generator SHALL, about every 15 min, take 20% of online nodes nearest a random point offline within 10 s and return each after an exponential downtime with mean 5 min.
2. WHERE the bursty scenario is enabled, THE Network SHALL multiply the RTT of nodes in the burst region by 2 for 3 min.

### Requirement 28: Ablations (optional)

**User Story:** As a researcher, I want ablations, so that I can attribute RL performance to its components.

#### Acceptance Criteria

1. WHERE ablations are enabled, THE Experiments SHALL run γ=0, no-PROBE, shuffled-reward, and feature drop-one variants.
2. WHERE ablations are enabled, THE Experiments SHALL sweep w_M ∈ {0.05, 0.2, 0.5} × w_L ∈ {0.1, 0.3, 0.6}.
3. WHERE ablations are enabled, THE Experiments SHALL evaluate a memoryless held-out scenario.
