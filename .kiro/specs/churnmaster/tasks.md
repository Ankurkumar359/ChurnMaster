# Implementation Plan: ChurnMaster

## Overview

Incremental Python 3.11+ implementation following the v2 plan task order (1–12). Each task delivers code plus its required pytest tests (acceptance I1–I6 depend on them). Hypothesis property tests for the design's Correctness Properties are separate optional sub-tasks (`*`). Tasks 11 and 12 are optional, post-MVP. Layering rule throughout: `sim/` never imports `rl/` or `policies/`, and only `experiments/*` wires components together.

## Tasks

- [x] 1. Scaffold, configuration, seeding, ID space, keyed RNG
  - [x] 1.1 Create project scaffold
    - `pyproject.toml` with exact pins (numpy, scipy, pandas, matplotlib, pytest, hypothesis), `src/` layout, console script `churnmaster`, no Gymnasium
    - Package `__init__.py` files for `churnmaster`, `sim`, `rl`, `policies`, `experiments`; `.gitignore` with `results/`
    - `tests/conftest.py` with `slow` marker and Hypothesis profile (≥100 examples); `tests/props/` directory
    - `configs/base.toml`, `configs/scenarios/{stable,high_churn,heldout_medium,bursty}.toml`, `configs/experiments/{headroom,train,tune,eval}.toml` with design values
    - _Requirements: 1.5, 1.6, 26.5_
  - [x] 1.2 Implement `config.py`
    - Frozen dataclasses (SimConfig, ScenarioConfig, RewardConfig, AgentConfig, ExperimentConfig, Config); `tomllib` loading; deep merge base → scenario → experiment; `validate` raising ConfigError naming the key; `config_hash` (SHA-256 of canonical JSON)
    - `tests/test_config.py`: merge order, defaults, frozen, invalid config rejection (unknown key, wrong type, out of range)
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_
  - [x] 1.3 Implement `seeding.py`
    - `Split` enum, `episode_seed` via `SeedSequence(root, spawn_key=(split, run, episode))`, `EpisodeStreams` (ids, types/positions, churn, congestion, workload generators and 64-bit keyed root), agent/random-baseline generators; eval seeds use run=0
    - `tests/test_seeding.py`: split disjointness over all splits used by experiments, eval seeds independent of run
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_
  - [x] 1.4 Implement `sim/idspace.py`
    - `xor_dist`, `bucket_index` (K == F → 0 for attribution), `closest`, `draw_ids`
    - `tests/test_idspace.py`: XOR examples, bucket boundaries, K == F edge case
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_
  - [x] 1.5 Implement `sim/keyed_rng.py`
    - Pure-Python `splitmix64`, `keyed_uniform(root, purpose, *ids)`, `keyed_normal` (Box–Muller from sub-keys 0 and 1), `Purpose`/`PingKind` enums
    - `tests/test_keyed_rng.py`: determinism, independence across purposes/ids, KS uniformity on 10k keys (α=0.01), normal mean/std sanity
    - _Requirements: 4.1, 4.2, 4.3, 4.4_
  - [x] 1.6 Implement `cli.py` with `show-config`
    - argparse subcommand skeleton for all seven commands (others stubbed); `show-config` prints merged config and hash
    - `tests/test_cli.py`: show-config output contains hash
    - _Requirements: 1.7, 26.1_
  - [ ]* 1.7 Write property test for config merge and hash
    - **Property 1: Config merge precedence and hash canonicality**
    - **Validates: Requirements 1.1, 1.4**
  - [ ]* 1.8 Write property test for invalid config rejection
    - **Property 2: Invalid configs are rejected**
    - **Validates: Requirements 1.3**
  - [ ]* 1.9 Write property test for seed split disjointness
    - **Property 3: Seed split disjointness**
    - **Validates: Requirements 2.3**
  - [ ]* 1.10 Write property test for XOR geometry
    - **Property 4: XOR geometry and bucket index**
    - **Validates: Requirements 3.1, 3.2, 3.3**
  - [ ]* 1.11 Write property test for closest selection
    - **Property 5: Closest selection matches sorting model**
    - **Validates: Requirements 3.5, 8.4**
  - [ ]* 1.12 Write property test for keyed RNG
    - **Property 6: Keyed RNG determinism and purity**
    - **Validates: Requirements 4.1, 4.2, 4.3**

- [x] 2. Churn, workload, network ground truth
  - [x] 2.1 Implement `sim/churn.py`
    - Node types by stable share; Weibull(0.6) sessions (scale = mean/Γ(1+1/0.6)) and exponential downtimes from −2 h burn-in; F always online; `ChurnTrace` with bisect `is_online`, sorted events, `online_at`, `trace_hash`; `analytic_online_fraction`
    - `tests/test_churn.py`: trace determinism, no session overlap, F always online, online fraction within 3 pp of analytic, nodes with older sessions survive the next 10 min more often
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_
  - [x] 2.2 Implement `sim/workload.py`
    - Pre-generated Poisson 1/s arrivals, integer ms, uniform keys, sequential lookup_id
    - `tests/test_workload.py`: rate within tolerance, id sequence
    - _Requirements: 6.1, 6.2_
  - [x] 2.3 Implement `sim/network.py`
    - Positions, types, congestion intervals (10% of nodes, ~120 s every ~20 min, ×3), keyed RTT formula, `query` (keyed loss, 500 ms timeout), `holders`, `bootstrap_entries` (BOOT keys)
    - `tests/test_network.py`: loss ≈ 2% within 0.5 pp, RTT/loss invariance for the same key tuple, holders exclude F
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 9.3_
  - [x] 2.4 Implement initial `plots.py`
    - Online-count-over-time and session-length histogram figures from a ChurnTrace
    - `tests/test_plots.py`: figures render to a temp dir without error
    - _Requirements: 26.3_
  - [ ]* 2.5 Write property test for churn traces
    - **Property 7: Churn trace well-formedness**
    - **Validates: Requirements 5.3, 5.4, 5.7**
  - [ ]* 2.6 Write property test for workload
    - **Property 8: Workload well-formedness**
    - **Validates: Requirements 6.1, 6.2**
  - [ ]* 2.7 Write property test for network query and RTT
    - **Property 9: Network query and RTT semantics**
    - **Validates: Requirements 7.3, 7.4**
  - [ ]* 2.8 Write property test for replica holders
    - **Property 10: Replica holders model**
    - **Validates: Requirements 7.6**

- [x] 3. LocalView, background tables, atomic lookup
  - [x] 3.1 Implement `sim/routing_table.py` (LocalView)
    - Entry/CacheEntry, buckets k=4, cache 4 with 300 s TTL, `record_response` (EWMA β=0.3), `record_failure`, `offer`, `autofill`, `incumbent` (LRS, tie earlier insertion), `candidate` (most recent, tie lower RTT), `evict`/`insert`/`cache_drop`, 120 s timeout window, `bootstrap(entries)`
    - `tests/test_routing_table.py`: auto-fill vs cache, LRS tie-break, candidate tie-break, TTL expiry
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8_
  - [x] 3.2 Implement `sim/background.py`
    - Initial build at t=0, staggered rebuild schedule `node_idx·300/N + 300·m` s, keyed (BG, node, rebuild_idx) selection of up to 4 online nodes per bucket, stale `closest_known`
    - `tests/test_background.py`: schedule, staleness, may return dead contacts
    - _Requirements: 8.1, 8.2, 8.3, 8.4_
  - [x] 3.3 Implement `sim/lookup.py`
    - Atomic iterative lookup per design pseudocode: shortlist of 4 without liveness filter, α=2 per round, round = slowest query, holder success latency, deadline/rounds/exhaustion failures with give-up latency, F stats side effects, responder offers, attribution to b(K)
    - `tests/test_lookup.py`: static network (no churn, p_loss=0) 1000/1000 success with mean rounds ≤ 3; deadline, round-limit, exhaustion cases; hand-computed latency; attribution; side effects
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 10.10_
  - [x] 3.4 Write import-guard test
    - `tests/test_leakage.py`: AST scan proving `rl/*`, `policies/*` (except `oracle.py`), `sim/routing_table.py`, `sim/observation.py` import none of network, churn, background, lookup, env, keyed_rng; `sim/*` imports nothing from `rl/` or `policies/`
    - _Requirements: 12.2, 12.3_
  - [ ]* 3.5 Write property test for background tables
    - **Property 11: Background table rebuild and staleness**
    - **Validates: Requirements 8.1, 8.2, 8.3**
  - [ ]* 3.6 Write property test for LocalView invariants
    - **Property 12: LocalView invariants under arbitrary operation sequences**
    - **Validates: Requirements 9.1, 9.2, 9.4, 9.5, 14.1**
  - [ ]* 3.7 Write property test for incumbent/candidate/timeout selection
    - **Property 13: Incumbent, candidate, and timeout-fraction selection**
    - **Validates: Requirements 9.6, 9.7, 9.8**
  - [ ]* 3.8 Write property test for lookup termination
    - **Property 14: Lookup termination and latency contract**
    - **Validates: Requirements 10.2, 10.4, 10.5, 10.9**
  - [ ]* 3.9 Write property test for lookup atomicity
    - **Property 15: Lookup atomicity**
    - **Validates: Requirements 10.6**

- [x] 4. Event loop, metrics, maintenance-off run
  - [x] 4.1 Implement `sim/env.py` event loop and record types
    - In `sim/observation.py`: `ObservedLookup`, `DecisionRecord`, `GTRecord` frozen dataclasses
    - `Env` with integer-ms priority queue (churn → rebuild → maintenance → lookup), bootstrap, 600 s warm-up + 2400 s measured, `measured` flags, maintenance-off mode, callbacks (`on_lookup`, `on_decision`, `gt_annex`)
    - `tests/test_env.py`: equal-timestamp ordering, warm-up flagged unmeasured, episode ends on time limit
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 12.6_
  - [x] 4.2 Implement `metrics.py`
    - `EpisodeMetrics`/`summarize` (success rate, p95 successful, p95 all with failures at give-up, pings/hour, mean latency, evictions/hour, messages/lookup, J, dead-entry fraction, actions by true incumbent state), `summary_hash`; measured window only
    - `tests/test_metrics.py`: hand-made record sets, warm-up excluded
    - _Requirements: 20.1, 20.2, 20.3, 11.3_
  - [x] 4.3 Add `simulate` command and determinism tests
    - `cli simulate` runs one maintenance-off episode and prints metrics
    - `tests/test_determinism.py`: identical summary hash for same seed; `@pytest.mark.slow` maintenance-off 3000 s episode ≤ 2 s
    - _Requirements: 11.5, 11.6, 26.1_
  - [ ]* 4.4 Write property test for determinism
    - **Property 19: Determinism**
    - **Validates: Requirements 11.5, 5.7**
  - [x] 4.5 Checkpoint - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Maintenance, DecisionContext, discretizer, baselines
  - [x] 5.1 Implement `DecisionContext` and `build_context` in `sim/observation.py`
    - Frozen, observable fields only; never-contacted entries use insertion time
    - `tests/test_observation.py`: field set equals the observable list, values from a scripted LocalView
    - _Requirements: 12.1_
  - [x] 5.2 Implement `sim/maintenance.py`
    - `Action` enum; `apply_action` for KEEP/PROBE/REPLACE with epoch ping keys (INC/CAND), IllegalActionError for REPLACE without candidate; `kademlia_std_on_offer` with STD keys
    - `tests/test_maintenance.py`: every action outcome branch with ping/eviction counts, kademlia_std fail/success branches
    - _Requirements: 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 14.9, 17.3_
  - [x] 5.3 Implement `rl/state.py`
    - Bins, `discretize`, mixed-radix `encode`/`decode`, `allowed_actions`, `reachable_states` (540)
    - `tests/test_state.py`: bin thresholds, cand_faster rules, 648/540 counts, masking
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5_
  - [x] 5.4 Implement `policies/base.py` and `policies/baselines.py`
    - Policy protocol; never_replace, always_replace, uniform_random, matched_random (from supplied frequencies split by candidate presence), kademlia_epoch, kademlia_std (decides KEEP, `shares_decision_points=False`); separate seeded generators
    - `tests/test_baselines.py`: actions on hand-made contexts
    - _Requirements: 17.1, 17.2, 17.4, 17.5_
  - [x] 5.5 Implement `policies/heuristic.py`
    - `HeuristicConfig`, `all_configs()` (32), rule function per design, `as_table`
    - `tests/test_heuristic.py`: rule order on hand-made contexts, 32 configs; representability: each config's 540-state table equals its rule function on 10k generated contexts
    - _Requirements: 18.1, 18.2, 18.3, 18.4_
  - [x] 5.6 Implement `policies/oracle.py`
    - `OracleFacts` (incumbent online, candidate online) and `OracleHeuristic`; labeled headroom reference, not an upper bound
    - `tests/test_oracle.py`: all GT combinations
    - _Requirements: 19.1, 19.2_
  - [x] 5.7 Wire maintenance epochs into `sim/env.py`
    - 30 s epochs: cache expiry, autofill, one decision per non-empty bucket via `decide`, `DecisionRecord` emission (null record for empty buckets), `GTRecord` emission, `oracle_view` hook, `mode="kademlia_std"` (post-lookup pings, costs as null-state DecisionRecords)
    - Extend `tests/test_env.py`: decision counts equal non-empty buckets, candidates only from lookup responders, kademlia_std costs recorded
    - _Requirements: 14.1, 14.2, 14.10, 17.3, 17.4, 21.4_
  - [x] 5.8 Implement `experiments/runner.py` and `compare` command
    - Sequential job runner for (policy, scenario, seed) returning metrics and records; policy factory; `cli compare` demo
    - `tests/test_runner.py`: small compare over two policies returns per-seed metrics
    - _Requirements: 26.1_
  - [x] 5.9 Write counterfactual leakage and fairness tests
    - Extend `tests/test_leakage.py`: counterfactual (same DecisionContext, different hidden state → same non-oracle action)
    - `tests/test_fairness.py`: trace hash and workload identical across policies; same keyed outcome for same key tuple; shared decision times for epoch-mode policies
    - _Requirements: 12.4, 21.1, 21.2, 21.4_
  - [ ]* 5.10 Write property test for discretizer
    - **Property 16: Discretizer round-trip and binning**
    - **Validates: Requirements 13.1, 13.2, 13.3, 13.4**
  - [ ]* 5.11 Write property test for action masking across policies
    - **Property 17: Action masking** (baselines and heuristics here; agent masking is covered in 8.1)
    - **Validates: Requirements 13.5, 16.3, 17.1, 17.5**
  - [ ]* 5.12 Write property test for action outcomes
    - **Property 18: Action outcome contract**
    - **Validates: Requirements 14.3, 14.4, 14.5, 14.6, 14.7**
  - [ ]* 5.13 Write property test for cross-policy fairness
    - **Property 20: Cross-policy fairness**
    - **Validates: Requirements 21.1, 21.2, 21.4**
  - [ ]* 5.14 Write property test for counterfactual leakage
    - **Property 21: Counterfactual leakage**
    - **Validates: Requirements 12.4**
  - [ ]* 5.15 Write property test for heuristic representability
    - **Property 28: Heuristic representability**
    - **Validates: Requirements 18.1, 18.2, 18.4**
  - [ ]* 5.16 Write property test for oracle rule
    - **Property 29: Oracle decision rule**
    - **Validates: Requirements 19.1**
  - [ ] 5.17 Checkpoint - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Headroom gate and environment freeze
  - [x] 6.1 Implement paired bootstrap CI in `experiments/stats.py`
    - Paired per-seed differences, 10k-resample percentile 95% CI (seeded)
    - `tests/test_stats.py`: CI on synthetic data with known shift
    - _Requirements: 22.2, 25.4_
  - [x] 6.2 Implement `experiments/headroom.py` and env freeze
    - 10 calibration seeds × {stable, high_churn} for never_replace, kademlia_epoch, always_replace, oracle_heuristic; G1/G2 evaluation on J / p95 latency (revised: gates on J/p95 per user decision after checkpoint 5.17); G1-failure tuning over deadline/loss/replicas on calibration seeds only; G2-failure "limited headroom" report
    - Manifest helpers in `experiments/runner.py` (write frozen env hash; refuse runs whose env hash differs); `configs/experiments/headroom.toml`; `cli headroom`
    - `tests/test_headroom.py`: gate logic on synthetic metrics, frozen-hash mismatch refusal
    - _Requirements: 22.1, 22.2, 22.3, 22.4, 22.5, 22.6, 22.7_

- [x] 7. Reward, ledger, diagnostics
  - [x] 7.1 Implement `rl/reward.py` and `rl/ledger.py`
    - `lookup_loss`, `action_cost`; Ledger accepting only ObservedLookup/DecisionRecord (TypeError otherwise), per-bucket-epoch SUM rewards, transitions with τ=1, dropping open transitions at episode end/empty bucket, `total_reward`, `score`
    - `tests/test_reward.py` (ℓ range), `tests/test_ledger.py` (scripted ledger, masking of null records, TypeError)
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.6, 15.7, 12.5_
  - [x] 7.2 Wire Ledger into runner and metrics
    - Runner feeds every record to the Ledger; `metrics.py` uses `rl/reward.py`; assert Metrics.J == Ledger.score() including kademlia_std costs
    - Extend `tests/test_runner.py`: identity on real short episodes for every policy
    - _Requirements: 15.4, 15.5, 20.4_
  - [x] 7.3 Implement `experiments/diagnostics.py`
    - Reward-total identity check; reward sanity (discounted returns for dead top/second-tier incumbents: PROBE/REPLACE vs KEEP; live: KEEP vs REPLACE; CIs); coverage (≥80% decisions in states visited ≥50 times over 20 episodes); J vs success Spearman
    - `tests/test_diagnostics.py`: each diagnostic on synthetic records
    - _Requirements: 23.1, 23.2, 23.3, 23.4_
  - [ ]* 7.4 Write property test for reward-total identity
    - **Property 22: Reward-total identity**
    - **Validates: Requirements 15.2, 15.3, 15.4, 15.5, 20.4**
  - [ ]* 7.5 Write property test for lookup loss range
    - **Property 23: Lookup loss range**
    - **Validates: Requirements 15.1**
  - [ ]* 7.6 Write property test for transition construction
    - **Property 24: Transition construction**
    - **Validates: Requirements 15.6, 15.7**

- [x] 8. Q-learning and training
  - [x] 8.1 Implement `rl/qlearning.py`
    - QAgent (648×3 Q init 0, counts N), masked ε-greedy `act` with fixed tie order KEEP<PROBE<REPLACE, `update` with α_n = max(α_min,(1+n)^−0.7), `epsilon` schedule, `.npy` save/load
    - `tests/test_qlearning.py`: hand-computed update, toy 2-state MDP within 0.05 of analytic Q, masking, schedules, save/load
    - _Requirements: 16.1, 16.2, 16.3, 16.4, 16.5, 16.7_
  - [x] 8.2 Implement `policies/learned.py`
    - Greedy policy over a frozen Q-table via `discretize`/`encode`/`allowed_actions`
    - `tests/test_learned.py`: greedy choice and masking on a fixed table
    - _Requirements: 16.4_
  - [x] 8.3 Implement `experiments/train.py` and `train` command
    - Training loop: 200 episodes alternating stable/high_churn, learning in warm-up and measured periods, transitions from Ledger; validation at episodes 0, 20, …, 200 on 5 validation seeds per training scenario (ε=0); `.npy` snapshots, `.csv` curves, mean |ΔQ| per episode
    - Modes: 8a debug (high_churn, 1 run, 100 ep), 8b grid γ{0.8,0.9}×α_min{0.02,0.05} selected by mean validation J over last 3 checkpoints, 8c 3 runs; I4 check (final vs episode-0 paired CI, shrinking |ΔQ|)
    - `configs/experiments/train.toml`, `cli train`; `tests/test_train.py`: short training is deterministic, snapshots/curves written, grid selection logic
    - _Requirements: 16.6, 24.1, 24.2, 24.3, 24.4, 24.5, 24.6_
  - [x] 8.4 Add multiprocessing to `experiments/runner.py`
    - Process pool over jobs with deterministic ordering of results
    - Extend `tests/test_runner.py`: multiprocessing results equal sequential results
    - _Requirements: 21.3_
  - [ ]* 8.5 Write property test for Q update
    - **Property 25: Q-update contract**
    - **Validates: Requirements 16.2**
  - [ ]* 8.6 Write property test for epsilon schedule
    - **Property 26: Epsilon schedule**
    - **Validates: Requirements 16.3**
  - [ ]* 8.7 Write property test for Q-table persistence
    - **Property 27: Q-table save/load round-trip**
    - **Validates: Requirements 16.5**
  - [x] 8.8 Checkpoint - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Heuristic tuning, matched random, evaluation, statistics
  - [x] 9.1 Implement `experiments/tune_heuristic.py`
    - All 32 configs on 10 train seeds per scenario; global best (stable + high_churn) by mean J; per-scenario best labeled privileged; `configs/experiments/tune.toml`
    - `tests/test_tune_heuristic.py`: selection logic on synthetic results
    - _Requirements: 25.1_
  - [x] 9.2 Add matched_random frequency extraction
    - In `policies/baselines.py`: build per-scenario frequencies split by candidate presence from RL decisions on validation seeds
    - Extend `tests/test_baselines.py`: frequencies from scripted decisions; sampled actions respect masking
    - _Requirements: 17.2_
  - [x] 9.3 Implement `experiments/evaluate.py` with eval lock and `evaluate` command
    - 20 eval seeds × {stable, high_churn, heldout_medium} × all policies × 3 RL runs; eval split lock in manifest; SplitLockError on missing or mismatched lock; per-seed and per-run CSVs; `configs/experiments/eval.toml`
    - `tests/test_evaluate.py`: lock enforcement, eval seeds disjoint from calibration/train/validation
    - _Requirements: 25.2, 25.3, 25.6, 2.3_
  - [x] 9.4 Extend `experiments/stats.py`
    - Wilcoxon signed-rank, Holm over the 9 primary tests (RL vs kademlia_epoch, tuned heuristic, matched_random × 3 scenarios on J), RL averaged over runs, per-run results, automatic H1–H5 table (CSV + Markdown)
    - Extend `tests/test_stats.py`: Wilcoxon/Holm on synthetic data, H-table generation
    - _Requirements: 25.4, 25.5, 25.6_
  - [ ]* 9.5 Write property test for Holm correction
    - **Property 30: Holm correction monotonicity**
    - **Validates: Requirements 25.5**

- [ ] 10. Report and reproducibility (MVP complete)
  - [x] 10.1 Implement `experiments/reproduce.py` and `reproduce` command
    - `--quick` (1 run, 40 episodes, 5 eval seeds, <10 min) and `--full` (headroom → diagnostics → train → tune → evaluate → stats → plots); manifest with git hash, config hashes, package versions; learning-curve and comparison figures in `plots.py`
    - `tests/test_reproduce.py` (`slow`): quick mode writes manifest, tables, and figures
    - _Requirements: 26.2, 26.3, 26.4_
  - [ ] 10.2 Write `README.md` and `docs/report.md`
    - README: install, CLI usage, reproduce commands; report (4–6 pages): setup, POMDP framing, atomic-lookup limitation, results tables/figures from reproduce, H1–H5 outcomes including negative results
    - _Requirements: 26.6_
  - [ ] 10.3 Final checkpoint - Ensure all tests pass, ask the user if questions arise.

- [ ] 11. Bursty extension (optional, post-MVP)
  - [ ]* 11.1 Implement bursty churn and regional RTT
    - In `sim/churn.py` and `sim/network.py`: ~15 min exponential burst gaps, 20% of online nodes near a random point leave within 10 s, return after Exp(5 min); region RTT ×2 for 3 min
    - `tests/test_bursty.py`: burst sizes, timing, RTT multiplier
    - _Requirements: 27.1, 27.2_
  - [ ]* 11.2 Evaluate all policies on bursty
    - Extend `experiments/evaluate.py` with an opt-in bursty scenario (reported separately from primary tests)
    - _Requirements: 27.1, 27.2_

- [ ] 12. Ablations (optional)
  - [ ]* 12.1 Implement agent ablations in `experiments/ablations.py`
    - γ=0, no PROBE (masked), shuffled reward, feature drop-one (discretizer wrapper)
    - _Requirements: 28.1_
  - [ ]* 12.2 Implement reward-weight sweep
    - w_M {0.05, 0.2, 0.5} × w_L {0.1, 0.3, 0.6} in `experiments/ablations.py`
    - _Requirements: 28.2_
  - [ ]* 12.3 Implement memoryless held-out scenario
    - Exponential-session scenario config and its evaluation in `experiments/ablations.py`
    - _Requirements: 28.3_

## Notes

- Tasks marked `*` are optional. Hypothesis property tests are optional because each module's required pytest tests (needed for I1–I6) are part of its implementation task.
- Tasks 11 and 12 are optional and post-MVP; MVP is complete after Task 10.
- Task 6 freezes the env config hash; Tasks 8–10 must run against it.
- Each property test is tagged `Feature: churnmaster, Property N: <title>` and runs ≥100 examples (episode-level properties use shortened episodes with a documented lower `max_examples`).
- Pin hypothesis alongside the other dependencies in `pyproject.toml`.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "1.3", "1.4", "1.5"] },
    { "id": 2, "tasks": ["1.6", "1.7", "1.9", "1.10", "1.12", "2.1", "2.2"] },
    { "id": 3, "tasks": ["1.8", "1.11", "2.3", "2.4", "2.5", "2.6", "3.1"] },
    { "id": 4, "tasks": ["2.7", "3.2", "3.4", "3.6"] },
    { "id": 5, "tasks": ["2.8", "3.3", "3.5", "3.7"] },
    { "id": 6, "tasks": ["3.8", "4.1"] },
    { "id": 7, "tasks": ["3.9", "4.2"] },
    { "id": 8, "tasks": ["4.3", "4.4"] },
    { "id": 9, "tasks": ["4.5"] },
    { "id": 10, "tasks": ["5.1", "5.2"] },
    { "id": 11, "tasks": ["5.3", "5.6", "5.7"] },
    { "id": 12, "tasks": ["5.4", "5.5", "5.10", "5.12"] },
    { "id": 13, "tasks": ["5.8", "5.15", "5.16"] },
    { "id": 14, "tasks": ["5.9", "5.11"] },
    { "id": 15, "tasks": ["5.13", "5.14"] },
    { "id": 16, "tasks": ["5.17"] },
    { "id": 17, "tasks": ["6.1", "7.1"] },
    { "id": 18, "tasks": ["6.2", "7.4", "7.5"] },
    { "id": 19, "tasks": ["7.2", "7.6"] },
    { "id": 20, "tasks": ["7.3", "8.1"] },
    { "id": 21, "tasks": ["8.2", "8.5"] },
    { "id": 22, "tasks": ["8.3", "8.6"] },
    { "id": 23, "tasks": ["8.4", "8.7"] },
    { "id": 24, "tasks": ["8.8"] },
    { "id": 25, "tasks": ["9.1", "9.2", "9.4"] },
    { "id": 26, "tasks": ["9.3", "9.5"] },
    { "id": 27, "tasks": ["10.1"] },
    { "id": 28, "tasks": ["10.2"] },
    { "id": 29, "tasks": ["10.3", "11.1", "12.1"] },
    { "id": 30, "tasks": ["11.2", "12.2"] },
    { "id": 31, "tasks": ["12.3"] }
  ]
}
```
