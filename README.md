# ChurnMaster

ChurnMaster asks whether tabular Q-learning can learn a better peer-replacement policy for a DHT routing table than hand-written rules. One focal node F in a simulated Kademlia-like network (128 nodes, 8-bit IDs, k=4, α=2, 3 replicas) sees only what a real node could observe. Every 30 s it makes one KEEP / PROBE / REPLACE decision per non-empty bucket, while the network churns underneath it. The learned policy is compared against Kademlia-style maintenance, random controls with matched action frequencies, a tuned rule-based heuristic, and a ground-truth oracle. Everything runs on a deterministic, CPU-only, pure-Python event simulator.

Research question: under churn, does a Q-learned maintenance policy beat standard Kademlia-style maintenance and a tuned heuristic on a combined lookup-quality/maintenance-cost score J, and does it generalize to a churn regime it was not trained on?


**Deliverables.** Two-page paper: [`report/churnmaster_report.pdf`](report/churnmaster_report.pdf) (LaTeX source alongside). Full technical record: [`FINAL_RESULTS.md`](FINAL_RESULTS.md). Long-form study report: [`docs/report.md`](docs/report.md). Release tags: `v1.0.0` (locked evaluation, MVP), `v1.1.0` (exploratory ablations), `v1.2.0` (submission package).

## RL formulation

| element | definition |
|---|---|
| environment | discrete-event simulator of a 128-node Kademlia-like DHT; only the focal node F learns; 600 s warm-up + 2400 s measured per episode; one decision per non-empty bucket every 30 s |
| state | 7 observable features of the bucket's least-recently-seen incumbent and freshest cached candidate: tier (3), consecutive failures (3), time since last reply (2), age (3), candidate none/fresh/stale (3), candidate faster (2), churn signal (2) → 3·3·2·3·3·2·2 = 648 indices, 540 reachable ([`rl/state.py`](src/churnmaster/rl/state.py)) |
| actions | `KEEP` (no message), `PROBE` (ping incumbent; evict on failure), `REPLACE` (ping candidate, swap in if it answers; masked without a candidate) |
| loss / reward | per lookup ℓ = (1 − success) + w_L·min(latency / 2000 ms, 1); bucket-epoch reward r = −Σℓ − (w_M·pings + w_R·evictions), w_L = 0.3, w_M = 0.1, w_R = 0.15 ([`rl/reward.py`](src/churnmaster/rl/reward.py), [`rl/ledger.py`](src/churnmaster/rl/ledger.py)) |
| score | J = 1 − mean(ℓ) − (w_M·pings + w_R·evictions)/N_lookups = 1 + Σr / N_lookups |
| learner | tabular Q-learning, Q(s,a) ← Q(s,a) + α_n [r + γ max_{a′} Q(s′,a′) − Q(s,a)], α_n = max(α_min, (1+n)^−0.7), ε-greedy 1.0 → 0.05 over 60% of 200 episodes; grid-selected γ = 0.8, α_min = 0.02; 3 independent runs ([`rl/qlearning.py`](src/churnmaster/rl/qlearning.py)) |

The observation hides the true network state, so this is a POMDP treated as an MDP; all claims are empirical.

## Install

Requires Python 3.11+. All dependencies are pinned exactly in `pyproject.toml` (numpy, scipy, pandas, matplotlib; pytest and hypothesis as dev extras). Gymnasium is not used.

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

Linux / macOS:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

The examples below use the Windows path `.\.venv\Scripts\python`. On Linux/macOS, use `.venv/bin/python` instead.

## Tests

```powershell
.\.venv\Scripts\python -m pytest                 # full suite (includes slow tests; ~5 min)
.\.venv\Scripts\python -m pytest -m "not slow"   # skip slow episode/timing tests (~1.5 min)
.\.venv\Scripts\python report\verify_report.py   # cross-check every number in the PDF report against results/
```

## Using the trained agent

The three canonical Q-tables are tracked in `results/train/final/run{0,1,2}/q_final.npy` (with `_counts.npy`, `_mask.npy` and `.json` sidecars).

```python
from churnmaster.rl.qlearning import QAgent
from churnmaster.rl.state import Features, encode

agent = QAgent.load("results/train/final/run0/q_final.npy")   # 648 x 3 table, gamma 0.8, alpha_min 0.02
s = encode(Features(tier=0, inc_fail=2, inc_stale=1, inc_age=0, cand=1, cand_faster=0, churn=1))
print(agent.greedy(s).name)                                    # greedy action for that state
```

`churnmaster.policies.learned.Learned.load(path)` wraps a Q-table as a frozen policy for the runner; `evaluate` uses all three runs. The locked evaluation is single-use: re-running it on the canonical seeds needs `evaluate --allow-rerun --out DIR`, and `reproduce --full` (below) retrains and re-evaluates from scratch and checks the results byte-for-byte.

## CLI

All commands are available as `churnmaster <command>` or `python -m churnmaster <command>`. Add `--help` to any command to see its options.

| command | purpose |
|---|---|
| `show-config` | print the merged configuration (base → scenario → experiment) and its hash |
| `simulate` | run one maintenance-off episode and print metrics |
| `compare` | compare policies over seeds (quick demo) |
| `headroom` | headroom gates G1/G2 on calibration seeds; freezes the environment hashes |
| `diagnostics` | reward identity, reward sanity, state coverage and J/success Spearman (uniform_random behaviour) |
| `train` | Q-learning training: `--mode debug`, `grid` or `final` |
| `tune` | evaluate the 32 heuristic configs on train seeds; pick global and privileged bests |
| `evaluate` | locked evaluation on the eval split (all policies, 3 RL runs), stats and H1–H5 table |
| `reproduce` | regenerate everything: `--quick`, `--full` or `--figures-only` |
| `ablate` | exploratory ablations on TEST seeds (not part of the locked evaluation): `--suite agent`, `reward` or `memoryless` |

The matched-random frequencies come from a module entry point rather than a CLI subcommand: `python -m churnmaster.experiments.matched`.

### Canonical study order

The canonical results in `results/` were produced in this order. Each step reads the frozen environment manifest written by `headroom`.

```powershell
$py = ".\.venv\Scripts\python"
& $py -m churnmaster headroom --workers 8                              # results/headroom (G1/G2, env freeze)
& $py -m churnmaster diagnostics --workers 8                           # results/diagnostics (20 episodes)
& $py -m churnmaster diagnostics --episodes 60 --out results/diagnostics_60 --workers 8
& $py -m churnmaster train --mode debug --workers 8                    # results/train/debug
& $py -m churnmaster train --mode grid --workers 8                     # results/train/grid (+ selected.json)
& $py -m churnmaster train --mode final --workers 8                    # results/train/final/run{0,1,2}
& $py -m churnmaster tune --workers 8                                  # results/tune
& $py -m churnmaster.experiments.matched --workers 8                   # results/matched/freqs.json
& $py -m churnmaster evaluate --create-lock                            # writes the eval-split lock
& $py -m churnmaster evaluate --workers 8                              # results/eval (single use)
& $py -m churnmaster reproduce --figures-only                          # results/figures (+ tables/*.md)
```

The `--workers` count never changes results. Runs are identical for any N. Most commands run sequentially by default (`reproduce` defaults to min(cpu_count − 1, 8) workers). The manifests record 8 workers for the canonical runs.

## Reproducing the study

```powershell
.\.venv\Scripts\python -m churnmaster reproduce --quick         # ~2 min: 1 run x 40 episodes, 5 eval seeds -> results/reproduce_quick
.\.venv\Scripts\python -m churnmaster reproduce --full          # ~11 min (8 workers): whole pipeline + verification -> results/reproduce_full
.\.venv\Scripts\python -m churnmaster reproduce --figures-only  # figures/tables from existing results -> results/figures
```

- `--quick` is a smoke run. Its numbers are not the study's results. With 5 eval seeds, for example, H1 comes out inconclusive.
- `--full` reruns headroom → diagnostics → train (grid + 3 final runs) → tune → matched → evaluate → figures in a fresh directory. It then verifies the run against the canonical `results/`: `eval/per_seed.csv` must be byte-identical (SHA-256), and the grid and tuning selections must match. On the reference machine this took 659 s, and verification passed with all 780 per-seed rows identical.
- Canonical-artifact guard: `reproduce` refuses to write into the canonical results directories (`results/headroom`, `results/eval`, and so on). Pass `--out DIR` to choose a different target. The locked evaluation itself is single-use (see below).
- The canonical results are tracked in Git (`results/headroom`, `diagnostics`, `diagnostics_60`, `checkpoint_5_17`, `train`, `tune`, `matched`, `eval`, `ablations`, `figures`), so a clean clone can load the agents, regenerate figures, run every test and verify `reproduce --full`. Only the verification reruns (`results/reproduce_quick`, `results/reproduce_full`) and other new output directories are gitignored. `reproduce --figures-only` (or `--full`) regenerates the primary figures and tables; `python scripts/export_ablation_assets.py` copies the ablation figures and tables into `results/figures/ablations/`.
- Exploratory scenarios live in `configs/ablations/`, not `configs/scenarios/`. `headroom` and `reproduce` freeze every file in `configs/scenarios/`, so the canonical pipeline and its env manifest are unaffected by the ablations.

## Repository layout

```
configs/base.toml                       simulator, reward and agent defaults
configs/scenarios/*.toml                stable, high_churn, heldout_medium (unseen), bursty (not yet evaluated)
configs/experiments/*.toml              headroom, train, tune, eval protocols
configs/ablations/*.toml                exploratory-only scenarios (heldout_memoryless); never frozen
src/churnmaster/
  config.py seeding.py cli.py metrics.py plots.py
  sim/          ground truth + F's local view (idspace, keyed_rng, churn, workload, network,
                background, routing_table, lookup, maintenance, observation, env)
  rl/           observable-only: state discretizer, reward, ledger, Q-learning
  policies/     baselines, heuristic (32 configs), learned, oracle (only module with GT access)
  experiments/  runner, headroom, diagnostics, train, tune_heuristic, matched, evaluate, stats, reproduce,
                ablations (exploratory, Task 12)
scripts/        export_ablation_assets.py (copy ablation figures/tables into results/figures/ablations)
report/         two-page paper: churnmaster_report.{tex,pdf}, fig_results.pdf, make_figure.py, verify_report.py
tests/          pytest + Hypothesis tests, including leakage, determinism and fairness guards
docs/report.md  long-form study report
FINAL_RESULTS.md  consolidated technical record of the project
results/        canonical experiment outputs (tracked; reproduce_* verification reruns are local)
```

## Key results (locked evaluation, 20 eval seeds per scenario)

- RL beats Kademlia-style epoch maintenance on J in all three scenarios (ΔJ +0.0085 to +0.0097, Holm-corrected p ≤ 1.7e-5) and uses 87–98% fewer pings.
- RL beats a random policy with matched action frequencies in every scenario (H1 pass, ΔJ +0.0126 to +0.0149), so the gain comes from *where* it acts, not *how often*.
- The gain carries over to the unseen `heldout_medium` scenario (ΔJ +0.0089 vs kademlia_epoch).
- RL is slightly but significantly worse than the tuned heuristic on J (ΔJ −0.0020 to −0.0030), and its p95 latency is 18–38% higher than the best baseline's (H3 fail).
- H2 and H4 are inconclusive, and H5 (per-run consistency) passes. Full details, including negative results and protocol deviations, are in [docs/report.md](docs/report.md).

## Exploratory ablations (Task 12; not part of the locked evaluation)

These were run after the locked evaluation, on TEST seeds (run 0, episodes 0–19) that are disjoint from the eval seeds. p-values are **not corrected for multiple comparisons**, and the hypothesis verdicts above are unchanged. Details are in [docs/report.md §10](docs/report.md); tables are in `results/figures/ablations/tables/`.

```powershell
.\.venv\Scripts\python -m churnmaster ablate --suite agent --workers 8       # ~20 min: gamma0, no_probe, shuffled_reward, buffered_control, drop_<feature>
.\.venv\Scripts\python -m churnmaster ablate --suite reward --workers 8      # ~17 min: 12 (w_L, w_M) cells x 3 runs, scored under canonical weights
.\.venv\Scripts\python -m churnmaster ablate --suite memoryless --workers 8  # ~30 s: canonical policies on exponential sessions
.\.venv\Scripts\python scripts\export_ablation_assets.py                     # copy figures/tables into results/figures/ablations
```

- Shuffling rewards across transitions costs ΔJ ≈ −0.009 to −0.012 against an unshuffled buffered control, and γ = 0 is worse in every scenario. So the learned behaviour depends on the reward signal and, to a lesser degree, on multi-epoch credit.
- Raising the training latency weight w_L lowers RL's p95, especially in stable and heldout_medium, but costs more pings. No weight cell closes the J gap to the tuned heuristic in high_churn. The one cell with a positive J difference (stable) was selected post hoc and needs confirmation on fresh seeds.
- On a memoryless (exponential-session) variant of heldout_medium, all policies change little. RL's gap to the tuned heuristic does not shift detectably (+0.0007 [−0.0004, +0.0018]).

## Report

```powershell
.\.venv\Scripts\python report\make_figure.py     # Figure 1 from results/eval/summary.csv
cd report; pdflatex churnmaster_report.tex; pdflatex churnmaster_report.tex   # needs a LaTeX install (e.g. MiKTeX, TeX Live)
.\.venv\Scripts\python report\verify_report.py   # every number, verdict and parameter checked against results/, configs/, src/
```

## Determinism and reproducibility

- Every episode is seeded by `SeedSequence(root, spawn_key=(split, run, episode))`, with disjoint calibration / train / validation / eval / agent splits.
- Randomness that must not depend on the policy (packet loss, RTT jitter, background-table rebuilds, bootstrap) comes from keyed hashing (splitmix64) of the event identity. Two policies that send the same query see the same outcome.
- `headroom` freezes per-scenario environment hashes (the `sim` + `scenario` config sections) into `results/headroom/manifest.json`. Later stages refuse to run if the hash differs. Changing the reward or agent settings does not change the env hash.
- The eval seed list is locked in the manifest with `evaluate --create-lock` before evaluation. Evaluation refuses a missing or mismatched lock, and it refuses to run twice unless `--allow-rerun` is given (results would be identical).
- Manifests record the git hash, config hashes and package versions.
