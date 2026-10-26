# ChurnMaster

ChurnMaster asks whether tabular Q-learning can learn a better peer-replacement policy for a DHT routing table than hand-written rules. One focal node F in a simulated Kademlia-like network (128 nodes, 8-bit IDs, k=4, α=2, 3 replicas) sees only what a real node could observe. Every 30 s it makes one KEEP / PROBE / REPLACE decision per non-empty bucket, while the network churns underneath it. The learned policy is compared against Kademlia-style maintenance, random controls with matched action frequencies, a tuned rule-based heuristic, and a ground-truth oracle. Everything runs on a deterministic, CPU-only, pure-Python event simulator.

Research question: under churn, does a Q-learned maintenance policy beat standard Kademlia-style maintenance and a tuned heuristic on a combined lookup-quality/maintenance-cost score J, and does it generalize to a churn regime it was not trained on?

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
.\.venv\Scripts\python -m pytest                 # full suite (includes slow tests)
.\.venv\Scripts\python -m pytest -m "not slow"   # skip slow episode/timing tests
```

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
- `results/` is gitignored. The figures referenced in `docs/report.md` live in `results/figures/` and are regenerated by `reproduce --figures-only` (or `--full`).

## Repository layout

```
configs/base.toml                       simulator, reward and agent defaults
configs/scenarios/*.toml                stable, high_churn, heldout_medium (unseen), bursty (not yet evaluated)
configs/experiments/*.toml              headroom, train, tune, eval protocols
src/churnmaster/
  config.py seeding.py cli.py metrics.py plots.py
  sim/          ground truth + F's local view (idspace, keyed_rng, churn, workload, network,
                background, routing_table, lookup, maintenance, observation, env)
  rl/           observable-only: state discretizer, reward, ledger, Q-learning
  policies/     baselines, heuristic (32 configs), learned, oracle (only module with GT access)
  experiments/  runner, headroom, diagnostics, train, tune_heuristic, matched, evaluate, stats, reproduce
tests/          pytest + Hypothesis tests, including leakage, determinism and fairness guards
docs/report.md  study report
results/        all experiment outputs (gitignored)
```

## Key results (locked evaluation, 20 eval seeds per scenario)

- RL beats Kademlia-style epoch maintenance on J in all three scenarios (ΔJ +0.0085 to +0.0097, Holm-corrected p ≤ 1.7e-5) and uses 87–98% fewer pings.
- RL beats a random policy with matched action frequencies in every scenario (H1 pass, ΔJ +0.0126 to +0.0149), so the gain comes from *where* it acts, not *how often*.
- The gain carries over to the unseen `heldout_medium` scenario (ΔJ +0.0089 vs kademlia_epoch).
- RL is slightly but significantly worse than the tuned heuristic on J (ΔJ −0.0020 to −0.0030), and its p95 latency is 18–38% higher than the best baseline's (H3 fail).
- H2 and H4 are inconclusive, and H5 (per-run consistency) passes. Full details, including negative results and protocol deviations, are in [docs/report.md](docs/report.md).

## Determinism and reproducibility

- Every episode is seeded by `SeedSequence(root, spawn_key=(split, run, episode))`, with disjoint calibration / train / validation / eval / agent splits.
- Randomness that must not depend on the policy (packet loss, RTT jitter, background-table rebuilds, bootstrap) comes from keyed hashing (splitmix64) of the event identity. Two policies that send the same query see the same outcome.
- `headroom` freezes per-scenario environment hashes (the `sim` + `scenario` config sections) into `results/headroom/manifest.json`. Later stages refuse to run if the hash differs. Changing the reward or agent settings does not change the env hash.
- The eval seed list is locked in the manifest with `evaluate --create-lock` before evaluation. Evaluation refuses a missing or mismatched lock, and it refuses to run twice unless `--allow-rerun` is given (results would be identical).
- Manifests record the git hash, config hashes and package versions.
