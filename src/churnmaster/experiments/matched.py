"""Per-scenario matched_random frequencies (Requirement 17.2, task 9.2).

matched_random samples actions from the RL policy's action frequencies split
by candidate presence, measured per scenario on VALIDATION seeds:

* every trained Q-table in ``q_paths`` is run greedily (:class:`Learned`) on
  ``Split.VALIDATION`` episodes ``0..n_seeds-1`` (run index 0, so all RL runs
  see the same validation environments) of each scenario;
* the non-null decisions are pooled over all RL runs and seeds of the
  scenario and turned into frequencies by
  :func:`~churnmaster.policies.baselines.action_frequencies`.

``heldout_medium`` is evaluated too (task 9.3), so it is measured on its
validation seeds as well. Training never ran on heldout_medium; using its
validation split (never the eval split) keeps eval seeds untouched while
giving matched_random a per-scenario profile there.

Every scenario config is checked against the frozen env manifest first
(``FrozenEnvError``; nothing is run). Output: ``freqs.json`` with
``scenario -> {"no_cand": [pK, pP, pR], "cand": [...], "counts": {"no_cand":
[...], "cand": [...]}, ...}`` (extra metadata keys are ignored by
:func:`load_matched_freqs`).

Entry point: ``python -m churnmaster.experiments.matched``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from churnmaster.experiments.diagnostics import _frozen_loader
from churnmaster.experiments.runner import Job, check_frozen, default_workers, run_jobs
from churnmaster.policies.baselines import _validate_freqs, action_counts, action_frequencies
from churnmaster.policies.learned import Learned
from churnmaster.seeding import Split

__all__ = ["DEFAULT_SCENARIOS", "FREQS_NAME", "measure_matched_freqs", "write_matched_freqs",
           "load_matched_freqs", "main"]

DEFAULT_SCENARIOS: tuple[str, ...] = ("stable", "high_churn", "heldout_medium")
FREQS_NAME = "freqs.json"
_KEYS = {False: "no_cand", True: "cand"}


def measure_matched_freqs(q_paths: Sequence[Path | str], configs_dir: Path | str,
                          manifest_path: Path | str, *,
                          scenarios: Sequence[str] = DEFAULT_SCENARIOS, n_seeds: int = 5,
                          workers: int | None = None, experiment_path: Path | str | None = None,
                          out_path: Path | str | None = None,
                          ) -> dict[str, dict[bool, tuple[float, float, float]]]:
    """Measure per-scenario matched_random frequencies (see module docstring).

    Writes ``out_path`` (if given) via :func:`write_matched_freqs`.
    """
    if not q_paths:
        raise ValueError("measure_matched_freqs needs at least one Q-table")
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be >= 1, got {n_seeds}")
    configs_dir, manifest_path = Path(configs_dir), Path(manifest_path)
    exp = None if experiment_path is None else Path(experiment_path)
    loader = _frozen_loader(configs_dir, exp, manifest_path)
    cfgs = {s: loader(s) for s in scenarios}
    env_hashes = {s: check_frozen(manifest_path, c) for s, c in cfgs.items()}

    policies = [Learned.load(p, name=f"learned_run{i}", id_bits=cfgs[scenarios[0]].sim.id_bits)
                for i, p in enumerate(q_paths)]
    freqs: dict[str, dict[bool, tuple[float, float, float]]] = {}
    records: dict[str, dict[str, Any]] = {}
    for s in scenarios:
        jobs = [Job(pol, s, Split.VALIDATION, 0, ep) for pol in policies for ep in range(n_seeds)]
        results = run_jobs(jobs, cfg_loader=loader, keep_records=True, workers=workers)
        decisions = [d for r in results for d in (r.decisions or [])]
        counts = action_counts(decisions)
        freqs[s] = action_frequencies(counts)
        records[s] = {"counts": counts, "env_hash": env_hashes[s], "n_episodes": len(jobs)}
    if out_path is not None:
        write_matched_freqs(out_path, freqs, records, q_paths=q_paths, n_seeds=n_seeds)
    return freqs


def write_matched_freqs(path: Path | str, freqs: Mapping[str, Mapping[bool, Sequence[float]]],
                        records: Mapping[str, Mapping[str, Any]] | None = None, *,
                        q_paths: Sequence[Path | str] = (), n_seeds: int | None = None) -> Path:
    out = Path(path)
    data: dict[str, Any] = {}
    for s, f in freqs.items():
        rec = dict((records or {}).get(s, {}))
        entry: dict[str, Any] = {_KEYS[k]: [float(x) for x in f[k]] for k in (False, True)}
        if "counts" in rec:
            counts = rec.pop("counts")
            entry["counts"] = {_KEYS[k]: [int(x) for x in counts[k]] for k in (False, True)}
        entry.update(rec)
        entry.update({"split": "validation", "n_seeds": n_seeds,
                      "q_paths": [str(p) for p in q_paths]})
        data[s] = entry
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def load_matched_freqs(path: Path | str) -> dict[str, dict[bool, tuple[float, float, float]]]:
    """Read ``freqs.json`` back into ``scenario -> {has_cand: (pK, pP, pR)}`` (validated)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, dict[bool, tuple[float, float, float]]] = {}
    for s, entry in data.items():
        f = {k: tuple(float(x) for x in entry[_KEYS[k]]) for k in (False, True)}
        _validate_freqs(f)
        out[s] = f  # type: ignore[assignment]
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m churnmaster.experiments.matched",
                                 description="Measure matched_random frequencies (Req. 17.2).")
    ap.add_argument("--q", nargs="+", type=Path,
                    default=[Path(f"results/train/final/run{i}/q_final.npy") for i in range(3)])
    ap.add_argument("--configs-dir", type=Path, default=Path("configs"))
    ap.add_argument("--manifest", type=Path, default=Path("results/headroom/manifest.json"))
    ap.add_argument("--experiment-file", type=Path, default=None)
    ap.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS))
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("results/matched") / FREQS_NAME)
    a = ap.parse_args(argv)
    workers = default_workers() if a.workers is None else a.workers
    measure_matched_freqs(a.q, a.configs_dir, a.manifest, scenarios=a.scenarios,
                          n_seeds=a.n_seeds, workers=workers, experiment_path=a.experiment_file,
                          out_path=a.out)
    data = json.loads(a.out.read_text(encoding="utf-8"))
    for s in a.scenarios:
        e = data[s]
        for k in ("no_cand", "cand"):
            n = e["counts"][k]
            print(f"{s:15s} {k:8s} n={sum(n):7d} counts(K,P,R)={n} "
                  f"freqs=({', '.join(f'{x:.4f}' for x in e[k])})")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
