"""Command-line entry point (Requirement 26.1).

Subcommands: show-config, simulate, compare, headroom, train, evaluate,
reproduce. ``show-config``, ``simulate`` and ``compare`` are implemented; the
others are stubs that exit non-zero until their tasks land.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

from churnmaster.config import ConfigError, config_hash, load_config, to_dict

COMMANDS = ("show-config", "simulate", "compare", "headroom", "train", "evaluate", "reproduce")


def default_configs_dir() -> Path:
    """``./configs`` if present, else the repository's ``configs`` directory."""
    cwd = Path.cwd() / "configs"
    if (cwd / "base.toml").is_file():
        return cwd
    return Path(__file__).resolve().parents[2] / "configs"


def _configs_root(args: argparse.Namespace) -> Path:
    return Path(args.configs_dir) if args.configs_dir else default_configs_dir()


def _experiment_path(args: argparse.Namespace) -> Path | None:
    """``--experiment-file PATH`` or ``--experiment NAME`` (configs/experiments/NAME.toml)."""
    if getattr(args, "experiment_file", None):
        return Path(args.experiment_file)
    if args.experiment:
        return _configs_root(args) / "experiments" / f"{args.experiment}.toml"
    return None


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path | None]:
    root = _configs_root(args)
    base = root / "base.toml"
    scenario = root / "scenarios" / f"{args.scenario}.toml"
    return base, scenario, _experiment_path(args)


def _add_config_args(p: argparse.ArgumentParser, scenario: bool = True) -> None:
    if scenario:
        p.add_argument("--scenario", default="stable",
                       help="scenario name (configs/scenarios/NAME.toml)")
    exp = p.add_mutually_exclusive_group()
    exp.add_argument("--experiment", default=None,
                     help="experiment name (configs/experiments/NAME.toml)")
    exp.add_argument("--experiment-file", default=None, metavar="PATH",
                     help="experiment TOML file (any path; same layer as --experiment)")
    p.add_argument("--configs-dir", default=None, help="override the configs directory")


def cmd_show_config(args: argparse.Namespace) -> int:
    cfg = load_config(*_resolve_paths(args))
    print(json.dumps(to_dict(cfg), indent=2, sort_keys=True))
    print(f"config_hash: {config_hash(cfg)}")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    """One maintenance-off episode; prints scalar metrics and hashes.

    Only ``--maintenance off`` exists until policies land (task 5.8).
    """
    from churnmaster.experiments.runner import parse_split, run_episode
    from churnmaster.metrics import summary_hash

    try:
        split = parse_split(args.seed_split)
    except ValueError as e:
        print(f"churnmaster simulate: {e}", file=sys.stderr)
        return 1
    if args.run < 0 or args.episode < 0:
        print("churnmaster simulate: --run and --episode must be >= 0", file=sys.stderr)
        return 1
    cfg = load_config(*_resolve_paths(args))
    t0, c0 = time.perf_counter(), time.process_time()
    metrics, trace = run_episode(cfg, split, args.run, args.episode, maintenance=False)
    wall_s, cpu_s = time.perf_counter() - t0, time.process_time() - c0

    out = {
        "scenario": cfg.scenario.name, "seed_split": split.name.lower(), "run": args.run,
        "episode": args.episode, "maintenance": args.maintenance,
        **metrics.to_dict(diagnostics=False),
        "config_hash": config_hash(cfg), "trace_hash": trace.trace_hash,
        "workload_hash": trace.workload_hash, "summary_hash": summary_hash(metrics),
        "wall_s": round(wall_s, 3), "cpu_s": round(cpu_s, 3),
    }
    if args.json:
        # NaN metrics become null so the output is strict JSON.
        clean = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in out.items()}
        print(json.dumps(clean, indent=2))
    else:
        width = max(len(k) for k in out)
        for k, v in out.items():
            text = f"{v:.6g}" if isinstance(v, float) else str(v)
            print(f"{k:<{width}}  {text}")
    return 0


COMPARE_PRINT_METRICS = ("success_rate", "p95_latency_all_ms", "pings_per_hour",
                         "evictions_per_hour", "J")
COMPARE_PLOT_METRICS = ("success_rate", "p95_latency_all_ms", "pings_per_hour")


def _split_list(values: list[str] | None, default: str) -> list[str]:
    items: list[str] = []
    for v in values or [default]:
        items.extend(s.strip() for s in v.split(",") if s.strip())
    return list(dict.fromkeys(items))


def _fmt_pm(mean: float, std: float) -> str:
    if math.isnan(mean):
        return "nan"
    digits = 4 if abs(mean) < 10 else 1
    s = f"{mean:.{digits}f}"
    return s if math.isnan(std) else f"{s} ± {std:.{digits}f}"


def cmd_compare(args: argparse.Namespace) -> int:
    """Run (policy, scenario, seed) jobs and print mean ± std per policy."""
    from churnmaster.experiments.runner import (DEFAULT_COMPARE_POLICIES, ConfigLoader, Job,
                                                parse_split, policy_env_mode, results_table,
                                                run_jobs, summary_by_policy)

    def fail(msg: str) -> int:
        print(f"churnmaster compare: {msg}", file=sys.stderr)
        return 1

    try:
        split = parse_split(args.seed_split)
    except ValueError as e:
        return fail(str(e))
    if args.seeds < 1 or args.run < 0:
        return fail("--seeds must be >= 1 and --run >= 0")
    scenarios = _split_list(args.scenario, "high_churn")
    policies = _split_list(args.policies, ",".join(DEFAULT_COMPARE_POLICIES))
    if "matched_random" in policies:
        return fail("matched_random needs action frequencies; not available from the CLI")
    for p in policies:
        try:
            policy_env_mode(p)  # validates the name
        except ValueError as e:
            return fail(str(e))

    loader = ConfigLoader(_configs_root(args), _experiment_path(args))
    for s in scenarios:
        loader(s)  # surface config errors before running anything
    jobs = [Job(p, s, split, args.run, ep)
            for s in scenarios for p in policies for ep in range(args.seeds)]
    t0 = time.perf_counter()
    df = results_table(run_jobs(jobs, cfg_loader=loader))
    wall_s = time.perf_counter() - t0
    summary = summary_by_policy(df)

    print(f"compare: split={split.name.lower()} run={args.run} seeds={args.seeds} "
          f"jobs={len(jobs)} wall={wall_s:.1f}s")
    headers = ["scenario", "policy", "shares_dp", "n", *COMPARE_PRINT_METRICS]
    rows = [[r["scenario"], r["policy"], str(bool(r["shares_decision_points"])), str(r["n"]),
             *(_fmt_pm(r[f"{m}_mean"], r[f"{m}_std"]) for m in COMPARE_PRINT_METRICS)]
            for _, r in summary.iterrows()]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))
    if (~summary["shares_decision_points"]).any():
        print("note: shares_dp=False (kademlia_std) does not share epoch decision points")

    if args.out:
        from churnmaster.plots import plot_policy_bars

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "per_seed.csv", index=False)
        summary.to_csv(out / "summary.csv", index=False)
        for m in COMPARE_PLOT_METRICS:
            plot_policy_bars(df, m, out / f"{m}.png")
        print(f"wrote {out / 'per_seed.csv'}, {out / 'summary.csv'} and "
              f"{len(COMPARE_PLOT_METRICS)} figures")
    return 0


def _stub(name: str):
    def run(args: argparse.Namespace) -> int:
        print(f"churnmaster {name}: not implemented yet", file=sys.stderr)
        return 2
    return run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="churnmaster", description="ChurnMaster DHT maintenance study")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser("show-config", help="print the merged configuration and its hash")
    _add_config_args(p)
    p.set_defaults(func=cmd_show_config)

    p = sub.add_parser("simulate", help="run one maintenance-off episode and print metrics")
    _add_config_args(p)
    p.add_argument("--seed-split", default="test",
                   help="environment split: calibration, train, validation, eval, test (default test)")
    p.add_argument("--run", type=int, default=0, help="run index (default 0; ignored for eval)")
    p.add_argument("--episode", type=int, default=0, help="episode index (default 0)")
    p.add_argument("--maintenance", choices=("off",), default="off",
                   help="maintenance mode; only 'off' until policies exist")
    p.add_argument("--json", action="store_true", help="print metrics as JSON")
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("compare", help="compare policies over seeds")
    _add_config_args(p, scenario=False)
    p.add_argument("--scenario", action="append", default=None,
                   help="scenario name(s); repeatable or comma-separated (default high_churn)")
    p.add_argument("--policies", action="append", default=None,
                   help="comma-separated policy names (default: never_replace,always_replace,"
                        "kademlia_epoch,kademlia_std,heuristic,oracle_heuristic,uniform_random); "
                        "'heuristic:<config id>' and 'none' are accepted")
    p.add_argument("--seed-split", default="calibration",
                   help="environment split (default calibration)")
    p.add_argument("--seeds", type=int, default=3, help="episodes (seeds) per policy (default 3)")
    p.add_argument("--run", type=int, default=0, help="run index (default 0; ignored for eval)")
    p.add_argument("--out", default=None, metavar="DIR",
                   help="write per_seed.csv, summary.csv and bar charts to DIR")
    p.set_defaults(func=cmd_compare)

    helps = {
        "headroom": "run the headroom study",
        "train": "train the Q-learning agent",
        "evaluate": "evaluate policies on test seeds",
        "reproduce": "regenerate all tables and figures",
    }
    for name, text in helps.items():
        p = sub.add_parser(name, help=f"{text} (not implemented yet)")
        _add_config_args(p)
        if name == "reproduce":
            mode = p.add_mutually_exclusive_group()
            mode.add_argument("--quick", action="store_true")
            mode.add_argument("--full", action="store_true")
        p.set_defaults(func=_stub(name))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as e:
        print(f"churnmaster: config error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
