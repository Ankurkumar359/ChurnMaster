"""Command-line entry point (Requirement 26.1).

Subcommands: show-config, simulate, compare, headroom, diagnostics, train,
tune, evaluate, reproduce (``--quick``, ``--full`` or ``--figures-only``),
ablate (exploratory ablations, ``--suite agent``, ``--suite reward`` or
``--suite memoryless``).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

from churnmaster.config import ConfigError, config_hash, load_config, to_dict

COMMANDS = ("show-config", "simulate", "compare", "headroom", "diagnostics", "train", "tune",
            "evaluate", "reproduce", "ablate")


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


def _add_workers_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--workers", type=int, default=1, metavar="N",
                   help="parallel worker processes (default 1 = sequential; results are "
                        "identical for any N)")


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
    if args.workers < 1:
        return fail("--workers must be >= 1")
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
    df = results_table(run_jobs(jobs, cfg_loader=loader, workers=args.workers))
    wall_s = time.perf_counter() - t0
    summary = summary_by_policy(df)

    print(f"compare: split={split.name.lower()} run={args.run} seeds={args.seeds} "
          f"jobs={len(jobs)} workers={args.workers} wall={wall_s:.1f}s")
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


def _fmt_ci(ci: dict, digits: int = 4) -> str:
    return f"{ci['mean']:+.{digits}f} [{ci['lo']:+.{digits}f}, {ci['hi']:+.{digits}f}]"


def cmd_headroom(args: argparse.Namespace) -> int:
    """Headroom study + env freeze (Requirement 22).

    Exit codes: 0 when G1 passes (on the default env or after the tuning
    fallback), even if G2 fails ("limited headroom"); 3 when G1 fails and no
    tuning setting passes (manifest written with ``frozen = false``); 1 on
    usage / config errors.
    """
    from churnmaster.experiments.headroom import (G1_FAILED_EXIT, manifest_path_for,
                                                  run_headroom, write_outputs)
    from churnmaster.experiments.runner import parse_split

    if args.seeds < 1 or args.workers < 1:
        print("churnmaster headroom: --seeds and --workers must be >= 1", file=sys.stderr)
        return 1
    try:
        split = parse_split(args.seed_split)
    except ValueError as e:
        print(f"churnmaster headroom: {e}", file=sys.stderr)
        return 1
    root = _configs_root(args)
    exp = _experiment_path(args) or root / "experiments" / "headroom.toml"
    out = Path(args.out)
    t0 = time.perf_counter()
    rep = run_headroom(root, exp, n_seeds=args.seeds, seed_split=split,
                       manifest_path=manifest_path_for(out),
                       progress=lambda m: print(m, flush=True), workers=args.workers)
    files = write_outputs(rep, out)
    wall = time.perf_counter() - t0

    print(f"headroom: split={split.name.lower()} seeds={args.seeds} wall={wall:.1f}s")
    headers = ["scenario", "policy", "success", "p95_all_ms", "pings/h", "J"]
    rows = [[r["scenario"], r["policy"], _fmt_pm(r["success_rate_mean"], r["success_rate_std"]),
             _fmt_pm(r["p95_latency_all_ms_mean"], r["p95_latency_all_ms_std"]),
             _fmt_pm(r["pings_per_hour_mean"], r["pings_per_hour_std"]),
             _fmt_pm(r["J_mean"], r["J_std"])] for _, r in rep.summary.iterrows()]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))

    g = rep.gates.to_dict()
    g1, g2 = g["G1"], g["G2"]
    print(f"G1 ({g['scenario']}): {'PASS' if g1['passed'] else 'FAIL'} "
          f"route={g1['route']} best={g1['best']}")
    for pol, c in g1["candidates"].items():
        print(f"  {pol:<17} dJ={_fmt_ci(c['dJ'])}  p95 nr-p (ms)={_fmt_ci(c['p95_diff_ms'], 1)}"
              f"  p95_rel={c['p95_rel']:+.1%}  J-route={c['j_route']} p95-route={c['p95_route']}")
    print(f"G2 ({g['scenario']}): {'PASS' if g2['passed'] else 'FAIL (limited headroom)'} "
          f"route={g2['route']} dJ(oracle-kademlia)={_fmt_ci(g2['dJ'])} "
          f"ping_ratio={g2['ping_ratio']:.3f}")
    for scn, info in g["G2_info"].items():
        print(f"  info {scn}: dJ={_fmt_ci(info['dJ'])} ping_ratio={info['ping_ratio']:.3f} "
              f"would_pass={info['passed']}")
    if rep.tuning is not None:
        print(f"tuning fallback: ran {len(rep.tuning.tried)} settings; "
              f"chosen={rep.tuning.chosen}")
    print(f"frozen={rep.manifest['frozen']} env_hashes:")
    for scn, h in rep.manifest["env_hashes"].items():
        print(f"  {scn:<15} {h}")
    print(f"wrote {len(files)} files and {manifest_path_for(out)}")
    return 0 if rep.g1_passed else G1_FAILED_EXIT


def cmd_diagnostics(args: argparse.Namespace) -> int:
    """Reward / coverage diagnostics (Requirement 23).

    Exit codes: 0 when identity, reward sanity and coverage all pass; 4
    (``DIAG_FAILED_EXIT``) when any fails (outputs are still written); 1 on
    usage errors or when the env is not frozen / differs from the manifest.
    """
    from churnmaster.experiments.diagnostics import DIAG_FAILED_EXIT, run_diagnostics, write_outputs
    from churnmaster.experiments.runner import FrozenEnvError, parse_split

    def fail(msg: str) -> int:
        print(f"churnmaster diagnostics: {msg}", file=sys.stderr)
        return 1

    if args.episodes < 1 or args.horizon < 1 or args.workers < 1:
        return fail("--episodes, --horizon and --workers must be >= 1")
    try:
        split = parse_split(args.seed_split)
    except ValueError as e:
        return fail(str(e))
    scenarios = _split_list(args.scenario, "high_churn,stable")
    t0 = time.perf_counter()
    try:
        rep = run_diagnostics(_configs_root(args), Path(args.manifest),
                              experiment_path=_experiment_path(args), n_episodes=args.episodes,
                              split=split, scenarios=scenarios, gamma=args.gamma,
                              horizon=args.horizon, per_seed_csv=args.per_seed_csv,
                              progress=(lambda m: print(m, flush=True)) if args.verbose else None,
                              workers=args.workers)
    except FrozenEnvError as e:
        return fail(str(e))
    files = write_outputs(rep, Path(args.out))
    wall = time.perf_counter() - t0

    print(f"diagnostics: policy={rep.policy} split={split.name.lower()} "
          f"episodes/scenario={rep.n_episodes} gamma={rep.gamma} H={rep.horizon} wall={wall:.1f}s")
    for scope, idr in rep.identity.items():
        print(f"identity {scope:<10} {'PASS' if idr.passed else 'FAIL'} "
              f"{idr.n_passed}/{idr.n_episodes} max|J-score|={idr.max_abs_err_J:.3g} "
              f"max|total|={idr.max_abs_err_total:.3g}")
    for scope, sr in rep.sanity.items():
        print(f"reward sanity {scope}: {'PASS' if sr.passed else 'FAIL'} (n={sr.n_samples})")
        for c in sr.comparisons:
            ci = "n/a" if c.ci is None else _fmt_ci(c.ci.to_dict(), 4)
            tag = "primary" if c.primary else "info"
            print(f"  {c.name:<32} {tag:<7} n={c.n_better}/{c.n_worse} "
                  f"G={c.mean_better:.4f}/{c.mean_worse:.4f} diff={ci} "
                  f"{'PASS' if c.passed else 'FAIL'}")
    for scope, cv in rep.coverage.items():
        print(f"coverage {scope:<10} {'PASS' if cv.passed else 'FAIL'} "
              f"fraction={cv.fraction:.3f} (target {cv.target:.2f}, >= {cv.min_visits} visits) "
              f"states visited={cv.n_states_visited} covered={cv.n_states_covered} "
              f"decisions={cv.n_decisions}")
    for s in rep.spearman:
        print(f"spearman J~success {s.level:<12} {s.scope:<10} n={s.n:<4} "
              f"rho={s.rho:+.3f} p={s.p:.3g}")
    print(f"wrote {len(files)} files to {args.out}")
    return 0 if rep.all_passed else DIAG_FAILED_EXIT


def cmd_train(args: argparse.Namespace) -> int:
    """Train the Q-learning agent (Requirement 24): ``--mode debug|grid|final``.

    Exit codes: 0 on success; 1 on usage errors or a frozen-env mismatch.
    """
    from churnmaster.experiments.runner import FrozenEnvError
    from churnmaster.experiments.train import run_training

    def fail(msg: str) -> int:
        print(f"churnmaster train: {msg}", file=sys.stderr)
        return 1

    for name in ("episodes", "runs", "validate_every", "validation_seeds"):
        v = getattr(args, name)
        if v is not None and v < 1:
            return fail(f"--{name.replace('_', '-')} must be >= 1")
    if args.workers < 1:
        return fail("--workers must be >= 1")
    root = _configs_root(args)
    exp = _experiment_path(args) or root / "experiments" / "train.toml"
    out = Path(args.out) if args.out else Path("results/train") / args.mode
    try:
        rep = run_training(args.mode, root, experiment_path=exp, manifest_path=Path(args.manifest),
                           out_dir=out, episodes=args.episodes, runs=args.runs, gamma=args.gamma,
                           alpha_min=args.alpha_min, validate_every=args.validate_every,
                           validation_seeds=args.validation_seeds, grid_dir=args.grid_dir,
                           allow_unfrozen=args.allow_unfrozen, workers=args.workers,
                           log_every=args.log_every,
                           progress=lambda m: print(m, flush=True))
    except FrozenEnvError as e:
        return fail(str(e))
    except ValueError as e:
        return fail(str(e))

    print(f"train: mode={rep.mode} scenarios={','.join(rep.scenarios)} "
          f"validation_seeds={rep.validation_seeds} wall={rep.wall_s:.1f}s"
          + ("" if rep.frozen_checked else " (UNFROZEN: env check skipped)"))
    ref = rep.references.groupby(["scenario", "policy"], sort=False)["J"].mean()
    for (scn, pol), j in ref.items():
        print(f"  reference {scn:<11} {pol:<17} J={j:.4f}")
    for rr in rep.runs:
        means = rr.validation.groupby(["checkpoint", "scenario"])["J"].mean().unstack()
        print(f"run {rr.settings.tag}/run{rr.run} ({rr.wall_s:.0f}s train): validation J by checkpoint")
        for c, row in means.iterrows():
            print(f"  ep {int(c):>4}  " + "  ".join(f"{s}={row[s]:.4f}" for s in means.columns))
        i4 = rr.i4
        for name, c in i4["comparisons"].items():
            print(f"  I4 final-vs-{name:<15} dJ={_fmt_ci(c)} beats={c['beats']}")
        dq = i4["dq"]
        print(f"  I4 |dQ| first{dq['window']}={dq['first_mean']:.4g} last{dq['window']}="
              f"{dq['last_mean']:.4g} shrinks={dq['shrinks']}  -> I4 {'PASS' if i4['passed'] else 'FAIL'}")
        g = rr.greedy
        w = g["actions"]["weighted"]
        print(f"  greedy (visit-weighted): KEEP={w['KEEP']:.3f} PROBE={w['PROBE']:.3f} "
              f"REPLACE={w['REPLACE']:.3f}; inc_fail>=2 not-KEEP="
              f"{g['inc_fail_ge2']['weighted']['frac_not_keep']:.3f}; healthy KEEP="
              f"{g['healthy']['weighted']['frac_keep']:.3f}")
    if rep.selected is not None:
        print(rep.grid.to_string(index=False))
        print(f"selected: gamma={rep.selected['gamma']} alpha_min={rep.selected['alpha_min']} "
              f"({rep.selected['reason']})")
    print(f"wrote outputs to {rep.out_dir}")
    return 0


def cmd_tune(args: argparse.Namespace) -> int:
    """Tune the heuristic family (Requirement 25.1).

    Exit codes: 0 on success; 1 on usage errors or a frozen-env mismatch.
    """
    from churnmaster.experiments.runner import FrozenEnvError
    from churnmaster.experiments.tune_heuristic import run_tuning, write_outputs

    def fail(msg: str) -> int:
        print(f"churnmaster tune: {msg}", file=sys.stderr)
        return 1

    if (args.seeds is not None and args.seeds < 1) or args.workers < 1:
        return fail("--seeds and --workers must be >= 1")
    root = _configs_root(args)
    exp = _experiment_path(args) or root / "experiments" / "tune.toml"
    try:
        rep = run_tuning(root, Path(args.manifest), experiment_path=exp, n_seeds=args.seeds,
                         workers=args.workers, progress=lambda m: print(m, flush=True))
    except (FrozenEnvError, ValueError) as e:
        return fail(str(e))
    files = write_outputs(rep, Path(args.out))

    sel = rep.selection
    scns = sel.scenarios
    print(f"tune: split={rep.seeds['split']} run={rep.seeds['run']} seeds={rep.seeds['n_seeds']} "
          f"jobs={rep.n_jobs} wall={rep.wall_s:.1f}s")
    headers = ["rank", "config", "J_overall", *(f"J {s}" for s in scns), "success",
               "p95_all_ms", "pings/h"]
    rows = []
    for _, r in rep.summary.head(args.top).iterrows():
        rows.append([str(r["rank"]), r["config"] + (" (default)" if r["is_default"] else ""),
                     f"{r['J_overall']:.4f}",
                     *(_fmt_pm(r[f"J_mean_{s}"], r[f"J_std_{s}"]) for s in scns),
                     f"{r['success_rate_overall']:.4f}", f"{r['p95_latency_all_ms_overall']:.1f}",
                     f"{r['pings_per_hour_overall']:.1f}"])
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))
    d = rep.selected_json()
    print(f"global best (tuned_heuristic): {sel.global_id}")
    for s, c in sel.per_scenario.items():
        print(f"privileged best {s}: {c}")
    print(f"default {d['default_config']} rank: {d['default_rank']}")
    print(f"wrote {', '.join(str(f) for f in files)}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Locked evaluation (Requirements 25.2, 25.3, 25.6).

    ``--create-lock`` only writes the eval-split lock into the manifest (a
    separate, explicit step). Without it, the lock must exist and match.
    Exit codes: 0 on success; 1 on usage errors, frozen-env mismatch,
    missing / mismatched lock (``SplitLockError``, nothing written) or an
    already recorded evaluation without ``--allow-rerun``.
    """
    from churnmaster.experiments.evaluate import (EvalRerunError, SplitLockError,
                                                  create_lock_from_configs, evaluate)
    from churnmaster.experiments.runner import FrozenEnvError

    def fail(msg: str) -> int:
        print(f"churnmaster evaluate: {msg}", file=sys.stderr)
        return 1

    if (args.seeds is not None and args.seeds < 1) or args.workers < 1:
        return fail("--seeds and --workers must be >= 1")
    root = _configs_root(args)
    exp = _experiment_path(args) or root / "experiments" / "eval.toml"
    try:
        if args.create_lock:
            lock = create_lock_from_configs(root, Path(args.manifest), exp, n_seeds=args.seeds)
            print(f"eval lock in {args.manifest}: split={lock['split']} run={lock['run']} "
                  f"episodes={lock['episodes'][0]}..{lock['episodes'][-1]} "
                  f"(n={len(lock['episodes'])}) scenarios={','.join(lock['scenarios'])} "
                  f"root_seed={lock['root_seed']} seed_list_hash={lock['seed_list_hash'][:16]} "
                  f"created={lock.get('created_utc')}")
            return 0
        rep = evaluate(root, Path(args.manifest), Path(args.out), q_paths=args.rl_q,
                       tune_selected=args.tune_selected, matched_freqs=args.matched_freqs,
                       experiment_path=exp, n_seeds=args.seeds, workers=args.workers,
                       allow_rerun=args.allow_rerun, progress=lambda m: print(m, flush=True))
    except (FrozenEnvError, SplitLockError, EvalRerunError, ValueError, FileNotFoundError) as e:
        return fail(str(e))

    s = rep.summary
    headers = ["scenario", "policy", "n", "J", "success", "p95_all_ms", "pings/h", "evict/h"]
    cols = ("J", "success_rate", "p95_latency_all_ms", "pings_per_hour", "evictions_per_hour")
    rows = [[r["scenario"], r["policy"] + (" (priv*)" if r["privileged_fallback"] else
                                          " (priv)" if r["privileged"] else
                                          " (ref)" if r["reference"] else
                                          " (no-dp)" if not r["shares_decision_points"] else ""),
             str(r["n"]), *(_fmt_pm(r[f"{m}_mean"], r[f"{m}_std"]) for m in cols)]
            for _, r in s.iterrows()]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))
    print("primary tests (rl_mean - comparator, J; Holm over 9):")
    for _, r in rep.primary.iterrows():
        print(f"  {r['scenario']:<15} vs {r['comparator']:<16} dJ={r['mean_diff']:+.4f} "
              f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] p={r['p']:.3g} p_holm={r['p_holm']:.3g} "
              f"significant={bool(r['significant'])}")
    for _, r in rep.hypotheses.iterrows():
        print(f"{r['id']}: {r['verdict'].upper()} - {r['statement']}\n    {r['evidence']}")
    print(f"evaluate: {rep.n_jobs} episodes, wall={rep.wall_s:.1f}s; wrote {len(rep.files)} files "
          f"to {args.out}")
    return 0


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
    _add_workers_arg(p)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("headroom", help="run the headroom gate study and freeze the env")
    _add_config_args(p, scenario=False)
    p.add_argument("--seeds", type=int, default=10,
                   help="calibration episodes (seeds) per policy and scenario (default 10)")
    p.add_argument("--seed-split", default="calibration",
                   help="environment split (default calibration)")
    p.add_argument("--out", default="results/headroom", metavar="DIR",
                   help="output directory (default results/headroom)")
    _add_workers_arg(p)
    p.set_defaults(func=cmd_headroom)

    p = sub.add_parser("diagnostics", help="reward-identity, reward-sanity, coverage and "
                                           "J/success diagnostics (uniform_random)")
    _add_config_args(p, scenario=False)
    p.add_argument("--scenario", action="append", default=None,
                   help="scenario name(s); repeatable or comma-separated (default high_churn,stable)")
    p.add_argument("--episodes", type=int, default=20,
                   help="uniform_random episodes per scenario (default 20)")
    p.add_argument("--seed-split", default="calibration",
                   help="environment split (default calibration)")
    p.add_argument("--gamma", type=float, default=0.9, help="discount for returns (default 0.9)")
    p.add_argument("--horizon", type=int, default=10, help="return horizon in epochs (default 10)")
    p.add_argument("--manifest", default="results/headroom/manifest.json",
                   help="frozen env manifest (default results/headroom/manifest.json)")
    p.add_argument("--per-seed-csv", default="results/headroom/per_seed.csv",
                   help="per-seed table for the J/success Spearman (run if missing)")
    p.add_argument("--out", default="results/diagnostics", metavar="DIR",
                   help="output directory (default results/diagnostics)")
    p.add_argument("--verbose", action="store_true", help="print per-episode progress")
    _add_workers_arg(p)
    p.set_defaults(func=cmd_diagnostics)

    p = sub.add_parser("train", help="train the Q-learning agent (debug, grid or final mode)")
    _add_config_args(p, scenario=False)
    p.add_argument("--mode", choices=("debug", "grid", "final"), default="debug",
                   help="debug (default): high_churn 1x100; grid: gamma x alpha_min; "
                        "final: 3 runs with the grid selection")
    p.add_argument("--episodes", type=int, default=None, help="override training episodes")
    p.add_argument("--runs", type=int, default=None, help="override number of runs")
    p.add_argument("--gamma", type=float, default=None, help="override gamma")
    p.add_argument("--alpha-min", type=float, default=None, help="override alpha_min")
    p.add_argument("--validate-every", type=int, default=None, help="override checkpoint period")
    p.add_argument("--validation-seeds", type=int, default=None,
                   help="override validation seeds per scenario")
    p.add_argument("--grid-dir", default=None, metavar="DIR",
                   help="grid output with selected.json (final mode; default <out>/../grid)")
    p.add_argument("--manifest", default="results/headroom/manifest.json",
                   help="frozen env manifest (default results/headroom/manifest.json)")
    p.add_argument("--allow-unfrozen", action="store_true",
                   help="skip the frozen-env check (tests / debugging only)")
    p.add_argument("--out", default=None, metavar="DIR",
                   help="output directory (default results/train/<mode>)")
    p.add_argument("--log-every", type=int, default=0, metavar="N",
                   help="print a training progress line every N episodes (default off)")
    _add_workers_arg(p)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("tune", help="tune the 32 heuristic configs on train seeds")
    _add_config_args(p, scenario=False)
    p.add_argument("--seeds", type=int, default=None,
                   help="train episodes per config and scenario (default: tune.toml n_seeds, 10)")
    p.add_argument("--manifest", default="results/headroom/manifest.json",
                   help="frozen env manifest (default results/headroom/manifest.json)")
    p.add_argument("--out", default="results/tune", metavar="DIR",
                   help="output directory (default results/tune)")
    p.add_argument("--top", type=int, default=5, help="configs to print (default 5)")
    _add_workers_arg(p)
    p.set_defaults(func=cmd_tune)

    p = sub.add_parser("evaluate", help="locked evaluation on the eval split (all policies, "
                                        "all RL runs) with statistics and H1-H5")
    _add_config_args(p, scenario=False)
    p.add_argument("--create-lock", action="store_true",
                   help="only write the eval-split lock into the manifest (separate step)")
    p.add_argument("--allow-rerun", action="store_true",
                   help="run even if an evaluation is already recorded (results are identical)")
    p.add_argument("--seeds", type=int, default=None,
                   help="eval episodes per scenario (default: eval.toml n_seeds, 20)")
    p.add_argument("--manifest", default="results/headroom/manifest.json",
                   help="frozen env manifest holding the eval lock")
    p.add_argument("--rl-q", nargs="+", default=None, metavar="PATH",
                   help="final Q-tables (default results/train/final/run{0,1,2}/q_final.npy)")
    p.add_argument("--tune-selected", default="results/tune/selected.json",
                   help="heuristic tuning selection (default results/tune/selected.json)")
    p.add_argument("--matched-freqs", default="results/matched/freqs.json",
                   help="matched_random frequencies (default results/matched/freqs.json)")
    p.add_argument("--out", default="results/eval", metavar="DIR",
                   help="output directory (default results/eval)")
    _add_workers_arg(p)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("reproduce", help="regenerate tables and figures (--quick, --full or "
                                         "--figures-only)")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true",
                      help="1 run x 40 episodes, 5 eval seeds (< 10 min)")
    mode.add_argument("--full", action="store_true",
                      help="headroom -> diagnostics -> train -> tune -> evaluate -> stats -> plots")
    mode.add_argument("--figures-only", action="store_true",
                      help="regenerate figures and tables from existing results (no simulation)")
    p.add_argument("--out", default=None, metavar="DIR",
                   help="output directory (default results/reproduce_quick, "
                        "results/reproduce_full or results/figures); canonical results "
                        "directories are refused")
    p.add_argument("--results-dir", default="results", metavar="DIR",
                   help="input results root for --figures-only (default results)")
    p.add_argument("--configs-dir", default=None, help="override the configs directory")
    p.add_argument("--workers", type=int, default=None, metavar="N",
                   help="parallel worker processes for --quick / --full (default "
                        "min(cpu_count - 1, 8); results are identical for any N)")
    p.set_defaults(func=cmd_reproduce)

    p = sub.add_parser("ablate", help="exploratory ablations on TEST seeds (separate from the "
                                      "locked evaluation)")
    _add_config_args(p, scenario=False)
    p.add_argument("--suite", choices=("agent", "reward", "memoryless"), default="agent",
                   help="ablation suite (default agent: gamma0, no_probe, shuffled_reward, "
                        "buffered_control, drop_<feature>; reward: w_M x w_L sweep scored "
                        "under the canonical weights; memoryless: canonical policies on the "
                        "exponential-session scenario heldout_memoryless vs heldout_medium)")
    p.add_argument("--variants", default=None,
                   help="comma-separated subset of the suite's variants / reward cells / "
                        "memoryless reference policies (default: all)")
    p.add_argument("--episodes", type=int, default=None,
                   help="override training episodes (default 200; shortened runs skip the "
                        "canonical-settings check)")
    p.add_argument("--seeds", type=int, default=20, help="TEST episodes per scenario (default 20)")
    p.add_argument("--manifest", default="results/headroom/manifest.json",
                   help="frozen env manifest (read only)")
    p.add_argument("--rl-q", nargs="+", default=None, metavar="PATH",
                   help="canonical Q-tables (default results/train/final/run{0,1,2}/q_final.npy)")
    p.add_argument("--tune-selected", default="results/tune/selected.json",
                   help="heuristic tuning selection (default results/tune/selected.json)")
    p.add_argument("--out", default="results/ablations", metavar="DIR",
                   help="output root (default results/ablations); canonical dirs are refused")
    p.add_argument("--overwrite", action="store_true",
                   help="replace an existing suite output")
    p.add_argument("--log-every", type=int, default=0, metavar="N",
                   help="print a training progress line every N episodes (default off)")
    p.add_argument("--workers", type=int, default=None, metavar="N",
                   help="parallel worker processes (default min(cpu_count - 1, 8); results "
                        "are identical for any N)")
    p.set_defaults(func=cmd_ablate)
    return parser


def cmd_ablate(args: argparse.Namespace) -> int:
    """Exploratory ablations (Requirement 28.1). Exit 0 ok, 1 on usage / input errors."""
    from churnmaster.experiments.ablations import (AblationSeedError, run_agent_suite,
                                                   run_memoryless_suite, run_reward_suite)
    from churnmaster.experiments.runner import FrozenEnvError, default_workers

    def fail(msg: str) -> int:
        print(f"churnmaster ablate: {msg}", file=sys.stderr)
        return 1

    workers = default_workers() if args.workers is None else args.workers
    if workers < 1 or args.seeds < 1 or (args.episodes is not None and args.episodes < 1):
        return fail("--workers, --seeds and --episodes must be >= 1")
    variants = [v.strip() for v in args.variants.split(",") if v.strip()] if args.variants else None
    root = _configs_root(args)
    exp = _experiment_path(args) or root / "experiments" / "train.toml"
    kw = dict(experiment_path=exp, episodes=args.episodes, n_seeds=args.seeds,
              canonical_q=args.rl_q, tune_selected=args.tune_selected, workers=workers,
              overwrite=args.overwrite, log_every=args.log_every,
              progress=lambda m: print(m, flush=True))
    try:
        if args.suite == "memoryless":
            if args.episodes is not None:
                return fail("--episodes does not apply to --suite memoryless (no training)")
            for k in ("episodes", "log_every"):
                kw.pop(k)
            rep = run_memoryless_suite(root, Path(args.manifest), Path(args.out),
                                       policies=variants, **kw)
            _print_memoryless_table(rep)
            print(f"ablate[{rep.suite}]: wall={rep.wall_s:.1f}s (eval {rep.eval_wall_s:.1f}s); "
                  f"wrote {len(rep.files)} files to {rep.out_dir}")
            return 0
        if args.suite == "reward":
            rep = run_reward_suite(root, Path(args.manifest), Path(args.out), cells=variants, **kw)
        else:
            rep = run_agent_suite(root, Path(args.manifest), Path(args.out), variants=variants,
                                  **kw)
    except (FrozenEnvError, AblationSeedError, ValueError, FileNotFoundError) as e:
        return fail(str(e))
    if args.suite == "reward":
        _print_reward_table(rep)
        print(f"ablate[{rep.suite}]: wall={rep.wall_s:.1f}s (train {rep.train_wall_s:.1f}s, "
              f"eval {rep.eval_wall_s:.1f}s); wrote {len(rep.files)} files to {rep.out_dir}")
        return 0
    c = rep.comparison
    print("exploratory; p uncorrected. dX = rl_mean(variant) - reference, paired over TEST seeds")
    for _, r in c[c["metric"] == "J"].iterrows():
        print(f"  {r['comparison']:<20} {r['variant']:<20} vs {r['reference']:<16} "
              f"{r['scenario']:<15} dJ={r['mean_diff']:+.4f} [{r['ci_lo']:+.4f}, "
              f"{r['ci_hi']:+.4f}] p={r['p_uncorrected']:.3g}")
    print(f"ablate[{rep.suite}]: wall={rep.wall_s:.1f}s (train {rep.train_wall_s:.1f}s, eval "
          f"{rep.eval_wall_s:.1f}s); wrote {len(rep.files)} files to {rep.out_dir}")
    return 0


def _print_memoryless_table(rep) -> None:
    """Per policy x scenario means, then RL vs references and the gap shift."""
    s, c = rep.summary, rep.comparison
    print("exploratory; p uncorrected; RL = rl_mean over the canonical runs, TEST seeds")
    for _, r in s[s["kind"] != "run"].iterrows():
        print(f"  {r['policy']:<17} {r['scenario']:<19} J={r['J_mean']:.4f} "
              f"p95={r['p95_latency_all_ms_mean']:7.1f} success={r['success_rate_mean']:.4f} "
              f"pings/h={r['pings_per_hour_mean']:.1f}")
    for _, r in c[(c["metric"] == "J") & (c["comparison"] != "shape_effect")].iterrows():
        print(f"  {r['comparison']:<16} {r['variant']} vs {r['reference']:<16} "
              f"{r['scenario']:<34} dJ={r['mean_diff']:+.4f} [{r['ci_lo']:+.4f}, "
              f"{r['ci_hi']:+.4f}] p={r['p_uncorrected']:.3g}")


def _print_reward_table(rep) -> None:
    """Per cell x scenario: RL p95, Δp95 / ΔJ vs canonical [CI], gaps to tuned_heuristic."""
    c, s = rep.comparison, rep.summary
    print("exploratory; p uncorrected; J under the canonical weights; "
          "d = rl_mean(cell) - reference, paired over TEST seeds")

    def get(comp: str, cell: str, scn: str, m: str):
        r = c[(c["comparison"] == comp) & (c["variant"] == cell) & (c["scenario"] == scn)
              & (c["metric"] == m)]
        return None if r.empty else r.iloc[0]

    for _, row in s[s["kind"] == "rl_mean"].iterrows():
        g, scn = row["group"], row["scenario"]
        dp, dj = get("vs_canonical", g, scn, "p95_latency_all_ms"), get("vs_canonical", g, scn, "J")
        gj, gp = (get("vs_tuned_heuristic", g, scn, "J"),
                  get("vs_tuned_heuristic", g, scn, "p95_latency_all_ms"))
        fmt = (lambda r, d: "-" if r is None else
               f"{r['mean_diff']:+.{d}f} [{r['ci_lo']:+.{d}f}, {r['ci_hi']:+.{d}f}]")
        print(f"  {row['cell']:<13} {scn:<15} p95={row['p95_latency_all_ms_mean']:7.1f} "
              f"dp95={fmt(dp, 1)} dJ={fmt(dj, 4)} gapJ={fmt(gj, 4)} gapP95={fmt(gp, 1)} "
              f"pings/h={row['pings_per_hour_mean']:.1f}")
    if rep.trend is not None:
        for _, t in rep.trend.iterrows():
            print(f"  trend {t['scenario']:<15} rho(w_L, p95) = {t['spearman_rho']:+.2f} "
                  f"(n={t['n_points']}, descriptive)")


def _reproduce_pipeline(args: argparse.Namespace, mode: str) -> int:
    from churnmaster.experiments.reproduce import run_pipeline

    out = Path(args.out) if args.out else Path(f"results/reproduce_{mode}")
    t0 = time.perf_counter()
    try:
        res = run_pipeline(mode, out, configs_dir=args.configs_dir, workers=args.workers,
                           progress=lambda m: print(m, flush=True), args=args)
    except ValueError as e:
        print(f"churnmaster reproduce: {e}", file=sys.stderr)
        return 1
    wall = time.perf_counter() - t0
    m = res.manifest
    print(f"reproduce --{mode}: status={m['status']} exit={res.exit_code} wall={wall:.1f}s "
          f"workers={m['workers']} out={res.out_dir}")
    for s in m["stages"]:
        print(f"  {s['name']:<12} {s['status']:<9} {s['wall_s']:>8.1f}s"
              + (f"  {s['error']}" if "error" in s else ""))
    hyp = next((s.get("hypotheses") for s in m["stages"] if s["name"] == "evaluate"), None)
    if hyp:
        print("  hypotheses: " + ", ".join(f"{k}={v}" for k, v in hyp.items()))
    v = m.get("verify")
    if v:
        print(f"  verify vs {v.get('canonical_root', 'canonical')}: {v['status']}")
        ps = v.get("per_seed") or {}
        if ps:
            print(f"    per_seed.csv sha256_equal={ps['sha256_equal']} keys_equal={ps['keys_equal']} "
                  f"columns_equal={ps['columns_equal']} differing={ps['differing_columns']}")
            for c in ps["differing_columns"]:
                d = ps["max_abs_diff"].get(c, ps["n_text_diff"].get(c))
                print(f"      {c}: {d}")
        for rel, sv in (v.get("selections") or {}).items():
            print(f"    {rel}: {sv['status']}" + (f" {sv['differences']}" if sv.get("differences")
                                                   else ""))
    print(f"wrote {res.out_dir / 'reproduce_manifest.json'}")
    return res.exit_code


def cmd_reproduce(args: argparse.Namespace) -> int:
    """Reproduction (Requirement 26). Exit codes: 0 ok; 1 usage / input errors;
    3 ``--full`` output differs from the canonical results; 5 a pipeline stage failed."""
    from churnmaster.experiments.reproduce import figures_only

    def fail(msg: str, code: int = 1) -> int:
        print(f"churnmaster reproduce: {msg}", file=sys.stderr)
        return code

    if args.workers is not None and args.workers < 1:
        return fail("--workers must be >= 1")
    if args.quick or args.full:
        return _reproduce_pipeline(args, "quick" if args.quick else "full")
    out = Path(args.out) if args.out else Path("results/figures")
    try:
        files = figures_only(args.results_dir, out, progress=lambda m: print(m, flush=True))
    except (ValueError, FileNotFoundError) as e:
        return fail(str(e))
    man = json.loads(files[-1].read_text(encoding="utf-8"))
    print(f"reproduce --figures-only: wrote {len(files) - 1} files + manifest to {out}; "
          f"skipped {len(man['skipped'])}")
    for s in man["skipped"]:
        print(f"  skipped {s['kind']} {s['output']}: {s['reason']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as e:
        print(f"churnmaster: config error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
