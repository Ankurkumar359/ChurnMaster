"""Tests for experiments/evaluate.py (Requirements 2.3, 25.2, 25.3, 25.6; task 9.3)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churnmaster.cli import main
from churnmaster.experiments.evaluate import (LOCK_KEY, RUNS_KEY, USED_SPLITS, EvalRerunError,
                                              SplitLockError, build_eval_lock, check_eval_disjoint,
                                              check_eval_lock, create_eval_lock,
                                              create_lock_from_configs, evaluate, used_seed_states)
from churnmaster.experiments.matched import write_matched_freqs
from churnmaster.experiments.runner import ConfigLoader, freeze_env, read_manifest, write_manifest
from churnmaster.rl.qlearning import QAgent
from churnmaster.seeding import Split, eval_seed, seed_state

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SHORT = "[sim]\nwarmup_s = 60\nmeasured_s = 240\n"
SCENARIOS = ("stable", "high_churn", "heldout_medium")
GLOBAL = "f=2,po=0,rs=1,ps=0,tg=all"
HC_PRIV = "f=2,po=0,rs=1,ps=0,tg=top_two"
N_RL = 2


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    d = tmp_path_factory.mktemp("evaluate")
    exp = d / "short.toml"
    exp.write_text(SHORT, encoding="utf-8")
    loader = ConfigLoader(CONFIGS, exp)
    cfgs = {s: loader(s) for s in SCENARIOS}
    qs = []
    for i in range(N_RL):
        ag = QAgent()
        ag.Q = np.random.default_rng(10 + i).normal(size=ag.Q.shape)
        qs.append(ag.save(d / f"run{i}" / "q_final.npy"))
    sel = d / "selected.json"
    sel.write_text(json.dumps({"global": GLOBAL, "per_scenario": {"stable": GLOBAL,
                                                                  "high_churn": HC_PRIV}}),
                   encoding="utf-8")
    fr = write_matched_freqs(d / "freqs.json", {s: {False: (0.6, 0.4, 0.0), True: (0.4, 0.3, 0.3)}
                                                for s in SCENARIOS})
    return {"dir": d, "exp": exp, "cfgs": cfgs, "qs": qs, "sel": sel, "freqs": fr}


def _manifest(base, path: Path) -> Path:
    freeze_env(path, base["cfgs"])
    return path


def _kw(base):
    return dict(q_paths=base["qs"], tune_selected=base["sel"], matched_freqs=base["freqs"],
                experiment_path=base["exp"], n_seeds=2, n_resamples=200)


# --------------------------------------------------------------------------- #
# Lock
# --------------------------------------------------------------------------- #

def test_missing_lock_raises_and_writes_nothing(base, tmp_path):
    man = _manifest(base, tmp_path / "manifest.json")
    out = tmp_path / "out"
    with pytest.raises(SplitLockError, match="not locked"):
        evaluate(CONFIGS, man, out, **_kw(base))
    assert not out.exists()
    with pytest.raises(SplitLockError):
        check_eval_lock(tmp_path / "missing.json", base["cfgs"]["stable"], [0, 1], SCENARIOS)
    assert RUNS_KEY not in read_manifest(man)


def test_lock_mismatches_raise(base, tmp_path):
    man = _manifest(base, tmp_path / "manifest.json")
    cfg = base["cfgs"]["stable"]
    lock = create_eval_lock(man, cfg, [0, 1], SCENARIOS)
    assert lock["episodes"] == [0, 1] and lock["root_seed"] == cfg.root_seed
    assert check_eval_lock(man, cfg, [0, 1], list(reversed(SCENARIOS))) == lock
    with pytest.raises(SplitLockError, match="episodes"):
        check_eval_lock(man, cfg, [0, 1, 2], SCENARIOS)
    with pytest.raises(SplitLockError, match="scenarios"):
        check_eval_lock(man, cfg, [0, 1], SCENARIOS[:2])
    with pytest.raises(SplitLockError, match="root_seed"):
        check_eval_lock(man, dataclasses.replace(cfg, root_seed=cfg.root_seed + 1), [0, 1], SCENARIOS)
    m = read_manifest(man)
    m[LOCK_KEY]["seed_states"]["1"][0] ^= 1          # tamper with one fingerprint
    write_manifest(man, m)
    with pytest.raises(SplitLockError, match="seed_states"):
        check_eval_lock(man, cfg, [0, 1], SCENARIOS)
    out = tmp_path / "out"
    with pytest.raises(SplitLockError):
        evaluate(CONFIGS, man, out, **_kw(base))
    assert not out.exists()


def test_create_lock_idempotent_and_refuses_overwrite(base, tmp_path):
    man = _manifest(base, tmp_path / "manifest.json")
    cfg = base["cfgs"]["high_churn"]
    first = create_eval_lock(man, cfg, [0, 1], SCENARIOS)
    again = create_eval_lock(man, cfg, [0, 1], SCENARIOS)
    assert again == first and again["created_utc"] == first["created_utc"]
    with pytest.raises(SplitLockError, match="refusing to overwrite"):
        create_eval_lock(man, cfg, [0, 1, 2], SCENARIOS)
    assert read_manifest(man)[LOCK_KEY] == first
    assert read_manifest(man)["frozen"] is True      # rest of the manifest preserved


def test_eval_seeds_disjoint_from_other_splits(base):
    root = base["cfgs"]["stable"].root_seed
    eps = range(20)
    other = used_seed_states(root)
    names = {v.split("/")[0] for v in other.values()}
    assert names == {"calibration", "train", "tuning", "validation"}
    assert len(other) == 20 + 3 * 200 + 10 + 5
    evals = {seed_state(eval_seed(root, e)) for e in eps}
    assert len(evals) == 20 and evals.isdisjoint(other)
    assert check_eval_disjoint(root, eps)["overlap"] is False
    # An injected overlapping split is detected.
    with pytest.raises(SplitLockError, match="overlaps"):
        check_eval_disjoint(root, eps, USED_SPLITS + (("fake", Split.EVAL, (0,), range(3, 4)),))
    with pytest.raises(ValueError):
        build_eval_lock(root, [0, 0], SCENARIOS)


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def e2e(base, tmp_path_factory):
    d = tmp_path_factory.mktemp("e2e")
    man = _manifest(base, d / "manifest.json")
    create_lock_from_configs(CONFIGS, man, base["exp"], n_seeds=2)
    rep = evaluate(CONFIGS, man, d / "out", **_kw(base))
    return d, man, rep


def test_end_to_end_outputs(e2e):
    d, man, rep = e2e
    out = d / "out"
    for f in ("per_seed.csv", "per_run.csv", "primary_tests.csv", "summary.csv", "hypotheses.csv",
              "hypotheses.md", "eval_manifest.json", "J.png", "p95_latency_all_ms.png",
              "pings_per_hour.png", *(f"success_vs_pings_{s}.png" for s in SCENARIOS)):
        assert (out / f).is_file(), f
    df = pd.read_csv(out / "per_seed.csv")
    labels = ["never_replace", "always_replace", "uniform_random", "matched_random",
              "kademlia_epoch", "kademlia_std", "heuristic", "tuned_heuristic",
              "tuned_heuristic_privileged", "oracle_heuristic", "rl_run0", "rl_run1"]
    assert list(dict.fromkeys(df["policy"])) == labels
    assert len(df) == len(labels) * len(SCENARIOS) * 2
    assert set(df["split"]) == {"eval"} and set(df["run"]) == {0} and set(df["episode"]) == {0, 1}
    # Identical eval environments for every label (same churn trace per scenario × episode).
    assert (df.groupby(["scenario", "episode"])["trace_hash"].nunique() == 1).all()

    def impl(label, scn):
        return df[(df["policy"] == label) & (df["scenario"] == scn)]["policy_impl"].iloc[0]

    assert impl("tuned_heuristic", "heldout_medium") == f"heuristic:{GLOBAL}"
    assert impl("tuned_heuristic_privileged", "high_churn") == f"heuristic:{HC_PRIV}"
    assert impl("rl_run1", "stable") == "learned"
    priv = df[df["policy"] == "tuned_heuristic_privileged"]
    assert priv["privileged"].all()
    assert priv.groupby("scenario")["privileged_fallback"].first().to_dict() == {
        "heldout_medium": True, "high_churn": False, "stable": False}
    assert df.loc[df["policy"] == "oracle_heuristic", "reference"].all()
    assert not df.loc[df["policy"] == "kademlia_std", "shares_decision_points"].any()
    assert df.loc[df["policy"] != "kademlia_std", "shares_decision_points"].all()
    # Same impl + same env => identical results across labels.
    a = df[(df["policy"] == "tuned_heuristic") & (df["scenario"] == "stable")]["J"].to_numpy()
    b = df[(df["policy"] == "tuned_heuristic_privileged") & (df["scenario"] == "stable")]["J"].to_numpy()
    assert np.array_equal(a, b)

    primary = pd.read_csv(out / "primary_tests.csv")
    assert len(primary) == 9
    assert set(primary["comparator"]) == {"kademlia_epoch", "tuned_heuristic", "matched_random"}
    assert (primary["p_holm"] >= primary["p"] - 1e-12).all()
    per_run = pd.read_csv(out / "per_run.csv")
    assert len(per_run) == N_RL * 9 * 4
    summary = pd.read_csv(out / "summary.csv")
    assert len(summary) == (len(labels) + 1) * len(SCENARIOS)   # + rl_mean
    rl_mean = summary[(summary["policy"] == "rl_mean") & (summary["scenario"] == "stable")]
    runs = df[df["policy"].isin(["rl_run0", "rl_run1"]) & (df["scenario"] == "stable")]
    assert rl_mean["J_mean"].iloc[0] == pytest.approx(runs["J"].mean())
    hyp = pd.read_csv(out / "hypotheses.csv")
    assert list(hyp["id"]) == ["H1", "H2", "H3", "H4", "H5"]
    assert set(hyp["verdict"]) <= {"pass", "fail", "inconclusive"}

    em = json.loads((out / "eval_manifest.json").read_text(encoding="utf-8"))
    assert em["eval_lock"]["seed_list_hash"] == read_manifest(man)[LOCK_KEY]["seed_list_hash"]
    assert len(em["rl_q_tables"]) == N_RL and all(len(q["sha256"]) == 64 for q in em["rl_q_tables"])
    assert em["privileged_fallback"] == ["heldout_medium"]
    assert set(em["env_hashes"]) == set(SCENARIOS)
    runs_rec = read_manifest(man)[RUNS_KEY]
    assert len(runs_rec) == 1 and runs_rec[0]["out_dir"] == str(out)


def test_rerun_refused_then_identical(e2e, base):
    d, man, rep = e2e
    out2 = d / "out2"
    with pytest.raises(EvalRerunError):
        evaluate(CONFIGS, man, out2, **_kw(base))
    assert not out2.exists()
    rep2 = evaluate(CONFIGS, man, out2, allow_rerun=True, **_kw(base))
    assert (out2 / "per_seed.csv").read_bytes() == (d / "out" / "per_seed.csv").read_bytes()
    pd.testing.assert_frame_equal(rep2.primary, rep.primary)
    assert len(read_manifest(man)[RUNS_KEY]) == 2


def test_cli_evaluate_smoke(base, tmp_path, capsys):
    man = _manifest(base, tmp_path / "manifest.json")
    common = ["--configs-dir", str(CONFIGS), "--experiment-file", str(base["exp"]),
              "--manifest", str(man), "--seeds", "1"]
    out = tmp_path / "out"
    run = ["evaluate", *common, "--out", str(out), "--rl-q", *map(str, base["qs"]),
           "--tune-selected", str(base["sel"]), "--matched-freqs", str(base["freqs"])]
    assert main(run) == 1
    assert "not locked" in capsys.readouterr().err and not out.exists()
    assert main(["evaluate", "--create-lock", *common]) == 0
    assert "eval lock" in capsys.readouterr().out
    assert main(run) == 0, capsys.readouterr().err
    text = capsys.readouterr().out
    assert "H1:" in text and "tuned_heuristic_privileged (priv*)" in text
    assert len(pd.read_csv(out / "per_seed.csv")) == 12 * 3
    assert main(run) == 1
    assert "allow-rerun" in capsys.readouterr().err
