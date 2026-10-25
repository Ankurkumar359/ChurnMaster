"""Training protocol (task 8.3; Requirements 16.6, 24)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churnmaster.actions import Action
from churnmaster.cli import main
from churnmaster.experiments.runner import ConfigLoader, FrozenEnvError, freeze_env
from churnmaster.experiments.train import (TrainSettings, TrainSpec, greedy_stats,
                                           i4_check, i4_from_frames, run_trainings, run_training,
                                           select_grid, train_agent)
from churnmaster.rl.qlearning import QAgent, epsilon
from churnmaster.rl.state import Features, encode

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SHORT = """
[sim]
warmup_s = 60
measured_s = 240

[experiment]
name = "train_short"
split = "train"
scenarios = ["stable", "high_churn"]
validation_seeds = 2
last_checkpoints = 2

[experiment.debug]
scenario = "high_churn"
runs = 1
episodes = 4

[experiment.grid]
gamma = [0.8, 0.9]
alpha_min = [0.05]

[agent]
episodes = 4
runs = 2
validate_every = 2
"""


@pytest.fixture(scope="module")
def short_exp(tmp_path_factory):
    p = tmp_path_factory.mktemp("exp") / "train_short.toml"
    p.write_text(SHORT, encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def loader(short_exp):
    return ConfigLoader(CONFIGS, short_exp)


@pytest.fixture(scope="module")
def manifest(tmp_path_factory, loader):
    man = tmp_path_factory.mktemp("man") / "manifest.json"
    freeze_env(man, {s: loader(s) for s in ("stable", "high_churn")})
    return man


SETTINGS = TrainSettings(episodes=4, validate_every=2)


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #

def test_settings_checkpoints_alternation_and_tag():
    assert SETTINGS.checkpoints() == [0, 2, 4]
    assert TrainSettings(episodes=5, validate_every=2).checkpoints() == [0, 2, 4, 5]
    assert [SETTINGS.scenario_for(e) for e in range(4)] == ["stable", "high_churn"] * 2
    assert TrainSettings(scenarios=("high_churn",)).scenario_for(3) == "high_churn"
    assert TrainSettings(gamma=0.8, alpha_min=0.02).tag == "g0.8_a0.02"


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def test_short_training_deterministic_online_and_snapshots(loader):
    a = train_agent(TrainSpec(loader, SETTINGS, 0))
    b = train_agent(TrainSpec(loader, SETTINGS, 0))
    np.testing.assert_array_equal(a.final.Q, b.final.Q)
    np.testing.assert_array_equal(a.final.N, b.final.N)
    pd.testing.assert_frame_equal(a.curves, b.curves)
    # transitions consumed online: updates happened and Q moved away from 0
    assert (a.curves["n_updates"] > 0).all()
    assert (a.curves["n_updates"] == a.curves["n_transitions"]).all()
    assert np.abs(a.final.Q).sum() > 0
    assert list(a.curves["scenario"]) == ["stable", "high_churn"] * 2
    assert list(a.curves["epsilon"]) == [epsilon(e, SETTINGS) for e in range(4)]
    # snapshots at checkpoints; checkpoint 0 is untrained, later ones are frozen copies
    assert sorted(a.snapshots) == [0, 2, 4]
    assert not a.snapshots[0].Q.any()
    np.testing.assert_array_equal(a.snapshots[4].Q, a.final.Q)
    assert a.snapshots[2].N.sum() < a.snapshots[4].N.sum()


def test_runs_independent_and_parallel_equals_sequential(loader):
    specs = [TrainSpec(loader, SETTINGS, r) for r in (0, 1)]
    seq = run_trainings(specs, workers=1)
    par = run_trainings(specs, workers=2)
    for s, p in zip(seq, par):
        np.testing.assert_array_equal(s.final.Q, p.final.Q)
        pd.testing.assert_frame_equal(s.curves, p.curves)
    # run 1 alone == run 1 in a batch (own exploration stream), and differs from run 0
    alone = train_agent(TrainSpec(loader, SETTINGS, 1))
    np.testing.assert_array_equal(alone.final.Q, seq[1].final.Q)
    assert not np.array_equal(seq[0].final.Q, seq[1].final.Q)


# --------------------------------------------------------------------------- #
# Grid selection
# --------------------------------------------------------------------------- #

def _grid_table(scores):
    """scores: {(gamma, alpha): [J per checkpoint 0, 1, 2, 3]} -> table (2 scenarios x 2 seeds)."""
    rows = []
    for (g, a), js in scores.items():
        for c, j in enumerate(js):
            for scn in ("stable", "high_churn"):
                for ep in range(2):
                    rows.append({"gamma": g, "alpha_min": a, "checkpoint": c * 20,
                                 "scenario": scn, "episode": ep, "J": j})
    return pd.DataFrame(rows)


def test_select_grid_uses_last_k_checkpoints():
    t = _grid_table({(0.8, 0.02): [0.0, 0.9, 0.9, 0.9],   # best over last 3
                     (0.8, 0.05): [0.99, 0.5, 0.5, 0.95],  # best at final only
                     (0.9, 0.02): [0.0, 0.1, 0.1, 0.1],
                     (0.9, 0.05): [0.0, 0.8, 0.8, 0.8]})
    summary, sel = select_grid(t, 3, default=(0.9, 0.05))
    assert (sel["gamma"], sel["alpha_min"]) == (0.8, 0.02)
    assert sel["reason"] == "highest score"
    assert summary["selected"].sum() == 1
    assert summary.loc[1, "score"] == pytest.approx(0.65)
    assert summary.loc[0, "last_checkpoints"] == "20,40,60"
    # last_k = 1 -> the final checkpoint decides
    _, sel1 = select_grid(t, 1, default=(0.9, 0.05))
    assert (sel1["gamma"], sel1["alpha_min"]) == (0.8, 0.05)


def test_select_grid_tie_break_default_then_order():
    tie = {(0.8, 0.02): [0, 0.5, 0.5], (0.8, 0.05): [0, 0.5, 0.5], (0.9, 0.05): [0, 0.5, 0.5],
           (0.9, 0.02): [0, 0.1, 0.1]}
    _, sel = select_grid(_grid_table(tie), 2, default=(0.9, 0.05))
    assert (sel["gamma"], sel["alpha_min"], sel["reason"]) == (0.9, 0.05, "tie: default setting")
    assert sel["n_tied"] == 3
    # default not among the tied -> first in grid order
    _, sel2 = select_grid(_grid_table(tie), 2, default=(0.9, 0.02))
    assert (sel2["gamma"], sel2["alpha_min"], sel2["reason"]) == (0.8, 0.02, "tie: grid order")
    _, sel3 = select_grid(_grid_table(tie), 2, default=(0.9, 0.05),
                          order=[(0.9, 0.02), (0.8, 0.05), (0.8, 0.02), (0.9, 0.05)])
    assert (sel3["gamma"], sel3["alpha_min"]) == (0.9, 0.05)


# --------------------------------------------------------------------------- #
# I4
# --------------------------------------------------------------------------- #

def _i4_frames(final_shift, dq):
    rng = np.random.default_rng(0)
    keys = [(s, e) for s in ("stable", "high_churn") for e in range(5)]
    base = rng.normal(0.9, 0.01, len(keys))
    val = pd.DataFrame([{"checkpoint": c, "scenario": s, "episode": e,
                         "J": base[i] + (final_shift if c == 40 else 0.0)}
                        for c in (0, 20, 40) for i, (s, e) in enumerate(keys)])
    refs = pd.DataFrame([{"policy": p, "scenario": s, "episode": e, "J": base[i] + off}
                         for p, off in (("uniform_random", -0.01), ("never_replace", 0.0))
                         for i, (s, e) in enumerate(keys)])
    curves = pd.DataFrame({"episode": range(len(dq)), "mean_abs_dq": dq})
    return val, refs, curves


def test_i4_pass_and_fail_logic(tmp_path):
    val, refs, curves = _i4_frames(0.02, np.linspace(1.0, 0.1, 40))
    out = i4_from_frames(val, refs, curves, n_resamples=500)
    assert out["primary_reference"] == "uniform_random"
    assert out["comparisons"]["uniform_random"]["mean"] == pytest.approx(0.03)
    assert out["comparisons"]["checkpoint0"]["mean"] == pytest.approx(0.02)
    assert out["comparisons"]["never_replace"]["beats"]
    assert out["dq"]["window"] == 20 and out["dq"]["shrinks"]
    assert [t["from_episode"] for t in out["dq"]["trend"]] == [0, 20]
    assert out["passed"]
    # final worse than uniform_random -> fail even though |dQ| shrinks
    val2, refs2, curves2 = _i4_frames(-0.05, np.linspace(1.0, 0.1, 40))
    assert not i4_from_frames(val2, refs2, curves2, n_resamples=500)["passed"]
    # growing |dQ| -> fail
    val3, refs3, curves3 = _i4_frames(0.02, np.linspace(0.1, 1.0, 40))
    out3 = i4_from_frames(val3, refs3, curves3, n_resamples=500)
    assert out3["beats_primary"] and not out3["dq"]["shrinks"] and not out3["passed"]
    # i4_check reads a run dir and the mode-level references.csv
    run = tmp_path / "run0"
    run.mkdir()
    val.to_csv(run / "validation.csv", index=False)
    curves.to_csv(run / "curves.csv", index=False)
    refs.to_csv(tmp_path / "references.csv", index=False)
    got = i4_check(run, n_resamples=500)
    assert got["passed"] and (run / "i4.json").is_file()


def test_greedy_stats_inc_fail_diagnostics():
    ag = QAgent()
    s_fail = encode(Features(0, 2, 1, 1, 1, 0, 1))
    s_ok = encode(Features(0, 0, 0, 1, 1, 0, 0))
    ag.Q[s_fail, int(Action.REPLACE)] = 1.0
    ag.N[s_fail, int(Action.REPLACE)] = 3
    ag.N[s_ok, int(Action.KEEP)] = 1
    g = greedy_stats(ag)
    assert g["inc_fail_ge2"]["weighted"]["frac_not_keep"] == 1.0
    assert g["inc_fail_ge2"]["visited"]["n"] == 1
    assert 0 < g["inc_fail_ge2"]["all"]["frac_not_keep"] < 0.05
    assert g["healthy"]["weighted"]["frac_keep"] == 1.0
    assert g["states_visited"] == 2


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #

def test_run_training_refuses_unfrozen_env(tmp_path, short_exp):
    with pytest.raises(FrozenEnvError):
        run_training("debug", CONFIGS, experiment_path=short_exp,
                     manifest_path=tmp_path / "missing.json", out_dir=tmp_path / "o")
    man = tmp_path / "full.json"
    freeze_env(man, {"high_churn": ConfigLoader(CONFIGS)("high_churn")})   # full-length env
    with pytest.raises(FrozenEnvError, match="differs"):
        run_training("debug", CONFIGS, experiment_path=short_exp, manifest_path=man,
                     out_dir=tmp_path / "o")
    assert not (tmp_path / "o").exists()


def test_final_mode_end_to_end_files_and_checkpoint0(tmp_path, short_exp, manifest):
    rep = run_training("final", CONFIGS, experiment_path=short_exp, manifest_path=manifest,
                       out_dir=tmp_path / "final", gamma=0.9, alpha_min=0.05, n_resamples=200)
    assert [r.run for r in rep.runs] == [0, 1]
    refs = rep.references
    assert set(refs["policy"]) == {"never_replace", "uniform_random", "kademlia_epoch",
                                   "heuristic", "oracle_heuristic"}
    nr = refs[refs["policy"] == "never_replace"].set_index(["scenario", "episode"])["J"]
    for rr in rep.runs:
        d = rr.out_dir
        for f in ("q_ep0.npy", "q_ep2.npy", "q_ep4.npy", "q_final.npy", "curves.csv",
                  "validation.csv", "greedy_table.csv", "i4.json", "summary.json"):
            assert (d / f).is_file(), f
        val = pd.read_csv(d / "validation.csv")
        assert sorted(val["checkpoint"].unique()) == [0, 2, 4]
        assert len(val) == 3 * 2 * 2                      # checkpoints x scenarios x seeds
        # untrained greedy checkpoint == never_replace on every validation seed
        c0 = val[val["checkpoint"] == 0].set_index(["scenario", "episode"])["J"]
        pd.testing.assert_series_equal(c0.sort_index(), nr.reindex(c0.index).sort_index(),
                                       check_names=False)
        np.testing.assert_array_equal(QAgent.load(d / "q_final.npy").Q,
                                      QAgent.load(d / "q_ep4.npy").Q)
        assert len(pd.read_csv(d / "curves.csv")) == 4
        assert set(json.loads((d / "i4.json").read_text())["comparisons"]) == {
            "checkpoint0", "uniform_random", "never_replace"}
    man = json.loads((tmp_path / "final" / "train_manifest.json").read_text())
    assert man["frozen_checked"] and man["selection_source"] == "cli"


def test_grid_then_final_uses_selection(tmp_path, short_exp, manifest):
    grid = run_training("grid", CONFIGS, experiment_path=short_exp, manifest_path=manifest,
                        out_dir=tmp_path / "grid", episodes=2, validation_seeds=1,
                        reference_policies=("never_replace", "uniform_random"), n_resamples=100)
    assert [(r.settings.gamma, r.settings.alpha_min) for r in grid.runs] == [(0.8, 0.05), (0.9, 0.05)]
    assert (tmp_path / "grid" / "g0.8_a0.05" / "run0" / "validation.csv").is_file()
    sel = json.loads((tmp_path / "grid" / "selected.json").read_text())
    assert len(pd.read_csv(tmp_path / "grid" / "grid.csv")) == 2
    fin = run_training("final", CONFIGS, experiment_path=short_exp, manifest_path=manifest,
                       out_dir=tmp_path / "final", episodes=2, runs=1, validation_seeds=1,
                       reference_policies=("uniform_random",), n_resamples=100)
    assert (fin.runs[0].settings.gamma, fin.runs[0].settings.alpha_min) == \
        (sel["gamma"], sel["alpha_min"])
    assert fin.selection_source.endswith("selected.json")


def test_cli_train_debug_smoke(tmp_path, short_exp, manifest, capsys):
    out = tmp_path / "debug"
    rc = main(["train", "--mode", "debug", "--configs-dir", str(CONFIGS),
               "--experiment-file", str(short_exp), "--manifest", str(manifest),
               "--out", str(out), "--episodes", "2", "--validation-seeds", "1"])
    text = capsys.readouterr().out
    assert rc == 0, text
    assert "I4" in text and "inc_fail>=2" in text
    val = pd.read_csv(out / "run0" / "validation.csv")
    assert set(val["scenario"]) == {"high_churn"}
    assert main(["train", "--mode", "debug", "--configs-dir", str(CONFIGS),
                 "--experiment-file", str(short_exp), "--manifest", str(tmp_path / "nope.json"),
                 "--out", str(out)]) == 1
