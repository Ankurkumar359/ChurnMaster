"""Headroom gate logic and env freeze (Requirements 22.1-22.7, revised J/p95 gates)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churnmaster.cli import main
from churnmaster.config import GatesConfig, TuningConfig, load_config
from churnmaster.experiments.headroom import (CHALLENGERS, GATE_POLICIES, evaluate_gates,
                                              tuning_candidates)
from churnmaster.experiments.runner import (FrozenEnvError, check_frozen, env_config_hash,
                                            freeze_env, read_manifest, write_manifest)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
N = 10
NOISE = np.random.default_rng(7).normal(0.0, 1.0, size=N)  # shared, deterministic jitter


def _frame(spec: dict[str, dict], scenario: str = "high_churn") -> pd.DataFrame:
    """Per-seed table; spec[policy] = {J, p95, pings} as scalars or length-N arrays."""
    rows = []
    for pol, v in spec.items():
        J = np.broadcast_to(np.asarray(v.get("J", 0.80), float), (N,))
        p95 = np.broadcast_to(np.asarray(v.get("p95", 1000.0), float), (N,))
        pings = np.broadcast_to(np.asarray(v.get("pings", 100.0), float), (N,))
        for ep in range(N):
            rows.append({"policy": pol, "scenario": scenario, "split": "calibration", "run": 0,
                         "episode": ep, "J": J[ep], "p95_latency_all_ms": p95[ep],
                         "pings_per_hour": pings[ep]})
    return pd.DataFrame(rows)


def _base_J() -> np.ndarray:
    return 0.80 + 0.01 * NOISE


def _spec(**over) -> dict[str, dict]:
    """All gate policies equal to never_replace unless overridden."""
    s = {p: {"J": _base_J(), "p95": 1000.0 + 20 * NOISE, "pings": 100.0} for p in GATE_POLICIES}
    for pol, v in over.items():
        s[pol] = {**s[pol], **v}
    return s


def _eval(df: pd.DataFrame):
    return evaluate_gates(df, GatesConfig(), n_resamples=2000)


# ----------------------------------------------------------------- G1 ----- #

def test_g1_passes_via_J_only():
    res = _eval(_frame(_spec(oracle_heuristic={"J": _base_J() + 0.03 + 0.001 * NOISE})))
    assert res.best == "oracle_heuristic"
    c = res.candidate("oracle_heuristic")
    assert c.j_route and not c.p95_route
    assert res.g1_passed and res.g1_route == "J"
    assert c.dJ.lo > 0 and c.dJ.mean == pytest.approx(0.03, abs=1e-3)


def test_g1_passes_via_p95_only():
    nr_p95 = 1000.0 + 20 * NOISE
    res = _eval(_frame(_spec(
        never_replace={"p95": nr_p95},
        kademlia_epoch={"J": _base_J() + 0.004, "p95": nr_p95 * 0.6})))
    c = res.candidate("kademlia_epoch")
    assert res.best == "kademlia_epoch"
    assert not c.j_route and c.p95_route
    assert c.p95_rel == pytest.approx(0.4)
    assert c.p95_diff_ms.lo > 0
    assert res.g1_passed and res.g1_route == "p95"


def test_g1_fails_when_both_routes_fall_short():
    nr_p95 = 1000.0 + 20 * NOISE
    res = _eval(_frame(_spec(
        never_replace={"p95": nr_p95},
        always_replace={"J": _base_J() + 0.005, "p95": nr_p95 * 0.9})))
    assert res.best == "always_replace"
    assert not res.g1_passed and res.g1_route is None


def test_g1_fails_when_ci_includes_zero():
    # Mean ΔJ = +0.02 (>= 0.01) but the paired differences flip sign heavily.
    diffs = np.array([0.30, -0.25, 0.28, -0.24, 0.26, -0.22, 0.25, -0.21, 0.20, -0.17])
    assert diffs.mean() == pytest.approx(0.02)
    res = _eval(_frame(_spec(oracle_heuristic={"J": _base_J() + diffs})))
    c = res.candidate("oracle_heuristic")
    assert c.dJ.mean >= 0.01 and c.dJ.lo <= 0
    assert not c.j_route and not res.g1_passed


def test_best_is_largest_mean_dJ_and_all_three_reported():
    nr_p95 = 1000.0 + 20 * NOISE
    res = _eval(_frame(_spec(
        never_replace={"p95": nr_p95},
        always_replace={"J": _base_J() + 0.03},
        oracle_heuristic={"J": _base_J() + 0.02, "p95": nr_p95 * 0.5},
        kademlia_epoch={"J": _base_J() + 0.01})))
    assert res.best == "always_replace"
    assert [c.policy for c in res.candidates] == list(CHALLENGERS)
    d = res.to_dict()["G1"]
    assert set(d["candidates"]) == set(CHALLENGERS) and d["best"] == "always_replace"


# ----------------------------------------------------------------- G2 ----- #

def test_g2_passes_via_J():
    res = _eval(_frame(_spec(oracle_heuristic={"J": _base_J() + 0.02})))
    assert res.g2_passed and res.g2["route"] == "J" and not res.limited_headroom


def test_g2_passes_via_pings_with_J_within_tolerance():
    res = _eval(_frame(_spec(kademlia_epoch={"pings": 100.0},
                             oracle_heuristic={"J": _base_J() - 0.003, "pings": 55.0})))
    assert res.g2["ping_ratio"] == pytest.approx(0.55)
    assert res.g2_passed and res.g2["route"] == "pings"


def test_g2_fails_sets_limited_headroom():
    res = _eval(_frame(_spec(kademlia_epoch={"pings": 100.0},
                             oracle_heuristic={"J": _base_J() - 0.003, "pings": 90.0})))
    assert not res.g2_passed and res.limited_headroom
    # Cheap but too much J loss also fails the ping route.
    res2 = _eval(_frame(_spec(oracle_heuristic={"J": _base_J() - 0.02, "pings": 10.0})))
    assert not res2.g2["ping_route"] and res2.limited_headroom


def test_g2_info_reported_for_other_scenarios():
    df = pd.concat([_frame(_spec(oracle_heuristic={"J": _base_J() + 0.02})),
                    _frame(_spec(), scenario="stable")])
    res = _eval(df)
    assert "stable" in res.g2_info and res.g2_passed
    assert not res.g2_info["stable"]["passed"]


# ------------------------------------------------------------- tuning ----- #

def test_tuning_candidates_order_closest_to_default_first():
    default = {"deadline_ms": 2000, "p_loss": 0.02, "replicas": 3}
    cands = tuning_candidates(TuningConfig(), default)
    assert len(cands) == 26 and default not in cands
    one_step = cands[:4]
    for c in one_step:
        assert sum(c[k] != default[k] for k in default) == 1
    assert cands[0] == {"deadline_ms": 1500, "p_loss": 0.02, "replicas": 3}
    assert sum(cands[-1][k] != default[k] for k in default) == 3


# ------------------------------------------------------------- freeze ----- #

def _cfg(scenario: str = "high_churn"):
    return load_config(CONFIGS / "base.toml", CONFIGS / "scenarios" / f"{scenario}.toml",
                       CONFIGS / "experiments" / "headroom.toml")


def test_manifest_round_trip(tmp_path):
    data = {"a": 1, "b": [1, 2], "c": {"x": "y"}, "none": None}
    p = write_manifest(tmp_path / "sub" / "manifest.json", data)
    assert read_manifest(p) == data


def test_frozen_hash_match_and_mismatch(tmp_path):
    m = tmp_path / "manifest.json"
    cfgs = {s: _cfg(s) for s in ("stable", "high_churn", "heldout_medium")}
    data = freeze_env(m, cfgs, extra={"n_seeds": 10})
    assert data["frozen"] is True and set(data["env_hashes"]) == set(cfgs)
    assert read_manifest(m)["env_hashes"]["high_churn"] == env_config_hash(cfgs["high_churn"])
    assert "packages" in data and "git_hash" in data

    hc = cfgs["high_churn"]
    assert check_frozen(m, hc) == env_config_hash(hc)
    with pytest.raises(FrozenEnvError):
        check_frozen(m, dataclasses.replace(hc, sim=dataclasses.replace(hc.sim, deadline_ms=1500)))
    with pytest.raises(FrozenEnvError):
        check_frozen(m, dataclasses.replace(
            hc, scenario=dataclasses.replace(hc.scenario, downtime_mean_s=1.0)))
    with pytest.raises(FrozenEnvError):
        check_frozen(m, dataclasses.replace(hc, root_seed=hc.root_seed + 1))


def test_reward_agent_experiment_changes_do_not_change_env_hash(tmp_path):
    m = tmp_path / "manifest.json"
    hc = _cfg()
    freeze_env(m, {"high_churn": hc})
    changed = dataclasses.replace(
        hc, reward=dataclasses.replace(hc.reward, w_M=0.9),
        agent=dataclasses.replace(hc.agent, gamma=0.5),
        experiment=dataclasses.replace(hc.experiment, name="eval", split="eval"))
    assert env_config_hash(changed) == env_config_hash(hc)
    check_frozen(m, changed)  # no raise
    # Train / eval experiment layers resolve to the same frozen env.
    for exp in ("train", "eval"):
        c = load_config(CONFIGS / "base.toml", CONFIGS / "scenarios" / "high_churn.toml",
                        CONFIGS / "experiments" / f"{exp}.toml")
        check_frozen(m, c)


def test_check_frozen_refuses_missing_unknown_or_unfrozen(tmp_path):
    with pytest.raises(FrozenEnvError):
        check_frozen(tmp_path / "nope.json", _cfg())
    m = tmp_path / "manifest.json"
    freeze_env(m, {"stable": _cfg("stable")})
    with pytest.raises(FrozenEnvError, match="not in the frozen manifest"):
        check_frozen(m, _cfg("high_churn"))
    freeze_env(m, {"high_churn": _cfg()}, frozen=False)
    with pytest.raises(FrozenEnvError, match="does not hold a frozen env"):
        check_frozen(m, _cfg())


# ---------------------------------------------------------------- CLI ----- #

SHORT_HEADROOM = """\
[sim]
warmup_s = 60
measured_s = 240

[experiment]
name = "headroom_smoke"
split = "calibration"

[experiment.stats]
bootstrap_resamples = 500

# Default-only grid: no fallback settings to try in the smoke test.
[experiment.tuning]
deadline_ms = [2000]
p_loss = [0.02]
replicas = [3]
"""


def test_cli_headroom_smoke_writes_gates_and_manifest(tmp_path, capsys):
    exp = tmp_path / "short.toml"
    exp.write_text(SHORT_HEADROOM, encoding="utf-8")
    out = tmp_path / "hr"
    rc = main(["headroom", "--seeds", "2", "--experiment-file", str(exp),
               "--configs-dir", str(CONFIGS), "--out", str(out)])
    text = capsys.readouterr().out
    assert rc in (0, 3), text
    assert "G1 (high_churn)" in text and "G2 (high_churn)" in text

    gates = json.loads((out / "gates.json").read_text(encoding="utf-8"))
    assert {"G1", "G2"} <= set(gates["gates"])
    assert set(gates["gates"]["G1"]["candidates"]) == set(CHALLENGERS)
    assert isinstance(gates["limited_headroom"], bool)
    assert rc == (0 if gates["g1_passed_final"] else 3)

    manifest = read_manifest(out / "manifest.json")
    scenario_files = {p.stem for p in (CONFIGS / "scenarios").glob("*.toml")}
    assert set(manifest["env_hashes"]) == scenario_files
    assert manifest["frozen"] is gates["g1_passed_final"]

    per_seed = pd.read_csv(out / "per_seed.csv")
    assert len(per_seed) == 6 * 2 * 2
    assert {"heuristic", "kademlia_std"} <= set(per_seed["policy"])
    for f in ("headroom_J_p95.png", "success_vs_pings.png", "dead_fraction_high_churn.png"):
        assert (out / f).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
