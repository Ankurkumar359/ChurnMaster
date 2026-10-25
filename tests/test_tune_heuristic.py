"""Tests for experiments/tune_heuristic.py (Task 9.1, Requirement 25.1)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from churnmaster.cli import main
from churnmaster.experiments.runner import ConfigLoader, FrozenEnvError, freeze_env
from churnmaster.experiments.tune_heuristic import (GLOBAL_LABEL, PRIVILEGED_LABEL, TUNING_RUN,
                                                    config_priority, load_selected, run_tuning,
                                                    select, tuned_policy_names, write_outputs)
from churnmaster.policies.heuristic import DEFAULT_CONFIG, all_configs

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SHORT = '[sim]\nwarmup_s = 60\nmeasured_s = 240\n\n[experiment]\nname = "tune_short"\n' \
        'split = "train"\nn_seeds = 2\nscenarios = ["stable", "high_churn"]\n'

IDS = [c.id for c in all_configs()]
A, B, C = [i for i in IDS if i != DEFAULT_CONFIG.id][:3]


def _rows(cid: str, scenario: str, js: list[float]) -> list[dict]:
    return [{"policy": f"heuristic:{cid}", "scenario": scenario, "run": TUNING_RUN,
             "episode": e, "J": j, "success_rate": 0.9, "p95_latency_all_ms": 500.0,
             "pings_per_hour": 100.0} for e, j in enumerate(js)]


def _table(spec: dict[str, dict[str, list[float]]]) -> pd.DataFrame:
    rows = []
    for cid, per in spec.items():
        for s, js in per.items():
            rows += _rows(cid, s, js)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Selection on synthetic tables
# --------------------------------------------------------------------------- #

def test_global_and_per_scenario_best_differ():
    df = _table({A: {"stable": [0.90, 0.92], "high_churn": [0.50, 0.52]},   # overall 0.71
                 B: {"stable": [0.80, 0.80], "high_churn": [0.60, 0.60]},   # overall 0.70
                 C: {"stable": [0.70, 0.70], "high_churn": [0.65, 0.65]}})  # overall 0.675
    summary, sel = select(df)
    assert sel.global_id == A
    assert sel.per_scenario == {"stable": A, "high_churn": C}
    assert list(summary["config"]) == [A, B, C]
    assert list(summary["rank"]) == [1, 2, 3]
    assert summary.loc[0, "J_overall"] == pytest.approx(0.71)
    assert summary.loc[0, "J_mean_stable"] == pytest.approx(0.91)
    assert bool(summary.loc[2, "privileged_best_high_churn"])
    d = sel.to_dict()
    assert d["global_policy"] == f"heuristic:{A}"
    assert {p["scenario"]: p["config"] for p in d["privileged"]} == {"stable": A, "high_churn": C}
    assert all(p["privileged"] and p["label"] == PRIVILEGED_LABEL for p in d["privileged"])


def test_equal_scenario_weighting_with_unequal_row_counts():
    # Row-pooled mean would favour A (many stable rows); per-scenario equal weight favours B.
    df = _table({A: {"stable": [1.0] * 9, "high_churn": [0.0]},       # pooled 0.9, equal 0.5
                 B: {"stable": [0.7], "high_churn": [0.7]}})          # pooled 0.7, equal 0.7
    summary, sel = select(df)
    assert sel.global_id == B
    assert summary.set_index("config").loc[A, "J_overall"] == pytest.approx(0.5)
    assert summary.set_index("config").loc[A, "n_stable"] == 9


def test_tie_break_default_first_then_config_order():
    same = {"stable": [0.8, 0.8], "high_churn": [0.6, 0.6]}
    # B listed before A in the table, DEFAULT last: DEFAULT still wins.
    _, sel = select(_table({B: same, A: same, DEFAULT_CONFIG.id: same}))
    assert sel.global_id == DEFAULT_CONFIG.id
    assert set(sel.per_scenario.values()) == {DEFAULT_CONFIG.id}
    # Without DEFAULT: all_configs() order, regardless of table order.
    summary, sel = select(_table({B: same, A: same}))
    first = A if IDS.index(A) < IDS.index(B) else B
    assert sel.global_id == first
    assert summary.loc[0, "config"] == first
    # Differences below TIE_TOL count as ties.
    tiny = {"stable": [0.8 + 1e-14, 0.8], "high_churn": [0.6, 0.6]}
    _, sel = select(_table({B: tiny, A: same}))
    assert sel.global_id == first


def test_priority_order():
    p = config_priority()
    assert p[0] == DEFAULT_CONFIG.id and len(p) == 32 and set(p) == set(IDS)


def test_missing_scenario_rows_rejected():
    df = _table({A: {"stable": [0.5], "high_churn": [0.5]}, B: {"stable": [0.5]}})
    with pytest.raises(ValueError, match="no rows"):
        select(df)


# --------------------------------------------------------------------------- #
# Frozen env, end-to-end, load_selected
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def short_exp(tmp_path_factory):
    p = tmp_path_factory.mktemp("exp") / "tune_short.toml"
    p.write_text(SHORT, encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def manifest(tmp_path_factory, short_exp):
    loader = ConfigLoader(CONFIGS, short_exp)
    man = tmp_path_factory.mktemp("man") / "manifest.json"
    freeze_env(man, {s: loader(s) for s in ("stable", "high_churn")})
    return man


def test_frozen_env_mismatch_refused(tmp_path, short_exp):
    with pytest.raises(FrozenEnvError):
        run_tuning(CONFIGS, tmp_path / "missing.json", experiment_path=short_exp)
    man = tmp_path / "full.json"
    freeze_env(man, {s: ConfigLoader(CONFIGS)(s) for s in ("stable", "high_churn")})
    with pytest.raises(FrozenEnvError, match="differs"):
        run_tuning(CONFIGS, man, experiment_path=short_exp)


def test_end_to_end_short_and_load_selected(tmp_path, short_exp, manifest):
    configs = [DEFAULT_CONFIG] + [c for c in all_configs() if c.id in (A, B)]
    rep = run_tuning(CONFIGS, manifest, experiment_path=short_exp, configs=configs, workers=1)
    assert rep.n_jobs == 3 * 2 * 2
    assert len(rep.per_seed) == 12
    assert set(rep.per_seed["split"]) == {"train"}
    assert set(rep.per_seed["run"]) == {TUNING_RUN}
    assert set(rep.per_seed["episode"]) == {0, 1}
    assert rep.seeds["episodes"] == [0, 1]
    assert set(rep.env_hashes) == {"stable", "high_churn"}

    files = write_outputs(rep, tmp_path / "out")
    assert [f.name for f in files] == ["per_seed.csv", "summary.csv", "selected.json"]
    assert all(f.is_file() for f in files)
    assert len(pd.read_csv(files[1])) == 3

    sel = load_selected(files[2])
    assert sel["global"] == rep.selection.global_id
    assert sel["per_scenario"] == rep.selection.per_scenario
    assert sel["env_hashes"] == rep.env_hashes
    names = tuned_policy_names(sel)
    assert names[GLOBAL_LABEL] == f"heuristic:{rep.selection.global_id}"
    assert set(names[PRIVILEGED_LABEL]) == {"stable", "high_churn"}


def test_load_selected_rejects_bad_files(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"foo": 1}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_selected(p)
    p.write_text(json.dumps({"global": "nope", "per_scenario": {}}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_selected(p)


def test_cli_tune_refuses_unfrozen(tmp_path, short_exp, capsys):
    rc = main(["tune", "--experiment-file", str(short_exp), "--manifest",
               str(tmp_path / "none.json"), "--out", str(tmp_path / "o")])
    assert rc == 1
    assert "headroom" in capsys.readouterr().err
