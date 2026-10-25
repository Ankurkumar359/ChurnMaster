"""Unit tests for config.py (Requirements 1.1-1.5).

Property tests for Properties 1 and 2 live in separate tasks (1.7, 1.8).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from churnmaster.config import (
    AgentConfig, Config, ConfigError, RewardConfig, SimConfig, config_hash, from_dict,
    load_config, merge, read_toml,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
BASE = CONFIGS / "base.toml"
SCENARIOS = ["stable", "high_churn", "heldout_medium", "bursty"]
EXPERIMENTS = ["headroom", "train", "tune", "eval"]


def scn(name: str) -> Path:
    return CONFIGS / "scenarios" / f"{name}.toml"


def exp(name: str) -> Path:
    return CONFIGS / "experiments" / f"{name}.toml"


def base_dict() -> dict:
    return merge(read_toml(BASE), read_toml(scn("high_churn")))


# ---------------------------------------------------------------- loading ---

@pytest.mark.parametrize("s", SCENARIOS)
@pytest.mark.parametrize("e", EXPERIMENTS + [None])
def test_all_shipped_configs_load(s, e):
    cfg = load_config(BASE, scn(s), exp(e) if e else None)
    assert cfg.scenario.name == s
    assert (cfg.scenario.bursts is not None) == (s == "bursty")
    if e:
        assert cfg.experiment.name == e
    assert len(config_hash(cfg)) == 64


def test_scenario_values():
    expected = {
        "stable": (0.70, 4 * 3600, 30 * 60, 20 * 60),
        "high_churn": (0.30, 3600, 5 * 60, 10 * 60),
        "heldout_medium": (0.45, 2 * 3600, 10 * 60, 12 * 60),
        "bursty": (0.50, 2 * 3600, 15 * 60, 15 * 60),
    }
    for name, vals in expected.items():
        c = load_config(BASE, scn(name)).scenario
        assert (c.stable_share, c.stable_session_mean_s, c.flaky_session_mean_s,
                c.downtime_mean_s) == vals


def test_bursty_extras():
    b = load_config(BASE, scn("bursty")).scenario.bursts
    assert (b.gap_mean_s, b.fraction, b.leave_window_s, b.return_mean_s,
            b.rtt_mult, b.rtt_duration_s) == (900.0, 0.2, 10, 300.0, 2.0, 180)


def test_defaults_match_requirement_1_5():
    s, r, a = SimConfig(), RewardConfig(), AgentConfig()
    assert (s.n_nodes, s.id_bits, s.k, s.alpha, s.replicas) == (128, 8, 4, 2, 3)
    assert (s.cache_size, s.cache_ttl_s, s.rebuild_period_s, s.epoch_s) == (4, 300, 300, 30)
    assert (s.warmup_s, s.measured_s, s.lookup_rate_hz, s.p_loss) == (600, 2400, 1.0, 0.02)
    assert (s.timeout_ms, s.deadline_ms, s.max_rounds) == (500, 2000, 8)
    assert (r.w_L, r.w_M, r.w_R) == (0.3, 0.2, 0.3)
    assert a.gamma == 0.9
    # base.toml agrees with dataclass defaults
    cfg = load_config(BASE, scn("stable"))
    assert cfg.sim == SimConfig() and cfg.reward == RewardConfig() and cfg.agent == AgentConfig()


def test_missing_sections_default():
    d = {"root_seed": 1, "scenario": read_toml(scn("stable"))["scenario"]}
    cfg = from_dict(d)
    assert cfg.sim == SimConfig()
    assert cfg.experiment.name == "default"


def test_frozen():
    cfg = load_config(BASE, scn("stable"), exp("train"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.sim.k = 5  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.root_seed = 3  # type: ignore[misc]
    assert isinstance(cfg.experiment.scenarios, tuple)
    hash(cfg)  # fully hashable => all nested values immutable


def test_valid_config_roundtrip_equal():
    d = base_dict()
    assert from_dict(d) == from_dict(merge(d)) and isinstance(from_dict(d), Config)


# ------------------------------------------------------------------ merge ---

def test_merge_order_experiment_overrides():
    cfg = load_config(BASE, scn("stable"), exp("train"))
    assert cfg.experiment.grid.gamma == (0.8, 0.9)
    assert cfg.agent.episodes == 200 and cfg.agent.gamma == 0.9


def test_merge_later_wins_and_no_mutation():
    a = {"x": 1, "t": {"a": 1, "b": 2}}
    b = {"t": {"b": 3, "c": 4}}
    c = {"x": 5, "t": {"c": 6}}
    m = merge(a, b, c)
    assert m == {"x": 5, "t": {"a": 1, "b": 3, "c": 6}}
    assert a == {"x": 1, "t": {"a": 1, "b": 2}} and b == {"t": {"b": 3, "c": 4}}


def test_load_config_overlay_precedence(tmp_path):
    ov = tmp_path / "exp.toml"
    ov.write_text('[sim]\np_loss = 0.05\n[scenario]\nstable_share = 0.9\n[experiment]\nname="x"\n')
    cfg = load_config(BASE, scn("stable"), ov)
    assert cfg.sim.p_loss == 0.05 and cfg.scenario.stable_share == 0.9
    assert cfg.sim.k == 4 and cfg.scenario.name == "stable"


# ------------------------------------------------------------------- hash ---

def test_hash_stable_under_key_reordering():
    d = base_dict()
    reordered = {k: d[k] for k in reversed(list(d))}
    reordered["sim"] = {k: d["sim"][k] for k in reversed(list(d["sim"]))}
    assert config_hash(from_dict(reordered)) == config_hash(from_dict(d))


def test_hash_changes_with_value_and_int_float_equivalence():
    d = base_dict()
    h0 = config_hash(from_dict(d))
    assert config_hash(from_dict(merge(d, {"sim": {"p_loss": 0.05}}))) != h0
    # 1 and 1.0 for a float field normalize to the same canonical value
    assert config_hash(from_dict(merge(d, {"sim": {"lookup_rate_hz": 1}}))) == h0


# ---------------------------------------------------------------- invalid ---

@pytest.mark.parametrize("patch, key", [
    # unknown keys
    ({"sim": {"bogus": 1}}, "sim.bogus"),
    ({"nonsense": {}}, "nonsense"),
    ({"scenario": {"extra": 1}}, "scenario.extra"),
    ({"scenario": {"bursts": {"oops": 1}}}, "scenario.bursts.oops"),
    ({"experiment": {"stats": {"foo": 1}}}, "experiment.stats.foo"),
    # wrong types
    ({"sim": {"k": "4"}}, "sim.k"),
    ({"sim": {"k": 4.0}}, "sim.k"),
    ({"sim": {"k": True}}, "sim.k"),
    ({"reward": {"w_L": "x"}}, "reward.w_L"),
    ({"sim": {"n_nodes": {}}}, "sim.n_nodes"),
    ({"experiment": {"scenarios": "stable"}}, "experiment.scenarios"),
    ({"experiment": {"split": "holdout"}}, "experiment.split"),
    # out of range
    ({"sim": {"p_loss": 1.0}}, "sim.p_loss"),
    ({"sim": {"p_loss": -0.1}}, "sim.p_loss"),
    ({"sim": {"k": 0}}, "sim.k"),
    ({"scenario": {"stable_share": 1.5}}, "scenario.stable_share"),
    ({"scenario": {"downtime_mean_s": 0.0}}, "scenario.downtime_mean_s"),
    ({"agent": {"gamma": 1.1}}, "agent.gamma"),
    ({"reward": {"w_M": -1.0}}, "reward.w_M"),
    ({"root_seed": -1}, "root_seed"),
    ({"sim": {"n_nodes": 300}}, "sim.n_nodes"),
    ({"experiment": {"tuning": {"p_loss": [0.01, 1.5]}}}, "experiment.tuning.p_loss[1]"),
])
def test_invalid_rejected_naming_key(patch, key):
    with pytest.raises(ConfigError) as ei:
        from_dict(merge(base_dict(), patch))
    assert ei.value.key == key
    assert key in str(ei.value)


def test_missing_required_scenario_key():
    d = base_dict()
    del d["scenario"]["stable_share"]
    with pytest.raises(ConfigError) as ei:
        from_dict(d)
    assert ei.value.key == "scenario.stable_share"


def test_load_config_error_names_file(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("[sim]\nunknown_knob = 3\n")
    with pytest.raises(ConfigError) as ei:
        load_config(BASE, scn("stable"), bad)
    assert ei.value.key == "sim.unknown_knob"
    assert ei.value.file == str(bad)
    assert "bad.toml" in str(ei.value)
