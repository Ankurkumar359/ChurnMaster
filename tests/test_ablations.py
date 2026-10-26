"""Agent ablations (task 12.1, Requirement 28.1)."""

from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churnmaster.actions import Action
from churnmaster.experiments.ablations import (AGENT_VARIANTS, ANALYSIS_PLAN, AblationSeedError,
                                               check_test_seeds_disjoint, guard_ablation_out,
                                               run_agent_suite)
from churnmaster.experiments.runner import ConfigLoader, Job, freeze_env, run_job
from churnmaster.experiments.train import (TrainHooks, TrainSettings, TrainSpec, apply_buffered,
                                           buffered_rewards, shuffle_rng, train_agent)
from churnmaster.policies.ablated import EncodedLearned
from churnmaster.rl.ledger import Transition
from churnmaster.rl.qlearning import QAgent, epsilon
from churnmaster.rl.state import (CAND_FRESH, CAND_NONE, N_ACTIONS, N_STATES, allowed_actions,
                                  decode, discretize, encode, is_reachable, reachable_states)
from churnmaster.rl.variants import FEATURE_NAMES, FeatureDrop, no_probe_mask
from churnmaster.seeding import Split, agent_rng
from churnmaster.sim.observation import DecisionContext

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"
SHORT = """
[sim]
warmup_s = 60
measured_s = 240

[experiment]
name = "ablate_short"
split = "train"
scenarios = ["stable", "high_churn"]

[agent]
episodes = 2
runs = 1
validate_every = 2
"""
SCENARIOS = ("stable", "high_churn", "heldout_medium")
SETTINGS = TrainSettings(gamma=0.8, alpha_min=0.02, episodes=2, validate_every=2)


@pytest.fixture(scope="module")
def short_exp(tmp_path_factory):
    p = tmp_path_factory.mktemp("exp") / "ablate_short.toml"
    p.write_text(SHORT, encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def loader(short_exp):
    return ConfigLoader(CONFIGS, short_exp)


@pytest.fixture(scope="module")
def manifest(tmp_path_factory, loader):
    man = tmp_path_factory.mktemp("man") / "manifest.json"
    freeze_env(man, {s: loader(s) for s in SCENARIOS})
    return man


# --------------------------------------------------------------------------- #
# Wrappers and masks
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("feature", FEATURE_NAMES)
def test_feature_drop_collapses_only_that_feature_and_keeps_masking(feature):
    drop = FeatureDrop(feature)
    images = set()
    for s in reachable_states():
        f, s2 = decode(s), drop.apply_state(s)
        g = decode(s2)
        images.add(s2)
        assert is_reachable(s2)
        # masking unchanged: candidate presence is preserved
        assert allowed_actions(s2) == allowed_actions(s)
        for name in FEATURE_NAMES:
            if name == feature:
                continue
            assert getattr(g, name) == getattr(f, name)
        if feature == "cand":
            assert g.cand == (CAND_NONE if f.cand == CAND_NONE else CAND_FRESH)
        else:
            assert getattr(g, feature) == 0
    vals = {getattr(decode(s), feature) for s in images}
    assert vals == ({CAND_NONE, CAND_FRESH} if feature == "cand" else {0})
    assert drop.apply_state(drop.apply_state(0)) == drop.apply_state(0)   # idempotent
    assert pickle.loads(pickle.dumps(drop)) == drop


def test_feature_drop_on_context_and_rejects_unknown():
    ctx = DecisionContext(bucket=7, epoch_idx=3, t_ms=90_000, inc_consec_fail=2,
                          inc_since_response_s=100.0, inc_age_s=50.0, inc_srtt_ms=100.0,
                          cand_since_seen_s=40.0, cand_last_rtt_ms=50.0, timeout_fraction=0.3)
    f = discretize(ctx, 8)
    assert FeatureDrop("inc_fail")(ctx, 8) == encode(FeatureDrop("inc_fail").apply(f))
    assert decode(FeatureDrop("cand")(ctx, 8)).cand == CAND_FRESH and f.cand != CAND_FRESH
    with pytest.raises(ValueError):
        FeatureDrop("nope")


def test_no_probe_mask():
    m = no_probe_mask()
    assert m.shape == (N_STATES, N_ACTIONS)
    assert not m[:, int(Action.PROBE)].any()
    assert m[:, int(Action.KEEP)].all()
    for s in range(N_STATES):
        assert m[s, int(Action.REPLACE)] == (Action.REPLACE in allowed_actions(s))


def test_variant_registry():
    names = list(AGENT_VARIANTS)
    assert names[:4] == ["gamma0", "no_probe", "shuffled_reward", "buffered_control"]
    assert {"drop_tier", "drop_inc_fail", "drop_inc_stale", "drop_inc_age",
            "drop_cand_freshness", "drop_cand_faster", "drop_churn"} <= set(names)
    assert len(names) == 11 and "drop_cand" not in names
    assert AGENT_VARIANTS["gamma0"].settings(SETTINGS).gamma == 0.0
    assert AGENT_VARIANTS["no_probe"].settings(SETTINGS) == SETTINGS
    assert AGENT_VARIANTS["drop_cand_freshness"].encoder() == FeatureDrop("cand")
    assert "NO multiple-comparison correction" in ANALYSIS_PLAN and "Split TEST" in ANALYSIS_PLAN


# --------------------------------------------------------------------------- #
# Buffered / shuffled updates
# --------------------------------------------------------------------------- #

def _transitions(n=30):
    rng = np.random.default_rng(5)
    return [Transition(0, k, int(rng.integers(N_STATES)) // 2 * 2, 0, -float(rng.random()),
                       0, (0, 1)) for k in range(n)]


def test_shuffled_permutes_rewards_and_buffered_keeps_them():
    trs = _transitions()
    orig = [t.r for t in trs]
    assert buffered_rewards(trs, None) == orig
    sh = buffered_rewards(trs, shuffle_rng(1, 0))
    assert sorted(sh) == sorted(orig) and sh != orig                # multiset preserved, permuted
    assert buffered_rewards(trs, shuffle_rng(1, 0)) == sh             # seeded
    # the shuffle stream is distinct from the run's exploration stream
    assert shuffle_rng(1, 0).random() != agent_rng(1, 0).random()
    a, b = QAgent(0.8, 0.02), QAgent(0.8, 0.02)
    used = apply_buffered(a, trs, None, use_mask=False)
    assert used == orig and a.n_updates == len(trs)
    for t in trs:
        b.update_transition(t)
    np.testing.assert_array_equal(a.Q, b.Q)                           # buffered == online order


class _LegacyPolicy:
    """The pre-hook TrainingPolicy (no encode_state attribute)."""

    name = "training"
    shares_decision_points = True

    def __init__(self, agent, eps, rng, id_bits):
        self.agent, self.eps, self.rng, self.id_bits = agent, eps, rng, id_bits

    def decide(self, ctx):
        return self.agent.act(encode(discretize(ctx, self.id_bits)), self.eps, self.rng)


def _legacy_train(loader, st, run):
    """Verbatim copy of the training loop before the ablation hooks were added."""
    cfgs = {s: loader(s) for s in dict.fromkeys(st.scenarios)}
    cfg0 = next(iter(cfgs.values()))
    agent = QAgent(st.gamma, st.alpha_min, st.alpha_exp)
    rng = agent_rng(cfg0.root_seed, run)
    for e in range(st.episodes):
        scn = st.scenario_for(e)
        agent.reset_stats()
        pol = _LegacyPolicy(agent, epsilon(e, st), rng, cfg0.sim.id_bits)
        run_job(Job(pol, scn, Split.TRAIN, run, e), cfg=cfgs[scn],
                on_transition=agent.update_transition)
    return agent


def test_default_hooks_reproduce_unmodified_training(loader):
    ref = _legacy_train(loader, SETTINGS, 1)
    new = train_agent(TrainSpec(loader, SETTINGS, 1)).final
    ident = train_agent(TrainSpec(loader, SETTINGS, 1, hooks=TrainHooks())).final
    for o in (new, ident):
        np.testing.assert_array_equal(o.Q, ref.Q)
        np.testing.assert_array_equal(o.N, ref.N)


@pytest.fixture(scope="module")
def trained(loader):
    out = {}
    for name in ("no_probe", "shuffled_reward", "buffered_control", "drop_inc_fail",
                 "drop_cand_freshness"):
        v = AGENT_VARIANTS[name]
        out[name] = train_agent(TrainSpec(loader, v.settings(SETTINGS), 0, hooks=v.hooks()))
    out["canonical"] = train_agent(TrainSpec(loader, SETTINGS, 0))
    return out


def test_no_probe_never_probes(trained, loader):
    o = trained["no_probe"]
    assert o.final.N[:, int(Action.PROBE)].sum() == 0
    assert not o.final.mask[:, int(Action.PROBE)].any()
    pol = EncodedLearned(o.final, name="np", id_bits=8)
    res = run_job(Job(pol, "high_churn", Split.TEST, 0, 0), cfg=loader("high_churn"),
                  keep_records=True)
    acts = Counter(int(d.action) for d in res.decisions if d.action is not None)
    assert acts and acts[int(Action.PROBE)] == 0
    assert res.metrics.pings_per_hour >= 0


def test_drop_variants_only_visit_collapsed_states(trained, loader):
    for name, feat in (("drop_inc_fail", "inc_fail"), ("drop_cand_freshness", "cand")):
        o = trained[name]
        visited = np.flatnonzero(o.final.N.sum(axis=1) > 0)
        assert visited.size > 0
        drop = FeatureDrop(feat)
        assert all(drop.apply_state(int(s)) == int(s) for s in visited)
        assert o.final.Q.shape == (N_STATES, N_ACTIONS)
        # evaluation records the wrapped state as well (picklable policy)
        pol = pickle.loads(pickle.dumps(EncodedLearned(o.final, name=name, id_bits=8,
                                                       encode=drop)))
        res = run_job(Job(pol, "stable", Split.TEST, 0, 0), cfg=loader("stable"),
                      keep_records=True)
        states = [int(d.state) for d in res.decisions if d.state is not None]
        assert states and all(drop.apply_state(s) == s for s in states)
    # the canonical agent does visit states the drop_inc_fail wrapper would collapse
    can = np.flatnonzero(trained["canonical"].final.N.sum(axis=1) > 0)
    assert any(FeatureDrop("inc_fail").apply_state(int(s)) != int(s) for s in can)


def test_buffered_and_shuffled_runs(trained):
    b, s, c = trained["buffered_control"], trained["shuffled_reward"], trained["canonical"]
    for o in (b, s):
        assert (o.curves["n_updates"] == o.curves["n_transitions"]).all()
        assert (o.curves["n_updates"] > 0).all()
    # episode 0 acts on the same (all-zero) Q and the same exploration draws in all three,
    # so the same transitions are collected; buffering alone gives the canonical update set
    assert b.curves["n_transitions"].iloc[0] == c.curves["n_transitions"].iloc[0]
    assert not np.array_equal(s.final.Q, b.final.Q)


# --------------------------------------------------------------------------- #
# Seeds and guard
# --------------------------------------------------------------------------- #

def test_test_seeds_disjoint_and_overlap_detected(monkeypatch):
    info = check_test_seeds_disjoint(12345, range(20))
    assert info["overlap"] is False and info["split"] == "test"
    names = {c["name"] for c in info["checked_against"]}
    assert {"calibration", "train", "tuning", "validation", "eval"} <= names
    with pytest.raises(AblationSeedError):
        check_test_seeds_disjoint(12345, [0, 0])
    import churnmaster.experiments.ablations as ab
    monkeypatch.setattr(ab, "USED_SPLITS", (("fake", Split.TEST, (0,), range(3)),))
    with pytest.raises(AblationSeedError):
        ab.check_test_seeds_disjoint(12345, [2])


@pytest.mark.parametrize("bad", ["results", "results/headroom", "results/eval/x", "results/train",
                                 "results/figures", "results/reproduce_full/a",
                                 "results/diagnostics_60", "results/checkpoint_5_17", "."])
def test_guard_refuses_canonical_dirs(bad, monkeypatch):
    monkeypatch.chdir(REPO)
    with pytest.raises(ValueError):
        guard_ablation_out(bad)


def test_guard_allows_ablations_dir(monkeypatch):
    monkeypatch.chdir(REPO)
    assert guard_ablation_out("results/ablations") == (REPO / "results" / "ablations").resolve()


def test_ablations_is_canonical_for_reproduce(monkeypatch):
    """After the agent suite ran, results/ablations is protected from reproduce; the
    ablation guard protects every other canonical dir."""
    from churnmaster.experiments.ablations import PROTECTED_NAMES
    from churnmaster.experiments.reproduce import CANONICAL_NAMES, guard_out_dir

    monkeypatch.chdir(REPO)
    assert "ablations" in CANONICAL_NAMES
    assert set(CANONICAL_NAMES) - {"ablations"} <= set(PROTECTED_NAMES)
    with pytest.raises(ValueError):
        guard_out_dir("results/ablations")
    with pytest.raises(ValueError):
        guard_out_dir("results/ablations/figures")


# --------------------------------------------------------------------------- #
# End-to-end (short)
# --------------------------------------------------------------------------- #

def test_suite_end_to_end(tmp_path, loader, short_exp, manifest, trained):
    can = trained["canonical"].final.save(tmp_path / "canon" / "run0" / "q_final.npy")
    sel = tmp_path / "selected.json"
    sel.write_text(json.dumps({"global": "f=2,po=0,rs=1,ps=0,tg=all", "per_scenario": {}}),
                   encoding="utf-8")
    man_before = manifest.read_bytes()
    out = tmp_path / "abl"
    rep = run_agent_suite(CONFIGS, manifest, out, experiment_path=short_exp, runs=(0,),
                          episodes=2, n_seeds=2, canonical_q=[can], tune_selected=sel,
                          workers=1, n_resamples=200)
    assert manifest.read_bytes() == man_before                      # frozen manifest untouched
    d = out / "agent"
    for f in ("per_seed.csv", "comparison.csv", "per_run.csv", "summary.csv", "greedy_stats.json"):
        assert (d / f).is_file()
    for v in AGENT_VARIANTS:
        assert (d / "variants" / v / "run0" / "q_final.npy").is_file()
        assert (d / "variants" / v / "run0" / "curves.csv").is_file()
    assert (out / "figures" / "agent_ablation_J.png").is_file()
    assert (out / "figures" / "agent_ablation_metrics.png").is_file()
    ps = pd.read_csv(d / "per_seed.csv")
    assert set(ps["split"]) == {"test"} and set(ps["run"]) == {0}
    assert set(ps["scenario"]) == set(SCENARIOS) and set(ps["episode"]) == {0, 1}
    # no_probe never probes on the TEST seeds
    np_rows = ps[ps["group"] == "no_probe"]
    assert (np_rows["pings_per_hour"] >= 0).all()
    comp = pd.read_csv(d / "comparison.csv")
    vc = comp[comp["comparison"] == "vs_canonical"]
    assert len(vc) == 11 * 3 * 4
    assert set(comp["comparison"]) == {"vs_canonical", "shuffled_vs_buffered",
                                       "vs_tuned_heuristic"}
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    sm = m["suites"]["agent"]
    assert "NO multiple-comparison correction" in m["analysis_plan"]
    assert sm["eval_seeds"]["overlap"] is False and sm["frozen_manifest"]["access"].startswith("read")
    # drop_cand_freshness: per-seed REPLACE remains possible (presence kept) -> masking correct
    g = json.loads((d / "greedy_stats.json").read_text(encoding="utf-8"))
    assert g["no_probe"]["run0"]["greedy"]["actions"]["all"]["PROBE"] == 0.0
    # refuses to overwrite without the flag
    with pytest.raises(ValueError, match="overwrite"):
        run_agent_suite(CONFIGS, manifest, out, experiment_path=short_exp, runs=(0,),
                        episodes=2, n_seeds=2, canonical_q=[can], tune_selected=sel, workers=1)


def test_suite_refuses_canonical_out(tmp_path, short_exp, manifest, monkeypatch):
    monkeypatch.chdir(REPO)
    with pytest.raises(ValueError):
        run_agent_suite(CONFIGS, manifest, "results/eval", experiment_path=short_exp, episodes=2)


@pytest.mark.slow
def test_default_path_reproduces_canonical_run0():
    """Full 200-episode retrain of canonical run 0 equals results/train/final/run0."""
    from churnmaster.experiments.diagnostics import _frozen_loader

    q = REPO / "results" / "train" / "final" / "run0" / "q_final.npy"
    man = REPO / "results" / "headroom" / "manifest.json"
    if not q.is_file() or not man.is_file():
        pytest.skip("canonical results not present")
    ld = _frozen_loader(CONFIGS, CONFIGS / "experiments" / "train.toml", man)
    st = TrainSettings.from_config(ld("stable"), ("stable", "high_churn"), gamma=0.8,
                                   alpha_min=0.02)
    o = train_agent(TrainSpec(ld, st, 0))
    np.testing.assert_array_equal(o.final.Q, np.load(q))
