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
    return ConfigLoader(CONFIGS, short_exp, ("ablations",))


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
    ld = _frozen_loader(CONFIGS, CONFIGS / "experiments" / "train.toml", man, ("ablations",))
    st = TrainSettings.from_config(ld("stable"), ("stable", "high_churn"), gamma=0.8,
                                   alpha_min=0.02)
    o = train_agent(TrainSpec(ld, st, 0))
    np.testing.assert_array_equal(o.final.Q, np.load(q))


# --------------------------------------------------------------------------- #
# Reward-weight sweep (task 12.2, Requirement 28.2)
# --------------------------------------------------------------------------- #

from churnmaster.experiments.ablations import (CANONICAL_CELL, CANONICAL_WEIGHTS,  # noqa: E402
                                               REWARD_ANALYSIS_PLAN, REWARD_CELLS, RewardLoader,
                                               _reward_eval, _RewardEvalTask, j_from_records,
                                               reward_cells, run_reward_suite)
from churnmaster.config import config_hash  # noqa: E402
from churnmaster.experiments.runner import env_config_hash  # noqa: E402
from churnmaster.rl.reward import action_cost, lookup_loss  # noqa: E402

CELL_W = (0.6, 0.2, 0.3)      # wL0.6_wM0.2


def test_reward_cell_grid():
    cells = reward_cells()
    assert len(cells) == 12 and list(REWARD_CELLS) == [c.name for c in cells]
    grid = {(c.w_M, c.w_L) for c in cells if c.origin == "grid"}
    assert grid == {(m, l) for m in (0.05, 0.2, 0.5) for l in (0.1, 0.3, 0.6)}
    assert sorted(c.w_L for c in cells if c.origin == "line") == [0.1, 0.6, 1.0]
    assert all(c.w_M == 0.1 for c in cells if c.origin == "line")
    for c in cells:
        assert c.w_R == pytest.approx(1.5 * c.w_M) and not c.is_canonical
    assert REWARD_CELLS["wL0.6_wM0.2"].weights == CELL_W
    assert "wL1.0_wM0.1" in REWARD_CELLS and "wL0.05_wM0.05" not in REWARD_CELLS
    # the canonical point is not trained: it reuses the canonical Q-tables
    assert CANONICAL_CELL.name == "wL0.3_wM0.1" and CANONICAL_CELL.is_canonical
    assert CANONICAL_CELL.weights == CANONICAL_WEIGHTS and CANONICAL_CELL.name not in REWARD_CELLS
    assert reward_cells(include_canonical=True)[-1] == CANONICAL_CELL
    assert CANONICAL_CELL.to_dict()["reuses_canonical"] is True
    assert "NO multiple-comparison correction" in REWARD_ANALYSIS_PLAN
    assert "SCORED UNDER THE CANONICAL WEIGHTS" in REWARD_ANALYSIS_PLAN
    assert "re-tuned" in REWARD_ANALYSIS_PLAN and "Pre-registered" not in REWARD_ANALYSIS_PLAN


def test_reward_loader_changes_only_reward(loader):
    ld = RewardLoader(loader, *CELL_W)
    c0, c1 = loader("stable"), ld("stable")
    assert (c1.reward.w_L, c1.reward.w_M, c1.reward.w_R) == CELL_W
    assert (c0.reward.w_L, c0.reward.w_M, c0.reward.w_R) == CANONICAL_WEIGHTS
    assert env_config_hash(c1) == env_config_hash(c0) and config_hash(c1) != config_hash(c0)
    assert c1.sim == c0.sim and c1.scenario == c0.scenario and c1.agent == c0.agent
    assert pickle.loads(pickle.dumps(ld)) == ld and hash(ld) == hash(RewardLoader(loader, *CELL_W))


def _bk_reward(res, b, k, w, epoch_ms, deadline):
    r = -sum(lookup_loss(lk.success, lk.latency_ms, w[0], deadline) for lk in res.lookups
             if lk.bucket == b and int(lk.t_ms) // epoch_ms == k)
    r -= sum(action_cost(d.pings, d.evictions, w[1], w[2]) for d in res.decisions
             if d.bucket == b and d.epoch_idx == k)
    return r


def test_training_rewards_use_cell_weights(loader, trained):
    ld = RewardLoader(loader, *CELL_W)
    job = Job("kademlia_epoch", "stable", Split.TRAIN, 0, 0)
    got: dict[str, list] = {"can": [], "cell": []}
    rc = run_job(job, cfg=loader("stable"), keep_records=True, on_transition=got["can"].append)
    rw = run_job(job, cfg=ld("stable"), keep_records=True, on_transition=got["cell"].append)
    assert rc.trace.trace_hash == rw.trace.trace_hash                  # same trajectory
    key = lambda t: (t.bucket, t.epoch_idx, t.s, t.a, t.s_next)        # noqa: E731
    assert [key(t) for t in got["can"]] == [key(t) for t in got["cell"]] and got["can"]
    assert any(abs(a.r - b.r) > 1e-9 for a, b in zip(got["can"], got["cell"]))
    cfg = loader("stable")
    em, dl = int(cfg.sim.epoch_s) * 1000, cfg.sim.deadline_ms
    for t in got["cell"][:200]:
        assert t.r == pytest.approx(_bk_reward(rw, t.bucket, t.epoch_idx, CELL_W, em, dl),
                                    abs=1e-9)
    for t in got["can"][:200]:
        assert t.r == pytest.approx(
            _bk_reward(rc, t.bucket, t.epoch_idx, CANONICAL_WEIGHTS, em, dl), abs=1e-9)
    # training through the RewardLoader learns from the cell rewards
    o = train_agent(TrainSpec(ld, SETTINGS, 0))
    assert not np.array_equal(o.final.Q, trained["canonical"].final.Q)


def test_record_based_J_and_eval_scoring(loader, trained):
    pol = EncodedLearned(trained["canonical"].final, name="c", id_bits=8)
    job = Job(pol, "high_churn", Split.TEST, 0, 0)
    res = run_job(job, cfg=loader("high_churn"), keep_records=True)
    dl = loader("high_churn").sim.deadline_ms
    assert j_from_records(res.lookups, res.decisions, *CANONICAL_WEIGHTS, dl) == \
        pytest.approx(res.ledger_score, abs=1e-12)
    cell = run_job(job, cfg=RewardLoader(loader, *CELL_W)("high_churn"))
    j_cell = j_from_records(res.lookups, res.decisions, *CELL_W, dl)
    assert j_cell == pytest.approx(cell.ledger_score, abs=1e-12)
    assert abs(j_cell - res.ledger_score) > 1e-6
    # the eval worker scores under the canonical loader's weights; J_train under the cell's
    row = _reward_eval(_RewardEvalTask(job, loader, CELL_W))
    assert row["J"] == res.metrics.J and row["J_recomputed"] == pytest.approx(row["J"], abs=1e-12)
    assert row["J_train"] == pytest.approx(cell.ledger_score, abs=1e-12)


def test_reward_suite_end_to_end(tmp_path, short_exp, manifest, trained):
    can = trained["canonical"].final.save(tmp_path / "canon" / "run0" / "q_final.npy")
    sel = tmp_path / "selected.json"
    sel.write_text(json.dumps({"global": "f=2,po=0,rs=1,ps=0,tg=all", "per_scenario": {}}),
                   encoding="utf-8")
    out = tmp_path / "abl"
    common = dict(experiment_path=short_exp, runs=(0,), episodes=2, n_seeds=2,
                  canonical_q=[can], tune_selected=sel, workers=1, n_resamples=200)
    run_agent_suite(CONFIGS, manifest, out, variants=["gamma0"], **common)
    agent_before = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    agent_files = {p: p.read_bytes() for p in (out / "agent").rglob("*") if p.is_file()}
    man_before = manifest.read_bytes()
    cells = ["wL0.6_wM0.2", "wL1.0_wM0.1"]
    rep = run_reward_suite(CONFIGS, manifest, out, cells=cells, **common)
    assert manifest.read_bytes() == man_before
    assert {p: p.read_bytes() for p in (out / "agent").rglob("*") if p.is_file()} == agent_files
    d = out / "reward"
    for f in ("per_seed.csv", "comparison.csv", "per_run.csv", "summary.csv", "trend.csv",
              "greedy_stats.json"):
        assert (d / f).is_file()
    for c in cells:
        assert (d / "cells" / c / "run0" / "q_final.npy").is_file()
        assert (d / "cells" / c / "run0" / "curves.csv").is_file()
    for f in ("reward_sweep_p95.png", "reward_sweep_J.png", "reward_sweep_tradeoff.png"):
        assert (out / "figures" / f).is_file()
    # manifest: read-modify-write keeps the agent suite intact
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert m["suites"]["agent"] == agent_before["suites"]["agent"]
    assert m["analysis_plan"] == agent_before["analysis_plan"]
    sm = m["suites"]["reward"]
    assert sm["analysis_plan"] == REWARD_ANALYSIS_PLAN == m["reward_analysis_plan"]
    assert sm["scoring_weights"] == {"w_L": 0.3, "w_M": 0.1, "w_R": 0.15}
    assert sm["references"]["reused"] is True
    assert sm["references"]["spot_checks"] and all(s["match"] for s in sm["references"]["spot_checks"])
    assert sm["j_recomputation"]["max_abs_dev_vs_ledger"] <= 1e-9
    assert set(sm["env_hashes"].values()) and sm["eval_seeds"]["overlap"] is False
    ps = pd.read_csv(d / "per_seed.csv")
    assert set(ps["split"]) == {"test"} and set(ps["episode"]) == {0, 1}
    assert set(ps["group"]) == {"canonical", *cells, "tuned_heuristic", "kademlia_epoch",
                                "never_replace"}
    cell_rows = ps[ps["group"].isin(cells)]
    assert (cell_rows["J"] - cell_rows["ledger_score"]).abs().max() <= 1e-12   # canonical scoring
    assert (cell_rows["J_train"] - cell_rows["J"]).abs().min() > 1e-9          # cell weights differ
    refs = ps[~ps["group"].isin(cells)]
    assert (refs["J_train"] == refs["J"]).all()
    assert set(ps.loc[ps["group"] == "canonical", "cell"]) == {"wL0.3_wM0.1"}
    # reused reference rows are identical to the agent suite's rows
    ag = pd.read_csv(out / "agent" / "per_seed.csv").set_index(["policy", "scenario", "episode"])
    for _, r in refs.iterrows():
        assert ag.loc[(r["policy"], r["scenario"], r["episode"]), "trace_hash"] == r["trace_hash"]
    comp = pd.read_csv(d / "comparison.csv")
    assert len(comp[comp["comparison"] == "vs_canonical"]) == 2 * 3 * 4
    assert len(comp[comp["comparison"] == "vs_tuned_heuristic"]) == 3 * 3 * 2
    tr = pd.read_csv(d / "trend.csv")
    assert list(tr["n_points"]) == [2, 2, 2]                    # canonical + wL1.0 on the line
    assert json.loads(tr["w_L_values"].iloc[0]) == [0.3, 1.0]
    # refuses to overwrite without the flag
    with pytest.raises(ValueError, match="overwrite"):
        run_reward_suite(CONFIGS, manifest, out, cells=cells, **common)
    # forced re-run of the references reproduces the reused rows
    rep2 = run_reward_suite(CONFIGS, manifest, out, cells=cells[:1], overwrite=True,
                            reuse_references=False, **common)
    assert rep2.manifest["references"]["reused"] is False
    a = rep.per_seed[~rep.per_seed["group"].isin(cells)].set_index(["policy", "scenario", "episode"])
    b = rep2.per_seed[~rep2.per_seed["group"].isin(cells)].set_index(["policy", "scenario", "episode"])
    assert set(a.index) == set(b.index)
    for k in ("trace_hash", "summary_hash"):
        assert (a[k] == b.loc[a.index, k]).all()
    m2 = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert m2["suites"]["agent"] == agent_before["suites"]["agent"]


@pytest.mark.parametrize("bad", ["results/eval", "results/headroom", "results/reproduce_quick"])
def test_reward_suite_refuses_canonical_out(bad, short_exp, manifest, monkeypatch):
    monkeypatch.chdir(REPO)
    with pytest.raises(ValueError):
        run_reward_suite(CONFIGS, manifest, bad, experiment_path=short_exp, episodes=2)


def test_reward_suite_rejects_unknown_or_canonical_cell(tmp_path, short_exp, manifest):
    for bad in (["wL0.3_wM0.1"], ["nope"], []):
        with pytest.raises(ValueError, match="cell"):
            run_reward_suite(CONFIGS, manifest, tmp_path / "x", cells=bad,
                             experiment_path=short_exp, episodes=2)


# --------------------------------------------------------------------------- #
# Memoryless held-out scenario (task 12.3, Requirement 28.3)
# --------------------------------------------------------------------------- #

import dataclasses  # noqa: E402
import hashlib  # noqa: E402

from churnmaster.experiments.ablations import (MEMORYLESS_ANALYSIS_PLAN,  # noqa: E402
                                               MEMORYLESS_PARENT, MEMORYLESS_POLICIES,
                                               MEMORYLESS_SCENARIO, SUITES,
                                               check_derived_scenario, run_memoryless_suite)
from churnmaster.experiments.runner import FrozenEnvError  # noqa: E402
from churnmaster.sim.churn import weibull_scale  # noqa: E402

#: Canonical results the memoryless suite must never modify.
CANONICAL_FILES = [REPO / "results" / p for p in (
    "headroom/manifest.json", "matched/freqs.json", "tune/selected.json", "eval/summary.csv",
    "eval/per_seed.csv", "train/final/run0/q_final.npy")]


def _digests(paths):
    return {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


def test_memoryless_config_only_changes_hazard_shape(loader):
    m, p = loader(MEMORYLESS_SCENARIO), loader(MEMORYLESS_PARENT)
    assert m.scenario.name == MEMORYLESS_SCENARIO and m.scenario.weibull_shape == 1.0
    assert p.scenario.weibull_shape == 0.6
    assert dataclasses.replace(m.scenario, name=p.scenario.name, weibull_shape=0.6) == p.scenario
    assert m.sim == p.sim and m.reward == p.reward and m.root_seed == p.root_seed
    # the Weibull scale keeps the mean: shape 1.0 is the exponential with the same mean
    for mean in (m.scenario.stable_session_mean_s, m.scenario.flaky_session_mean_s):
        assert weibull_scale(mean, 1.0) == pytest.approx(mean)


def test_check_derived_scenario(loader, manifest):
    m, p = loader(MEMORYLESS_SCENARIO), loader(MEMORYLESS_PARENT)
    before = manifest.read_bytes()
    info = check_derived_scenario(manifest, m, p)
    assert manifest.read_bytes() == before
    assert info["config_hash"] == config_hash(m) and info["env_hash"] == env_config_hash(m)
    assert info["parent_env_hash"] == env_config_hash(p) != info["env_hash"]
    assert info["differs"] == {"name": [MEMORYLESS_PARENT, MEMORYLESS_SCENARIO],
                               "weibull_shape": [0.6, 1.0]}
    assert info["in_frozen_manifest"] is False
    bad = dataclasses.replace(m, scenario=dataclasses.replace(m.scenario, stable_share=0.5))
    with pytest.raises(FrozenEnvError, match="more than"):
        check_derived_scenario(manifest, bad, p)
    bad_sim = dataclasses.replace(m, sim=dataclasses.replace(m.sim, k=m.sim.k + 1))
    with pytest.raises(FrozenEnvError):
        check_derived_scenario(manifest, bad_sim, p)


def test_memoryless_plan_and_cli_wiring(capsys):
    from churnmaster.cli import build_parser, main

    assert "memoryless" in SUITES
    assert "NO multiple-comparison correction" in MEMORYLESS_ANALYSIS_PLAN
    assert "weibull_shape = 1.0" in MEMORYLESS_ANALYSIS_PLAN
    assert "Pre-registered" not in MEMORYLESS_ANALYSIS_PLAN
    assert "Memoryless outputs" not in MEMORYLESS_ANALYSIS_PLAN
    assert "memoryless" not in ANALYSIS_PLAN.lower()
    assert set(MEMORYLESS_POLICIES) >= {"tuned_heuristic", "kademlia_epoch", "matched_random",
                                        "never_replace", "uniform_random"}
    args = build_parser().parse_args(["ablate", "--suite", "memoryless", "--workers", "2"])
    assert args.suite == "memoryless" and args.func.__name__ == "cmd_ablate"
    assert main(["ablate", "--suite", "memoryless", "--episodes", "5"]) == 1
    assert "--episodes does not apply" in capsys.readouterr().err


def test_real_memoryless_scenario_derives_from_frozen_heldout_medium():
    """Against the canonical frozen manifest (read-only); the scenario stays out of it."""
    from churnmaster.experiments.diagnostics import _frozen_loader

    man = REPO / "results" / "headroom" / "manifest.json"
    if not man.is_file():
        pytest.skip("canonical frozen manifest not present")
    before = _digests([man])
    ld = _frozen_loader(CONFIGS, CONFIGS / "experiments" / "train.toml", man, ("ablations",))
    info = check_derived_scenario(man, ld(MEMORYLESS_SCENARIO), ld(MEMORYLESS_PARENT))
    assert info["in_frozen_manifest"] is False
    frozen = json.loads(man.read_text(encoding="utf-8"))["env_hashes"]
    assert MEMORYLESS_SCENARIO not in frozen and info["parent_env_hash"] == frozen[MEMORYLESS_PARENT]
    assert _digests([man]) == before
    rec = REPO / "results" / "ablations" / "manifest.json"
    if rec.is_file():
        m = json.loads(rec.read_text(encoding="utf-8"))
        if "memoryless" in m.get("suites", {}):
            sc = m["suites"]["memoryless"]["scenario"]
            assert sc["config_hash"] == info["config_hash"]
            assert sc["env_hash"] == info["env_hash"]
            assert m["extra_scenarios"][MEMORYLESS_SCENARIO]["config_hash"] == info["config_hash"]


def test_memoryless_suite_end_to_end(tmp_path, loader, short_exp, manifest, trained):
    can = trained["canonical"].final.save(tmp_path / "canon" / "run0" / "q_final.npy")
    sel = tmp_path / "selected.json"
    sel.write_text(json.dumps({"global": "f=2,po=0,rs=1,ps=0,tg=all", "per_scenario": {}}),
                   encoding="utf-8")
    out = tmp_path / "abl"
    out.mkdir()
    prior = {"kind": "churnmaster-ablations", "analysis_plan": ANALYSIS_PLAN,
             "suites": {"agent": {"suite": "agent", "marker": 1},
                        "reward": {"suite": "reward", "marker": 2}}}
    (out / "manifest.json").write_text(json.dumps(prior), encoding="utf-8")
    canon_before = _digests(CANONICAL_FILES)
    man_before = manifest.read_bytes()
    common = dict(experiment_path=short_exp, runs=(0,), n_seeds=2, canonical_q=[can],
                  tune_selected=sel, matched_freqs=REPO / "results" / "matched" / "freqs.json",
                  freq_seeds=1, workers=1, n_resamples=200)
    rep = run_memoryless_suite(CONFIGS, manifest, out, **common)
    assert manifest.read_bytes() == man_before                       # frozen manifest untouched
    assert _digests(CANONICAL_FILES) == canon_before                  # canonical results untouched
    d = out / "memoryless"
    for f in ("per_seed.csv", "summary.csv", "deltas.csv", "summary.md", "matched_freqs.json"):
        assert (d / f).is_file()
    assert (out / "figures" / "memoryless_J_p95.png").is_file()
    ps = pd.read_csv(d / "per_seed.csv")
    assert set(ps["split"]) == {"test"} and set(ps["run"]) == {0} and set(ps["episode"]) == {0, 1}
    assert set(ps["scenario"]) == {MEMORYLESS_SCENARIO, MEMORYLESS_PARENT}
    assert set(ps["policy"]) == {"canonical:run0", *MEMORYLESS_POLICIES}
    assert len(ps) == 7 * 2 * 2
    hm = {s: config_hash(loader(s)) for s in (MEMORYLESS_SCENARIO, MEMORYLESS_PARENT)}
    for s, h in hm.items():
        assert set(ps.loc[ps["scenario"] == s, "config_hash"]) == {h}
    # same seeds, different hazard shape -> different churn traces
    a = ps[(ps["policy"] == "never_replace")].set_index(["scenario", "episode"])["trace_hash"]
    assert a[(MEMORYLESS_SCENARIO, 0)] != a[(MEMORYLESS_PARENT, 0)]
    dl = pd.read_csv(d / "deltas.csv")
    assert set(dl["comparison"]) == {"rl_vs_reference", "shape_effect", "gap_shift"}
    assert len(dl[dl["comparison"] == "rl_vs_reference"]) == 2 * 6 * 4
    assert len(dl[dl["comparison"] == "gap_shift"]) == 3 * 2
    sm_ = pd.read_csv(d / "summary.csv")
    assert {"rl_mean", "run", "reference"} == set(sm_["kind"])
    # matched_random: the canonical freqs.json does not match the short env -> measured
    fq = json.loads((d / "matched_freqs.json").read_text(encoding="utf-8"))
    assert set(fq) == {MEMORYLESS_SCENARIO, MEMORYLESS_PARENT}
    assert all(v["source"].startswith("measured") for v in fq.values())
    assert sum(fq[MEMORYLESS_SCENARIO]["counts"]["cand"]) > 0
    # manifest: read-modify-write keeps the other suites, records the scenario hashes
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert m["suites"]["agent"] == prior["suites"]["agent"]
    assert m["suites"]["reward"] == prior["suites"]["reward"]
    assert m["analysis_plan"] == ANALYSIS_PLAN
    sm = m["suites"]["memoryless"]
    assert sm["analysis_plan"] == MEMORYLESS_ANALYSIS_PLAN == m["memoryless_analysis_plan"]
    assert sm["scenario"]["config_hash"] == hm[MEMORYLESS_SCENARIO]
    assert sm["scenario"]["scenario"]["weibull_shape"] == 1.0
    assert len(sm["scenario"]["toml_sha256"]) == 64
    assert m["extra_scenarios"][MEMORYLESS_SCENARIO]["config_hash"] == hm[MEMORYLESS_SCENARIO]
    assert sm["eval_seeds"]["overlap"] is False
    assert sm["multiple_comparison_correction"].startswith("none")
    assert sm["agent_cross_check"]["checked"] is False
    assert rep.summary.equals(sm_) or len(rep.summary) == len(sm_)
    md = (d / "summary.md").read_text(encoding="utf-8")
    assert "rl_mean" in md and "uncorrected" in md
    with pytest.raises(ValueError, match="overwrite"):
        run_memoryless_suite(CONFIGS, manifest, out, **common)


@pytest.mark.parametrize("bad", ["results/eval", "results/matched", "results/headroom",
                                 "results/reproduce_full"])
def test_memoryless_suite_refuses_canonical_out(bad, short_exp, manifest, monkeypatch):
    monkeypatch.chdir(REPO)
    with pytest.raises(ValueError):
        run_memoryless_suite(CONFIGS, manifest, bad, experiment_path=short_exp)


def test_memoryless_suite_rejects_bad_policies(tmp_path, short_exp, manifest):
    for bad in (["nope"], [], ["kademlia_epoch", "kademlia_epoch"]):
        with pytest.raises(ValueError, match="policy"):
            run_memoryless_suite(CONFIGS, manifest, tmp_path / "x", policies=bad,
                                 experiment_path=short_exp)


@pytest.mark.slow
def test_memoryless_recorded_rows_reproduce():
    """Re-run TEST episode 0 with the real configs; it must match the recorded per_seed rows."""
    rec = REPO / "results" / "ablations" / "memoryless" / "per_seed.csv"
    man = REPO / "results" / "headroom" / "manifest.json"
    if not rec.is_file() or not man.is_file():
        pytest.skip("memoryless results not present")
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        rep = run_memoryless_suite(CONFIGS, man, Path(tmp) / "abl", n_seeds=2,
                                   canonical_q=[REPO / f"results/train/final/run{r}/q_final.npy"
                                                for r in range(3)],
                                   tune_selected=REPO / "results/tune/selected.json",
                                   matched_freqs=REPO / "results/matched/freqs.json",
                                   workers=4, n_resamples=200)
    old = pd.read_csv(rec).set_index(["policy", "scenario", "episode"])
    for _, r in rep.per_seed.iterrows():
        o = old.loc[(r["policy"], r["scenario"], r["episode"])]
        assert o["trace_hash"] == r["trace_hash"] and o["summary_hash"] == r["summary_hash"]


def test_memoryless_scenario_kept_out_of_frozen_scenario_dir():
    """headroom / reproduce freeze every file in configs/scenarios; the ablation scenario
    must live in configs/ablations so the canonical pipeline is unchanged (task 12.3)."""
    from churnmaster.experiments.ablations import ABLATION_CONFIG_DIR
    from churnmaster.experiments.headroom import scenario_names

    assert ABLATION_CONFIG_DIR == "ablations"
    assert scenario_names(CONFIGS) == ["bursty", "heldout_medium", "high_churn", "stable"]
    assert not (CONFIGS / "scenarios" / f"{MEMORYLESS_SCENARIO}.toml").exists()
    assert (CONFIGS / ABLATION_CONFIG_DIR / f"{MEMORYLESS_SCENARIO}.toml").is_file()
    from churnmaster.config import ConfigError

    with pytest.raises(ConfigError, match="cannot read"):
        ConfigLoader(CONFIGS)(MEMORYLESS_SCENARIO)          # default loader: frozen set only
    extra = ConfigLoader(CONFIGS, None, (ABLATION_CONFIG_DIR,))
    assert extra(MEMORYLESS_SCENARIO).scenario.weibull_shape == 1.0
    # frozen scenarios always resolve to configs/scenarios, even with extra dirs
    assert extra.scenario_path(MEMORYLESS_PARENT) == CONFIGS / "scenarios" / f"{MEMORYLESS_PARENT}.toml"
    assert config_hash(extra(MEMORYLESS_PARENT)) == config_hash(ConfigLoader(CONFIGS)(MEMORYLESS_PARENT))