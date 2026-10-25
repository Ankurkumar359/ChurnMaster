"""Import guard: static AST check of the observation boundary (Requirements 12.2, 12.3).

Every ``.py`` under ``src/churnmaster`` is parsed (never imported) and each import is
resolved to fully-qualified module names, covering absolute imports, relative imports,
``from pkg import submodule`` forms, and best-effort ``importlib.import_module("...")`` /
``__import__("...")`` string literals.
"""

from __future__ import annotations

import ast
import pathlib
import sys

import pytest

SRC_ROOT = pathlib.Path(__file__).resolve().parents[1] / "src"
PKG_ROOT = SRC_ROOT / "churnmaster"

HIDDEN_MODULES = tuple(
    f"churnmaster.sim.{m}" for m in ("network", "churn", "background", "lookup", "env", "keyed_rng")
)
LEARNER_PACKAGES = ("churnmaster.rl", "churnmaster.policies")


# --------------------------------------------------------------------------- resolver


def module_name_for(path: pathlib.Path) -> tuple[str, bool]:
    """Return (dotted module name, is_package) for a file under ``SRC_ROOT``."""
    parts = list(path.resolve().relative_to(SRC_ROOT).with_suffix("").parts)
    is_pkg = parts[-1] == "__init__"
    if is_pkg:
        parts = parts[:-1]
    return ".".join(parts), is_pkg


def _resolve_relative(name: str, level: int, module: str, is_pkg: bool) -> str:
    """Resolve a relative import of ``level`` dots against ``module``."""
    package = module if is_pkg else module.rpartition(".")[0]
    pkg_parts = package.split(".") if package else []
    if level - 1 > len(pkg_parts) - 1:
        raise ImportError(f"relative import beyond top-level package in {module}")
    base = pkg_parts[: len(pkg_parts) - (level - 1)]
    return ".".join(base + ([name] if name else []))


def _string_import_target(node: ast.Call) -> tuple[str, str | None] | None:
    """Return (target, package-arg) for import_module/__import__ calls with literal args."""
    func = node.func
    fname = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
    if fname not in ("import_module", "__import__") or not node.args:
        return None
    first = node.args[0]
    if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
        return None
    pkg: str | None = None
    if fname == "import_module":
        cand = node.args[1] if len(node.args) > 1 else next(
            (k.value for k in node.keywords if k.arg == "package"), None
        )
        if isinstance(cand, ast.Constant) and isinstance(cand.value, str):
            pkg = cand.value
    return first.value, pkg


def imported_modules(source: str, module: str, is_pkg: bool = False) -> set[str]:
    """All fully-qualified names a module's source may import.

    ``from X import a`` yields both ``X`` and ``X.a`` (``a`` may be a submodule), and
    ``from X import *`` yields ``X.*``.
    """
    tree = ast.parse(source)
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = (
                _resolve_relative(node.module or "", node.level, module, is_pkg)
                if node.level
                else node.module or ""
            )
            out.add(base)
            for a in node.names:
                out.add(f"{base}.{a.name}" if base else a.name)
        elif isinstance(node, ast.Call):
            hit = _string_import_target(node)
            if hit is None:
                continue
            target, pkg = hit
            if target.startswith("."):
                level = len(target) - len(target.lstrip("."))
                if pkg is not None:
                    out.add(_resolve_relative(target[level:], level, pkg, True))
                else:
                    out.add(_resolve_relative(target[level:], level, module, is_pkg))
            else:
                out.add(target)
    return out


def violations(imports: set[str], forbidden: tuple[str, ...]) -> list[str]:
    """Imported names that equal, live under, or star-import over a forbidden module."""
    bad = set()
    for name in imports:
        for f in forbidden:
            if name == f or name.startswith(f + "."):
                bad.add(name)
            elif name.endswith(".*") and f.startswith(name[:-1]):
                bad.add(name)  # star import from a package containing a forbidden module
    return sorted(bad)


# --------------------------------------------------------------------------- file sets


def _rel(p: pathlib.Path) -> str:
    return p.relative_to(SRC_ROOT).as_posix()


def observable_only_files() -> list[pathlib.Path]:
    files = list((PKG_ROOT / "rl").rglob("*.py"))
    files += [p for p in (PKG_ROOT / "policies").rglob("*.py") if p.name != "oracle.py"]
    for name in ("routing_table.py", "observation.py"):
        p = PKG_ROOT / "sim" / name
        if p.exists():
            files.append(p)
    return sorted(files)


def sim_files() -> list[pathlib.Path]:
    return sorted((PKG_ROOT / "sim").rglob("*.py"))


def _check(path: pathlib.Path, forbidden: tuple[str, ...]) -> list[str]:
    module, is_pkg = module_name_for(path)
    imports = imported_modules(path.read_text(encoding="utf-8"), module, is_pkg)
    return violations(imports, forbidden)


# --------------------------------------------------------------------------- guards


@pytest.mark.parametrize("path", observable_only_files(), ids=_rel)
def test_observable_only_modules_do_not_import_hidden_state(path):
    bad = _check(path, HIDDEN_MODULES)
    assert not bad, f"{_rel(path)} imports hidden-state module(s): {', '.join(bad)}"


@pytest.mark.parametrize("path", sim_files(), ids=_rel)
def test_sim_does_not_import_learners(path):
    bad = _check(path, LEARNER_PACKAGES)
    assert not bad, f"{_rel(path)} imports learner module(s): {', '.join(bad)}"


def test_guarded_set_covers_routing_table():
    assert PKG_ROOT / "sim" / "routing_table.py" in observable_only_files()


def test_oracle_is_exempt():
    assert all(p.name != "oracle.py" for p in observable_only_files())


# --------------------------------------------------------------------------- resolver self-tests

RL_MOD = "churnmaster.rl.agent"
SIM_MOD = "churnmaster.sim.routing_table"


@pytest.mark.parametrize(
    "src, module, expected",
    [
        ("import churnmaster.sim.network", RL_MOD, "churnmaster.sim.network"),
        ("import churnmaster.sim.network as n", RL_MOD, "churnmaster.sim.network"),
        ("from churnmaster.sim import network", RL_MOD, "churnmaster.sim.network"),
        ("from churnmaster.sim.network import Network", RL_MOD, "churnmaster.sim.network.Network"),
        ("from churnmaster.sim import *", RL_MOD, "churnmaster.sim.*"),
        ("from ..sim import network", RL_MOD, "churnmaster.sim.network"),
        ("from ..sim.lookup import run", RL_MOD, "churnmaster.sim.lookup.run"),
        ("from . import keyed_rng", SIM_MOD, "churnmaster.sim.keyed_rng"),
        ("from .churn import Q", SIM_MOD, "churnmaster.sim.churn.Q"),
        ("from ...sim import env", "churnmaster.policies.sub.x", "churnmaster.sim.env"),
        ("import importlib\nimportlib.import_module('churnmaster.sim.network')", RL_MOD, "churnmaster.sim.network"),
        ("from importlib import import_module\nimport_module('..sim.env', 'churnmaster.rl')", RL_MOD, "churnmaster.sim.env"),
        ("import_module('.background', package='churnmaster.sim')", RL_MOD, "churnmaster.sim.background"),
        ("__import__('churnmaster.sim.lookup')", RL_MOD, "churnmaster.sim.lookup"),
    ],
)
def test_resolver_resolves(src, module, expected):
    assert expected in imported_modules(src, module)


@pytest.mark.parametrize(
    "src, module",
    [
        ("from ..sim import network", RL_MOD),
        ("from churnmaster.sim import network", RL_MOD),
        ("import churnmaster.sim.keyed_rng", RL_MOD),
        ("from .churn import ChurnProcess", SIM_MOD),
        ("from churnmaster.sim import *", RL_MOD),
        ("import importlib\nimportlib.import_module('churnmaster.sim.network')", RL_MOD),
        ("def f():\n    from churnmaster.sim.env import Env", RL_MOD),
    ],
)
def test_resolver_flags_hidden_imports(src, module):
    assert violations(imported_modules(src, module), HIDDEN_MODULES)


@pytest.mark.parametrize(
    "src, module",
    [
        ("from churnmaster.sim.idspace import bucket_index", RL_MOD),
        ("from ..sim import idspace", RL_MOD),
        ("from .idspace import xor", SIM_MOD),
        ("from churnmaster.sim.routing_table import RoutingView", RL_MOD),
        ("import churnmaster.sim.networking_utils", RL_MOD),  # prefix lookalike, not network
        ("from churnmaster import config", RL_MOD),
        ("import numpy as np", RL_MOD),
    ],
)
def test_resolver_allows_observable_imports(src, module):
    assert not violations(imported_modules(src, module), HIDDEN_MODULES)


@pytest.mark.parametrize(
    "src, flagged",
    [
        ("from churnmaster import rl", True),
        ("from churnmaster.policies.q import QPolicy", True),
        ("from ..rl import agent", True),
        ("from ..policies import oracle", True),
        ("from churnmaster import config", False),
        ("from .idspace import xor", False),
    ],
)
def test_resolver_learner_guard(src, flagged):
    assert bool(violations(imported_modules(src, SIM_MOD), LEARNER_PACKAGES)) is flagged


def test_package_init_relative_resolution():
    # In churnmaster/sim/__init__.py, "." is churnmaster.sim itself.
    assert "churnmaster.sim.network" in imported_modules("from . import network", "churnmaster.sim", True)


def test_violation_message_names_file_and_module(tmp_path, monkeypatch):
    fake_src = tmp_path / "src"
    fake = fake_src / "churnmaster" / "rl" / "leaky.py"
    fake.parent.mkdir(parents=True)
    fake.write_text("from ..sim import network\n", encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "SRC_ROOT", fake_src)
    with pytest.raises(AssertionError) as exc:
        test_observable_only_modules_do_not_import_hidden_state(fake)
    msg = str(exc.value)
    assert "churnmaster/rl/leaky.py" in msg and "churnmaster.sim.network" in msg


# --------------------------------------------------------------------------- counterfactual
# Requirement 12.4: same DecisionContext, different hidden state -> same action from
# every non-oracle policy. Hidden worlds are hand-built Networks that share F's scripted
# LocalView but disagree on who is online (plus positions and keyed root); the oracle,
# fed ground truth from each world via the real ``Env.oracle_facts``, must notice.

import dataclasses  # noqa: E402
import inspect  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
from hypothesis import given, strategies as st  # noqa: E402

from churnmaster.config import load_config  # noqa: E402
from churnmaster.policies.baselines import BASELINE_NAMES, RANDOM_BASELINES, make_baseline  # noqa: E402
from churnmaster.policies.heuristic import DEFAULT_CONFIG, Heuristic, all_configs  # noqa: E402
from churnmaster.policies.oracle import OracleFacts, OracleHeuristic  # noqa: E402
from churnmaster.seeding import Split, env_seed, make_streams  # noqa: E402
from churnmaster.sim.churn import ChurnTrace, NodeType, Session  # noqa: E402
from churnmaster.sim.env import Env  # noqa: E402
from churnmaster.sim.network import Network  # noqa: E402
from churnmaster.sim.observation import DecisionContext, build_context  # noqa: E402
from churnmaster.sim.routing_table import LocalView  # noqa: E402

CONFIGS = SRC_ROOT.parent / "configs"
_CFG = load_config(CONFIGS / "base.toml", CONFIGS / "scenarios" / "high_churn.toml")
_HORIZON = 3_000_000
_FREQS = {False: (0.5, 0.5, 0.0), True: (0.2, 0.3, 0.5)}
_RNG_SEEDS = {"uniform_random": 123, "matched_random": 456}

FOCAL = 0
# bucket 7: 128..255, bucket 6: 64..127, bucket 5: 32..63, bucket 4: 16..31
_BOOT = (128, 140, 150, 160, 64, 70, 32, 33, 34, 35, 16)
_CACHED = (200, 210, 40)
NODES = _BOOT + _CACHED
_INCUMBENT_SET = (128, 64, 33, 16)  # LRS incumbents after the script below
_CTX_TIMES = ((1, 30_000), (3, 90_000), (10, 300_000), (44, 1_320_000))

EXPECTED_CONTEXT_FIELDS = {
    "bucket", "epoch_idx", "t_ms", "inc_consec_fail", "inc_since_response_s", "inc_age_s",
    "inc_srtt_ms", "cand_since_seen_s", "cand_last_rtt_ms", "timeout_fraction",
}


def scripted_view() -> LocalView:
    """F's observations: identical in every hidden world (no network involved)."""
    s = _CFG.sim
    v = LocalView(FOCAL, id_bits=s.id_bits, k=s.k, cache_size=s.cache_size,
                  cache_ttl_ms=int(s.cache_ttl_s) * 1000,
                  window_ms=int(s.window_timeout_s) * 1000, rtt_beta=s.rtt_beta)
    v.bootstrap((n, 0, None) for n in _BOOT)
    v.record_response(140, 5_000, 40.0)
    v.record_response(150, 6_000, 80.0)
    v.record_response(160, 6_500, 120.0)
    v.record_failure(128, 7_000)
    v.record_failure(128, 8_000)
    v.record_response(70, 9_000, 25.0)
    v.record_response(32, 9_500, 200.0)
    v.record_response(34, 9_600, 210.0)
    v.record_response(35, 9_700, 220.0)
    v.record_failure(33, 9_800)
    for t, timed_out in ((5_000, False), (7_000, True), (8_000, True), (9_800, True),
                         (12_000, False)):
        v.record_query_outcome(t, timed_out)
    v.cache_put(7, 210, 10_000, 30.0)
    v.cache_put(7, 200, 20_000, 15.0)
    v.cache_put(5, 40, 25_000, 300.0)
    return v


def scripted_contexts(view: LocalView) -> list[DecisionContext]:
    return [build_context(view, b, k, t) for k, t in _CTX_TIMES
            for b in range(view.id_bits) if view.bucket(b)]


def make_world(online: set[int], keyed_root: int = 7, pos_seed: int = 0) -> Network:
    """Hidden world over IDs ``(FOCAL,) + NODES``: ``online`` IDs are up the whole episode."""
    ids = [FOCAL, *NODES]
    n = len(ids)
    sessions = [[Session(i, 0, _HORIZON)] if i == 0 or ids[i] in online else []
                for i in range(n)]
    positions = np.random.default_rng(pos_seed).random((n, 2))
    return Network(_CFG.sim, ids, positions, [NodeType(True)] * n,
                   ChurnTrace(n, _HORIZON, sessions), {}, keyed_root)


def oracle_for(view: LocalView, net: Network) -> OracleHeuristic:
    """Oracle with ground truth from ``net`` via the real ``Env.oracle_facts``."""
    env_like = SimpleNamespace(view=view, net=net)
    return OracleHeuristic(lambda b, t: OracleFacts(*Env.oracle_facts(env_like, b, t)))


def non_oracle_policies() -> dict[str, object]:
    """Fresh instances of every non-oracle policy (random ones identically seeded)."""
    out: dict[str, object] = {}
    for name in BASELINE_NAMES:
        if name in RANDOM_BASELINES:
            rng = np.random.default_rng(_RNG_SEEDS[name])
            out[name] = make_baseline(name, rng=rng, freqs=_FREQS)
        else:
            out[name] = make_baseline(name)
    out["heuristic"] = Heuristic(DEFAULT_CONFIG, id_bits=_CFG.sim.id_bits)
    for hc in all_configs():
        out[f"heuristic:{hc.id}"] = Heuristic(hc, id_bits=_CFG.sim.id_bits)
    return out


def actions_of(policy, ctxs) -> list[int]:
    return [int(policy.decide(c)) for c in ctxs]


WORLDS = {
    "all_online": set(NODES),
    "all_offline": set(),
    "inc_off_cand_on": set(NODES) - set(_INCUMBENT_SET),
    "inc_on_cand_off": set(NODES) - set(_CACHED),
}


def test_build_context_takes_no_hidden_state():
    params = list(inspect.signature(build_context).parameters)
    assert params == ["view", "b", "epoch_idx", "t_ms"]
    assert not {"net", "network", "env", "trace", "bg"} & set(params)


def test_decision_context_has_no_node_ids():
    names = {f.name for f in dataclasses.fields(DecisionContext)}
    assert names == EXPECTED_CONTEXT_FIELDS
    assert not any("node" in n or n.endswith("_id") for n in names)


def test_script_covers_interesting_contexts():
    ctxs = scripted_contexts(scripted_view())
    assert any(c.cand_since_seen_s is not None for c in ctxs)
    assert any(c.cand_since_seen_s is None for c in ctxs)
    assert any(c.inc_consec_fail >= 2 for c in ctxs)
    assert any(c.timeout_fraction > 0 for c in ctxs)
    view = scripted_view()
    assert {view.incumbent(b).node for b in range(8) if view.bucket(b)} == set(_INCUMBENT_SET)


def test_counterfactual_same_context_same_non_oracle_action():
    ctx_by_world, actions_by_world, oracle_by_world = {}, {}, {}
    for wname, online in WORLDS.items():
        view = scripted_view()
        net = make_world(online, keyed_root=len(online) + 1, pos_seed=len(wname))
        ctxs = scripted_contexts(view)
        ctx_by_world[wname] = ctxs
        actions_by_world[wname] = {n: actions_of(p, ctxs) for n, p in non_oracle_policies().items()}
        oracle_by_world[wname] = actions_of(oracle_for(view, net), ctxs)

    ref = next(iter(WORLDS))
    for wname in WORLDS:
        assert ctx_by_world[wname] == ctx_by_world[ref]
        assert actions_by_world[wname] == actions_by_world[ref], wname
    # The oracle does read ground truth, so the test can detect GT dependence.
    assert len({tuple(a) for a in oracle_by_world.values()}) > 1
    assert oracle_by_world["all_online"] != oracle_by_world["all_offline"]
    assert 2 in oracle_by_world["inc_off_cand_on"]  # REPLACE needs an online candidate


@given(online=st.sets(st.sampled_from(NODES)), keyed_root=st.integers(0, 2**64 - 1),
       pos_seed=st.integers(0, 2**32 - 1))
def test_counterfactual_property_random_hidden_worlds(online, keyed_root, pos_seed):
    """**Validates: Requirements 12.4**"""
    ref_ctxs = scripted_contexts(scripted_view())
    ref = {n: actions_of(p, ref_ctxs) for n, p in non_oracle_policies().items()}
    view = scripted_view()
    net = make_world(online, keyed_root, pos_seed)
    ctxs = scripted_contexts(view)
    assert ctxs == ref_ctxs
    # The oracle runs alongside on this world's ground truth; it must not perturb others.
    oracle = oracle_for(view, net)
    pols = non_oracle_policies()
    got: dict[str, list[int]] = {n: [] for n in pols}
    for c in ctxs:
        oracle.decide(c)
        for n, p in pols.items():
            got[n].append(int(p.decide(c)))
    assert got == ref


def test_end_to_end_recorded_contexts_replay_to_same_actions(tmp_path):
    """Contexts captured during a real episode replay to the same actions on fresh
    policy instances, without any env (policies are pure functions of ctx)."""
    exp = tmp_path / "short.toml"
    exp.write_text("[sim]\nwarmup_s = 60\nmeasured_s = 240\n", encoding="utf-8")
    cfg = load_config(CONFIGS / "base.toml", CONFIGS / "scenarios" / "high_churn.toml", exp)
    env = Env(cfg, make_streams(env_seed(cfg.root_seed, Split.CALIBRATION, 0, 0)))
    driver = make_baseline("uniform_random", rng=np.random.default_rng(99))
    observers = non_oracle_policies()
    ctxs: list[DecisionContext] = []
    live: dict[str, list[int]] = {n: [] for n in observers}

    def decide(ctx):
        ctxs.append(ctx)
        for n, p in observers.items():
            live[n].append(int(p.decide(ctx)))
        return driver.decide(ctx)

    env.run(decide, lambda d: None, lambda l: None)
    assert len(ctxs) > 50
    assert len({c.cand_since_seen_s is None for c in ctxs}) == 2  # both candidate cases

    for n, p in non_oracle_policies().items():
        if n in RANDOM_BASELINES:  # stateful stream: same seed, same order
            assert actions_of(p, ctxs) == live[n], n
        else:                      # pure: any order
            assert actions_of(p, ctxs[::-1]) == live[n][::-1], n
