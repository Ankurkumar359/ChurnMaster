"""Layered, validated, hashable configuration (Requirement 1).

TOML files are parsed with ``tomllib`` and deep-merged in the order
base -> scenario -> experiment (later layers win). The merged dict is turned
into frozen dataclasses by a small generic builder that rejects unknown keys
and wrong types; ``validate`` then checks value ranges. Every error is a
``ConfigError`` naming the dotted key (and, from ``load_config``, the file
that set it).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import tomllib
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal


class ConfigError(ValueError):
    """Invalid configuration. ``key`` is the dotted path of the offending key."""

    def __init__(self, key: str, message: str, file: str | None = None):
        self.key = key
        self.file = file
        self.reason = message
        where = f" (in {file})" if file else ""
        super().__init__(f"{key}: {message}{where}")


# --------------------------------------------------------------------------- #
# Range-check helpers (stored in dataclass field metadata)
# --------------------------------------------------------------------------- #

def _range(lo: float | None = None, hi: float | None = None, *,
           lo_open: bool = False, hi_open: bool = False) -> dict[str, Callable[[Any], str | None]]:
    def check(v: Any) -> str | None:
        if isinstance(v, float) and not math.isfinite(v):
            return "must be finite"
        if lo is not None and (v <= lo if lo_open else v < lo):
            return f"must be {'>' if lo_open else '>='} {lo}"
        if hi is not None and (v >= hi if hi_open else v > hi):
            return f"must be {'<' if hi_open else '<='} {hi}"
        return None
    return {"check": check}


def _nonempty() -> dict[str, Callable[[Any], str | None]]:
    return {"check": lambda v: "must not be empty" if len(v) == 0 else None}


POS = _range(0, lo_open=True)        # > 0
NONNEG = _range(0)                   # >= 0
UNIT = _range(0, 1)                  # [0, 1]


def _f(default: Any = dataclasses.MISSING, meta: dict | None = None) -> Any:
    return field(default=default, metadata=meta or {})


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SimConfig:
    n_nodes: int = _f(128, _range(2))
    id_bits: int = _f(8, _range(1, 16))
    k: int = _f(4, _range(1))
    alpha: int = _f(2, _range(1))
    replicas: int = _f(3, _range(1))
    cache_size: int = _f(4, _range(1))
    cache_ttl_s: int = _f(300, POS)
    rebuild_period_s: int = _f(300, POS)
    epoch_s: int = _f(30, POS)
    warmup_s: int = _f(600, NONNEG)
    measured_s: int = _f(2400, POS)
    lookup_rate_hz: float = _f(1.0, POS)
    p_loss: float = _f(0.02, _range(0, 1, hi_open=True))
    timeout_ms: int = _f(500, POS)
    deadline_ms: int = _f(2000, POS)
    max_rounds: int = _f(8, _range(1))
    rtt_base_ms: float = _f(10.0, NONNEG)
    rtt_span_ms: float = _f(240.0, NONNEG)
    rtt_sigma: float = _f(0.25, NONNEG)
    congestion_share: float = _f(0.10, UNIT)
    congestion_len_s: int = _f(120, POS)
    congestion_gap_s: int = _f(1200, POS)
    congestion_mult: float = _f(3.0, _range(1))
    window_timeout_s: int = _f(120, POS)
    rtt_beta: float = _f(0.3, _range(0, 1, lo_open=True))


@dataclass(frozen=True)
class BurstConfig:
    gap_mean_s: float = _f(900.0, POS)
    fraction: float = _f(0.20, UNIT)
    leave_window_s: int = _f(10, NONNEG)
    return_mean_s: float = _f(300.0, POS)
    rtt_mult: float = _f(2.0, _range(1))
    rtt_duration_s: int = _f(180, NONNEG)


@dataclass(frozen=True)
class ScenarioConfig:
    name: str = _f(meta=_nonempty())
    stable_share: float = _f(meta=UNIT)
    stable_session_mean_s: float = _f(meta=POS)
    flaky_session_mean_s: float = _f(meta=POS)
    downtime_mean_s: float = _f(meta=POS)
    weibull_shape: float = _f(0.6, POS)
    burn_in_s: int = _f(7200, NONNEG)
    bursts: BurstConfig | None = _f(None)


@dataclass(frozen=True)
class RewardConfig:
    w_L: float = _f(0.3, NONNEG)
    w_M: float = _f(0.2, NONNEG)
    w_R: float = _f(0.3, NONNEG)


@dataclass(frozen=True)
class AgentConfig:
    gamma: float = _f(0.9, UNIT)
    alpha_min: float = _f(0.05, _range(0, 1, lo_open=True))
    alpha_exp: float = _f(0.7, POS)
    eps_start: float = _f(1.0, UNIT)
    eps_end: float = _f(0.05, UNIT)
    eps_decay_frac: float = _f(0.6, _range(0, 1, lo_open=True))
    episodes: int = _f(200, _range(1))
    runs: int = _f(3, _range(1))
    validate_every: int = _f(20, _range(1))


# ---- experiment sub-tables ------------------------------------------------- #

@dataclass(frozen=True)
class GatesConfig:
    """Headroom gates on J / p95 latency (revised after checkpoint 5.17)."""
    g1_min_dJ: float = _f(0.01, NONNEG)          # G1 route A: mean ΔJ vs never_replace
    g1_min_p95_rel: float = _f(0.20, UNIT)       # G1 route B: mean relative p95_all reduction
    g2_min_dJ: float = _f(0.01, NONNEG)          # G2 route A: oracle − kademlia_epoch ΔJ
    g2_max_ping_ratio: float = _f(0.6, NONNEG)   # G2 route B: pings(oracle)/pings(kademlia)
    g2_J_tol: float = _f(0.005, NONNEG)          # G2 route B: mean ΔJ >= −tol


@dataclass(frozen=True)
class TuningConfig:
    deadline_ms: tuple[int, ...] = _f((1000, 1500, 2000), _nonempty())
    p_loss: tuple[float, ...] = _f((0.01, 0.02, 0.05), _nonempty())
    replicas: tuple[int, ...] = _f((1, 2, 3), _nonempty())


@dataclass(frozen=True)
class DebugConfig:
    scenario: str = _f("high_churn", _nonempty())
    runs: int = _f(1, _range(1))
    episodes: int = _f(100, _range(1))


@dataclass(frozen=True)
class GridConfig:
    gamma: tuple[float, ...] = _f((0.8, 0.9), _nonempty())
    alpha_min: tuple[float, ...] = _f((0.02, 0.05), _nonempty())


@dataclass(frozen=True)
class StatsConfig:
    bootstrap_resamples: int = _f(10000, _range(1))
    ci_level: float = _f(0.95, _range(0, 1, lo_open=True, hi_open=True))


SplitName = Literal["calibration", "train", "validation", "eval", "agent", "test"]


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = _f("default", _nonempty())
    split: SplitName = _f("test")
    n_seeds: int | None = _f(None, _range(1))
    scenarios: tuple[str, ...] = _f(())
    policies: tuple[str, ...] = _f(())
    validation_seeds: int | None = _f(None, _range(1))
    last_checkpoints: int | None = _f(None, _range(1))
    gates: GatesConfig | None = _f(None)
    tuning: TuningConfig | None = _f(None)
    debug: DebugConfig | None = _f(None)
    grid: GridConfig | None = _f(None)
    stats: StatsConfig | None = _f(None)


@dataclass(frozen=True)
class Config:
    sim: SimConfig = _f()
    scenario: ScenarioConfig = _f()
    reward: RewardConfig = _f()
    agent: AgentConfig = _f()
    experiment: ExperimentConfig = _f()
    root_seed: int = _f(meta=_range(0, 2**64 - 1))


# Sections that may be omitted from the merged dict and are then defaulted.
_DEFAULTABLE = {"sim", "reward", "agent", "experiment"}


# --------------------------------------------------------------------------- #
# Merge
# --------------------------------------------------------------------------- #

def merge(base: dict, *overlays: dict) -> dict:
    """Deep-merge dicts; for every key the last layer defining it wins.

    Nested dicts are merged recursively; any other value (including lists)
    replaces the earlier value. Inputs are not mutated.
    """
    out = _deepcopy(base)
    for ov in overlays:
        _merge_into(out, ov)
    return out


def _deepcopy(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: _deepcopy(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_deepcopy(v) for v in d]
    return d


def _merge_into(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge_into(dst[k], v)
        else:
            dst[k] = _deepcopy(v)


# --------------------------------------------------------------------------- #
# Dict -> dataclass builder (unknown keys, types)
# --------------------------------------------------------------------------- #

def _type_name(tp: Any) -> str:
    return getattr(tp, "__name__", str(tp))


def _coerce(tp: Any, value: Any, key: str) -> Any:
    origin = typing.get_origin(tp)

    # Optional / unions
    if origin in (typing.Union, types.UnionType):
        args = typing.get_args(tp)
        if value is None and type(None) in args:
            return None
        errors = []
        for a in args:
            if a is type(None):
                continue
            try:
                return _coerce(a, value, key)
            except ConfigError as e:
                errors.append(e)
        # Report the most specific (deepest) error
        raise max(errors, key=lambda e: len(e.key))

    if origin is Literal:
        allowed = typing.get_args(tp)
        if value not in allowed or not isinstance(value, str):
            raise ConfigError(key, f"must be one of {list(allowed)}, got {value!r}")
        return value

    if origin is tuple:
        (elem, _ell) = typing.get_args(tp)
        if not isinstance(value, (list, tuple)):
            raise ConfigError(key, f"expected list, got {type(value).__name__}")
        return tuple(_coerce(elem, v, f"{key}[{i}]") for i, v in enumerate(value))

    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise ConfigError(key, f"expected table, got {type(value).__name__}")
        return _build(tp, value, key)

    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(key, f"expected bool, got {type(value).__name__}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(key, f"expected int, got {type(value).__name__}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(key, f"expected float, got {type(value).__name__}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(key, f"expected str, got {type(value).__name__}")
        return value
    raise TypeError(f"unsupported schema type {_type_name(tp)} at {key}")  # pragma: no cover


def _build(cls: type, data: dict, prefix: str = "") -> Any:
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    for k in data:
        if k not in fields:
            raise ConfigError(_join(prefix, k), "unknown key")
    kwargs = {}
    for name, f in fields.items():
        key = _join(prefix, name)
        if name in data:
            kwargs[name] = _coerce(hints[name], data[name], key)
        elif f.default is dataclasses.MISSING:
            if cls is Config and name in _DEFAULTABLE:
                kwargs[name] = _build(hints[name], {}, key)
            else:
                raise ConfigError(key, "missing required key")
    return cls(**kwargs)


def _join(prefix: str, k: str) -> str:
    return f"{prefix}.{k}" if prefix else k


def from_dict(data: dict) -> Config:
    """Build and validate a Config from a merged dict."""
    if not isinstance(data, dict):
        raise ConfigError("<root>", "expected table")
    cfg = _build(Config, data)
    validate(cfg)
    return cfg


# --------------------------------------------------------------------------- #
# Validation (ranges, cross-field)
# --------------------------------------------------------------------------- #

def _check_ranges(obj: Any, prefix: str) -> None:
    for f in dataclasses.fields(obj):
        key = _join(prefix, f.name)
        v = getattr(obj, f.name)
        if v is None:
            continue
        if dataclasses.is_dataclass(v):
            _check_ranges(v, key)
            continue
        check = f.metadata.get("check")
        if check is None:
            continue
        msg = check(v)
        if msg:
            raise ConfigError(key, f"{msg}, got {v!r}")


def validate(cfg: Config) -> None:
    """Range and cross-field checks; raises ConfigError naming the key."""
    _check_ranges(cfg, "")
    s = cfg.sim
    if s.n_nodes > 2 ** s.id_bits:
        raise ConfigError("sim.n_nodes", f"must be <= 2**id_bits = {2 ** s.id_bits}, got {s.n_nodes}")
    if s.timeout_ms > s.deadline_ms:
        raise ConfigError("sim.timeout_ms", "must be <= sim.deadline_ms")
    if s.replicas >= s.n_nodes:
        raise ConfigError("sim.replicas", "must be < sim.n_nodes")
    a = cfg.agent
    if a.eps_end > a.eps_start:
        raise ConfigError("agent.eps_end", "must be <= agent.eps_start")
    t = cfg.experiment.tuning
    if t is not None:
        for i, p in enumerate(t.p_loss):
            if not 0 <= p < 1:
                raise ConfigError(f"experiment.tuning.p_loss[{i}]", f"must be in [0, 1), got {p}")
        for name in ("deadline_ms", "replicas"):
            for i, v in enumerate(getattr(t, name)):
                if v < 1:
                    raise ConfigError(f"experiment.tuning.{name}[{i}]", f"must be >= 1, got {v}")
    g = cfg.experiment.grid
    if g is not None:
        for i, v in enumerate(g.gamma):
            if not 0 <= v <= 1:
                raise ConfigError(f"experiment.grid.gamma[{i}]", f"must be in [0, 1], got {v}")
        for i, v in enumerate(g.alpha_min):
            if not 0 < v <= 1:
                raise ConfigError(f"experiment.grid.alpha_min[{i}]", f"must be in (0, 1], got {v}")


# --------------------------------------------------------------------------- #
# Hash, loading
# --------------------------------------------------------------------------- #

def to_dict(cfg: Config) -> dict:
    """Plain JSON-compatible dict (tuples become lists)."""
    return json.loads(canonical_json(cfg))


def canonical_json(cfg: Config) -> str:
    return json.dumps(dataclasses.asdict(cfg), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def config_hash(cfg: Config) -> str:
    """SHA-256 hex digest of the canonical JSON serialization."""
    return hashlib.sha256(canonical_json(cfg).encode("utf-8")).hexdigest()


def read_toml(path: Path) -> dict:
    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError("<toml>", f"parse error: {e}", file=str(path)) from e
    except OSError as e:
        raise ConfigError("<file>", f"cannot read: {e}", file=str(path)) from e


def _provenance(d: dict, file: str, out: dict[str, str], prefix: str = "") -> None:
    for k, v in d.items():
        key = _join(prefix, k)
        out[key] = file
        if isinstance(v, dict):
            _provenance(v, file, out, key)


def _file_for(key: str, prov: dict[str, str]) -> str | None:
    k = key
    while k:
        base = k.split("[", 1)[0]
        if base in prov:
            return prov[base]
        if "." not in base:
            return None
        k = base.rsplit(".", 1)[0]
    return None


def load_config(base: Path, scenario: Path, experiment: Path | None = None) -> Config:
    """Load base -> scenario -> experiment TOML files, merge, build, validate."""
    paths = [Path(base), Path(scenario)] + ([Path(experiment)] if experiment is not None else [])
    layers, prov = [], {}
    for p in paths:
        d = read_toml(p)
        layers.append(d)
        _provenance(d, str(p), prov)
    merged = merge(*layers)
    try:
        return from_dict(merged)
    except ConfigError as e:
        if e.file is None:
            raise ConfigError(e.key, e.reason, file=_file_for(e.key, prov)) from e
        raise


__all__ = [
    "AgentConfig", "BurstConfig", "Config", "ConfigError", "DebugConfig", "ExperimentConfig",
    "GatesConfig", "GridConfig", "RewardConfig", "ScenarioConfig", "SimConfig", "StatsConfig",
    "TuningConfig", "canonical_json", "config_hash", "from_dict", "load_config", "merge",
    "read_toml", "to_dict", "validate",
]
