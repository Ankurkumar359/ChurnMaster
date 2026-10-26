"""CLI tests (Requirements 1.7, 26.1)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from churnmaster.cli import COMMANDS, build_parser, main
from churnmaster.config import config_hash, load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _split_output(out: str) -> tuple[dict, str]:
    body, _, tail = out.rpartition("config_hash: ")
    return json.loads(body), tail.strip()


def test_show_config_prints_merged_config_and_hash(capsys):
    rc = main(["show-config", "--scenario", "high_churn", "--configs-dir", str(CONFIGS)])
    assert rc == 0
    cfg_dict, printed_hash = _split_output(capsys.readouterr().out)
    expected = load_config(CONFIGS / "base.toml", CONFIGS / "scenarios" / "high_churn.toml")
    assert re.fullmatch(r"[0-9a-f]{64}", printed_hash)
    assert printed_hash == config_hash(expected)
    assert cfg_dict["scenario"]["name"] == "high_churn"


def test_show_config_with_experiment(capsys):
    rc = main(["show-config", "--scenario", "stable", "--experiment", "eval",
               "--configs-dir", str(CONFIGS)])
    assert rc == 0
    _, printed_hash = _split_output(capsys.readouterr().out)
    expected = load_config(CONFIGS / "base.toml", CONFIGS / "scenarios" / "stable.toml",
                           CONFIGS / "experiments" / "eval.toml")
    assert printed_hash == config_hash(expected)


def test_show_config_unknown_scenario_is_clean_error(capsys):
    rc = main(["show-config", "--scenario", "no_such_scenario", "--configs-dir", str(CONFIGS)])
    assert rc != 0
    err = capsys.readouterr().err
    assert "config error" in err and "no_such_scenario" in err


def test_all_ten_commands_registered():
    parser = build_parser()
    for name in COMMANDS:
        argv = [name, "--figures-only"] if name == "reproduce" else [name]
        assert parser.parse_args(argv).command == name
    assert len(COMMANDS) == 10


def test_reproduce_requires_exactly_one_mode():
    parser = build_parser()
    for flag in ("--quick", "--full", "--figures-only"):
        assert parser.parse_args(["reproduce", flag]).command == "reproduce"
    for argv in (["reproduce"], ["reproduce", "--quick", "--full"],
                 ["reproduce", "--full", "--figures-only"]):
        with pytest.raises(SystemExit):
            parser.parse_args(argv)


@pytest.mark.parametrize("flag", ["--quick", "--full"])
def test_reproduce_pipeline_args(flag):
    a = build_parser().parse_args(["reproduce", flag, "--out", "x", "--workers", "3"])
    assert (a.quick, a.full) == (flag == "--quick", flag == "--full")
    assert a.out == "x" and a.workers == 3
    assert build_parser().parse_args(["reproduce", flag]).workers is None  # default_workers()


@pytest.mark.parametrize("flag", ["--quick", "--full"])
@pytest.mark.parametrize("out", ["results/headroom", "results/eval/sub", "results", "."])
def test_reproduce_pipelines_refuse_canonical_out(flag, out, capsys):
    assert main(["reproduce", flag, "--out", out]) == 1
    err = capsys.readouterr().err
    assert "canonical" in err or "results root" in err


def test_simulate_prints_metrics_and_hashes(capsys):
    rc = main(["simulate", "--scenario", "high_churn", "--configs-dir", str(CONFIGS)])
    assert rc == 0
    out = capsys.readouterr().out
    for key in ("success_rate", "config_hash", "trace_hash", "workload_hash", "summary_hash", "wall_s"):
        assert key in out
    assert re.search(r"summary_hash\s+[0-9a-f]{64}", out)


def test_simulate_json_is_deterministic(capsys):
    argv = ["simulate", "--scenario", "stable", "--episode", "1", "--json",
            "--configs-dir", str(CONFIGS)]
    assert main(argv) == 0
    first = json.loads(capsys.readouterr().out)
    assert main(argv) == 0
    second = json.loads(capsys.readouterr().out)
    assert first["summary_hash"] == second["summary_hash"]
    assert first["episode"] == 1 and first["maintenance"] == "off"
    assert first["n_lookups"] > 0


def test_simulate_rejects_agent_split(capsys):
    rc = main(["simulate", "--seed-split", "agent", "--configs-dir", str(CONFIGS)])
    assert rc != 0
    assert "agent" in capsys.readouterr().err


def test_python_dash_m_entry_point():
    proc = subprocess.run([sys.executable, "-m", "churnmaster", "show-config",
                           "--configs-dir", str(CONFIGS)],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert "config_hash: " in proc.stdout


SHORT_SIM = "[sim]\nwarmup_s = 60\nmeasured_s = 240\n"


def test_compare_short_writes_outputs(tmp_path, capsys):
    exp = tmp_path / "short.toml"
    exp.write_text(SHORT_SIM, encoding="utf-8")
    out = tmp_path / "out"
    rc = main(["compare", "--scenario", "high_churn", "--policies", "never_replace,kademlia_std",
               "--seeds", "2", "--experiment-file", str(exp), "--configs-dir", str(CONFIGS),
               "--out", str(out)])
    assert rc == 0, capsys.readouterr().err
    text = capsys.readouterr().out
    assert "never_replace" in text and "kademlia_std" in text and "±" in text
    import pandas as pd
    per_seed = pd.read_csv(out / "per_seed.csv")
    assert len(per_seed) == 4 and set(per_seed["episode"]) == {0, 1}
    summary = pd.read_csv(out / "summary.csv")
    assert list(summary["policy"]) == ["never_replace", "kademlia_std"]
    for m in ("success_rate", "p95_latency_all_ms", "pings_per_hour"):
        assert (out / f"{m}.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_compare_unknown_policy_is_clean_error(capsys):
    rc = main(["compare", "--policies", "bogus", "--configs-dir", str(CONFIGS)])
    assert rc != 0
    assert "bogus" in capsys.readouterr().err
