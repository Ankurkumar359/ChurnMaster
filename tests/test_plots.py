"""Tests for plots.py (Requirement 26.3): figures render without error."""

from __future__ import annotations

import matplotlib
import numpy as np

from churnmaster import plots
from churnmaster.config import ScenarioConfig
from churnmaster.seeding import Split, env_seed, make_streams
from churnmaster.sim.churn import ChurnTrace, Session, draw_node_types, generate_trace

HIGH_CHURN = ScenarioConfig(name="high_churn", stable_share=0.30, stable_session_mean_s=3600.0,
                            flaky_session_mean_s=300.0, downtime_mean_s=600.0)
STABLE = ScenarioConfig(name="stable", stable_share=0.70, stable_session_mean_s=7200.0,
                        flaky_session_mean_s=900.0, downtime_mean_s=600.0)
N, FOCAL, HORIZON_MS = 64, 0, 3000 * 1000

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _trace(scn: ScenarioConfig, seed: int = 3) -> ChurnTrace:
    streams = make_streams(env_seed(12345, Split.CALIBRATION, 0, seed))
    types = draw_node_types(streams.types_positions, N, scn.stable_share, focal=FOCAL)
    return generate_trace(scn, types, streams.churn, HORIZON_MS, FOCAL)


def _is_png(path) -> bool:
    return path.is_file() and path.stat().st_size > 0 and path.read_bytes()[:8] == PNG_MAGIC


def test_online_count_single_trace_renders(tmp_path):
    out = plots.plot_online_count(_trace(HIGH_CHURN), tmp_path / "online.png", focal=FOCAL)
    assert out == tmp_path / "online.png"
    assert _is_png(out)


def test_online_count_multiple_scenarios_renders(tmp_path):
    traces = [("high_churn", _trace(HIGH_CHURN)), ("stable", _trace(STABLE))]
    out = plots.plot_online_count(traces, tmp_path / "sub" / "online_multi.png",
                                  step_ms=60_000, focal=FOCAL, title="Scenarios")
    assert _is_png(out)


def test_session_histogram_renders_log_and_linear(tmp_path):
    tr = _trace(HIGH_CHURN)
    assert _is_png(plots.plot_session_histogram(tr, tmp_path / "hist_log.png", focal=FOCAL))
    assert _is_png(plots.plot_session_histogram(tr, tmp_path / "hist_lin.png",
                                                focal=FOCAL, log=False))


def test_session_histogram_with_only_focal_renders(tmp_path):
    tr = ChurnTrace(1, 1000, [[Session(0, 0, 1000)]])
    assert _is_png(plots.plot_session_histogram(tr, tmp_path / "empty.png", focal=0))


def test_online_count_series_matches_online_at():
    tr = _trace(HIGH_CHURN)
    times, counts = plots.online_count_series(tr, step_ms=30_000, focal=FOCAL)
    assert times[0] == 0 and times[-1] <= HORIZON_MS
    for t, c in zip(times, counts):
        if t < HORIZON_MS:
            assert c == len(tr.online_at(int(t)) - {FOCAL})


def test_session_lengths_exclude_focal():
    tr = _trace(HIGH_CHURN)
    lengths = plots.session_lengths_s(tr, focal=FOCAL)
    expected = [(s.end_ms - s.start_ms) / 1000 for s in tr.all_sessions() if s.node != FOCAL]
    assert len(lengths) == len(tr.all_sessions()) - 1
    assert np.allclose(lengths, expected)


def test_rendering_does_not_change_global_backend(tmp_path):
    before = matplotlib.get_backend()
    plots.plot_online_count(_trace(STABLE), tmp_path / "o.png")
    assert matplotlib.get_backend() == before
    assert "pyplot" not in vars(plots)
