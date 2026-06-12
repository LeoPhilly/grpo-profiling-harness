"""Unit tests for the pure aggregation core of analysis/steady_state.py.
The wandb API shell can only be validated live against a real run."""

import importlib.util
import statistics
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parent.parent / "analysis" / "steady_state.py"
_spec = importlib.util.spec_from_file_location("steady_state", _path)
steady_state = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(steady_state)


def _rows(n=100, residual=0.0001):
    # Hand-computable synthetic history: generate 7.0s of an 8.5s wall.
    return [
        {
            "_step": i,
            "_timestamp": 1000.0 + i * 8.5,
            "time/generate": 7.0,
            "time/forward_loss": 1.5,
            "time/wall_clock": 8.5,
            "time/tokens_per_sec_generate": 1200.0,
            "train/reward_mean": 0.8,
            "train/format_rate": 0.95,
            "check/timing_residual_frac": residual,
            "train/notes": "non-numeric, must be ignored",
        }
        for i in range(n)
    ]


def test_window_and_shares():
    table, scalars, n, span, mean_abs_res, var_decomp = (
        steady_state.aggregate_window(_rows(), 50, 100)
    )
    assert n == 50
    assert span == (1000.0 + 50 * 8.5, 1000.0 + 99 * 8.5)

    by_key = {row[0]: row[1:] for row in table}
    # tokens_per_sec is a rate: not in the phase table, but in scalars.
    assert steady_state.TPS not in by_key
    assert scalars[steady_state.TPS] == pytest.approx(1200.0)

    mean, std, p10, p90, share = by_key["time/generate"]
    assert mean == pytest.approx(7.0)
    assert std == pytest.approx(0.0)
    assert p10 == pytest.approx(7.0) and p90 == pytest.approx(7.0)
    assert share == pytest.approx(100.0 * 7.0 / 8.5)  # 82.35...%
    # wall_clock itself has no share and is ordered last.
    assert by_key["time/wall_clock"][4] is None
    assert table[-1][0] == "time/wall_clock"

    assert scalars["train/reward_mean"] == pytest.approx(0.8)
    assert mean_abs_res == pytest.approx(0.0001)
    # Constant wall -> var(wall) ~ 0 -> decomposition skipped, not a crash.
    assert var_decomp is None


def _varying_rows(generate, other):
    return [
        {
            "_step": i,
            "time/generate": g,
            "time/other": o,
            "time/wall_clock": g + o,
        }
        for i, (g, o) in enumerate(zip(generate, other))
    ]


def test_variance_decomposition_single_source():
    # Only generate varies: it owns 100% of wall variance, remainder 0.
    rows = _varying_rows([6.0, 8.0, 6.0, 8.0], [1.0, 1.0, 1.0, 1.0])
    *_, var_decomp = steady_state.aggregate_window(rows, 0, 4)
    decomp = dict(var_decomp)
    assert decomp["time/generate"] == pytest.approx(100.0)
    assert decomp["time/other"] == pytest.approx(0.0)
    assert decomp["covariance remainder"] == pytest.approx(0.0)


def test_variance_decomposition_covariance_remainder():
    # Two perfectly correlated phases: var(a)=var(b)=0.5, wall=[2,4] var=2.
    # Each phase shows 25%; the 2*cov term (50%) lands in the remainder —
    # exactly the case where per-phase variances alone would mislead.
    rows = _varying_rows([1.0, 2.0], [1.0, 2.0])
    *_, var_decomp = steady_state.aggregate_window(rows, 0, 2)
    decomp = dict(var_decomp)
    assert decomp["time/generate"] == pytest.approx(25.0)
    assert decomp["time/other"] == pytest.approx(25.0)
    assert decomp["covariance remainder"] == pytest.approx(50.0)
    assert sum(pct for _, pct in var_decomp) == pytest.approx(100.0)


def test_derived_forward_loss_row():
    # New-style rows: parts logged, no aggregate -> derived per-step series.
    rows = [
        {
            "_step": i,
            "time/forward": f,
            "time/loss_compute": 0.1,
            "time/backward": b,
            "time/wall_clock": f + 0.1 + b + 1.0,
        }
        for i, (f, b) in enumerate([(0.5, 1.0), (0.7, 1.4)])
    ]
    table, *_, var_decomp = steady_state.aggregate_window(rows, 0, 2)
    by_key = {row[0]: row[1:] for row in table}
    label = steady_state.FORWARD_LOSS_DERIVED
    mean, std, p10, p90, share = by_key[label]
    # Per-step sums: 1.6 and 2.2 -> mean 1.9; std/percentiles from the
    # summed series, NOT sums of per-part stds/percentiles.
    assert mean == pytest.approx(1.9)
    assert std == pytest.approx(statistics.stdev([1.6, 2.2]))
    # Floor-index percentile convention (see test_percentiles): with n=2,
    # both p10 and p90 land on index int(q*1) = 0.
    assert (p10, p90) == (1.6, 1.6)
    assert share == pytest.approx(100.0 * 1.9 / statistics.fmean([2.6, 3.2]))
    # Ordered before wall_clock; excluded from the variance decomposition.
    assert table[-1][0] == "time/wall_clock"
    assert table[-2][0] == label
    assert label not in dict(var_decomp)


def test_no_derived_row_when_aggregate_logged():
    # r0-style rows already log time/forward_loss: no derived duplicate.
    table, *_ = steady_state.aggregate_window(_rows(20), 0, 20)
    keys = [row[0] for row in table]
    assert steady_state.FORWARD_LOSS in keys
    assert steady_state.FORWARD_LOSS_DERIVED not in keys


def test_percentiles():
    vals = sorted(float(v) for v in range(1, 11))  # 1..10
    assert steady_state.percentile(vals, 0.1) == 1.0  # int(0.1*9)=0
    assert steady_state.percentile(vals, 0.9) == 9.0  # int(0.9*9)=8
    rows = _varying_rows([float(v) for v in range(1, 11)], [0.0] * 10)
    table, *_ = steady_state.aggregate_window(rows, 0, 10)
    by_key = {row[0]: row[1:] for row in table}
    _, _, p10, p90, _ = by_key["time/generate"]
    assert (p10, p90) == (1.0, 9.0)


def test_half_open_window():
    # [start, end): step 100 excluded, step 50 included.
    _, _, n, _, _, _ = steady_state.aggregate_window(_rows(150), 50, 100)
    assert n == 50


def test_residual_caveat_threshold():
    # Above 0.05 -> caveat condition true; signed mean can't hide it
    # because the aggregate uses |residual|.
    rows = _rows(20, residual=0.1)
    rows[1]["check/timing_residual_frac"] = -0.1  # sign flip
    _, _, _, _, mean_abs, _ = steady_state.aggregate_window(rows, 0, 20)
    assert mean_abs == pytest.approx(0.1)
    assert mean_abs > 0.05

    _, _, _, _, mean_abs_ok, _ = steady_state.aggregate_window(_rows(20), 0, 20)
    assert mean_abs_ok < 0.05


def test_empty_window_and_missing_keys():
    _, _, n, span, mean_abs, var_decomp = steady_state.aggregate_window(
        _rows(), 500, 600
    )
    assert n == 0 and span is None and mean_abs is None and var_decomp is None
    # Rows lacking timing keys entirely don't crash the aggregation.
    table, scalars, n, _, _, _ = steady_state.aggregate_window(
        [{"_step": 0, "train/reward_mean": 1.0}], 0, 10
    )
    assert n == 1 and table == [] and scalars["train/reward_mean"] == 1.0
