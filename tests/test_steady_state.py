"""Unit tests for the pure aggregation core of analysis/steady_state.py.
The wandb API shell can only be validated live against a real run."""

import importlib.util
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
    table, scalars, n, span, mean_abs_res = steady_state.aggregate_window(
        _rows(), 50, 100
    )
    assert n == 50
    assert span == (1000.0 + 50 * 8.5, 1000.0 + 99 * 8.5)

    by_key = {key: (mean, std, share) for key, mean, std, share in table}
    # tokens_per_sec is a rate: not in the phase table, but in scalars.
    assert steady_state.TPS not in by_key
    assert scalars[steady_state.TPS] == pytest.approx(1200.0)

    mean, std, share = by_key["time/generate"]
    assert mean == pytest.approx(7.0)
    assert std == pytest.approx(0.0)
    assert share == pytest.approx(100.0 * 7.0 / 8.5)  # 82.35...%
    # wall_clock itself has no share and is ordered last.
    assert by_key["time/wall_clock"][2] is None
    assert table[-1][0] == "time/wall_clock"

    assert scalars["train/reward_mean"] == pytest.approx(0.8)
    assert mean_abs_res == pytest.approx(0.0001)


def test_half_open_window():
    # [start, end): step 100 excluded, step 50 included.
    _, _, n, _, _ = steady_state.aggregate_window(_rows(150), 50, 100)
    assert n == 50


def test_residual_caveat_threshold():
    # Above 0.05 -> caveat condition true; signed mean can't hide it
    # because the aggregate uses |residual|.
    rows = _rows(20, residual=0.1)
    rows[1]["check/timing_residual_frac"] = -0.1  # sign flip
    _, _, _, _, mean_abs = steady_state.aggregate_window(rows, 0, 20)
    assert mean_abs == pytest.approx(0.1)
    assert mean_abs > 0.05

    _, _, _, _, mean_abs_ok = steady_state.aggregate_window(_rows(20), 0, 20)
    assert mean_abs_ok < 0.05


def test_empty_window_and_missing_keys():
    _, _, n, span, mean_abs = steady_state.aggregate_window(_rows(), 500, 600)
    assert n == 0 and span is None and mean_abs is None
    # Rows lacking timing keys entirely don't crash the aggregation.
    table, scalars, n, _, _ = steady_state.aggregate_window(
        [{"_step": 0, "train/reward_mean": 1.0}], 0, 10
    )
    assert n == 1 and table == [] and scalars["train/reward_mean"] == 1.0
