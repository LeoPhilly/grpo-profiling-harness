"""CPU-backend validation of the timing harness. Per CLAUDE.md these sleeps
validate the ACCOUNTING (phase boundaries, residual, key mapping) — CUDA
event timing itself is validated on the GPU box against a known workload."""

import time

import pytest

from grpo.instrumentation.timing import (
    RESIDUAL_FRAC,
    TIMING_RESIDUAL_KEY,
    WALL_CLOCK,
    WALL_CLOCK_KEY,
    PhaseTimer,
    to_wandb_metrics,
)


def test_known_workload_200ms():
    timer = PhaseTimer(use_cuda=False)
    timer.start_step()
    with timer.phase("work"):
        time.sleep(0.2)
    summary = timer.step_summary()
    assert 0.19 <= summary["work"] <= 0.25


def test_residual_small_when_everything_is_timed():
    timer = PhaseTimer(use_cuda=False)
    timer.start_step()
    for name in ("a", "b", "c"):
        with timer.phase(name):
            time.sleep(0.05)
    summary = timer.step_summary()
    assert abs(summary[RESIDUAL_FRAC]) < 0.05
    assert summary[WALL_CLOCK] >= 0.15


def test_residual_detects_injected_untimed_gap():
    # Inject the failure the check exists to detect: 50ms of untimed work.
    timer = PhaseTimer(use_cuda=False)
    timer.start_step()
    with timer.phase("a"):
        time.sleep(0.1)
    time.sleep(0.05)  # deliberately untimed
    with timer.phase("b"):
        time.sleep(0.1)
    summary = timer.step_summary()
    # gap / wall ~= 0.05 / 0.25 = 0.2
    assert 0.12 < summary[RESIDUAL_FRAC] < 0.30


def test_repeated_phase_name_accumulates():
    timer = PhaseTimer(use_cuda=False)
    timer.start_step()
    for _ in range(2):
        with timer.phase("x"):
            time.sleep(0.05)
    summary = timer.step_summary()
    assert 0.09 <= summary["x"] <= 0.14


def test_wall_clock_is_independent_of_phases():
    # An untimed step still has a wall clock; phases sum to 0 -> residual 1.
    timer = PhaseTimer(use_cuda=False)
    timer.start_step()
    time.sleep(0.05)
    summary = timer.step_summary()
    assert summary[WALL_CLOCK] >= 0.05
    assert summary[RESIDUAL_FRAC] == 1.0


def test_wandb_key_mapping_is_pinned():
    summary = {"generate": 1.0, WALL_CLOCK: 2.0, RESIDUAL_FRAC: 0.5}
    assert to_wandb_metrics(summary) == {
        "time/generate": 1.0,
        WALL_CLOCK_KEY: 2.0,
        TIMING_RESIDUAL_KEY: 0.5,
    }
    # The canonical key strings themselves, pinned literally.
    assert WALL_CLOCK_KEY == "time/wall_clock"
    assert TIMING_RESIDUAL_KEY == "check/timing_residual_frac"


def test_misuse_raises():
    timer = PhaseTimer(use_cuda=False)
    with pytest.raises(RuntimeError):
        timer.step_summary()
    timer.start_step()
    with pytest.raises(ValueError):
        with timer.phase(WALL_CLOCK):
            pass
