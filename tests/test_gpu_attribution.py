"""Calibrated per-phase attribution validation. GPU-only (pytest -m gpu),
runs at smoke time on the box, skipped on the Mac.

Workload is torch.cuda._sleep: a busy-wait with no memory traffic, so the
injection has no cache/allocator side effects on neighboring phases.

Calibration chain (each step catches the previous step's failure mode):
probe -> fit (physics-anchored against the nominal clock) -> scale ->
measured ground truth W with a sanity range. W, not the 200ms target, is
the reference for every assertion; baselines are measured, never assumed.
"""

import statistics
import time

import pytest
import torch

from grpo.instrumentation.timing import RESIDUAL_FRAC, PhaseTimer

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="GPU-only: calibrated attribution needs CUDA",
    ),
]

CUDA_SLEEP = getattr(torch.cuda, "_sleep", None)

TARGET_MS = 200  # only a steering wheel; measured W is the reference
PHASES = ("p0", "p1", "p2", "p3")


def _time_sleep_standalone(cycles):
    """Event-pair-timed _sleep with clean sync boundaries. Milliseconds."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    CUDA_SLEEP(cycles)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


@pytest.fixture(scope="module")
def calibration():
    """Steps 1-4: probe, fit, scale, ground truth."""
    if CUDA_SLEEP is None:
        pytest.skip("torch.cuda._sleep unavailable in this torch build")

    n_probe = 10_000_000
    probe_ms = _time_sleep_standalone(n_probe)
    cycles_per_ms = n_probe / probe_ms

    # Physics anchor: clock_rate() returns the SM clock in MHz — observed on
    # the A100 (2026-06-12): empirical fit 1,047,589 cycles/ms (~1.05 GHz),
    # clock_rate() = 1095, ratio 0.957 after the *1000 conversion. The
    # original kHz assumption tripped this assert at ratio ~957, exactly the
    # ~1e3 unit-botch failure mode it exists to catch.
    nominal_cycles_per_ms = torch.cuda.clock_rate() * 1000  # MHz -> cycles/ms
    ratio = cycles_per_ms / nominal_cycles_per_ms
    assert 0.1 < ratio < 10, (
        f"calibration {cycles_per_ms:.0f} cyc/ms vs nominal "
        f"{nominal_cycles_per_ms} (ratio {ratio:.4f}) — unit botch?"
    )

    n_inject = int(cycles_per_ms * TARGET_MS)
    w_ms = statistics.median(_time_sleep_standalone(n_inject) for _ in range(3))
    # Range check that catches a botched fit before it can produce a
    # vacuously passing injection test.
    assert 50.0 < w_ms < 2000.0, f"ground-truth dose {w_ms:.1f}ms out of range"
    return {"n_inject": n_inject, "w_ms": w_ms, "cycles_per_ms": cycles_per_ms}


def _mini_step(work_cycles, inject=None):
    """Synthetic mini-step on the REAL substrate (PhaseTimer, CUDA backend):
    4 event-timed phases of ~10ms trivial GPU work each. inject maps
    phase name -> cycles enqueued at the top of that phase, NO sync inside
    the phase; everything harvests at step end."""
    timer = PhaseTimer()
    assert timer.use_cuda
    timer.start_step()
    for name in PHASES:
        with timer.phase(name):
            if inject and name in inject:
                CUDA_SLEEP(inject[name])  # enqueue only — no sync in-phase
            CUDA_SLEEP(work_cycles)
    return timer.step_summary()


@pytest.mark.parametrize("site", PHASES)
def test_injection_charged_to_site(calibration, site):
    """One site hot keeps this a localization test, not an arithmetic
    puzzle. Fixed dose at every site regardless of phase size — the
    question is 'is time issued here charged here', not proportional
    perturbation."""
    work = int(calibration["cycles_per_ms"] * 10)
    w_s = calibration["w_ms"] / 1000.0

    baseline = _mini_step(work)  # measured control, never assumed
    injected = _mini_step(work, inject={site: calibration["n_inject"]})

    for name in PHASES:
        if name == site:
            extra = injected[name] - baseline[name]
            assert abs(extra - w_s) < 0.2 * w_s, (
                f"{name}: charged {extra:.4f}s, expected W={w_s:.4f}s ±20%"
            )
        else:
            assert injected[name] < 3 * baseline[name], (
                f"{name} leaked: {injected[name]:.4f}s vs "
                f"baseline {baseline[name]:.4f}s"
            )
    assert abs(injected[RESIDUAL_FRAC]) < 0.05


def test_cpu_phase_work_lands_in_residual(calibration):
    """CONFIRMED SEMANTIC (A100, 2026-06-12): async GPU work issued during
    a perf_counter-timed CPU phase is charged to NOBODY — not the CPU phase
    (host clock only measures host time), not the next event-timed phase
    (its start event timestamps at GPU execution, after the in-flight work
    completes). It lands in the unattributed gap: the step RESIDUAL.

    The step residual check is therefore the safety net that catches
    CPU-issued GPU work — which is why a production residual of ~4e-5
    certifies that no such leakage is happening in the real loop.

    History: the original spec predicted drain-into-next-phase; the
    pre-registered alternative diagnosis was confirmed by measurement
    (next-phase delta -0.0000s, wall grew 0.1467s ~= W=0.1493s).

    PhaseTimer is single-backend, so the mixed step is hand-rolled here:
    event pairs for the flanking GPU phases, perf_counter for the middle
    CPU phase — exactly the mixed accounting the semantic describes."""
    work = int(calibration["cycles_per_ms"] * 10)
    w_s = calibration["w_ms"] / 1000.0

    def mixed_step(inject):
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        ev[0].record()
        CUDA_SLEEP(work)  # gpu_a
        ev[1].record()
        cpu_start = time.perf_counter()
        if inject:
            CUDA_SLEEP(calibration["n_inject"])  # async enqueue, no sync
        deadline = time.perf_counter() + 0.010  # ~10ms pure host work
        while time.perf_counter() < deadline:
            pass
        cpu_elapsed = time.perf_counter() - cpu_start
        ev[2].record()
        CUDA_SLEEP(work)  # gpu_b
        ev[3].record()
        torch.cuda.synchronize()
        wall = time.perf_counter() - wall_start
        return {
            "gpu_a": ev[0].elapsed_time(ev[1]) / 1000.0,
            "cpu": cpu_elapsed,
            "gpu_b": ev[2].elapsed_time(ev[3]) / 1000.0,
            "wall": wall,
        }

    base = mixed_step(inject=False)
    inj = mixed_step(inject=True)

    # Host clock must not inflate: the CPU phase did the same host work.
    assert inj["cpu"] < 3 * base["cpu"], (base["cpu"], inj["cpu"])
    # Upstream phase untouched.
    assert inj["gpu_a"] < 3 * base["gpu_a"], (base["gpu_a"], inj["gpu_a"])
    # Next phase is NOT charged: its start event ticks after W completes.
    next_delta = inj["gpu_b"] - base["gpu_b"]
    assert abs(next_delta) < 0.1 * w_s, (
        f"next-phase delta {next_delta:.4f}s should be ~0 (W={w_s:.4f}s)"
    )
    # The work lands in the unattributed gap: wall grows by ~W while no
    # phase claims it — exactly what the residual check exists to catch.
    wall_growth = inj["wall"] - base["wall"]
    assert abs(wall_growth - w_s) < 0.2 * w_s, (
        f"wall grew {wall_growth:.4f}s, expected ~W={w_s:.4f}s"
    )


def test_queued_wait_charged_to_issuing_phase(calibration):
    """Single-stream queueing semantics: p1 enqueues W; p2 enqueues W/2 of
    its own work. p2's events queue behind p1's W on the stream, but events
    timestamp at execution — so the wait sits inside p1's bracket. Assert
    p1 ~= W and p2 ~= W/2: queued-behind wait time is charged to the
    ISSUING phase. This is the intended accounting."""
    n_inject = calibration["n_inject"]
    w_s = calibration["w_ms"] / 1000.0
    work = int(calibration["cycles_per_ms"] * 10)

    timer = PhaseTimer()
    timer.start_step()
    with timer.phase("p0"):
        CUDA_SLEEP(work)
    with timer.phase("p1"):
        CUDA_SLEEP(n_inject)  # W, enqueue only
    with timer.phase("p2"):
        CUDA_SLEEP(n_inject // 2)  # its own W/2
    with timer.phase("p3"):
        CUDA_SLEEP(work)
    summary = timer.step_summary()

    assert abs(summary["p1"] - w_s) < 0.2 * w_s, (summary["p1"], w_s)
    assert abs(summary["p2"] - w_s / 2) < 0.2 * (w_s / 2), (summary["p2"], w_s / 2)
    assert abs(summary[RESIDUAL_FRAC]) < 0.05
