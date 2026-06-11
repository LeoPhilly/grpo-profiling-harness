"""Phase-timing harness. The CLAUDE.md timing rules apply verbatim.

CUDA is asynchronous: a bare Python timer around a kernel launch returns
before the kernel finishes and lies. GPU phases therefore use CUDA event
pairs, read only at step_summary() after one explicit synchronize (a sync
per phase would serialize the very pipeline being measured).

On the Mac (no CUDA) everything falls back to perf_counter. Those numbers
validate ONLY the accounting logic — phase boundaries, residual, wandb
keys — never performance.
"""

import time
from contextlib import contextmanager

import torch

# The wandb keys, defined once here and imported everywhere else.
PHASE_KEY_PREFIX = "time/"
WALL_CLOCK_KEY = "time/wall_clock"
TIMING_RESIDUAL_KEY = "check/timing_residual_frac"
TOKENS_PER_SEC_GENERATE_KEY = "time/tokens_per_sec_generate"

# Reserved step_summary() entries (they share the dict with phase names).
WALL_CLOCK = "wall_clock"
RESIDUAL_FRAC = "residual_frac"
_RESERVED = (WALL_CLOCK, RESIDUAL_FRAC)


def phase_wandb_key(name: str) -> str:
    return f"{PHASE_KEY_PREFIX}{name}"


def to_wandb_metrics(summary: dict) -> dict:
    """Map a step_summary() dict onto the canonical wandb keys."""
    out = {}
    for name, value in summary.items():
        if name == WALL_CLOCK:
            out[WALL_CLOCK_KEY] = value
        elif name == RESIDUAL_FRAC:
            out[TIMING_RESIDUAL_KEY] = value
        else:
            out[phase_wandb_key(name)] = value
    return out


class PhaseTimer:
    """Per-step phase timing: `with timer.phase("generate"): ...`.

    Backend is chosen once at construction: CUDA events when CUDA is
    available, perf_counter otherwise (use_cuda overrides, for tests).
    No global state — the train loop owns one instance.
    """

    def __init__(self, use_cuda=None):
        self.use_cuda = torch.cuda.is_available() if use_cuda is None else use_cuda
        self._records = []
        self._step_start = None

    def start_step(self):
        self._records = []
        if self.use_cuda:
            # Clean boundary: GPU work still in flight from before the step
            # must not leak into the first phase or the wall clock.
            torch.cuda.synchronize()
        self._step_start = time.perf_counter()

    @contextmanager
    def phase(self, name: str):
        if name in _RESERVED:
            raise ValueError(f"phase name {name!r} is reserved")
        if self.use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            yield
            end.record()  # queued on the stream; read only after the
            self._records.append((name, start, end))  # step_summary() sync
        else:
            t0 = time.perf_counter()
            yield
            self._records.append((name, t0, time.perf_counter()))

    def step_summary(self) -> dict:
        """{phase: seconds} (repeated names accumulate) plus:
        - "wall_clock": the whole step, measured independently of the phases
          (perf_counter from start_step; on CUDA, after a final synchronize).
        - "residual_frac": (wall_clock - sum(phases)) / wall_clock. If this
          drifts past 5%, the harness is broken and no downstream number is
          trustworthy (standing check #2).
        """
        if self._step_start is None:
            raise RuntimeError("step_summary() called before start_step()")
        if self.use_cuda:
            torch.cuda.synchronize()  # events are async; complete them first
        wall = time.perf_counter() - self._step_start

        summary = {}
        for name, a, b in self._records:
            seconds = a.elapsed_time(b) / 1000.0 if self.use_cuda else b - a
            summary[name] = summary.get(name, 0.0) + seconds
        phase_total = sum(summary.values())
        summary[WALL_CLOCK] = wall
        summary[RESIDUAL_FRAC] = (wall - phase_total) / wall
        return summary
