"""Integration test: 2 real steps on CPU with FakeGenerator and a tiny HF
model. Asserts the plumbing — loss finite, standing-check key logged, no NaN.
It does NOT claim the model learns anything (RL fails silently; behavioral
verification happens on real runs, per CLAUDE.md)."""

import inspect
import math

from transformers import AutoTokenizer

import grpo.train_loop as train_loop_module
from grpo.data.gsm8k_data import render_prompt
from grpo.instrumentation.timing import (
    TIMING_RESIDUAL_KEY,
    TOKENS_PER_SEC_GENERATE_KEY,
    WALL_CLOCK_KEY,
    phase_wandb_key,
)
from grpo.rollout.fake_generator import FakeGenerator
from grpo.train_loop import TrainConfig, train

TIMED_PHASES = (
    "render",
    "sync_weights",
    "generate",
    "reward",
    "build_batch",
    "forward_loss",
    "optimizer",
)

TINY_MODEL = "sshleifer/tiny-gpt2"

PAIRS = [
    ("What is 2+2? Reply with #### then the number.", "4"),
    ("What is 3+4? Reply with #### then the number.", "7"),
    ("What is 10-3? Reply with #### then the number.", "7"),
    ("What is 5*2? Reply with #### then the number.", "10"),
]


class _SpyFakeGenerator(FakeGenerator):
    """Records what the loop hands the generator as prompts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.received_prompts = []

    def generate(self, prompts, group_size, ground_truths=None):
        self.received_prompts.append(prompts)
        return super().generate(prompts, group_size, ground_truths=ground_truths)


class _RecordingRun:
    def finish(self):
        pass


class _RecordingWandb:
    def __init__(self):
        self.logged = []

    def init(self, **kwargs):
        return _RecordingRun()

    def log(self, metrics, step=None):
        self.logged.append(dict(metrics))


def test_two_steps_end_to_end(monkeypatch):
    recorder = _RecordingWandb()
    monkeypatch.setattr(train_loop_module, "wandb", recorder)

    cfg = TrainConfig(
        model_name=TINY_MODEL,
        device="cpu",
        group_size=4,
        prompts_per_step=2,
        max_steps=2,
        lr=1e-4,
    )
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    generator = _SpyFakeGenerator(tokenizer, completion_tokens=(6, 12), seed=0)

    history = train(cfg, generator, PAIRS)

    assert len(history) == 2
    for metrics in history:
        for key, value in metrics.items():
            assert math.isfinite(value), f"non-finite {key}={value}"

    # The standing-check wandb keys were actually logged, every step.
    assert len(recorder.logged) == 2
    for logged in recorder.logged:
        assert "check/logprob_identity" in logged
        assert "train/loss" in logged
        assert "train/reward_mean" in logged

    # FakeGenerator makes exactly half of each group correct, so if rewards
    # flowed through the real scorer the mean is exactly 0.5.
    assert history[0]["train/reward_mean"] == 0.5

    # Timing keys: every phase, wall clock, residual, tokens/sec — all
    # logged, all finite, and the residual within the standing 5% bound.
    for logged in recorder.logged:
        for phase_name in TIMED_PHASES:
            key = phase_wandb_key(phase_name)
            assert key in logged and logged[key] > 0
        assert logged[WALL_CLOCK_KEY] > 0
        assert logged[TOKENS_PER_SEC_GENERATE_KEY] > 0
        assert math.isfinite(logged[TIMING_RESIDUAL_KEY])
        assert abs(logged[TIMING_RESIDUAL_KEY]) < 0.05

    # Single-tokenization invariant: the generator received exactly the
    # token ids render_prompt produces — never prompt text.
    assert len(generator.received_prompts) == 2  # one call per step
    step0_questions = [q for q, _ in PAIRS[:2]]
    expected_ids = [render_prompt(tokenizer, q) for q in step0_questions]
    assert generator.received_prompts[0] == expected_ids
    for batch in generator.received_prompts:
        for ids in batch:
            assert all(isinstance(t, int) for t in ids)


def test_build_batch_takes_no_tokenizer():
    # Structural guard: re-encoding prompt text inside build_batch is the
    # misalignment bug class; the function must not even receive a tokenizer.
    params = inspect.signature(train_loop_module.build_batch_phase).parameters
    assert "tokenizer" not in params
    assert list(params) == [
        "prompt_token_ids",
        "outs",
        "group_size",
        "pad_token_id",
        "device",
    ]
