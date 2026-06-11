"""Integration test: 2 real steps on CPU with FakeGenerator and a tiny HF
model. Asserts the plumbing — loss finite, standing-check key logged, no NaN.
It does NOT claim the model learns anything (RL fails silently; behavioral
verification happens on real runs, per CLAUDE.md)."""

import inspect
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import grpo.train_loop as train_loop_module
from grpo.data.gsm8k_data import render_prompt
from grpo.loss import grpo_loss_from_token_logprobs, shifted_token_logprobs
from grpo.train_loop import build_batch_phase, forward_loss_phase
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
        model_dtype="float32",  # CPU plumbing path; R0 bf16 is GPU-only
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
    # flowed through the real scorer the mean is exactly 0.5. ALL of its
    # completions carry the #### marker (even wrong ones), so format_rate
    # is exactly 1.0 — anything else means extraction/accounting broke.
    assert history[0]["train/reward_mean"] == 0.5
    assert all(m["train/format_rate"] == 1.0 for m in history)
    assert all("train/format_rate" in logged for logged in recorder.logged)

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


def test_grad_accum_exactly_matches_full_batch():
    """THE accumulation test: chunked loss AND gradients must equal the
    full-batch computation. Chunks are built with unequal token counts so a
    mean-of-chunk-means implementation fails (negative control below)."""
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    model.eval()  # tiny-gpt2 has dropout; eval makes both passes deterministic

    # 3 prompts of different lengths x G=2, completion lengths 5..14: B=6
    # with unequal per-chunk completion-token counts.
    questions = ["a?", "what is 2+2?", "a much longer question about apples?"]
    prompt_ids = [render_prompt(tokenizer, q) for q in questions]
    generator = FakeGenerator(tokenizer, completion_tokens=(5, 14), seed=3)
    outs = generator.generate(prompt_ids, 2, ground_truths=["1", "2", "3"])
    batch = build_batch_phase(prompt_ids, outs, 2, tokenizer.pad_token_id, "cpu")
    advantages = torch.randn(6, generator=torch.Generator().manual_seed(7))

    def run(micro_batch_size):
        model.zero_grad(set_to_none=True)
        loss, identity = forward_loss_phase(model, batch, advantages, micro_batch_size)
        grads = [
            None if p.grad is None else p.grad.clone() for p in model.parameters()
        ]
        return loss, identity, grads

    loss_full, identity_full, grads_full = run(6)  # one chunk = full batch
    loss_chunked, identity_chunked, grads_chunked = run(2)  # 3 unequal chunks

    assert torch.allclose(loss_chunked, loss_full, atol=1e-6)
    assert abs(identity_chunked - identity_full) < 1e-6
    for g_full, g_chunked in zip(grads_full, grads_chunked):
        assert (g_full is None) == (g_chunked is None)
        if g_full is not None:
            assert torch.allclose(g_full, g_chunked, atol=1e-6)

    # Negative control (inject-the-failure): the naive mean of per-chunk
    # losses must NOT equal the full-batch token-mean on this construction —
    # otherwise this test couldn't catch the mean-of-means bug.
    with torch.no_grad():
        chunk_losses = []
        for start in range(0, 6, 2):
            sl = slice(start, start + 2)
            logits = model(
                input_ids=batch["input_ids"][sl],
                attention_mask=batch["attention_mask"][sl],
            ).logits
            token_logprobs = shifted_token_logprobs(logits, batch["input_ids"][sl])
            chunk_losses.append(
                grpo_loss_from_token_logprobs(
                    token_logprobs, batch["completion_mask"][sl], advantages[sl]
                )
            )
        naive_mean_of_means = torch.stack(chunk_losses).mean()
    assert not torch.isclose(naive_mean_of_means, loss_full, atol=1e-6)


def test_config_defaults_match_locked_decisions():
    cfg = TrainConfig()
    assert cfg.model_dtype == "bfloat16"  # R0 baseline trainer dtype
    assert cfg.gpu_memory_utilization == 0.3  # R2 starting split


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
