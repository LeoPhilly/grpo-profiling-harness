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
    PhaseTimer,
    phase_wandb_key,
)
from grpo.rollout.fake_generator import FakeGenerator
from grpo.train_loop import TrainConfig, train

TIMED_PHASES = (
    "render",
    "sync_weights",
    "generate",
    "reward",
    "advantages",
    "build_batch",
    "forward",
    "loss_compute",
    "identity_check",
    "backward",
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


def test_two_steps_end_to_end(monkeypatch, tmp_path, capsys):
    recorder = _RecordingWandb()
    monkeypatch.setattr(train_loop_module, "wandb", recorder)

    anomaly_path = tmp_path / "anomalies.txt"
    cfg = TrainConfig(
        model_name=TINY_MODEL,
        device="cpu",
        model_dtype="float32",  # CPU plumbing path; R0 bf16 is GPU-only
        group_size=4,
        prompts_per_step=2,
        max_steps=2,
        lr=1e-4,
        # FakeGenerator's identity is ~-9 by construction: every step trips
        # the 0.5 threshold, which IS the injected-failure validation of the
        # anomaly tripwire.
        anomaly_dump_path=str(anomaly_path),
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
        # The r0-era aggregate is no longer logged: the flat decomposition
        # is exactly the eleven phases; analysis derives forward_loss.
        assert phase_wandb_key("forward_loss") not in logged

    # New identity stats: token-weighted mean bracketed by the per-sequence
    # extremes; FakeGenerator never truncates, so truncated_frac is exactly 0.
    for m in history:
        assert m["check/logprob_identity_min"] <= m["check/logprob_identity"]
        assert m["check/logprob_identity"] <= m["check/logprob_identity_max"]
        assert m["train/truncated_frac"] == 0.0

    # Anomaly tripwire fired on both steps (injected failure: fake identity
    # ~-9 vs threshold 0.5), capturing every completion with its stats.
    anomaly_text = anomaly_path.read_text()
    assert anomaly_text.count("===== ANOMALY step") == 2
    assert anomaly_text.count("--- completion") == 16  # 8 per step
    assert "seq_logratio=" in anomaly_text
    assert "truncated=False" in anomaly_text
    assert "tokens=" in anomaly_text

    # Per-step stdout line, built from the metrics dict.
    out = capsys.readouterr().out
    assert "step 1/2 | reward 0.50 | format 1.00 | gen " in out
    assert "| identity -9." in out

    # Single-tokenization invariant: the generator received exactly the
    # token ids render_prompt produces — never prompt text.
    assert len(generator.received_prompts) == 2  # one call per step
    step0_questions = [q for q, _ in PAIRS[:2]]
    expected_ids = [render_prompt(tokenizer, q) for q in step0_questions]
    assert generator.received_prompts[0] == expected_ids
    for batch in generator.received_prompts:
        for ids in batch:
            assert all(isinstance(t, int) for t in ids)


def test_anomaly_tripwire_respects_threshold(monkeypatch, tmp_path):
    # Negative control: threshold far above the fake's ~9 -> no dump at all.
    recorder = _RecordingWandb()
    monkeypatch.setattr(train_loop_module, "wandb", recorder)
    anomaly_path = tmp_path / "anomalies.txt"
    cfg = TrainConfig(
        model_name=TINY_MODEL,
        device="cpu",
        model_dtype="float32",
        group_size=4,
        prompts_per_step=2,
        max_steps=1,
        anomaly_threshold=1e6,
        anomaly_dump_path=str(anomaly_path),
    )
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    train(cfg, FakeGenerator(tokenizer, completion_tokens=(6, 12), seed=0), PAIRS)
    assert not anomaly_path.exists()


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
        timer = PhaseTimer(use_cuda=False)
        loss, (identity, per_seq, valid) = forward_loss_phase(
            model, batch, advantages, micro_batch_size, timer
        )
        grads = [
            None if p.grad is None else p.grad.clone() for p in model.parameters()
        ]
        return loss, identity, per_seq[valid], grads

    loss_full, identity_full, per_seq_full, grads_full = run(6)  # full batch
    loss_chunked, identity_chunked, per_seq_chunked, grads_chunked = run(2)

    assert torch.allclose(loss_chunked, loss_full, atol=1e-6)
    assert abs(identity_chunked - identity_full) < 1e-6
    assert torch.allclose(per_seq_chunked, per_seq_full, atol=1e-6)
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
    assert cfg.micro_batch_size == 4  # 8 OOMed at 1.5B steady state


def test_adam_preallocation_is_side_effect_free():
    """Pre-allocated Adam state must produce EXACTLY the trajectory lazy
    allocation produces — same grads in, bit-identical weights out. (A
    zero-grad warmup step() fails this: weight decay + step-count shift.)"""
    torch.manual_seed(0)
    model_lazy = torch.nn.Linear(8, 8)
    model_pre = torch.nn.Linear(8, 8)
    model_pre.load_state_dict(model_lazy.state_dict())

    opt_lazy = torch.optim.AdamW(model_lazy.parameters(), lr=1e-2)
    opt_pre = torch.optim.AdamW(model_pre.parameters(), lr=1e-2)
    train_loop_module.preallocate_optimizer_state(opt_pre)

    # State exists up front (the memory effect we want)...
    assert all(len(opt_pre.state[p]) > 0 for p in model_pre.parameters())
    assert all(len(opt_lazy.state[p]) == 0 for p in model_lazy.parameters())

    # ...and two real steps with identical grads give bit-identical weights.
    for _ in range(2):
        x = torch.randn(4, 8)
        for model, opt in ((model_lazy, opt_lazy), (model_pre, opt_pre)):
            opt.zero_grad(set_to_none=True)
            model(x).pow(2).sum().backward()
            opt.step()
    for p_lazy, p_pre in zip(model_lazy.parameters(), model_pre.parameters()):
        assert torch.equal(p_lazy, p_pre)


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
