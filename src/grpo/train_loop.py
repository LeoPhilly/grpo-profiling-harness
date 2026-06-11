"""Minimal end-to-end GRPO loop.

Pipeline: data -> generate -> reward -> advantages -> build batch -> forward
-> loss -> optimizer step -> log. Each phase is one named function — these
are the timing seams for the (next-task) instrumentation; no timers yet.

Alignment rule that makes standing check #1 possible: completions are NEVER
re-tokenized. The generator returns token ids and behavior logprobs; the
trainer builds sequences as prompt_ids + completion_token_ids, so behavior
logprobs stay aligned 1:1 with the tokens the loss sees.
"""

from dataclasses import asdict, dataclass

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo.advantages import compute_group_advantages
from grpo.data.gsm8k_data import iter_prompt_batches, render_prompt
from grpo.instrumentation.timing import (
    TOKENS_PER_SEC_GENERATE_KEY,
    PhaseTimer,
    to_wandb_metrics,
)
from grpo.loss import grpo_loss_from_token_logprobs, shifted_token_logprobs
from grpo.rewards.gsm8k import gsm8k_reward


@dataclass
class TrainConfig:
    model_name: str = "sshleifer/tiny-gpt2"
    device: str = "cpu"  # the one device knob; the GPU box passes "cuda"
    group_size: int = 4
    prompts_per_step: int = 2
    max_steps: int = 2
    lr: float = 1e-5
    seed: int = 0
    # vLLM/trainer memory split (R2 rung) — consumed by whoever constructs
    # the VLLMGenerator; FakeGenerator ignores it.
    gpu_memory_utilization: float = 0.3
    wandb_project: str = "grpo-profiling"
    wandb_mode: str = "offline"


# --- pipeline phases (timing seams; one function per phase) ---


def generate_phase(generator, prompt_token_ids, ground_truths, group_size):
    outs = generator.generate(
        prompt_token_ids, group_size, ground_truths=ground_truths
    )
    assert len(outs) == len(prompt_token_ids) * group_size
    return outs


def reward_phase(outs, ground_truths, group_size):
    rewards = [
        gsm8k_reward(out["text"], ground_truths[i // group_size])
        for i, out in enumerate(outs)
    ]
    return torch.tensor(rewards)


def build_batch_phase(prompt_token_ids, outs, group_size, pad_token_id, device):
    """Right-padded (B, T) tensors: input_ids, attention_mask, completion_mask
    (1 on completion tokens in the input frame), behavior_logprobs (filled at
    completion positions, 0 elsewhere).

    Consumes token ids only — prompt text was tokenized exactly once, in
    render_prompt. Deliberately takes no tokenizer: re-encoding text here
    could shift prompt lengths and misalign behavior logprobs (check #1)."""
    assert all(len(p) > 0 for p in prompt_token_ids)

    seqs = []
    for i, out in enumerate(outs):
        pids = prompt_token_ids[i // group_size]
        seqs.append((pids, list(out["token_ids"]), list(out["logprobs"])))

    B = len(seqs)
    T = max(len(p) + len(c) for p, c, _ in seqs)
    input_ids = torch.full((B, T), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((B, T), dtype=torch.long)
    completion_mask = torch.zeros((B, T), dtype=torch.long)
    behavior_logprobs = torch.zeros((B, T))
    for b, (pids, cids, blps) in enumerate(seqs):
        n_p, n_c = len(pids), len(cids)
        input_ids[b, : n_p + n_c] = torch.tensor(pids + cids)
        attention_mask[b, : n_p + n_c] = 1
        completion_mask[b, n_p : n_p + n_c] = 1
        behavior_logprobs[b, n_p : n_p + n_c] = torch.tensor(blps)

    # Position 0 must never be a completion token: in the shifted loss frame
    # token 0 is never a prediction target, so marking it would silently drop
    # a generated token from the loss.
    assert completion_mask[:, 0].sum() == 0

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "completion_mask": completion_mask.to(device),
        "behavior_logprobs": behavior_logprobs.to(device),
    }


def logprob_identity(token_logprobs, batch):
    """Standing check #1: mean log-ratio (trainer recompute - behavior) over
    completion tokens. ~0 on the first inner step when generation is truly
    on-policy (~0.01 is bf16/engine numerics; ~0.3 is a bug). With
    FakeGenerator the value is meaningless — only the plumbing is exercised.

    Consumes the step's shared token logprobs (see shifted_token_logprobs) —
    recomputing a full-vocab log_softmax here OOMed the first GPU run."""
    with torch.no_grad():
        mask = batch["completion_mask"][:, 1:].to(token_logprobs.dtype)
        behavior = batch["behavior_logprobs"][:, 1:]
        return (((token_logprobs - behavior) * mask).sum() / mask.sum()).item()


def forward_loss_phase(model, batch, advantages):
    logits = model(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
    ).logits
    # The one full-vocab log_softmax of the step, shared by loss and identity.
    token_logprobs = shifted_token_logprobs(logits, batch["input_ids"])
    loss = grpo_loss_from_token_logprobs(
        token_logprobs, batch["completion_mask"], advantages
    )
    identity = logprob_identity(token_logprobs, batch)
    return loss, identity


def optimizer_phase(optimizer, loss):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


def log_phase(metrics, step):
    wandb.log(metrics, step=step)


# --- the loop ---


def train(cfg: TrainConfig, generator, pairs) -> list:
    """Run up to cfg.max_steps GRPO steps over (prompt, ground_truth) pairs.
    Returns the per-step metric dicts (also sent to wandb)."""
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    run = wandb.init(
        project=cfg.wandb_project, mode=cfg.wandb_mode, config=asdict(cfg)
    )
    timer = PhaseTimer()
    history = []
    step = 0
    for batch_pairs in iter_prompt_batches(pairs, cfg.prompts_per_step):
        if step >= cfg.max_steps:
            break
        timer.start_step()
        with timer.phase("render"):
            # The single tokenization of prompt text; ids flow from here.
            prompt_token_ids = [
                render_prompt(tokenizer, question) for question, _ in batch_pairs
            ]
            ground_truths = [gt for _, gt in batch_pairs]
        with timer.phase("sync_weights"):
            generator.sync_weights(model)  # on-policy: engine matches trainer
        with timer.phase("generate"):
            outs = generate_phase(
                generator, prompt_token_ids, ground_truths, cfg.group_size
            )
        with timer.phase("reward"):
            rewards = reward_phase(outs, ground_truths, cfg.group_size)
            advantages = compute_group_advantages(rewards, cfg.group_size).to(
                device
            )
        with timer.phase("build_batch"):
            batch = build_batch_phase(
                prompt_token_ids,
                outs,
                cfg.group_size,
                tokenizer.pad_token_id,
                device,
            )
        with timer.phase("forward_loss"):
            loss, identity = forward_loss_phase(model, batch, advantages)
        with timer.phase("optimizer"):
            optimizer_phase(optimizer, loss)
        # log_phase is NOT timed: it emits this step's own summary, and its
        # cost falls outside wall_clock (taken here), keeping the residual
        # honest instead of silently absorbing logging time.
        summary = timer.step_summary()

        completion_tokens = int(batch["completion_mask"].sum().item())
        metrics = {
            "train/loss": loss.item(),
            "train/reward_mean": rewards.mean().item(),
            "train/completion_tokens": completion_tokens,
            "check/logprob_identity": identity,
            # Util alone is deceptive; tokens/sec of useful work is the
            # number that has to accompany any utilization claim.
            TOKENS_PER_SEC_GENERATE_KEY: completion_tokens / summary["generate"],
            **to_wandb_metrics(summary),
        }
        log_phase(metrics, step)
        history.append(metrics)
        step += 1

    run.finish()
    return history
