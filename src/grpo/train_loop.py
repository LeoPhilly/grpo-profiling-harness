"""Minimal end-to-end GRPO loop.

Pipeline: data -> generate -> reward -> advantages -> build batch -> forward
-> loss -> optimizer step -> log. Each phase is one named function — these
are the timing seams for the (next-task) instrumentation; no timers yet.

Alignment rule that makes standing check #1 possible: completions are NEVER
re-tokenized. The generator returns token ids and behavior logprobs; the
trainer builds sequences as prompt_ids + completion_token_ids, so behavior
logprobs stay aligned 1:1 with the tokens the loss sees.
"""

import os

# Fragmentation: the first 1.5B run OOMed with 5.3GB reserved-but-unallocated
# while coexisting with vLLM's resident pool; expandable segments lets the
# allocator grow/shrink instead of pinning fixed blocks. Read lazily at first
# CUDA allocation; harmless no-op on CPU.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from dataclasses import asdict, dataclass

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo.advantages import compute_group_advantages
from grpo.data.gsm8k_data import iter_prompt_batches, render_prompt
from grpo.instrumentation.timing import (
    TOKENS_PER_SEC_GENERATE_KEY,
    WALL_CLOCK,
    PhaseTimer,
    to_wandb_metrics,
)
from grpo.loss import grpo_loss_from_token_logprobs, shifted_token_logprobs
from grpo.rewards.gsm8k import gsm8k_reward_with_format


# Trainer-side peak memory key (R2 needs this number alongside vLLM's pool).
MEM_PEAK_KEY = "mem/trainer_peak_gb"


@dataclass
class TrainConfig:
    model_name: str = "sshleifer/tiny-gpt2"
    device: str = "cpu"  # the one device knob; the GPU box passes "cuda"
    # R0 baseline is a bf16 trainer (fp32 Adam state OOMed at 1.5B). The Mac
    # tiny-gpt2 test path passes "float32" here — config, not branching logic.
    # Pure-bf16 Adam (bf16 moments); fp32 master weights is the documented
    # fallback if the learning gate shows instability (see future_work.md).
    model_dtype: str = "bfloat16"
    group_size: int = 4
    prompts_per_step: int = 2
    # Trainer-forward chunk size along B (gradient accumulation). The full
    # step batch (prompts_per_step * group_size sequences) OOMed at 1.5B on
    # 40GB from activation memory (8 still OOMed at steady state once Adam
    # moments were resident); gradients are mathematically identical to
    # full-batch (see forward_loss_phase).
    micro_batch_size: int = 4
    max_steps: int = 2
    lr: float = 1e-5
    seed: int = 0
    # vLLM/trainer memory split (R2 rung) — consumed by whoever constructs
    # the VLLMGenerator; FakeGenerator ignores it.
    gpu_memory_utilization: float = 0.3
    # Standing-check-#1 tripwire: when |mean identity| exceeds the threshold,
    # the full batch is appended to anomaly_dump_path immediately (r0's
    # 15-step excursion fell between the every-25-step dumps). Empty path
    # disables the dump (FakeGenerator's identity is meaningless by design).
    anomaly_threshold: float = 0.5
    anomaly_dump_path: str = ""
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
    """Returns (rewards (B,), format_rate). One extraction per completion
    serves both — formatted means strict extraction found an answer."""
    results = [
        gsm8k_reward_with_format(out["text"], ground_truths[i // group_size])
        for i, out in enumerate(outs)
    ]
    rewards = torch.tensor([reward for reward, _ in results])
    format_rate = sum(formatted for _, formatted in results) / len(results)
    return rewards, format_rate


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
    """Standing check #1: log-ratio (trainer recompute - behavior) over
    completion tokens. ~0 on the first inner step when generation is truly
    on-policy (~0.01 is bf16/engine numerics; ~0.3 is a bug). With
    FakeGenerator the value is meaningless — only the plumbing is exercised.

    Returns (mean, per_sequence, valid): mean is the token-weighted mean over
    the whole batch (semantics unchanged from r0); per_sequence (B,) holds
    each sequence's own mean log-ratio — its extremes localize whether one
    sequence or the whole batch diverges; valid flags sequences with at
    least one completion token.

    Consumes the step's shared token logprobs (see shifted_token_logprobs) —
    recomputing a full-vocab log_softmax here OOMed the first GPU run."""
    with torch.no_grad():
        mask = batch["completion_mask"][:, 1:].to(token_logprobs.dtype)
        diff = (token_logprobs - batch["behavior_logprobs"][:, 1:]) * mask
        seq_tokens = mask.sum(dim=1)
        mean = (diff.sum() / seq_tokens.sum()).item()
        per_sequence = diff.sum(dim=1) / seq_tokens.clamp(min=1)
        return mean, per_sequence, seq_tokens > 0


def forward_loss_phase(model, batch, advantages, micro_batch_size, timer):
    """Chunked forward+backward along B (gradient accumulation).

    Each chunk's loss is weighted by chunk_completion_tokens /
    total_completion_tokens, so the summed gradients AND the returned
    aggregate loss equal the full-batch token-mean EXACTLY. Mean-of-chunk-
    means would be wrong: chunks have unequal token counts.

    backward() runs per chunk: that frees each chunk's graph (and its
    full-vocab logsumexp buffers) before the next forward — the whole point,
    since 1.5B OOMed on activations. Only the detached (b, T-1) gathered
    logprobs are kept, concatenated for the identity check.

    Timing: this function owns four timer phases — "forward" (model to
    logits), "loss_compute" (gather + loss + weighting), "backward" (both
    accumulated across chunks), and "identity_check" (once, on the
    concatenated logprobs). It is NOT wrapped in an outer phase: nesting
    would double-count time in the residual. No forward_loss aggregate is
    logged; analysis derives it (=forward+loss_compute+backward) for
    comparability with r0.

    Returns (detached aggregate loss, identity stats).
    """
    assert micro_batch_size >= 1
    B = batch["input_ids"].shape[0]
    total_tokens = batch["completion_mask"][:, 1:].sum()
    if total_tokens == 0:
        raise ValueError("completion_mask selects no tokens in the shifted frame")

    aggregate_loss = torch.zeros((), device=batch["input_ids"].device)
    chunk_logprobs = []
    for start in range(0, B, micro_batch_size):
        sl = slice(start, start + micro_batch_size)
        with timer.phase("forward"):
            logits = model(
                input_ids=batch["input_ids"][sl],
                attention_mask=batch["attention_mask"][sl],
            ).logits
        with timer.phase("loss_compute"):
            token_logprobs = shifted_token_logprobs(logits, batch["input_ids"][sl])
            del logits  # the (b, T, V) tensor drives peak memory; drop it now
            chunk_mask = batch["completion_mask"][sl]
            chunk_tokens = chunk_mask[:, 1:].sum()
            weighted = None
            if chunk_tokens > 0:  # all-empty-completion chunk contributes 0
                chunk_loss = grpo_loss_from_token_logprobs(
                    token_logprobs, chunk_mask, advantages[sl]
                )
                weighted = chunk_loss * (chunk_tokens / total_tokens)
            chunk_logprobs.append(token_logprobs.detach())
        if weighted is not None:
            with timer.phase("backward"):
                weighted.backward()  # accumulates grads; frees chunk's graph
                aggregate_loss += weighted.detach()

    with timer.phase("identity_check"):
        identity_stats = logprob_identity(torch.cat(chunk_logprobs, dim=0), batch)
    return aggregate_loss, identity_stats


def optimizer_phase(optimizer):
    """Gradients were accumulated in forward_loss_phase; step once, then
    clear for the next step's accumulation."""
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def preallocate_optimizer_state(optimizer):
    """Force AdamW's moment buffers into memory before step 0.

    AdamW allocates exp_avg/exp_avg_sq lazily at the first step(), so step 0
    ran with ~12GB more headroom than every later step — its memory picture
    lied, and step 1 OOMed. Mirror the lazy init exactly (zero moments, zero
    step counter): the first real step() then behaves identically.

    Deliberately NOT a zero-grad warmup step(): AdamW's decoupled weight
    decay shrinks weights even at zero gradient, and the extra step count
    would shift bias correction for the whole run.
    """
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state[p]
            state["step"] = torch.tensor(0.0)
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)


def log_phase(metrics, step):
    wandb.log(metrics, step=step)


def dump_anomaly_step(
    path, step, identity_mean, outs, ground_truths, per_sequence, valid, group_size
):
    """Standing-check-#1 tripwire payload: the full batch the moment the
    threshold is crossed, so transient excursions (r0: ~15 steps, gone before
    the next periodic dump) leave their completions behind for reading."""
    with open(path, "a") as f:
        f.write(
            f"\n===== ANOMALY step {step}: mean identity {identity_mean:.4f} =====\n"
        )
        for i, out in enumerate(outs):
            seq_logratio = per_sequence[i].item() if valid[i] else float("nan")
            f.write(
                f"--- completion {i} (prompt {i // group_size}) | "
                f"seq_logratio={seq_logratio:.4f} | "
                f"tokens={len(out['token_ids'])} | "
                f"truncated={out.get('truncated', False)} | "
                f"gt={ground_truths[i // group_size]!r} ---\n"
            )
            f.write(out["text"] + "\n")


# --- the loop ---


def train(cfg: TrainConfig, generator, pairs) -> list:
    """Run up to cfg.max_steps GRPO steps over (prompt, ground_truth) pairs.
    Returns the per-step metric dicts (also sent to wandb)."""
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=getattr(torch, cfg.model_dtype)
    ).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    # Step 0's memory picture must equal steady state (lazy Adam alloc made
    # step 0 lie about headroom; step 1 OOMed).
    preallocate_optimizer_state(optimizer)

    run = wandb.init(
        project=cfg.wandb_project, mode=cfg.wandb_mode, config=asdict(cfg)
    )
    timer = PhaseTimer()
    history = []
    step = 0
    for batch_pairs in iter_prompt_batches(pairs, cfg.prompts_per_step):
        if step >= cfg.max_steps:
            break
        if device.type == "cuda":
            # Per-step peak: reset here so the number is this step's, not a
            # high-water mark from warm-up or earlier steps.
            torch.cuda.reset_peak_memory_stats(device)
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
            rewards, format_rate = reward_phase(outs, ground_truths, cfg.group_size)
        with timer.phase("advantages"):
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
        # Times its own forward/loss_compute/backward sub-phases (an outer
        # phase here would double-count in the residual).
        loss, (identity, identity_per_seq, identity_valid) = forward_loss_phase(
            model, batch, advantages, cfg.micro_batch_size, timer
        )
        with timer.phase("optimizer"):
            optimizer_phase(optimizer)
        # log_phase is NOT timed: it emits this step's own summary, and its
        # cost falls outside wall_clock (taken here), keeping the residual
        # honest instead of silently absorbing logging time.
        summary = timer.step_summary()

        completion_tokens = int(batch["completion_mask"].sum().item())
        # Per-sequence extremes localize an excursion: one bad sequence vs
        # the whole batch drifting.
        seq_logratios = identity_per_seq[identity_valid]
        identity_min = seq_logratios.min().item() if seq_logratios.numel() else 0.0
        identity_max = seq_logratios.max().item() if seq_logratios.numel() else 0.0
        truncated = [bool(out.get("truncated", False)) for out in outs]
        metrics = {
            "train/loss": loss.item(),
            "train/reward_mean": rewards.mean().item(),
            # GRPO can't bootstrap from all-zero groups; format compliance is
            # the leading indicator of that failure mode at small scale.
            "train/format_rate": format_rate,
            # Truncation (hit max_new_tokens) is the lead suspect for the r0
            # identity excursion; vLLM reports it via finish_reason.
            "train/truncated_frac": sum(truncated) / len(truncated),
            "train/completion_tokens": completion_tokens,
            "check/logprob_identity": identity,
            "check/logprob_identity_min": identity_min,
            "check/logprob_identity_max": identity_max,
            # Util alone is deceptive; tokens/sec of useful work is the
            # number that has to accompany any utilization claim.
            TOKENS_PER_SEC_GENERATE_KEY: completion_tokens / summary["generate"],
            **to_wandb_metrics(summary),
        }
        if device.type == "cuda":
            metrics[MEM_PEAK_KEY] = (
                torch.cuda.max_memory_allocated(device) / 1024**3  # GiB
            )
        log_phase(metrics, step)
        # Tripwire AFTER logging so a dump failure can't lose the metrics.
        if cfg.anomaly_dump_path and abs(identity) > cfg.anomaly_threshold:
            dump_anomaly_step(
                cfg.anomaly_dump_path,
                step,
                identity,
                outs,
                ground_truths,
                identity_per_seq,
                identity_valid,
                cfg.group_size,
            )
        print(
            f"step {step + 1}/{cfg.max_steps} | "
            f"reward {metrics['train/reward_mean']:.2f} | "
            f"format {metrics['train/format_rate']:.2f} | "
            f"gen {summary['generate']:.1f}s | "
            f"wall {summary[WALL_CLOCK]:.1f}s | "
            f"identity {identity:.3f}",
            flush=True,
        )
        history.append(metrics)
        step += 1

    run.finish()
    return history
