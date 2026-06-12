#!/usr/bin/env python
"""Real measurement run for one rung. Does the work and exits (CLAUDE.md —
never leaves an interactive GPU session idle). Reuses train() unchanged.

Every 25 steps the first 4 raw completions (+ extracted/ground-truth/reward)
are APPENDED to results/dumps/<run-name>.txt. That file is the step-0 vs
step-~100 reading artifact for standing check #3: curves are never evidence
of learning; reward hacking shows up in completions.

Every run of this script is paid GPU time: add the costs.md row.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

from grpo.data.gsm8k_data import gsm8k_pairs  # noqa: E402
from grpo.rewards.gsm8k import gsm8k_reward_with_format  # noqa: E402
from grpo.rewards.verl_gsm8k import extract_solution  # noqa: E402
from grpo.train_loop import TrainConfig, train  # noqa: E402

DUMP_EVERY = 25
DUMP_N = 4


class _PeriodicDump:
    """Pass-through generator wrapper: every `every` steps, append the first
    `n` completions exactly as training saw them to the reading artifact."""

    def __init__(self, inner, path, every=DUMP_EVERY, n=DUMP_N):
        self.inner = inner
        self.path = path
        self.every = every
        self.n = n
        self._step = 0

    def sync_weights(self, model):
        self.inner.sync_weights(model)

    def generate(self, prompt_token_ids, group_size, ground_truths=None):
        outs = self.inner.generate(
            prompt_token_ids, group_size, ground_truths=ground_truths
        )
        if self._step % self.every == 0:
            with open(self.path, "a") as f:
                f.write(f"\n===== step {self._step} =====\n")
                for k, out in enumerate(outs[: self.n]):
                    gt = ground_truths[k // group_size] if ground_truths else None
                    reward, _ = gsm8k_reward_with_format(out["text"], gt)
                    extracted = extract_solution(out["text"], method="strict")
                    f.write(f"--- completion {k} (prompt {k // group_size}) ---\n")
                    f.write(out["text"] + "\n")
                    f.write(
                        f">>> extracted={extracted!r} ground_truth={gt!r} "
                        f"reward={reward}\n"
                    )
        self._step += 1
        return outs


def main():
    parser = argparse.ArgumentParser(
        description="One real GRPO measurement run; dumps raw completions "
        "every 25 steps to results/dumps/<run-name>.txt"
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--prompts-per-step", type=int, default=4)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="vLLM sampling cap; truncated completions lose the #### line "
        "and get reward 0 naturally via the strict parser",
    )
    parser.add_argument("--gpu-mem-util", type=float, default=0.3)
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=4,
        help="trainer forward/backward chunk size (gradient accumulation; "
        "full 32-seq batch OOMed at 1.5B/40GB on activations, 8 OOMed at "
        "steady state with Adam moments resident)",
    )
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dumps_dir = Path(__file__).resolve().parent.parent / "results" / "dumps"
    dumps_dir.mkdir(parents=True, exist_ok=True)
    dump_path = dumps_dir / f"{args.run_name}.txt"
    anomaly_path = dumps_dir / f"{args.run_name}-anomalies.txt"
    # Appended blocks from retries of the same run name must be
    # distinguishable when reading the artifacts later. The periodic dump
    # always gets content (step 0), so its header is written up front; the
    # anomaly file is created lazily at first firing (header passed via
    # config) so clean runs leave no header-only file.
    start_header = (
        f"=== run start ({args.run_name}, "
        f"{datetime.now().isoformat(timespec='seconds')}) ===\n"
    )
    with open(dump_path, "a") as f:
        f.write(start_header)

    cfg = TrainConfig(
        model_name=args.model,
        device=device,
        group_size=args.group_size,
        prompts_per_step=args.prompts_per_step,
        max_steps=args.steps,
        gpu_memory_utilization=args.gpu_mem_util,
        micro_batch_size=args.micro_batch_size,
        anomaly_dump_path=str(anomaly_path),
        anomaly_dump_header=start_header,
        wandb_mode="online",
    )

    # train() owns wandb.init; the run name travels via wandb's own env var
    # so train() stays unchanged.
    os.environ["WANDB_NAME"] = args.run_name

    from grpo.rollout.vllm_generator import VLLMGenerator  # GPU box only

    generator = VLLMGenerator(
        model_name=args.model,
        max_tokens=args.max_new_tokens,
        gpu_memory_utilization=args.gpu_mem_util,
    )

    generator = _PeriodicDump(generator, dump_path)

    pairs = gsm8k_pairs("train")[: cfg.prompts_per_step * cfg.max_steps]
    history = train(cfg, generator, pairs)

    final = history[-1]
    print(f"RUN COMPLETE: {len(history)} steps. Read the dump: {dump_path}")
    print(
        f"final step: reward_mean={final['train/reward_mean']:.3f} "
        f"format_rate={final['train/format_rate']:.3f} "
        f"identity={final['check/logprob_identity']:.4f} "
        f"residual={final['check/timing_residual_frac']:.4f}"
    )


if __name__ == "__main__":
    main()
