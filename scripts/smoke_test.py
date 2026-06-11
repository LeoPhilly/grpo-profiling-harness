#!/usr/bin/env python
"""Smoke test: a few real steps, asserting the standing checks' plumbing.

GPU box workflow (CLAUDE.md): pull from git -> run this -> only then launch
anything long. On the Mac it runs with --generator fake and validates only
accounting/plumbing, never numbers.

Does its work and exits; never leaves a session idle.
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

from grpo.data.gsm8k_data import gsm8k_pairs  # noqa: E402
from grpo.instrumentation.timing import TIMING_RESIDUAL_KEY  # noqa: E402
from grpo.train_loop import TrainConfig, train  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", choices=["fake", "vllm"], default="fake")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    # The one device decision point.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.generator == "fake":
        model_name = args.model or "sshleifer/tiny-gpt2"
    else:
        model_name = args.model or "Qwen/Qwen2.5-0.5B-Instruct"

    cfg = TrainConfig(
        model_name=model_name,
        device=device,
        max_steps=args.steps,
        wandb_mode="offline",
    )

    if args.generator == "fake":
        from transformers import AutoTokenizer

        from grpo.rollout.fake_generator import FakeGenerator

        generator = FakeGenerator(AutoTokenizer.from_pretrained(model_name))
    else:
        from grpo.rollout.vllm_generator import VLLMGenerator  # GPU box only

        generator = VLLMGenerator(
            model_name=model_name,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
        )

    pairs = gsm8k_pairs("train")[: cfg.prompts_per_step * args.steps]
    history = train(cfg, generator, pairs)

    assert len(history) == args.steps, f"ran {len(history)} steps, wanted {args.steps}"
    for i, metrics in enumerate(history):
        assert "check/logprob_identity" in metrics, f"identity missing at step {i}"
        for key, value in metrics.items():
            assert isinstance(value, (int, float)) and math.isfinite(
                value
            ), f"non-finite metric at step {i}: {key}={value}"
        # Standing check #2: if the residual drifts past 5%, the timing
        # harness is broken and no downstream number is trustworthy.
        residual = metrics[TIMING_RESIDUAL_KEY]
        assert abs(residual) < 0.05, f"timing residual {residual:.4f} at step {i}"
    if args.generator == "vllm":
        # Standing check #1 is only meaningful with a real engine: on-policy
        # at step 0, so the log-ratio must be ~0 (bf16 numerics ~0.01).
        first = abs(history[0]["check/logprob_identity"])
        assert first < 0.05, f"on-policy identity violated at step 0: {first:.4f}"

    print(f"SMOKE PASS ({args.generator}, {args.steps} steps, device={device})")
    for i, metrics in enumerate(history):
        print(f"  step {i}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
