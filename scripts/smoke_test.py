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
from grpo.rewards.gsm8k import gsm8k_reward, gsm8k_reward_with_format  # noqa: E402
from grpo.rewards.verl_gsm8k import extract_solution  # noqa: E402
from grpo.train_loop import TrainConfig, train  # noqa: E402


class _DumpFirstStep:
    """Pass-through generator wrapper that prints the first N completions of
    step 0 exactly as training sees them (standing check #3: read the raw
    completions; curves are never evidence)."""

    def __init__(self, inner, tokenizer, n):
        self.inner = inner
        self.tokenizer = tokenizer
        self.n = n
        self._dumped = False

    def sync_weights(self, model):
        self.inner.sync_weights(model)

    def generate(self, prompt_token_ids, group_size, ground_truths=None):
        outs = self.inner.generate(
            prompt_token_ids, group_size, ground_truths=ground_truths
        )
        if not self._dumped:
            self._dumped = True
            print("=== DUMP: step-0 rendered prompt (first of batch) ===")
            print(self.tokenizer.decode(prompt_token_ids[0]))
            for k, out in enumerate(outs[: self.n]):
                gt = ground_truths[k // group_size] if ground_truths else None
                extracted = extract_solution(out["text"], method="strict")
                print(f"--- completion {k} (prompt {k // group_size}) ---")
                print(out["text"])
                print(
                    f">>> extracted={extracted!r} ground_truth={gt!r} "
                    f"reward={gsm8k_reward(out['text'], gt)}"
                )
            # Summary over the WHOLE step-0 batch, not just the N shown.
            stats = [
                gsm8k_reward_with_format(
                    out["text"],
                    ground_truths[i // group_size] if ground_truths else None,
                )
                for i, out in enumerate(outs)
            ]
            n_formatted = sum(formatted for _, formatted in stats)
            n_correct = sum(
                1 for reward, formatted in stats if formatted and reward == 1.0
            )
            print(
                f">>> format rate: {n_formatted}/{len(outs)}, "
                f"correct-given-formatted: {n_correct}/{n_formatted}"
            )
            print("=== END DUMP ===")
        return outs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", choices=["fake", "vllm"], default="fake")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--dump",
        type=int,
        default=0,
        metavar="N",
        help="print prompt + first N raw completions of step 0 with "
        "extracted answer / ground truth / reward (standing check #3)",
    )
    args = parser.parse_args()

    # The one device decision point.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        # Timer-attribution validation gates everything else: if attribution
        # is broken, no number from the run below is trustworthy.
        import subprocess

        repo_root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-m", "gpu", "-q",
             str(repo_root / "tests" / "test_gpu_attribution.py")],
            cwd=repo_root,
        )
        assert result.returncode == 0, "GPU attribution tests failed — see above"

    if args.generator == "fake":
        model_name = args.model or "sshleifer/tiny-gpt2"
    else:
        model_name = args.model or "Qwen/Qwen2.5-0.5B-Instruct"

    cfg = TrainConfig(
        model_name=model_name,
        device=device,
        max_steps=args.steps,
        wandb_mode="offline",
        # R0 trainer is bf16; the tiny-gpt2 plumbing path stays fp32 so the
        # Mac bit-identity baseline is preserved (config, not branching).
        model_dtype="float32" if args.generator == "fake" else "bfloat16",
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if args.generator == "fake":
        from grpo.rollout.fake_generator import FakeGenerator

        generator = FakeGenerator(tokenizer)
    else:
        from grpo.rollout.vllm_generator import VLLMGenerator  # GPU box only

        generator = VLLMGenerator(
            model_name=model_name,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
        )
    if args.dump > 0:
        generator = _DumpFirstStep(generator, tokenizer, args.dump)

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
