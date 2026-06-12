#!/usr/bin/env python
"""Identity-drift autopsy: produce DISCRIMINATING EVIDENCE, apply no fixes.

CONTEXT: check/logprob_identity drifts negative late in runs, correlated
with rising train/truncated_frac. Pre-registered hypotheses and signatures:
  (A) bookkeeping bug at a boundary (masking / shift / EOS / truncation
      final position) — log-ratio ~0 everywhere EXCEPT localized positions,
      specifically final position(s) of truncated sequences.
  (B) vLLM and HF genuinely differ for identical tokens through identical
      weights (kernel differences) — small uniform offset across ALL
      positions, truncated and non-truncated alike.
  (B1) dtype/accumulation sub-case — the fp32 trainer rescore closes the
      gap that the bf16 rescore shows.

Rescoring uses the IMPORTED production code path (build_batch_phase +
shifted_token_logprobs) — never a reimplementation, which could 'fix' a
bookkeeping bug by accident and diagnose nothing.

GPU box only, ~minutes (no training, fresh weights, 16 prompts, G=1).
Every run gets a costs.md row.
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from grpo.data.gsm8k_data import gsm8k_pairs, render_prompt  # noqa: E402
from grpo.loss import shifted_token_logprobs  # noqa: E402
from grpo.train_loop import build_batch_phase  # noqa: E402

N_PROMPTS = 16
MICRO_BATCH = 4
MAX_NEW_TOKEN_CONFIGS = (128, 512)  # 128 forces truncations; 512 is normal
BUCKETS = ("first", "middle", "last3", "final")


def position_bucket(i, length):
    """Disjoint relative-position buckets over a completion of `length`
    tokens: final token, first token, the 3 before final, everything else.
    Priority final > first > last3 so short completions stay disjoint."""
    if i == length - 1:
        return "final"
    if i == 0:
        return "first"
    if i >= length - 4:
        return "last3"
    return "middle"


def rescore(model, batch):
    """Per-position recomputed logprobs through the PRODUCTION shift/gather
    path (shifted_token_logprobs), chunked to bound logits memory."""
    chunks = []
    B = batch["input_ids"].shape[0]
    with torch.no_grad():
        for start in range(0, B, MICRO_BATCH):
            sl = slice(start, start + MICRO_BATCH)
            logits = model(
                input_ids=batch["input_ids"][sl],
                attention_mask=batch["attention_mask"][sl],
            ).logits
            chunks.append(shifted_token_logprobs(logits, batch["input_ids"][sl]))
            del logits
    return torch.cat(chunks, dim=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--gpu-mem-util", type=float, default=0.3)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("identity_autopsy is GPU-box only (needs vLLM + CUDA).")
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # bf16, exactly as the real loop loads the trainer.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    from grpo.rollout.vllm_generator import VLLMGenerator  # GPU box only

    generator = VLLMGenerator(
        model_name=args.model,
        max_tokens=MAX_NEW_TOKEN_CONFIGS[0],
        gpu_memory_utilization=args.gpu_mem_util,
    )
    generator.sync_weights(model)  # same sync as the real loop

    pairs = gsm8k_pairs("train")[:N_PROMPTS]
    # Same render path as the real loop: token ids, single tokenization.
    prompt_ids = [render_prompt(tokenizer, question) for question, _ in pairs]

    # One engine for both configs (a second engine would double load time
    # and memory). max_tokens is read per generate() call when building
    # SamplingParams.  # VERIFY-ON-GPU: cap honored per call on one engine
    gens, batches = {}, {}
    for max_new in MAX_NEW_TOKEN_CONFIGS:
        generator.max_tokens = max_new
        outs = generator.generate(prompt_ids, 1)
        n_trunc = sum(out["truncated"] for out in outs)
        print(f"max_new_tokens={max_new}: {n_trunc}/{len(outs)} truncated")
        gens[max_new] = outs
        batches[max_new] = build_batch_phase(
            prompt_ids, outs, 1, tokenizer.pad_token_id, device
        )

    # Rescore identical token sequences twice: bf16 (as the real loop),
    # then the model cast to fp32 (hypothesis B1 probe).
    recomputed = {"bf16": {}, "fp32": {}}
    for max_new in MAX_NEW_TOKEN_CONFIGS:
        recomputed["bf16"][max_new] = rescore(model, batches[max_new])
    model = model.float()
    for max_new in MAX_NEW_TOKEN_CONFIGS:
        recomputed["fp32"][max_new] = rescore(model, batches[max_new])

    out_dir = Path(__file__).resolve().parent.parent / "results" / "identity_autopsy"
    out_dir.mkdir(parents=True, exist_ok=True)

    for max_new in MAX_NEW_TOKEN_CONFIGS:
        batch, outs = batches[max_new], gens[max_new]
        mask = batch["completion_mask"][:, 1:].bool().cpu()
        behavior = batch["behavior_logprobs"][:, 1:].cpu()
        targets = batch["input_ids"][:, 1:].cpu()
        ratio = {
            dtype: recomputed[dtype][max_new].float().cpu() - behavior
            for dtype in ("bf16", "fp32")
        }

        csv_rows = []
        # (bucket, truncated, dtype) -> list of |log-ratio|
        agg = {}
        for b, out in enumerate(outs):
            positions = torch.nonzero(mask[b]).flatten().tolist()
            length = len(positions)
            if length == 0:
                continue
            for i, p in enumerate(positions):
                bucket = position_bucket(i, length)
                lr = {dt: float(ratio[dt][b, p]) for dt in ("bf16", "fp32")}
                csv_rows.append(
                    [b, i, bucket, int(targets[b, p]), lr["bf16"], lr["fp32"],
                     out["truncated"], length]
                )
                for dt in ("bf16", "fp32"):
                    agg.setdefault((bucket, out["truncated"], dt), []).append(
                        abs(lr[dt])
                    )

        csv_path = out_dir / f"max{max_new}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["seq_id", "position", "rel_position_bucket", "token_id",
                 "logratio_bf16", "logratio_fp32", "truncated", "completion_len"]
            )
            writer.writerows(csv_rows)
        print(f"wrote {csv_path} ({len(csv_rows)} positions)")

        print(f"\n=== max_new_tokens={max_new} ===")
        print(f"{'bucket':<8} {'truncated':<10} {'dtype':<6} "
              f"{'mean|lr|':>10} {'max|lr|':>10} {'n':>6}")
        for bucket in BUCKETS:
            for truncated in (True, False):
                for dtype in ("bf16", "fp32"):
                    vals = agg.get((bucket, truncated, dtype))
                    if not vals:
                        continue
                    print(
                        f"{bucket:<8} {str(truncated):<10} {dtype:<6} "
                        f"{sum(vals) / len(vals):>10.5f} {max(vals):>10.5f} "
                        f"{len(vals):>6}"
                    )
        print()

    print(
        "Signatures (human reads the tables above; no auto-verdict):\n"
        "(A)  bookkeeping bug: ~0 everywhere EXCEPT localized positions —\n"
        "     especially final position(s) of TRUNCATED sequences.\n"
        "(B)  engine-vs-HF kernels: small uniform offset across ALL\n"
        "     positions, truncated and non-truncated alike.\n"
        "(B1) dtype/accumulation: the fp32 columns close the gap that the\n"
        "     bf16 columns show."
    )


if __name__ == "__main__":
    main()
