#!/usr/bin/env python
"""Verify sync_weights actually syncs — R1 gate before clocking strategies.

Stages:
  A. load trainer (bf16) + VLLMGenerator as in the real loop; sync once.
  B. perturb a handful of trainer params across depth (+0.123).
  C. BEFORE re-syncing, confirm the engine still holds the OLD values
     (diff ~0.123). If it already matches, the 'extraction' is aliasing
     trainer memory and any PASS would be vacuous.
  D. sync again; perturbed targets must match engine-side exactly and a
     random sample of ~20 unperturbed tensors must be bit-identical.
Any tensor unmatchable by name (after fused-layout mapping) is listed BY
NAME on both sides — silently unmapped weights are the lie being hunted.

GPU box only, minutes. costs.md row applies.
"""

import argparse
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

PERTURB = 0.123
# vLLM fuses these for Qwen2-style models; rows concatenated in this order.
# VERIFY-ON-GPU: fusion layout (names and row order) may differ by version.
FUSED = {
    "q_proj": ("qkv_proj", ("q_proj", "k_proj", "v_proj")),
    "k_proj": ("qkv_proj", ("q_proj", "k_proj", "v_proj")),
    "v_proj": ("qkv_proj", ("q_proj", "k_proj", "v_proj")),
    "gate_proj": ("gate_up_proj", ("gate_proj", "up_proj")),
    "up_proj": ("gate_up_proj", ("gate_proj", "up_proj")),
}


def _find(name, engine_params):
    for cand in (
        name,
        name[len("model."):] if name.startswith("model.") else "model." + name,
    ):
        if cand in engine_params:
            return cand
    return None


def locate(name, trainer_sd, engine_params):
    """Map one trainer param name to the engine tensor (or fused slice).
    Returns (engine_tensor, engine_name_used, how) or None."""
    direct = _find(name, engine_params)
    if direct is not None:
        return engine_params[direct], direct, "direct"
    for part, (fused, order) in FUSED.items():
        token = f".{part}."
        if token not in name:
            continue
        fused_key = _find(name.replace(token, f".{fused}."), engine_params)
        if fused_key is None:
            return None
        offset = 0
        for member in order:
            if member == part:
                break
            offset += trainer_sd[name.replace(token, f".{member}.")].shape[0]
        rows = trainer_sd[name].shape[0]
        sliced = engine_params[fused_key][offset : offset + rows]
        return sliced, fused_key, f"fused[{offset}:{offset + rows}]"
    return None


def max_abs_diff(a, b):
    return (a.detach().float() - b.detach().float()).abs().max().item()


def pick_targets(names):
    """Perturbation sites across depth: embedding, early attn (exercises the
    fused-qkv mapping), mid MLP, lm_head (final row)."""
    layers = sorted({int(m.group(1)) for n in names
                     for m in [re.search(r"layers\.(\d+)\.", n)] if m})
    mid = layers[len(layers) // 2]
    wanted = [
        ("embedding", "model.embed_tokens.weight", None),
        ("early_attn", f"model.layers.{layers[0]}.self_attn.q_proj.weight", None),
        ("mid_mlp", f"model.layers.{mid}.mlp.up_proj.weight", None),
        ("lm_head_final_row", "lm_head.weight", -1),  # row slice
    ]
    targets = []
    for label, name, row in wanted:
        if name in names:
            targets.append((label, name, row))
        else:
            print(f"NOTE: {label} target {name!r} absent from trainer "
                  "named_parameters (tied weights?) — not perturbed; if the "
                  "engine materializes it separately it will appear in the "
                  "unmatched-engine list below.")
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--gpu-mem-util", type=float, default=0.3)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        sys.exit("verify_sync is GPU-box only (needs vLLM + CUDA).")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to("cuda")
    model.eval()
    from grpo.rollout.vllm_generator import VLLMGenerator  # GPU box only

    generator = VLLMGenerator(
        model_name=args.model, gpu_memory_utilization=args.gpu_mem_util
    )
    generator.sync_weights(model)  # sync #1, exactly as the real loop

    def engine_param_dict():
        # VERIFY-ON-GPU: engine module navigation; re-fetched after each sync
        # in case load_weights swaps modules instead of copying in place.
        runner = generator.llm.llm_engine.model_executor.driver_worker.model_runner
        return dict(runner.model.named_parameters())

    trainer_sd = dict(model.named_parameters())
    engine_params = engine_param_dict()

    # --- name-coverage pass: every tensor accounted, both sides ---
    mapping, unmatched_trainer, used_engine = {}, [], set()
    for name in trainer_sd:
        hit = locate(name, trainer_sd, engine_params)
        if hit is None:
            unmatched_trainer.append(name)
        else:
            mapping[name] = hit
            used_engine.add(hit[1])
    unmatched_engine = sorted(set(engine_params) - used_engine)
    for side, lst in (("trainer", unmatched_trainer), ("engine", unmatched_engine)):
        print(f"unmatched {side}-side tensors: {len(lst)}")
        for n in lst:
            print(f"  UNMATCHED ({side}): {n}")

    targets = pick_targets(set(trainer_sd))
    sample_pool = sorted(set(mapping) - {n for _, n, _ in targets})
    sample = random.Random(0).sample(sample_pool, min(20, len(sample_pool)))

    # --- stage B: perturb the trainer ---
    with torch.no_grad():
        for label, name, row in targets:
            if row is None:
                trainer_sd[name].add_(PERTURB)
            else:
                trainer_sd[name][row].add_(PERTURB)
            print(f"perturbed {label}: {name}" + ("" if row is None else f" row {row}"))

    # --- stage C: engine must still hold OLD values (no aliasing) ---
    aliasing_ok = True
    for label, name, row in targets:
        tensor, _, how = mapping[name]
        t = trainer_sd[name] if row is None else trainer_sd[name][row]
        e = tensor if row is None else tensor[row]
        diff = max_abs_diff(t, e)
        ok = 0.05 < diff < 0.2  # ~0.123 through bf16
        aliasing_ok &= ok
        print(f"pre-sync  {label:<18} ({how}) max|diff|={diff:.6f} "
              f"{'ok (engine holds old values)' if ok else 'FAIL — aliasing?'}")

    # --- stage D: sync #2, then compare ---
    generator.sync_weights(model)
    engine_params = engine_param_dict()

    perturbed_ok = True
    for label, name, row in targets:
        hit = locate(name, trainer_sd, engine_params)
        tensor = hit[0] if row is None else hit[0][row]
        t = trainer_sd[name] if row is None else trainer_sd[name][row]
        diff = max_abs_diff(t, tensor)
        ok = diff < 1e-6
        perturbed_ok &= ok
        print(f"post-sync {label:<18} ({hit[2]}) max|diff|={diff:.6f} "
              f"{'ok' if ok else 'FAIL'}")

    sample_ok = True
    for name in sample:
        hit = locate(name, trainer_sd, engine_params)
        diff = max_abs_diff(trainer_sd[name], hit[0])
        if diff != 0.0:
            sample_ok = False
            print(f"post-sync sample MISMATCH {name}: max|diff|={diff:.6g}")
    print(f"unperturbed sample: {len(sample)} tensors, "
          f"{'all bit-identical' if sample_ok else 'MISMATCHES above'}")

    print(
        "\nPASS criteria: (1) pre-sync diffs ~0.123 on every perturbed target "
        "(engine memory distinct, perturbation real); (2) post-sync perturbed "
        "diffs ~0; (3) unperturbed sample bit-identical; (4) zero unmatched "
        "tensors on either side."
    )
    coverage_ok = not unmatched_trainer and not unmatched_engine
    failures = [
        reason
        for ok, reason in (
            (aliasing_ok, "engine aliased trainer memory or perturbation lost"),
            (perturbed_ok, "perturbed weights not synced"),
            (sample_ok, "unperturbed weights drifted"),
            (coverage_ok, "unmapped tensors (see UNMATCHED lines)"),
        )
        if not ok
    ]
    if failures:
        print("SYNC VERIFY: FAIL — " + "; ".join(failures))
        sys.exit(1)
    print("SYNC VERIFY: PASS")


if __name__ == "__main__":
    main()
