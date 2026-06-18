# GRPO Profiling Harness

**Design Doc:** [design_doc.md](design_doc.md)

**Weights & Biases:** https://wandb.ai/an2353-project/grpo-profiling

## Summary

We run a co-located 1xA100 GRPO RL infrastructure using Qwen2.5-1.5B-Instruct and GSM8K dataset with a focus on profiling across varying G values using HF for training and vLLM for generation. The baseline of G=8 shows the generation phase taking 82% of the total wall clock, with backward (11%) and forward (5%) coming in next and everything else under 1%. Increasing G from 4 to 32 improved throughput metrics such as sec/completion by ~4x and token/sec by 5.7x. At the same time, the bottleneck's share moved: generation’s share of wall-clock fell from 88.7% at G=4 to 56.0% at G=32, while backward rose from 6.9% to 29.5%. The completion length tail that demonstrates the straggler problem worsened most visibly at G=32. The trustworthiness of these numbers is demonstrated by injection tests on CUDA events and runtime checks such as on-policy logprob identity check and a GPU residual time check. All metrics and runs are logged in Weights & Biases and are available here: https://wandb.ai/an2353-project/grpo-profiling.

## Repository Structure

```
.
├── src/grpo/        # Core GRPO training + profiling code
├── scripts/         # Run scripts, checkpoint/CSV utilities
├── analysis/        # Steady-state analysis across G=4,8,16,32
├── results/         # Profiling outputs and run artifacts
├── tests/           # Test suite
├── requirements.txt        # Base dependencies
├── requirements-gpu.txt    # GPU / CUDA dependencies
└── pytest.ini
```

## Setup

```bash
# Base dependencies
pip install -r requirements.txt

# GPU dependencies (CUDA environment)
pip install -r requirements-gpu.txt
```

## Hardware

Profiling was run on a single NVIDIA A100 with training (HuggingFace) and generation (vLLM) co-located on the same GPU.
