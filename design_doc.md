## Summary:

We run a co-located 1xA100 GRPO RL infrastructure using Qwen2.5-1.5B-Instruct and GSM8K dataset with a focus on profiling across varying G values using HF for training and vLLM for generation. The baseline of G=8 shows the generation phase taking 82% of the total wall clock, with backward (11%) and forward (5%) coming in next and everything else under 1%. Increasing G from 4 to 32 improved throughput metrics such as sec/completion by ~4x and token/sec by 5.7x. At the same time, the bottleneck's share moved: generation’s share of wall-clock fell from 88.7% at G=4 to 56.0% at G=32, while backward rose from 6.9% to 29.5%. The completion length tail that demonstrates the straggler problem worsened most visibly at G=32. The trustworthiness of these numbers is demonstrated by injection tests on CUDA events and runtime checks such as on-policy logprob identity check and a GPU residual time check. All metrics and runs are logged in Weights & Biases and are available here: https://wandb.ai/an2353-project/grpo-profiling.



## Contents

- [Summary:](#summary)
- [What is your goal/motivation here?](#what-is-your-goalmotivation-here)
- [Experiment Framework Details](#experiment-framework-details)
  - [Why choose the Qwen1.5B model instead of the 7B model? Why not the math specific family of models?](#why-choose-the-qwen15b-model-instead-of-the-7b-model-why-not-the-math-specific-family-of-models)
  - [Why choose A100 40GB instead of H100 80GB?](#why-choose-a100-40gb-instead-of-h100-80gb)
- [Loss Formula Calculation](#loss-formula-calculation)
  - [Why is your inner epoch == 1? Why even keep the ratio if it's always constant?](#why-is-your-inner-epoch--1-why-even-keep-the-ratio-if-its-always-constant)
  - [Why is there no clipping mechanism?](#why-is-there-no-clipping-mechanism)
  - [Why is there no KL penalty in your GRPO loss formula?](#why-is-there-no-kl-penalty-in-your-grpo-loss-formula)
  - [How did you choose the reward formula?](#how-did-you-choose-the-reward-formula)
- [Experiment Set-up/Debugging Details](#experiment-set-updebugging-details)
  - [What OOM errors did you run into?](#what-oom-errors-did-you-run-into)
    - [vLLM gpu_memory_utilization](#vllm-gpu_memory_utilization)
    - [FP32 to BF16](#fp32-to-bf16)
    - [Logsumexp trick](#logsumexp-trick)
- [Timing Design](#timing-design)
  - [Stability: Why were only steps 50-100 considered?](#stability-why-were-only-steps-50-100-considered)
  - [Coverage: How were the phases divided up?](#coverage-how-were-the-phases-divided-up)
  - [Correctness: How do you know the wall clock phases accounted only for those phases?](#correctness-how-do-you-know-the-wall-clock-phases-accounted-only-for-those-phases)
    - [How did you know how long a cycle would take on your GPU?](#how-did-you-know-how-long-a-cycle-would-take-on-your-gpu)
    - [Why didn't you run the injection tests on a real run with a real pipeline?](#why-didnt-you-run-the-injection-tests-on-a-real-run-with-a-real-pipeline)
    - [Two Independent Checks:](#two-independent-checks)
    - [When did you sync GPU/CPU? Why not sync it after every phase?](#when-did-you-sync-gpucpu-why-not-sync-it-after-every-phase)
    - [How did you deal with vLLM running its own engine?](#how-did-you-deal-with-vllm-running-its-own-engine)
  - [What surprised you?](#what-surprised-you)
  - [Audit: Is everything accounted for? What are the checks?](#audit-is-everything-accounted-for-what-are-the-checks)
- [Part 1 run: Breakdown of timing](#part-1-run-breakdown-of-timing)
  - [RL Training Step Profile](#rl-training-step-profile)
    - [Table: Timing breakdown (seconds per step)](#timing-breakdown-seconds-per-step)
    - [Table: Variance share (contribution to wall-clock variance, %)](#variance-share-contribution-to-wall-clock-variance-)
  - [Analysis:](#analysis)
	  - [Generation is the bottleneck](#generation-is-the-bottleneck)
	  - [Variance of the wall clock](#variance-of-the-wall-clock)
	  - [Checks:](#checks)
  - [Why is the log ratio falling off after step 80? How did you debug or fix this?](#why-is-the-log-ratio-falling-off-after-step-80-how-did-you-debug-or-fix-this)
  - [Identity Bug Autopsy](#identity-bug-autopsy)
    - [Identity Autopsy — V2 (signed mean_lr, absolute-position bins)](#identity-autopsy--v2-signed-mean_lr-absolute-position-bins)
      - [Table: V2 · max_new_tokens = 128](#v2--max_new_tokens--128)
      - [Table: V2 · max_new_tokens = 512](#v2--max_new_tokens--512)
  - [How did you check whether the drift was a function of the RL-updated weights?](#how-did-you-check-whether-the-drift-was-a-function-of-the-rl-updated-weights)
- [Part 2 runs: Straggler & G variants](#part-2-runs-straggler--g-variants)
  - [Why are your straggler metrics based on length and not timings?](#why-are-your-straggler-metrics-based-on-length-and-not-timings)
  - [Which straggler metrics did you choose and why?](#which-straggler-metrics-did-you-choose-and-why)
  - [Table: Key Metrics from G = {4, 8, 16, 32}](#key-metric-from-g--4-8-16-32)
	  - [Why does the sec/completions improve ~4x? Why is this not a linear improvement?](#why-does-the-seccompletions-improve-4x-why-is-this-not-a-linear-improvement)
	  - [How does the bottleneck change as G increases and why? How come each doubling lowers its share more and more?](#how-does-the-bottleneck-change-as-g-increases-and-why-how-come-each-doubling-lowers-its-share-more-and-more)
	  - [How do stragglers worsen with increased G?](#how-do-stragglers-worsen-with-increased-g)
	  - [How does the overall wall clock increase? Is it in proportion to G?](#how-does-the-overall-wall-clock-increase-is-it-in-proportion-to-g)
	  - [Why does the log ratio drift occur more drastically in G=32 but not at all in G=16?](#why-does-the-log-ratio-drift-occur-more-drastically-in-g32-but-not-at-all-in-g16)
  - [How noisy are these numbers? Are the multiples trustworthy?](#how-noisy-are-these-numbers-are-the-multiples-trustworthy)
  - [Was Amdahl's law noticable?](#was-amdahls-law-noticable)
- [Future work & Limitations](#future-work--limitations)
  - [Which other forms of optimization would you profile next?](#which-other-forms-of-optimization-would-you-profile-next)
  - [Further work into the straggler problems](#further-work-into-the-straggler-problems)
  - [Limitations](#limitations)
- [How was AI used?](#how-was-ai-used)
- [References](#references)
- [Appendix](#appendix)

## What is your goal/motivation here?

My goal is to build a profiling tool, with a validation framework demonstrating proof of correctness, for the wall-clock of each phases of the GRPO RL process. This can be extended to variants like Dr.GRPO/DAPO etc; essentially any RL algo that lacks a policy critic and has multiple generations per prompt. From there, I was curious to see what aspects of profiling change with varying G (number of generations per prompt), and how this causes the bottleneck to move.

For context, I was aware that generation takes the majority of time in this particular type of RL process, but was curious to see how prominent this would be at a small scale setting, and how much time each of the other phases would take.

## Experiment Framework Details

### Why choose the Qwen1.5B model instead of the 7B model? Why not the math specific family of models?

Since I was using the GSM8K dataset, I wanted there to be room for the model to learn (as proof of correctness), and therefore did not use the math specific model (which are known to perform well on their particular dataset). I also chose the 1.5B model for compute and memory reasons (more below on the GPU). Also, this way I could run the 0.5B model as a proxy test for OOM errors as a first check.

### Why choose A100 40GB instead of H100 80GB?

I initially planned to use A100 80GB, but unfortunately lambda labs only had the A100 40GB available, and the A100 was about ⅓ of the cost of the H100.

## Loss Formula Calculation

### Why is your inner epoch == 1? Why even keep the ratio if it's always constant?

The goal for inner epoch == 1 was to catch policy drift (since in this case the ratio in the GRPO formula is always 1, or the log ratio is always 0). Moreover, since the goal was plain profiling, the learning/efficiency improvements from high inner epoch were not of interest, but making sure that this RL was on-policy (with proof) was important for correctness. Also, this was useful as a diagnostic not only for on-policy, but also for catching bugs (weight syncing and masking/ token-shifting). More on this later in the doc.

### Why is there no clipping mechanism?

Note again that since we are using inner epoch ==1, there is no need for the importance sampling clipping mechanism since that is used when inner epochs > 1 (and off-policy) to keep the updates under a certain range.

### Why is there no KL penalty in your GRPO loss formula?

Note that many of the current variations of GRPO (including Dr.GRPO and DAPO) both remove the reference model KL penalty and see better returns. Moreover, from a principle standpoint, as I was running it on GSM8K and only for a limited number of steps with profiling being the goal, the current policy drifting too far was not a concern. Also, from a practical standpoint, I was running my A100 at > 80% memory, and storing the reference model would be limiting (especially since I was going to increase G and was already hitting OOM errors).

### How did you choose the reward formula?

The reward/answers format is one of the challenges of using GSM8K, and the metrics that clearly can show with examples that RL is working. I used the standard rewards formula for GSM8K that is used in the popular verl library (https://github.com/verl-project/verl). However, in order to prevent reward hacking, I used only the strict version of the format.

## Experiment Set-up/Debugging Details

### What OOM errors did you run into?

Note that since I was running this on a single A100 40GB, this was a colocated GPU set up (using the same GPU is used for both generation/training) with training done via HF and generation done via vLLM. I ran into quite a few OOM errors and I have listed some of the major ones (in order of occurrence), along with their solutions/diagnostics below:

#### vLLM gpu_memory_utilization

Firstly, the vLLM gpu_memory_utilization was at 0.9 by default which is already ~36GB of memory. Since vLLM is used primarily for inference on a disaggregated infrastructure, I reduced this to 0.3. Note again this is a colocated setup, and hence HF training was sharing the same memory space as vLLM. This divides ~12GB for vLLM and ~28GB for HF.

#### FP32 to BF16

Next, I reduced the model load type from FP32 to BF16. By default, HF stores the AdamW params (weights, grads and the 2 moments) at FP32 (same as model load type). By rough calculation, at 4 bytes for each of the 1.5B params, this takes up 6GB (weights) + 6GB(grads) + 12GB (both moments) = 24GB. While less than 28GB, this is before any of the forward/backward activations and hence also ran into OOM errors. By changing all of the above to BF16, this would take about 2 bytes instead of 4, for a total of 12GB.

Note that this brings some precision and training stability tradeoff (since even in mixed-precision training master weights are always fp32), and is one of the many limitations of this experiment.

#### Logsumexp trick

After running into OOM errors still, I used the logsumexp trick from this article: https://omkaark.com/posts/cce.html (which got it from this paper originally: https://arxiv.org/html/2411.09009v1). GRPO requires per token_log_prob at a token level, and I was initially building the whole log_softmax tensor (for logits - log(sum(exp(logits))), and then grabbing the value for the chosen log index. Using this trick, I changed it to log (chosen) - logsumexp(logits), which worked. Also note that this was needed because I was running 4 micro batches at a time, with initial G=8, for a total of 32 sequences per step.

## Timing Design

### Stability: Why were only steps 50-100 considered?

In order to not factor in the initial startup costs of cuda, vLLM and others, I used only steps 50-100 for stability in the analysis (called steady_state in the files).

### Coverage: How were the phases divided up?

In order to ensure end-to-end coverage, I divided the phases into the following 11 (as also shown in wandb) in order:
1. time/render – note this is tokenizing the prompt.
2. time/sync_weights
3. time/generate
4. time/reward
5. time/advantages
6. time/build_batch
7. time/forward
8. time/loss_compute
9. time/backward
10. time/identity_check
11. time/optimizer

Two quick notes here: 1. Some of these times are technically majority CPU functions (like reward) but this tracks the GPU phase attributed to those phases and 2. forward/backward/loss_compute are per micro-batch and aggregate in the time/ metrics.

In addition, I also logged time/wall_clock which measures the entire process (used for auditing) – see below.

### Correctness: How do you know the wall clock phases accounted only for those phases?

GPU phases are timed via pairs of cuda event timers (using a context manager), and CPU is timed via time.perf_counter().
	Injection Tests:
	To ensure that the cuda events were correctly in my Phase Timer design, I created 4 dummy phases (in order, each of ~10ms) and added cuda sleep cycles to each and measured the wall clock changes for each phase to account for any spillovers. Essentially, the injection should show up only on the particular phase it was added to and the surrounding phases should remain the same as measured by an initial baseline. As a side note, I initially considered adding an async operation (like a large matmul) since this would actually add a known operation to the GPU instead of sleep, but chose against it because this might have effects on memory too.

#### How did you know how long a cycle would take on your GPU?

The key here is that cuda sleep is in terms of cycles, not seconds. And because different GPUs can have different kernel behavior, I first calculated cycles/ms for my specific A100 GPU. Cross checking it with the GPU's own clock_rate(), I caught a subtle metric difference in that I was assuming kHz (vs the actual MHz). I then scaled that ratio to be ~200 ms, and measured that as a change. Note that this was not exactly 200ms since cycles/ms can be noisy, but approximate.

#### Why didn't you run the injection tests on a real run with a real pipeline?

Primarily because the actual wall clocks are noisy, and secondly because it was not needed since for phase attribution the surrounding phases could be dummy. We only had to check the injection when showing in the correct phase, and there was no spillover – both of which could be done without a real run. The actual mechanisms, like the same context manager and real GPU streams with real CUDA kernels was all that was needed.

#### Two Independent Checks:
I ran two independent checks – one in the CPU that used python's time.perf_counter() to measure full end to end wall clock, and the other in the GPU which logged each of the RL phases via cuda events. Note that the total timer accounted for by both clocks is not exactly equal, but the residual between them serves a good audit check that all times are accounted for (more below in the audit section).

#### When did you sync GPU/CPU? Why not sync it after every phase?

Firstly, we can't sync after every phase because although that gives an accurate reading of the particular phase, it would not reflect the actual pipeline/wall-clock since it would cause artificial slow down at each sync step and would not let async work be reflected. Therefore, I sync GPU/CPU twice, one at the very end to get all the values from the GPU, and once at the very beginning to make sure that the next step has a clean boundary that only accounts for its time and doesn't have any spillover from the previous step.

#### How did you deal with vLLM running its own engine?

The time/generate surrounds vLLM, and because it runs its own engine, I wasn't sure if it would show up in my cuda phase (for example, if it had its own stream). However, running a real run showed that it does show up in the time/generate (~80% of the time goes here, which is hard to miss), and the timing_residual_frac (more below in the audit section) was also at a minimum overall.

### What surprised you?

For the injection tests, I tried installing a GPU injection tests during a CPU phase (like reward), and because we used perf.timer for python, I knew it wouldn't show up here, so I assumed it would spill over to the next GPU clocked phase (where we used cuda events). However, from running this it seems that the next phase cuda events actually don't clock this extra GPU sleep either, it goes unattributed to any phase. It only shows up in the residual = wall clock - sum of all phases clocked section. This was interesting because I assumed that GPU work launched under a CPU phase would show up in the next GPU's wall clock and essentially inflate that by spilling over. Note that the reason for this is that next phase's cuda start event is behind the cuda sleep in the GPU queue, so the cuda start event clock starts only after the cuda sleep is finished. This is also proof that if residual is within the expected margins, it means that there is no major unattributed or unexpected hidden GPU work being done.

### Audit: Is everything accounted for? What are the checks?

There were two key checks: on-policy RL and no unaccounted time. This was done using these 4 metrics on wandb:
1. check/timing_residual_frac
2. check/logprob_identity
3. check/logprob_identity_min
4. check/logprob_identity_max

For the on-policy check, I logged the log ratio (which should be 0 since this is in log terms) to watch for weight-sync bugs or off-policy drift. The min/max additions were extra to see if there was anything systematically wrong.

The timing_residual_frac was calculated as the gap of the two independent checks as described above:

```math
\text{timing\_residual\_frac} = \frac{t_{\text{wall}}^{\text{CPU}} - \sum_{i} t_{i}^{\text{GPU}}}{t_{\text{wall}}^{\text{CPU}}}
```

This was to ensure that there is no unaccounted for time, and that there was no spillover from one GPU phase to another.

## Part 1 run: Breakdown of timing

## RL Training Step Profile

**Window:** `[50, 100)` · **Steps:** 50 · **Mean wall-clock:** 9.62 s/step
**Throughput:** 1,144.6 generate tokens/sec · **Mean abs. timing residual:** 0.00024

### Timing breakdown (seconds per step)

Sorted by share of wall-clock time.

| Phase | Mean | Std | p10 | p90 | % of wall |
|---|---:|---:|---:|---:|---:|
| generate | 7.893519 | 0.923142 | 5.963585 | 8.480594 | 82.038 |
| backward | 1.116587 | 0.112870 | 0.927363 | 1.218327 | 11.605 |
| forward | 0.487464 | 0.052998 | 0.389522 | 0.539876 | 5.066 |
| optimizer | 0.058869 | 0.035627 | 0.053601 | 0.054071 | 0.612 |
| loss_compute | 0.041803 | 0.004345 | 0.035114 | 0.047245 | 0.434 |
| sync_weights | 0.012367 | 0.001203 | 0.011707 | 0.012972 | 0.129 |
| build_batch | 0.005721 | 0.000605 | 0.004856 | 0.006363 | 0.059 |
| render | 0.002384 | 0.000173 | 0.002189 | 0.002680 | 0.025 |
| advantages | 0.000331 | 0.000143 | 0.000176 | 0.000454 | 0.003 |
| reward | 0.000221 | 0.000073 | 0.000182 | 0.000246 | 0.002 |
| identity_check | 0.000210 | 0.000028 | 0.000184 | 0.000229 | 0.002 |
| **forward_loss** *(forward+loss+backward)* | 1.645853 | 0.169108 | 1.356228 | 1.803391 | 17.106 |
| **wall_clock (total)** | 9.621753 | 1.090920 | 7.396900 | 10.341626 | — |

### Variance share (contribution to wall-clock variance, %)

| Source | Var share |
|---|---:|
| generate | 71.606 |
| covariance remainder | 26.979 |
| backward | 1.070 |
| forward | 0.236 |
| optimizer | 0.107 |
| loss_compute | 0.002 |
| advantages | 0.000 |
| build_batch | 0.000 |
| identity_check | 0.000 |
| render | 0.000 |
| reward | 0.000 |
| sync_weights | 0.000 |

This run is logged as 'r0-base-1.5b-g8-profile' in the wandb link. G == 8.

### Analysis:

#### Generation is the bottleneck:
As expected, generation accounts for the biggest piece of the wall clock coming in at 82%. From there, the other major numbers are backward at ~11.5% and  forward ~5% (backward is ~2.3x forward, more than the expected ~2×). All other phases are negligible in the calculation and make up < 2% altogether.

#### Variance of the wall clock:
Generator's standard deviation is also quite high compared to the rest, which hints at the straggler problem. The more interesting aspect is that the variance in total wall clock is only 71% explained by the variance of the generated phase, with the next factor being the covariance remainder at 27% instead of an individual phase. This is likely explained by the fact that most of the phases move together, i.e that a longer generation takes longer forward/backward pass instead of all them being independent.

#### Checks: 
Furthermore, our checks for check/timing_residual_frac and check/logprob_identity
are also functioning, with logprob_identity showing a concerning drop at the end (addressed below). Residual timing during the steady state (steps 50-100) is <.05%.

### Why is the log ratio falling off after step 80? How did you debug or fix this?

Note that log ratio seems to be ~0 until about step 80, after which even the min/max checks fall down, pointing towards a consistent, one-sided drift. Firstly, it could indicate that our RL is no longer on-policy (at least from the check), and I suspect the bug is one of the following:
1. Since this is happening only steps 80 onwards, I suspect it might be something to do with GRPO itself. It is a feature of GRPO that as training progresses, generation length also increases (although I doubt that this would show up so soon in step 80). However, if that is the case, there may be an EOS bug or truncation bug.
2. Since all our calculations are in BF16 (instead of the ideal FP32), it's possible that there could be subtle differences in the calculation that show up for longer sequences or after a certain number of calculations (or in other forms like the KV cache).
3. Lastly, this might be because vLLM and HF run different engines and so compute logprobs slightly differently for the same tokens.

Note that as long as the weight-sync works, it is technically still on-policy, and this is a limitation of the check. My bet is on #1 (see below: turns out to be wrong) since this is not something I accounted for in my code. To debug this, I created an autopsy test file that reduces max new tokens to 128/512, and then checked each token position by position via vLLM and HF, in both FP32 and BF16.


### Identity Bug Autopsy

### Identity Autopsy — V2 (signed mean_lr, absolute-position bins)

### V2 · max_new_tokens = 128

| bucket | truncated | dtype | mean_lr | mean\|lr\| | max\|lr\| | n |
|---|---|---|---:|---:|---:|---:|
| 0-64    | True  | bf16 |  0.00321 | 0.02609 | 0.29177 | 896 |
| 0-64    | True  | fp32 |  0.00049 | 0.01177 | 0.25294 | 896 |
| 0-64    | False | bf16 | -0.00388 | 0.02628 | 0.26283 | 128 |
| 0-64    | False | fp32 |  0.00515 | 0.01439 | 0.36183 | 128 |
| final   | True  | bf16 |  0.00515 | 0.00854 | 0.03531 | 14 |
| final   | True  | fp32 |  0.00099 | 0.00105 | 0.00783 | 14 |
| final   | False | bf16 |  0.01235 | 0.02527 | 0.03762 | 2 |
| final   | False | fp32 | -0.00034 | 0.00105 | 0.00139 | 2 |

### V2 · max_new_tokens = 512

| bucket | truncated | dtype | mean_lr | mean\|lr\| | max\|lr\| | n |
|---|---|---|---:|---:|---:|---:|
| 0-64    | True  | bf16 |  0.00636 | 0.02958 | 0.11886 | 64 |
| 0-64    | True  | fp32 | -0.00167 | 0.01107 | 0.07356 | 64 |
| 0-64    | False | bf16 |  0.00436 | 0.02454 | 0.25493 | 897 |
| 0-64    | False | fp32 | -0.00001 | 0.01232 | 0.38305 | 897 |
| 128-256 | True  | bf16 | -0.00068 | 0.03686 | 0.24894 | 128 |
| 128-256 | True  | fp32 |  0.00027 | 0.01642 | 0.14535 | 128 |
| 128-256 | False | bf16 |  0.00322 | 0.01481 | 0.27011 | 1205 |
| 128-256 | False | fp32 | -0.00057 | 0.00690 | 0.21076 | 1205 |
| final   | True  | bf16 | -0.13527 | 0.13527 | 0.13527 | 1 |
| final   | True  | fp32 |  0.02351 | 0.02351 | 0.02351 | 1 |
| final   | False | bf16 |  0.01037 | 0.02296 | 0.09690 | 15 |
| final   | False | fp32 |  0.01129 | 0.01161 | 0.11730 | 15 |

Note full charts are in the appendix. 

As the chart shows, there seems to be no spike in the log ratio numbers in any particular aspects, rather there happens to be small disagreements everywhere. Truncation or position of the token doesn't have any standout impact on the log ratio either, hence ruling out the KV cache or masking/truncation bug hypothesis. Furthermore, the difference is roughly halved when calculations are done in FP32 as opposed to BF16, yet the notable aspect is that the difference remains. This therefore points to kernel/engine level differences between HF and vLLM, which are compounded because my model runs in BF16.

This discrepancy between HF and vLLM also matches the documented vLLM train-inference mismatch, which can be traced down to logprob-processing and lm_head precision as per https://huggingface.co/blog/ServiceNow-AI/correctness-before-corrections

### How did you check whether the drift was a function of the RL-updated weights?

The only aspect I wanted to confirm further was this was not a downstream effect of updated weights from RL learning. In order to test this, I checkpointed the weights at step 100 and reran the autopsy. The numbers came back nearly the same as above (fp32 halving the log ratio, subtle difference everywhere, no relation to truncation or position), which rules out that the drift is derived from updated weights and once again points to being a property of the live generation via vLLM and recompute via HF dynamics.

## Part 2 runs: Straggler & G variants

### Why are your straggler metrics based on length and not timings?

As I use VLLM for inference – which incorporates continuous batching – timing would be an inaccurate metric. For example, completion could take longer or shorter depending on which batch they are paired up with, and hence the timing is not solely dependent on the generation. This makes it an unreliable variable for comparison. Note that length is not perfect either, since the metric we really want is latency of each generation; however, length is a good proxy in this case.

### Which straggler metrics did you choose and why?

The main metric is the straggler/p99_p50 ratio, which is the ratio between the 99% percentile over the median. I chose p99 over p90 or max because it accurately captures almost all of the tail yet is robust to a single worst case outlier dictating the ratio. Moreover, the max is also more affected by the sample size (high samples tend to have higher maxes), which would be less stable compared to p99 among various values of G. Note the above is true in principle, but in practice for small values like G=4 p99 is essentially the same as max. Hence, I also logged the max along with the median. 

### Key Metric from G = {4, 8, 16, 32}

| metric | G=4 | G=8 | G=16 | G=32 |
|---|---|---|---|---|
| tokens/sec | 639 | 1166 | 2208 | 3680 |
| wall/step (s) | 7.25 | 9.47 | 9.94 | 14.20 |
| completions/step | 16 | 32 | 64 | 128 |
| sec/completion (wall÷comp) | 0.453 | 0.296 | 0.155 | 0.111 |
| generate share | 88.7% | 81.8% | 69.5% | 56.0% |
| backward share | 6.9% | 11.8% | 20.2% | 29.5% |
| identity steady → last_20 | +0.0026 → +0.0024 | −0.014 → −0.029 | −0.0014 → −0.0029 | −0.042 → −0.095 |
| p99/p50 ratio (mean / max) | 1.72 / 2.35 | 1.80 / 2.24 | 1.80 / 2.53 | 1.92 / 4.23 |

Note that the above runs start with 'r2...' in the wandb link and I re ran G = 8 in the same GPU session for consistency. 

### Why does the sec/completions improve ~4x? Why is this not a linear improvement?

The throughput improves the most from 0.453 down to 0.11, an ~4x drop in cost/completion, as G increases from 4 to 32. Note that interestingly this is not a linear improvement, and each G doubling has the following multiples: 1.53x, then 1.9x and last is 1.4x. My initial guess was that we could see most gains in the first multiple due to batching efficiency that amortizes the fixed costs (kernel launches, prefill etc), and hence are captured in the first doubling. However, G=4 is not enough to saturate the GPU for these optimizations to show up, and hence the sweet spot is in the middle. 


### How does the bottleneck change as G increases and why? How come each doubling lowers its share more and more?

The share of generation falls from 88.7% (G=4) -> 81% -> 69% down to 56% (G=32), whereas the wall clock share by backward (and forward) increases (1.71x to 1.71x to 1.46x, with an overall 4.28x). Note from above that generation benefits from the amortized fixed costs of grouping, but backward (and forward) costs increase almost linearly with the number of generations. This explains why the share of the bottleneck moves from generation to backward/forward, because generations overall get cheaper, but there is no major efficiency gain in the backward/forward pass.  It is worth noting that even for G=32, the majority share of the wall clock is still the  generation phase (but trending downwards). 

### How do stragglers worsen with increased G?

The p99_p50_ratio does increase, but not in a concrete way. The change goes from 1.72 -> 1.8 -> 1.8 -> 1.92, with G=32 having the most increase for stragglers. However, the max metric tells a much clearer picture, as it increases from 2.35 -> 2.24 -> 2.53 -> 4.23, again with G=32 having a dramatically higher max. My earlier reasoning about capturing the long tail to demonstrate the straggler problem wasn't quite right, and that the real signal is actually with the worst case stragglers! 

### How does the overall wall clock increase? Is it in proportion to G?

Wall clock increases by multiples of 1.31x -> 1.05x -> 1.43x. Two things of note here: one is that the jump from G=8 to G=16 is lower than the previous multiple (1.05 < 1.31), while the jump from G=16 to G=32 is the biggest. As mentioned above, the decrease in multiple for the second multiple is largely due to batching efficiency gains, while at the same time the straggler problem is mild. This changes drastically for G=32 with the straggler showing up, and hence pushes generation time, which shows up here in wall clock multiple. 


### Why does the log ratio drift occur more drastically in G=32 but not at all in G=16?

This was a very surprising result as I expected the drift to be more substantially for G= 16 and even more so for G=32. While it was significant for G=32, and moderate for G=8 (from earlier) there was no noticeable drift for G=16 or G= 4. I'm not sure why this is the case, it could be due to my single seed run or it's possible there are certain batch sizes that take a different kernel path (for example, using powers of 2 is a known GPU optimization, perhaps in this case using power of 4 is the differentiator?)

### How noisy are these numbers? Are the multiples trustworthy?

The most significant limitation is that these numbers are done on a relatively small scale and hence they are bound to be quite noisy; and while the multiple themselves are specific to this config of this experiment, the more important takeaways are the reasoning and direction of each of the metrics.

### Was Amdahl's law noticable?

As a general pattern, all the throughput type gains (tokens/sec, wall clock, per completion cost) were sublinear in that each doubling of G got less than <2x change. This is Amdahl's law in spirit -- the generations get optimized, but the un-amortized backward pass's share increases and hence overall gain starts to flatten. 


## Future work & Limitations

### Which other forms of optimization would you profile next?

I would like to create a flowchart/ladder of RL specific optimizations and see how to improve & move the bottlenecks from a naive implementation. By RL specific optimizations, I mean improvements related only to GRPO (in this case). For example, flash attention would not count as a phase since that is a standard training loop optimization, but radix attention would count since that is GRPO specific optimization.

### Further work into the straggler problems

It has been shown in research that truncating the number of generations by first to 6 (out of 8, for example) has a bias for shorter length answers. An interesting next step would be to actually let these 2 unaccounted for generations run in the background, and then do an analysis of what their rewards actually were compared to the rewards of the first 6, and how much exactly do they change the mean and sd, and whether the wall clock gain is worth the tradeoff. Note that this bias is different from the GRPO length bias that is laid out in the Dr.GRPO paper, which is due to per token dilution bias; the above described one comes solely from considering generations with faster timings, and hence shorter lengths (assuming generations all have the same speed). Another interesting aspect of GRPO is that we can have dynamic G based on the prompt's inherent variance, in which case everyone starts with G == 4, but only if the reward variation is below a threshold do we increase G.

### Limitations

All weights are used in bf16 form, instead of the ideal fp32 (for master weights) which are used in real training.
Most of the other limitations come from the small scale of the experiment, that is the low model parameter, single GPU and relatively straightforward dataset.
For example, in real infrastructure for RL, especially for agents and domain specific fields, the rewards are unlikely to be a straightforward calculation. Moreover, with tool use capacity, the generations themselves might not be outputted at the same speed, and the tail end is likely to be significantly higher.
Furthermore, practical scale RL infra also introduces clusters of GPUs to hold GPUs for both training and generation, and therefore we have to take into account the design of those too. For example, using DDP methods such as FSDP or various forms of ZeRO introduces communication costs (ring all reduce, all to all) etc.

## How was AI used?

Design & Analysis is my work, and essentially all of the code is AI written under my review.

## References

- [How to Accurately Time CUDA Kernels in PyTorch](https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch) 
- [Best Practices for Profiling PyTorch Models on GPUs](https://medium.com/@dayashankar.bhakuni/best-practices-for-profiling-pytorch-models-on-gpus-7a791d17e2b9)
- [Keep the Tokens Flowing: Lessons from 16 Open-Source RL Libraries](https://huggingface.co/blog/async-rl-training-landscape)
- [vLLM discussion](https://discuss.vllm.ai/t/difference-in-log-probabilities-between-vllm-and-hf-model-in-identical-environment/183)
- [vLLM V0→V1: Correctness Before Corrections in RL](https://huggingface.co/blog/ServiceNow-AI/correctness-before-corrections)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)
- [Dr.GRPO Paper](https://arxiv.org/abs/2503.20783)

## Appendix

### V2 · max_new_tokens = 128

| bucket | truncated | dtype | mean_lr | mean\|lr\| | max\|lr\| | n |
|---|---|---|---:|---:|---:|---:|
| 0-64    | True  | bf16 |  0.00321 | 0.02609 | 0.29177 | 896 |
| 0-64    | True  | fp32 |  0.00049 | 0.01177 | 0.25294 | 896 |
| 0-64    | False | bf16 | -0.00388 | 0.02628 | 0.26283 | 128 |
| 0-64    | False | fp32 |  0.00515 | 0.01439 | 0.36183 | 128 |
| 64-128  | True  | bf16 |  0.00463 | 0.01894 | 0.27660 | 840 |
| 64-128  | True  | fp32 |  0.00082 | 0.00729 | 0.16176 | 840 |
| 64-128  | False | bf16 |  0.03210 | 0.04428 | 0.25003 | 16 |
| 64-128  | False | fp32 |  0.02190 | 0.03659 | 0.32403 | 16 |
| last3   | True  | bf16 | -0.00074 | 0.01946 | 0.09296 | 42 |
| last3   | True  | fp32 | -0.00072 | 0.01051 | 0.08790 | 42 |
| last3   | False | bf16 | -0.00197 | 0.01436 | 0.03577 | 6 |
| last3   | False | fp32 | -0.00143 | 0.00194 | 0.00566 | 6 |
| final   | True  | bf16 |  0.00515 | 0.00854 | 0.03531 | 14 |
| final   | True  | fp32 |  0.00099 | 0.00105 | 0.00783 | 14 |
| final   | False | bf16 |  0.01235 | 0.02527 | 0.03762 | 2 |
| final   | False | fp32 | -0.00034 | 0.00105 | 0.00139 | 2 |

### V2 · max_new_tokens = 512

| bucket | truncated | dtype | mean_lr | mean\|lr\| | max\|lr\| | n |
|---|---|---|---:|---:|---:|---:|
| 0-64    | True  | bf16 |  0.00636 | 0.02958 | 0.11886 | 64 |
| 0-64    | True  | fp32 | -0.00167 | 0.01107 | 0.07356 | 64 |
| 0-64    | False | bf16 |  0.00436 | 0.02454 | 0.25493 | 897 |
| 0-64    | False | fp32 | -0.00001 | 0.01232 | 0.38305 | 897 |
| 64-128  | True  | bf16 | -0.00193 | 0.02653 | 0.27624 | 64 |
| 64-128  | True  | fp32 | -0.00435 | 0.01704 | 0.10743 | 64 |
| 64-128  | False | bf16 |  0.00442 | 0.01964 | 0.56264 | 838 |
| 64-128  | False | fp32 |  0.00026 | 0.00782 | 0.32450 | 838 |
| 128-256 | True  | bf16 | -0.00068 | 0.03686 | 0.24894 | 128 |
| 128-256 | True  | fp32 |  0.00027 | 0.01642 | 0.14535 | 128 |
| 128-256 | False | bf16 |  0.00322 | 0.01481 | 0.27011 | 1205 |
| 128-256 | False | fp32 | -0.00057 | 0.00690 | 0.21076 | 1205 |
| 256-512 | True  | bf16 |  0.00016 | 0.02489 | 0.17549 | 252 |
| 256-512 | True  | fp32 | -0.00196 | 0.01105 | 0.11694 | 252 |
| 256-512 | False | bf16 | -0.00001 | 0.01676 | 0.24835 | 473 |
| 256-512 | False | fp32 | -0.00211 | 0.00612 | 0.20067 | 473 |
| last3   | True  | bf16 | -0.00425 | 0.00963 | 0.02008 | 3 |
| last3   | True  | fp32 |  0.01940 | 0.04012 | 0.08927 | 3 |
| last3   | False | bf16 |  0.00975 | 0.02432 | 0.17426 | 45 |
| last3   | False | fp32 |  0.00351 | 0.00726 | 0.08525 | 45 |
| final   | True  | bf16 | -0.13527 | 0.13527 | 0.13527 | 1 |
| final   | True  | fp32 |  0.02351 | 0.02351 | 0.02351 | 1 |
| final   | False | bf16 |  0.01037 | 0.02296 | 0.09690 | 15 |
| final   | False | fp32 |  0.01129 | 0.01161 | 0.11730 | 15 |
