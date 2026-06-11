"""vLLM-backed generator. GPU box ONLY — this module must never import on the Mac.

Every vLLM API call below was written without being able to run vLLM and is
marked # VERIFY-ON-GPU. Validate each against the real installed version on
the GPU box (run scripts/smoke_test.py --generator vllm first); fix from real
tracebacks pasted into the conversation, never by guessing and never by
editing FakeGenerator. Pin the installed vllm version in the GPU box's
requirements when it is first installed.
"""

try:
    from vllm import LLM, SamplingParams  # VERIFY-ON-GPU: import path
    from vllm.inputs import TokensPrompt  # VERIFY-ON-GPU: moved across versions
except ImportError as e:
    raise ImportError(
        "vllm is not installed. VLLMGenerator only runs on the GPU box; "
        "on the Mac use FakeGenerator (vLLM/CUDA are never installed here)."
    ) from e


class VLLMGenerator:
    """Same interface as FakeGenerator: generate(prompt_token_ids, group_size,
    ground_truths=None) -> flat list of {"text", "token_ids", "logprobs"}
    in group order. Prompts arrive as TOKEN IDS (rendered once upstream by
    render_prompt) — vLLM must never re-tokenize text, or prompt lengths can
    shift and misalign behavior logprobs (standing check #1). ground_truths
    is accepted and ignored (the real model must never see them)."""

    def __init__(
        self,
        model_name: str,
        max_tokens: int = 256,
        gpu_memory_utilization: float = 0.45,
        temperature: float = 1.0,
        seed: int = 0,
    ):
        self.max_tokens = max_tokens
        self.temperature = temperature
        # VERIFY-ON-GPU: LLM constructor kwargs
        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            seed=seed,
        )

    def sync_weights(self, model):
        """Push current trainer weights into the vLLM engine so generation is
        on-policy. Without this, standing check #1 fails from step 2 onward."""
        # VERIFY-ON-GPU: the weight-sync path differs across vllm versions
        runner = self.llm.llm_engine.model_executor.driver_worker.model_runner  # VERIFY-ON-GPU
        runner.model.load_weights(model.state_dict().items())  # VERIFY-ON-GPU

    def generate(self, prompt_token_ids, group_size, ground_truths=None):
        # VERIFY-ON-GPU: SamplingParams kwargs (logprobs=0 means "chosen token only")
        params = SamplingParams(
            n=group_size,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            logprobs=0,
        )
        # VERIFY-ON-GPU: TokensPrompt carries pre-tokenized prompts verbatim
        inputs = [TokensPrompt(prompt_token_ids=ids) for ids in prompt_token_ids]
        request_outputs = self.llm.generate(inputs, params)  # VERIFY-ON-GPU
        outs = []
        # VERIFY-ON-GPU: one RequestOutput per prompt, input order preserved
        for req in request_outputs:
            assert len(req.outputs) == group_size  # VERIFY-ON-GPU
            for comp in req.outputs:
                token_ids = list(comp.token_ids)  # VERIFY-ON-GPU
                logprobs = [
                    pos[tid].logprob  # VERIFY-ON-GPU: dict token_id -> Logprob
                    for tid, pos in zip(token_ids, comp.logprobs)
                ]
                outs.append(
                    {"text": comp.text, "token_ids": token_ids, "logprobs": logprobs}
                )
        assert len(outs) == len(prompts) * group_size
        return outs
