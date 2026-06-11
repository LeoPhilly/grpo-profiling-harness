"""FakeGenerator: canned completions so the plumbing runs CPU-only on the Mac.

NEVER add realism or features here (CLAUDE.md). It exists to exercise the
train loop's masks, rewards, batching and logging — not to imitate vLLM.
vLLM bugs are debugged on the GPU box from real tracebacks, never here.

Behavior logprobs are random floats, so with this generator the on-policy
logprob identity (standing check #1) is meaningless BY CONSTRUCTION — only
the plumbing that computes and logs it is under test on the Mac.
"""

import torch


class FakeGenerator:
    def __init__(self, tokenizer, completion_tokens=(8, 24), seed=0):
        self.tokenizer = tokenizer
        self.min_tokens, self.max_tokens = completion_tokens
        self.rng = torch.Generator().manual_seed(seed)

    def sync_weights(self, model):
        """No-op. Exists so the train loop calls one seam for both backends."""

    def generate(self, prompts, group_size, ground_truths=None):
        """Flat list of len(prompts) * group_size dicts in group order (each
        prompt's group_size completions consecutive), matching the grouping
        compute_group_advantages expects:
            {"text": str, "token_ids": list[int], "logprobs": list[float]}

        When ground_truths is given, completions at even in-group indices end
        with a correct "#### <gt>" (so ~half of each group scores 1.0).
        """
        filler_ids = self.tokenizer.encode(" step", add_special_tokens=False)
        outs = []
        for i in range(len(prompts)):
            for j in range(group_size):
                if ground_truths is None:
                    answer = "0"
                elif j % 2 == 0:
                    answer = ground_truths[i]
                else:
                    # Appending a digit is never numerically equal to gt.
                    answer = ground_truths[i] + "1"
                suffix_ids = self.tokenizer.encode(
                    f"\n#### {answer}", add_special_tokens=False
                )
                target = int(
                    torch.randint(
                        self.min_tokens, self.max_tokens + 1, (1,), generator=self.rng
                    )
                )
                n_filler = max(0, target - len(suffix_ids))
                token_ids = (filler_ids * n_filler)[:n_filler] + suffix_ids
                logprobs = (
                    -(torch.rand(len(token_ids), generator=self.rng) * 2.9 + 0.1)
                ).tolist()
                outs.append(
                    {
                        "text": self.tokenizer.decode(token_ids),
                        "token_ids": token_ids,
                        "logprobs": logprobs,
                    }
                )
        return outs
