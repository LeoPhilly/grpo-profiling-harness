"""GSM8K loading: (prompt + PROMPT_SUFFIX, ground_truth) pairs.

GSM8K is the only training set and the only source of quality numbers
(the GSM8K / SynthLen discipline wall lives in CLAUDE.md).
"""

from datasets import load_dataset

from grpo.rewards.gsm8k import PROMPT_SUFFIX

DATASET_NAME = "openai/gsm8k"
DATASET_CONFIG = "main"

# Fixed held-out eval slice: the first 256 examples of the *test* split.
# All quality claims (pass@1) use exactly this slice; it is never trained on.
HELDOUT_EVAL_SLICE = slice(0, 256)


def extract_ground_truth(answer: str) -> str:
    """GSM8K reference answers end with '#### <number>'. Commas are stripped
    so the ground truth matches what the strict scorer extracts."""
    return answer.split("####")[-1].strip().replace(",", "")


def format_prompt(question: str) -> str:
    """The single place PROMPT_SUFFIX is attached to a question."""
    return question.strip() + PROMPT_SUFFIX


def render_prompt(tokenizer, question: str) -> list:
    """question -> prompt token ids. The ONLY place prompt text is tokenized:
    these ids flow through generation and batch building unchanged. A second
    tokenization anywhere (trainer or vLLM) can shift the prompt length and
    misalign behavior logprobs, poisoning standing check #1.

    Chat models get their template (user role, add_generation_prompt=True);
    template-less tokenizers (tiny-gpt2 on the Mac) get a plain encode of
    the same content.
    """
    content = format_prompt(question)
    if tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=True,
        )
    return tokenizer.encode(content, add_special_tokens=False)


def gsm8k_pairs(split: str = "train") -> list:
    """List of (question, ground_truth) for a split. Questions are raw —
    PROMPT_SUFFIX is applied later, inside render_prompt, exactly once.
    Network/cache access."""
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split)
    return [(ex["question"], extract_ground_truth(ex["answer"])) for ex in ds]


def heldout_eval_pairs() -> list:
    return gsm8k_pairs("test")[HELDOUT_EVAL_SLICE]


def iter_prompt_batches(pairs, prompts_per_batch: int):
    """Yield consecutive full batches; a trailing partial batch is dropped so
    every step has identical shape (comparison axis is equal gradient steps)."""
    for i in range(0, len(pairs) - prompts_per_batch + 1, prompts_per_batch):
        yield pairs[i : i + prompts_per_batch]
