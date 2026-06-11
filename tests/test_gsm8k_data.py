"""Unit tests for the pure parts of the data module (no network — the actual
hub load is verified by running scripts/smoke_test.py)."""

import pytest
from transformers import AutoTokenizer

from grpo.data.gsm8k_data import (
    HELDOUT_EVAL_SLICE,
    extract_ground_truth,
    format_prompt,
    iter_prompt_batches,
    render_prompt,
)
from grpo.rewards.gsm8k import PROMPT_SUFFIX


@pytest.fixture()
def tiny_tokenizer():
    return AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")


def test_extract_ground_truth():
    answer = "Natalia sold 48/2 = <<48/2=24>>24 clips.\n#### 72"
    assert extract_ground_truth(answer) == "72"
    assert extract_ground_truth("reasoning...\n#### 1,234") == "1234"
    assert extract_ground_truth("#### -5") == "-5"


def test_format_prompt_appends_suffix():
    prompt = format_prompt("  What is 2+2?  ")
    assert prompt == "What is 2+2?" + PROMPT_SUFFIX


def test_iter_prompt_batches_full_batches_only():
    pairs = [(f"p{i}", str(i)) for i in range(5)]
    batches = list(iter_prompt_batches(pairs, 2))
    assert batches == [pairs[0:2], pairs[2:4]]  # trailing partial dropped


def test_heldout_slice_is_fixed():
    assert HELDOUT_EVAL_SLICE == slice(0, 256)


def test_render_prompt_fallback_matches_plain_encode(tiny_tokenizer):
    # tiny-gpt2 has no chat template: render must be byte-identical to the
    # plain encode of question + PROMPT_SUFFIX (the pre-existing behavior).
    assert tiny_tokenizer.chat_template is None
    question = "What is 2+2?"
    ids = render_prompt(tiny_tokenizer, question)
    expected = tiny_tokenizer.encode(
        format_prompt(question), add_special_tokens=False
    )
    assert ids == expected
    assert len(ids) > 0
    assert all(isinstance(t, int) for t in ids)


def test_render_prompt_uses_chat_template_when_present(tiny_tokenizer):
    # No chat model can be downloaded on the Mac, so pin the *mechanics* with
    # a minimal template: user content present, generation prompt appended.
    # The real Qwen template renders only on the GPU box (VERIFY-ON-GPU).
    tiny_tokenizer.chat_template = (
        "{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}"
        "{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}"
    )
    question = "What is 2+2?"
    ids = render_prompt(tiny_tokenizer, question)
    text = tiny_tokenizer.decode(ids)
    assert text.startswith("<user>What is 2+2?")
    assert PROMPT_SUFFIX in text
    assert text.endswith("<assistant>")
