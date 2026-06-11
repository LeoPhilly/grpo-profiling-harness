import importlib

import pytest
from transformers import AutoTokenizer

from grpo.rewards.gsm8k import gsm8k_reward
from grpo.rollout.fake_generator import FakeGenerator

TINY_MODEL = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TINY_MODEL)


def test_fake_generator_shapes_grouping_and_alignment(tokenizer):
    gen = FakeGenerator(tokenizer, completion_tokens=(8, 16), seed=1)
    prompts = ["a?", "b?", "c?"]
    ground_truths = ["7", "13", "240"]
    outs = gen.generate(prompts, group_size=4, ground_truths=ground_truths)

    assert len(outs) == 12  # len(prompts) * G, group order
    for out in outs:
        assert len(out["token_ids"]) == len(out["logprobs"])
        assert 8 <= len(out["token_ids"]) <= 16
        assert all(lp < 0 for lp in out["logprobs"])


def test_fake_generator_half_correct_under_real_reward(tokenizer):
    gen = FakeGenerator(tokenizer, completion_tokens=(8, 16), seed=2)
    prompts = ["a?", "b?"]
    ground_truths = ["7", "13"]
    group_size = 4
    outs = gen.generate(prompts, group_size, ground_truths=ground_truths)
    # Even in-group indices are correct: exactly 2 of each group of 4 score
    # 1.0 under the REAL reward function (so the integration test's reward
    # signal is known, not assumed).
    for g, gt in enumerate(ground_truths):
        group = outs[g * group_size : (g + 1) * group_size]
        scores = [gsm8k_reward(out["text"], gt) for out in group]
        assert scores == [1.0, 0.0, 1.0, 0.0]


def test_vllm_generator_unimportable_on_mac():
    # The only vLLM thing verifiable on the Mac: the import guard's message.
    with pytest.raises(ImportError, match="GPU box"):
        importlib.import_module("grpo.rollout.vllm_generator")
