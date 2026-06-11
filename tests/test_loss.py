import math

import pytest
import torch

from grpo.loss import grpo_loss


def _hand_example():
    """B=1, T=4, V=5. Tokens [0,1,2,3]; positions 2,3 are the completion.

    In the shifted frame the contributing predictions are:
      logits[0, 1] -> token 2 (first completion token)
      logits[0, 2] -> token 3
    logits[0, 0] predicts token 1 (prompt, masked out); logits[0, 3] predicts
    past the end of the sequence. Both rows are filled with garbage on purpose.
    """
    input_ids = torch.tensor([[0, 1, 2, 3]])
    completion_mask = torch.tensor([[0, 0, 1, 1]])
    logits = torch.zeros(1, 4, 5)
    logits[0, 0] = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])  # garbage, must not matter
    logits[0, 1] = torch.tensor([0.0, 0.0, 2.0, 0.0, 0.0])
    logits[0, 2] = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])
    logits[0, 3] = torch.tensor([9.0, 9.0, 9.0, 9.0, 9.0])  # garbage, must not matter
    advantages = torch.tensor([2.0])
    return logits, input_ids, completion_mask, advantages


def test_hand_computed_loss():
    logits, input_ids, completion_mask, advantages = _hand_example()

    # Plain-math ground truth, independent of torch:
    # logprob of token 2 from row [0,0,2,0,0]: 2 - log(4*e^0 + e^2)
    lp1 = 2.0 - math.log(4.0 + math.exp(2.0))
    # logprob of token 3 from row [0,0,0,1,0]: 1 - log(4*e^0 + e^1)
    lp2 = 1.0 - math.log(4.0 + math.exp(1.0))
    # loss = -(adv * (lp1 + lp2)) / num_completion_tokens = -(2*(lp1+lp2))/2
    expected = -(lp1 + lp2)

    loss = grpo_loss(logits, input_ids, completion_mask, advantages)
    assert loss.shape == ()
    assert math.isclose(loss.item(), expected, abs_tol=1e-6)


def test_token_shift_alignment():
    logits, input_ids, completion_mask, advantages = _hand_example()
    base = grpo_loss(logits, input_ids, completion_mask, advantages)

    # (a) Last position predicts past the end of the sequence: no effect.
    l = logits.clone()
    l[0, 3] += 7.0
    assert torch.equal(grpo_loss(l, input_ids, completion_mask, advantages), base)

    # (b) Position 0 predicts token 1, a prompt token (masked): no effect.
    l = logits.clone()
    l[0, 0] += 7.0
    assert torch.equal(grpo_loss(l, input_ids, completion_mask, advantages), base)

    # (c) Position 1 predicts token 2, the first completion token: must change.
    l = logits.clone()
    l[0, 1, 0] += 1.0
    changed = grpo_loss(l, input_ids, completion_mask, advantages)
    assert abs(changed.item() - base.item()) > 1e-4


def test_prompt_tokens_contribute_exactly_zero():
    logits, input_ids, completion_mask, advantages = _hand_example()
    base = grpo_loss(logits, input_ids, completion_mask, advantages)

    # Double the prompt with garbage tokens (mask=0) and garbage logits at the
    # new positions; copy the original rows verbatim after them.
    ext_ids = torch.tensor([[4, 4, 0, 1, 2, 3]])
    ext_mask = torch.tensor([[0, 0, 0, 0, 1, 1]])
    ext_logits = torch.full((1, 6, 5), 3.3)
    ext_logits[0, 2:] = logits[0]

    ext = grpo_loss(ext_logits, ext_ids, ext_mask, advantages)
    assert torch.equal(ext, base)  # bit-identical, not just close


def test_zero_advantage_gives_zero_loss():
    logits, input_ids, completion_mask, _ = _hand_example()
    loss = grpo_loss(logits, input_ids, completion_mask, torch.tensor([0.0]))
    assert loss.item() == 0.0


def test_invalid_inputs_raise():
    logits, input_ids, completion_mask, advantages = _hand_example()
    with pytest.raises(ValueError):
        grpo_loss(logits, input_ids, torch.zeros_like(completion_mask), advantages)
    # Mask only on position 0: token 0 is never a prediction target in the
    # shifted frame, so this also selects no tokens and must refuse loudly.
    with pytest.raises(ValueError):
        grpo_loss(logits, input_ids, torch.tensor([[1, 0, 0, 0]]), advantages)
    with pytest.raises(ValueError):
        grpo_loss(logits, input_ids, completion_mask, torch.tensor([1.0, 2.0]))
    with pytest.raises(ValueError):
        grpo_loss(logits, input_ids[:, :3], completion_mask, advantages)
