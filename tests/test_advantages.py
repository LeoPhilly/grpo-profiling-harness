import math

import pytest
import torch

from grpo.advantages import compute_group_advantages

EPS = 1e-4


def test_hand_computed_two_groups_of_four():
    # Expected values derived with plain Python math, not torch, so the test
    # is an independent ground truth. Sample std (unbiased) convention.
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 1.0, 1.0])

    # Group A: mean 2.5, sample var = (1.5^2 + 0.5^2 + 0.5^2 + 1.5^2) / 3 = 5/3
    std_a = math.sqrt(5.0 / 3.0)
    # Group B: mean 0.5, sample var = (4 * 0.5^2) / 3 = 1/3
    std_b = math.sqrt(1.0 / 3.0)
    expected = torch.tensor(
        [
            (1.0 - 2.5) / (std_a + EPS),
            (2.0 - 2.5) / (std_a + EPS),
            (3.0 - 2.5) / (std_a + EPS),
            (4.0 - 2.5) / (std_a + EPS),
            (0.0 - 0.5) / (std_b + EPS),
            (0.0 - 0.5) / (std_b + EPS),
            (1.0 - 0.5) / (std_b + EPS),
            (1.0 - 0.5) / (std_b + EPS),
        ]
    )

    adv = compute_group_advantages(rewards, group_size=4)
    assert adv.shape == rewards.shape
    assert torch.allclose(adv, expected, atol=1e-6)


def test_all_equal_rewards_give_zero_not_nan():
    # std = 0, so only the +1e-4 epsilon keeps this finite.
    rewards = torch.tensor([3.0, 3.0, 3.0, 3.0, 7.0, 7.0, 7.0, 7.0])
    adv = compute_group_advantages(rewards, group_size=4)
    assert torch.isfinite(adv).all()
    assert torch.equal(adv, torch.zeros(8))


def test_normalization_is_per_group_not_batch():
    # Two groups with the same within-group pattern but wildly different
    # scales. Group-wise normalization gives both groups the identical
    # zero-mean advantage pattern; batch-wise normalization would push all of
    # group A negative and all of group B positive.
    rewards = torch.tensor([0.0, 0.0, 1.0, 1.0, 100.0, 100.0, 101.0, 101.0])
    adv = compute_group_advantages(rewards, group_size=4)

    # Each group is zero-mean on its own (batch norm cannot satisfy this here).
    assert torch.allclose(adv[:4].mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(adv[4:].mean(), torch.tensor(0.0), atol=1e-6)
    # Identical within-group pattern -> identical advantages across groups.
    assert torch.allclose(adv[:4], adv[4:], atol=1e-6)
    # And the exact group-wise values: +/- 0.5 / (sqrt(1/3) + eps).
    v = 0.5 / (math.sqrt(1.0 / 3.0) + EPS)
    expected_block = torch.tensor([-v, -v, v, v])
    assert torch.allclose(adv[:4], expected_block, atol=1e-6)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        compute_group_advantages(torch.ones(7), group_size=4)  # 7 % 4 != 0
    with pytest.raises(ValueError):
        compute_group_advantages(torch.ones(4), group_size=1)
    with pytest.raises(ValueError):
        compute_group_advantages(torch.ones(2, 4), group_size=4)  # not 1-D
