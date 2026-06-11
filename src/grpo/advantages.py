import torch


def compute_group_advantages(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    """Per-group normalized advantages: (r - group_mean) / (group_std + 1e-4).

    rewards has shape (B,) with B = num_groups * group_size; each consecutive
    block of group_size entries is one group (completions of the same prompt).
    Std is the sample std (unbiased=True, torch's default) — the tests'
    hand-computed values assume this convention.
    """
    if rewards.dim() != 1:
        raise ValueError(f"rewards must be 1-D, got shape {tuple(rewards.shape)}")
    if group_size < 2:
        # With one completion per prompt the group std is undefined (NaN under
        # the unbiased estimator) and GRPO has no baseline; refuse loudly.
        raise ValueError(f"group_size must be >= 2, got {group_size}")
    if rewards.numel() % group_size != 0:
        raise ValueError(
            f"batch size {rewards.numel()} not divisible by group_size {group_size}"
        )
    grouped = rewards.reshape(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True)
    return ((grouped - mean) / (std + 1e-4)).reshape(-1)
