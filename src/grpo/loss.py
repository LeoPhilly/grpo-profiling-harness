import torch
import torch.nn.functional as F


def grpo_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    """Plain REINFORCE-style GRPO loss (on-policy, no ratio/clip).

    logits (B, T, V) from a forward pass on full prompt+completion sequences;
    input_ids (B, T); completion_mask (B, T), 1 on completion tokens in the
    *input frame*; advantages (B,) per sequence.

    Token shift: logits at position t predict the token at t+1, so targets are
    input_ids[:, 1:] scored by logits[:, :-1], and the mask is aligned to the
    targets as completion_mask[:, 1:]. The first completion token (predicted
    from the last prompt position) is therefore included, as it should be.

    loss = -(advantage * token_logprob * mask).sum() / mask.sum()
    """
    B, T, V = logits.shape
    if input_ids.shape != (B, T):
        raise ValueError(f"input_ids shape {tuple(input_ids.shape)} != {(B, T)}")
    if completion_mask.shape != (B, T):
        raise ValueError(
            f"completion_mask shape {tuple(completion_mask.shape)} != {(B, T)}"
        )
    if advantages.shape != (B,):
        raise ValueError(f"advantages shape {tuple(advantages.shape)} != {(B,)}")

    targets = input_ids[:, 1:]
    mask = completion_mask[:, 1:].to(logits.dtype)
    denom = mask.sum()
    if denom == 0:
        # 0/0 would silently feed NaN into a training run; refuse instead.
        raise ValueError("completion_mask selects no tokens in the shifted frame")

    logprobs = F.log_softmax(logits[:, :-1, :], dim=-1)
    token_logprob = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return -(advantages[:, None] * token_logprob * mask).sum() / denom
