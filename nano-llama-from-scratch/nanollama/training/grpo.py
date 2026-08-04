"""grpo loss: group-normalised advantages, clipped ppo surrogate, kl penalty (k1/k2/k3).

new objectives (opd, dpo, sft) can sit next to this file and reuse rollout.get_logprobs.
"""
import torch


def get_grpo_advantage(rewards, std_normalise=True, eps=1e-6):
    assert rewards.shape[-1] > 1, "Groups need more than one"
    rewards_mean = rewards.mean(dim=-1, keepdim=True)
    rewards_std = rewards.std(dim=-1, keepdim=True, unbiased=True)
    return (rewards - rewards_mean) / (rewards_std + eps)

def get_ppo_loss(advantage, logprob, logprob_old, mask, clip=0.2):
    log_ratio = logprob - logprob_old.detach()
    ratio = torch.clamp(log_ratio, -20, 20).exp() # cannot overflow to inf
    unclipped = -advantage * ratio
    clipped = -advantage * torch.clamp(ratio, 1-clip, 1+clip)
    per_token_loss = torch.maximum(unclipped, clipped)
    return per_token_loss


def get_kl_loss(logprob, logprob_ref, estimator='k3'):
    log_ratio = logprob - logprob_ref
    if estimator == 'k1': return log_ratio                      # unbiased, noisy, signed
    if estimator == 'k2': return 0.5 * log_ratio.pow(2)         # biased, low variance, >= 0
    if estimator == 'k3': return log_ratio + (-log_ratio).exp() - 1    # unbiased, >= 0
    raise ValueError(estimator)


def get_grpo_loss(advantage, logprob, logprob_old, logprob_ref, mask, num_seqs,
                  beta=0.0, clip=0.2):   # (added the missing colon here)

    per_token_loss = get_ppo_loss(advantage, logprob, logprob_old, mask, clip)
    kl = get_kl_loss(logprob, logprob_ref) if beta else torch.zeros_like(logprob)
    per_token_loss = per_token_loss + beta * kl
    # GRPO reduction i.e. lr per token gets diluted
    per_seq_loss = (per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1)
    loss = per_seq_loss.sum() / num_seqs
    return loss
