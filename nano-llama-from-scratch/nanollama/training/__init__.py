"""training objectives + rollout. new objectives (opd, dpo, sft) go here as siblings."""
from .grpo import get_grpo_advantage, get_ppo_loss, get_kl_loss, get_grpo_loss
from .rollout import chat_prompt, calculate_reward, get_logprobs, rollout

__all__ = [
    "get_grpo_advantage", "get_ppo_loss", "get_kl_loss", "get_grpo_loss",
    "chat_prompt", "calculate_reward", "get_logprobs", "rollout",
]
