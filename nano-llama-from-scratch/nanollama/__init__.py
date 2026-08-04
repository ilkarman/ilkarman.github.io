"""
nano llama from scratch: a small llama-style lm and a grpo trainer, all my own code.

layout:
    nanollama/config.py       hyperparams + device
    nanollama/losses.py       softmax / log_softmax / causal-lm loss
    nanollama/data.py         collate_fun (padding + mask + labels)
    nanollama/model/          rope, gqa, norms, swiglu, block, llama, sampling
    nanollama/training/       grpo losses, rollout / reward / logprobs
    scripts/train_grpo.py     the training loop

four small glue functions are still mine to write (grep "TODO glue", see GLUE.md).
"""
from .model import LLama

__all__ = ["LLama"]
