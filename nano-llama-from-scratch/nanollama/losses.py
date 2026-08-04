"""softmax, log-softmax and the causal-lm loss."""
import torch


# Softmax function
def softmax_fun(x, dim=-1):
    x_scaled = x - x.amax(dim=dim, keepdim=True)
    out = x_scaled.exp() / x_scaled.exp().sum(dim=dim, keepdim=True)
    return out

def log_softmax_fun(x, dim=-1):
    # log(e^x / S(e^x)) = log(e^x) - log(S(e^x))
    x_scaled = x - x.amax(dim=dim, keepdim=True)
    # torch.log(x_scaled.exp().sum(dim=dim, keepdim=True))
    out = x_scaled - torch.logsumexp(x_scaled, dim=dim, keepdim=True)
    return out

# LLM loss function
def llm_loss_fun(logits, input_ids, attention_mask, prompt_len):
    # Alignment
    # Trim last token, and remove (prompt-1)
    # P2 P3, R1, R2, R3, ?
    logits = logits[:, prompt_len - 1: -1, :]
    # Remove prompt
    # P1, P2, P3, R1, R2, R3
    input_ids = input_ids[:, prompt_len:]
    response_mask = attention_mask[:, prompt_len:]

    # Get logprobs at value of input_ids
    logprobs = torch.log_softmax(logits, dim=-1).gather(
        dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)

    loss_per_tok = -1 * (logprobs * response_mask)
    loss = loss_per_tok.sum(dim=-1) / response_mask.sum(dim=-1).clamp(min=1)
    return loss.mean()
