"""sampling from logits: temperature, top-k, top-p."""
import torch


# Sample-Logits from Scratch
def sample_topk_topp(logits, temperature=1.0, top_k=0, top_p=1.0):
    # logits: (N*T,D)
    # Temp > 1 will make distribution flatter
    logits = logits / max(temperature, 1e-5)

    if top_k > 0:
        # (B,1)
        top_k_val = logits.topk(k=top_k, dim=-1).values[:,-1]
        top_k_val = top_k_val.unsqueeze(-1)
        # Zero-out post-softmax
        logits = logits.masked_fill(logits < top_k_val, float('-inf'))

    if top_p < 1.0:
        sorted_logit_val, sorted_logit_indx = logits.sort(descending=True)
        proba_val = torch.softmax(sorted_logit_val, dim=-1)
        # p = 0.6, 0.2, 0.1
        # c = 0.6, 0.8, 0.9
        # d = 0, 0.6, 0.8
        # e.g. top_p is 0.7
        p_cumsum = proba_val.cumsum(dim=-1)
        mask = (p_cumsum - proba_val) > top_p

        # (B,D)
        sorted_logit_val = sorted_logit_val.masked_fill(mask, float('-inf'))
        logits = torch.empty_like(logits).scatter(dim=-1, src=sorted_logit_val, index=sorted_logit_indx)

    proba = torch.softmax(logits, dim=-1)
    return torch.multinomial(proba, 1)
