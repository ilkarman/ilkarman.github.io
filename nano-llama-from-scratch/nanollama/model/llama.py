"""the model: token embedding, N decoder blocks, rmsnorm, tied lm head."""
import torch
import torch.nn as nn

from .block import DecoderBlock
from .norm import RMSNorm
from .sampling import sample_topk_topp  # noqa: F401  (used by generate below)


class LLama(nn.Module):

    def __init__(self, vocab_size, d_model, n_layers, n_heads, n_kv_heads):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight # Weight tying for small model

    def forward(self, idx):

        x = self.embed(idx) # Embed tokenised word

        # Push through the SA and MLP blocks
        for blk in self.blocks:
            x = blk(x)

        # LM head
        out = self.lm_head(self.norm(x))
        return out


# TODO glue #2 of 4: a cache-aware forward so generate can feed one token at a time.
# needs glue #1 first, so blk(..., return_cache=True) gives back (x, cache).
#
# def _forward_cached(self, idx, caches, start_pos):
#     x = self.embed(idx)
#     new_caches = []
#     for blk, cache in zip(self.blocks, caches):
#         x, cache = blk(x, kv_cache=cache, start_pos=start_pos, return_cache=True)
#         new_caches.append(cache)
#     return self.lm_head(self.norm(x)), new_caches
#
#
# TODO glue #3 of 4: the generate loop. prefill the prompt once, then sample one token at
# a time off the cache. i already have sample_topk_topp and the kv-cache so this is just
# the loop. in rollout i sample GROUP_SIZE completions from one prompt, so every row has
# the same prompt length and i don't need a padding mask here.
#
# @torch.no_grad()
# def generate(self, idx, max_new_tokens, temperature=1.0, top_k=0, top_p=1.0,
#              eos_id=None, num_return_sequences=1):
#     self.eval()
#     if num_return_sequences > 1:
#         idx = idx.repeat_interleave(num_return_sequences, dim=0)   # (G, T_prompt)
#     B, T = idx.shape
#     caches = [None] * len(self.blocks)
#     logits, caches = self._forward_cached(idx, caches, start_pos=0)   # prefill
#     start_pos = T
#     finished = torch.zeros(B, dtype=torch.bool, device=idx.device)
#     for _ in range(max_new_tokens):                                   # decode
#         next_tok = sample_topk_topp(logits[:, -1, :], temperature, top_k, top_p)  # (B,1)
#         if eos_id is not None:
#             next_tok[finished] = eos_id
#         idx = torch.cat([idx, next_tok], dim=1)
#         if eos_id is not None:
#             finished |= (next_tok.squeeze(-1) == eos_id)
#             if finished.all():
#                 break
#         logits, caches = self._forward_cached(next_tok, caches, start_pos=start_pos)
#         start_pos += 1
#     return idx
