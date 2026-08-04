"""one decoder block: pre-norm, attention then swiglu, both with residuals."""
import torch.nn as nn

from .norm import LayerNorm, RMSNorm  # noqa: F401  (rmsnorm handy if i switch norms later)
from .attention import GQ_MHSA
from .ffn import SwiGLU


class DecoderBlock(nn.Module):

    def __init__(self, in_features):
        super().__init__()
        self.norm1 = LayerNorm(in_features)
        self.norm2 = LayerNorm(in_features)

        self.sa = GQ_MHSA(in_features, in_features, 32, 32)
        self.mlp = SwiGLU(in_features)

    def forward(self, x):

        x = x + self.sa(self.norm1(x), causal_masking=True)
        x = x + self.mlp(self.norm2(x))
        return x


# TODO glue #1 of 4: thread the kv-cache through the block.
# forward has to pass kv_cache + start_pos into self.sa and unpack the (out, new_cache)
# tuple it returns. right now `x + self.sa(...)` adds a tensor to a tuple and blows up,
# so nothing runs until this is done. the return_cache flag keeps the plain training path
# (llama.forward calls blk(x)) returning just x, and lets generate ask for the cache.
# grep "TODO glue" to find all 4. checklist in GLUE.md.
#
# def forward(self, x, kv_cache=None, start_pos=0, return_cache=False):
#     attn_out, new_cache = self.sa(self.norm1(x), kv_cache=kv_cache,
#                                   start_pos=start_pos, causal_masking=True)
#     x = x + attn_out
#     x = x + self.mlp(self.norm2(x))
#     return (x, new_cache) if return_cache else x
