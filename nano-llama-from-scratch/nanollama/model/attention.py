"""
rope + grouped-query attention. the kv-cache path is built into the forward.

if i add other attention variants later (linear, sliding window) they go here with the
same (out, new_cache) return, so the block doesn't care which one it's running.
"""
import math
import torch
import torch.nn as nn


# Rope from Scratch
class RoPE(nn.Module):

    def __init__(self, max_len, head_dim, base=10000):

        super().__init__()

        positions = torch.arange(0, max_len)
        dim_indices = torch.arange(0, head_dim, 2)
        inv_freq = 1/base**(dim_indices/head_dim)

        angles = positions[:, None] * inv_freq[None, :]
        # (T,D)
        angles = torch.cat([angles, angles], dim=-1)

        self.register_buffer('cos', angles.cos())
        self.register_buffer('sin', angles.sin())


    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2,x1], dim=-1)

    def forward(self, q, k, start_pos=None):

        B,H,T,D = q.shape

        cos = self.cos[None, None, start_pos:start_pos + T, :]
        sin = self.sin[None, None, start_pos:start_pos + T, :]

        # Rotmatrix
        # cos(theta) - sin(theta)
        # sin(theta) + cos(theta)

        q_rotated = cos*q + sin*self.rotate_half(q)
        k_rotated = cos*k + sin*self.rotate_half(k)

        return q_rotated, k_rotated


# Grouped Query MHSA from Scratch
# With cache
class GQ_MHSA(nn.Module):

    def __init__(self, d_model, d_embed, q_heads, kv_heads):

        super().__init__()

        # Have q_heads as a * kv_heads where a is an int
        assert d_embed % q_heads == 0
        assert d_embed % kv_heads == 0
        assert q_heads % kv_heads == 0

        self.head_dim = d_embed // q_heads
        self.q_heads = q_heads
        self.kv_heads = kv_heads

        # Each head will be self.head_dim; fewer kv heads so total dim of k,v less
        self.q_proj = nn.Linear(d_model, self.head_dim * q_heads, bias=False)
        self.k_proj = nn.Linear(d_model, self.head_dim * kv_heads, bias=False)
        self.v_proj = nn.Linear(d_model, self.head_dim * kv_heads, bias=False)

        # Project to mix heads
        self.o_proj = nn.Linear(d_embed, d_model)

        # Positional Embeddings
        self.positional_embedding = RoPE(10000, self.head_dim)

    def forward(self, x, kv_cache=None, start_pos=0, causal_masking=True):

        B,T,_ = x.shape
        scale = 1/math.sqrt(self.head_dim)

        # Q: B,T,H*dk -> (B,Hq,T,dk)
        q = self.q_proj(x).view(B,T,self.q_heads,self.head_dim).transpose(1,2)
        # kv_heads is smaller than q_heads
        k = self.k_proj(x).view(B,T,self.kv_heads,self.head_dim).transpose(1,2)
        v = self.v_proj(x).view(B,T,self.kv_heads,self.head_dim).transpose(1,2)

        # Apply positional embedding to Q and K
        q, k = self.positional_embedding(q, k, start_pos)

        if kv_cache is not None:
            # (B,H,T,D)
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_cache = (k,v)

        # Match heads now: Q,K,V are B,Hq,T,dk
        scale_kv_to_q = self.q_heads // self.kv_heads
        # Repeat interleave keeps closer in memory access, e.g.
        # k1k1k1k2k2k2k3k3k3
        k = k.repeat_interleave(scale_kv_to_q, dim=1)
        v = v.repeat_interleave(scale_kv_to_q, dim=1)

        # SA -> var will increase by dk
        scores = torch.matmul(q, k.transpose(-1,-2)) * scale

        if causal_masking:
            Tk = k.shape[2]

            mask_out = torch.triu(
                torch.ones(T, Tk, device=x.device, dtype=torch.bool),
                diagonal=1+(Tk-T))
            scores = scores.masked_fill(mask_out, float('-inf'))

        attn = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        # (T,T) (T,D)
        out = torch.matmul(attn, v)

        # (B,T,H,D)
        out = out.transpose(1,2).contiguous().view(B,T,-1)
        return self.o_proj(out), new_cache
        # note: this returns (out, new_cache). the decoderblock still does
        # `x + self.sa(...)`, which adds a tensor to a tuple and breaks. i unpack it
        # properly in glue #1.
