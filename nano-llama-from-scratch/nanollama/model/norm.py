"""layernorm and rmsnorm, from scratch."""
import torch
import torch.nn as nn


# LayerNorm from Scratch
class LayerNorm(nn.Module):

    def __init__(self, feature_dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(feature_dim))
        self.delta = nn.Parameter(torch.zeros(feature_dim))
        self.eps = eps

    def forward(self, x):
        # x: (B,T,D)

        x_mean = x.mean(dim=-1, keepdim=True) # (B,T,1)
        x_var = (x - x_mean).pow(2).mean(dim=-1, keepdim=True)

        x_norm = (x - x_mean) / torch.sqrt(x_var + self.eps)
        return x_norm * self.gamma + self.delta

class RMSNorm(nn.Module):

    def __init__(self, feature_dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(feature_dim))
        self.eps = eps

    def forward(self, x):
        x_var = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x/torch.sqrt(x_var + self.eps)
        return x_norm*self.gamma
