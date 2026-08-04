"""swiglu feed-forward."""
import torch
import torch.nn as nn


# SwiGLU + pre-norm decoder
class SwiGLU(nn.Module):

    def __init__(self, in_features, expansion=4):
        super().__init__()
        out_features = int(in_features*expansion*2/3) # We have three matricies now not 2
        self.up_proj = nn.Linear(in_features, out_features, bias=False)
        self.gate_proj = nn.Linear(in_features, out_features, bias=False)
        self.down_proj = nn.Linear(out_features, in_features, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        return self.down_proj(torch.sigmoid(gate) * gate * self.up_proj(x))
