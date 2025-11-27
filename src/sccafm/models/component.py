import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) * (1.0 / (x.shape[-1] ** 0.5))
        return self.weight * x / (norm + self.eps)


class SwiGLU(nn.Module):
    def __init__(self, dim_in, dim_hidden):
        super().__init__()
        self.w1 = nn.Linear(dim_in, dim_hidden)
        self.w2 = nn.Linear(dim_in, dim_hidden)
        
        self.proj_weight = nn.Parameter(torch.zeros(dim_in, dim_hidden))  # (out, in)
        self.proj_bias = nn.Parameter(torch.zeros(dim_in))

        nn.init.normal_(self.proj_weight, mean=0.0, std=1e-6)

    def forward(self, x):
        hidden = F.silu(self.w1(x)) * self.w2(x)
        return hidden @ self.proj_weight.T + self.proj_bias
