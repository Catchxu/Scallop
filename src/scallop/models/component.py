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
        self.w3 = nn.Linear(dim_hidden, dim_in)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
