import torch
import torch.nn as nn
import torch.nn.functional as F
from math import factorial
import numpy as np  

class ResidualDialConvBlock(nn.Module):
    def __init__(self, in_dim, dilations):
        super().__init__()
        self.net = nn.Sequential(
            *[ResidualDialConv(in_dim, i) for i in dilations]
        )
    
    def forward(self, inp):
        itermidiate = []
        for layer in self.net:
            inp = layer(inp)
            itermidiate.append(inp)
        return itermidiate

class ResidualDialConv(nn.Module):
    def __init__(self, in_dim, dilation, avg_pool=True):
        super().__init__()

        k_size = 3
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, in_dim, kernel_size=k_size, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_dim, in_dim, kernel_size=1)
        )
        self.un = nn.Unfold(kernel_size=[k_size, 1], dilation=dilation)
        self.k_size = k_size

        self.apply(init_weights)

        if avg_pool:
            w = [1/k_size,] * k_size
        else:
            w = list(range(k_size))
            mid = max(w)/2
            w = [-1 * abs(i - mid) for i in w]
            w = np.exp(w)/sum(np.exp(w))
        w = torch.tensor(w).float()
        self.register_buffer('w', w)
    
    def forward(self, inp):
        B, C, _ = inp.shape
        T = self.k_size
        conv = self.conv(inp) # shape [B, C, S]
        seg = self.un(inp.unsqueeze(-1)).view(B, C, T, -1) # shape [B, C, T, S]
        w = self.w.view(1, 1, -1, 1).expand_as(seg)
        residual = torch.sum(seg * w, dim=2)

        out = conv + residual
        return out