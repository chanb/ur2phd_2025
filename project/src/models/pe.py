import math
import torch

from torch import nn


class NoPE(nn.Module):
    """
    No positional encoding (identity).
    """
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class SinusoidalPE(nn.Module):
    """
    Sinusoidal Positional Encoding from Vaswani et al. (2017).
    Implementation from: https://stackoverflow.com/questions/77444485/using-positional-encoding-in-pytorch
    Reference: Vaswani et al. (2017) https://arxiv.org/pdf/1706.03762
    """
    def __init__(self, embed_dim: int, dropout: float = 0.1, max_len: int = 10, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(1, max_len, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
