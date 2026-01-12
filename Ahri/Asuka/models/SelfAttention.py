"""
Self-Attention 注意力机制
"""

from torch import Tensor, nn


class SelfAttention(nn.Module):

    def __init__(self, dim: int, dk: int, dv: int) -> None:
        super().__init__()
        self.scale = dk**-0.5
        self.q = nn.Linear(dim, dk)  # queries
        self.k = nn.Linear(dim, dk)  # keys
        self.v = nn.Linear(dim, dv)  # values

    def forward(self, x):
        q: Tensor = self.q(x)
        k: Tensor = self.k(x)
        v: Tensor = self.v(x)

        attn: Tensor = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        return out
