"""
Vision Transformer
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from Vision.utils import DEVICE


class PatchEmbedding(nn.Module):

    def __init__(self, in_channels, patch_size, embed_dim, num_patchs, dropout) -> None:
        super().__init__()
        self.patcher = nn.Sequential(nn.Conv2d(in_channels, embed_dim, patch_size, patch_size), nn.Flatten(2))
        self.cls_token = nn.Parameter(torch.randn((1, 1, embed_dim), requires_grad=True))
        self.position_embedding = nn.Parameter(torch.randn(size=(1, num_patchs + 1, embed_dim), requires_grad=True))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = self.patcher(x).permute(0, 2, 1)
        x = torch.cat([cls_token, x], dim=2)
        x = x + self.position_embedding
        x = self.dropout(x)
        return x


class Vit(nn.Module):
    def __init__(
        self, in_channels, patch_size, embed_dim, num_patchs, dropout, num_heads, activation, num_encoders, num_classes
    ) -> None:
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embed_dim, num_patchs, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            embed_dim, num_heads, dropout=dropout, activation=activation, batch_first=True, norm_first=True
        )
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)
        self.mlp = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.encoder_blocks(x)
        x = self.mlp(x[:, 0, :])
        return x


def main():
    vit = Vit(224, 16, 128, 20, True, 1, F.relu, 4, 10).to(DEVICE)
    x = torch.randn(1, 3, 224, 224).to(DEVICE)
    out: Tensor = vit(x)
    print(out.shape)


if __name__ == "__main__":
    main()
