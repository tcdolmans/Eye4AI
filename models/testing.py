import torch
import torch.nn as nn
from embedders import ETPatchEmbed, ImagePatchEmbed, SemanticEmbedding


class ETPatchEmbed(nn.Module):
    def __init__(self, in_channels=3, embed_dim=768, kernel_size=15, stride=1, padding=7):
        super().__init__()
        self.norm = nn.AdaptiveAvgPool1d(embed_dim)
        self.projection = nn.Conv1d(in_channels=in_channels,
                                    out_channels=embed_dim,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding)

    def forward(self, x):
        print(f"Input shape: {x.shape}")
        x = self.projection(x)
        print(f"Shape after projection: {x.shape}")
        x = self.norm(x)
        print(f"Shape after normalization: {x.shape}")
        x = x.transpose(1, 2)
        print(f"Output shape: {x.shape}")
        return x


# Test case
device = 'cuda' if torch.cuda.is_available() else 'cpu'
et_patch_embed = ETPatchEmbed(in_channels=4, embed_dim=192, kernel_size=15, stride=1, padding=7).to(device)
img_patch_embed = ImagePatchEmbed(in_channels=3, embed_dim=192, patch_size=25, stride=12, padding=12).to(device)
sem_patch_embed = SemanticEmbedding(in_channels=12, embed_dim=192, patch_size=25, stride=12, padding=12).to(device)
et_data = torch.randn(8, 4, 300).to(device)
img_data = torch.randn(8, 3, 600, 800).to(device)
sem_data = torch.randn(8, 12, 600, 800).to(device)
et_embed = et_patch_embed(et_data)
img_embed = img_patch_embed(img_data)
sem_embed = sem_patch_embed(sem_data)
print(et_embed.shape, img_embed.shape, sem_embed.shape)
