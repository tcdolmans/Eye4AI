import torch.nn as nn


class ETPatchEmbed(nn.Module):
    """Takes a batch of ET data and returns their embeddings."""
    def __init__(self,
                 in_channels=4,
                 embed_dim=768,
                 kernel_size=15,
                 stride=1,
                 padding=7):
        super().__init__()
        self.norm = nn.AdaptiveAvgPool1d(embed_dim)
        self.projection = nn.Conv1d(in_channels=in_channels,
                                    out_channels=embed_dim,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding)

    def forward(self, x):
        x = self.projection(x)
        x = self.norm(x)
        return x


class ImagePatchEmbed(nn.Module):
    """Takes batch of images and returns their embeddings."""
    def __init__(self,
                 in_channels=3,
                 embed_dim=768,
                 patch_size=25,
                 stride=1,
                 padding=12):
        super().__init__()
        self.patch_size = patch_size
        self.norm = nn.AdaptiveAvgPool2d(embed_dim)
        self.projection = nn.Conv2d(in_channels=in_channels,
                                    out_channels=embed_dim,
                                    kernel_size=patch_size,
                                    stride=stride,
                                    padding=padding)

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2)
        x = self.norm(x)
        return x


class SemanticEmbedding(nn.Module):
    """Takes batch of semantic labels and returns their embeddings."""
    def __init__(self,
                 in_channels=12,
                 embed_dim=768,
                 patch_size=25,
                 stride=1,
                 padding=12):
        super().__init__()
        self.patch_size = patch_size
        self.norm = nn.AdaptiveAvgPool2d(embed_dim)
        self.projection = nn.Conv2d(in_channels=in_channels,
                                    out_channels=embed_dim,
                                    kernel_size=patch_size,
                                    stride=stride,
                                    padding=padding)

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2)
        x = self.norm(x)
        return x
