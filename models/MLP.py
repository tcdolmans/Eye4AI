import torch.nn as nn
from embedders import ETPatchEmbed


class ETMLP(nn.Module):
    """
    MLP for ET data. Simple implementation so serve as a baseline.
    """
    def __init__(self, config):
        super().__init__()
        self.patch_embed = ETPatchEmbed(
                    in_channels=config["et_dim"],
                    embed_dim=config["embed_dim"],
                    kernel_size=config["et_patch_size"],
                    stride=config["et_stride"]
                    )
        self.num_classes = config["num_classes"]
        self.dropout = nn.Dropout(config["dropout"])
        self.mode = config["mode"]
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(config["embed_dim"]**2, 2048)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(2048, 2048)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(2048, 1024)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(1024, config["num_classes"])
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout(self.fc4(x))
        x = self.relu4(x)
        return x
