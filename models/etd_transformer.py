import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from pipeline.utils import list_files # noqa
from pipeline.dataset_constructor import return_data_loaders, replace_nans # noqa


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ETPatchEmbed(nn.Module):
    """
    Takes a 1D convolution of a patch of ET data to embed in a lower dimensional space.
    """
    def __init__(self, sample_height=300, sample_width=3,
                 patch_size=15, in_chans=64, embed_dim=15):
        super().__init__()
        sample_size = tuple((sample_height, sample_width))
        patch_size = tuple((patch_size, sample_width))
        num_patches = (sample_size[1] // patch_size[1] * sample_size[0] // patch_size[0])
        self.sample_size = sample_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.conv_layer = nn.Conv1d(in_channels=3, out_channels=embed_dim,
                                    kernel_size=15, stride=15)

    def forward(self, x):
        x = self.conv_layer(x.permute(0, 2, 1))
        print(x.shape)
        return x


class ETMLP(nn.Module):
    def __init__(self, input_size=9216, hidden_size=4096, num_classes=31,
                 patch_size=100, in_chans=4, embed_dim=15):
        super().__init__()
        self.patch_embed = ETPatchEmbed(
            sample_height=300,
            sample_width=3,
            patch_size=15,
            in_chans=3,
            embed_dim=12)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(900, 2048)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(2048, 1024)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(1024, 512)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(512, 1)
        # self.soft = nn.Softmax(dim=0)
        # Change to output batch_size, num_classes
        # Softmax for top 5?

    def forward(self, x):
        x = x.to(torch.device('cuda'))
        x = torch.cuda.FloatTensor(x.float())
        # out = self.patch_embed(x)

        out = self.flat(x)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        # out = self.soft(out)
        return out


class ETAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor can be set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3,
                                  self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # torchscript cannot use tensor as tuple

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = ETAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop = nn.Dropout(attn_drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class ETTransformer(nn.Module):
    def __init__(self, sample_height=3000, sample_width=4, patch_size=250, in_chans=4,
                 num_classes=31, embed_dim=12, depth=4, num_heads=4, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = ETPatchEmbed(sample_height=3000, sample_width=4, patch_size=patch_size,
                                        in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads,
                  mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  qk_scale=qk_scale, drop=drop_rate,
                  attn_drop=attn_drop_rate, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def train_one_epoch(epoch_index):
    running_loss = 0
    last_loss = 0

    for i, data in enumerate(train_dataloader):
        labels = data[:, 0, 0]
        # labels = labels.to(torch.device('cuda'))
        inputs = data[:, 1:, 1:]  # Whole batch, exclude label, exclude timestamps
        optimiser.zero_grad()
        output = model(inputs).squeeze()
        # output = output.view(64, -1, 31)
        output = output.to(torch.device('cpu'))
        # print(output.shape)
        # outputs = torch.tensor([list(out).index(max(out)) for out in outputs])
        # outputs = outputs.to(torch.device('cuda'))
        # outputs = torch.cuda.FloatTensor(outputs.float())
        # print(outputs, labels)
        loss = loss_fn(output, labels)
        # loss.requires_grad = True
        loss.backward()
        optimiser.step()
        running_loss += loss.item()
        if i % 15 == 14:
            last_loss = running_loss / 1000
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.
    return last_loss


def train_model(model, EPOCHS, init_loss):
    epoch_number = 0
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number)
        # We don't need gradients on to do reporting
        model.train(False)

        running_vloss = 0.0
        for i, vdata in enumerate(test_dataloader):
            vlabels = vdata[:, 0, 0].float()
            vlabels = vlabels.to(torch.device('cuda'))
            vinputs = vdata[:, 1:, 1:]  # Whole batch, exclude label, exclude timestamps
            vinputs = replace_nans(vinputs)
            voutputs = model(vinputs).squeeze()
            # voutputs = torch.tensor([list(out).index(max(out)) for out in voutputs])
            # voutputs = voutputs.to(torch.device('cuda'))
            # voutputs = torch.cuda.FloatTensor(voutputs.float())
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

        avg_test_loss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_test_loss))


if __name__ == "__main__":
    model = ETMLP()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    # print(summary(model, (300, 3)))
    loss_fn = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=1e-3, capturable=True)
    folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DS10'))
    files = list_files(folder)
    train_dataloader, test_dataloader = return_data_loaders(files, 124, 124)
    # train_sample = next(iter(train_dataloader))
    # test_sample = next(iter(test_dataloader))
    # print(train_sample.shape, test_sample.shape)

    EPOCHS = 5
    best_test_loss = 1_000_000.
    train_model(model, EPOCHS, best_test_loss)
