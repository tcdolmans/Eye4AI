# NOTE: This file is depricated and should not be used.

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ETPatchEmbed(nn.Module):
    """Takes a batch of ET data and returns their embeddings."""
    def __init__(self,
                 in_channels=3,
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
        x = self.norm(x).transpose(1, 2)
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
        self.projection = nn.Conv2d(in_channels=in_channels,
                                    out_channels=embed_dim,
                                    kernel_size=patch_size,
                                    stride=stride,
                                    padding=padding)

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ResNet18ImageEmbed(nn.Module):
    def __init__(self, num_classes=335):
        super(ResNet18ImageEmbed, self).__init__()
        self.resnet18 = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-1])
        for param in self.resnet18.parameters():
            param.requires_grad = False

    def forward(self, x):
        # TODO: Replace this line with a function that resizes the image to 224x224 and saves them
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.resnet18(x)  # Outputs have shape (batch_size, 512, 1, 1)


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
        # TODO: Should this be Conv3d?
        self.projection = nn.Conv2d(in_channels=in_channels,
                                    out_channels=embed_dim,
                                    kernel_size=patch_size,
                                    stride=stride,
                                    padding=padding)

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class SelfAttention(nn.Module):
    # Based on the implementation from Aladdin Persson: youtube.com/watch?v=U0s0f995w14
    def __init__(self, embed_size, heads, device):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.device = device
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.query = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        query = self.query(query)  # (N, query_len, heads, head_dim)

        # Good source on Einsum: youtube.com/watch?v=pkVwUVEHmfI
        energy = torch.einsum("nqhd,nkhd->nhqk", [query, keys])
        # Queries shape: (N, query_len, heads, head_dim)
        # Keys shape: (N, key_len, heads, head_dim)
        # Energy shape: (N, heads, query_len, key_len)

        if mask is not None:
            # TODO: Energy shape changes with bottleneck, but mask does not
            # if not energy.shape[2] == mask.shape[2]:
            # Change mask shape from torch.Size([8, 2, 192, 192]) to torch.Size([8, 2, 196, 196])

            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim)
        # Attention shape: (N, heads, query_len, key_len)
        # Values shape: (N, value_len, heads, head_dim)\
        # Out shape : (N, query_len, heads, head_dim), then flatten last 2 dims
        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, device):
        super(TransformerBlock, self).__init__()
        self.device = device
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size=embed_size,
                                       heads=heads,
                                       device=device)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value,
                                   key,
                                   query,
                                   mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        embed_size,
        num_layers,
        heads,
        fusion_layer,
        forward_expansion,
        dropout,
        device,
        modalities,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.heads = heads
        self.fusion_layer = fusion_layer
        self.forward_expansion = forward_expansion
        self.dropout = dropout
        self.device = device
        self.modalities = modalities
        self.position_embedding = nn.Embedding(embed_size, embed_size)

    def forward(self, x: dict[str, any], bottleneck, src_key_padding_mask):
        for modality in self.modalities:
            embed, expand = x[modality].shape[1], x[modality].shape[0]
            x[modality] += self.position_embedding(torch.arange(0, embed)
                                                   .expand(expand, -1)
                                                   .to(self.device))
        for lyr in range(self.num_layers):
            encoders = {}
            for modality in self.modalities:
                encoders[modality] = TransformerBlock(embed_size=self.embed_size,
                                                      heads=self.heads,
                                                      dropout=self.dropout,
                                                      forward_expansion=self.forward_expansion,
                                                      device=self.device).to(self.device)
            if lyr < self.fusion_layer or lyr == 1:
                for modality in self.modalities:
                    x[modality] = encoders[modality](x[modality],
                                                     x[modality],
                                                     x[modality],
                                                     src_key_padding_mask)
            else:
                bottle = []
                for modality in self.modalities:
                    t_mod = x[modality].shape[1]
                    in_mod = torch.cat([x[modality], bottleneck], dim=1)
                    out_mod = encoders[modality](in_mod,
                                                 in_mod,
                                                 in_mod,
                                                 src_key_padding_mask)
                    x[modality] = out_mod[:, :t_mod]
                    bottle.append(out_mod[:, t_mod:])
                bottleneck = torch.mean(torch.stack(bottle, dim=-1), dim=-1)
        x_out = torch.cat([x[modality] for modality in self.modalities], dim=-1)
        return nn.LayerNorm(x_out)


class MultimodalBottleneckTransformer(nn.Module):
    def __init__(self,
                 num_layers,
                 heads,
                 embed_dim,
                 forward_expansion,
                 fusion_layer,
                 dropout,
                 et_embed_dim,
                 et_patch_size,
                 et_stride,
                 img_embed_dim,
                 img_patch_size,
                 img_stride,
                 sem_embed_dim,
                 sem_patch_size,
                 sem_stride,
                 modalities,
                 num_classes,
                 mode,
                 device,
                 n_bottlenecks,
                 use_bottleneck=True,
                 test_with_bottle_neck=False):
        super(MultimodalBottleneckTransformer, self).__init__()
        self.et_embed = ETPatchEmbed(in_channels=3,
                                     embed_dim=et_embed_dim,
                                     kernel_size=et_patch_size,
                                     stride=et_stride)
        self.img_embed = ImagePatchEmbed(in_channels=3,
                                         embed_dim=img_embed_dim,
                                         patch_size=img_patch_size,
                                         stride=img_stride)
        self.sem_embed = SemanticEmbedding(in_channels=12,
                                           embed_dim=sem_embed_dim,
                                           patch_size=sem_patch_size,
                                           stride=sem_stride)
        self.d_model = et_embed_dim + img_embed_dim + sem_embed_dim
        self.mode = mode
        self.device = device
        self.use_bottleneck = use_bottleneck
        self.n_bottlenecks = n_bottlenecks
        self.modalities = modalities
        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=4 * self.d_model,
            custom_encoder=Encoder(embed_size=embed_dim,
                                   num_layers=num_layers,
                                   heads=heads,
                                   fusion_layer=fusion_layer,
                                   forward_expansion=forward_expansion,
                                   dropout=dropout,
                                   device=self.device,
                                   modalities=modalities))
        self.classification_head = nn.Linear(self.d_model, num_classes)
        self.prediction_head = nn.Linear(self.d_model, et_embed_dim)

    def create_src_tgt_sequences(self, x: dict[str, any], start_token=None):
        """Create the source and target sequences for the transformer model."""
        src = {}
        tgt = {}
        for modality in self.modalities:
            src_batch = x[modality]
            if start_token is None:
                # Currently not using a proper start token, just generating one
                start_token = src_batch[:, 0].unsqueeze(1)

            # Shift the source sequence to create the target sequence
            tgt_batch = src_batch[:, :-1]  # Remove the last element
            tgt_batch = torch.cat((start_token, tgt_batch), dim=1)  # Add the start token

            src[modality] = src_batch
            tgt[modality] = tgt_batch
        return src, tgt

    def adjust_time_dimension(self, embedding, target_time_steps=192):
        """Adjust the time dimension of the embedding to match the target time steps."""
        batch_size, current_steps, feature_dim = embedding.shape
        if current_steps == target_time_steps:
            return embedding
        else:
            embedding = embedding.unsqueeze(1)
            embedding = nn.functional.interpolate(embedding,
                                                  size=(target_time_steps, feature_dim),
                                                  mode='bilinear',
                                                  align_corners=False)
            return embedding.squeeze(1)

    def make_src_mask(self, src, src_pad_index=0):
        src_mask = {}
        for modality in src:
            src_mask[modality] = (
                src[modality] != src_pad_index).unsqueeze(1)
        return src_mask

    def make_tgt_mask(self, tgt):
        tgt_mask = {}
        if 'et' in tgt:
            tgt_tensor = tgt['et']
            N, tgt_len = tgt_tensor.shape[:-1]
            tgt_mask['et'] = torch.tril(torch.ones((tgt_len, tgt_len))).expand(
                N, 2, tgt_len, tgt_len).to(self.device)
        return tgt_mask

    def forward(self, et_data, img_data, sem_data):
        # Initialise the input dictionary and add the embeddings
        x = {}
        x['et'] = self.adjust_time_dimension(self.et_embed(et_data))
        x['img'] = self.adjust_time_dimension(self.img_embed(img_data))
        x['sem'] = self.adjust_time_dimension(self.sem_embed(sem_data))
        # TODO: Figure out if I need class tokens.. This would be the place to init them.
        if self.use_bottleneck:
            n, c, t = x['et'].shape
            bottleneck = nn.init.normal_(
                torch.empty(n, self.n_bottlenecks, t),
                mean=0, std=0.02).to(self.device)

        src, tgt = self.create_src_tgt_sequences(x)
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        out_src = self.transformer.encoder(src, bottleneck, src_key_padding_mask=src_mask['et'])
        out = self.transformer.decoder(tgt, out_src, tgt_mask=tgt_mask['et'])

        if self.mode == "classification":
            out = self.classification_head(out)
            return torch.mean(out, dim=1)  # NOTE:May not have to mean it here
        elif self.mode == "prediction":
            out = self.prediction_head(out)
            # Extract only the ET embeddings from the output
            et_embeddings = out[:, :, :self.et_embed.embed_dim]
            return torch.mean(et_embeddings, dim=1)  # NOTE:May not have to mean it here
        else:
            raise ValueError("Invalid mode. Must be either 'classification' or 'prediction'.")


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Instantiate the model
model = MultimodalBottleneckTransformer(num_layers=3,
                                        heads=2,
                                        embed_dim=192,
                                        forward_expansion=4,
                                        fusion_layer=2,
                                        dropout=0.1,
                                        et_embed_dim=192,
                                        et_patch_size=15,
                                        et_stride=1,
                                        img_embed_dim=192,
                                        img_patch_size=25,
                                        img_stride=12,
                                        sem_embed_dim=192,
                                        sem_patch_size=25,
                                        sem_stride=12,
                                        modalities=['et', 'img', 'sem'],
                                        num_classes=335,
                                        device=device,
                                        n_bottlenecks=4,
                                        mode="classification").to(device)


# Initialize input data
et_data = torch.randn(8, 3, 300).to(device)
img_data = torch.randn(8, 3, 600, 800).to(device)
sem_data = torch.randn(8, 12, 600, 800).to(device)

# Forward pass
output = model(et_data, img_data, sem_data)
print(output.shape)
