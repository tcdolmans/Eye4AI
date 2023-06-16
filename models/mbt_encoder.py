"""
 * @author [Tenzing Dolmans]
 * @email [t.c.dolmans@gmail.com]
 * @create date 2023-05-11 12:04:49
 * @modify date 2023-05-11 12:04:49
 * @desc [description]
"""
import torch
import torch.nn as nn
import numpy as np


class SelfAttention(nn.Module):
    """
    Based on the implementation from Aladdin Persson: youtube.com/watch?v=U0s0f995w14
    """
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

    def adjust_mask_size(self, mask, target_size, num_channels, padding_value) -> torch.Tensor:
        """
        Adjusts the size of the mask to the target size if they are not equal.
        """
        target_rows, target_cols = target_size
        if mask.size(1) < target_rows:
            padding_rows = torch.full((mask.size(0), target_rows - mask.size(1), mask.size(2)),
                                      padding_value, dtype=mask.dtype, device=mask.device)
            mask = torch.cat((mask, padding_rows), dim=1)
        if mask.size(2) < target_cols:
            padding_cols = torch.full((mask.size(0), target_rows, target_cols - mask.size(2)),
                                      padding_value, dtype=mask.dtype, device=mask.device)
            mask = torch.cat((mask, padding_cols), dim=2)
        mask = mask.unsqueeze(1)
        mask = mask.expand(-1, num_channels, -1, -1)
        return mask

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
            mask = self.adjust_mask_size(mask, energy.shape[-2:], energy.shape[1], mask[-1, -1, -1])
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
    """
    Standard transformer block with the above defined self-attention.
    """
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

    def forward(self, value, key, query, mask=None):
        attention = self.attention(value,
                                   key,
                                   query,
                                   mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    """
    Custom encoder module that stacks the transformer blocks and uses bottleneck fusion.
    Takes in a dictionary of modalities and their embeddings and generates blocks for each modality.
    Bottleneck tokens are added to the embeddings of each modality.
    """
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
        self.position_embeddings = {
            modality: nn.Embedding(embed_size+1, embed_size).to(self.device)
            for modality in self.modalities}
        self.layer_norm = nn.LayerNorm(embed_size * len(self.modalities))

        self.transformer_blocks = nn.ModuleList([
            nn.ModuleDict({
                modality: TransformerBlock(embed_size=self.embed_size,
                                           heads=self.heads,
                                           dropout=self.dropout,
                                           forward_expansion=self.forward_expansion,
                                           device=self.device) for modality in self.modalities
                            })for _ in range(num_layers)
            ])

    def generate_positional_encoding(self, seq_len, embed_size):
        PE = np.zeros((seq_len, embed_size))
        for pos in range(seq_len):
            for i in range(0, embed_size, 2):
                PE[pos, i] = np.sin(pos / (10000 ** ((2 * i) / embed_size)))
                if i + 1 < embed_size:
                    PE[pos, i + 1] = np.cos(pos / (10000 ** ((2 * i) / embed_size)))
        return torch.tensor(PE, dtype=torch.float32).to(self.device)

    def forward(self, x: dict[str, any], bottleneck, src_key_padding_mask=None):
        # Assuming x contains the latent embeddings
        for modality in self.modalities:
            expand, seq_len, embed = x[modality].shape
            pos = self.generate_positional_encoding(seq_len, embed)
            self.register_buffer('position_encoding', pos)
            # pos = self.position_embeddings[modality](torch.arange(0, embed)
            #                                          .expand(expand, -1)
            #                                          .to(self.device))
            x[modality] += pos
        for lyr in range(self.num_layers):
            encoders = self.transformer_blocks[lyr]
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
        return self.layer_norm(x_out)
