class MBT(nn.Module):
    def __init__(self,
                 num_patches,
                 embed_dim):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.fsn_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.transformer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8)

    def forward(self, x):
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        fsn_tokens = self.fsn_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, fsn_tokens, x), dim=1)
        x = self.transformer(x)
        return x
