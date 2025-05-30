import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_dim=256, img_size=224):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, emb_dim))

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x)  # (B, emb_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # (B, N, emb_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, emb_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, emb_dim)
        x = x + self.pos_embed
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim=256, num_heads=8, num_layers=6, mlp_dim=512, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.transformer(x)

class Encoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, emb_dim=768, num_layers=12, num_heads=12, mlp_dim=3072, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels=3, patch_size=patch_size, emb_dim=emb_dim, img_size=img_size)
        self.transformer = TransformerEncoder(
            emb_dim=emb_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_dim=mlp_dim,
            dropout=dropout
        )
        self.norm = nn.LayerNorm(emb_dim)
        self.feature_dim = emb_dim  # Output feature dimension

    def forward(self, x):
        x = self.patch_embed(x)      # (B, N+1, emb_dim)
        x = self.transformer(x)      # (B, N+1, emb_dim)
        x = self.norm(x[:, 0])       # CLS token
        return x