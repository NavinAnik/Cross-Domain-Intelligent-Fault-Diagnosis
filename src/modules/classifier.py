import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class ClassificationModel(nn.Module):
    def __init__(self, input_dim, model_dim=128, num_classes=2, num_heads=4, num_layers=2, dropout=0.1, max_len=512):
        super().__init__()

        self.embedding = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, num_classes)
        )

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        If using tabular data, reshape to (batch_size, 1, input_dim)
        """
        x = self.embedding(x)               # (B, S, model_dim)
        x = self.pos_encoder(x)             # (B, S, model_dim)
        x = self.transformer_encoder(x)     # (B, S, model_dim)
        cls_token = x[:, 0, :]              # use first token as aggregate (CLS)
        return self.classifier(cls_token)
