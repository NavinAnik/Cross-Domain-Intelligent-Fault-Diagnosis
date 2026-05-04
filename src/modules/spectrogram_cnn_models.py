"""
CNN baselines and CNN+Attention / CNN+Transformer hybrids for spectrogram-image classification.

Uses timm backbones (ResNet, EfficientNet) with global_pool='' for feature maps.
Factory: build_model(name, num_classes, **kwargs).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    _HAS_TIMM = True
except ImportError:
    timm = None
    _HAS_TIMM = False


class CNNBaseline(nn.Module):
    """ImageNet-pretrained CNN (timm) + global average pool + linear classifier."""

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 2,
        pretrained: bool = True,
    ):
        super().__init__()
        if not _HAS_TIMM:
            raise ImportError("timm is required: pip install timm")
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        self.num_features = self.backbone.num_features
        self.head = nn.Linear(self.num_features, num_classes)
        self.backbone_name = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        if feat.dim() == 4:
            feat = feat.mean(dim=(-2, -1))
        return self.head(feat)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        if feat.dim() == 4:
            return feat.mean(dim=(-2, -1))
        return feat


class SpatialAttention2D(nn.Module):
    """CBAM-style spatial attention on (B, C, H, W); returns gated features and attention map."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        p = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=p, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        attn = torch.cat([avg, mx], dim=1)
        attn = torch.sigmoid(self.conv(attn))
        return x * attn, attn.squeeze(1)


class CNNSpatialAttention(nn.Module):
    """CNN backbone + spatial attention on last feature map before GAP."""

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 2,
        pretrained: bool = True,
        attn_kernel: int = 7,
    ):
        super().__init__()
        if not _HAS_TIMM:
            raise ImportError("timm is required")
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        self.num_features = self.backbone.num_features
        self.spatial_attn = SpatialAttention2D(kernel_size=attn_kernel)
        self.head = nn.Linear(self.num_features, num_classes)
        self._last_spatial_attn: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        feat, attn = self.spatial_attn(feat)
        self._last_spatial_attn = attn.detach()
        pooled = feat.mean(dim=(-2, -1))
        return self.head(pooled)

    def get_last_spatial_attention(self) -> Optional[torch.Tensor]:
        """(B, H, W) attention weights from last forward."""
        return self._last_spatial_attn


def _build_2d_sincos_pe(h: int, w: int, d_model: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """(1, h*w, d_model) sinusoidal PE from normalized row/col indices."""
    pe = torch.zeros(h * w, d_model, device=device, dtype=dtype)
    d2 = d_model // 2
    if d2 == 0:
        return pe.unsqueeze(0)
    rows = torch.linspace(0.0, 1.0, steps=h, device=device, dtype=dtype)
    cols = torch.linspace(0.0, 1.0, steps=w, device=device, dtype=dtype)
    grid_r = rows.unsqueeze(1).expand(h, w).reshape(-1)
    grid_c = cols.unsqueeze(0).expand(h, w).reshape(-1)
    div_r = torch.exp(torch.arange(0, d2, 2, device=device, dtype=dtype) * (-math.log(10000.0) / max(d2, 1)))
    div_c = torch.exp(torch.arange(0, d_model - d2, 2, device=device, dtype=dtype) * (-math.log(10000.0) / max(d_model - d2, 1)))
    for i in range(0, d2, 2):
        pe[:, i] = torch.sin(grid_r * div_r[i // 2])
        if i + 1 < d2:
            pe[:, i + 1] = torch.cos(grid_r * div_r[i // 2])
    base = d2
    for j in range(0, d_model - d2, 2):
        idx = base + j
        if idx < d_model:
            pe[:, idx] = torch.sin(grid_c * div_c[j // 2])
        if idx + 1 < d_model:
            pe[:, idx + 1] = torch.cos(grid_c * div_c[j // 2])
    return pe.unsqueeze(0)


class CNNTransformerHybrid(nn.Module):
    """
    CNN feature extractor (feature map B,C,H,W) -> flatten -> linear proj ->
    2D sinusoidal PE + optional CLS -> TransformerEncoder -> classify from CLS (or mean pool).
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 2,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pretrained: bool = True,
        use_cls_token: bool = True,
    ):
        super().__init__()
        if not _HAS_TIMM:
            raise ImportError("timm is required")
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        c = self.backbone.num_features
        self.proj = nn.Linear(c, d_model)
        self.use_cls_token = use_cls_token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.d_model = d_model
        self._last_attn: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        b, c, h, w = feat.shape
        seq = feat.flatten(2).transpose(1, 2)
        seq = self.proj(seq)
        pe = _build_2d_sincos_pe(h, w, self.d_model, seq.device, seq.dtype)
        seq = seq + pe
        if self.use_cls_token:
            cls = self.cls_token.expand(b, -1, -1)
            seq = torch.cat([cls, seq], dim=1)
        seq = self.encoder(seq)
        if self.use_cls_token:
            pooled = seq[:, 0]
        else:
            pooled = seq.mean(dim=1)
        pooled = self.norm(pooled)
        return self.head(pooled)

    def forward_with_attention(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning (logits, attn_weights) where attn_weights is (B, L, L)
        from the last encoder layer (matches norm_first=True TransformerEncoderLayer).
        """
        feat = self.backbone(x)
        b, c, h, w = feat.shape
        seq = feat.flatten(2).transpose(1, 2)
        seq = self.proj(seq)
        pe = _build_2d_sincos_pe(h, w, self.d_model, seq.device, seq.dtype)
        seq = seq + pe
        if self.use_cls_token:
            cls = self.cls_token.expand(b, -1, -1)
            seq = torch.cat([cls, seq], dim=1)
        x_tok = seq
        layers = list(self.encoder.layers)
        for layer in layers[:-1]:
            x_tok = layer(x_tok, is_causal=False)
        last = layers[-1]
        x2 = last.norm1(x_tok)
        attn_out, attn_w = last.self_attn(
            x2, x2, x2, need_weights=True, average_attn_weights=True
        )
        x_tok = x_tok + last.dropout1(attn_out)
        x_tok = x_tok + last._ff_block(last.norm2(x_tok))
        self._last_attn = attn_w
        if self.use_cls_token:
            pooled = self.norm(x_tok[:, 0])
        else:
            pooled = self.norm(x_tok.mean(dim=1))
        logits = self.head(pooled)
        if attn_w is None:
            attn_w = torch.zeros(b, x_tok.size(1), x_tok.size(1), device=x.device, dtype=x.dtype)
        return logits, attn_w

    def get_last_attention(self) -> Optional[torch.Tensor]:
        return self._last_attn


def build_model(
    name: str,
    num_classes: int,
    backbone: str = "resnet50",
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    """
    Factory for spectrogram CNN models.

    Names:
      - resnet50, efficientnet_b0: timm backbone names as model name (shortcut)
      - cnn_baseline: backbone from kwargs default resnet50
      - cnn_se: ResNet50 + spatial attention (backbone configurable via backbone=)
      - cnn_transformer: CNN + Transformer hybrid
    """
    name_l = name.lower().replace("-", "_")
    if name_l in ("resnet50", "efficientnet_b0", "efficientnet_b1", "resnet34"):
        return CNNBaseline(backbone=name_l, num_classes=num_classes, pretrained=pretrained)
    if name_l == "cnn_baseline":
        bb = kwargs.get("backbone", backbone)
        return CNNBaseline(backbone=bb, num_classes=num_classes, pretrained=pretrained)
    if name_l in ("cnn_se", "cnn_spatial_attention"):
        bb = kwargs.get("backbone", backbone)
        return CNNSpatialAttention(
            backbone=bb,
            num_classes=num_classes,
            pretrained=pretrained,
            attn_kernel=int(kwargs.get("attn_kernel", 7)),
        )
    if name_l in ("cnn_transformer", "hybrid_cnn_transformer"):
        bb = kwargs.get("backbone", backbone)
        return CNNTransformerHybrid(
            backbone=bb,
            num_classes=num_classes,
            d_model=int(kwargs.get("d_model", 256)),
            nhead=int(kwargs.get("nhead", 8)),
            num_layers=int(kwargs.get("num_layers", 2)),
            dim_feedforward=int(kwargs.get("dim_feedforward", 1024)),
            dropout=float(kwargs.get("dropout", 0.1)),
            pretrained=pretrained,
            use_cls_token=bool(kwargs.get("use_cls_token", True)),
        )
    raise ValueError(
        f"Unknown model name: {name}. Try resnet50, efficientnet_b0, cnn_baseline, cnn_se, cnn_transformer"
    )


def list_model_names() -> List[str]:
    return [
        "resnet50",
        "efficientnet_b0",
        "cnn_baseline",
        "cnn_se",
        "cnn_transformer",
    ]
