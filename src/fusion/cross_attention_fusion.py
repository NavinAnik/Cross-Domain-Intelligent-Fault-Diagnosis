"""
Cross-Modal Attention Fusion: bidirectional cross-attention between modalities.

Query from one modality, Key/Value from the other (and vice versa).
Enables rich multimodal interaction before classification.
"""

import torch
import torch.nn as nn
from typing import Optional


def _get_feature_dim(encoder: nn.Module) -> int:
    return getattr(
        encoder,
        "feature_dim",
        getattr(getattr(encoder, "encoder", None), "feature_dim", 768),
    )


class CrossAttentionFusion(nn.Module):
    """
    Fuse audio and vision via bidirectional cross-attention, then classify.

    - Cross-attn 1: Q=audio, K,V=vision -> audio_attended
    - Cross-attn 2: Q=vision, K,V=audio -> vision_attended
    - Concat [audio_attended, vision_attended] and project to logits.

    Args:
        audio_encoder: Encoder returning (B, audio_dim).
        vision_encoder: Encoder returning (B, vision_dim).
        num_classes: Number of output classes.
        hidden_dim: Common projection dim for cross-attention (default: min(audio_dim, vision_dim)).
        num_heads: Number of attention heads.
        dropout: Dropout for attention and classifier.
    """

    def __init__(
        self,
        audio_encoder: nn.Module,
        vision_encoder: nn.Module,
        num_classes: int = 2,
        hidden_dim: Optional[int] = None,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.vision_encoder = vision_encoder
        audio_dim = _get_feature_dim(audio_encoder)
        vision_dim = _get_feature_dim(vision_encoder)
        h = hidden_dim or min(audio_dim, vision_dim)
        self.hidden_dim = h

        self.audio_proj = nn.Linear(audio_dim, h)
        self.vision_proj = nn.Linear(vision_dim, h)
        self.cross_attn_av = nn.MultiheadAttention(
            embed_dim=h,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_va = nn.MultiheadAttention(
            embed_dim=h,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_av = nn.LayerNorm(h)
        self.norm_va = nn.LayerNorm(h)
        fused_dim = 2 * h
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, num_classes),
        )
        self._num_classes = num_classes

    def forward(
        self,
        audio_input: torch.Tensor,
        vision_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            Logits (B, num_classes).
        """
        audio_feat = self.audio_encoder(audio_input)  # (B, D_a)
        vision_feat = self.vision_encoder(vision_input)  # (B, D_v)
        B = audio_feat.size(0)
        # Treat as sequence of length 1 for cross-attention
        q_audio = self.audio_proj(audio_feat).unsqueeze(1)  # (B, 1, h)
        kv_vision = self.vision_proj(vision_feat).unsqueeze(1)  # (B, 1, h)
        q_vision = self.vision_proj(vision_feat).unsqueeze(1)
        kv_audio = self.audio_proj(audio_feat).unsqueeze(1)

        audio_attended, _ = self.cross_attn_av(
            query=q_audio,
            key=kv_vision,
            value=kv_vision,
        )
        vision_attended, _ = self.cross_attn_va(
            query=q_vision,
            key=kv_audio,
            value=kv_audio,
        )
        audio_attended = self.norm_av(q_audio + audio_attended).squeeze(1)
        vision_attended = self.norm_va(q_vision + vision_attended).squeeze(1)

        fused = torch.cat([audio_attended, vision_attended], dim=1)
        return self.classifier(fused)

    def get_features(
        self,
        audio_input: torch.Tensor,
        vision_input: torch.Tensor,
    ):
        """Return audio features, vision features, and fused representation (for analysis)."""
        audio_feat = self.audio_encoder(audio_input)
        vision_feat = self.vision_encoder(vision_input)
        B = audio_feat.size(0)
        q_audio = self.audio_proj(audio_feat).unsqueeze(1)
        kv_vision = self.vision_proj(vision_feat).unsqueeze(1)
        q_vision = self.vision_proj(vision_feat).unsqueeze(1)
        kv_audio = self.audio_proj(audio_feat).unsqueeze(1)
        audio_attended, _ = self.cross_attn_av(
            query=q_audio, key=kv_vision, value=kv_vision
        )
        vision_attended, _ = self.cross_attn_va(
            query=q_vision, key=kv_audio, value=kv_audio
        )
        audio_attended = self.norm_av(q_audio + audio_attended).squeeze(1)
        vision_attended = self.norm_va(q_vision + vision_attended).squeeze(1)
        fused = torch.cat([audio_attended, vision_attended], dim=1)
        return audio_feat, vision_feat, fused

    @property
    def feature_dim(self) -> int:
        return 2 * self.hidden_dim
