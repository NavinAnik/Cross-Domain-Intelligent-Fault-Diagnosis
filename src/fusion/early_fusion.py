"""
Early Fusion: concatenate audio and vision features before classifier.

Implementation: torch.cat([audio_features, vision_features], dim=1) then MLP.
Simple but effective baseline for multimodal fusion.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


def _get_feature_dim(encoder: nn.Module) -> int:
    return getattr(encoder, "feature_dim", None) or getattr(
        getattr(encoder, "encoder", None), "feature_dim", 768
    )


class EarlyFusion(nn.Module):
    """
    Concatenate audio and vision features, then pass through a classifier.

    Args:
        audio_encoder: Encoder that returns (B, audio_dim). Must have .feature_dim.
        vision_encoder: Encoder that returns (B, vision_dim). Must have .feature_dim.
        num_classes: Number of output classes.
        hidden_dim: Optional hidden dim for classifier MLP. If None, single linear layer.
        dropout: Dropout rate for classifier.
    """

    def __init__(
        self,
        audio_encoder: nn.Module,
        vision_encoder: nn.Module,
        num_classes: int = 2,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.vision_encoder = vision_encoder
        audio_dim = _get_feature_dim(audio_encoder)
        vision_dim = _get_feature_dim(vision_encoder)
        fused_dim = audio_dim + vision_dim

        if hidden_dim is None:
            self.classifier = nn.Sequential(
                nn.LayerNorm(fused_dim),
                nn.Dropout(dropout),
                nn.Linear(fused_dim, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(fused_dim),
                nn.Linear(fused_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        self._fused_dim = fused_dim
        self._num_classes = num_classes

    def forward(
        self,
        audio_input: torch.Tensor,
        vision_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            audio_input: (B, ...) as expected by audio_encoder (e.g. spectrograms).
            vision_input: (B, ...) as expected by vision_encoder (e.g. images).

        Returns:
            Logits (B, num_classes).
        """
        audio_feat = self.audio_encoder(audio_input)
        vision_feat = self.vision_encoder(vision_input)
        fused = torch.cat([audio_feat, vision_feat], dim=1)
        return self.classifier(fused)

    def get_features(
        self,
        audio_input: torch.Tensor,
        vision_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return audio features, vision features, and concatenated fusion (for analysis)."""
        audio_feat = self.audio_encoder(audio_input)
        vision_feat = self.vision_encoder(vision_input)
        fused = torch.cat([audio_feat, vision_feat], dim=1)
        return audio_feat, vision_feat, fused

    @property
    def feature_dim(self) -> int:
        return self._fused_dim
