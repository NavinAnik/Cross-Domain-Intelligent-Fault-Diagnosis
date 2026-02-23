"""
ConFu (Contrastive Fusion): higher-order contrastive alignment.

Jointly embed individual modalities AND their fusion.
Loss: L = L_pairwise(audio, vision) + L_fusion(concat(audio, vision), targets).
Captures XOR-like dependencies missed by pairwise methods.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


def _get_feature_dim(encoder: nn.Module) -> int:
    return getattr(
        encoder,
        "feature_dim",
        getattr(getattr(encoder, "encoder", None), "feature_dim", 768),
    )


class ConFuFusion(nn.Module):
    """
    Early fusion plus pairwise contrastive alignment.

    Forward returns logits and (audio_feat, vision_feat) so the caller can compute:
    L = CE(logits, targets) + λ * L_contrastive(audio_feat, vision_feat).

    Args:
        audio_encoder: Encoder for audio (B, audio_dim).
        vision_encoder: Encoder for vision (B, vision_dim).
        num_classes: Number of output classes.
        hidden_dim: Optional hidden dim for fusion classifier.
        dropout: Dropout for classifier.
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
            self.fusion_classifier = nn.Sequential(
                nn.LayerNorm(fused_dim),
                nn.Dropout(dropout),
                nn.Linear(fused_dim, num_classes),
            )
        else:
            self.fusion_classifier = nn.Sequential(
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
        Returns fusion logits (B, num_classes).
        Use get_features() to obtain audio_feat, vision_feat for contrastive loss.
        """
        audio_feat = self.audio_encoder(audio_input)
        vision_feat = self.vision_encoder(vision_input)
        fused = torch.cat([audio_feat, vision_feat], dim=1)
        return self.fusion_classifier(fused)

    def get_features_and_logits(
        self,
        audio_input: torch.Tensor,
        vision_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (audio_feat, vision_feat, fused_features, logits).
        Caller can compute:
          L_fusion = F.cross_entropy(logits, targets)
          L_pairwise = contrastive_loss(audio_feat, vision_feat)  # e.g. NT-Xent
          L = L_fusion + lambda_pair * L_pairwise
        """
        audio_feat = self.audio_encoder(audio_input)
        vision_feat = self.vision_encoder(vision_input)
        fused = torch.cat([audio_feat, vision_feat], dim=1)
        logits = self.fusion_classifier(fused)
        return audio_feat, vision_feat, fused, logits

    @property
    def feature_dim(self) -> int:
        return self._fused_dim
