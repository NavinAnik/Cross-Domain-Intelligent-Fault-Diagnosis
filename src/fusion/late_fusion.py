"""
Late Fusion: ensemble predictions from separate models via weighted averaging.

Implementation: α * audio_logits + (1-α) * vision_logits.
α can be a trainable parameter or fixed (e.g. grid search).
"""

import torch
import torch.nn as nn
from typing import Optional


class LateFusion(nn.Module):
    """
    Two separate encoders + classifiers; fuse logits with learnable weight α.

    L_fused = α * audio_logits + (1 - α) * vision_logits.

    Args:
        audio_encoder: Encoder for audio. Must have .feature_dim.
        vision_encoder: Encoder for vision. Must have .feature_dim.
        num_classes: Number of output classes.
        alpha_init: Initial value for fusion weight α in [0, 1]. If None, learned from 0.5.
        learn_alpha: If True, α is trainable; otherwise fixed to alpha_init.
        hidden_dim: Optional hidden dim for per-modality classifier MLPs.
        dropout: Dropout for classifiers.
    """

    def __init__(
        self,
        audio_encoder: nn.Module,
        vision_encoder: nn.Module,
        num_classes: int = 2,
        alpha_init: float = 0.5,
        learn_alpha: bool = True,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.vision_encoder = vision_encoder

        def _fd(m):
            return getattr(m, "feature_dim", None) or getattr(
                getattr(m, "encoder", None), "feature_dim", 768
            )

        audio_dim = _fd(audio_encoder)
        vision_dim = _fd(vision_encoder)

        def _make_head(in_dim: int):
            if hidden_dim is None:
                return nn.Sequential(
                    nn.LayerNorm(in_dim),
                    nn.Dropout(dropout),
                    nn.Linear(in_dim, num_classes),
                )
            return nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

        self.audio_classifier = _make_head(audio_dim)
        self.vision_classifier = _make_head(vision_dim)

        # α in [0,1]: logit so that sigmoid(logit) = alpha_init
        import math

        self._alpha_logit = nn.Parameter(
            torch.tensor(math.log(alpha_init / (1.0 - alpha_init + 1e-8))),
            requires_grad=learn_alpha,
        )
        self._num_classes = num_classes

    def forward(
        self,
        audio_input: torch.Tensor,
        vision_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            Fused logits (B, num_classes).
        """
        audio_feat = self.audio_encoder(audio_input)
        vision_feat = self.vision_encoder(vision_input)
        audio_logits = self.audio_classifier(audio_feat)
        vision_logits = self.vision_classifier(vision_feat)
        alpha = torch.sigmoid(self._alpha_logit)
        return alpha * audio_logits + (1.0 - alpha) * vision_logits

    def get_modality_logits(
        self,
        audio_input: torch.Tensor,
        vision_input: torch.Tensor,
    ):
        """Return audio logits, vision logits, and fused logits (for analysis)."""
        audio_feat = self.audio_encoder(audio_input)
        vision_feat = self.vision_encoder(vision_input)
        audio_logits = self.audio_classifier(audio_feat)
        vision_logits = self.vision_classifier(vision_feat)
        alpha = torch.sigmoid(self._alpha_logit)
        fused = alpha * audio_logits + (1.0 - alpha) * vision_logits
        return audio_logits, vision_logits, fused

    @property
    def alpha(self) -> float:
        return torch.sigmoid(self._alpha_logit).item()
