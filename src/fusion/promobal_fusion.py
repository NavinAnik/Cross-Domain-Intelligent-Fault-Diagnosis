"""
ProMoBal (Prototype-Guided Modality Balancing): address modality imbalance.

Create class prototypes on hypersphere (regular simplex geometry).
Sample-adaptive fusion weights: w_audio, w_vision = f(sample_difficulty).
Reported to improve accuracy when one modality dominates.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


def _get_feature_dim(encoder: nn.Module) -> int:
    return getattr(
        encoder,
        "feature_dim",
        getattr(getattr(encoder, "encoder", None), "feature_dim", 768),
    )


def regular_simplex_prototypes(
    num_classes: int, dim: int, device: torch.device
) -> torch.Tensor:
    """
    Place num_classes prototypes on the unit hypersphere as a regular simplex
    (vertices of equilateral simplex). Returns (num_classes, dim) L2-normalized.
    """
    if num_classes == 1:
        return torch.ones(1, dim, device=device) / math.sqrt(dim)
    # First vertex
    prototypes = torch.zeros(num_classes, dim, device=device)
    prototypes[0, 0] = 1.0
    # Subsequent vertices: equal angles with previous
    for k in range(1, num_classes):
        # Place k-th vertex so that it has equal dot product with all previous
        # and lies on unit sphere. Standard simplex: e_1, then ( -1/k, ..., -1/k, 1 - 1/k, 0, ... )
        # then normalize.
        prototypes[k, :k] = -1.0 / (k + 1)
        prototypes[k, k] = 1.0 - (k + 1) * (-1.0 / (k + 1))
    # Normalize rows to unit sphere
    prototypes = prototypes / (prototypes.norm(dim=1, keepdim=True) + 1e-8)
    return prototypes


class ProMoBalFusion(nn.Module):
    """
    Prototype-guided modality balancing with sample-adaptive fusion weights.

    - Project audio and vision features to a common sphere (L2 norm).
    - Learnable class prototypes on hypersphere (initialized as regular simplex).
    - Per-sample weights w_audio, w_vision from a small MLP that takes
      [audio_feat, vision_feat] or prediction confidence as difficulty.
    - Fused logits = w_audio * audio_logits + w_vision * vision_logits.

    Args:
        audio_encoder: Encoder for audio.
        vision_encoder: Encoder for vision.
        num_classes: Number of output classes.
        projection_dim: Common dimension for projection to sphere (default: 256).
        hidden_dim: Hidden dim for modality classifiers and weight MLP.
        dropout: Dropout.
    """

    def __init__(
        self,
        audio_encoder: nn.Module,
        vision_encoder: nn.Module,
        num_classes: int = 2,
        projection_dim: int = 256,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.vision_encoder = vision_encoder
        self.num_classes = num_classes
        self.projection_dim = projection_dim
        audio_dim = _get_feature_dim(audio_encoder)
        vision_dim = _get_feature_dim(vision_encoder)
        h = hidden_dim or min(256, audio_dim, vision_dim)

        # Project to common sphere
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )
        # Class prototypes on hypersphere (learnable); init as regular simplex
        self.prototypes = nn.Parameter(
            regular_simplex_prototypes(
                num_classes, projection_dim, torch.device("cpu")
            ).clone(),
        )
        # Logits from cosine similarity to prototypes (projected features are L2-normalized)
        # scale for temperature
        self.logit_scale = nn.Parameter(torch.tensor(10.0))
        # Sample-adaptive weight: input = concat(audio_proj, vision_proj), output = w_audio
        self.weight_mlp = nn.Sequential(
            nn.Linear(2 * projection_dim, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, 1),
            nn.Sigmoid(),
        )
        self._dropout = dropout

    def _l2_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x / (x.norm(dim=1, keepdim=True) + 1e-8)

    def forward(
        self,
        audio_input: torch.Tensor,
        vision_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            Fused logits (B, num_classes) using sample-adaptive weights.
        """
        audio_feat = self.audio_encoder(audio_input)
        vision_feat = self.vision_encoder(vision_input)
        audio_proj = self._l2_normalize(self.audio_proj(audio_feat))
        vision_proj = self._l2_normalize(self.vision_proj(vision_feat))
        # Prototype-based logits: cosine similarity to prototypes (both on sphere)
        prototypes = self._l2_normalize(self.prototypes)
        audio_logits = self.logit_scale * (audio_proj @ prototypes.T)
        vision_logits = self.logit_scale * (vision_proj @ prototypes.T)

        # Sample-adaptive weight from concatenated projected features
        concat_proj = torch.cat([audio_proj, vision_proj], dim=1)
        w_audio = self.weight_mlp(concat_proj)
        w_vision = 1.0 - w_audio
        fused_logits = w_audio * audio_logits + w_vision * vision_logits
        return fused_logits

    def get_weights_and_logits(
        self,
        audio_input: torch.Tensor,
        vision_input: torch.Tensor,
    ):
        """Return w_audio, w_vision, audio_logits, vision_logits, fused_logits (for analysis)."""
        audio_feat = self.audio_encoder(audio_input)
        vision_feat = self.vision_encoder(vision_input)
        audio_proj = self._l2_normalize(self.audio_proj(audio_feat))
        vision_proj = self._l2_normalize(self.vision_proj(vision_feat))
        prototypes = self._l2_normalize(self.prototypes)
        audio_logits = self.logit_scale * (audio_proj @ prototypes.T)
        vision_logits = self.logit_scale * (vision_proj @ prototypes.T)
        concat_proj = torch.cat([audio_proj, vision_proj], dim=1)
        w_audio = self.weight_mlp(concat_proj)
        w_vision = 1.0 - w_audio
        fused_logits = w_audio * audio_logits + w_vision * vision_logits
        return w_audio, w_vision, audio_logits, vision_logits, fused_logits

    @property
    def feature_dim(self) -> int:
        return 2 * self.projection_dim
