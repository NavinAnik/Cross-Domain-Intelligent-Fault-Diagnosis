"""
Hierarchical Temporal-Spectral Fusion (HTSF) for Audio-Visual Anomaly Detection.

Novel architecture for Q1 journal publication that addresses the specific
challenge of multimodal industrial machine anomaly detection.

Key Contributions:
1. Temporal Attention Pooling: Learns to focus on discriminative temporal segments
2. Spatial Attention Pooling: Learns to focus on relevant spatial regions
3. Cross-Modal Transformer: Bidirectional attention between modalities
4. Anomaly-Aware Classifier: Special attention to anomaly patterns

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │ Audio Branch                    Vision Branch               │
    │ ┌───────────┐                  ┌───────────┐                │
    │ │AST Encoder│                  │DINOv2/EVA02│               │
    │ └─────┬─────┘                  └─────┬─────┘                │
    │       │                              │                      │
    │ ┌─────▼─────┐                  ┌─────▼─────┐                │
    │ │Temporal   │                  │Spatial    │                │
    │ │Attention  │                  │Attention  │                │
    │ │Pooling    │                  │Pooling    │                │
    │ └─────┬─────┘                  └─────┬─────┘                │
    │       │         ┌───────────┐        │                      │
    │       └────────►│Cross-Modal│◄───────┘                      │
    │                 │Transformer│                               │
    │                 └─────┬─────┘                               │
    │                       │                                     │
    │                 ┌─────▼─────┐                               │
    │                 │Anomaly-   │                               │
    │                 │Aware      │                               │
    │                 │Classifier │                               │
    │                 └───────────┘                               │
    └─────────────────────────────────────────────────────────────┘
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_feature_dim(encoder: nn.Module) -> int:
    """Extract feature dimension from encoder."""
    return getattr(
        encoder,
        "feature_dim",
        getattr(getattr(encoder, "encoder", None), "feature_dim", 768),
    )


class TemporalAttentionPooling(nn.Module):
    """
    Temporal attention pooling for audio features.
    
    Learns to weight different time steps based on their relevance
    for anomaly detection. Anomalous sounds often occur in specific
    temporal segments.
    """

    def __init__(
        self,
        feature_dim: int,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_heads = n_heads

        self.temporal_query = nn.Parameter(torch.randn(1, 1, feature_dim))
        
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        self.norm = nn.LayerNorm(feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, D) temporal features from audio encoder.
        
        Returns:
            pooled: (B, D) pooled features.
            attention_weights: (B, T) attention weights over time.
        """
        B, T, D = x.shape
        
        query = self.temporal_query.expand(B, -1, -1)
        
        attn_output, attn_weights = self.attention(
            query, x, x, need_weights=True
        )
        
        attn_output = self.norm(attn_output + query)
        
        mlp_output = self.mlp(attn_output)
        output = self.norm2(mlp_output + attn_output)
        
        pooled = output.squeeze(1)
        weights = attn_weights.squeeze(1)
        
        return pooled, weights


class SpatialAttentionPooling(nn.Module):
    """
    Spatial attention pooling for vision features.
    
    Learns to weight different spatial regions based on their relevance.
    Mechanical anomalies often appear in specific spatial locations.
    """

    def __init__(
        self,
        feature_dim: int,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_heads = n_heads

        self.spatial_query = nn.Parameter(torch.randn(1, 1, feature_dim))
        
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        self.norm = nn.LayerNorm(feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, H*W, D) spatial features from vision encoder.
        
        Returns:
            pooled: (B, D) pooled features.
            attention_weights: (B, H*W) attention weights over space.
        """
        B, S, D = x.shape
        
        query = self.spatial_query.expand(B, -1, -1)
        
        attn_output, attn_weights = self.attention(
            query, x, x, need_weights=True
        )
        
        attn_output = self.norm(attn_output + query)
        
        mlp_output = self.mlp(attn_output)
        output = self.norm2(mlp_output + attn_output)
        
        pooled = output.squeeze(1)
        weights = attn_weights.squeeze(1)
        
        return pooled, weights


class CrossModalTransformer(nn.Module):
    """
    Cross-modal transformer for bidirectional attention between audio and vision.
    
    Enables each modality to attend to relevant features from the other,
    capturing complementary information for anomaly detection.
    """

    def __init__(
        self,
        audio_dim: int,
        vision_dim: int,
        hidden_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        
        self.audio_to_vision = nn.ModuleList([
            CrossAttentionLayer(hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.vision_to_audio = nn.ModuleList([
            CrossAttentionLayer(hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        audio_feat: torch.Tensor,
        vision_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            audio_feat: (B, D_a) or (B, T, D_a) audio features.
            vision_feat: (B, D_v) or (B, S, D_v) vision features.
        
        Returns:
            fused: (B, hidden_dim) fused representation.
            attention_info: dict with cross-attention weights.
        """
        if audio_feat.dim() == 2:
            audio_feat = audio_feat.unsqueeze(1)
        if vision_feat.dim() == 2:
            vision_feat = vision_feat.unsqueeze(1)
        
        audio = self.audio_proj(audio_feat)
        vision = self.vision_proj(vision_feat)
        
        a2v_attn_weights = []
        v2a_attn_weights = []
        
        audio_enhanced = audio
        vision_enhanced = vision
        
        for a2v, v2a in zip(self.audio_to_vision, self.vision_to_audio):
            audio_enhanced, a2v_attn = a2v(audio_enhanced, vision_enhanced)
            vision_enhanced, v2a_attn = v2a(vision_enhanced, audio_enhanced)
            a2v_attn_weights.append(a2v_attn)
            v2a_attn_weights.append(v2a_attn)
        
        audio_pooled = audio_enhanced.mean(dim=1)
        vision_pooled = vision_enhanced.mean(dim=1)
        audio_orig = audio.mean(dim=1)
        vision_orig = vision.mean(dim=1)
        
        concat = torch.cat([
            audio_pooled, vision_pooled,
            audio_orig, vision_orig
        ], dim=-1)
        
        fused = self.fusion(concat)
        fused = self.output_norm(fused)
        
        attention_info = {
            "audio_to_vision": a2v_attn_weights,
            "vision_to_audio": v2a_attn_weights,
        }
        
        return fused, attention_info


class CrossAttentionLayer(nn.Module):
    """Single cross-attention layer with residual connection."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self, query: torch.Tensor, key_value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.attention(
            query, key_value, key_value, need_weights=True
        )
        x = self.norm1(query + attn_out)
        x = self.norm2(x + self.mlp(x))
        return x, attn_weights


class AnomalyAwareClassifier(nn.Module):
    """
    Anomaly-aware classification head.
    
    Incorporates anomaly-specific attention mechanisms and multi-scale
    feature processing for improved anomaly detection.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        hidden_dims: list = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [512, 256, 128]
        
        self.anomaly_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid(),
        )
        
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers)
        
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, 1),
            nn.Softplus(),
        )
        
        self._features_before_head = None

    def forward(
        self, x: torch.Tensor, return_uncertainty: bool = False
    ) -> torch.Tensor:
        attention = self.anomaly_attention(x)
        x_attended = x * attention + x
        
        for i, layer in enumerate(self.classifier[:-1]):
            x_attended = layer(x_attended)
            if i == len(self.classifier) - 2:
                self._features_before_head = x_attended
        
        logits = self.classifier[-1](x_attended)
        
        if return_uncertainty:
            uncertainty = self.uncertainty_head(self._features_before_head)
            return logits, uncertainty
        
        return logits


class HTSFFusion(nn.Module):
    """
    Hierarchical Temporal-Spectral Fusion for multimodal anomaly detection.
    
    This is the main novel contribution combining:
    - Temporal attention pooling for audio
    - Spatial attention pooling for vision
    - Cross-modal transformer for multimodal fusion
    - Anomaly-aware classifier
    
    Args:
        audio_encoder: Pre-trained audio encoder (e.g., AST).
        vision_encoder: Pre-trained vision encoder (e.g., DINOv2).
        num_classes: Number of output classes (default: 2).
        hidden_dim: Hidden dimension for cross-modal transformer.
        n_heads: Number of attention heads.
        n_cross_layers: Number of cross-attention layers.
        dropout: Dropout rate.
        freeze_encoders: Whether to freeze encoder weights.
    """

    def __init__(
        self,
        audio_encoder: nn.Module,
        vision_encoder: nn.Module,
        num_classes: int = 2,
        hidden_dim: int = 512,
        n_heads: int = 8,
        n_cross_layers: int = 2,
        dropout: float = 0.1,
        freeze_encoders: bool = True,
    ):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.vision_encoder = vision_encoder
        
        audio_dim = _get_feature_dim(audio_encoder)
        vision_dim = _get_feature_dim(vision_encoder)
        
        if freeze_encoders:
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
        
        self.temporal_pool = TemporalAttentionPooling(
            audio_dim, n_heads=n_heads, dropout=dropout
        )
        self.spatial_pool = SpatialAttentionPooling(
            vision_dim, n_heads=n_heads, dropout=dropout
        )
        
        self.cross_modal = CrossModalTransformer(
            audio_dim=audio_dim,
            vision_dim=vision_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_cross_layers,
            dropout=dropout,
        )
        
        self.classifier = AnomalyAwareClassifier(
            input_dim=hidden_dim,
            num_classes=num_classes,
            hidden_dims=[hidden_dim, hidden_dim // 2, hidden_dim // 4],
            dropout=dropout,
        )
        
        self.hidden_dim = hidden_dim
        self._attention_info = None

    def forward(
        self,
        audio_input: torch.Tensor,
        vision_input: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through HTSF.
        
        Args:
            audio_input: Audio spectrogram or waveform.
            vision_input: Image tensor.
            return_attention: Whether to return attention weights.
        
        Returns:
            logits: (B, num_classes) classification logits.
            attention_info (optional): dict with attention weights.
        """
        audio_feat = self.audio_encoder(audio_input)
        vision_feat = self.vision_encoder(vision_input)
        
        if audio_feat.dim() == 2:
            audio_feat = audio_feat.unsqueeze(1)
        if vision_feat.dim() == 2:
            vision_feat = vision_feat.unsqueeze(1)
        
        audio_pooled, temporal_attn = self.temporal_pool(audio_feat)
        vision_pooled, spatial_attn = self.spatial_pool(vision_feat)
        
        fused, cross_attn = self.cross_modal(audio_pooled, vision_pooled)
        
        logits = self.classifier(fused)
        
        self._attention_info = {
            "temporal_attention": temporal_attn,
            "spatial_attention": spatial_attn,
            "cross_modal": cross_attn,
        }
        
        if return_attention:
            return logits, self._attention_info
        
        return logits

    def get_attention_maps(self) -> dict:
        """Return last computed attention maps for visualization."""
        return self._attention_info

    @property
    def feature_dim(self) -> int:
        return self.hidden_dim


class HTSFWithUncertainty(HTSFFusion):
    """
    HTSF with uncertainty estimation using Monte Carlo Dropout.
    
    Extends HTSF to provide prediction uncertainty estimates.
    """

    def forward_with_uncertainty(
        self,
        audio_input: torch.Tensor,
        vision_input: torch.Tensor,
        n_samples: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with uncertainty estimation.
        
        Args:
            audio_input: Audio spectrogram.
            vision_input: Image tensor.
            n_samples: Number of stochastic forward passes.
        
        Returns:
            mean_logits: Mean predictions across samples.
            uncertainty: Prediction uncertainty (std).
            all_logits: All sampled logits.
        """
        self.train()
        
        all_logits = []
        for _ in range(n_samples):
            logits = self.forward(audio_input, vision_input)
            all_logits.append(logits)
        
        all_logits = torch.stack(all_logits)
        mean_logits = all_logits.mean(dim=0)
        uncertainty = all_logits.std(dim=0)
        
        return mean_logits, uncertainty, all_logits

    def predict_with_rejection(
        self,
        audio_input: torch.Tensor,
        vision_input: torch.Tensor,
        uncertainty_threshold: float = 0.3,
        n_samples: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict with rejection option for uncertain samples.
        
        Args:
            audio_input: Audio spectrogram.
            vision_input: Image tensor.
            uncertainty_threshold: Threshold for rejection.
            n_samples: Number of stochastic forward passes.
        
        Returns:
            predictions: Predicted classes (-1 for rejected).
            confidences: Prediction confidences.
            uncertainties: Per-sample uncertainties.
        """
        mean_logits, uncertainty, _ = self.forward_with_uncertainty(
            audio_input, vision_input, n_samples
        )
        
        probs = F.softmax(mean_logits, dim=-1)
        confidences, predictions = probs.max(dim=-1)
        
        max_uncertainty = uncertainty.max(dim=-1)[0]
        
        rejected = max_uncertainty > uncertainty_threshold
        predictions[rejected] = -1
        
        return predictions, confidences, max_uncertainty
