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


class UncertaintyProMoBal(ProMoBalFusion):
    """
    ProMoBal with Monte Carlo Dropout for uncertainty estimation.
    
    Extends ProMoBal to provide:
    - Epistemic uncertainty via MC Dropout
    - Confidence-weighted fusion
    - Reject option for uncertain samples
    
    This enhancement is particularly useful for anomaly detection where
    we may want to flag uncertain predictions for manual review.
    """

    def __init__(
        self,
        audio_encoder: nn.Module,
        vision_encoder: nn.Module,
        num_classes: int = 2,
        projection_dim: int = 256,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.2,
        mc_dropout_rate: float = 0.1,
    ):
        super().__init__(
            audio_encoder=audio_encoder,
            vision_encoder=vision_encoder,
            num_classes=num_classes,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.mc_dropout_rate = mc_dropout_rate
        
        self.mc_dropout_audio = nn.Dropout(mc_dropout_rate)
        self.mc_dropout_vision = nn.Dropout(mc_dropout_rate)
        self.mc_dropout_fusion = nn.Dropout(mc_dropout_rate)
        
        h = hidden_dim or min(256, projection_dim)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(2 * projection_dim, h),
            nn.ReLU(),
            nn.Linear(h, num_classes),
            nn.Softplus(),
        )
        
        self._last_uncertainty = None

    def forward_with_uncertainty(
        self,
        audio_input: torch.Tensor,
        vision_input: torch.Tensor,
        n_samples: int = 10,
    ):
        """
        Forward pass with Monte Carlo Dropout uncertainty estimation.
        
        Args:
            audio_input: Audio spectrogram.
            vision_input: Image tensor.
            n_samples: Number of stochastic forward passes.
        
        Returns:
            mean_logits: Mean predictions across samples.
            epistemic_uncertainty: Epistemic (model) uncertainty.
            aleatoric_uncertainty: Aleatoric (data) uncertainty.
            all_logits: All sampled logits for further analysis.
        """
        was_training = self.training
        self.train()
        
        all_logits = []
        all_weights_audio = []
        all_weights_vision = []
        
        for _ in range(n_samples):
            audio_feat = self.audio_encoder(audio_input)
            vision_feat = self.vision_encoder(vision_input)
            
            audio_feat = self.mc_dropout_audio(audio_feat)
            vision_feat = self.mc_dropout_vision(vision_feat)
            
            audio_proj = self._l2_normalize(self.audio_proj(audio_feat))
            vision_proj = self._l2_normalize(self.vision_proj(vision_feat))
            
            audio_proj = self.mc_dropout_fusion(audio_proj)
            vision_proj = self.mc_dropout_fusion(vision_proj)
            
            prototypes = self._l2_normalize(self.prototypes)
            audio_logits = self.logit_scale * (audio_proj @ prototypes.T)
            vision_logits = self.logit_scale * (vision_proj @ prototypes.T)
            
            concat_proj = torch.cat([audio_proj, vision_proj], dim=1)
            w_audio = self.weight_mlp(concat_proj)
            w_vision = 1.0 - w_audio
            
            fused_logits = w_audio * audio_logits + w_vision * vision_logits
            all_logits.append(fused_logits)
            all_weights_audio.append(w_audio)
            all_weights_vision.append(w_vision)
        
        if not was_training:
            self.eval()
        
        all_logits = torch.stack(all_logits, dim=0)
        mean_logits = all_logits.mean(dim=0)
        epistemic_uncertainty = all_logits.std(dim=0)
        
        with torch.no_grad():
            audio_feat = self.audio_encoder(audio_input)
            vision_feat = self.vision_encoder(vision_input)
            audio_proj = self._l2_normalize(self.audio_proj(audio_feat))
            vision_proj = self._l2_normalize(self.vision_proj(vision_feat))
            concat_proj = torch.cat([audio_proj, vision_proj], dim=1)
            aleatoric_uncertainty = self.uncertainty_head(concat_proj)
        
        self._last_uncertainty = {
            "epistemic": epistemic_uncertainty,
            "aleatoric": aleatoric_uncertainty,
            "mean_weight_audio": torch.stack(all_weights_audio).mean(dim=0),
            "weight_std_audio": torch.stack(all_weights_audio).std(dim=0),
        }
        
        return mean_logits, epistemic_uncertainty, aleatoric_uncertainty, all_logits

    def predict_with_confidence(
        self,
        audio_input: torch.Tensor,
        vision_input: torch.Tensor,
        n_samples: int = 10,
    ):
        """
        Get predictions with confidence scores.
        
        Returns:
            predictions: Predicted class labels.
            confidences: Per-sample confidence scores (higher = more confident).
            uncertainties: Total uncertainty (epistemic + aleatoric).
        """
        mean_logits, epistemic, aleatoric, _ = self.forward_with_uncertainty(
            audio_input, vision_input, n_samples
        )
        
        probs = torch.softmax(mean_logits, dim=-1)
        confidences, predictions = probs.max(dim=-1)
        
        total_uncertainty = epistemic.mean(dim=-1) + aleatoric.mean(dim=-1)
        
        confidence_adjusted = confidences * torch.exp(-total_uncertainty * 0.5)
        
        return predictions, confidence_adjusted, total_uncertainty

    def predict_with_rejection(
        self,
        audio_input: torch.Tensor,
        vision_input: torch.Tensor,
        uncertainty_threshold: float = 0.5,
        confidence_threshold: float = 0.7,
        n_samples: int = 10,
    ):
        """
        Predict with rejection option for uncertain samples.
        
        Samples are rejected if:
        - Total uncertainty > uncertainty_threshold, OR
        - Confidence < confidence_threshold
        
        Args:
            audio_input: Audio spectrogram.
            vision_input: Image tensor.
            uncertainty_threshold: Max allowable uncertainty.
            confidence_threshold: Min required confidence.
            n_samples: Number of MC samples.
        
        Returns:
            predictions: Predicted classes (-1 for rejected samples).
            confidences: Raw confidence scores.
            rejection_reasons: Reason for rejection ('none', 'uncertainty', 'confidence').
        """
        predictions, confidences, uncertainties = self.predict_with_confidence(
            audio_input, vision_input, n_samples
        )
        
        batch_size = predictions.size(0)
        rejection_reasons = ['none'] * batch_size
        final_predictions = predictions.clone()
        
        uncertain_mask = uncertainties > uncertainty_threshold
        low_conf_mask = confidences < confidence_threshold
        
        for i in range(batch_size):
            if uncertain_mask[i]:
                final_predictions[i] = -1
                rejection_reasons[i] = 'uncertainty'
            elif low_conf_mask[i]:
                final_predictions[i] = -1
                rejection_reasons[i] = 'confidence'
        
        return final_predictions, confidences, rejection_reasons

    def get_calibrated_probabilities(
        self,
        audio_input: torch.Tensor,
        vision_input: torch.Tensor,
        n_samples: int = 20,
        temperature: float = 1.0,
    ):
        """
        Get calibrated probability estimates using MC Dropout ensemble.
        
        Returns:
            calibrated_probs: Temperature-scaled mean probabilities.
            predictive_entropy: Entropy of the predictive distribution.
        """
        _, _, _, all_logits = self.forward_with_uncertainty(
            audio_input, vision_input, n_samples
        )
        
        all_probs = torch.softmax(all_logits / temperature, dim=-1)
        calibrated_probs = all_probs.mean(dim=0)
        
        predictive_entropy = -(calibrated_probs * torch.log(calibrated_probs + 1e-10)).sum(dim=-1)
        
        return calibrated_probs, predictive_entropy

    def get_last_uncertainty(self):
        """Return uncertainty breakdown from last forward pass."""
        return self._last_uncertainty


class ConfidenceWeightedProMoBal(UncertaintyProMoBal):
    """
    ProMoBal with confidence-weighted modality fusion.
    
    Weights each modality based on its prediction confidence,
    giving more weight to the more confident modality.
    """

    def forward(
        self,
        audio_input: torch.Tensor,
        vision_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with confidence-weighted fusion.
        
        Instead of learning fixed weights, dynamically weight modalities
        based on their prediction confidence.
        """
        audio_feat = self.audio_encoder(audio_input)
        vision_feat = self.vision_encoder(vision_input)
        
        audio_proj = self._l2_normalize(self.audio_proj(audio_feat))
        vision_proj = self._l2_normalize(self.vision_proj(vision_feat))
        
        prototypes = self._l2_normalize(self.prototypes)
        audio_logits = self.logit_scale * (audio_proj @ prototypes.T)
        vision_logits = self.logit_scale * (vision_proj @ prototypes.T)
        
        audio_conf = torch.softmax(audio_logits, dim=-1).max(dim=-1, keepdim=True)[0]
        vision_conf = torch.softmax(vision_logits, dim=-1).max(dim=-1, keepdim=True)[0]
        
        total_conf = audio_conf + vision_conf + 1e-8
        w_audio = audio_conf / total_conf
        w_vision = vision_conf / total_conf
        
        concat_proj = torch.cat([audio_proj, vision_proj], dim=1)
        w_learned = self.weight_mlp(concat_proj)
        
        w_audio_final = 0.5 * w_audio + 0.5 * w_learned
        w_vision_final = 0.5 * w_vision + 0.5 * (1 - w_learned)
        
        fused_logits = w_audio_final * audio_logits + w_vision_final * vision_logits
        
        return fused_logits
