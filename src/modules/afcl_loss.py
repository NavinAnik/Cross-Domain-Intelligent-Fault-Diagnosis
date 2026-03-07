"""
Asymmetric Focal Contrastive Loss (AFCL) for Imbalanced Anomaly Detection.

Novel loss function combining:
1. Focal Loss: Down-weights easy examples, focuses on hard ones
2. Asymmetric Contrastive: Harder penalties on minority class pairs
3. Class-aware InfoNCE: Modified contrastive loss aware of class labels

This addresses the severe class imbalance in anomaly detection by:
- Preventing the majority class from dominating the loss
- Encouraging tighter clusters for the minority (abnormal) class
- Maintaining separation between normal and abnormal representations

Mathematical Formulation:
    L_AFCL = L_focal + λ_contrast * L_asymmetric_contrastive + λ_align * L_alignment

Where:
    L_focal = -α * (1-p)^γ * log(p)  [standard focal loss]
    L_asymmetric_contrastive = weighted InfoNCE with class-aware weights
    L_alignment = cosine alignment between representations and class prototypes
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossComponent(nn.Module):
    """
    Focal Loss component of AFCL.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Where α is class weight and γ controls focus on hard examples.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            focal_weight = alpha_t * focal_weight
        
        loss = focal_weight * ce
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class AsymmetricContrastiveLoss(nn.Module):
    """
    Asymmetric contrastive loss that weights pairs based on class membership.
    
    Key insight: In anomaly detection, we want:
    - Abnormal-abnormal pairs: Strong positive pull (tight cluster)
    - Normal-abnormal pairs: Strong negative push (separation)
    - Normal-normal pairs: Moderate positive pull
    
    This asymmetry helps the minority class form a distinct, tight cluster.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        abnormal_weight: float = 2.0,
        base_weight: float = 1.0,
        abnormal_class: int = 1,
    ):
        super().__init__()
        self.temperature = temperature
        self.abnormal_weight = abnormal_weight
        self.base_weight = base_weight
        self.abnormal_class = abnormal_class

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: (B, D) L2-normalized feature vectors.
            labels: (B,) class labels.
        
        Returns:
            Asymmetric contrastive loss.
        """
        features = F.normalize(features, dim=1)
        batch_size = features.size(0)
        
        sim = features @ features.T / self.temperature
        
        mask_diag = ~torch.eye(batch_size, dtype=torch.bool, device=features.device)
        
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        pos_mask = labels_eq & mask_diag
        neg_mask = ~labels_eq
        
        is_abnormal = (labels == self.abnormal_class).float()
        abnormal_pair = is_abnormal.unsqueeze(0) * is_abnormal.unsqueeze(1)
        
        pos_weight = self.base_weight + (self.abnormal_weight - self.base_weight) * abnormal_pair
        neg_weight = self.base_weight + (self.abnormal_weight - self.base_weight) * (
            is_abnormal.unsqueeze(0) + is_abnormal.unsqueeze(1)
        ).clamp(max=1.0)
        
        exp_sim = torch.exp(sim)
        
        neg_sum = (exp_sim * neg_mask.float() * neg_weight).sum(dim=1, keepdim=True)
        
        log_prob = sim - torch.log(exp_sim + neg_sum + 1e-8)
        
        pos_count = pos_mask.float().sum(dim=1).clamp(min=1)
        weighted_log_prob = (log_prob * pos_mask.float() * pos_weight).sum(dim=1) / pos_count
        
        loss = -weighted_log_prob.mean()
        
        return loss


class PrototypeAlignmentLoss(nn.Module):
    """
    Loss encouraging features to align with learnable class prototypes.
    
    Helps form well-separated class clusters in the feature space.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int = 2,
        margin: float = 0.5,
    ):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, feature_dim))
        nn.init.xavier_uniform_(self.prototypes)
        self.margin = margin
        self.num_classes = num_classes

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: (B, D) feature vectors.
            labels: (B,) class labels.
        
        Returns:
            Prototype alignment loss.
        """
        features = F.normalize(features, dim=1)
        prototypes = F.normalize(self.prototypes, dim=1)
        
        sim = features @ prototypes.T
        
        batch_size = features.size(0)
        correct_sim = sim[torch.arange(batch_size), labels]
        
        wrong_mask = torch.ones_like(sim, dtype=torch.bool)
        wrong_mask[torch.arange(batch_size), labels] = False
        wrong_sim = sim.masked_fill(~wrong_mask, -float('inf')).max(dim=1)[0]
        
        loss = F.relu(wrong_sim - correct_sim + self.margin)
        
        return loss.mean()


class AsymmetricFocalContrastiveLoss(nn.Module):
    """
    Full AFCL Loss combining focal, contrastive, and alignment components.
    
    L_AFCL = L_focal + λ_contrast * L_contrastive + λ_align * L_alignment
    
    Args:
        feature_dim: Dimension of feature vectors for alignment.
        num_classes: Number of output classes.
        focal_alpha: Class weights for focal loss [normal_weight, abnormal_weight].
        focal_gamma: Focusing parameter for focal loss.
        temperature: Temperature for contrastive loss.
        abnormal_weight: Extra weight for abnormal class pairs.
        lambda_contrast: Weight for contrastive loss term.
        lambda_align: Weight for alignment loss term.
        margin: Margin for prototype alignment.
    """

    def __init__(
        self,
        feature_dim: int = 768,
        num_classes: int = 2,
        focal_alpha: Optional[list] = None,
        focal_gamma: float = 2.0,
        temperature: float = 0.07,
        abnormal_weight: float = 2.0,
        lambda_contrast: float = 0.5,
        lambda_align: float = 0.1,
        margin: float = 0.5,
        abnormal_class: int = 1,
    ):
        super().__init__()
        
        alpha = None
        if focal_alpha is not None:
            alpha = torch.tensor(focal_alpha, dtype=torch.float32)
        
        self.focal = FocalLossComponent(
            alpha=alpha,
            gamma=focal_gamma,
            reduction="mean",
        )
        
        self.contrastive = AsymmetricContrastiveLoss(
            temperature=temperature,
            abnormal_weight=abnormal_weight,
            abnormal_class=abnormal_class,
        )
        
        self.alignment = PrototypeAlignmentLoss(
            feature_dim=feature_dim,
            num_classes=num_classes,
            margin=margin,
        )
        
        self.lambda_contrast = lambda_contrast
        self.lambda_align = lambda_align
        
        self._last_losses = {}

    def forward(
        self,
        features: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor:
        """
        Compute AFCL loss.
        
        Args:
            features: (B, D) feature vectors from encoder.
            logits: (B, C) classification logits.
            labels: (B,) ground truth labels.
            return_components: If True, return individual loss components.
        
        Returns:
            Total AFCL loss (and optionally component losses).
        """
        l_focal = self.focal(logits, labels)
        l_contrastive = self.contrastive(features, labels)
        l_alignment = self.alignment(features, labels)
        
        total_loss = (
            l_focal
            + self.lambda_contrast * l_contrastive
            + self.lambda_align * l_alignment
        )
        
        self._last_losses = {
            "focal": l_focal.item(),
            "contrastive": l_contrastive.item(),
            "alignment": l_alignment.item(),
            "total": total_loss.item(),
        }
        
        if return_components:
            return total_loss, l_focal, l_contrastive, l_alignment
        
        return total_loss

    def get_last_losses(self) -> dict:
        """Return breakdown of last computed losses."""
        return self._last_losses

    def get_prototypes(self) -> torch.Tensor:
        """Return learned class prototypes."""
        return F.normalize(self.alignment.prototypes, dim=1)


class AFCLWithHardMining(AsymmetricFocalContrastiveLoss):
    """
    AFCL with hard negative mining for improved separation.
    
    Extends AFCL to:
    - Identify hard negatives (normal samples close to abnormal cluster)
    - Up-weight loss contribution from hard samples
    - Track mining statistics for analysis
    """

    def __init__(
        self,
        feature_dim: int = 768,
        num_classes: int = 2,
        focal_alpha: Optional[list] = None,
        focal_gamma: float = 2.0,
        temperature: float = 0.07,
        abnormal_weight: float = 2.0,
        lambda_contrast: float = 0.5,
        lambda_align: float = 0.1,
        margin: float = 0.5,
        hard_negative_ratio: float = 0.3,
        hard_weight: float = 2.0,
        abnormal_class: int = 1,
    ):
        super().__init__(
            feature_dim=feature_dim,
            num_classes=num_classes,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            temperature=temperature,
            abnormal_weight=abnormal_weight,
            lambda_contrast=lambda_contrast,
            lambda_align=lambda_align,
            margin=margin,
            abnormal_class=abnormal_class,
        )
        
        self.hard_negative_ratio = hard_negative_ratio
        self.hard_weight = hard_weight
        self.abnormal_class = abnormal_class
        self._hard_negative_stats = {}

    def _identify_hard_negatives(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Identify hard negatives: normal samples closest to abnormal cluster.
        
        Returns:
            Hard negative mask (B,) - True for hard negatives.
        """
        abnormal_mask = labels == self.abnormal_class
        normal_mask = ~abnormal_mask
        
        if abnormal_mask.sum() == 0 or normal_mask.sum() == 0:
            return torch.zeros(labels.size(0), dtype=torch.bool, device=labels.device)
        
        features_norm = F.normalize(features, dim=1)
        
        abnormal_centroid = features_norm[abnormal_mask].mean(dim=0)
        
        normal_features = features_norm[normal_mask]
        distances = 1 - (normal_features @ abnormal_centroid)
        
        k = max(1, int(self.hard_negative_ratio * normal_mask.sum()))
        _, hard_idx = distances.topk(k, largest=False)
        
        hard_mask = torch.zeros(labels.size(0), dtype=torch.bool, device=labels.device)
        normal_indices = torch.where(normal_mask)[0]
        hard_mask[normal_indices[hard_idx]] = True
        
        self._hard_negative_stats = {
            "n_hard": k,
            "n_normal": normal_mask.sum().item(),
            "mean_hard_distance": distances[hard_idx].mean().item(),
            "mean_easy_distance": distances.mean().item(),
        }
        
        return hard_mask

    def forward(
        self,
        features: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor:
        hard_mask = self._identify_hard_negatives(features, labels)
        
        sample_weights = torch.ones(labels.size(0), device=labels.device)
        sample_weights[hard_mask] = self.hard_weight
        sample_weights[labels == self.abnormal_class] = self.hard_weight
        
        ce = F.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce)
        
        if self.focal.alpha is not None:
            alpha_t = self.focal.alpha.to(logits.device)[labels]
        else:
            alpha_t = 1.0
        
        focal_weight = alpha_t * (1 - pt) ** self.focal.gamma
        l_focal = (focal_weight * sample_weights * ce).mean()
        
        l_contrastive = self.contrastive(features, labels)
        l_alignment = self.alignment(features, labels)
        
        total_loss = (
            l_focal
            + self.lambda_contrast * l_contrastive
            + self.lambda_align * l_alignment
        )
        
        self._last_losses = {
            "focal": l_focal.item(),
            "contrastive": l_contrastive.item(),
            "alignment": l_alignment.item(),
            "total": total_loss.item(),
            "hard_negatives": hard_mask.sum().item(),
        }
        
        if return_components:
            return total_loss, l_focal, l_contrastive, l_alignment
        
        return total_loss

    def get_hard_negative_stats(self) -> dict:
        """Return statistics about hard negative mining."""
        return self._hard_negative_stats


def create_afcl_loss(
    feature_dim: int = 768,
    num_classes: int = 2,
    imbalance_ratio: float = 5.0,
    use_hard_mining: bool = True,
) -> nn.Module:
    """
    Factory function to create appropriately configured AFCL loss.
    
    Args:
        feature_dim: Feature dimension from encoder.
        num_classes: Number of classes.
        imbalance_ratio: Ratio of normal to abnormal samples.
        use_hard_mining: Whether to use hard negative mining.
    
    Returns:
        Configured AFCL loss module.
    """
    normal_weight = 1.0 / (1.0 + imbalance_ratio)
    abnormal_weight = imbalance_ratio / (1.0 + imbalance_ratio)
    focal_alpha = [normal_weight, abnormal_weight]
    
    gamma = min(3.0, 1.5 + 0.3 * math.log(imbalance_ratio + 1))
    
    if use_hard_mining:
        return AFCLWithHardMining(
            feature_dim=feature_dim,
            num_classes=num_classes,
            focal_alpha=focal_alpha,
            focal_gamma=gamma,
            temperature=0.07,
            abnormal_weight=2.0,
            lambda_contrast=0.5,
            lambda_align=0.1,
            hard_negative_ratio=0.3,
            hard_weight=2.0,
        )
    else:
        return AsymmetricFocalContrastiveLoss(
            feature_dim=feature_dim,
            num_classes=num_classes,
            focal_alpha=focal_alpha,
            focal_gamma=gamma,
            temperature=0.07,
            abnormal_weight=2.0,
            lambda_contrast=0.5,
            lambda_align=0.1,
        )
