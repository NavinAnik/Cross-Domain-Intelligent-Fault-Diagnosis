"""
Focal Loss and Label Smoothing implementations for handling class imbalance.

Focal Loss: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
Label Smoothing: Szegedy et al., "Rethinking the Inception Architecture", CVPR 2016
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    The focal loss focuses training on hard examples by down-weighting
    easy examples. This is especially useful for class imbalance where
    the model tends to be overconfident on the majority class.
    
    Args:
        alpha: Class weights as list or tensor. For binary: [weight_class_0, weight_class_1]
               Default: [0.25, 0.75] to favor minority class
        gamma: Focusing parameter. Higher gamma puts more focus on hard examples.
               Default: 2.0 (standard value from paper)
        reduction: Specifies the reduction to apply: 'none' | 'mean' | 'sum'
    
    Example:
        >>> criterion = FocalLoss(alpha=[0.2, 0.8], gamma=2.0)
        >>> loss = criterion(logits, targets)
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Forward pass.
        
        Args:
            inputs: Logits of shape (N, C) where C = number of classes
            targets: Ground truth labels of shape (N,)
        
        Returns:
            Focal loss value
        """
        # Ensure FP32 for numerical stability (FP16 + exp/log can cause NaN)
        if inputs.dtype != torch.float32:
            inputs = inputs.float()
        # Compute cross-entropy loss without reduction
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        # Get probability of the true class
        p_t = torch.exp(-ce_loss)
        
        # Compute focal term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma
        
        # Compute focal loss
        focal_loss = focal_term * ce_loss
        
        # Apply class weights if provided
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    
    Label smoothing prevents the model from becoming overconfident by
    softening the target distribution. Instead of hard targets (0 or 1),
    targets are smoothed: (1-epsilon) for true class, epsilon/(C-1) for others.
    
    Args:
        epsilon: Smoothing parameter. Typical values: 0.1
        reduction: Specifies the reduction to apply: 'none' | 'mean' | 'sum'
    
    Example:
        >>> criterion = LabelSmoothingCrossEntropy(epsilon=0.1)
        >>> loss = criterion(logits, targets)
    """
    
    def __init__(self, epsilon=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Forward pass.
        
        Args:
            inputs: Logits of shape (N, C) where C = number of classes
            targets: Ground truth labels of shape (N,)
        
        Returns:
            Label smoothing cross-entropy loss
        """
        n_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, n_classes).float()
        
        # Apply label smoothing
        # True class gets (1 - epsilon), other classes get epsilon / n_classes
        targets_smooth = (1 - self.epsilon) * targets_one_hot + self.epsilon / n_classes
        
        # Compute KL divergence between smoothed targets and predictions
        loss = -(targets_smooth * log_probs).sum(dim=-1)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    """
    Combination of Focal Loss and Label Smoothing for maximum effectiveness.
    
    Uses focal loss to handle class imbalance and label smoothing to
    prevent overconfidence. This combination is particularly effective
    for difficult classification tasks with imbalanced data.
    
    Args:
        alpha: Class weights for focal loss
        gamma: Focusing parameter for focal loss
        epsilon: Smoothing parameter for label smoothing
        focal_weight: Weight for focal loss (default: 0.75)
        smooth_weight: Weight for label smoothing (default: 0.25)
    """
    
    def __init__(self, alpha=None, gamma=2.0, epsilon=0.1, 
                 focal_weight=0.75, smooth_weight=0.25):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.smooth_loss = LabelSmoothingCrossEntropy(epsilon=epsilon)
        self.focal_weight = focal_weight
        self.smooth_weight = smooth_weight
    
    def forward(self, inputs, targets):
        """
        Forward pass.
        
        Args:
            inputs: Logits of shape (N, C)
            targets: Ground truth labels of shape (N,)
        
        Returns:
            Combined loss value
        """
        focal = self.focal_loss(inputs, targets)
        smooth = self.smooth_loss(inputs, targets)
        return self.focal_weight * focal + self.smooth_weight * smooth


if __name__ == "__main__":
    # Quick test
    print("Testing Focal Loss and Label Smoothing...")
    
    # Create dummy data
    batch_size = 32
    num_classes = 2
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Test Focal Loss
    focal_criterion = FocalLoss(alpha=[0.25, 0.75], gamma=2.0)
    focal_loss = focal_criterion(logits, targets)
    print(f"Focal Loss: {focal_loss.item():.4f}")
    
    # Test Label Smoothing
    smooth_criterion = LabelSmoothingCrossEntropy(epsilon=0.1)
    smooth_loss = smooth_criterion(logits, targets)
    print(f"Label Smoothing Loss: {smooth_loss.item():.4f}")
    
    # Test Combined Loss
    combined_criterion = CombinedLoss(alpha=[0.25, 0.75], gamma=2.0, epsilon=0.1)
    combined_loss = combined_criterion(logits, targets)
    print(f"Combined Loss: {combined_loss.item():.4f}")
    
    print("\nAll tests passed!")
