"""
Mixup and CutMix data augmentation for improved model generalization.

Mixup: Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018
CutMix: Yun et al., "CutMix: Regularization Strategy to Train Strong Classifiers", ICCV 2019
"""

import numpy as np
import torch


def mixup_data(x, y, alpha=0.2):
    """
    Apply Mixup augmentation to a batch of images and labels.
    
    Mixup creates virtual training examples by mixing pairs of examples
    and their labels: mixed_x = lambda * x_i + (1 - lambda) * x_j
    
    Args:
        x: Input images of shape (B, C, H, W)
        y: Labels of shape (B,)
        alpha: Beta distribution parameter. Higher alpha = more mixing.
               Typical values: 0.2-0.4 for images
    
    Returns:
        mixed_x: Mixed images
        y_a, y_b: Original labels for the pair
        lam: Mixing coefficient
    
    Example:
        >>> mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2)
        >>> loss = lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    
    # Random permutation for mixing pairs
    index = torch.randperm(batch_size).to(x.device)
    
    # Mix images
    mixed_x = lam * x + (1 - lam) * x[index]
    
    # Return both labels for mixed loss computation
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """
    Apply CutMix augmentation to a batch of images and labels.
    
    CutMix cuts and pastes patches between training images. The ground
    truth labels are mixed proportionally to the area of the patches.
    
    Args:
        x: Input images of shape (B, C, H, W)
        y: Labels of shape (B,)
        alpha: Beta distribution parameter. Typical value: 1.0
    
    Returns:
        mixed_x: Images with cut and pasted patches
        y_a, y_b: Original labels for the pair
        lam: Mixing coefficient based on cut area
    
    Example:
        >>> mixed_x, y_a, y_b, lam = cutmix_data(x, y, alpha=1.0)
        >>> loss = lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    
    # Random permutation for mixing pairs
    index = torch.randperm(batch_size).to(x.device)
    
    # Get image dimensions
    _, _, h, w = x.shape
    
    # Generate random bounding box
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)
    
    # Uniform sampling of box center
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    
    # Bounding box coordinates
    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)
    
    # Apply CutMix: cut and paste the patch
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda to exactly match the ratio of cut area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
    
    y_a, y_b = y, y[index]
    
    return x, y_a, y_b, lam


def rand_bbox(size, lam):
    """
    Generate random bounding box for CutMix.
    
    Args:
        size: Image size (h, w)
        lam: Lambda parameter determining box size
    
    Returns:
        Bounding box coordinates (x1, y1, x2, y2)
    """
    h, w = size
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)
    
    # Uniform
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)
    
    return bbx1, bby1, bbx2, bby2


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Compute mixed loss for Mixup/CutMix.
    
    Args:
        criterion: Loss function (e.g., nn.CrossEntropyLoss())
        pred: Model predictions
        y_a: First set of labels
        y_b: Second set of labels
        lam: Mixing coefficient
    
    Returns:
        Mixed loss value
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class MixupCutmixTransform:
    """
    Combined Mixup and CutMix transform that randomly applies one or the other.
    
    Args:
        mixup_alpha: Alpha for mixup (default: 0.2)
        cutmix_alpha: Alpha for cutmix (default: 1.0)
        cutmix_prob: Probability of applying cutmix vs mixup (default: 0.5)
        switch_prob: Probability of applying any augmentation (default: 0.5)
    
    Example:
        >>> transform = MixupCutmixTransform(mixup_alpha=0.2, cutmix_alpha=1.0)
        >>> mixed_x, y_a, y_b, lam = transform(x, y)
        >>> loss = mixup_criterion(criterion, pred, y_a, y_b, lam)
    """
    
    def __init__(self, mixup_alpha=0.2, cutmix_alpha=1.0, 
                 cutmix_prob=0.5, switch_prob=0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_prob = cutmix_prob
        self.switch_prob = switch_prob
    
    def __call__(self, x, y):
        """
        Apply mixup or cutmix randomly.
        
        Args:
            x: Input images
            y: Labels
        
        Returns:
            mixed_x, y_a, y_b, lam (or x, y, y, 1.0 if not applied)
        """
        # Decide whether to apply augmentation
        if np.random.rand() > self.switch_prob:
            return x, y, y, 1.0
        
        # Randomly choose between Mixup and CutMix
        if np.random.rand() < self.cutmix_prob:
            return cutmix_data(x, y, alpha=self.cutmix_alpha)
        else:
            return mixup_data(x, y, alpha=self.mixup_alpha)


if __name__ == "__main__":
    # Quick test
    print("Testing Mixup and CutMix...")
    
    # Create dummy data
    batch_size = 8
    x = torch.randn(batch_size, 3, 224, 224)
    y = torch.randint(0, 2, (batch_size,))
    
    print(f"Original shapes: x={x.shape}, y={y.shape}")
    
    # Test Mixup
    mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2)
    print(f"\nMixup:")
    print(f"  Mixed x shape: {mixed_x.shape}")
    print(f"  Lambda: {lam:.3f}")
    print(f"  y_a: {y_a[:4].tolist()}")
    print(f"  y_b: {y_b[:4].tolist()}")
    
    # Test CutMix
    cut_x, y_a, y_b, lam = cutmix_data(x, y, alpha=1.0)
    print(f"\nCutMix:")
    print(f"  Cut x shape: {cut_x.shape}")
    print(f"  Lambda: {lam:.3f}")
    print(f"  y_a: {y_a[:4].tolist()}")
    print(f"  y_b: {y_b[:4].tolist()}")
    
    # Test Combined Transform
    transform = MixupCutmixTransform(mixup_alpha=0.2, cutmix_alpha=1.0)
    aug_x, y_a, y_b, lam = transform(x, y)
    print(f"\nCombined Transform:")
    print(f"  Augmented x shape: {aug_x.shape}")
    print(f"  Lambda: {lam:.3f}")
    
    print("\nAll tests passed!")
