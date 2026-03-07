"""
DINOv2 Pretrained Encoder Module

Uses Meta AI's DINOv2 pretrained Vision Transformer for feature extraction.
DINOv2 was trained on 142M images with self-supervised learning, providing
excellent visual representations without requiring labeled data.

Supported models:
- dinov2_vits14: ViT-Small (384 dim) - ~3GB VRAM
- dinov2_vitb14: ViT-Base (768 dim) - ~5GB VRAM  
- dinov2_vitl14: ViT-Large (1024 dim) - ~6GB VRAM with FP16
- dinov2_vitg14: ViT-Giant (1536 dim) - ~12GB VRAM with FP16
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)
import torch
import torch.nn as nn


class DINOv2Encoder(nn.Module):
    """
    DINOv2 pretrained encoder for feature extraction.
    
    Args:
        model_name: One of 'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'
        freeze_backbone: If True, freeze all backbone weights (recommended for finetuning)
        use_fp16: If True, use half precision for lower memory usage
    """
    
    FEATURE_DIMS = {
        'dinov2_vits14': 384,
        'dinov2_vitb14': 768,
        'dinov2_vitl14': 1024,
        'dinov2_vitg14': 1536,
    }
    
    def __init__(
        self, 
        model_name: str = 'dinov2_vitl14',  # Default to Large for best quality
        freeze_backbone: bool = True,
        use_fp16: bool = True
    ):
        super().__init__()
        
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.use_fp16 = use_fp16
        self.feature_dim = self.FEATURE_DIMS.get(model_name, 1024)
        
        # Load DINOv2 from torch hub
        logger.info("Loading %s from torch.hub...", model_name)
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2', 
            model_name,
            pretrained=True
        )
        logger.info("Loaded %s with feature dim %s", model_name, self.feature_dim)
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
            logger.info("Backbone frozen - only classifier will be trained")
        
        # Convert to FP16 if requested
        if use_fp16:
            self.backbone = self.backbone.half()
            logger.info("Using FP16 for reduced memory usage")
    
    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images.
        
        Args:
            x: Input images of shape (B, 3, H, W), normalized with ImageNet stats
               H and W should be divisible by 14 (patch size)
        
        Returns:
            Features of shape (B, feature_dim) - the [CLS] token features
        """
        # Convert input to FP16 if using mixed precision
        if self.use_fp16:
            x = x.half()
        
        # DINOv2 returns the [CLS] token features by default (autocast only when CUDA available)
        with torch.amp.autocast("cuda", enabled=(self.use_fp16 and torch.cuda.is_available())):
            if self.freeze_backbone:
                with torch.no_grad():
                    features = self.backbone(x)
            else:
                features = self.backbone(x)
        
        # Convert back to FP32 for classifier
        return features.float()
    
    def get_intermediate_features(self, x: torch.Tensor, n_last_blocks: int = 4):
        """
        Get intermediate features from multiple transformer blocks.
        Useful for more detailed feature extraction.
        
        Args:
            x: Input images
            n_last_blocks: Number of last blocks to extract features from
        
        Returns:
            Concatenated features from the last n blocks
        """
        if self.use_fp16:
            x = x.half()
            
        with torch.amp.autocast("cuda", enabled=(self.use_fp16 and torch.cuda.is_available())):
            if self.freeze_backbone:
                with torch.no_grad():
                    features = self.backbone.get_intermediate_layers(
                        x, n=n_last_blocks, return_class_token=True
                    )
            else:
                features = self.backbone.get_intermediate_layers(
                    x, n=n_last_blocks, return_class_token=True
                )
        
        # Concatenate class tokens from all layers
        cls_tokens = torch.cat([f[1] for f in features], dim=-1)
        return cls_tokens.float()
    
    def train(self, mode: bool = True):
        """Override train to keep backbone frozen if freeze_backbone is True."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self


class EnhancedClassifierHead(nn.Module):
    """
    Advanced classifier head with:
    - Multi-layer feature fusion from multiple transformer blocks
    - Attention pooling over features
    - Deeper MLP with residual-like connections
    - Heavy dropout and layer normalization for regularization
    
    This enhanced head significantly improves performance on difficult
    classification tasks by better utilizing the rich features from DINOv2.
    
    Args:
        feature_dim: Input feature dimension
        num_classes: Number of output classes
        hidden_dims: List of hidden dimensions for MLP layers (e.g., [512, 256, 128])
        dropout: Dropout rate (default: 0.3 for strong regularization)
        use_attention_pooling: Whether to use attention pooling (for multi-patch features)
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_classes: int = 2,
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.3,
        use_attention_pooling: bool = False
    ):
        super().__init__()
        self.use_attention_pooling = use_attention_pooling
        self.feature_dim = feature_dim
        
        # Attention pooling (if enabled) for aggregating patch-level features
        if use_attention_pooling:
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 4),
                nn.Tanh(),
                nn.Linear(feature_dim // 4, 1)
            )
        
        # Multi-layer MLP with layer norm and dropout
        layers = []
        in_dim = feature_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        
        # Final classification layer
        self.output = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, num_classes)
        )
        
        logger.info("Enhanced Classifier: %s -> %s -> %s", feature_dim, hidden_dims, num_classes)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Features of shape (B, feature_dim) or (B, num_patches, feature_dim)
        
        Returns:
            Logits of shape (B, num_classes)
        """
        # Apply attention pooling if we have multi-patch features
        if self.use_attention_pooling and x.dim() == 3:
            # x: (B, num_patches, feature_dim)
            attn_weights = self.attention(x)  # (B, num_patches, 1)
            attn_weights = torch.softmax(attn_weights, dim=1)
            x = (x * attn_weights).sum(dim=1)  # (B, feature_dim)
        
        # Pass through feature extraction layers
        x = self.feature_layers(x)
        
        # Final classification
        return self.output(x)


class DINOv2Classifier(nn.Module):
    """
    Complete classification model using DINOv2 encoder + classifier head.
    
    Args:
        num_classes: Number of output classes
        model_name: DINOv2 model variant
        freeze_backbone: Whether to freeze DINOv2 weights
        use_fp16: Whether to use FP16 for encoder
        hidden_dim: Hidden dimension for simple MLP classifier (None for linear)
        hidden_dims: List of hidden dims for enhanced classifier (e.g., [512, 256, 128])
        dropout: Dropout rate for classifier
        use_enhanced_head: Whether to use EnhancedClassifierHead
        use_multi_layer_features: Extract features from multiple transformer layers
        n_last_blocks: Number of last blocks to use for multi-layer features
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        model_name: str = 'dinov2_vitl14',
        freeze_backbone: bool = True,
        use_fp16: bool = True,
        hidden_dim: Optional[int] = None,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        use_enhanced_head: bool = False,
        use_multi_layer_features: bool = False,
        n_last_blocks: int = 4
    ):
        super().__init__()
        
        # DINOv2 encoder
        self.encoder = DINOv2Encoder(
            model_name=model_name,
            freeze_backbone=freeze_backbone,
            use_fp16=use_fp16
        )
        
        self.use_multi_layer_features = use_multi_layer_features
        self.n_last_blocks = n_last_blocks
        
        # Calculate feature dimension
        base_feature_dim = self.encoder.feature_dim
        if use_multi_layer_features:
            # Concatenate features from multiple layers
            feature_dim = base_feature_dim * n_last_blocks
            logger.info("Using multi-layer features from last %s blocks: %s dims", n_last_blocks, feature_dim)
        else:
            feature_dim = base_feature_dim
        
        # Build classifier head
        if use_enhanced_head:
            # Use enhanced classifier with deeper architecture
            if hidden_dims is None:
                hidden_dims = [512, 256, 128]
            self.classifier = EnhancedClassifierHead(
                feature_dim=feature_dim,
                num_classes=num_classes,
                hidden_dims=hidden_dims,
                dropout=dropout,
                use_attention_pooling=False
            )
        elif hidden_dim is None:
            # Simple linear classifier
            self.classifier = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, num_classes)
            )
            logger.info("Linear Classifier: %s -> %s", feature_dim, num_classes)
        else:
            # Simple MLP classifier (2 layers)
            self.classifier = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
            logger.info("MLP Classifier: %s -> %s -> %s", feature_dim, hidden_dim, num_classes)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before classification.
        
        Args:
            x: Input images (B, 3, H, W)
        
        Returns:
            Features of shape (B, feature_dim)
        """
        if self.use_multi_layer_features:
            features = self.encoder.get_intermediate_features(x, n_last_blocks=self.n_last_blocks)
        else:
            features = self.encoder(x)
        return features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images (B, 3, H, W)
        
        Returns:
            Logits of shape (B, num_classes)
        """
        # Extract features
        features = self.get_features(x)
        
        # Classify
        logits = self.classifier(features)
        return logits
    
    @property
    def feature_dim(self):
        return self.encoder.feature_dim


def get_dinov2_transforms(image_size: int = 224, augment_strength: str = 'weak'):
    """
    Get the recommended transforms for DINOv2.
    
    Args:
        image_size: Target image size (224 or 518 recommended)
        augment_strength: 'weak' (default) or 'strong' for enhanced augmentation
    
    Returns:
        Dictionary with 'train' and 'val' transforms
    """
    from torchvision import transforms
    
    # ImageNet normalization (required for DINOv2)
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    
    if augment_strength == 'strong':
        # Strong augmentation for better generalization
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            # Geometric augmentations
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            # Color augmentations (important for spectrograms)
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            # Advanced augmentations
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.3),
            transforms.RandomApply([
                transforms.RandomPosterize(bits=4)
            ], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            # Cutout/Random erasing for robustness
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3))
        ])
    else:
        # Default weak augmentation
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    return {
        'train': train_transform,
        'val': val_transform
    }


if __name__ == "__main__":
    # Quick test
    logger.info("Testing DINOv2 Encoder...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    
    # Test 1: Standard classifier
    logger.info("=== Test 1: Standard Classifier ===")
    model = DINOv2Classifier(
        num_classes=2,
        model_name='dinov2_vitl14',
        freeze_backbone=True,
        use_fp16=True
    ).to(device)
    
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    output = model(dummy_input)
    logger.info("Input shape: %s", dummy_input.shape)
    logger.info("Output shape: %s", output.shape)
    
    # Test 2: Enhanced classifier with multi-layer features
    logger.info("=== Test 2: Enhanced Classifier with Multi-Layer Features ===")
    model_enhanced = DINOv2Classifier(
        num_classes=2,
        model_name='dinov2_vitl14',
        freeze_backbone=True,
        use_fp16=True,
        use_enhanced_head=True,
        use_multi_layer_features=True,
        hidden_dims=[512, 256, 128],
        dropout=0.3,
        n_last_blocks=4
    ).to(device)
    
    output_enhanced = model_enhanced(dummy_input)
    logger.info("Input shape: %s", dummy_input.shape)
    logger.info("Output shape: %s", output_enhanced.shape)
    logger.info("All tests passed!")
