"""
Swin Transformer v2 Pretrained Encoder Module

Uses Microsoft's Swin Transformer (and Swin V2) from timm for feature extraction.
Swin uses a hierarchical architecture with shifted windows for local attention,
making it more memory-efficient than vanilla ViT and well-suited to spectrogram patches.

Supported models (plan: swin_base_patch4_window7_224):
- swin_tiny_patch4_window7_224: 768 dim
- swin_small_patch4_window7_224: 768 dim
- swin_base_patch4_window7_224: 1024 dim (recommended)
- swinv2_tiny_window16_256: 768 dim (Swin V2, 256 input)
- swinv2_small_window16_256: 768 dim
- swinv2_base_window16_256: 1024 dim (Swin V2 base)
- Use mixed precision (FP16) and batch size 16-24 on 8GB GPU
"""

import torch
import torch.nn as nn

try:
    import timm

    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class SwinEncoder(nn.Module):
    """
    Swin Transformer (v1/v2) pretrained encoder for feature extraction.

    Args:
        model_name: Timm model name (default: 'swin_base_patch4_window7_224')
        freeze_backbone: If True, freeze all backbone weights (recommended for finetuning)
        use_fp16: If True, use half precision for lower memory usage
        use_gradient_checkpointing: If True, enable gradient checkpointing when backbone is trained
    """

    FEATURE_DIMS = {
        # Swin v1 (patch4 window7 224)
        "swin_tiny_patch4_window7_224": 768,
        "swin_small_patch4_window7_224": 768,
        "swin_base_patch4_window7_224": 1024,
        # Swin V2 (window16 256 and variants)
        "swinv2_tiny_window16_256": 768,
        "swinv2_tiny_window12_192_22k": 768,
        "swinv2_small_window16_256": 768,
        "swinv2_small_window12_192_22k": 768,
        "swinv2_base_window16_256": 1024,
        "swinv2_base_window12_192_22k": 1024,
        "swinv2_base_window12to16_192to256_22k": 1024,
        "swinv2_large_window16_256": 1536,
        "swinv2_large_window12to16_192to256_22k": 1536,
    }

    def __init__(
        self,
        model_name: str = "swin_base_patch4_window7_224",
        freeze_backbone: bool = True,
        use_fp16: bool = True,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()

        if not TIMM_AVAILABLE:
            raise ImportError(
                "timm is required for Swin encoder. Install with: pip install timm"
            )

        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.use_fp16 = use_fp16
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.feature_dim = self.FEATURE_DIMS.get(model_name, 1024)

        # Load Swin from timm (num_classes=0 returns features only)
        print(f"Loading {model_name} from timm...")
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,  # Feature extractor only
            global_pool="avg",  # Global average pooling over spatial dims
        )
        print(f"Loaded {model_name} with feature dim {self.feature_dim}")

        if freeze_backbone:
            self._freeze_backbone()
            print("Backbone frozen - only classifier will be trained")
        elif use_gradient_checkpointing:
            if hasattr(self.backbone, "gradient_checkpointing_enable"):
                self.backbone.gradient_checkpointing_enable()
                print("Gradient checkpointing enabled for memory efficiency")

        if use_fp16:
            self.backbone = self.backbone.half()
            print("Using FP16 for reduced memory usage")

    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images.

        Args:
            x: Input images of shape (B, 3, H, W), normalized with ImageNet stats.
               For swin_base_patch4_window7_224 use H=W=224; for swinv2_base_window16_256 use 256.

        Returns:
            Features of shape (B, feature_dim)
        """
        if self.use_fp16:
            x = x.half()

        with torch.cuda.amp.autocast(enabled=self.use_fp16):
            if self.freeze_backbone:
                with torch.no_grad():
                    features = self.backbone(x)
            else:
                features = self.backbone(x)

        return features.float()

    def train(self, mode: bool = True):
        """Override train to keep backbone frozen if freeze_backbone is True."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self


class SwinClassifier(nn.Module):
    """
    Complete classification model using Swin encoder + classifier head.

    Args:
        num_classes: Number of output classes
        model_name: Swin model variant (e.g. swin_base_patch4_window7_224)
        freeze_backbone: Whether to freeze Swin weights
        use_fp16: Whether to use FP16 for encoder
        use_gradient_checkpointing: Whether to use gradient checkpointing (when not freezing)
        hidden_dim: Hidden dimension for classifier MLP (None for linear classifier)
        dropout: Dropout rate for classifier
    """

    def __init__(
        self,
        num_classes: int = 2,
        model_name: str = "swin_base_patch4_window7_224",
        freeze_backbone: bool = True,
        use_fp16: bool = True,
        use_gradient_checkpointing: bool = False,
        hidden_dim: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = SwinEncoder(
            model_name=model_name,
            freeze_backbone=freeze_backbone,
            use_fp16=use_fp16,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        feature_dim = self.encoder.feature_dim

        if hidden_dim is None:
            self.classifier = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

        print(f"Classifier head: {feature_dim} -> {num_classes}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input images (B, 3, H, W)

        Returns:
            Logits of shape (B, num_classes)
        """
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits

    @property
    def feature_dim(self):
        return self.encoder.feature_dim


def get_swin_transforms(image_size: int = 224):
    """
    Get the recommended transforms for Swin (ImageNet normalization).

    Use image_size=224 for swin_*_patch4_window7_224, or 256 for swinv2_*_window16_256.

    Args:
        image_size: Target image size (224 or 256 depending on model)

    Returns:
        Dictionary with 'train' and 'val' transforms
    """
    from torchvision import transforms

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return {
        "train": train_transform,
        "val": val_transform,
    }


if __name__ == "__main__":
    print("Testing Swin Encoder...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SwinClassifier(
        num_classes=2,
        model_name="swin_base_patch4_window7_224",
        freeze_backbone=True,
        use_fp16=True,
    ).to(device)

    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Test passed!")
