"""
BEiT v2 (Bidirectional Encoder from Image Transformers) Pretrained Encoder Module

Uses Microsoft's BEiT / BEiT v2 from timm for feature extraction.
BEiT was trained with masked image modeling; BEiT v2 uses vector-quantized visual tokenizers.
Strong performance on fine-grained classification. Use official Microsoft weights via timm.

Supported models (plan: beit_base_patch16_224):
- beit_base_patch16_224: BEiT v1 base, 768 dim, 224 input
- beitv2_base_patch16_224: BEiT v2 base, 768 dim (recommended)
- beit_large_patch16_224: BEiT v1 large, 1024 dim
- beitv2_large_patch16_224: BEiT v2 large, 1024 dim
- Use mixed precision (FP16) and batch size 12-16 on 8GB GPU
"""

import torch
import torch.nn as nn

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class BEiTEncoder(nn.Module):
    """
    BEiT / BEiT v2 pretrained encoder for feature extraction.

    Args:
        model_name: Timm model name (default: 'beitv2_base_patch16_224' for BEiT v2)
        freeze_backbone: If True, freeze all backbone weights (recommended for finetuning)
        use_fp16: If True, use half precision for lower memory usage
        use_gradient_checkpointing: If True, enable gradient checkpointing (saves memory when backbone is trained)
    """

    FEATURE_DIMS = {
        "beit_base_patch16_224": 768,
        "beit_base_patch16_384": 768,
        "beit_large_patch16_224": 1024,
        "beit_large_patch16_384": 1024,
        "beit_large_patch16_512": 1024,
        "beitv2_base_patch16_224": 768,
        "beitv2_large_patch16_224": 1024,
    }

    def __init__(
        self,
        model_name: str = "beitv2_base_patch16_224",
        freeze_backbone: bool = True,
        use_fp16: bool = True,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()

        if not TIMM_AVAILABLE:
            raise ImportError(
                "timm is required for BEiT encoder. Install with: pip install timm"
            )

        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.use_fp16 = use_fp16
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.feature_dim = self.FEATURE_DIMS.get(model_name, 768)

        # Load BEiT from timm (num_classes=0 returns features only, no classifier head)
        # BEiT uses global_pool 'avg' by default; use it for a single feature vector per image
        print(f"Loading {model_name} from timm...")
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,  # Feature extractor only
            global_pool="avg",  # Global average pooling over tokens
        )
        print(f"Loaded {model_name} with feature dim {self.feature_dim}")

        if freeze_backbone:
            self._freeze_backbone()
            print("Backbone frozen - only classifier will be trained")
        elif use_gradient_checkpointing:
            if hasattr(self.backbone, "set_grad_checkpointing"):
                self.backbone.set_grad_checkpointing(enable=True)
                print("Gradient checkpointing enabled for memory efficiency")
            elif hasattr(self.backbone, "gradient_checkpointing_enable"):
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
            x: Input images of shape (B, 3, H, W), normalized with ImageNet stats
               (BEiT v2) or (0.5, 0.5, 0.5) for BEiT v1. H and W should be 224 (or 384 for 384 variants).

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


class BEiTClassifier(nn.Module):
    """
    Complete classification model using BEiT encoder + classifier head.

    Args:
        num_classes: Number of output classes
        model_name: BEiT model variant (e.g. beitv2_base_patch16_224)
        freeze_backbone: Whether to freeze BEiT weights
        use_fp16: Whether to use FP16 for encoder
        use_gradient_checkpointing: Whether to use gradient checkpointing (when not freezing)
        hidden_dim: Hidden dimension for classifier MLP (None for linear classifier)
        dropout: Dropout rate for classifier
    """

    def __init__(
        self,
        num_classes: int = 2,
        model_name: str = "beitv2_base_patch16_224",
        freeze_backbone: bool = True,
        use_fp16: bool = True,
        use_gradient_checkpointing: bool = False,
        hidden_dim: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = BEiTEncoder(
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


def get_beit_transforms(image_size: int = 224, use_imagenet_norm: bool = True):
    """
    Get the recommended transforms for BEiT.

    BEiT v2 uses ImageNet normalization. BEiT v1 originally used mean=0.5, std=0.5;
    timm pretrained BEiT v2 configs use ImageNet stats.

    Args:
        image_size: Target image size (224 for base 224 models, 384 for 384 variants)
        use_imagenet_norm: If True, use ImageNet mean/std (for BEiT v2 and common finetuning)

    Returns:
        Dictionary with 'train' and 'val' transforms
    """
    from torchvision import transforms

    if use_imagenet_norm:
        MEAN = (0.485, 0.456, 0.406)
        STD = (0.229, 0.224, 0.225)
    else:
        MEAN = (0.5, 0.5, 0.5)
        STD = (0.5, 0.5, 0.5)

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    return {
        "train": train_transform,
        "val": val_transform,
    }


if __name__ == "__main__":
    print("Testing BEiT Encoder...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BEiTClassifier(
        num_classes=2,
        model_name="beitv2_base_patch16_224",
        freeze_backbone=True,
        use_fp16=True,
    ).to(device)

    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Test passed!")
