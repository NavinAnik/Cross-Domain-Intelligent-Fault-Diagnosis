"""
DINOv3 Pretrained Encoder Module

Uses Meta AI's DINOv3 pretrained Vision Transformer for feature extraction.
DINOv3 (Aug 2025) scales self-supervised learning with models up to 7B parameters,
outperforming DINOv2 and weakly-supervised models across vision tasks.

Supported models (Hugging Face):
- facebook/dinov3-vits16-pretrain-lvd1689m: ViT-Small (384 dim)
- facebook/dinov3-vits16plus-pretrain-lvd1689m: ViT-S+ (384 dim)
- facebook/dinov3-vitb16-pretrain-lvd1689m: ViT-Base (768 dim)
- facebook/dinov3-vitl16-pretrain-lvd1689m: ViT-Large (1024 dim) - recommended
- facebook/dinov3-vith16plus-pretrain-lvd1689m: ViT-H+ (1280 dim)
- facebook/dinov3-vit7b16-pretrain-lvd1689m: ViT-7B (4096 dim)
- facebook/dinov3-convnext-tiny-pretrain-lvd1689m: ConvNeXt-Tiny
- facebook/dinov3-convnext-base-pretrain-lvd1689m: ConvNeXt-Base
"""

from typing import List, Optional
import torch
import torch.nn as nn


# Known feature dims for common DINOv3 models (others inferred from config)
DINOV3_FEATURE_DIMS = {
    "facebook/dinov3-vits16-pretrain-lvd1689m": 384,
    "facebook/dinov3-vits16plus-pretrain-lvd1689m": 384,
    "facebook/dinov3-vitb16-pretrain-lvd1689m": 768,
    "facebook/dinov3-vitl16-pretrain-lvd1689m": 1024,
    "facebook/dinov3-vith16plus-pretrain-lvd1689m": 1280,
    "facebook/dinov3-vit7b16-pretrain-lvd1689m": 4096,
    "facebook/dinov3-convnext-tiny-pretrain-lvd1689m": 768,
    "facebook/dinov3-convnext-base-pretrain-lvd1689m": 1024,
    "facebook/dinov3-convnext-small-pretrain-lvd1689m": 768,
    "facebook/dinov3-convnext-large-pretrain-lvd1689m": 1536,
}


class DINOv3Encoder(nn.Module):
    """
    DINOv3 pretrained encoder for feature extraction via Hugging Face Transformers.

    Args:
        model_id: Hugging Face model ID (e.g. facebook/dinov3-vitl16-pretrain-lvd1689m)
        freeze_backbone: If True, freeze all backbone weights (recommended for finetuning)
        use_fp16: If True, use half precision for lower memory usage
    """

    def __init__(
        self,
        model_id: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        freeze_backbone: bool = True,
        use_fp16: bool = True,
    ):
        super().__init__()
        from transformers import AutoModel

        self.model_id = model_id
        self.freeze_backbone = freeze_backbone
        self.use_fp16 = use_fp16

        print(f"Loading {model_id} from Hugging Face...")
        self.backbone = AutoModel.from_pretrained(model_id)
        self.feature_dim = DINOV3_FEATURE_DIMS.get(
            model_id, getattr(self.backbone.config, "hidden_size", 1024)
        )
        if model_id not in DINOV3_FEATURE_DIMS:
            # Infer from config
            if hasattr(self.backbone.config, "hidden_size"):
                self.feature_dim = self.backbone.config.hidden_size
            elif hasattr(self.backbone.config, "hidden_sizes"):
                self.feature_dim = self.backbone.config.hidden_sizes[-1]
        print(f"Loaded {model_id} with feature dim {self.feature_dim}")

        if freeze_backbone:
            self._freeze_backbone()
            print("Backbone frozen - only classifier will be trained")

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

        Returns:
            Features of shape (B, feature_dim) - pooler_output or CLS token
        """
        if self.use_fp16:
            x = x.half()

        with torch.amp.autocast("cuda", enabled=(self.use_fp16 and torch.cuda.is_available())):
            if self.freeze_backbone:
                with torch.no_grad():
                    outputs = self.backbone(pixel_values=x)
            else:
                outputs = self.backbone(pixel_values=x)

        # Prefer pooler_output; fallback to last_hidden_state[:, 0]
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state[:, 0, :]

        return features.float()


def get_dinov3_transforms(image_size: int = 224, augment_strength: str = "weak"):
    """
    Get the recommended transforms for DINOv3.

    Args:
        image_size: Target image size (224 default)
        augment_strength: 'weak' (default) or 'strong'

    Returns:
        Dictionary with 'train' and 'val' transforms
    """
    from torchvision import transforms

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    if augment_strength == "strong":
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
        ])
    else:
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

    return {"train": train_transform, "val": val_transform}


# Reuse EnhancedClassifierHead from dinov2 for consistency
def _get_enhanced_classifier():
    from modules.dinov2_encoder import EnhancedClassifierHead
    return EnhancedClassifierHead


class DINOv3Classifier(nn.Module):
    """
    Complete classification model using DINOv3 encoder + classifier head.

    Args:
        num_classes: Number of output classes
        model_id: DINOv3 Hugging Face model ID
        freeze_backbone: Whether to freeze DINOv3 weights
        use_fp16: Whether to use FP16 for encoder
        hidden_dim: Hidden dimension for simple MLP (None for linear)
        hidden_dims: List for enhanced classifier
        dropout: Dropout rate
        use_enhanced_head: Whether to use EnhancedClassifierHead
    """

    def __init__(
        self,
        num_classes: int = 2,
        model_id: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        freeze_backbone: bool = True,
        use_fp16: bool = True,
        hidden_dim: Optional[int] = None,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        use_enhanced_head: bool = False,
    ):
        super().__init__()
        self.encoder = DINOv3Encoder(
            model_id=model_id,
            freeze_backbone=freeze_backbone,
            use_fp16=use_fp16,
        )
        feature_dim = self.encoder.feature_dim

        EnhancedClassifierHead = _get_enhanced_classifier()
        if use_enhanced_head:
            if hidden_dims is None:
                hidden_dims = [512, 256, 128]
            self.classifier = EnhancedClassifierHead(
                feature_dim=feature_dim,
                num_classes=num_classes,
                hidden_dims=hidden_dims,
                dropout=dropout,
                use_attention_pooling=False,
            )
        elif hidden_dim is None:
            self.classifier = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, num_classes),
            )
            print(f"Linear Classifier: {feature_dim} -> {num_classes}")
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
            print(f"MLP Classifier: {feature_dim} -> {hidden_dim} -> {num_classes}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.classifier(features)

    @property
    def feature_dim(self):
        return self.encoder.feature_dim


if __name__ == "__main__":
    print("Testing DINOv3 Encoder...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DINOv3Classifier(
        num_classes=2,
        model_id="facebook/dinov3-vits16-pretrain-lvd1689m",
        freeze_backbone=True,
        use_fp16=True,
    ).to(device)
    x = torch.randn(2, 3, 224, 224).to(device)
    out = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}")
    print("All tests passed!")
