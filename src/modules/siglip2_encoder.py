"""
SigLIP 2 Encoder Module

Google's SigLIP 2 vision-language encoders (Feb 2025) with improved semantic
understanding, localization, and dense features. Outperforms original SigLIP.

Uses Hugging Face Transformers (timm does not include SigLIP 2 model definitions).

Supported models (Hugging Face):
- google/siglip2-base-patch16-224: ViT-Base, 224 res
- google/siglip2-base-patch16-256: ViT-Base, 256 res
- google/siglip2-large-patch16-256: ViT-Large, 256 res
- google/siglip2-so400m-patch14-384: ViT-SO400M
"""

from typing import Optional
import torch
import torch.nn as nn

try:
    from transformers import AutoModel
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

SIGLIP2_FEATURE_DIMS = {
    "google/siglip2-base-patch16-224": 768,
    "google/siglip2-base-patch16-256": 768,
    "google/siglip2-large-patch16-256": 1024,
    "google/siglip2-so400m-patch14-384": 1152,
    "google/siglip2-base-patch16-naflex": 768,
}


class SigLIP2Encoder(nn.Module):
    """
    SigLIP 2 pretrained encoder for feature extraction via Hugging Face Transformers.
    """

    def __init__(
        self,
        model_id: str = "google/siglip2-base-patch16-224",
        freeze_backbone: bool = True,
        use_fp16: bool = True,
    ):
        super().__init__()
        if not HF_AVAILABLE:
            raise ImportError("transformers>=4.56.0 required for SigLIP 2. Install: pip install transformers")

        self.model_id = model_id
        self.freeze_backbone = freeze_backbone
        self.use_fp16 = use_fp16

        print(f"Loading {model_id} from Hugging Face...")
        dtype = torch.float16 if use_fp16 else torch.float32
        self.backbone = AutoModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )
        self.feature_dim = SIGLIP2_FEATURE_DIMS.get(
            model_id,
            getattr(
                getattr(self.backbone.config, "vision_config", self.backbone.config),
                "hidden_size",
                768,
            ),
        )
        if hasattr(self.backbone.config, "vision_config"):
            vc = self.backbone.config.vision_config
            self.feature_dim = getattr(vc, "hidden_size", self.feature_dim)
        elif hasattr(self.backbone.config, "projection_dim"):
            self.feature_dim = self.backbone.config.projection_dim
        print(f"Loaded {model_id} with feature dim {self.feature_dim}")

        if freeze_backbone:
            self._freeze_backbone()
            print("Backbone frozen - only classifier will be trained")

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fp16:
            x = x.half()
        with torch.amp.autocast("cuda", enabled=(self.use_fp16 and torch.cuda.is_available())):
            if self.freeze_backbone:
                with torch.no_grad():
                    features = self.backbone.get_image_features(pixel_values=x)
            else:
                features = self.backbone.get_image_features(pixel_values=x)
        return features.float()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self


def get_siglip2_transforms(image_size: int = 224, augment_strength: str = "weak"):
    """SigLIP 2 uses mean=0.5, std=0.5 (not ImageNet). Default 224 for base model."""
    from torchvision import transforms
    # SigLIP 2 normalization (from Siglip2ImageProcessor defaults)
    SIGLIP2_MEAN = (0.5, 0.5, 0.5)
    SIGLIP2_STD = (0.5, 0.5, 0.5)

    if augment_strength == "strong":
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=SIGLIP2_MEAN, std=SIGLIP2_STD),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=SIGLIP2_MEAN, std=SIGLIP2_STD),
        ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=SIGLIP2_MEAN, std=SIGLIP2_STD),
    ])
    return {"train": train_transform, "val": val_transform}


class SigLIP2Classifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        model_id: str = "google/siglip2-base-patch16-224",
        freeze_backbone: bool = True,
        use_fp16: bool = True,
        hidden_dim: Optional[int] = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = SigLIP2Encoder(
            model_id=model_id,
            freeze_backbone=freeze_backbone,
            use_fp16=use_fp16,
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
        print(f"SigLIP2 Classifier: {feature_dim} -> {num_classes}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.classifier(features)

    @property
    def feature_dim(self):
        return self.encoder.feature_dim


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SigLIP2Classifier(
        num_classes=2,
        model_id="google/siglip2-base-patch16-224",
        freeze_backbone=True,
    ).to(device)
    x = torch.randn(2, 3, 224, 224).to(device)
    out = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}")
