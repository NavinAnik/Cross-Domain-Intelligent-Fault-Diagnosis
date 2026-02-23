"""
EVA-02 Encoder Module

EVA-02: strong visual representation model (90% ImageNet fine-tune with 304M params).
Available via timm. Good baseline for image classification.
"""

from typing import Optional
import torch
import torch.nn as nn

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

EVA02_FEATURE_DIMS = {
    "timm/eva02_base_patch14_448": 768,
    "timm/eva02_large_patch14_224": 1024,
    "timm/eva02_large_patch14_448": 1024,
    "eva02_base_patch14_448": 768,
    "eva02_large_patch14_224": 1024,
    "eva02_large_patch14_448": 1024,
}


class EVA02Encoder(nn.Module):
    def __init__(
        self,
        model_id: str = "timm/eva02_large_patch14_224",
        freeze_backbone: bool = True,
        use_fp16: bool = True,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required. Install: pip install timm>=1.0.20")

        self.model_id = model_id
        self.freeze_backbone = freeze_backbone
        self.use_fp16 = use_fp16

        print(f"Loading {model_id} from timm...")
        self.backbone = timm.create_model(
            model_id,
            pretrained=True,
            num_classes=0,
            global_pool="token",
        )
        self.feature_dim = EVA02_FEATURE_DIMS.get(model_id, getattr(self.backbone, "num_features", 1024))
        print(f"Loaded {model_id} with feature dim {self.feature_dim}")

        if freeze_backbone:
            self._freeze_backbone()
            print("Backbone frozen - only classifier will be trained")
        elif use_gradient_checkpointing and hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()

        if use_fp16:
            self.backbone = self.backbone.half()
            print("Using FP16 for reduced memory usage")

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
                    features = self.backbone(x)
            else:
                features = self.backbone(x)
        return features.float()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self


def get_eva02_transforms(image_size: int = 224, augment_strength: str = "weak"):
    """EVA-02 uses ImageNet normalization."""
    from torchvision import transforms
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    if augment_strength == "strong":
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
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


class EVA02Classifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        model_id: str = "timm/eva02_large_patch14_224",
        freeze_backbone: bool = True,
        use_fp16: bool = True,
        use_gradient_checkpointing: bool = False,
        hidden_dim: Optional[int] = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = EVA02Encoder(
            model_id=model_id,
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
        print(f"EVA02 Classifier: {feature_dim} -> {num_classes}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.classifier(features)

    @property
    def feature_dim(self):
        return self.encoder.feature_dim


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EVA02Classifier(
        num_classes=2,
        model_id="timm/eva02_base_patch14_448",
        freeze_backbone=True,
    ).to(device)
    x = torch.randn(2, 3, 224, 224).to(device)
    out = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}")
