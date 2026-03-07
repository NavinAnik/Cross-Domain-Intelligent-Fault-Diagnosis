"""
MetaCLIP 2 Encoder Module

Meta's MetaCLIP 2 (Aug 2025) - multilingual vision-language model that surpasses
SigLIP-2. Use the vision encoder only for image classification.

Model: facebook/metaclip-2-worldwide-huge-quickgelu (ViT-H, 2B params)
"""

from typing import Optional
import torch
import torch.nn as nn

try:
    from transformers import AutoModel
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class MetaCLIP2Encoder(nn.Module):
    """
    MetaCLIP 2 vision encoder for feature extraction.
    Extracts image features from the vision tower (CLIP-style).
    """

    def __init__(
        self,
        model_id: str = "facebook/metaclip-2-worldwide-huge-quickgelu",
        freeze_backbone: bool = True,
        use_fp16: bool = True,
    ):
        super().__init__()
        if not HF_AVAILABLE:
            raise ImportError("transformers>=4.56.0 required for MetaCLIP 2")

        self.model_id = model_id
        self.freeze_backbone = freeze_backbone
        self.use_fp16 = use_fp16

        print(f"Loading {model_id} from Hugging Face...")
        dtype = torch.bfloat16 if use_fp16 else torch.float32
        self.backbone = AutoModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            attn_implementation="sdpa" if hasattr(torch.nn.functional, "scaled_dot_product_attention") else "eager",
        )
        # get_image_features returns pooler_output after visual_projection (hidden_size -> projection_dim)
        # Use projection_dim, not vision hidden_size, since we use the projected features
        self.feature_dim = getattr(
            self.backbone.config, "projection_dim",
            getattr(self.backbone.config.vision_config, "projection_dim", 1024)
            if hasattr(self.backbone.config, "vision_config") else 1024,
        )
        print(f"Loaded {model_id} with feature dim {self.feature_dim}")

        if freeze_backbone:
            self._freeze_backbone()
            print("Backbone frozen - only classifier will be trained")

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MetaCLIP 2 expects pixel_values; get_image_features returns BaseModelOutputWithPooling
        if hasattr(self.backbone, "get_image_features"):
            with torch.amp.autocast("cuda", enabled=(self.use_fp16 and torch.cuda.is_available())):
                if self.freeze_backbone:
                    with torch.no_grad():
                        out = self.backbone.get_image_features(pixel_values=x)
                else:
                    out = self.backbone.get_image_features(pixel_values=x)
            features = out.pooler_output
        else:
            outputs = self.backbone(pixel_values=x)
            features = outputs.get("pooler_output") or outputs.get("last_hidden_state", outputs)[:, 0, :]
        return features.float()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self


def get_metaclip2_transforms(image_size: int = 224, augment_strength: str = "weak"):
    """MetaCLIP 2 uses standard CLIP/ImageNet-like normalization."""
    from torchvision import transforms
    # MetaCLIP typically uses ImageNet stats (same as CLIP for compatibility)
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


class MetaCLIP2Classifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        model_id: str = "facebook/metaclip-2-worldwide-huge-quickgelu",
        freeze_backbone: bool = True,
        use_fp16: bool = True,
        hidden_dim: Optional[int] = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = MetaCLIP2Encoder(
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
        print(f"MetaCLIP2 Classifier: {feature_dim} -> {num_classes}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.classifier(features)

    @property
    def feature_dim(self):
        return self.encoder.feature_dim


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MetaCLIP2Classifier(
        num_classes=2,
        freeze_backbone=True,
    ).to(device)
    x = torch.randn(2, 3, 224, 224).to(device)
    out = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}")
