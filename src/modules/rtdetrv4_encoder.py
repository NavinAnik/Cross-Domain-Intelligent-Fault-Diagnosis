"""
RT-DETRv4 Encoder Module

Uses RT-DETR backbone/encoder for feature extraction, then adds a classifier head.
Supports RT-DETRv4-L and RT-DETRv4-L+distill via Hugging Face transformers.

Note: RT-DETRv4-L and RT-DETRv4-L+distill from RT-DETRs/RT-DETRv4 use HGNetV2-L backbone.
When available from that repo, prefer those checkpoints. This module uses the HF
RT-DETR/RT-DETRv2 models (PekingU/rtdetr_r101vd, etc.) as a compatible backbone
for classification until RT-DETRv4 is available in transformers.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

try:
    from transformers import RTDetrForObjectDetection, RTDetrImageProcessor, AutoImageProcessor
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

logger = logging.getLogger(__name__)

# Model IDs: L-sized variants for RT-DETRv4-L style
RTDETRV4_L_MODEL_IDS = {
    "rtdetrv4_l": "PekingU/rtdetr_r101vd",  # R101 ~54% AP, large
    "rtdetrv4_l_distill": "PekingU/rtdetr_r101vd",  # +distill uses different weights when available
}
RTDETR_HIDDEN_DIM = 256  # RT-DETR d_model


class RTDetrv4Encoder(nn.Module):
    """
    RT-DETR encoder for feature extraction (backbone + hybrid encoder).
    Pools decoder output over object queries for image-level features.
    """

    def __init__(
        self,
        model_id: str = "PekingU/rtdetr_r101vd",
        freeze_backbone: bool = True,
        use_fp16: bool = True,
    ):
        super().__init__()
        if not _HAS_TRANSFORMERS:
            raise ImportError("transformers required. Install: pip install transformers>=4.56.0")

        self.model_id = model_id
        self.freeze_backbone = freeze_backbone
        self.use_fp16 = use_fp16

        logger.info("Loading %s from Hugging Face (RT-DETR)...", model_id)
        # PekingU checkpoint uses RTDetrForObjectDetection format; use it and extract base model
        full_model = RTDetrForObjectDetection.from_pretrained(model_id)
        self.rtdetr = full_model.model  # Base RT-DETR encoder for feature extraction
        # d_model from config
        self.feature_dim = getattr(
            self.rtdetr.config, "d_model", RTDETR_HIDDEN_DIM
        )

        if freeze_backbone:
            for p in self.rtdetr.parameters():
                p.requires_grad = False
            self.rtdetr.eval()
            logger.info("Backbone frozen - only classifier will be trained")

        if use_fp16:
            self.rtdetr = self.rtdetr.half()
            logger.info("Using FP16 for reduced memory usage")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract pooled features from images. x: (B, 3, H, W)."""
        if self.use_fp16:
            x = x.half()
        with torch.amp.autocast("cuda", enabled=(self.use_fp16 and torch.cuda.is_available())):
            if self.freeze_backbone:
                with torch.no_grad():
                    out = self.rtdetr(pixel_values=x)
            else:
                out = self.rtdetr(pixel_values=x)
        # last_hidden_state: (B, num_queries, d_model)
        h = out.last_hidden_state.float()
        # Mean pool over object queries -> (B, d_model)
        features = h.mean(dim=1)
        return features


def get_rtdetrv4_transforms(image_size: int = 640, augment_strength: str = "weak"):
    """RT-DETR typically uses 640x640. For 224 we resize."""
    from torchvision import transforms
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    size = (image_size, image_size)
    if augment_strength == "strong":
        train_t = transforms.Compose([
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
        ])
    else:
        train_t = transforms.Compose([
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    val_t = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return {"train": train_t, "val": val_t}


class RTDetrv4Classifier(nn.Module):
    """
    RT-DETRv4-based classifier for binary/multi-class classification.
    """

    def __init__(
        self,
        num_classes: int = 2,
        model_id: str = "PekingU/rtdetr_r101vd",
        freeze_backbone: bool = True,
        use_fp16: bool = True,
        hidden_dim: Optional[int] = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = RTDetrv4Encoder(
            model_id=model_id,
            freeze_backbone=freeze_backbone,
            use_fp16=use_fp16,
        )
        fd = self.encoder.feature_dim
        if hidden_dim is None:
            self.classifier = nn.Sequential(
                nn.LayerNorm(fd),
                nn.Dropout(dropout),
                nn.Linear(fd, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(fd),
                nn.Linear(fd, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        logger.info("RT-DETRv4 Classifier: %s -> %s", fd, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.classifier(features)
