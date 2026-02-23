"""
CLIP (Contrastive Language-Image Pre-training) Encoder Module

Uses OpenAI's CLIP ViT-B/16 for multimodal alignment and feature extraction.
CLIP was pretrained on 400M image-text pairs, enabling zero-shot classification
and alignment of visual inputs (e.g. spectrograms) with textual descriptions.

ViT-B/16 variant:
- Feature dim: 512
- Input: 224x224 RGB images
- Use for zero-shot classification and multimodal alignment
- Can align audio spectrograms with textual descriptions

Install: pip install git+https://github.com/openai/CLIP.git
Dependencies: ftfy, regex
"""

import torch
import torch.nn as nn
from typing import List, Optional

# Optional: try HuggingFace CLIP as fallback if openai/CLIP not installed
try:
    import clip as openai_clip
    OPENAI_CLIP_AVAILABLE = True
except ImportError:
    OPENAI_CLIP_AVAILABLE = False

try:
    from transformers import CLIPProcessor, CLIPModel
    HF_CLIP_AVAILABLE = True
except ImportError:
    HF_CLIP_AVAILABLE = False


# CLIP ViT-B/16 normalization (used by OpenAI CLIP)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Supported model names and their feature dimensions
CLIP_FEATURE_DIMS = {
    "ViT-B/32": 512,
    "ViT-B/16": 512,
    "ViT-L/14": 768,
    "ViT-L/14@336px": 768,
}


class CLIPEncoder(nn.Module):
    """
    CLIP pretrained image encoder for feature extraction and multimodal alignment.

    Uses ViT-B/16 by default (512-dim embeddings). Supports frozen backbone and FP16.
    Optionally use encode_text() for zero-shot classification or multimodal alignment.

    Args:
        model_name: One of 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'
        freeze_backbone: If True, freeze all backbone weights (recommended for finetuning)
        use_fp16: If True, use half precision for lower memory usage
        use_hf: If True, use HuggingFace transformers CLIP; else use openai/CLIP
    """

    def __init__(
        self,
        model_name: str = "ViT-B/16",
        freeze_backbone: bool = True,
        use_fp16: bool = True,
        use_hf: bool = False,
    ):
        super().__init__()

        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.use_fp16 = use_fp16
        self.use_hf = use_hf
        self.feature_dim = CLIP_FEATURE_DIMS.get(model_name, 512)

        if use_hf and HF_CLIP_AVAILABLE:
            self._init_hf(model_name)
        elif OPENAI_CLIP_AVAILABLE:
            self._init_openai(model_name)
        else:
            raise ImportError(
                "CLIP not found. Install with: pip install git+https://github.com/openai/CLIP.git\n"
                "Or: pip install transformers (for HuggingFace CLIP)"
            )

        if freeze_backbone:
            self._freeze_backbone()
            print("CLIP backbone frozen - only classifier will be trained")

        if use_fp16 and not use_hf:
            self.backbone = self.backbone.half()
            print("Using FP16 for reduced memory usage")

    def _init_openai(self, model_name: str):
        """Load CLIP from OpenAI official repo (torch hub style via clip.load)."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.backbone, self._preprocess = openai_clip.load(model_name, device=device)
        # We only need the visual encoder for image features
        self.backbone = self.backbone.visual
        self._preprocess_fn = None  # Use get_clip_transforms() for dataset

    def _init_hf(self, model_name: str):
        """Load CLIP from HuggingFace transformers; full model for get_image_features (512-dim)."""
        hf_name = {
            "ViT-B/32": "openai/clip-vit-base-patch32",
            "ViT-B/16": "openai/clip-vit-base-patch16",
            "ViT-L/14": "openai/clip-vit-large-patch14",
            "ViT-L/14@336px": "openai/clip-vit-large-patch14-336",
        }.get(model_name, "openai/clip-vit-base-patch16")
        print(f"Loading {hf_name} from HuggingFace...")
        self.backbone = CLIPModel.from_pretrained(hf_name)
        self.processor = CLIPProcessor.from_pretrained(hf_name)
        # Keep full model: get_image_features returns projected 512-dim (ViT-B/16) embeddings
        self.feature_dim = self.backbone.config.projection_dim  # 512 for ViT-B/16
        self._preprocess_fn = None
        self._preprocess = None

    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract image features.

        Args:
            x: Input images (B, 3, H, W), normalized with CLIP stats (see get_clip_transforms).

        Returns:
            Features of shape (B, feature_dim).
        """
        if self.use_fp16 and not self.use_hf:
            x = x.half()

        with torch.cuda.amp.autocast(enabled=self.use_fp16):
            if self.freeze_backbone:
                with torch.no_grad():
                    features = self._encode_images(x)
            else:
                features = self._encode_images(x)

        return features.float()

    def _encode_images(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images through the visual backbone."""
        if self.use_hf:
            # Full CLIPModel: get_image_features returns (B, projection_dim) e.g. 512
            return self.backbone.get_image_features(pixel_values=x)
        else:
            return self.backbone(x)

    def encode_text(self, texts: List[str], device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Encode text for zero-shot classification or multimodal alignment.
        Only available when using OpenAI CLIP (not HuggingFace vision-only path).

        Args:
            texts: List of strings to encode.
            device: Device for tokenization (default: same as first backbone parameter).

        Returns:
            Text features (num_texts, feature_dim), L2-normalized if using OpenAI CLIP.
        """
        if self.use_hf:
            raise NotImplementedError(
                "encode_text is implemented only when using OpenAI CLIP (use_hf=False)."
            )
        if not OPENAI_CLIP_AVAILABLE:
            raise ImportError("OpenAI CLIP is required for encode_text. Install: pip install git+https://github.com/openai/CLIP.git")
        # Reload full model for text encoding (we only keep .visual in backbone)
        full_model, _ = openai_clip.load(self.model_name, device=device or next(self.parameters()).device)
        tokens = openai_clip.tokenize(texts, truncate=True).to(full_model.device)
        with torch.no_grad():
            text_features = full_model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.float()

    def train(self, mode: bool = True):
        """Override train to keep backbone frozen if freeze_backbone is True."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self


class CLIPClassifier(nn.Module):
    """
    Classification model using CLIP image encoder + classifier head.

    Args:
        num_classes: Number of output classes.
        model_name: CLIP model variant (default ViT-B/16).
        freeze_backbone: Whether to freeze CLIP weights.
        use_fp16: Whether to use FP16 for encoder.
        use_hf: Use HuggingFace CLIP if True, else OpenAI CLIP.
        hidden_dim: Hidden dimension for classifier MLP (None for linear).
        dropout: Dropout rate for classifier.
    """

    def __init__(
        self,
        num_classes: int = 2,
        model_name: str = "ViT-B/16",
        freeze_backbone: bool = True,
        use_fp16: bool = True,
        use_hf: bool = False,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = CLIPEncoder(
            model_name=model_name,
            freeze_backbone=freeze_backbone,
            use_fp16=use_fp16,
            use_hf=use_hf,
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

        print(f"CLIP classifier head: {feature_dim} -> {num_classes}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input images (B, 3, H, W).

        Returns:
            Logits (B, num_classes).
        """
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits

    @property
    def feature_dim(self) -> int:
        return self.encoder.feature_dim


def get_clip_transforms(image_size: int = 224):
    """
    Recommended transforms for CLIP (ViT-B/16).
    Uses CLIP's normalization (different from ImageNet).

    Args:
        image_size: Target size (224 for ViT-B/16).

    Returns:
        Dict with 'train' and 'val' transforms.
    """
    from torchvision import transforms

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])

    return {
        "train": train_transform,
        "val": val_transform,
    }


def get_openai_clip_preprocess():
    """
    Return the official OpenAI CLIP preprocess (for single-image inference).
    Use get_clip_transforms() for dataset loaders.
    """
    if not OPENAI_CLIP_AVAILABLE:
        return None
    _, preprocess = openai_clip.load("ViT-B/16", device="cpu")
    return preprocess


if __name__ == "__main__":
    print("Testing CLIP Encoder...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prefer OpenAI CLIP if available (ViT-B/16, 512 dim)
    use_hf = not OPENAI_CLIP_AVAILABLE
    if OPENAI_CLIP_AVAILABLE:
        print("Using OpenAI CLIP (ViT-B/16)")
    else:
        print("Using HuggingFace CLIP")

    model = CLIPClassifier(
        num_classes=2,
        model_name="ViT-B/16",
        freeze_backbone=True,
        use_fp16=True,
        use_hf=use_hf,
    ).to(device)

    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Feature dim: {model.feature_dim}")
    print("Test passed!")
