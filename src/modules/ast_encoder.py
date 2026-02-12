"""
Audio Spectrogram Transformer (AST) Pretrained Encoder Module

Uses MIT/ast-finetuned-audioset-10-10-0.4593 from HuggingFace for audio (spectrogram)
feature extraction. AST applies a Vision Transformer to mel spectrograms and achieves
strong results on AudioSet (0.485 mAP). Ideal for anomaly detection on 128-dim mel
spectrograms (e.g. MIMII).

- Feature dim: 768
- Input: (batch, time, num_mel_bins) e.g. (B, 128, 128)
- Input normalization: mean=0, std=0.5 (critical for fine-tuning; configurable)
- Patch size: 16×16 with overlap (frequency_stride=10, time_stride=10 in config)
- Window: 25ms Hamming, 10ms hop (handled by feature extractor if using raw audio)
"""

import torch
import torch.nn as nn

try:
    from transformers import ASTModel, ASTConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# Default normalization for fine-tuning (plan: mean=0, std=0.5)
# HuggingFace pretrained uses mean=-4.2677393, std=4.5689974 (AudioSet)
AST_INPUT_MEAN = 0.0
AST_INPUT_STD = 0.5


class ASTEncoder(nn.Module):
    """
    Audio Spectrogram Transformer encoder for feature extraction.

    Args:
        model_name: HuggingFace model id (default: 'MIT/ast-finetuned-audioset-10-10-0.4593')
        freeze_backbone: If True, freeze all backbone weights (recommended for finetuning)
        use_fp16: If True, use half precision for lower memory usage
        input_mean: Normalization mean for input spectrograms (plan: 0.0 for fine-tuning)
        input_std: Normalization std for input spectrograms (plan: 0.5 for fine-tuning)
    """

    FEATURE_DIM = 768

    def __init__(
        self,
        model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        freeze_backbone: bool = True,
        use_fp16: bool = True,
        input_mean: float = AST_INPUT_MEAN,
        input_std: float = AST_INPUT_STD,
    ):
        super().__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required for AST encoder. "
                "Install with: pip install transformers"
            )

        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.use_fp16 = use_fp16
        self.input_mean = input_mean
        self.input_std = input_std
        self.feature_dim = self.FEATURE_DIM

        # Load AST base model (no classification head; we add our own)
        print(f"Loading {model_name} from HuggingFace...")
        self.backbone = ASTModel.from_pretrained(model_name)
        print(f"Loaded AST with feature dim {self.feature_dim}")

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

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert batch to (B, max_length, num_mel_bins) and normalize.
        Accepts (B, 1, T, F) from MIMII dataset or (B, T, F).
        """
        # Squeeze channel dim if present (e.g. from data_loader (B, 1, 128, 128))
        if x.dim() == 4:
            x = x.squeeze(1)  # (B, T, F)
        # Normalize: mean=0, std=0.5 (critical for fine-tuning)
        x = (x - self.input_mean) / (self.input_std + 1e-8)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from mel spectrograms.

        Args:
            x: Input spectrograms of shape (B, 1, T, F) or (B, T, F),
               e.g. (B, 128, 128) for 128 time frames and 128 mel bins.
               Values should be log-mel (e.g. from librosa.power_to_db).

        Returns:
            Features of shape (B, feature_dim) - pooled [CLS] token
        """
        x = self._prepare_input(x)
        if self.use_fp16:
            x = x.half()

        with torch.cuda.amp.autocast(enabled=self.use_fp16):
            if self.freeze_backbone:
                with torch.no_grad():
                    out = self.backbone(input_values=x)
            else:
                out = self.backbone(input_values=x)

        # pooler_output is (B, hidden_size) - classification token after projection
        features = out.pooler_output if out.pooler_output is not None else out.last_hidden_state[:, 0]
        return features.float()

    def train(self, mode: bool = True):
        """Override train to keep backbone frozen if freeze_backbone is True."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self


class ASTClassifier(nn.Module):
    """
    Complete classification model using AST encoder + classifier head.

    Args:
        num_classes: Number of output classes
        model_name: HuggingFace AST model id
        freeze_backbone: Whether to freeze AST weights
        use_fp16: Whether to use FP16 for encoder
        input_mean: Spectrogram normalization mean
        input_std: Spectrogram normalization std
        hidden_dim: Hidden dimension for classifier MLP (None for linear classifier)
        dropout: Dropout rate for classifier
    """

    def __init__(
        self,
        num_classes: int = 2,
        model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        freeze_backbone: bool = True,
        use_fp16: bool = True,
        input_mean: float = AST_INPUT_MEAN,
        input_std: float = AST_INPUT_STD,
        hidden_dim: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = ASTEncoder(
            model_name=model_name,
            freeze_backbone=freeze_backbone,
            use_fp16=use_fp16,
            input_mean=input_mean,
            input_std=input_std,
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
            x: Input spectrograms (B, 1, T, F) or (B, T, F), e.g. (B, 128, 128)

        Returns:
            Logits of shape (B, num_classes)
        """
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits

    @property
    def feature_dim(self):
        return self.encoder.feature_dim


def get_ast_normalization():
    """
    Get recommended normalization for AST (mean=0, std=0.5 for fine-tuning).

    Returns:
        dict with 'mean' and 'std' for normalizing log-mel spectrograms.
    """
    return {"mean": AST_INPUT_MEAN, "std": AST_INPUT_STD}


if __name__ == "__main__":
    print("Testing AST Encoder...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ASTClassifier(
        num_classes=2,
        model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
        freeze_backbone=True,
        use_fp16=True,
        input_mean=0.0,
        input_std=0.5,
    ).to(device)

    # (B, 1, time, mel) as from MIMII data_loader
    dummy_input = torch.randn(2, 1, 128, 128).to(device)
    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Test passed!")
