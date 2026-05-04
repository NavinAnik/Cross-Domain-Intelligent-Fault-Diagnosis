"""
Multimodal fusion strategies for audio and vision.

- EarlyFusion: concatenate features then classify
- LateFusion: weighted average of per-modality logits
- CrossAttentionFusion: bidirectional cross-attention between modalities
- ConFuFusion: contrastive fusion (pairwise alignment + fusion classifier)
- ProMoBalFusion: prototype-guided modality balancing with sample-adaptive weights
- HTSFFusion: Hierarchical Temporal-Spectral Fusion for audio-visual anomaly detection
"""

from .early_fusion import EarlyFusion
from .late_fusion import LateFusion
from .cross_attention_fusion import CrossAttentionFusion
from .confu_fusion import ConFuFusion
from .promobal_fusion import ProMoBalFusion
from .htsf_fusion import HTSFFusion, HTSFWithUncertainty

__all__ = [
    "EarlyFusion",
    "LateFusion",
    "CrossAttentionFusion",
    "ConFuFusion",
    "ProMoBalFusion",
    "HTSFFusion",
    "HTSFWithUncertainty",
]
