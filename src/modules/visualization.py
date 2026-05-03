"""
Visualization utilities for model analysis and publication-quality figures.

Includes:
- t-SNE and UMAP embedding visualization (normal vs abnormal clustering)
- Attention maps (which spectrogram/image regions the model focuses on)
- Grad-CAM for interpretability
- Radar plots for multi-metric model comparison
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional: UMAP (prefer over t-SNE for large N and structure preservation)
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay

__all__ = [
    "plot_tsne",
    "plot_umap",
    "plot_tsne_umap",
    "plot_attention_map",
    "compute_grad_cam_vit",
    "compute_grad_cam_cnn",
    "plot_grad_cam",
    "plot_grad_cam_cnn",
    "plot_radar_chart",
    "plot_radar_from_metric_dicts",
    "plot_cv_metric_boxplots",
    "plot_roc_comparison",
    "plot_pr_comparison",
    "plot_confusion_matrices_top_k",
    "plot_training_curves",
    "plot_ablation_bars",
    "plot_computational_efficiency",
    "plot_architecture_diagram",
    "UMAP_AVAILABLE",
]

# ---------------------------------------------------------------------------
# t-SNE / UMAP embedding visualization
# ---------------------------------------------------------------------------

def plot_tsne(
    embeddings: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor, List[int]],
    output_path: str,
    title: str = "t-SNE of Learned Embeddings",
    class_names: Optional[List[str]] = None,
    perplexity: float = 30.0,
    random_state: int = 42,
    figsize: Tuple[int, int] = (10, 8),
    alpha: float = 0.7,
) -> np.ndarray:
    """
    Compute t-SNE and plot 2D embedding space colored by class (e.g. normal vs abnormal).

    Args:
        embeddings: (N, D) feature vectors.
        labels: (N,) integer class labels.
        output_path: Path to save the figure (e.g. 'results/tsne.png').
        title: Plot title.
        class_names: Optional names for classes (e.g. ['Normal', 'Abnormal']).
        perplexity: t-SNE perplexity (5–50 typical).
        random_state: For reproducibility.
        figsize: Figure size.
        alpha: Point transparency.

    Returns:
        (N, 2) t-SNE coordinates.
    """
    X = np.asarray(embeddings, dtype=np.float64)
    y = np.asarray(labels).ravel()

    if X.shape[0] != y.shape[0]:
        raise ValueError("embeddings and labels must have same length")

    perplexity_actual = min(perplexity, max(5, X.shape[0] // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity_actual, random_state=random_state)
    coords = tsne.fit_transform(X)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    unique_labels = np.unique(y)
    if class_names is None:
        class_names = [str(int(u)) for u in unique_labels]
    if len(class_names) < len(unique_labels):
        class_names = class_names + [str(int(u)) for u in unique_labels[len(class_names):]]

    for i, label in enumerate(unique_labels):
        mask = y == label
        name = class_names[i] if i < len(class_names) else str(int(label))
        ax.scatter(coords[mask, 0], coords[mask, 1], label=name, alpha=alpha, s=20)

    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return coords


def plot_umap(
    embeddings: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor, List[int]],
    output_path: str,
    title: str = "UMAP of Learned Embeddings",
    class_names: Optional[List[str]] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    metric: str = "cosine",
    figsize: Tuple[int, int] = (10, 8),
    alpha: float = 0.7,
) -> np.ndarray:
    """
    Compute UMAP and plot 2D embedding space colored by class.

    Args:
        embeddings: (N, D) feature vectors.
        labels: (N,) integer class labels.
        output_path: Path to save the figure.
        title: Plot title.
        class_names: Optional names for classes.
        n_neighbors: UMAP n_neighbors (larger = more global structure).
        min_dist: UMAP min_dist (0–1).
        random_state: For reproducibility.
        metric: Distance metric ('cosine', 'euclidean', etc.).
        figsize: Figure size.
        alpha: Point transparency.

    Returns:
        (N, 2) UMAP coordinates.
    """
    if not UMAP_AVAILABLE:
        raise ImportError("umap-learn is required. Install with: pip install umap-learn")

    X = np.asarray(embeddings, dtype=np.float64)
    y = np.asarray(labels).ravel()

    if X.shape[0] != y.shape[0]:
        raise ValueError("embeddings and labels must have same length")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(n_neighbors, X.shape[0] - 1),
        min_dist=min_dist,
        random_state=random_state,
        metric=metric,
    )
    coords = reducer.fit_transform(X)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    unique_labels = np.unique(y)
    if class_names is None:
        class_names = [str(int(u)) for u in unique_labels]
    if len(class_names) < len(unique_labels):
        class_names = class_names + [str(int(u)) for u in unique_labels[len(class_names):]]

    for i, label in enumerate(unique_labels):
        mask = y == label
        name = class_names[i] if i < len(class_names) else str(int(label))
        ax.scatter(coords[mask, 0], coords[mask, 1], label=name, alpha=alpha, s=20)

    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return coords


def plot_tsne_umap(
    embeddings: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor, List[int]],
    output_dir: str = "results",
    title_prefix: str = "",
    class_names: Optional[List[str]] = None,
    tsne_perplexity: float = 30.0,
    umap_n_neighbors: int = 15,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Generate both t-SNE and UMAP plots and save to output_dir.

    Args:
        embeddings: (N, D) feature vectors.
        labels: (N,) integer class labels.
        output_dir: Directory for saved figures (e.g. 'results').
        title_prefix: Optional prefix for plot titles.
        class_names: Optional list of class names (e.g. ['Normal', 'Abnormal']).
        tsne_perplexity: t-SNE perplexity.
        umap_n_neighbors: UMAP n_neighbors.
        random_state: Random seed.

    Returns:
        Dict with keys 'tsne_coords' and 'umap_coords' (and 'tsne' / 'umap' paths in saved files).
    """
    os.makedirs(output_dir, exist_ok=True)
    X = np.asarray(embeddings)
    y = np.asarray(labels).ravel()

    out = {}
    tsne_path = os.path.join(output_dir, "tsne.png")
    out["tsne_coords"] = plot_tsne(
        X, y,
        output_path=tsne_path,
        title=f"{title_prefix} t-SNE of Learned Embeddings".strip() or "t-SNE of Learned Embeddings",
        class_names=class_names,
        perplexity=tsne_perplexity,
        random_state=random_state,
    )

    if UMAP_AVAILABLE:
        umap_path = os.path.join(output_dir, "umap.png")
        out["umap_coords"] = plot_umap(
            X, y,
            output_path=umap_path,
            title=f"{title_prefix} UMAP of Learned Embeddings".strip() or "UMAP of Learned Embeddings",
            class_names=class_names,
            n_neighbors=umap_n_neighbors,
            random_state=random_state,
        )
    else:
        out["umap_coords"] = None  # UMAP not available

    return out


# ---------------------------------------------------------------------------
# Attention maps (ViT / spectrogram regions)
# ---------------------------------------------------------------------------

def plot_attention_map(
    attention_weights: Union[np.ndarray, torch.Tensor],
    output_path: str,
    input_image: Optional[Union[np.ndarray, torch.Tensor]] = None,
    patch_size: Optional[Tuple[int, int]] = None,
    title: str = "Attention Map",
    cmap: str = "jet",
    overlay_alpha: float = 0.5,
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    """
    Plot attention as a 2D heatmap, optionally overlaid on the input image.

    For ViT, attention_weights are typically (num_patches,) or (H_patches, W_patches)
    (e.g. [CLS] token attention over patch tokens). If 1D, patch_size or image shape
    is used to reshape to 2D.

    Args:
        attention_weights: 1D (num_patches,) or 2D (H, W) attention map.
        output_path: Path to save the figure.
        input_image: Optional (C, H, W) or (H, W, C) to overlay heatmap on.
        patch_size: (ph, pw) to reshape 1D attention to 2D grid (e.g. (14, 14) for 224/16).
        title: Plot title.
        cmap: Matplotlib colormap name.
        overlay_alpha: Alpha for heatmap when overlaying on image (0–1).
        figsize: Figure size.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    attn = np.asarray(attention_weights).squeeze()
    if attn.ndim == 1:
        n = attn.size
        if patch_size is not None:
            ph, pw = patch_size
            if ph * pw != n:
                # Infer grid from n (e.g. 196 = 14*14)
                ph = int(np.sqrt(n))
                pw = n // ph
            attn = attn.reshape(ph, pw)
        else:
            # Assume square grid
            side = int(np.sqrt(n))
            if side * side != n:
                side = int(np.ceil(np.sqrt(n)))
                attn = np.pad(attn, (0, side * side - n), mode="constant", constant_values=0)
                attn = attn.reshape(side, side)
            else:
                attn = attn.reshape(side, side)

    # Normalize for display
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

    fig, ax = plt.subplots(figsize=figsize)
    if input_image is not None:
        img = np.asarray(input_image).squeeze()
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        if img.shape[:2] != attn.shape:
            try:
                from scipy.ndimage import zoom
                scale = (img.shape[0] / attn.shape[0], img.shape[1] / attn.shape[1])
                attn_upsampled = zoom(attn, scale, order=1)
            except ImportError:
                # Fallback: repeat to approximate size
                sy, sx = attn.shape
                ty, tx = img.shape[0], img.shape[1]
                repeat_y = max(1, (ty + sy - 1) // sy)
                repeat_x = max(1, (tx + sx - 1) // sx)
                attn_upsampled = np.repeat(np.repeat(attn, repeat_y, axis=0)[:ty], repeat_x, axis=1)[:, :tx]
        else:
            attn_upsampled = attn
        ax.imshow(img)
        im = ax.imshow(attn_upsampled, cmap=cmap, alpha=overlay_alpha)
    else:
        im = ax.imshow(attn, cmap=cmap)
    plt.colorbar(im, ax=ax, label="Attention")
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Grad-CAM for ViT / CNN
# ---------------------------------------------------------------------------

def _get_vit_patch_features_and_grad(
    model: nn.Module,
    x: torch.Tensor,
    target_class: Optional[int] = None,
    encoder_attr: str = "encoder",
    backbone_attr: str = "backbone",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get patch-level features and their gradients w.r.t. target class logit.
    Assumes model has model.encoder.backbone (e.g. timm ViT) and model.classifier.
    Backbone must produce patch tokens we can hook (e.g. timm ViT stores in .blocks).
    """
    encoder = getattr(model, encoder_attr, model)
    backbone = getattr(encoder, backbone_attr, encoder)
    features = []
    grad_out = []

    def hook_fn(module, grad_input, grad_output):
        grad_out.append(grad_output[0])

    def fwd_fn(module, input, output):
        features.append(output)

    # Try to attach to last block's output (timm ViT: backbone.blocks[-1])
    if hasattr(backbone, "blocks") and len(backbone.blocks) > 0:
        handle_fwd = backbone.blocks[-1].register_forward_hook(fwd_fn)
        handle_bwd = backbone.blocks[-1].register_full_backward_hook(hook_fn)
    elif hasattr(backbone, "forward_features"):
        # Single block or different structure: hook forward_features output
        handle_fwd = backbone.register_forward_hook(fwd_fn)
        handle_bwd = backbone.register_full_backward_hook(hook_fn)
    else:
        handle_fwd = encoder.register_forward_hook(fwd_fn)
        handle_bwd = encoder.register_full_backward_hook(hook_fn)

    model.eval()
    x = x.requires_grad_(True)
    logits = model(x)
    if target_class is None:
        target_class = logits.argmax(dim=1).item()
    score = logits[0, target_class]
    model.zero_grad()
    score.backward()

    handle_fwd.remove()
    handle_bwd.remove()

    feats = features[0]
    grads = grad_out[0]
    # feats/grads: (1, num_patches + 1, dim) or (1, num_patches, dim); drop CLS for spatial map
    if feats.dim() == 3:
        feats = feats[0]
        grads = grads[0]
    if feats.size(0) > 1 and feats.size(0) == grads.size(0):
        # First token is often CLS
        feats = feats[1:]
        grads = grads[1:]
    return feats, grads


def compute_grad_cam_vit(
    model: nn.Module,
    x: torch.Tensor,
    target_class: Optional[int] = None,
    encoder_attr: str = "encoder",
    backbone_attr: str = "backbone",
    patch_grid: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Compute Grad-CAM style map for a ViT-style model (patch tokens + gradients).

    Args:
        model: Classification model with encoder (+ backbone) and classifier.
        x: Single input (1, C, H, W).
        target_class: Class index to explain; if None, use predicted class.
        encoder_attr: Attribute name of encoder on model.
        backbone_attr: Attribute name of backbone on encoder.
        patch_grid: (H_patches, W_patches). If None, inferred from feature length.

    Returns:
        2D numpy array (H_patches, W_patches) heatmap.
    """
    if x.dim() == 3:
        x = x.unsqueeze(0)
    model.eval()
    feats, grads = _get_vit_patch_features_and_grad(
        model, x, target_class, encoder_attr, backbone_attr
    )
    # Grad-CAM: weight each channel by its mean gradient (over patches), then sum over channels
    weights = grads.mean(dim=0)
    cam = (feats * weights).sum(dim=-1)
    cam = F.relu(cam).detach().cpu().numpy()
    n = cam.size
    if patch_grid is not None:
        ph, pw = patch_grid
        if ph * pw != n:
            ph, pw = int(np.sqrt(n)), n // int(np.sqrt(n))
        cam = cam.reshape(ph, pw)
    else:
        side = int(np.sqrt(n))
        if side * side != n:
            side = int(np.ceil(np.sqrt(n)))
            cam = np.pad(cam, (0, side * side - n), mode="constant", constant_values=0)
            cam = cam.reshape(side, side)
        else:
            cam = cam.reshape(side, side)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.astype(np.float32)


def compute_grad_cam_cnn(
    model: nn.Module,
    x: torch.Tensor,
    target_class: Optional[int] = None,
    backbone_attr: str = "backbone",
) -> np.ndarray:
    """
    Grad-CAM from the last 4D feature map of a CNN backbone (e.g. timm ResNet/EfficientNet).

    Expects ``model.<backbone_attr>(x)`` to yield (B, C, H, W) during ``model(x)``.
    Works with CNNBaseline, CNNSpatialAttention, and CNNTransformerHybrid (CNN stem only).
    """
    if x.dim() == 3:
        x = x.unsqueeze(0)
    backbone = getattr(model, backbone_attr, None)
    if backbone is None:
        raise ValueError(f"Model has no '{backbone_attr}' for CNN Grad-CAM.")

    feats_holder: Dict[str, torch.Tensor] = {}

    def hook(_module, _inp, out):
        t = out[0] if isinstance(out, tuple) else out
        if isinstance(t, torch.Tensor) and t.dim() == 4:
            t.retain_grad()
            feats_holder["f"] = t

    handle = backbone.register_forward_hook(hook)
    model.eval()
    logits = model(x)
    handle.remove()

    f = feats_holder.get("f")
    if f is None:
        raise ValueError("Backbone did not produce a 4D feature map for Grad-CAM.")

    if target_class is None:
        target_class = int(logits.argmax(dim=1).item())

    model.zero_grad()
    logits[0, target_class].backward()
    grads = f.grad
    if grads is None:
        raise RuntimeError("No gradients on feature map; ensure model is differentiable.")

    weights = grads[0].mean(dim=(1, 2), keepdim=True)
    cam = (weights * f[0]).sum(dim=0)
    cam = F.relu(cam).detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.astype(np.float32)


def plot_grad_cam(
    model: nn.Module,
    x: torch.Tensor,
    output_path: str,
    target_class: Optional[int] = None,
    input_image: Optional[Union[np.ndarray, torch.Tensor]] = None,
    patch_grid: Optional[Tuple[int, int]] = None,
    title: str = "Grad-CAM",
    encoder_attr: str = "encoder",
    backbone_attr: str = "backbone",
    figsize: Tuple[int, int] = (8, 6),
    overlay_alpha: float = 0.5,
    cmap: str = "jet",
) -> np.ndarray:
    """
    Compute Grad-CAM for a single input and save figure (optionally overlaid on image).

    Args:
        model: Model with encoder (+ backbone) and classifier.
        x: Input tensor (1, C, H, W); used for forward/backward.
        output_path: Path to save the figure.
        target_class: Class to explain; None = predicted class.
        input_image: Optional image to overlay (e.g. denormalized x for display).
        patch_grid: (H_patches, W_patches) for ViT; inferred if None.
        title: Plot title.
        encoder_attr, backbone_attr: For ViT-style models.
        figsize, overlay_alpha, cmap: Plot options.

    Returns:
        2D Grad-CAM heatmap (numpy).
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cam = compute_grad_cam_vit(
        model, x, target_class=target_class,
        encoder_attr=encoder_attr, backbone_attr=backbone_attr,
        patch_grid=patch_grid,
    )
    img_for_overlay = input_image
    if img_for_overlay is None and x.dim() == 4:
        # Denormalize for display (ImageNet mean/std)
        img = x[0].detach().cpu().float()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
        img = np.clip(img.numpy(), 0, 1)
        img_for_overlay = np.transpose(img, (1, 2, 0))
    plot_attention_map(
        cam,
        output_path=output_path,
        input_image=img_for_overlay,
        patch_size=None,
        title=title,
        cmap=cmap,
        overlay_alpha=overlay_alpha,
        figsize=figsize,
    )
    return cam


def plot_grad_cam_cnn(
    model: nn.Module,
    x: torch.Tensor,
    output_path: str,
    target_class: Optional[int] = None,
    input_image: Optional[Union[np.ndarray, torch.Tensor]] = None,
    backbone_attr: str = "backbone",
    title: str = "Grad-CAM (CNN)",
    figsize: Tuple[int, int] = (8, 6),
    overlay_alpha: float = 0.5,
    cmap: str = "jet",
) -> np.ndarray:
    """Grad-CAM on CNN backbone feature maps; save overlaid figure like ``plot_grad_cam``."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cam = compute_grad_cam_cnn(
        model, x, target_class=target_class, backbone_attr=backbone_attr
    )
    img_for_overlay = input_image
    if img_for_overlay is None and x.dim() == 4:
        img = x[0].detach().cpu().float()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
        img = np.clip(img.numpy(), 0, 1)
        img_for_overlay = np.transpose(img, (1, 2, 0))
    plot_attention_map(
        cam,
        output_path=output_path,
        input_image=img_for_overlay,
        patch_size=None,
        title=title,
        cmap=cmap,
        overlay_alpha=overlay_alpha,
        figsize=figsize,
    )
    return cam


# ---------------------------------------------------------------------------
# Radar (spider) plots for multi-metric comparison
# ---------------------------------------------------------------------------

def plot_radar_chart(
    metrics_by_model: Dict[str, Dict[str, float]],
    output_path: str,
    metrics_order: Optional[List[str]] = None,
    title: str = "Model Comparison",
    figsize: Tuple[int, int] = (8, 8),
    normalize: bool = True,
    fill_alpha: float = 0.25,
) -> None:
    """
    Plot a radar (spider) chart comparing multiple models on several metrics.

    Args:
        metrics_by_model: e.g. {"DINOv2": {"accuracy": 0.9, "f1": 0.88, ...}, "MAE": {...}}.
        output_path: Path to save the figure.
        metrics_order: Order of axes; if None, use union of all keys (alphabetical).
        title: Plot title.
        figsize: Figure size.
        normalize: If True, scale each metric to [0, 1] using min/max across models.
        fill_alpha: Alpha for polygon fill (0–1).
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    all_metrics = set()
    for d in metrics_by_model.values():
        all_metrics.update(d.keys())
    if metrics_order is None:
        metrics_order = sorted(all_metrics)
    else:
        metrics_order = [m for m in metrics_order if m in all_metrics]
    if not metrics_order:
        raise ValueError("No metrics to plot")

    # Build matrix (models x metrics)
    values = {}
    for name, d in metrics_by_model.items():
        values[name] = [d.get(m, np.nan) for m in metrics_order]
    arr = np.array([values[name] for name in metrics_by_model])
    nan_mask = np.isnan(arr)
    if nan_mask.any():
        arr = np.nan_to_num(arr, nan=0.0)

    if normalize:
        min_vals = arr.min(axis=0)
        max_vals = arr.max(axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        arr = (arr - min_vals) / range_vals

    num_vars = len(metrics_order)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection="polar"))
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(metrics_by_model))))

    for i, (name, row) in enumerate(zip(metrics_by_model, arr)):
        vals = row.tolist()
        vals += vals[:1]
        color = colors[i % len(colors)]
        ax.plot(angles, vals, "o-", linewidth=2, label=name, color=color)
        ax.fill(angles, vals, alpha=fill_alpha, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_order, size=9)
    ax.set_ylim(0, 1.0 if normalize else None)
    ax.set_title(title, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_radar_from_metric_dicts(
    model_names: List[str],
    metric_dicts: List[Dict[str, float]],
    output_path: str,
    metrics_order: Optional[List[str]] = None,
    title: str = "Model Comparison (Cross-Validation Metrics)",
    **kwargs: Any,
) -> None:
    """
    Convenience: build metrics_by_model from a list of metric dicts (e.g. one per model).

    Args:
        model_names: List of model names.
        metric_dicts: List of dicts (e.g. from compute_classification_metrics), one per model.
        output_path: Path to save radar figure.
        metrics_order: Optional order of metrics (e.g. ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'auc_pr']).
        title: Plot title.
        **kwargs: Passed to plot_radar_chart.
    """
    if len(model_names) != len(metric_dicts):
        raise ValueError("model_names and metric_dicts must have same length")
    metrics_by_model = {name: d for name, d in zip(model_names, metric_dicts)}
    if metrics_order is None:
        metrics_order = ["accuracy", "precision", "recall", "f1", "mcc", "cohen_kappa", "auc_roc", "auc_pr"]
    plot_radar_chart(metrics_by_model, output_path, metrics_order=metrics_order, title=title, **kwargs)


def plot_cv_metric_boxplots(
    metrics_by_model: Dict[str, List[Dict[str, float]]],
    metric_names: Optional[List[str]] = None,
    output_path: str = "cv_metric_boxplots.png",
    figsize: Optional[Tuple[int, int]] = None,
    title: str = "Cross-Validation Metric Distributions",
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot box plots of metric distributions across CV folds for multiple models.

    Used for publication: box plots for metric distributions across folds (plan Section 5).

    Args:
        metrics_by_model: Dict mapping model name -> list of metric dicts (one per fold).
            Each dict has keys like "accuracy", "f1", "auc_roc", etc.
        metric_names: Which metrics to plot (default: accuracy, precision, recall, f1, mcc, auc_roc, auc_pr).
        output_path: Path to save the figure.
        figsize: (width, height). Default (12, 4 * n_metrics) for one column of subplots.
        title: Overall figure suptitle.
        ylim: Optional (ymin, ymax) for all subplots (e.g. (0, 1)).
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    default_metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "mcc",
        "auc_roc",
        "auc_pr",
    ]
    if metric_names is None:
        metric_names = default_metrics
    # Restrict to metrics that appear in at least one model's fold dicts
    all_keys = set()
    for per_fold in metrics_by_model.values():
        for d in per_fold:
            all_keys.update(k for k, v in d.items() if v is not None)
    metric_names = [m for m in metric_names if m in all_keys]
    if not metric_names:
        return

    n_metrics = len(metric_names)
    n_models = len(metrics_by_model)
    if figsize is None:
        figsize = (max(10, n_models * 1.5), 4 * n_metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    model_names = list(metrics_by_model.keys())
    for ax_idx, metric in enumerate(metric_names):
        ax = axes[ax_idx]
        data_for_box = []
        labels = []
        for name in model_names:
            values = []
            for fold_dict in metrics_by_model[name]:
                if metric in fold_dict and fold_dict[metric] is not None:
                    values.append(float(fold_dict[metric]))
            if values:
                data_for_box.append(values)
                labels.append(name)
        if data_for_box:
            bp = ax.boxplot(
                data_for_box,
                labels=labels,
                patch_artist=True,
                showmeans=True,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor("lightblue")
                patch.set_alpha(0.7)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()} across folds")
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Publication figures: ROC/PR comparison, confusion matrices, training curves
# ---------------------------------------------------------------------------

def plot_roc_comparison(
    model_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: str,
    title: str = "ROC Curves Comparison",
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    """
    Plot ROC curves for multiple models on one figure (Figure 4 in plan).

    Args:
        model_results: Dict mapping model_name -> (y_true, y_pred_probs).
            y_pred_probs: 1D probability of positive class, or 2D (n, n_classes).
        output_path: Path to save the figure.
        title: Plot title.
        class_names: Optional names for classes (for legend/display).
        figsize: Figure size.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(model_results))))

    for i, (name, (y_true, y_probs)) in enumerate(model_results.items()):
        y_true = np.asarray(y_true).ravel()
        y_probs = np.asarray(y_probs)
        if y_probs.ndim == 2:
            pos_probs = y_probs[:, 1] if y_probs.shape[1] == 2 else y_probs[:, 1]
        else:
            pos_probs = y_probs.ravel()
        fpr, tpr, _ = roc_curve(y_true, pos_probs)
        roc_auc = auc(fpr, tpr)
        color = colors[i % len(colors)]
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pr_comparison(
    model_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: str,
    title: str = "Precision-Recall Curves Comparison",
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    """
    Plot Precision-Recall curves for multiple models (Figure 5 in plan).

    Args:
        model_results: Dict mapping model_name -> (y_true, y_pred_probs).
        output_path: Path to save the figure.
        title: Plot title.
        figsize: Figure size.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(model_results))))

    for i, (name, (y_true, y_probs)) in enumerate(model_results.items()):
        y_true = np.asarray(y_true).ravel()
        y_probs = np.asarray(y_probs)
        if y_probs.ndim == 2:
            pos_probs = y_probs[:, 1] if y_probs.shape[1] == 2 else y_probs[:, 1]
        else:
            pos_probs = y_probs.ravel()
        prec, rec, _ = precision_recall_curve(y_true, pos_probs)
        pr_auc = auc(rec, prec)
        color = colors[i % len(colors)]
        ax.plot(rec, prec, color=color, lw=2, label=f"{name} (AUC = {pr_auc:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrices_top_k(
    model_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: str,
    top_k: int = 3,
    class_names: Optional[List[str]] = None,
    sort_by: str = "accuracy",
    figsize_per: Tuple[float, float] = (4, 3),
) -> None:
    """
    Plot confusion matrices for top K models in a grid (Figure 3 in plan).

    Args:
        model_results: Dict mapping model_name -> (y_true, y_pred).
            y_pred can be labels (1D) or logits (2D); argmax used if 2D.
        output_path: Path to save the figure.
        top_k: Number of top models to show (default 3).
        class_names: Optional labels for classes (e.g. ['Normal', 'Abnormal']).
        sort_by: Metric to rank models: 'accuracy', 'f1', etc. (uses accuracy if missing).
        figsize_per: (width, height) per subplot.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    from sklearn.metrics import accuracy_score, f1_score

    scored = []
    for name, (y_true, y_pred) in model_results.items():
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred)
        if y_pred.ndim == 2:
            y_pred = y_pred.argmax(axis=1)
        else:
            y_pred = y_pred.ravel()
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
        scored.append((name, y_true, y_pred, acc, f1))

    metric_idx = 3 if sort_by == "accuracy" else 4
    scored.sort(key=lambda x: x[metric_idx], reverse=True)
    top = scored[:top_k]
    if not top:
        return

    n = len(top)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows), squeeze=False
    )
    axes = axes.ravel()
    for i, (name, y_true, y_pred, acc, f1) in enumerate(top):
        ax = axes[i]
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title(f"{name}\nAcc={acc:.3f}, F1={f1:.3f}")
    for j in range(len(top), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Confusion Matrices (Top {} Models)".format(top_k), y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_training_curves(
    history_by_model: Dict[str, Dict[str, List[float]]],
    output_path: str,
    metrics: Optional[List[str]] = None,
    title: str = "Training Curves",
    figsize: Tuple[int, int] = (12, 5),
) -> None:
    """
    Plot training/validation loss and accuracy (or other metrics) for multiple models (Figure 2).

    Args:
        history_by_model: Dict mapping model_name -> dict with keys e.g. 'train_loss', 'val_loss',
            'train_accuracy', 'val_accuracy' (or 'val_acc'). Each value is a list of per-epoch values.
        output_path: Path to save the figure (single figure with subplots).
        metrics: Pairs to plot as (loss, accuracy). Default: [('train_loss','val_loss'), ('train_accuracy','val_accuracy')]
            with fallback keys 'val_acc', 'train_acc'.
        title: Suptitle.
        figsize: (width, height).
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if metrics is None:
        metrics = [("train_loss", "val_loss"), ("train_accuracy", "val_accuracy")]
    # Normalize key names: accept val_acc / train_acc
    def get_series(hist: Dict[str, List[float]], keys: List[str]) -> Optional[List[float]]:
        for k in keys:
            if k in hist and hist[k]:
                return hist[k]
        return None

    n_plots = len(metrics)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize, squeeze=(n_plots == 1))
    if n_plots == 1:
        axes = [axes]
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(history_by_model))))

    for ax_idx, (key_a, key_b) in enumerate(metrics):
        ax = axes[ax_idx]
        key_b_alt = "val_acc" if key_b == "val_accuracy" else key_b
        key_a_alt = "train_acc" if key_a == "train_accuracy" else key_a
        for i, (name, hist) in enumerate(history_by_model.items()):
            series_a = get_series(hist, [key_a, key_a_alt])
            series_b = get_series(hist, [key_b, key_b_alt])
            if series_a is not None:
                epochs = range(1, len(series_a) + 1)
                ax.plot(epochs, series_a, color=colors[i % len(colors)], linestyle="-", label=f"{name} (train)" if key_b else f"{name}", lw=1.5)
            if series_b is not None and series_b != series_a:
                epochs_b = range(1, len(series_b) + 1)
                ax.plot(epochs_b, series_b, color=colors[i % len(colors)], linestyle="--", label=f"{name} (val)", lw=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key_a.replace("_", " ").title() if "loss" in key_a else "Accuracy")
        ax.set_title(f"{key_a.replace('_', ' ').title()} / {key_b.replace('_', ' ').title()}")
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_ablation_bars(
    results_by_study: Dict[str, Dict[str, float]],
    output_path: str,
    metric: str = "accuracy",
    title: str = "Ablation Study Results",
    figsize: Tuple[int, int] = (10, 5),
    ylim: Optional[Tuple[float, float]] = (0, 1),
    bar_colors: Optional[Dict[str, str]] = None,
) -> None:
    """
    Bar chart for ablation studies (Figure 8): e.g. fusion strategies, contrastive losses.

    Args:
        results_by_study: Dict mapping study_name -> {config_name: metric_value}.
            E.g. {"Fusion": {"Early": 0.85, "Late": 0.82, "ConFu": 0.88}, "Loss": {"SimCLR": 0.8, "VICReg": 0.84}}.
            Or single study: {"Fusion": {"Early": 0.85, "Late": 0.82}}.
        output_path: Path to save the figure.
        metric: Label for y-axis (e.g. 'accuracy', 'F1').
        title: Plot title.
        figsize: Figure size.
        ylim: Optional (ymin, ymax).
        bar_colors: Optional dict config_name -> color (applied across subplots).
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    n_studies = len(results_by_study)
    fig, axes = plt.subplots(1, n_studies, figsize=(figsize[0], figsize[1]), squeeze=(n_studies == 1))
    if n_studies == 1:
        axes = [axes]
    colors = plt.cm.tab10(np.linspace(0, 1, 12))

    for ax_idx, (study_name, config_metrics) in enumerate(results_by_study.items()):
        ax = axes[ax_idx]
        names = list(config_metrics.keys())
        values = [float(config_metrics[n]) for n in names]
        cols = [bar_colors.get(n, colors[i % len(colors)]) for i, n in enumerate(names)] if bar_colors else [colors[i % len(colors)] for i in range(len(names))]
        bars = ax.bar(names, values, color=cols)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(study_name)
        ax.tick_params(axis="x", rotation=45)
        if ylim:
            ax.set_ylim(ylim)
        ax.grid(axis="y", alpha=0.3)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_computational_efficiency(
    model_stats: Dict[str, Dict[str, Union[float, int, str]]],
    output_path: str,
    metrics_order: Optional[List[str]] = None,
    title: str = "Computational Efficiency Comparison",
    figsize: Tuple[int, int] = (12, 5),
) -> None:
    """
    Plot computational efficiency: params (M), FLOPs (G), inference time (ms), memory (GB) (Figure 9).

    Args:
        model_stats: Dict mapping model_name -> {param_count or params_M, flops_G, inference_time_ms, memory_GB, etc}.
            Numeric values can be int or float; they are plotted as bars (normalized or raw).
        output_path: Path to save the figure.
        metrics_order: Which metrics to plot and in what order (e.g. ['params_M', 'flops_G', 'inference_time_ms']).
        title: Plot title.
        figsize: Figure size.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    all_keys = set()
    for d in model_stats.values():
        for k, v in d.items():
            if isinstance(v, (int, float)) and k not in ("param_count",):
                all_keys.add(k)
        if "param_count" in d:
            all_keys.add("params_M")
    if metrics_order is None:
        metrics_order = ["params_M", "flops_G", "inference_time_ms", "memory_GB"]
    metrics_order = [m for m in metrics_order if m in all_keys or (m == "params_M" and any("param" in str(k) for d in model_stats.values() for k in d))]
    # Build params_M from param_count if present
    for name, d in list(model_stats.items()):
        if "params_M" not in d and "param_count" in d:
            model_stats[name] = {**d, "params_M": d["param_count"] / 1e6}
    if not metrics_order:
        metrics_order = sorted(all_keys)

    n_metrics = len(metrics_order)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize, squeeze=(n_metrics == 1))
    if n_metrics == 1:
        axes = [axes]
    model_names = list(model_stats.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(model_names))))

    for ax_idx, m in enumerate(metrics_order):
        ax = axes[ax_idx]
        key = m
        if m == "params_M":
            vals = [model_stats[n].get("params_M") or (model_stats[n].get("param_count", 0) / 1e6) for n in model_names]
        else:
            vals = [model_stats[n].get(m) for n in model_names]
        vals = [float(v) if v is not None else 0 for v in vals]
        bars = ax.bar(model_names, vals, color=[colors[i % len(colors)] for i in range(len(model_names))])
        ax.set_ylabel(m.replace("_", " ").title())
        ax.set_title(m.replace("_", " "))
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01 * (max(vals) or 1), f"{v:.2f}", ha="center", fontsize=8)
    fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_architecture_diagram(
    output_path: str,
    title: str = "Multimodal Fusion Architecture",
    figsize: Tuple[int, int] = (12, 8),
) -> None:
    """
    Draw a programmatic architecture diagram for multimodal (audio + vision) fusion (Figure 1).

    Shows: Audio input -> Audio Encoder; Image/Spectrogram -> Vision Encoder;
    Fusion (Early / Late / Cross-Attention / ConFu / ProMoBal) -> Classifier -> Output.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    # Colors
    audio_c = "#4A90D9"
    vision_c = "#E24A4A"
    fusion_c = "#50C878"
    cls_c = "#9B59B6"

    def box(ax, x, y, w, h, label, color, fontsize=10):
        from matplotlib.patches import FancyBboxPatch
        p = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02", facecolor=color, edgecolor="black", alpha=0.8)
        ax.add_patch(p)
        ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, wrap=True)

    def arrow(ax, x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=2))

    # Inputs
    box(ax, 2, 8, 1.8, 0.8, "Audio\nSpectrogram", audio_c)
    box(ax, 2, 6, 1.8, 0.8, "Image /\nSpectrogram", vision_c)
    # Encoders
    box(ax, 2, 5, 1.8, 0.8, "Audio Encoder\n(AST)", audio_c)
    box(ax, 2, 3, 1.8, 0.8, "Vision Encoder\n(DINOv2/MAE/Swin/...)", vision_c)
    # Fusion block
    box(ax, 5, 4, 2.2, 2.2, "Fusion\n(Early / Late /\nCross-Attn /\nConFu / ProMoBal)", fusion_c, fontsize=8)
    # Classifier
    box(ax, 8, 4, 1.6, 1.0, "Classifier\n(Normal / Abnormal)", cls_c, fontsize=9)

    # Arrows
    arrow(ax, 2, 7.2, 2, 5.4)
    arrow(ax, 2, 6.2, 2, 3.4)
    arrow(ax, 2, 4.6, 3.1, 4)
    arrow(ax, 2, 2.6, 3.1, 4)
    arrow(ax, 6.1, 4, 7.2, 4)
    ax.text(4.5, 4.5, "concat / attention\n/ contrastive", fontsize=7, ha="center", style="italic")
    ax.text(7, 4.5, "logits", fontsize=7, ha="center")
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Enhanced interpretability visualizations for Q1 publication
# ---------------------------------------------------------------------------

def plot_tsne_with_decision_boundary(
    embeddings: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    output_path: str,
    title: str = "t-SNE with Decision Regions",
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 10),
    perplexity: float = 30.0,
    random_state: int = 42,
) -> None:
    """
    t-SNE visualization showing misclassified samples and approximate decision regions.
    Useful for understanding model behavior at class boundaries.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    X = np.asarray(embeddings, dtype=np.float64)
    y = np.asarray(labels).ravel()
    preds = np.asarray(predictions).ravel()
    
    perplexity_actual = min(perplexity, max(5, X.shape[0] // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity_actual, random_state=random_state)
    coords = tsne.fit_transform(X)
    
    correct = y == preds
    misclassified = ~correct
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if class_names is None:
        class_names = ["Normal", "Abnormal"]
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i, label in enumerate([0, 1]):
        mask = (y == label) & correct
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[colors[i]], label=f"{class_names[i]} (correct)",
            alpha=0.6, s=30, marker='o'
        )
    
    for i, label in enumerate([0, 1]):
        mask = (y == label) & misclassified
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[colors[i]], label=f"{class_names[i]} (misclassified)",
            alpha=0.9, s=80, marker='x', linewidths=2
        )
    
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    n_correct = correct.sum()
    n_total = len(y)
    accuracy = n_correct / n_total
    ax.text(
        0.02, 0.98, f"Accuracy: {accuracy:.2%} ({n_correct}/{n_total})",
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_class_distribution_analysis(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 5),
) -> Dict[str, float]:
    """
    Analyze and visualize class distributions in embedding space.
    
    Computes:
    - Intra-class variance (tightness of clusters)
    - Inter-class distance (separation between classes)
    - Silhouette score
    
    Returns metrics dict for inclusion in publication tables.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    from sklearn.metrics import silhouette_score
    
    X = np.asarray(embeddings)
    y = np.asarray(labels).ravel()
    
    if class_names is None:
        class_names = ["Normal", "Abnormal"]
    
    unique_labels = np.unique(y)
    centroids = {}
    intra_class_var = {}
    
    for label in unique_labels:
        mask = y == label
        class_embeddings = X[mask]
        centroid = class_embeddings.mean(axis=0)
        centroids[label] = centroid
        
        distances = np.linalg.norm(class_embeddings - centroid, axis=1)
        intra_class_var[label] = distances.mean()
    
    if len(centroids) >= 2:
        inter_class_dist = np.linalg.norm(centroids[0] - centroids[1])
    else:
        inter_class_dist = 0.0
    
    try:
        sil_score = silhouette_score(X, y)
    except:
        sil_score = 0.0
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    ax1 = axes[0]
    vars_list = [intra_class_var.get(l, 0) for l in unique_labels]
    bars = ax1.bar([class_names[l] for l in unique_labels], vars_list, color=['#4A90D9', '#E24A4A'])
    ax1.set_ylabel("Mean Distance to Centroid")
    ax1.set_title("Intra-class Variance\n(lower = tighter clusters)")
    for bar, val in zip(bars, vars_list):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', fontsize=10)
    
    ax2 = axes[1]
    ax2.bar(["Inter-class"], [inter_class_dist], color='#50C878')
    ax2.set_ylabel("Distance")
    ax2.set_title("Inter-class Distance\n(higher = better separation)")
    ax2.text(0, inter_class_dist + 0.01, f'{inter_class_dist:.3f}', ha='center', fontsize=10)
    
    ax3 = axes[2]
    colors = ['#E24A4A' if sil_score < 0.3 else '#FFD700' if sil_score < 0.5 else '#50C878']
    ax3.bar(["Silhouette"], [sil_score], color=colors)
    ax3.set_ylabel("Score")
    ax3.set_ylim(-1, 1)
    ax3.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Good threshold')
    ax3.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Weak threshold')
    ax3.set_title("Silhouette Score\n(higher = better clustering)")
    ax3.text(0, sil_score + 0.03, f'{sil_score:.3f}', ha='center', fontsize=10)
    ax3.legend(loc='lower right', fontsize=8)
    
    fig.suptitle("Embedding Space Analysis", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    metrics = {
        "intra_class_var_normal": intra_class_var.get(0, 0),
        "intra_class_var_abnormal": intra_class_var.get(1, 0),
        "inter_class_distance": inter_class_dist,
        "silhouette_score": sil_score,
        "separation_ratio": inter_class_dist / (sum(vars_list) / len(vars_list) + 1e-8),
    }
    
    return metrics


def plot_attention_analysis_grid(
    attention_maps: List[np.ndarray],
    images: List[np.ndarray],
    labels: List[int],
    predictions: List[int],
    output_path: str,
    class_names: Optional[List[str]] = None,
    n_samples: int = 8,
    figsize: Tuple[int, int] = (16, 8),
) -> None:
    """
    Create a grid showing attention maps for correct and incorrect predictions.
    Useful for interpretability analysis in publications.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    if class_names is None:
        class_names = ["Normal", "Abnormal"]
    
    labels = np.array(labels)
    predictions = np.array(predictions)
    correct_mask = labels == predictions
    
    correct_idx = np.where(correct_mask)[0]
    incorrect_idx = np.where(~correct_mask)[0]
    
    n_correct = min(n_samples // 2, len(correct_idx))
    n_incorrect = min(n_samples // 2, len(incorrect_idx))
    
    if n_correct > 0:
        correct_samples = np.random.choice(correct_idx, n_correct, replace=False)
    else:
        correct_samples = []
    
    if n_incorrect > 0:
        incorrect_samples = np.random.choice(incorrect_idx, n_incorrect, replace=False)
    else:
        incorrect_samples = []
    
    n_total = len(correct_samples) + len(incorrect_samples)
    if n_total == 0:
        return
    
    ncols = 4
    nrows = (n_total + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.ravel() if n_total > 1 else [axes]
    
    all_samples = list(correct_samples) + list(incorrect_samples)
    
    for i, idx in enumerate(all_samples):
        ax = axes[i]
        
        img = images[idx]
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        if img.max() > 1:
            img = img / 255.0
        
        attn = attention_maps[idx]
        if attn.shape != img.shape[:2]:
            from scipy.ndimage import zoom
            scale = (img.shape[0] / attn.shape[0], img.shape[1] / attn.shape[1])
            attn = zoom(attn, scale, order=1)
        
        ax.imshow(img)
        ax.imshow(attn, cmap='jet', alpha=0.4)
        
        true_label = class_names[labels[idx]]
        pred_label = class_names[predictions[idx]]
        is_correct = labels[idx] == predictions[idx]
        
        color = 'green' if is_correct else 'red'
        title = f"True: {true_label}\nPred: {pred_label}"
        ax.set_title(title, color=color, fontsize=9)
        ax.axis('off')
    
    for j in range(len(all_samples), len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle("Attention Analysis: Correct vs Misclassified", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_interpretability_report(
    model: nn.Module,
    val_loader,
    output_dir: str,
    device: torch.device = None,
    class_names: Optional[List[str]] = None,
    n_grad_cam_samples: int = 16,
) -> Dict[str, Any]:
    """
    Generate comprehensive interpretability report for publication.
    
    Creates:
    1. t-SNE visualization with decision analysis
    2. Class distribution metrics
    3. Grad-CAM attention analysis grid
    4. Embedding space statistics
    
    Returns dict with all computed metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    
    all_embeddings = []
    all_labels = []
    all_predictions = []
    all_images = []
    all_attention_maps = []
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(val_loader):
            x = x.to(device)
            
            if hasattr(model, 'encoder'):
                emb = model.encoder(x)
            else:
                emb = model(x)
            
            if emb.dim() > 2:
                emb = emb.mean(dim=list(range(1, emb.dim() - 1)))
            
            logits = model(x) if emb.dim() == 2 else emb
            preds = logits.argmax(dim=1)
            
            all_embeddings.append(emb.cpu().numpy())
            all_labels.extend(y.numpy().tolist())
            all_predictions.extend(preds.cpu().numpy().tolist())
            
            if batch_idx < n_grad_cam_samples // x.size(0) + 1:
                for i in range(min(x.size(0), n_grad_cam_samples - len(all_images))):
                    img = x[i].cpu().numpy()
                    all_images.append(img)
                    
                    try:
                        cam = compute_grad_cam_vit(model, x[i:i+1])
                        all_attention_maps.append(cam)
                    except:
                        all_attention_maps.append(np.zeros((14, 14)))
    
    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.array(all_labels)
    predictions = np.array(all_predictions)
    
    plot_tsne_with_decision_boundary(
        embeddings, labels, predictions,
        os.path.join(output_dir, "tsne_decision_boundary.png"),
        class_names=class_names
    )
    
    distribution_metrics = plot_class_distribution_analysis(
        embeddings, labels,
        os.path.join(output_dir, "class_distribution_analysis.png"),
        class_names=class_names
    )
    
    if all_images and all_attention_maps:
        plot_attention_analysis_grid(
            all_attention_maps[:n_grad_cam_samples],
            all_images[:n_grad_cam_samples],
            labels[:n_grad_cam_samples].tolist(),
            predictions[:n_grad_cam_samples].tolist(),
            os.path.join(output_dir, "attention_analysis_grid.png"),
            class_names=class_names
        )
    
    plot_tsne_umap(
        embeddings, labels,
        output_dir=output_dir,
        class_names=class_names
    )
    
    accuracy = (labels == predictions).mean()
    
    report = {
        "accuracy": accuracy,
        "n_samples": len(labels),
        **distribution_metrics,
    }
    
    import json
    with open(os.path.join(output_dir, "interpretability_metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    
    return report
