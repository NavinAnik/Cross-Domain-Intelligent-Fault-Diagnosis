"""
Generate methodology and architecture schematic PNGs for the spectrogram-fault paper supplement.

Usage (from repository root):
  python scripts/paper_figures/generate_spectrogram_fault_figures.py

Outputs (300 DPI) under src/paper_assets/figures/:
  - methodology_pipeline.png
  - architecture_cnn_baseline.png
  - architecture_cnn_spatial_attention.png
  - architecture_cnn_transformer_hybrid.png
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _out_dir() -> Path:
    d = _repo_root() / "src" / "paper_assets" / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#F4F6FA",
            "axes.facecolor": "#F4F6FA",
            "savefig.facecolor": "#F4F6FA",
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Helvetica", "Arial"],
            "font.size": 9,
            "axes.edgecolor": "#D0D7E2",
            "text.color": "#1A2332",
            "axes.labelcolor": "#1A2332",
        }
    )


def _gradient_background(ax, x0: float, x1: float, y0: float, y1: float) -> None:
    """Subtle vertical wash behind the diagram."""
    ny, nx = 120, 2
    g = np.linspace(0.98, 0.92, ny).reshape(ny, 1)
    g = np.repeat(g, nx, axis=1)
    extent = [x0, x1, y0, y1]
    ax.imshow(
        g,
        extent=extent,
        aspect="auto",
        zorder=0,
        cmap="coolwarm",
        alpha=0.12,
        vmin=0,
        vmax=1,
    )


def _title_block(ax, title: str, subtitle: str | None, x0: float, x1: float, y1: float) -> None:
    ax.text(
        (x0 + x1) / 2,
        y1 - 0.05,
        title,
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
        color="#0D1B2A",
        zorder=20,
    )
    if subtitle:
        ax.text(
            (x0 + x1) / 2,
            y1 - 0.14,
            subtitle,
            ha="center",
            va="top",
            fontsize=8.5,
            color="#415A77",
            style="italic",
            zorder=20,
        )


def _fancy_box(
    ax,
    xy: tuple[float, float],
    w: float,
    h: float,
    title: str,
    body: str | None = None,
    *,
    fc: str,
    ec: str,
    z: int = 5,
) -> None:
    x, y = xy
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        linewidth=1.35,
        edgecolor=ec,
        facecolor=fc,
        zorder=z,
    )
    p.set_path_effects(
        [
            pe.SimplePatchShadow(offset=(2.8, -2.8), shadow_rgbFace="#1A2332", alpha=0.22),
            pe.Normal(),
        ]
    )
    ax.add_patch(p)
    if body:
        ax.text(
            x + w / 2,
            y + h * 0.58,
            title,
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="#0D1B2A",
            zorder=z + 1,
        )
        ax.text(
            x + w / 2,
            y + h * 0.28,
            body,
            ha="center",
            va="center",
            fontsize=7.8,
            color="#334155",
            linespacing=1.15,
            zorder=z + 1,
        )
    else:
        ax.text(
            x + w / 2,
            y + h / 2,
            title,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#0D1B2A",
            linespacing=1.2,
            zorder=z + 1,
        )


def _arrow(
    ax,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str = "#3D5A80",
    lw: float = 1.8,
) -> None:
    a = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=lw,
        color=color,
        shrinkA=2,
        shrinkB=2,
        zorder=4,
        path_effects=[pe.Stroke(linewidth=lw + 2.5, foreground="white", alpha=0.85), pe.Normal()],
    )
    ax.add_patch(a)


def _footer_note(ax, text: str, x_mid: float, y: float) -> None:
    pad_w, pad_h = 8.8, 0.22
    fx = x_mid - pad_w / 2
    foot = FancyBboxPatch(
        (fx, y),
        pad_w,
        pad_h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
        linewidth=1.0,
        alpha=0.95,
        zorder=3,
    )
    foot.set_path_effects(
        [pe.SimplePatchShadow(offset=(1.5, -1.5), shadow_rgbFace="#64748B", alpha=0.12), pe.Normal()]
    )
    ax.add_patch(foot)
    ax.text(
        x_mid,
        y + pad_h / 2,
        text,
        ha="center",
        va="center",
        fontsize=7.5,
        color="#475569",
        zorder=4,
    )


def draw_methodology_pipeline(path: Path) -> None:
    _apply_style()
    x0, x1, y0, y1 = 0.02, 9.98, 0.02, 1.38
    fig, ax = plt.subplots(figsize=(11.2, 3.35), dpi=150)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.axis("off")
    _gradient_background(ax, x0, x1, y0, y1)
    _title_block(
        ax,
        "Spectrogram-image fault classification",
        "End-to-end pipeline (supervised)",
        x0,
        x1,
        y1,
    )

    w, h = 1.38, 0.52
    y = 0.48
    stages = [
        (0.12, y, "Data", "Spectrogram\nPNG folders", "#DBEAFE", "#1D4ED8"),
        (1.72, y, "Loader", "ImageDataset\nlabels · groups", "#EDE9FE", "#5B21B6"),
        (3.32, y, "Split", "Stratified\nGroupKFold", "#D1FAE5", "#047857"),
        (4.92, y, "Augment", "weak → strong\nSpecAugment-style", "#FEF3C7", "#B45309"),
        (6.52, y, "Model", "timm CNN\nattn · transformer", "#FFE4E6", "#BE123C"),
        (8.12, y, "Output", "softmax\nmetrics · figures", "#E2E8F0", "#334155"),
    ]
    for i, (sx, sy, t, b, fc, ec) in enumerate(stages):
        _fancy_box(ax, (sx, sy), w, h, t, b, fc=fc, ec=ec, z=5 + i)
        if i > 0:
            prev_x = stages[i - 1][0] + w
            cy = sy + h / 2
            _arrow(ax, prev_x + 0.02, cy, sx - 0.02, cy, color="#475569")

    _footer_note(
        ax,
        "Validation: resize + ImageNet normalize  ·  Training: affine / color jitter + time–frequency masks + noise",
        (x0 + x1) / 2,
        0.12,
    )
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def draw_cnn_baseline(path: Path) -> None:
    _apply_style()
    x0, x1, y0, y1 = 0.02, 8.85, 0.02, 1.22
    fig, ax = plt.subplots(figsize=(9.2, 2.85), dpi=150)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.axis("off")
    _gradient_background(ax, x0, x1, y0, y1)
    _title_block(ax, "CNNBaseline", "ImageNet-pretrained timm backbone (e.g. ResNet50)", x0, x1, y1)

    y, h = 0.38, 0.52
    boxes = [
        (0.18, y, 1.42, h, "Input", "B × 3 × H × W", "#DBEAFE", "#1D4ED8"),
        (1.78, y, 2.15, h, "Backbone", "global_pool=''\nB × C × h × w", "#E0E7FF", "#4338CA"),
        (4.12, y, 1.38, h, "GAP", "B × C", "#D1FAE5", "#047857"),
        (5.72, y, 1.45, h, "Classifier", "Linear → K classes", "#FEF3C7", "#B45309"),
    ]
    for i, (bx, by, bw, bh, t, b, fc, ec) in enumerate(boxes):
        _fancy_box(ax, (bx, by), bw, bh, t, b, fc=fc, ec=ec)
        if i:
            cy = by + bh / 2
            prev = boxes[i - 1]
            _arrow(ax, prev[0] + prev[2] + 0.02, cy, bx - 0.02, cy)

    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def draw_cnn_spatial_attention(path: Path) -> None:
    _apply_style()
    x0, x1, y0, y1 = 0.02, 9.55, 0.02, 1.42
    fig, ax = plt.subplots(figsize=(9.8, 3.15), dpi=150)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.axis("off")
    _gradient_background(ax, x0, x1, y0, y1)
    _title_block(ax, "CNNSpatialAttention", "CBAM-style spatial gate before pooling (`cnn_se`)", x0, x1, y1)

    y, h = 0.42, 0.5
    boxes = [
        (0.12, y, 1.32, h, "Input", "B × 3 × H × W", "#DBEAFE", "#1D4ED8"),
        (1.58, y, 1.95, h, "CNN backbone", "feature map\nB × C × h × w", "#E0E7FF", "#4338CA"),
        (3.68, y, 2.05, h, "SpatialAttention", "avg + max → 7×7 conv\nσ gate", "#FCE7F3", "#BE185D"),
        (5.88, y, 1.22, h, "GAP", "B × C", "#D1FAE5", "#047857"),
        (7.28, y, 1.05, h, "Head", "Linear", "#FEF3C7", "#B45309"),
    ]
    for i, row in enumerate(boxes):
        bx, by, bw, bh, t, b, fc, ec = row
        _fancy_box(ax, (bx, by), bw, bh, t, b, fc=fc, ec=ec)
        if i:
            cy = by + bh / 2
            prev = boxes[i - 1]
            _arrow(ax, prev[0] + prev[2] + 0.02, cy, bx - 0.02, cy)

    _footer_note(
        ax,
        "Attention map (B, h, w) cached for publication-quality heatmaps",
        (x0 + x1) / 2,
        0.14,
    )
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def draw_cnn_transformer_hybrid(path: Path) -> None:
    _apply_style()
    x0, x1, y0, y1 = 0.02, 9.95, 0.02, 1.45
    fig, ax = plt.subplots(figsize=(10.2, 3.25), dpi=150)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.axis("off")
    _gradient_background(ax, x0, x1, y0, y1)
    _title_block(
        ax,
        "CNNTransformerHybrid",
        "2D sinusoidal PE on spatial tokens + PyTorch TransformerEncoder (`cnn_transformer`)",
        x0,
        x1,
        y1,
    )

    y, h = 0.48, 0.46
    boxes = [
        (0.1, y, 1.22, h, "Input", "B × 3 × H × W", "#DBEAFE", "#1D4ED8"),
        (1.48, y, 1.72, h, "CNN", "B × C × h × w", "#E0E7FF", "#4338CA"),
        (3.34, y, 1.42, h, "Tokens", "B × (hw) × C", "#EDE9FE", "#5B21B6"),
        (4.88, y, 1.25, h, "Project", "→ d_model", "#FCE7F3", "#BE185D"),
        (6.23, y, 1.28, h, "PE + CLS", "2D sin-cos", "#FEF3C7", "#B45309"),
        (7.63, y, 1.12, h, "Encoder", "L layers · pre-norm", "#CCFBF1", "#0F766E"),
    ]
    for i, row in enumerate(boxes):
        bx, by, bw, bh, t, b, fc, ec = row
        _fancy_box(ax, (bx, by), bw, bh, t, b, fc=fc, ec=ec)
        if i:
            cy = by + bh / 2
            prev = boxes[i - 1]
            _arrow(ax, prev[0] + prev[2] + 0.02, cy, bx - 0.02, cy)

    hx, hy, hw, hh = 3.25, 0.1, 3.85, 0.34
    _fancy_box(
        ax,
        (hx, hy),
        hw,
        hh,
        "Classification head",
        "LayerNorm → Linear  (CLS token or mean pool)",
        fc="#F1F5F9",
        ec="#475569",
        z=12,
    )
    last = boxes[-1]
    enc_cx = last[0] + last[2] / 2
    head_cx = hx + hw / 2
    top_y = hy + hh
    _arrow(ax, enc_cx, y, enc_cx, top_y + 0.02, color="#0F766E")
    _arrow(ax, enc_cx, top_y + 0.04, head_cx, top_y + 0.04, color="#475569")
    _arrow(ax, head_cx, top_y + 0.04, head_cx, top_y, color="#475569")

    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    _apply_style()
    out = _out_dir()
    draw_methodology_pipeline(out / "methodology_pipeline.png")
    draw_cnn_baseline(out / "architecture_cnn_baseline.png")
    draw_cnn_spatial_attention(out / "architecture_cnn_spatial_attention.png")
    draw_cnn_transformer_hybrid(out / "architecture_cnn_transformer_hybrid.png")
    print("Wrote:", out)


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    main()
