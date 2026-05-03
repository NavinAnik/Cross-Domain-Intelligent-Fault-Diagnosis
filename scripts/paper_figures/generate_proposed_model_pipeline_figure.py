"""
Generate a transfer-learning pipeline figure (journal-style) for the proposed
MIMII spectrogram fault diagnosis stack: DINOv2 pretrain → partial fine-tune,
HTSF (temporal / spatial pooling + cross-modal transformer), enhanced head,
AFCL training objective.

Output:
  src/paper_assets/figures/proposed_model_transfer_learning_pipeline.png
  results/all_metrics_plots/ProposedModel_transfer_learning_pipeline.png
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

# Repo root (parent of scripts/)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _box(ax, xy, w, h, text, fontsize=8, fc="#E8F4FC", ec="#2E5A88", lw=1.2, bold=False):
    x, y = xy
    fb = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.012",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(fb)
    weight = "bold" if bold else "normal"
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
        wrap=True,
    )
    return fb


def _arrow(ax, start, end, text=None, rad=0.0, color="#333333"):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.5,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)
    if text:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        ax.text(mx, my + 0.02, text, ha="center", fontsize=7, style="italic", color="#444444")
    return arr


def build_figure():
    fig, ax = plt.subplots(1, 1, figsize=(14.5, 9.2), dpi=150)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # --- Title band ---
    ax.text(
        0.5,
        0.97,
        "Proposed pipeline: transfer learning for MIMII spectrogram fault diagnosis",
        ha="center",
        va="top",
        fontsize=13,
        weight="bold",
    )
    ax.text(
        0.5,
        0.935,
        "Encoder weights from self-supervised ViT (DINOv2) → on-site adaptation with HTSF fusion + AFCL",
        ha="center",
        va="top",
        fontsize=9,
        color="#444444",
    )

    # --- Left column: physical + spectrogram ---
    _box(ax, (0.02, 0.62), 0.14, 0.28, "Industrial\nmachine &\nsensors\n(MIMII)", fontsize=8, fc="#F5F5F5", ec="#666666")
    # Stylized spectrogram placeholder
    sg = Rectangle((0.045, 0.68), 0.09, 0.12, facecolor="#1a1a2e", edgecolor="#333", linewidth=1)
    ax.add_patch(sg)
    for i in range(10):
        yi = 0.685 + i * 0.011
        ax.plot([0.05, 0.13], [yi, yi + 0.002], color="#4a9eff", linewidth=0.8, alpha=0.35 + 0.05 * i)
    ax.text(0.09, 0.66, "Spectrogram\n(RGB, 224×224)", ha="center", fontsize=7, color="#222")

    # --- SOURCE TASK (top path) ---
    ax.text(0.22, 0.88, "Source task (off-the-shelf pretraining)", fontsize=10, weight="bold", color="#1a5276")
    _box(
        ax,
        (0.20, 0.72),
        0.36,
        0.14,
        "DINOv2 ViT-B/14 encoder\n(torch.hub — LVD-scale SSL weights)\n12 × Transformer block + final LayerNorm\nOutput: patch tokens + [CLS]  (768-D)",
        fontsize=8,
        fc="#D4E9F7",
        ec="#1a5276",
        bold=True,
    )
    _box(ax, (0.58, 0.74), 0.18, 0.10, "Pretraining\nobjective\n(SSL on\nlarge image\ncorpus)", fontsize=7, fc="#EAF2F8", ec="#1a5276")

    _arrow(ax, (0.16, 0.76), (0.20, 0.79), rad=0.0)
    _arrow(ax, (0.56, 0.79), (0.58, 0.79), rad=0.0)

    # --- TRANSFER ---
    ax.text(0.5, 0.58, "⟵  Transfer pretrained ViT weights  ⟶", ha="center", fontsize=10, weight="bold", color="#c0392b")
    ax.plot([0.18, 0.82], [0.555, 0.555], color="#c0392b", linewidth=2, linestyle="--", alpha=0.7)

    # --- TARGET TASK (bottom path) — matches run_experiment_5 + DINOv2Classifier ---
    ax.text(0.22, 0.52, "Target task (MIMII fault classification — Exp. 5 pipeline)", fontsize=10, weight="bold", color="#145a32")
    _box(
        ax,
        (0.18, 0.36),
        0.20,
        0.12,
        "Spectrogram batch\n(B, 3, 224, 224)\n+ SpecAug-style\nmasks & jitter\n(run_improved_experiments)",
        fontsize=7,
        fc="#D5F5E3",
        ec="#145a32",
    )
    _box(
        ax,
        (0.40, 0.36),
        0.22,
        0.12,
        "DINOv2 ViT-B/14\nget_features() → 768-D\n[CLS] / patch tokens\n(partial_unfreeze:\nlast 4 blocks trainable)",
        fontsize=7,
        fc="#D5F5E3",
        ec="#145a32",
        bold=True,
    )
    _arrow(ax, (0.09, 0.62), (0.28, 0.42), rad=0.12)
    _arrow(ax, (0.28, 0.42), (0.40, 0.42), rad=0.0)

    _box(
        ax,
        (0.64, 0.36),
        0.32,
        0.12,
        "Optional multimodal extension — HTSF\n(fusion/htsf_fusion.py, Exp. 4)\nDual encoders + temporal & spatial\nattention pooling + cross-modal\nTransformer → fused embedding",
        fontsize=6.5,
        fc="#E8F8F5",
        ec="#117a65",
        lw=1.0,
    )
    ax.text(
        0.64,
        0.50,
        "when two views / modalities are wired\ninto audio + vision branches",
        fontsize=6,
        color="#117a65",
        style="italic",
    )
    _arrow(ax, (0.51, 0.42), (0.64, 0.42), rad=0.0, color="#888888")

    _box(
        ax,
        (0.20, 0.17),
        0.42,
        0.14,
        "Classifier head (modules/dinov2_encoder.py)\nEnhanced MLP: 768 → [512 → 256 → 128] → logits (K=2)\nAdamW: backbone_lr 1e-5, head_lr 1e-3 | AMP + grad clip\nMixup (α=0.2) | TTA (5 views) at eval | early stop on balanced acc",
        fontsize=7.5,
        fc="#FCF3CF",
        ec="#9a7d0a",
        bold=True,
    )
    _arrow(ax, (0.51, 0.36), (0.48, 0.31), rad=0.15)

    _box(
        ax,
        (0.66, 0.17),
        0.30,
        0.14,
        "Loss: AFCL (modules/afcl_loss.py)\nℒ = ℒ_focal + w(t)·λ_c·ℒ_contrast\n+ w(t)·λ_a·ℒ_align\nwarmup on contrast / align terms",
        fontsize=7,
        fc="#FADBD8",
        ec="#922b21",
    )
    _arrow(ax, (0.62, 0.24), (0.66, 0.24), rad=0.0)

    ax.text(
        0.5,
        0.06,
        "(a) Industrial fault diagnosis from MIMII spectrograms via transfer learning from a self-supervised ViT encoder,\n"
        "partial fine-tuning with an enhanced MLP head, and asymmetric focal contrastive learning (AFCL). "
        "HTSF denotes an optional multimodal fusion block used in separate fusion experiments.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        style="italic",
    )

    # Legend strip
    leg = [
        mpatches.Patch(facecolor="#D4E9F7", edgecolor="#1a5276", label="Pretrained encoder (source)"),
        mpatches.Patch(facecolor="#D5F5E3", edgecolor="#145a32", label="On-site spectrogram + ViT (target)"),
        mpatches.Patch(facecolor="#E8F8F5", edgecolor="#117a65", label="HTSF (optional multimodal)"),
        mpatches.Patch(facecolor="#FCF3CF", edgecolor="#9a7d0a", label="Enhanced head + training recipe"),
        mpatches.Patch(facecolor="#FADBD8", edgecolor="#922b21", label="AFCL composite loss"),
    ]
    ax.legend(
        handles=leg,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.32),
        fontsize=7,
        framealpha=0.95,
    )

    plt.tight_layout()
    return fig


def main():
    fig = build_figure()
    out1 = os.path.join(ROOT, "src", "paper_assets", "figures", "proposed_model_transfer_learning_pipeline.png")
    out2 = os.path.join(ROOT, "results", "all_metrics_plots", "ProposedModel_transfer_learning_pipeline.png")
    os.makedirs(os.path.dirname(out1), exist_ok=True)
    os.makedirs(os.path.dirname(out2), exist_ok=True)
    fig.savefig(out1, bbox_inches="tight", facecolor="white")
    fig.savefig(out2, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Wrote:", out1)
    print("Wrote:", out2)


if __name__ == "__main__":
    main()
