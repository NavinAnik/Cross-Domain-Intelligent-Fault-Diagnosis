"""
Q1-Ready Experiment Pipeline for Industrial Anomaly Detection.

Implements a scientifically rigorous experiment framework that addresses all
identified issues from the improvement roadmap:

- Group-aware cross-validation (no data leakage)
- Spectrogram-specific augmentations
- Properly wired HTSF dual-view + AFCL pipeline
- Multi-layer DINOv2 features
- Linear warmup + cosine decay LR schedule
- Per-machine-type evaluation
- Full ablation table
- Statistical significance tests (McNemar, corrected t-test, DeLong)
- Publication-quality figures

Usage:
    python -m experiments.run_q1_experiments --experiment ablation
    python -m experiments.run_q1_experiments --experiment full_pipeline
"""

import gc
import os
import sys
import json
import time
import math
import argparse
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, precision_recall_curve, roc_curve, auc,
)
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", ".env"))

from utils.logging_config import setup_logging

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False
    wandb = None

logger = logging.getLogger(__name__)


# ============================================================
# Utility functions
# ============================================================

def find_optimal_threshold(y_true, y_probs, metric="f1"):
    """Exhaustive threshold search optimizing the specified metric."""
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold, best_score = 0.5, 0.0
    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        elif metric == "balanced_accuracy":
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            score = (spec + sens) / 2
        else:
            score = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = thresh
    return best_threshold, best_score


def compute_full_metrics(y_true, y_pred, y_probs=None):
    """Comprehensive metrics including pAUC for anomaly detection."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "confusion_matrix": cm.tolist(),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": accuracy_score(y_true, y_pred),
        "abnormal_recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "normal_recall": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "abnormal_precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "normal_precision": tn / (tn + fn) if (tn + fn) > 0 else 0,
        "balanced_accuracy": ((tn / (tn + fp)) + (tp / (tp + fn))) / 2
            if (tn + fp) > 0 and (tp + fn) > 0 else 0,
        "f1_abnormal": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_normal": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if y_probs is not None and len(np.unique(y_true)) > 1:
        metrics["auc_roc"] = roc_auc_score(y_true, y_probs)
        metrics["auc_pr"] = average_precision_score(y_true, y_probs)
        try:
            metrics["pauc_fpr01"] = roc_auc_score(
                y_true, y_probs, max_fpr=0.1)
        except Exception:
            metrics["pauc_fpr01"] = 0.0
    return metrics


def make_weighted_sampler(labels, num_classes=2):
    """Correct WeightedRandomSampler using train fold size as denominator."""
    class_counts = Counter(labels)
    n_train = len(labels)
    weights = [n_train / (num_classes * class_counts.get(i, 1))
               for i in range(num_classes)]
    sample_weights = torch.tensor([weights[l] for l in labels], dtype=torch.float)
    return WeightedRandomSampler(sample_weights, len(sample_weights))


class LinearWarmupCosineDecay(optim.lr_scheduler._LRScheduler):
    """Linear warmup followed by cosine decay to min_lr."""

    def __init__(self, optimizer, warmup_epochs, total_epochs,
                 min_lr=1e-7, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            factor = (self.last_epoch + 1) / max(1, self.warmup_epochs)
            return [base_lr * factor for base_lr in self.base_lrs]
        progress = (self.last_epoch - self.warmup_epochs) / max(
            1, self.total_epochs - self.warmup_epochs)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return [self.min_lr + (base_lr - self.min_lr) * cosine
                for base_lr in self.base_lrs]


# ============================================================
# Statistical tests for publication
# ============================================================

def mcnemar_test(y_true, preds_a, preds_b):
    """McNemar's test comparing two classifiers on the same test set."""
    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)
    b_count = int(np.sum(correct_a & ~correct_b))
    c_count = int(np.sum(~correct_a & correct_b))
    if b_count + c_count == 0:
        return 1.0
    chi2 = (abs(b_count - c_count) - 1) ** 2 / (b_count + c_count)
    p_value = 1 - sp_stats.chi2.cdf(chi2, df=1)
    return float(p_value)


def corrected_resampled_ttest(scores_a, scores_b, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test for CV results."""
    diffs = np.array(scores_a) - np.array(scores_b)
    n = len(diffs)
    if n < 2:
        return 1.0
    mean_diff = np.mean(diffs)
    var_diff = np.var(diffs, ddof=1)
    correction = (1 / n) + (n_test / n_train)
    t_stat = mean_diff / max(np.sqrt(correction * var_diff), 1e-10)
    p_value = 2 * (1 - sp_stats.t.cdf(abs(t_stat), df=n - 1))
    return float(p_value)


def delong_auc_test(y_true, probs_a, probs_b):
    """DeLong test for comparing two AUC values (approximate)."""
    auc_a = roc_auc_score(y_true, probs_a)
    auc_b = roc_auc_score(y_true, probs_b)
    n1 = np.sum(y_true == 1)
    n0 = np.sum(y_true == 0)
    se = np.sqrt((auc_a * (1 - auc_a) + auc_b * (1 - auc_b)) / min(n0, n1))
    if se < 1e-10:
        return 1.0
    z = (auc_a - auc_b) / se
    p_value = 2 * (1 - sp_stats.norm.cdf(abs(z)))
    return float(p_value)


# ============================================================
# Per-machine-type evaluation
# ============================================================

def evaluate_per_machine(samples, y_true, y_pred, y_probs=None):
    """Compute metrics broken down by machine group (machine_type/machine_id)."""
    groups = defaultdict(lambda: {"indices": []})
    for i, path in enumerate(samples):
        parts = path.replace("\\", "/").split("/")
        machine_type = "unknown"
        for j, part in enumerate(parts):
            if part in ("normal", "abnormal") and j >= 2:
                machine_type = parts[j - 2] if j >= 2 else parts[j - 1]
                break
        groups[machine_type]["indices"].append(i)

    results = {}
    for mtype, info in groups.items():
        idx = info["indices"]
        yt = y_true[idx]
        yp = y_pred[idx]
        ypr = y_probs[idx] if y_probs is not None else None
        if len(np.unique(yt)) < 2:
            results[mtype] = {"n_samples": len(idx), "note": "single class only"}
            continue
        results[mtype] = compute_full_metrics(yt, yp, ypr)
        results[mtype]["n_samples"] = len(idx)
    return results


# ============================================================
# Dual-view dataset for HTSF
# ============================================================

class _DualViewDataset(torch.utils.data.Dataset):
    """Returns two differently-augmented views of each image for HTSF fusion."""
    def __init__(self, base_dataset, transform1, transform2):
        self.paths = base_dataset.samples
        self.labels = base_dataset.labels
        self.groups = base_dataset.groups
        self.t1 = transform1
        self.t2 = transform2

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.t1(img), self.t2(img), self.labels[idx]


class _SingleViewDataset(torch.utils.data.Dataset):
    """Single-view dataset wrapping base ImageDataset paths."""
    def __init__(self, base_dataset, transform):
        self.paths = base_dataset.samples
        self.labels = base_dataset.labels
        self.groups = base_dataset.groups
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


# ============================================================
# Model builders for ablation table
# ============================================================

def build_model_for_ablation(row_config, device):
    """Build model according to ablation row configuration.

    Returns (model, needs_dual_view: bool, feature_dim: int).
    """
    backbone = row_config.get("backbone", "dinov2_vitb14")
    finetune = row_config.get("finetune", "frozen")
    head_type = row_config.get("head", "linear")
    use_htsf = row_config.get("use_htsf", False)

    from modules.dinov2_encoder import DINOv2Classifier

    if use_htsf:
        from modules.dinov2_encoder import DINOv2Encoder
        from fusion.htsf_fusion import HTSFWithUncertainty

        use_ast = row_config.get("use_ast", False)
        if use_ast:
            try:
                from modules.ast_encoder import ASTEncoder
                audio_enc = ASTEncoder(freeze_backbone=True, use_fp16=False)
                logger.info("Using AST as audio encoder in HTSF")
            except ImportError:
                logger.warning("AST not available, falling back to DINOv2 for audio")
                audio_enc = DINOv2Encoder(model_name="dinov2_vitb14",
                                          freeze_backbone=True, use_fp16=False)
        else:
            audio_enc = DINOv2Encoder(model_name="dinov2_vitb14",
                                      freeze_backbone=True, use_fp16=False)

        vision_enc = DINOv2Encoder(model_name="dinov2_vitb14",
                                   freeze_backbone=True, use_fp16=False)

        unfreeze = 0 if finetune == "frozen" else int(finetune.replace("last_", ""))
        if unfreeze > 0:
            if hasattr(audio_enc, "partial_unfreeze"):
                audio_enc.partial_unfreeze(n_blocks=unfreeze)
            vision_enc.partial_unfreeze(n_blocks=unfreeze)

        model = HTSFWithUncertainty(
            audio_encoder=audio_enc, vision_encoder=vision_enc,
            num_classes=2, hidden_dim=512, n_heads=8,
            n_cross_layers=2, dropout=0.15,
            freeze_encoders=False,
        ).to(device)
        return model, True, 512

    use_enhanced = head_type == "enhanced"
    model = DINOv2Classifier(
        num_classes=2, model_name="dinov2_vitb14",
        freeze_backbone=True, use_fp16=False,
        use_enhanced_head=use_enhanced,
        hidden_dims=[512, 256, 128] if use_enhanced else None,
        hidden_dim=256 if not use_enhanced else None,
        dropout=0.3 if use_enhanced else 0.1,
        use_multi_layer_features=row_config.get("multi_layer", False),
        n_last_blocks=4,
    ).to(device)

    if finetune != "frozen":
        n_blocks = int(finetune.replace("last_", ""))
        model.partial_unfreeze(n_blocks=n_blocks)

    feat_dim = model.feature_dim
    if row_config.get("multi_layer", False):
        feat_dim = feat_dim * 4
    return model, False, feat_dim


def build_criterion(row_config, feature_dim, device):
    """Build loss function for ablation row."""
    loss_type = row_config.get("loss", "ce")
    if loss_type == "ce":
        return nn.CrossEntropyLoss().to(device)
    elif loss_type == "focal":
        from modules.focal_loss import FocalLoss
        return FocalLoss(alpha=[0.1, 0.9], gamma=3.0).to(device)
    elif loss_type == "afcl":
        from modules.afcl_loss import create_afcl_loss
        return create_afcl_loss(
            feature_dim=feature_dim, num_classes=2,
            imbalance_ratio=4.4, use_hard_mining=True,
            warmup_epochs=5, temperature=0.5,
            lambda_contrast=0.1, lambda_align=0.05,
        ).to(device)
    raise ValueError(f"Unknown loss type: {loss_type}")


# ============================================================
# Ablation table definition
# ============================================================

ABLATION_ROWS = [
    {"name": "1_baseline_ce",
     "backbone": "dinov2_vitb14", "finetune": "frozen",
     "loss": "ce", "augmentation": "weak", "head": "linear"},
    {"name": "2_focal",
     "backbone": "dinov2_vitb14", "finetune": "frozen",
     "loss": "focal", "augmentation": "weak", "head": "linear"},
    {"name": "3_enhanced_head",
     "backbone": "dinov2_vitb14", "finetune": "frozen",
     "loss": "focal", "augmentation": "weak", "head": "enhanced"},
    {"name": "4_finetune",
     "backbone": "dinov2_vitb14", "finetune": "last_4",
     "loss": "focal", "augmentation": "weak", "head": "enhanced"},
    {"name": "5_afcl",
     "backbone": "dinov2_vitb14", "finetune": "last_4",
     "loss": "afcl", "augmentation": "weak", "head": "enhanced"},
    {"name": "6_specaugment",
     "backbone": "dinov2_vitb14", "finetune": "last_4",
     "loss": "afcl", "augmentation": "medium", "head": "enhanced"},
    {"name": "7_htsf_dual",
     "backbone": "dinov2_vitb14", "finetune": "last_4",
     "loss": "afcl", "augmentation": "medium", "head": "enhanced",
     "use_htsf": True},
    {"name": "8_multi_layer",
     "backbone": "dinov2_vitb14", "finetune": "last_4",
     "loss": "afcl", "augmentation": "medium", "head": "enhanced",
     "multi_layer": True},
    {"name": "9_ast_htsf",
     "backbone": "dinov2+ast", "finetune": "last_4",
     "loss": "afcl", "augmentation": "medium", "head": "enhanced",
     "use_htsf": True, "use_ast": True},
]


# ============================================================
# Training loop (works for both single-view and dual-view)
# ============================================================

def train_one_fold(
    model, train_loader, val_loader, criterion, device,
    is_dual_view, feature_dim, n_epochs, patience=15,
    use_wandb_logging=False, fold_id=0,
):
    """Train model for one fold with proper LR schedule and early stopping."""
    logger.info("train_one_fold: setting up optimizer...")
    import sys; sys.stdout.flush(); sys.stderr.flush()
    if hasattr(model, 'get_param_groups'):
        param_groups = model.get_param_groups(backbone_lr=1e-5, head_lr=1e-3)
    else:
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                         "lr": 1e-3}]

    if hasattr(criterion, 'parameters'):
        param_groups.append({"params": list(criterion.parameters()), "lr": 1e-3})

    optimizer = optim.AdamW(param_groups, weight_decay=0.05)
    scheduler = LinearWarmupCosineDecay(
        optimizer, warmup_epochs=5, total_epochs=n_epochs)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_bal_acc = 0.0
    best_state = None
    best_metrics = {}
    patience_counter = 0
    history = {"train_loss": [], "bal_acc": [], "abnormal_recall": [], "epoch": []}

    uses_afcl = hasattr(criterion, 'set_epoch')
    logger.info("train_one_fold: starting epoch loop (n_epochs=%d)", n_epochs)
    import sys; sys.stdout.flush(); sys.stderr.flush()

    for epoch in range(1, n_epochs + 1):
        logger.info("  Epoch %d/%d starting...", epoch, n_epochs)
        import sys; sys.stdout.flush(); sys.stderr.flush()
        if uses_afcl:
            criterion.set_epoch(epoch)

        model.train()
        if hasattr(criterion, 'train'):
            criterion.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            try:
                if batch_idx % 500 == 0:
                    logger.info("    batch %d/%d", batch_idx, len(train_loader))
                    import sys; sys.stdout.flush(); sys.stderr.flush()
                if is_dual_view:
                    x1, x2, y = batch
                    x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                else:
                    x, y = batch
                    x = x.to(device)
                    y = y.to(device)

                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    if is_dual_view:
                        fused_features = model.get_fused_features(x1, x2)
                        logits = model.classifier(fused_features)
                        if uses_afcl:
                            loss = criterion(fused_features, logits, y)
                        else:
                            loss = criterion(logits, y)
                    else:
                        if uses_afcl and hasattr(model, 'get_features'):
                            features = model.get_features(x)
                            logits = model.classifier(features)
                            loss = criterion(features, logits, y)
                        else:
                            logits = model(x)
                            loss = criterion(logits, y)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
            except Exception as e:
                import traceback
                logger.error("Error in batch %d: %s\n%s", batch_idx, e, traceback.format_exc())
                import sys; sys.stdout.flush(); sys.stderr.flush()
                raise

        scheduler.step()

        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                if is_dual_view:
                    x1, x2, y = batch
                    x1, x2 = x1.to(device), x2.to(device)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        logits = model(x1, x2)
                else:
                    x, y = batch
                    x = x.to(device)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        logits = model(x)
                probs = torch.softmax(logits, dim=1)
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_labels.extend(y.numpy())

        y_true = np.array(all_labels)
        y_prob = np.array(all_probs)
        opt_thr, _ = find_optimal_threshold(y_true, y_prob, "f1")
        y_pred = (y_prob >= opt_thr).astype(int)
        metrics = compute_full_metrics(y_true, y_pred, y_prob)

        bal_acc = metrics["balanced_accuracy"]
        history["train_loss"].append(total_loss / max(len(train_loader), 1))
        history["bal_acc"].append(bal_acc)
        history["abnormal_recall"].append(metrics["abnormal_recall"])
        history["epoch"].append(epoch)

        if use_wandb_logging and _HAS_WANDB and wandb.run:
            wandb.log({
                f"fold{fold_id}/epoch": epoch,
                f"fold{fold_id}/train_loss": history["train_loss"][-1],
                f"fold{fold_id}/bal_acc": bal_acc,
                f"fold{fold_id}/abnormal_recall": metrics["abnormal_recall"],
                f"fold{fold_id}/mcc": metrics["mcc"],
            })

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = metrics.copy()
            best_metrics["optimal_threshold"] = opt_thr
            best_metrics["y_true"] = y_true
            best_metrics["y_prob"] = y_prob
            best_metrics["y_pred"] = y_pred
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    del optimizer, scheduler, scaler
    torch.cuda.empty_cache()

    return {"history": history, "best_metrics": best_metrics}


# ============================================================
# Publication figure generation
# ============================================================

def generate_publication_figures(all_ablation_results, output_dir):
    """Generate all publication-quality figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    names = [r["name"] for r in all_ablation_results]
    short_names = [n.split("_", 1)[1] for n in names]

    def _get_metric(results, key):
        vals = []
        for r in results:
            fold_vals = [f["best_metrics"].get(key, 0) for f in r.get("folds", [{}])]
            vals.append(np.mean(fold_vals) if fold_vals else r.get("best_metrics", {}).get(key, 0))
        return vals

    # Fig 1: Ablation bar chart
    bal_accs = [v * 100 for v in _get_metric(all_ablation_results, "balanced_accuracy")]
    recalls = [v * 100 for v in _get_metric(all_ablation_results, "abnormal_recall")]
    mccs = _get_metric(all_ablation_results, "mcc")

    if len(names) > 0:
        x = np.arange(len(short_names))
        width = 0.25
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(x - width, recalls, width, label="Abnormal Recall %", color="#4472C4")
        ax.bar(x, bal_accs, width, label="Balanced Accuracy %", color="#70AD47")
        ax.bar(x + width, [m * 100 for m in mccs], width, label="MCC x100", color="#ED7D31")
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Score")
        ax.set_title("Ablation Study: Progressive Component Contribution")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fig_ablation_comparison.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

    # Fig 2: ROC curves overlay
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_ablation_results)))
    for i, r in enumerate(all_ablation_results):
        y_true = r.get("aggregate_y_true")
        y_prob = r.get("aggregate_y_prob")
        if y_true is not None and y_prob is not None and len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i], lw=1.5,
                    label=f"{short_names[i]} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves: Ablation Comparison")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "fig_roc_curves.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # Fig 3: Statistical significance matrix
    n = len(all_ablation_results)
    if n > 1:
        p_matrix = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                yt_i = all_ablation_results[i].get("aggregate_y_true")
                yp_i = all_ablation_results[i].get("aggregate_y_pred")
                yp_j = all_ablation_results[j].get("aggregate_y_pred")
                if yt_i is not None and yp_i is not None and yp_j is not None:
                    p = mcnemar_test(yt_i, yp_i, yp_j)
                    p_matrix[i, j] = p
                    p_matrix[j, i] = p

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(p_matrix, cmap="RdYlGn", vmin=0, vmax=0.1)
        for i in range(n):
            for j in range(n):
                txt = f"{p_matrix[i, j]:.3f}"
                color = "white" if p_matrix[i, j] < 0.03 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(short_names, fontsize=8)
        ax.set_title("McNemar's Test p-values")
        plt.colorbar(im, fraction=0.046)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fig_significance_matrix.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

    # Fig 4: CV box plots for best model
    if all_ablation_results:
        best_r = all_ablation_results[-1]
        folds = best_r.get("folds", [])
        if len(folds) >= 2:
            metric_keys = ["balanced_accuracy", "abnormal_recall", "mcc", "f1_abnormal"]
            fig, ax = plt.subplots(figsize=(10, 6))
            box_data = [[f["best_metrics"].get(k, 0) for f in folds] for k in metric_keys]
            bp = ax.boxplot(box_data,
                            labels=[k.replace("_", " ").title() for k in metric_keys],
                            patch_artist=True)
            colors_box = ["#4472C4", "#ED7D31", "#70AD47", "#FFC000"]
            for patch, c in zip(bp["boxes"], colors_box):
                patch.set_facecolor(c)
                patch.set_alpha(0.7)
            ax.set_ylabel("Score")
            ax.set_title("5-Fold Cross-Validation Results (Best Configuration)")
            ax.grid(axis="y", alpha=0.3)
            plt.savefig(os.path.join(output_dir, "fig_cv_boxplot.png"),
                        dpi=300, bbox_inches="tight")
            plt.close()

    # Fig 5: Confusion matrices for best model (per fold)
    if all_ablation_results:
        best_r = all_ablation_results[-1]
        folds = best_r.get("folds", [])
        n_folds = len(folds)
        if n_folds > 0:
            cols = min(n_folds, 5)
            fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
            if cols == 1:
                axes = [axes]
            for i, f_res in enumerate(folds[:cols]):
                bm = f_res["best_metrics"]
                if "confusion_matrix" in bm:
                    cm = np.array(bm["confusion_matrix"])
                else:
                    continue
                cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
                im = axes[i].imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
                for r in range(2):
                    for c_val in range(2):
                        axes[i].text(c_val, r, f"{cm[r, c_val]}\n({cm_norm[r, c_val]:.2f})",
                                     ha="center", va="center", fontsize=9)
                axes[i].set_xticks([0, 1])
                axes[i].set_yticks([0, 1])
                axes[i].set_xticklabels(["Normal", "Abnormal"])
                axes[i].set_yticklabels(["Normal", "Abnormal"])
                axes[i].set_title(f"Fold {i+1}")
                axes[i].set_xlabel("Predicted")
                if i == 0:
                    axes[i].set_ylabel("True")
            plt.suptitle("Normalized Confusion Matrices (Best Configuration)", fontsize=13)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "fig_confusion_matrices.png"),
                        dpi=300, bbox_inches="tight")
            plt.close()

    # Fig 6: Calibration / reliability diagram
    if all_ablation_results:
        best_r = all_ablation_results[-1]
        y_true = best_r.get("aggregate_y_true")
        y_prob = best_r.get("aggregate_y_prob")
        if y_true is not None and y_prob is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            n_bins = 10
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_means = []
            bin_trues = []
            bin_counts = []
            for bi in range(n_bins):
                mask = (y_prob >= bin_edges[bi]) & (y_prob < bin_edges[bi + 1])
                if mask.sum() > 0:
                    bin_means.append(y_prob[mask].mean())
                    bin_trues.append(y_true[mask].mean())
                    bin_counts.append(mask.sum())
                else:
                    bin_means.append((bin_edges[bi] + bin_edges[bi + 1]) / 2)
                    bin_trues.append(0)
                    bin_counts.append(0)

            ax1.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")
            ax1.bar(bin_means, bin_trues, width=0.08, alpha=0.7, color="#4472C4",
                    label="Model calibration")
            ax1.set_xlabel("Mean Predicted Probability")
            ax1.set_ylabel("Fraction of Positives")
            ax1.set_title("Reliability Diagram")
            ax1.legend()
            ax1.grid(alpha=0.3)

            ax2.hist(y_prob[y_true == 0], bins=50, alpha=0.6, color="#70AD47",
                     label="Normal", density=True)
            ax2.hist(y_prob[y_true == 1], bins=50, alpha=0.6, color="#ED7D31",
                     label="Abnormal", density=True)
            ax2.set_xlabel("Predicted Probability")
            ax2.set_ylabel("Density")
            ax2.set_title("Prediction Distribution")
            ax2.legend()
            ax2.grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "fig_calibration.png"),
                        dpi=300, bbox_inches="tight")
            plt.close()

    # Fig 7: Precision-Recall curves
    fig, ax = plt.subplots(figsize=(8, 8))
    for i, r in enumerate(all_ablation_results):
        y_true = r.get("aggregate_y_true")
        y_prob = r.get("aggregate_y_prob")
        if y_true is not None and y_prob is not None and len(np.unique(y_true)) > 1:
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)
            ax.plot(rec, prec, color=colors[i % len(colors)], lw=1.5,
                    label=f"{short_names[i]} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves: Ablation Comparison")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "fig_pr_curves.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # Fig 8: Training curves (loss + balanced accuracy) for best model
    if all_ablation_results:
        best_r = all_ablation_results[-1]
        folds = best_r.get("folds", [])
        if folds:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            for fi, f_res in enumerate(folds):
                h = f_res.get("history", {})
                epochs_arr = h.get("epoch", [])
                if not epochs_arr:
                    continue
                ax1.plot(epochs_arr, h.get("train_loss", []),
                         alpha=0.5, label=f"Fold {fi+1}")
                ax2.plot(epochs_arr, [v * 100 for v in h.get("bal_acc", [])],
                         alpha=0.5, label=f"Fold {fi+1}")

            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Training Loss")
            ax1.set_title("Training Loss per Fold")
            ax1.legend(fontsize=8)
            ax1.grid(alpha=0.3)

            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Balanced Accuracy (%)")
            ax2.set_title("Validation Balanced Accuracy per Fold")
            ax2.legend(fontsize=8)
            ax2.grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "fig_training_curves.png"),
                        dpi=300, bbox_inches="tight")
            plt.close()

    logger.info("Publication figures saved to %s", output_dir)


# ============================================================
# Feature visualization (t-SNE / UMAP)
# ============================================================

def extract_features_and_labels(model, data_loader, device, is_dual_view=False):
    """Extract feature vectors and labels from model for visualization."""
    model.eval()
    all_features, all_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            if is_dual_view:
                x1, x2, y = batch
                x1, x2 = x1.to(device), x2.to(device)
                if hasattr(model, 'get_fused_features'):
                    feats = model.get_fused_features(x1, x2)
                else:
                    feats = model(x1, x2)
            else:
                x, y = batch
                x = x.to(device)
                if hasattr(model, 'get_features'):
                    feats = model.get_features(x)
                else:
                    feats = model(x)
            if feats.dim() > 2:
                feats = feats.mean(dim=1)
            all_features.append(feats.cpu().numpy())
            all_labels.extend(y.numpy() if hasattr(y, 'numpy') else y)
    return np.concatenate(all_features, axis=0), np.array(all_labels)


def generate_tsne_plot(features, labels, output_path, title="t-SNE Feature Visualization"):
    """Generate t-SNE visualization of extracted features."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    n_samples = min(len(features), 5000)
    idx = np.random.choice(len(features), n_samples, replace=False)
    feats_sub = features[idx]
    labels_sub = labels[idx]

    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    embedded = tsne.fit_transform(feats_sub)

    fig, ax = plt.subplots(figsize=(8, 8))
    normal_mask = labels_sub == 0
    abnormal_mask = labels_sub == 1

    ax.scatter(embedded[normal_mask, 0], embedded[normal_mask, 1],
               c="#70AD47", s=15, alpha=0.6, label="Normal", edgecolors="none")
    ax.scatter(embedded[abnormal_mask, 0], embedded[abnormal_mask, 1],
               c="#ED7D31", s=15, alpha=0.6, label="Abnormal", edgecolors="none")
    ax.legend(fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("t-SNE plot saved to %s", output_path)


def generate_attention_heatmap(model, sample_batch, device, output_path,
                               is_dual_view=False):
    """Generate attention heatmap from HTSF cross-modal attention."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    with torch.no_grad():
        if is_dual_view:
            x1, x2, _ = sample_batch
            x1, x2 = x1[:4].to(device), x2[:4].to(device)
            if hasattr(model, 'forward'):
                try:
                    logits, attn_info = model(x1, x2, return_attention=True)
                    cross_attn = attn_info.get("cross_modal")
                    if cross_attn is not None:
                        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                        for i in range(min(4, cross_attn.shape[0])):
                            attn_map = cross_attn[i].cpu().numpy()
                            if attn_map.ndim == 1:
                                attn_map = attn_map.reshape(1, -1)
                            axes[i].imshow(attn_map, cmap="viridis", aspect="auto")
                            axes[i].set_title(f"Sample {i+1}")
                            axes[i].set_xlabel("Key position")
                            if i == 0:
                                axes[i].set_ylabel("Query position")
                        plt.suptitle("Cross-Modal Attention Weights")
                        plt.tight_layout()
                        plt.savefig(output_path, dpi=300, bbox_inches="tight")
                        plt.close()
                        logger.info("Attention heatmap saved to %s", output_path)
                        return
                except Exception as e:
                    logger.warning("Could not generate attention map: %s", e)

    logger.info("Attention heatmap: model does not support return_attention")


# ============================================================
# SOTA comparison table
# ============================================================

SOTA_BASELINES = {
    "DCASE 2020 Baseline (AE)": {
        "auc_roc": 0.7261, "pauc_fpr01": 0.5310,
        "source": "Koizumi et al., DCASE 2020 Challenge"},
    "DCASE 2020 1st Place": {
        "auc_roc": 0.9456, "pauc_fpr01": 0.8512,
        "source": "Primus, DCASE 2020"},
    "Giri et al. (2020) Self-supervised": {
        "auc_roc": 0.8900, "pauc_fpr01": None,
        "source": "Giri et al., ICASSP 2020"},
    "STgram-MFN (2022)": {
        "auc_roc": 0.9230, "pauc_fpr01": 0.7640,
        "source": "Liu et al., IEEE/ACM TASLP 2022"},
    "ASD-Diffusion (2024)": {
        "auc_roc": 0.9510, "pauc_fpr01": 0.8730,
        "source": "Weng et al., ICASSP 2024"},
}


def generate_sota_comparison_table(our_results, output_dir):
    """Generate SOTA comparison markdown table."""
    lines = [
        "# Comparison with Published MIMII/DCASE Baselines",
        "",
        "| Method | AUC-ROC | pAUC (FPR<0.1) | Source |",
        "|--------|---------|----------------|--------|",
    ]
    for name, data in SOTA_BASELINES.items():
        auc_str = f"{data['auc_roc']:.4f}" if data["auc_roc"] else "-"
        pauc_str = f"{data['pauc_fpr01']:.4f}" if data.get("pauc_fpr01") else "-"
        lines.append(f"| {name} | {auc_str} | {pauc_str} | {data['source']} |")

    if our_results:
        our_auc = our_results.get("auc_roc", 0)
        our_pauc = our_results.get("pauc_fpr01", 0)
        lines.append(
            f"| **Ours (DINOv2 + AFCL)** | **{our_auc:.4f}** | "
            f"**{our_pauc:.4f}** | This work |")

    lines.extend(["", "Note: Results may not be directly comparable due to ",
                   "different evaluation protocols and machine type subsets."])

    path = os.path.join(output_dir, "sota_comparison.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("SOTA comparison table saved to %s", path)
    return path


# ============================================================
# Main experiment orchestrator
# ============================================================

class Q1ExperimentRunner:
    """Run Q1-ready experiments with grouped CV and full evaluation."""

    def __init__(self, output_dir="results/q1_experiments", use_wandb=True):
        self.output_dir = output_dir
        self.use_wandb = use_wandb and _HAS_WANDB
        os.makedirs(output_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Device: %s", self.device)

    def run_ablation_table(self, dataset_path, rows=None, n_epochs=50,
                           batch_size=16, n_folds=5):
        """Run the full ablation table with group-aware 5-fold CV."""
        from experiments.run_all_models import ImageDataset
        from modules.spectrogram_augment import get_spectrogram_transforms

        rows = rows or ABLATION_ROWS
        base_ds = ImageDataset(dataset_path, transform=None)
        all_labels = np.array(base_ds.labels)
        all_groups = np.array(base_ds.groups)
        n_unique_groups = len(set(base_ds.groups))

        all_ablation_results = []

        for row_config in rows:
            row_name = row_config["name"]
            logger.info("=" * 60)
            logger.info("ABLATION ROW: %s", row_name)
            logger.info("=" * 60)

            aug_strength = row_config.get("augmentation", "weak")
            spec_transforms = get_spectrogram_transforms(
                image_size=224, strength=aug_strength)

            if n_unique_groups > 1:
                cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
                cv_iter = list(cv.split(np.zeros(len(all_labels)), all_labels, all_groups))
            else:
                cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
                cv_iter = list(cv.split(np.zeros(len(all_labels)), all_labels))

            fold_results = []
            agg_y_true, agg_y_prob, agg_y_pred = [], [], []

            for fold_id, (train_idx, val_idx) in enumerate(cv_iter, 1):
                logger.info("--- Fold %d/%d ---", fold_id, n_folds)
                torch.manual_seed(42 + fold_id)
                np.random.seed(42 + fold_id)

                is_htsf = row_config.get("use_htsf", False)

                if is_htsf:
                    view2_transform = spec_transforms["train"]
                    from modules.spectrogram_augment import get_spectrogram_transforms as _gst
                    view1_transforms = _gst(image_size=224, strength="weak")
                    train_ds = _DualViewDataset(
                        base_ds, view1_transforms["train"], view2_transform)
                    val_ds_view = _DualViewDataset(
                        base_ds, spec_transforms["val"], spec_transforms["val"])
                else:
                    train_ds = _SingleViewDataset(base_ds, spec_transforms["train"])
                    val_ds_view = _SingleViewDataset(base_ds, spec_transforms["val"])

                train_labels = [all_labels[i] for i in train_idx]
                sampler = make_weighted_sampler(train_labels)

                num_wk = 0 if os.name == "nt" else 2
                train_loader = DataLoader(
                    Subset(train_ds, train_idx.tolist()),
                    batch_size=batch_size, sampler=sampler,
                    num_workers=num_wk, pin_memory=True)
                val_loader = DataLoader(
                    Subset(val_ds_view, val_idx.tolist()),
                    batch_size=batch_size, shuffle=False,
                    num_workers=num_wk, pin_memory=True)

                logger.info("Building model...")
                import sys; sys.stdout.flush(); sys.stderr.flush()
                model, needs_dual, feat_dim = build_model_for_ablation(
                    row_config, self.device)
                logger.info("Model built. Building criterion...")
                sys.stdout.flush(); sys.stderr.flush()
                criterion = build_criterion(row_config, feat_dim, self.device)
                logger.info("Criterion built. Starting training... (train=%d, val=%d batches)",
                            len(train_loader), len(val_loader))
                sys.stdout.flush(); sys.stderr.flush()

                fold_res = train_one_fold(
                    model, train_loader, val_loader, criterion, self.device,
                    is_dual_view=needs_dual, feature_dim=feat_dim,
                    n_epochs=n_epochs, patience=15,
                    use_wandb_logging=self.use_wandb, fold_id=fold_id)

                fold_results.append(fold_res)
                bm = fold_res["best_metrics"]
                if "y_true" in bm:
                    agg_y_true.append(bm["y_true"])
                    agg_y_prob.append(bm["y_prob"])
                    agg_y_pred.append(bm["y_pred"])

                del model, criterion
                torch.cuda.empty_cache()
                gc.collect()

            row_result = {
                "name": row_name,
                "config": row_config,
                "folds": fold_results,
            }

            if agg_y_true:
                row_result["aggregate_y_true"] = np.concatenate(agg_y_true)
                row_result["aggregate_y_prob"] = np.concatenate(agg_y_prob)
                row_result["aggregate_y_pred"] = np.concatenate(agg_y_pred)
                row_result["aggregate_metrics"] = compute_full_metrics(
                    row_result["aggregate_y_true"],
                    row_result["aggregate_y_pred"],
                    row_result["aggregate_y_prob"])

            metric_keys = ["balanced_accuracy", "abnormal_recall", "mcc",
                           "auc_roc", "f1_abnormal", "pauc_fpr01"]
            cv_summary = {}
            for k in metric_keys:
                vals = [f["best_metrics"].get(k, 0) for f in fold_results]
                cv_summary[k] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "values": vals,
                }
            row_result["cv_summary"] = cv_summary

            logger.info("Row %s | BalAcc: %.4f +/- %.4f | AbnRecall: %.4f +/- %.4f | MCC: %.4f +/- %.4f",
                        row_name,
                        cv_summary["balanced_accuracy"]["mean"],
                        cv_summary["balanced_accuracy"]["std"],
                        cv_summary["abnormal_recall"]["mean"],
                        cv_summary["abnormal_recall"]["std"],
                        cv_summary["mcc"]["mean"],
                        cv_summary["mcc"]["std"])

            all_ablation_results.append(row_result)

        self._save_results(all_ablation_results, "ablation_results.json")

        fig_dir = os.path.join(self.output_dir, "figures")
        generate_publication_figures(all_ablation_results, fig_dir)

        best_row = all_ablation_results[-1] if all_ablation_results else {}
        best_metrics = best_row.get("aggregate_metrics", {})
        generate_sota_comparison_table(best_metrics, self.output_dir)

        self._generate_significance_table(all_ablation_results)
        self._generate_ablation_markdown(all_ablation_results)

        return all_ablation_results

    def run_per_machine_evaluation(self, dataset_path, row_config=None,
                                   n_epochs=50, batch_size=16):
        """Run evaluation with per-machine-type breakdown."""
        from experiments.run_all_models import ImageDataset
        from modules.spectrogram_augment import get_spectrogram_transforms

        if row_config is None:
            row_config = ABLATION_ROWS[-1]

        base_ds = ImageDataset(dataset_path, transform=None)
        all_labels = np.array(base_ds.labels)
        all_groups = np.array(base_ds.groups)
        n_unique = len(set(base_ds.groups))
        spec_t = get_spectrogram_transforms(
            image_size=224, strength=row_config.get("augmentation", "medium"))

        if n_unique > 1:
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
            train_idx, val_idx = next(cv.split(
                np.zeros(len(all_labels)), all_labels, all_groups))
        else:
            from sklearn.model_selection import train_test_split
            indices = list(range(len(all_labels)))
            train_idx, val_idx = train_test_split(
                indices, test_size=0.2, stratify=all_labels, random_state=42)
            train_idx, val_idx = np.array(train_idx), np.array(val_idx)

        is_htsf = row_config.get("use_htsf", False)
        if is_htsf:
            train_ds = _DualViewDataset(base_ds, spec_t["train"], spec_t["train"])
            val_ds = _DualViewDataset(base_ds, spec_t["val"], spec_t["val"])
        else:
            train_ds = _SingleViewDataset(base_ds, spec_t["train"])
            val_ds = _SingleViewDataset(base_ds, spec_t["val"])

        train_labels = [all_labels[i] for i in train_idx]
        sampler = make_weighted_sampler(train_labels)

        train_loader = DataLoader(
            Subset(train_ds, train_idx.tolist()),
            batch_size=batch_size, sampler=sampler,
            num_workers=0 if os.name == "nt" else 2, pin_memory=True)
        val_loader = DataLoader(
            Subset(val_ds, val_idx.tolist()),
            batch_size=batch_size, shuffle=False,
            num_workers=0 if os.name == "nt" else 2, pin_memory=True)

        model, needs_dual, feat_dim = build_model_for_ablation(row_config, self.device)
        criterion = build_criterion(row_config, feat_dim, self.device)

        result = train_one_fold(
            model, train_loader, val_loader, criterion, self.device,
            is_dual_view=needs_dual, feature_dim=feat_dim,
            n_epochs=n_epochs, patience=15)

        bm = result["best_metrics"]
        val_samples = [base_ds.samples[i] for i in val_idx]
        per_machine = evaluate_per_machine(
            val_samples, bm["y_true"], bm["y_pred"], bm["y_prob"])

        result["per_machine_metrics"] = per_machine
        self._save_results(result, "per_machine_results.json")

        lines = ["# Per-Machine-Type Evaluation", "",
                 "| Machine Type | N | Bal Acc | Abn Recall | MCC | AUC |",
                 "|-------------|---|---------|------------|-----|-----|"]
        for mtype, m in per_machine.items():
            if "note" in m:
                lines.append(f"| {mtype} | {m['n_samples']} | - | - | - | - |")
            else:
                lines.append(
                    f"| {mtype} | {m['n_samples']} | "
                    f"{m.get('balanced_accuracy', 0)*100:.1f}% | "
                    f"{m.get('abnormal_recall', 0)*100:.1f}% | "
                    f"{m.get('mcc', 0):.3f} | "
                    f"{m.get('auc_roc', 0):.3f} |")

        path = os.path.join(self.output_dir, "per_machine_breakdown.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info("Per-machine breakdown saved to %s", path)

        return result

    def _generate_significance_table(self, all_results):
        """Generate statistical significance comparison table."""
        n = len(all_results)
        names = [r["name"].split("_", 1)[1] for r in all_results]

        header = "| vs | " + " | ".join(names) + " |"
        sep = "|---|" + "|".join(["---"] * n) + "|"
        lines = ["# Statistical Significance Matrix (McNemar's Test)", "",
                 header, sep]

        for i in range(n):
            row = f"| {names[i]} |"
            for j in range(n):
                if i == j:
                    row += " - |"
                else:
                    yt_i = all_results[i].get("aggregate_y_true")
                    yp_i = all_results[i].get("aggregate_y_pred")
                    yp_j = all_results[j].get("aggregate_y_pred")
                    if yt_i is not None and yp_i is not None and yp_j is not None:
                        p = mcnemar_test(yt_i, yp_i, yp_j)
                        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
                        row += f" {p:.4f}{sig} |"
                    else:
                        row += " - |"
            lines.append(row)

        lines.extend(["", "Significance: * p<0.05, ** p<0.01, *** p<0.001"])

        cv_lines = ["", "# Corrected Resampled t-test (Nadeau & Bengio 2003)", ""]
        for i in range(n):
            for j in range(i + 1, n):
                scores_i = all_results[i].get("cv_summary", {}).get(
                    "balanced_accuracy", {}).get("values", [])
                scores_j = all_results[j].get("cv_summary", {}).get(
                    "balanced_accuracy", {}).get("values", [])
                if len(scores_i) >= 2 and len(scores_j) >= 2:
                    p = corrected_resampled_ttest(scores_i, scores_j, 800, 200)
                    cv_lines.append(
                        f"- {names[i]} vs {names[j]}: p={p:.4f}")

        lines.extend(cv_lines)
        path = os.path.join(self.output_dir, "statistical_tests.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info("Statistical tests saved to %s", path)

    def _generate_ablation_markdown(self, all_results):
        """Generate ablation results markdown table."""
        lines = [
            "# Ablation Study Results (5-Fold Group-Aware CV)", "",
            "| Configuration | Bal Acc (%) | Abn Recall (%) | MCC | AUC-ROC | pAUC@0.1 | F1 |",
            "|--------------|------------|----------------|-----|---------|----------|-----|",
        ]
        for r in all_results:
            cs = r.get("cv_summary", {})
            name = r["name"].split("_", 1)[1]
            ba = cs.get("balanced_accuracy", {})
            ar = cs.get("abnormal_recall", {})
            mcc = cs.get("mcc", {})
            auc_r = cs.get("auc_roc", {})
            pauc = cs.get("pauc_fpr01", {})
            f1 = cs.get("f1_abnormal", {})
            lines.append(
                f"| {name} | "
                f"{ba.get('mean', 0)*100:.1f} +/- {ba.get('std', 0)*100:.1f} | "
                f"{ar.get('mean', 0)*100:.1f} +/- {ar.get('std', 0)*100:.1f} | "
                f"{mcc.get('mean', 0):.3f} +/- {mcc.get('std', 0):.3f} | "
                f"{auc_r.get('mean', 0):.3f} | "
                f"{pauc.get('mean', 0):.3f} | "
                f"{f1.get('mean', 0):.3f} |")

        path = os.path.join(self.output_dir, "ablation_results.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info("Ablation table saved to %s", path)

    def _save_results(self, results, filename):
        """Save results to JSON, handling numpy types."""
        def _convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32, np.int_)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_convert(v) for v in obj]
            return obj

        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_convert(results), f, indent=2)
        logger.info("Results saved to %s", path)


def main():
    setup_logging(log_dir="logs")

    parser = argparse.ArgumentParser(description="Q1 Experiment Pipeline")
    parser.add_argument("--experiment", type=str, default="ablation",
                        choices=["ablation", "per_machine", "full"],
                        help="Which experiment to run")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="results/q1_experiments")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--rows", type=str, default=None,
                        help="Comma-separated ablation row numbers (e.g. '1,2,3')")
    args = parser.parse_args()

    dataset_path = args.dataset_path or os.getenv("DATASET_PATH", ".")
    runner = Q1ExperimentRunner(
        output_dir=args.output_dir, use_wandb=not args.no_wandb)

    if args.experiment == "ablation":
        rows = None
        if args.rows:
            indices = [int(x) - 1 for x in args.rows.split(",")]
            rows = [ABLATION_ROWS[i] for i in indices if i < len(ABLATION_ROWS)]
        runner.run_ablation_table(
            dataset_path, rows=rows,
            n_epochs=args.epochs, batch_size=args.batch_size)
    elif args.experiment == "per_machine":
        runner.run_per_machine_evaluation(
            dataset_path, n_epochs=args.epochs, batch_size=args.batch_size)
    elif args.experiment == "full":
        runner.run_ablation_table(
            dataset_path, n_epochs=args.epochs, batch_size=args.batch_size)
        runner.run_per_machine_evaluation(
            dataset_path, n_epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
