"""
Publication-ready training entry point for spectrogram-image fault classification.

Usage (from src/):
  python -m experiments.train_spectrogram_fault --dataset_path %DATASET_PATH% --model resnet50

See experiments/README_SPECTROGRAM_FAULT.md for options.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_fscore_support,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", ".env"))

from experiments.run_all_models import ImageDataset
from modules.spectrogram_fault_data import (
    get_fault_transforms,
    get_train_val_indices,
    get_train_val_indices_mat_stem_stratified,
    default_num_workers,
    resolve_image_dataset_root,
)
from modules.spectrogram_cnn_models import build_model, list_model_names
from modules.focal_loss import FocalLoss
from modules.evaluation import compute_classification_metrics
from utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


class LinearWarmupCosineLR:
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, min_lr: float = 1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        e = self.current_epoch
        for i, base in enumerate(self.base_lrs):
            if e <= self.warmup_epochs:
                lr = base * (e / max(1, self.warmup_epochs))
            else:
                t = (e - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
                lr = self.min_lr + (base - self.min_lr) * 0.5 * (1 + math.cos(math.pi * t))
            self.optimizer.param_groups[i]["lr"] = lr


def _class_weights_from_labels(train_labels: List[int], num_classes: int) -> torch.Tensor:
    counts = Counter(train_labels)
    n = len(train_labels)
    w = [n / (num_classes * counts.get(c, 1)) for c in range(num_classes)]
    return torch.tensor(w, dtype=torch.float32)


def _make_criterion(
    loss_name: str,
    train_labels: List[int],
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    if loss_name == "ce_weighted":
        w = _class_weights_from_labels(train_labels, num_classes).to(device)
        return nn.CrossEntropyLoss(weight=w)
    if loss_name == "focal":
        counts = Counter(train_labels)
        n = len(train_labels)
        alpha = [n / (num_classes * counts.get(c, 1)) for c in range(num_classes)]
        return FocalLoss(alpha=alpha, gamma=2.0).to(device)
    raise ValueError(f"Unknown loss: {loss_name}")


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    use_amp: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_y: List[int] = []
    all_pred: List[int] = []
    all_prob: List[np.ndarray] = []
    for x, y in loader:
        x = x.to(device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda" and use_amp):
            logits = model(x)
        prob = torch.softmax(logits.float(), dim=1).cpu().numpy()
        pred = logits.argmax(dim=1).cpu().numpy()
        all_y.extend(y.numpy().tolist())
        all_pred.extend(pred.tolist())
        all_prob.append(prob)
    probs = np.vstack(all_prob)
    return np.array(all_y), np.array(all_pred), probs


def save_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.figure(figsize=(7, 6))
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.4f}")
        plt.plot([0, 1], [0, 1], "k--")
    else:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_true, classes=np.arange(n_classes))
        for c in range(n_classes):
            if y_bin[:, c].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, c], y_prob[:, c])
            plt.plot(fpr, tpr, label=f"class {c} AUC={auc(fpr, tpr):.3f}")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_confusion_matrix_fig(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=[str(i) for i in labels])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_training_curves(history: Dict[str, List[float]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="Train loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, history["val_bal_acc"], label="Val balanced acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Balanced accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_metric_curves(history: Dict[str, List[float]], path: str) -> None:
    """Save validation precision/recall/F1 curves across epochs."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    epochs = range(1, len(history["val_precision"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["val_precision"], label="Val precision")
    plt.plot(epochs, history["val_recall"], label="Val recall")
    plt.plot(epochs, history["val_f1"], label="Val F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Precision / Recall / F1")
    plt.ylim(0.0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def _save_attention_samples(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    out_dir: str,
    use_amp: bool,
    max_save: int = 4,
) -> None:
    """Save spatial attention (cnn_se) or transformer attention maps for a few batches."""
    model.eval()
    saved = 0
    for x, y in val_loader:
        if saved >= max_save:
            break
        x = x.to(device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda" and use_amp):
            if hasattr(model, "get_last_spatial_attention"):
                _ = model(x[:1])
                attn = model.get_last_spatial_attention()
                if attn is not None:
                    a = attn[0].float().cpu().numpy()
                    plt.figure(figsize=(4, 3))
                    plt.imshow(a, cmap="magma")
                    plt.colorbar()
                    plt.title("Spatial attention")
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, f"spatial_attn_{saved}.png"), dpi=200)
                    plt.close()
                    saved += 1
            elif hasattr(model, "forward_with_attention"):
                logits, aw = model.forward_with_attention(x[:1])
                if aw is not None and aw.dim() == 3:
                    cls_attn = aw[0, 0, 1:].cpu().numpy()
                    side = int(np.sqrt(len(cls_attn)))
                    if side * side == len(cls_attn):
                        plt.figure(figsize=(4, 3))
                        plt.imshow(cls_attn.reshape(side, side), cmap="viridis")
                        plt.colorbar()
                        plt.title("CLS attention over spatial tokens")
                        plt.tight_layout()
                        plt.savefig(os.path.join(out_dir, f"tfm_attn_{saved}.png"), dpi=200)
                        plt.close()
                        saved += 1
        if saved >= max_save:
            break


def _save_grad_cam_samples(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    out_dir: str,
    max_save: int = 4,
) -> None:
    """Grad-CAM on CNN backbone (``modules.visualization.plot_grad_cam_cnn``)."""
    from modules.visualization import plot_grad_cam_cnn

    if not hasattr(model, "backbone"):
        return
    model.eval()
    saved = 0
    for x, _y in val_loader:
        if saved >= max_save:
            break
        x = x[:1].to(device).float()
        path = os.path.join(out_dir, f"grad_cam_{saved}.png")
        try:
            plot_grad_cam_cnn(model, x, path, title=f"Grad-CAM sample {saved}")
            saved += 1
        except Exception as e:
            logger.warning("Grad-CAM skipped: %s", e)
            break


_MAT_STEM_WIN_RE = re.compile(r"^(.+?)_w\d+\.(?:png|jpg|jpeg|bmp|webp)$", re.IGNORECASE)


def _infer_mat_stem_groups_inplace(ds: Any) -> int:
    """
    Set ``ds.groups`` from CWRU export-style basenames (``<mat_stem>_wNNNNNN.png``).
    Returns the number of unique group ids.
    """
    groups: List[str] = []
    for path in ds.samples:
        base = os.path.basename(path)
        m = _MAT_STEM_WIN_RE.match(base)
        groups.append(m.group(1) if m else "default")
    ds.groups = groups
    return len(set(groups))


def _apply_dataset_subdir_from_env(args: argparse.Namespace) -> None:
    if getattr(args, "dataset_subdir", None) is not None:
        return
    ev = os.getenv("DATASET_SUBDIR")
    if ev and str(ev).strip():
        args.dataset_subdir = str(ev).strip()


def train_one_run(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    _apply_dataset_subdir_from_env(args)
    _sub = getattr(args, "dataset_subdir", None)
    dataset_subdir = None
    if _sub is not None:
        dataset_subdir = str(_sub).strip() or None
    data_root = resolve_image_dataset_root(args.dataset_path, dataset_subdir)
    logger.info("ImageDataset root: %s", data_root)

    tfm = get_fault_transforms(image_size=args.image_size, augment=args.augment)
    train_ds = ImageDataset(data_root, transform=tfm["train"])
    val_ds = ImageDataset(data_root, transform=tfm["val"])
    if len(train_ds) == 0:
        raise FileNotFoundError(f"No images under {data_root}")

    if getattr(args, "infer_mat_stem_groups", False):
        n_u_tr = _infer_mat_stem_groups_inplace(train_ds)
        _infer_mat_stem_groups_inplace(val_ds)
        logger.info(
            "infer_mat_stem_groups: %d unique mat/file stems (split by stem within each class; "
            "avoids val folds that are single-class).",
            n_u_tr,
        )
        if n_u_tr <= 1:
            logger.warning(
                "infer_mat_stem_groups produced <=1 group; filenames may not match ``<stem>_w######.png``."
            )
    elif args.group_split and len(set(train_ds.groups)) <= 1 and len(train_ds) < 50000:
        logger.warning(
            "All samples share one group id (e.g. flat normal/abnormal folders). Train/val splits can "
            "mix windows from the same recording into train and val, which often inflates val metrics. "
            "For CWRU PNGs from export_cwru_spectrogram_images, re-run with --infer_mat_stem_groups."
        )

    labels_list = train_ds.labels
    num_classes = max(labels_list) + 1
    stem_meta: Optional[Dict[str, Any]] = None
    if getattr(args, "infer_mat_stem_groups", False):
        train_idx, val_idx, stem_meta = get_train_val_indices_mat_stem_stratified(
            labels_list,
            train_ds.groups,
            val_split=args.val_split,
            random_state=args.seed,
        )
        slim = {k: stem_meta[k] for k in stem_meta if k not in ("stem_ids",)}
        logger.info(
            "Mat-stem split: train_windows=%d val_windows=%d | val_stems=%s | details=%s",
            len(train_idx),
            len(val_idx),
            stem_meta.get("val_stems"),
            slim,
        )
        if stem_meta.get("warning"):
            logger.warning("%s", stem_meta["warning"])
    else:
        try:
            train_idx, val_idx = get_train_val_indices(
                labels_list,
                train_ds.groups,
                val_split=args.val_split,
                random_state=args.seed,
                use_group_split=args.group_split,
            )
        except ValueError as e:
            if not args.group_split:
                raise
            logger.warning("Group-aware split failed (%s); using stratified sample split instead.", e)
            train_idx, val_idx = get_train_val_indices(
                labels_list,
                train_ds.groups,
                val_split=args.val_split,
                random_state=args.seed,
                use_group_split=False,
            )
    train_labels = [labels_list[i] for i in train_idx]

    nw = args.num_workers if args.num_workers >= 0 else default_num_workers()
    train_loader = DataLoader(
        Subset(train_ds, train_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        Subset(val_ds, val_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        args.model,
        num_classes=num_classes,
        backbone=args.backbone,
        pretrained=args.pretrained,
    ).to(device)

    if getattr(args, "init_checkpoint", None):
        ck = os.path.normpath(os.path.expandvars(os.path.expanduser(str(args.init_checkpoint))))
        if not os.path.isfile(ck):
            raise FileNotFoundError(f"--init_checkpoint not found: {ck}")
        try:
            warm = torch.load(ck, map_location=device, weights_only=False)
        except TypeError:
            warm = torch.load(ck, map_location=device)
        sd = warm.get("model_state_dict", warm)
        model.load_state_dict(sd, strict=True)
        logger.info("Warm-started model weights from %s", ck)

    criterion = _make_criterion(args.loss, train_labels, num_classes, device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = LinearWarmupCosineLR(
        optimizer, warmup_epochs=args.warmup_epochs, total_epochs=args.epochs
    )

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_bal_acc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
    }
    best_bal = -1.0
    best_state = None
    patience_cnt = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        scheduler.step()
        model.train()
        total_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            n_batches += 1

        train_loss_epoch = total_loss / max(n_batches, 1)
        y_t, y_p, y_pr = evaluate_model(model, val_loader, device, num_classes, use_amp)
        bal_acc = balanced_accuracy_score(y_t, y_p)
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            y_t,
            y_p,
            average="macro",
            zero_division=0,
            labels=list(range(num_classes)),
        )
        history["train_loss"].append(train_loss_epoch)
        history["val_bal_acc"].append(float(bal_acc))
        history["val_precision"].append(float(prec_macro))
        history["val_recall"].append(float(rec_macro))
        history["val_f1"].append(float(f1_macro))

        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_bal_acc=%.4f | val_precision=%.4f | val_recall=%.4f | val_f1=%.4f",
            epoch, args.epochs, train_loss_epoch, bal_acc, prec_macro, rec_macro, f1_macro,
        )

        if bal_acc > best_bal:
            best_bal = bal_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
            torch.save(
                {"model_state_dict": best_state, "epoch": epoch, "val_bal_acc": best_bal},
                os.path.join(out_dir, "best.pt"),
            )
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_t, y_p, y_pr = evaluate_model(model, val_loader, device, num_classes, use_amp)
    metrics = compute_classification_metrics(y_t, y_p, y_pr, n_classes=num_classes)
    metrics["val_balanced_accuracy"] = float(balanced_accuracy_score(y_t, y_p))
    metrics["train_time_sec"] = time.time() - t0

    labels_cm = list(range(num_classes))
    save_confusion_matrix_fig(y_t, y_p, labels_cm, os.path.join(out_dir, "confusion_matrix.png"))
    save_roc_curve(y_t, y_pr, num_classes, os.path.join(out_dir, "roc_curve.png"))
    save_training_curves(history, os.path.join(out_dir, "training_curves.png"))
    save_metric_curves(history, os.path.join(out_dir, "metric_curves.png"))

    attn_dir = os.path.join(out_dir, "interpretability")
    os.makedirs(attn_dir, exist_ok=True)
    try:
        _save_attention_samples(model, val_loader, device, attn_dir, use_amp)
    except Exception as e:
        logger.warning("Attention visualization skipped: %s", e)
    try:
        _save_grad_cam_samples(model, val_loader, device, attn_dir)
    except Exception as e:
        logger.warning("Grad-CAM visualization skipped: %s", e)

    config = {k: getattr(args, k) for k in dir(args) if not k.startswith("_") and not callable(getattr(args, k))}
    config["num_classes"] = num_classes
    config["n_train"] = len(train_idx)
    config["n_val"] = len(val_idx)
    if stem_meta is not None:
        config["mat_stem_split"] = {
            k: stem_meta[k]
            for k in ("split", "val_stems", "train_stems", "warning", "reason")
            if k in stem_meta and stem_meta[k] is not None
        }
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)

    def _json_safe(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, float):
            return obj
        return str(obj)

    metrics_out = {k: _json_safe(v) for k, v in metrics.items() if not isinstance(v, np.ndarray)}
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    torch.save({"model_state_dict": model.state_dict(), "metrics": metrics}, os.path.join(out_dir, "last.pt"))

    logger.info("Saved run to %s | best val_bal_acc=%.4f", out_dir, best_bal)
    return {"out_dir": out_dir, "metrics": metrics, "best_val_bal_acc": best_bal}


def build_train_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train spectrogram fault classifier")
    p.add_argument("--dataset_path", type=str, default=os.getenv("DATASET_PATH", "."))
    p.add_argument(
        "--dataset_subdir",
        type=str,
        default=None,
        help="Optional child of dataset_path (e.g. 0dB for MIMII spectrograms). "
        "Default: env DATASET_SUBDIR if set and non-empty.",
    )
    p.add_argument("--output_dir", type=str, default="results/spectrogram_fault")
    p.add_argument("--run_id", type=str, default=None)
    p.add_argument("--model", type=str, default="resnet50", help=f"One of: {list_model_names()} + timm names resnet50, efficientnet_b0")
    p.add_argument("--backbone", type=str, default="resnet50", help="Backbone for cnn_baseline / cnn_se / cnn_transformer")
    p.add_argument("--num_classes", type=int, default=None, help="Infer from data if omitted")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--augment", type=str, default="strong", choices=["off", "weak", "medium", "strong"])
    p.add_argument("--loss", type=str, default="ce_weighted", choices=["ce_weighted", "focal"])
    p.add_argument("--val_split", type=float, default=0.2)
    p.set_defaults(group_split=True)
    p.add_argument("--no_group_split", action="store_false", dest="group_split")
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no_pretrained", action="store_false", dest="pretrained")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no_amp", action="store_false", dest="amp")
    p.add_argument("--num_workers", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--init_checkpoint",
        type=str,
        default=None,
        help="Optional path to a .pt file with key model_state_dict to warm-start weights before training "
        "(e.g. MIMII best.pt before fine-tuning on another spectrogram folder).",
    )
    p.add_argument(
        "--infer_mat_stem_groups",
        action="store_true",
        help="Infer ImageDataset group ids from filenames like ``97_w000123.png`` (stem before ``_w``). "
        "Use for CWRU exports so validation can hold out whole .mat recordings instead of random windows.",
    )
    return p


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    return build_train_arg_parser().parse_args(argv)


def main():
    setup_logging(log_dir="logs")
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_one_run(args)


if __name__ == "__main__":
    main()
