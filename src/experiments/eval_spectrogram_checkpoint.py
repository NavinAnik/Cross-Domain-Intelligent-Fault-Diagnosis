"""
Evaluate a trained spectrogram fault checkpoint on a folder-based image dataset.

Expects ``normal/`` and ``abnormal/`` PNGs (same layout as ``ImageDataset`` in
``run_all_models``). Validation preprocessing matches ``get_fault_transforms(..., augment='off')``
(resize 224, ImageNet mean/std) without importing ``torchvision`` here so this entry point
does not pull in ``run_all_models`` / optional ``evaluate_models`` side effects.

Run from ``src/``:
  python -m experiments.eval_spectrogram_checkpoint \\
    --run_id 20260407_193339 --dataset_path ../data/cwru_spectrograms_png
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    auc,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.evaluation import compute_classification_metrics
from modules.spectrogram_cnn_models import build_model

logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _default_num_workers() -> int:
    return 0 if os.name == "nt" else 2


class SpectrogramImageFolderDataset(Dataset):
    """
    Same discovery rules as ``ImageDataset`` for ``normal`` / ``abnormal`` folders,
    without importing ``experiments.run_all_models``.
    """

    def __init__(self, root_dir: str, image_size: int = 224):
        self.samples: List[str] = []
        self.labels: List[int] = []
        self.image_size = int(image_size)
        label_map = {"normal": 0, "abnormal": 1}

        for item in os.listdir(root_dir):
            item_path = os.path.join(root_dir, item)
            if not os.path.isdir(item_path):
                continue
            if item in label_map:
                self._add_images_from_folder(item_path, label_map[item])
            else:
                for label_name in ["normal", "abnormal"]:
                    label_path = os.path.join(item_path, label_name)
                    if os.path.exists(label_path):
                        self._add_images_from_folder(label_path, label_map[label_name])
                    else:
                        for machine_id in os.listdir(item_path):
                            mid_path = os.path.join(item_path, machine_id)
                            if os.path.isdir(mid_path):
                                for ln in ["normal", "abnormal"]:
                                    lp = os.path.join(mid_path, ln)
                                    if os.path.exists(lp):
                                        self._add_images_from_folder(lp, label_map[ln])

        if self.samples:
            logger.info(
                "Loaded %d images, distribution: %s",
                len(self.samples),
                dict(Counter(self.labels)),
            )

    def _add_images_from_folder(self, folder_path: str, label: int) -> None:
        valid = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
        for f in os.listdir(folder_path):
            if any(f.lower().endswith(ext) for ext in valid):
                self.samples.append(os.path.join(folder_path, f))
                self.labels.append(label)

    def __len__(self) -> int:
        return len(self.samples)

    def _to_chw_normalized(self, pil: Image.Image) -> torch.Tensor:
        pil = pil.convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        chw = np.transpose(arr, (2, 0, 1))
        chw = (chw - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(np.ascontiguousarray(chw))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = Image.open(self.samples[idx])
        x = self._to_chw_normalized(image)
        return x, int(self.labels[idx])


@torch.no_grad()
def gather_probs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (y_true, softmax_probs) without applying a decision threshold."""
    model.eval()
    all_y: List[int] = []
    all_prob: List[np.ndarray] = []
    for x, y in loader:
        x = x.to(device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda" and use_amp):
            logits = model(x)
        prob = torch.softmax(logits.float(), dim=1).cpu().numpy()
        all_y.extend(y.numpy().tolist())
        all_prob.append(prob)
    probs = np.vstack(all_prob)
    return np.array(all_y), probs


def preds_at_threshold(probs: np.ndarray, num_classes: int, pos_threshold: Optional[float]) -> np.ndarray:
    if pos_threshold is None or num_classes != 2:
        return np.argmax(probs, axis=1).astype(np.int64)
    return (probs[:, 1] >= float(pos_threshold)).astype(np.int64)


def tune_binary_threshold(y_true: np.ndarray, pos_prob: np.ndarray, objective: str) -> Tuple[float, float]:
    """Grid-search threshold on P(class=1). Returns (best_threshold, best_objective_score on tune set)."""
    best_t = 0.5
    best_score = -1.0
    y_true = np.asarray(y_true).ravel()
    pos_prob = np.asarray(pos_prob).ravel()
    for t in np.linspace(0.001, 0.999, 500):
        pred = (pos_prob >= t).astype(np.int64)
        if objective == "f1":
            s = float(f1_score(y_true, pred, zero_division=0))
        else:
            s = float(balanced_accuracy_score(y_true, pred))
        if s > best_score:
            best_score = s
            best_t = float(t)
    return best_t, best_score


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    use_amp: bool,
    pos_threshold: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_t, probs = gather_probs(model, loader, device, use_amp)
    pred = preds_at_threshold(probs, num_classes, pos_threshold)
    return y_t, pred, probs


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


def save_confusion_matrix_fig(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[int],
    path: str,
    display_labels: Optional[List[str]] = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp_labels = display_labels if display_labels is not None else [str(i) for i in labels]
    disp = ConfusionMatrixDisplay(cm, display_labels=disp_labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int, path: str) -> None:
    """Precision–recall curve (binary: uses probability of class 1)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.figure(figsize=(7, 6))
    if n_classes == 2:
        pos = y_prob[:, 1]
        prec, rec, _ = precision_recall_curve(y_true, pos)
        ap = average_precision_score(y_true, pos)
        plt.plot(rec, prec, label=f"PR AUC (AP) = {ap:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision–recall curve (positive = class 1)")
    else:
        from sklearn.preprocessing import label_binarize

        y_bin = label_binarize(y_true, classes=np.arange(n_classes))
        for c in range(n_classes):
            if y_bin[:, c].sum() == 0:
                continue
            prec, rec, _ = precision_recall_curve(y_bin[:, c], y_prob[:, c])
            ap = average_precision_score(y_bin[:, c], y_prob[:, c])
            plt.plot(rec, prec, label=f"class {c} AP={ap:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision–recall curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_precrecf1_bar(precision: float, recall: float, f1: float, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.figure(figsize=(6, 4))
    vals = [float(precision), float(recall), float(f1)]
    plt.bar(["Precision", "Recall", "F1"], vals, color=["#4C72B0", "#55A868", "#C44E52"])
    plt.ylim(0.0, 1.05)
    for i, v in enumerate(vals):
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    plt.ylabel("Score (binary positive class)")
    plt.title("Precision, recall, and F1 (positive = class 1)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_per_class_precrecf1_bars(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    path: str,
    class_display_names: Optional[List[str]] = None,
) -> None:
    """Grouped bars: per-class precision, recall, F1 (sklearn average=None)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    labels = list(range(int(n_classes)))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    names = class_display_names if class_display_names is not None else [f"class {i}" for i in labels]
    x = np.arange(len(names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(names)), 4.5))
    ax.bar(x - w, prec, width=w, label="Precision", color="#4C72B0")
    ax.bar(x, rec, width=w, label="Recall", color="#55A868")
    ax.bar(x + w, f1, width=w, label="F1", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-class precision, recall, and F1")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_binary_threshold_prf_curves(y_true: np.ndarray, pos_prob: np.ndarray, path: str) -> None:
    """Precision, recall, and F1 vs decision threshold on P(class=1) (binary only)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    y_true = np.asarray(y_true).ravel()
    pos_prob = np.asarray(pos_prob).ravel()
    grid = np.linspace(0.0, 1.0, 501)
    precs: List[float] = []
    recs: List[float] = []
    f1s: List[float] = []
    for t in grid:
        pred = (pos_prob >= float(t)).astype(np.int64)
        p, r, f, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0
        )
        precs.append(float(p))
        recs.append(float(r))
        f1s.append(float(f))
    plt.figure(figsize=(7, 5))
    plt.plot(grid, precs, label="Precision", color="#4C72B0")
    plt.plot(grid, recs, label="Recall", color="#55A868")
    plt.plot(grid, f1s, label="F1", color="#C44E52")
    plt.xlabel("Threshold on P(abnormal)")
    plt.ylabel("Score (binary positive class)")
    plt.title("Precision, recall, and F1 vs threshold")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def _log_evaluation_summary(metrics: Dict[str, Any], num_classes: int) -> None:
    """Log main classification metrics to the console."""
    pos_name = "abnormal (class 1)" if num_classes == 2 else f"positive / macro (n_classes={num_classes})"
    logger.info("--- Metrics (binary average: positive = %s) ---", pos_name)
    logger.info(
        "precision=%.4f  recall=%.4f  f1=%.4f",
        float(metrics.get("precision", float("nan"))),
        float(metrics.get("recall", float("nan"))),
        float(metrics.get("f1", float("nan"))),
    )
    logger.info(
        "accuracy=%.4f  balanced_accuracy=%.4f",
        float(metrics.get("accuracy", float("nan"))),
        float(metrics.get("val_balanced_accuracy", float("nan"))),
    )
    logger.info(
        "auc_roc=%.4f  auc_pr=%.4f",
        float(metrics.get("auc_roc", float("nan"))),
        float(metrics.get("auc_pr", float("nan"))),
    )
    if num_classes == 2:
        logger.info(
            "macro: precision=%.4f recall=%.4f f1=%.4f",
            float(metrics.get("macro_precision", float("nan"))),
            float(metrics.get("macro_recall", float("nan"))),
            float(metrics.get("macro_f1", float("nan"))),
        )


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, float):
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _runs_parent_dir(results_root: str) -> str:
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", results_root)
    )


def _format_run_dir_hint(results_root: str) -> str:
    parent = _runs_parent_dir(results_root)
    if not os.path.isdir(parent):
        return f"(no directory at {parent!r} — train a model first or fix --results_root)"
    try:
        names = sorted(
            d
            for d in os.listdir(parent)
            if os.path.isdir(os.path.join(parent, d)) and not d.startswith(".")
        )
    except OSError:
        return f"(could not list {parent!r})"
    if not names:
        return f"(no run subfolders under {parent!r})"
    tail = ", ".join(names[-5:])
    return f"example: --run_id {names[-1]}  |  recent folders include: {tail}"


def load_config(checkpoint_path: str, config_path: Optional[str], results_root: str) -> Dict[str, Any]:
    ckpt = os.path.abspath(checkpoint_path)
    if config_path:
        cfg_path = os.path.abspath(config_path)
    else:
        cfg_path = os.path.join(os.path.dirname(ckpt), "config.json")
    if not os.path.isfile(cfg_path):
        hint = _format_run_dir_hint(results_root)
        raise FileNotFoundError(
            f"config.json not found: {cfg_path}\n"
            f"  Pass --config /path/to/config.json, or use a real run id (not the literal '<run_id>' placeholder).\n"
            f"  {hint}"
        )
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _labels_allow_stratify(labels: np.ndarray) -> bool:
    _, counts = np.unique(np.asarray(labels).ravel(), return_counts=True)
    return bool(len(counts) >= 2 and int(np.min(counts)) >= 2)


def _reject_doc_placeholder_checkpoint(path: str) -> None:
    if "<" in path or ">" in path:
        raise SystemExit(
            "The --checkpoint path looks like a documentation placeholder (it contains '<' or '>').\n"
            "Use your actual training run folder name, for example:\n"
            "  python -m experiments.eval_spectrogram_checkpoint "
            "--run_id 20260407_193339 --dataset_path ../data/cwru_spectrograms_png\n"
            "or:\n"
            "  python -m experiments.eval_spectrogram_checkpoint "
            "--checkpoint results/spectrogram_fault/20260407_193339/best.pt "
            "--dataset_path ../data/cwru_spectrograms_png"
        )


def main(argv: Optional[list] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description="Eval spectrogram_fault best.pt on an image folder dataset.",
        epilog=(
            "Examples (from src/):\n"
            "  python -m experiments.eval_spectrogram_checkpoint \\\n"
            "    --run_id 20260407_193339 --dataset_path ../data/cwru_spectrograms_png\n"
            "  python -m experiments.eval_spectrogram_checkpoint \\\n"
            "    --checkpoint results/spectrogram_fault/20260407_193339/best.pt \\\n"
            "    --dataset_path ../data/cwru_spectrograms_png\n"
            "\n"
            "Do not paste the literal text <run_id>; substitute your folder name under results/spectrogram_fault/.\n"
            "\n"
            "Cross-domain scores are often poor with argmax; try calibrating the decision threshold on data:\n"
            "  --tune_threshold_val_split 0.25 --tune_objective balanced_accuracy\n"
            "\n"
            "For strong CWRU accuracy, fine-tune on CWRU PNGs; use ``--infer_mat_stem_groups`` so val "
            "holds out whole .mat files (otherwise val can look unrealistically easy):\n"
            "  cd src && python -m experiments.train_spectrogram_fault "
            "--dataset_path ../data/cwru_spectrograms_png "
            "--init_checkpoint results/spectrogram_fault/20260407_193339/best.pt "
            "--infer_mat_stem_groups --epochs 30 --augment medium --run_id cwru_finetune_honest"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ckpt_group = p.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument("--checkpoint", type=str, default=None, help="Path to best.pt")
    ckpt_group.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Run folder name under results/spectrogram_fault/ (uses best.pt + config.json in that folder)",
    )
    p.add_argument(
        "--results_root",
        type=str,
        default="results/spectrogram_fault",
        help="Root directory (relative to src/) containing run_id subfolders (default: results/spectrogram_fault)",
    )
    p.add_argument("--config", type=str, default=None, help="Path to config.json (default: next to checkpoint)")
    p.add_argument("--dataset_path", type=str, required=True, help="Root with normal/ and abnormal/ images")
    p.add_argument("--output_dir", type=str, default=None, help="Where to write metrics/plots (default: checkpoint dir)")
    p.add_argument("--batch_size", type=int, default=None, help="Override batch size (default: from config or 32)")
    p.add_argument("--num_workers", type=int, default=-1, help="DataLoader workers (-1: default for OS)")
    p.add_argument("--no_amp", action="store_true", help="Disable AMP on CUDA")
    p.add_argument(
        "--tune_threshold_val_split",
        type=float,
        default=0.0,
        help="Fraction of samples (stratified) held out for evaluation; threshold is chosen on the "
        "complement to maximize --tune_objective (binary only). Example: 0.25.",
    )
    p.add_argument(
        "--tune_objective",
        type=str,
        default="balanced_accuracy",
        choices=["balanced_accuracy", "f1"],
        help="Score to maximize when searching P(abnormal) threshold (with --tune_threshold_val_split > 0).",
    )
    p.add_argument(
        "--pos_threshold",
        type=float,
        default=None,
        help="Binary only: predict abnormal if P(class=1) >= this value (instead of argmax). "
        "Do not combine with --tune_threshold_val_split > 0.",
    )
    p.add_argument(
        "--threshold_seed",
        type=int,
        default=42,
        help="RNG seed for stratified train vs hold-out split (threshold tuning).",
    )
    p.add_argument(
        "--also_report_argmax_full",
        action="store_true",
        help="With threshold tuning, also report argmax metrics on the full dataset (for comparison).",
    )
    args = p.parse_args(argv)

    if args.run_id:
        run_id = str(args.run_id).strip().strip("/\\")
        if not run_id or "<" in run_id or ">" in run_id:
            raise SystemExit("--run_id must be a real folder name (e.g. 20260407_193339), not a placeholder.")
        checkpoint_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            args.results_root,
            run_id,
            "best.pt",
        )
        checkpoint_path = os.path.normpath(checkpoint_path)
    else:
        checkpoint_path = os.path.normpath(
            os.path.expandvars(os.path.expanduser(str(args.checkpoint)))
        )

    _reject_doc_placeholder_checkpoint(checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        hint = _format_run_dir_hint(args.results_root)
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}\n  {hint}")

    cfg = load_config(checkpoint_path, args.config, args.results_root)
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    out_dir = os.path.abspath(args.output_dir) if args.output_dir else ckpt_dir
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    num_classes = int(cfg.get("num_classes", 2))
    model_name = str(cfg.get("model", "resnet50"))
    backbone = str(cfg.get("backbone", "resnet50"))
    image_size = int(cfg.get("image_size", 224))
    pretrained = bool(cfg.get("pretrained", True))
    batch_size = int(args.batch_size or cfg.get("batch_size", 32))
    nw = int(args.num_workers) if args.num_workers >= 0 else _default_num_workers()
    use_amp = (not args.no_amp) and device.type == "cuda"

    data_root = os.path.normpath(os.path.expandvars(os.path.expanduser(args.dataset_path)))
    ds = SpectrogramImageFolderDataset(data_root, image_size=image_size)
    if len(ds) == 0:
        raise FileNotFoundError(f"No images found under {data_root}")

    tune_split = float(args.tune_threshold_val_split)
    pos_thr = getattr(args, "pos_threshold", None)
    if tune_split > 0 and pos_thr is not None:
        raise SystemExit("Use only one of --pos_threshold and --tune_threshold_val_split (>0).")
    if tune_split > 0 and num_classes != 2:
        raise SystemExit("--tune_threshold_val_split requires a binary checkpoint (num_classes=2).")
    if pos_thr is not None and num_classes != 2:
        raise SystemExit("--pos_threshold requires a binary checkpoint (num_classes=2).")
    if tune_split > 0 and not (0.05 <= tune_split <= 0.9):
        raise SystemExit("--tune_threshold_val_split must be in [0.05, 0.9] (hold-out fraction).")

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=device.type == "cuda",
    )

    model = build_model(model_name, num_classes=num_classes, backbone=backbone, pretrained=pretrained).to(device)
    try:
        blob = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(checkpoint_path, map_location=device)
    state = blob.get("model_state_dict", blob)
    model.load_state_dict(state, strict=True)
    logger.info(
        "Loaded checkpoint %s | model=%s backbone=%s num_classes=%d",
        checkpoint_path,
        model_name,
        backbone,
        num_classes,
    )

    labels_np = np.asarray(ds.labels, dtype=np.int64)
    eval_mode = "argmax"
    threshold_used: Optional[float] = None
    tune_score_on_calibration: Optional[float] = None
    metrics_argmax_full: Optional[Dict[str, Any]] = None

    if tune_split > 0:
        if not _labels_allow_stratify(labels_np):
            raise SystemExit(
                "Each class needs at least 2 samples for stratified threshold tuning "
                "(or disable --tune_threshold_val_split)."
            )
        idx = np.arange(len(ds))
        tr_idx, te_idx = train_test_split(
            idx,
            test_size=tune_split,
            stratify=labels_np,
            random_state=int(args.threshold_seed),
        )
        tune_loader = DataLoader(
            Subset(ds, tr_idx.tolist()),
            batch_size=batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=device.type == "cuda",
        )
        test_loader = DataLoader(
            Subset(ds, te_idx.tolist()),
            batch_size=batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=device.type == "cuda",
        )
        y_tune, p_tune = gather_probs(model, tune_loader, device, use_amp)
        t_star, tune_obj = tune_binary_threshold(y_tune, p_tune[:, 1], str(args.tune_objective))
        y_t, y_pr = gather_probs(model, test_loader, device, use_amp)
        y_p = preds_at_threshold(y_pr, num_classes, t_star)
        threshold_used = t_star
        tune_score_on_calibration = tune_obj
        eval_mode = "stratified_holdout_tuned_threshold"
        logger.info(
            "Threshold tuning: objective=%s | calibration n=%d | hold-out n=%d | "
            "chosen P(abnormal)>=%.4f (calibration %s=%.4f)",
            args.tune_objective,
            len(tr_idx),
            len(te_idx),
            t_star,
            args.tune_objective,
            tune_obj,
        )
    elif pos_thr is not None:
        y_t, y_pr = gather_probs(model, loader, device, use_amp)
        y_p = preds_at_threshold(y_pr, num_classes, float(pos_thr))
        threshold_used = float(pos_thr)
        eval_mode = "fixed_pos_threshold_full_data"
        logger.info(
            "Using fixed --pos_threshold=%.4f on full dataset (n=%d).",
            threshold_used,
            len(y_t),
        )
    else:
        y_t, y_p, y_pr = evaluate_model(model, loader, device, num_classes, use_amp, None)

    metrics = compute_classification_metrics(y_t, y_p, y_pr, n_classes=num_classes)
    metrics["val_balanced_accuracy"] = float(balanced_accuracy_score(y_t, y_p))
    if threshold_used is not None:
        metrics["pos_threshold_used"] = float(threshold_used)

    if args.also_report_argmax_full and tune_split > 0:
        y_all, pr_all = gather_probs(model, loader, device, use_amp)
        y_arg = preds_at_threshold(pr_all, num_classes, None)
        m_arg = compute_classification_metrics(y_all, y_arg, pr_all, n_classes=num_classes)
        m_arg["val_balanced_accuracy"] = float(balanced_accuracy_score(y_all, y_arg))
        metrics_argmax_full = {
            k: _json_safe(v) for k, v in m_arg.items() if not isinstance(v, np.ndarray)
        }
        logger.info("Also_report_argmax_full: accuracy=%.4f bal_acc=%.4f", m_arg["accuracy"], m_arg["val_balanced_accuracy"])

    disclaimer = (
        "Cross-domain note: if this dataset was not part of training (e.g. CWRU vs MIMII spectrograms), "
        "low metrics can reflect domain shift rather than implementation errors."
    )
    if tune_split > 0:
        disclaimer += (
            " Reported numbers are on the stratified hold-out split only; the threshold was chosen "
            "on the complementary calibration split (no sample overlap)."
        )
    metrics["evaluation_note"] = disclaimer
    logger.info("%s", disclaimer)

    _log_evaluation_summary(metrics, num_classes)

    labels_cm = list(range(num_classes))
    cm_labels = (
        ["normal (0)", "abnormal (1)"] if num_classes == 2 else [str(i) for i in labels_cm]
    )
    suf = "_holdout_tuned" if eval_mode == "stratified_holdout_tuned_threshold" else ""
    path_cm = os.path.join(out_dir, f"eval_confusion_matrix{suf}.png")
    path_roc = os.path.join(out_dir, f"eval_roc_curve{suf}.png")
    path_pr = os.path.join(out_dir, f"eval_pr_curve{suf}.png")
    path_bars = os.path.join(out_dir, f"eval_precision_recall_f1{suf}.png")
    path_bars_per_class = os.path.join(out_dir, f"eval_per_class_precision_recall_f1{suf}.png")
    path_thr_curves = os.path.join(out_dir, f"eval_precision_recall_f1_vs_threshold{suf}.png")

    save_confusion_matrix_fig(y_t, y_p, labels_cm, path_cm, display_labels=cm_labels)
    save_roc_curve(y_t, y_pr, num_classes, path_roc)
    save_pr_curve(y_t, y_pr, num_classes, path_pr)
    save_precrecf1_bar(
        float(metrics["precision"]),
        float(metrics["recall"]),
        float(metrics["f1"]),
        path_bars,
    )
    save_per_class_precrecf1_bars(y_t, y_p, num_classes, path_bars_per_class, class_display_names=cm_labels)
    plot_paths: Dict[str, str] = {
        "confusion_matrix": path_cm,
        "roc_curve": path_roc,
        "pr_curve": path_pr,
        "precision_recall_f1_bars": path_bars,
        "per_class_precision_recall_f1_bars": path_bars_per_class,
    }
    if num_classes == 2:
        save_binary_threshold_prf_curves(y_t, y_pr[:, 1], path_thr_curves)
        plot_paths["precision_recall_f1_vs_threshold"] = path_thr_curves
    logger.info("Saved plots:")
    for k, p in plot_paths.items():
        logger.info("  %s: %s", k, p)

    metrics_out = {k: _json_safe(v) for k, v in metrics.items() if not isinstance(v, np.ndarray)}
    sidecar = {
        "checkpoint": os.path.abspath(checkpoint_path),
        "config_path": os.path.abspath(args.config) if args.config else os.path.join(ckpt_dir, "config.json"),
        "dataset_path": data_root,
        "plots": plot_paths,
        "export_meta": None,
        "eval_mode": eval_mode,
        "tune_threshold_val_split": tune_split if tune_split > 0 else None,
        "tune_objective": str(args.tune_objective) if tune_split > 0 else None,
        "tune_score_on_calibration": tune_score_on_calibration,
        "threshold_seed": int(args.threshold_seed) if tune_split > 0 else None,
    }
    meta_path = os.path.join(data_root, "export_meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            sidecar["export_meta"] = json.load(f)

    out_metrics_path = os.path.join(out_dir, "eval_metrics.json")
    with open(out_metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "eval_mode": eval_mode,
                "pos_threshold_used": threshold_used,
                "metrics": metrics_out,
                "metrics_argmax_full_dataset": metrics_argmax_full,
                "sidecar": sidecar,
            },
            f,
            indent=2,
        )
    logger.info("Wrote %s", out_metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
