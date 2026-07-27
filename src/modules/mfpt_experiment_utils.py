"""
Shared helpers for MFPT experiments (mirrors cwru_experiment_utils).
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from experiments.run_all_models import ImageDataset
from experiments.train_spectrogram_fault import _infer_mat_stem_groups_inplace
from modules.cwru_experiment_utils import (
    build_and_train_model,
    build_spectrogram_train_args,
    make_dataloaders,
    set_global_seeds,
)
from modules.spectrogram_fault_data import default_num_workers, get_fault_transforms, resolve_image_dataset_root

logger = logging.getLogger(__name__)

__all__ = [
    "load_manifest",
    "prepare_mfpt_image_dataset",
    "indices_for_label_fraction",
    "stem_stratified_group_kfold_splits",
    "build_and_train_model",
    "build_spectrogram_train_args",
    "make_dataloaders",
    "set_global_seeds",
    "save_test_evaluation_plots",
    "persist_training_history",
]


def stem_stratified_group_kfold_splits(
    labels: List[int],
    groups: List[str],
    k: int,
    seed: int,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], int, Dict[str, Any]]:
    """
    Recording-wise k-fold that holds out whole stems but keeps **both classes** in
    validation whenever each class has enough stems.

    Splits normal stems and abnormal stems separately (KFold), then pairs fold *i*
    so validation contains val stems from class 0 fold *i* and class 1 fold *i*.
    Mirrors CWRU ``get_train_val_indices_mat_stem_stratified`` at k-fold scale.
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    labels_arr = np.asarray(labels)
    groups_arr = np.asarray(groups)
    indices = np.arange(len(labels))

    stem_ids = sorted({str(g) for g in groups_arr.tolist() if str(g) != "default"})
    meta: Dict[str, Any] = {"split": "stem_stratified_group_kfold", "stem_ids": stem_ids}

    stem_to_label: Dict[str, int] = {}
    for s in stem_ids:
        idx = np.where(groups_arr == s)[0]
        ys = np.unique(labels_arr[idx])
        if len(ys) != 1:
            raise ValueError(f"Stem {s!r} mixes class labels; expected one class per stem.")
        stem_to_label[s] = int(ys[0])

    stems_0 = [s for s in stem_ids if stem_to_label[s] == 0]
    stems_1 = [s for s in stem_ids if stem_to_label[s] == 1]
    meta["stems_normal"] = stems_0
    meta["stems_abnormal"] = stems_1

    if not stems_0 or not stems_1:
        meta["warning"] = "Need stems in both classes; falling back to window stratified k-fold."
        k_eff = max(2, min(k, len(labels)))
        skf = StratifiedKFold(n_splits=k_eff, shuffle=True, random_state=seed)
        splits = list(skf.split(indices, labels_arr))
        meta["k_eff"] = k_eff
        meta["split"] = "fallback_stratified_window"
        return splits, k_eff, meta

    k_eff = min(k, len(stems_0), len(stems_1))
    if k_eff < 2:
        k_eff = 2
    meta["k_eff"] = k_eff

    kf0 = KFold(n_splits=k_eff, shuffle=True, random_state=seed)
    kf1 = KFold(n_splits=k_eff, shuffle=True, random_state=seed + 17)
    folds_0 = list(kf0.split(np.arange(len(stems_0))))
    folds_1 = list(kf1.split(np.arange(len(stems_1))))

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    fold_meta: List[Dict[str, Any]] = []

    for fold_i in range(k_eff):
        _, va0_local = folds_0[fold_i]
        _, va1_local = folds_1[fold_i]
        val_stems = {stems_0[j] for j in va0_local} | {stems_1[j] for j in va1_local}
        val_mask = np.isin(groups_arr, list(val_stems))
        val_idx = indices[val_mask]
        train_idx = indices[~val_mask]

        if len(val_idx) == 0 or len(train_idx) == 0:
            logger.warning("Skip degenerate stem-stratified fold %d (empty train or val)", fold_i)
            continue

        val_labels = sorted(set(int(labels_arr[i]) for i in val_idx))
        fold_meta.append(
            {
                "fold": fold_i,
                "val_stems": sorted(val_stems),
                "val_label_counts": {str(l): int((labels_arr[val_idx] == l).sum()) for l in val_labels},
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
            }
        )
        splits.append((train_idx, val_idx))

    meta["folds"] = fold_meta
    if not splits:
        raise ValueError("stem_stratified_group_kfold_splits produced no valid folds")
    return splits, k_eff, meta


def save_test_evaluation_plots(
    out_dir: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
    *,
    prefix: str = "test_",
    display_labels: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Save confusion matrix, ROC, PR, and precision/recall/F1 bar (CWRU eval style)."""
    from experiments.eval_spectrogram_checkpoint import (
        save_confusion_matrix_fig,
        save_pr_curve,
        save_precrecf1_bar,
        save_roc_curve,
    )
    from modules.evaluation import compute_classification_metrics

    os.makedirs(out_dir, exist_ok=True)
    labels_cm = list(range(num_classes))
    disp = display_labels or ["normal", "abnormal"][:num_classes]
    paths = {
        "confusion_matrix": os.path.join(out_dir, f"{prefix}confusion_matrix.png"),
        "roc_curve": os.path.join(out_dir, f"{prefix}roc_curve.png"),
        "pr_curve": os.path.join(out_dir, f"{prefix}pr_curve.png"),
        "precrecf1_bar": os.path.join(out_dir, f"{prefix}precrecf1_bar.png"),
    }
    save_confusion_matrix_fig(y_true, y_pred, labels_cm, paths["confusion_matrix"], display_labels=disp)
    save_roc_curve(y_true, y_prob, num_classes, paths["roc_curve"])
    save_pr_curve(y_true, y_prob, num_classes, paths["pr_curve"])
    metrics = compute_classification_metrics(y_true, y_pred, y_prob, n_classes=num_classes)
    save_precrecf1_bar(metrics["precision"], metrics["recall"], metrics["f1"], paths["precrecf1_bar"])
    return paths


def persist_training_history(out_dir: str, history: Dict[str, Any]) -> str:
    """Write per-epoch train/val history to JSON (loss, accuracy, precision, recall, F1)."""
    import json

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "history.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return path


def load_manifest(dataset_path: str) -> pd.DataFrame:
    path = os.path.join(dataset_path, "manifest.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"manifest.csv not found under {dataset_path}. "
            "Re-run export_mfpt_spectrogram_images on MFPT data."
        )
    return pd.read_csv(path)


def prepare_mfpt_image_dataset(
    dataset_path: str,
    *,
    image_size: int = 224,
    augment: str = "strong",
) -> Tuple[ImageDataset, ImageDataset, str]:
    data_root = resolve_image_dataset_root(dataset_path, None)
    tfm = get_fault_transforms(image_size=image_size, augment=augment)
    train_ds = ImageDataset(data_root, transform=tfm["train"])
    val_ds = ImageDataset(data_root, transform=tfm["val"])
    if len(train_ds) == 0:
        raise FileNotFoundError(f"No images under {data_root}")
    n_groups = _infer_mat_stem_groups_inplace(train_ds)
    _infer_mat_stem_groups_inplace(val_ds)
    logger.info("MFPT dataset: %d samples, %d groups", len(train_ds), n_groups)
    return train_ds, val_ds, data_root


def indices_for_label_fraction(
    labels: List[int],
    groups: List[str],
    fraction: float,
    *,
    seed: int = 42,
    pool_indices: Optional[List[int]] = None,
) -> Tuple[List[int], List[int]]:
    """
    Return (train_indices, test_indices) for few-shot fine-tuning.

    Selects ``fraction`` of unique recording stems (group-aware, stratified by class)
    for training; remaining pool indices form the test set.
    """
    from sklearn.model_selection import train_test_split

    fraction = float(fraction)
    if not 0 < fraction < 1:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")

    pool = list(range(len(labels))) if pool_indices is None else list(pool_indices)
    if not pool:
        return [], []

    pool_labels = [labels[i] for i in pool]
    pool_groups = [groups[i] for i in pool]

    stem_to_indices: Dict[str, List[int]] = {}
    for i, g in zip(pool, pool_groups):
        stem_to_indices.setdefault(str(g), []).append(i)

    stem_to_label: Dict[str, int] = {}
    for stem, idxs in stem_to_indices.items():
        ys = {labels[i] for i in idxs}
        if len(ys) != 1:
            raise ValueError(f"Stem {stem!r} mixes labels in pool")
        stem_to_label[stem] = int(next(iter(ys)))

    stems_0 = [s for s, y in stem_to_label.items() if y == 0]
    stems_1 = [s for s, y in stem_to_label.items() if y == 1]

    def _pick_stems(stems: List[str]) -> Tuple[List[str], List[str]]:
        if not stems:
            return [], []
        if len(stems) == 1:
            return stems, []
        n_train = max(1, int(round(len(stems) * fraction)))
        n_train = min(n_train, len(stems) - 1)  # leave at least one stem for test pool
        rng = random.Random(seed)
        shuffled = list(stems)
        rng.shuffle(shuffled)
        return shuffled[:n_train], shuffled[n_train:]

    tr0, _ = _pick_stems(stems_0)
    tr1, _ = _pick_stems(stems_1)
    train_stems = set(tr0) | set(tr1)

    # Ensure at least one stem per class when possible
    for stems_side, label in ((stems_0, 0), (stems_1, 1)):
        if not any(stem_to_label.get(s) == label for s in train_stems):
            if stems_side:
                train_stems.add(stems_side[0])

    train_idx: List[int] = []
    test_idx: List[int] = []
    for pos, i in enumerate(pool):
        if str(pool_groups[pos]) in train_stems:
            train_idx.append(i)
        else:
            test_idx.append(i)

    if not train_idx or not test_idx:
        # Fallback: random sample split at window level
        train_idx, test_idx = train_test_split(
            pool,
            train_size=fraction,
            random_state=seed,
            stratify=pool_labels if len(set(pool_labels)) > 1 else None,
        )
        train_idx = list(train_idx)
        test_idx = list(test_idx)

    logger.info(
        "Few-shot fraction=%.2f -> train=%d test=%d stems=%d",
        fraction,
        len(train_idx),
        len(test_idx),
        len(train_stems),
    )
    return train_idx, test_idx


def indices_train_val_from_pool(
    indices: List[int],
    labels: List[int],
    groups: List[str],
    val_split: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """Inner train/val split within a pool using mat-stem stratified logic."""
    from modules.spectrogram_fault_data import get_train_val_indices_mat_stem_stratified

    sub_labels = [labels[i] for i in indices]
    sub_groups = [groups[i] for i in indices]
    tr_local, va_local, _meta = get_train_val_indices_mat_stem_stratified(
        sub_labels, sub_groups, val_split=val_split, random_state=seed
    )
    train_idx = [indices[i] for i in tr_local.tolist()]
    val_idx = [indices[i] for i in va_local.tolist()]
    return train_idx, val_idx
