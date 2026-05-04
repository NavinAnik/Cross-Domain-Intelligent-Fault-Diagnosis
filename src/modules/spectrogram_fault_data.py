"""
Data pipeline for spectrogram-image fault classification.

- Train/val transforms: ImageNet normalize + optional SpecAugment (time/freq mask, noise).
- Group-aware train/val index split (StratifiedGroupKFold) reusing ImageDataset from run_all_models.
"""

from __future__ import annotations

import os
import collections
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torchvision import transforms

from modules.spectrogram_augment import (
    FrequencyMask,
    TimeMask,
    GaussianNoiseTensor,
    TimeShift,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Compatibility shim for torchvision builds that still reference collections.Iterable
# on Python 3.10+ / 3.14 where it moved to collections.abc.
if not hasattr(collections, "Iterable"):
    collections.Iterable = Iterable


def resolve_image_dataset_root(root_dir: str, subdir: Optional[str] = None) -> str:
    """
    Resolve the directory passed to ``ImageDataset``.

    For MIMII-style trees, ``root_dir`` may be ``.../spectrograms`` and first-level folders
    are SNR (e.g. ``-6dB``, ``0dB``, ``6dB``), then machine type, then ``normal``/``abnormal``.
    ``ImageDataset`` already discovers that layout when ``subdir`` is omitted.

    Set ``subdir`` (e.g. ``0dB``) to restrict loading to a single child folder under ``root_dir``.
    """
    root_dir = os.path.normpath(os.path.expandvars(os.path.expanduser(root_dir)))
    if not subdir:
        return root_dir
    sub = subdir.strip().strip("/\\")
    if not sub:
        return root_dir
    out = os.path.join(root_dir, sub)
    if not os.path.isdir(out):
        raise FileNotFoundError(
            f"Dataset subdir not found: {out!r} (under root {root_dir!r})"
        )
    return out


def get_fault_transforms(
    image_size: int = 224,
    augment: str = "strong",
) -> Dict[str, Any]:
    """
    Build train and val transform dicts.

    Args:
        image_size: Square resize side.
        augment: 'off' | 'weak' | 'strong' | 'medium'
            'off': resize, to tensor, ImageNet normalize only.
            Others: add tensor-stage SpecAugment-style ops after normalize.
    """
    base_val = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    if augment == "off":
        train = base_val
        return {"train": train, "val": base_val}

    if augment == "weak":
        train = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomAffine(degrees=0, translate=(0.02, 0.01), scale=(0.98, 1.02)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            FrequencyMask(max_mask_pct=0.10, num_masks=1),
            TimeMask(max_mask_pct=0.10, num_masks=1),
            GaussianNoiseTensor(std=0.005),
        ])
    elif augment == "medium":
        train = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomAffine(degrees=0, translate=(0.03, 0.01), scale=(0.97, 1.03)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            FrequencyMask(max_mask_pct=0.15, num_masks=2),
            TimeMask(max_mask_pct=0.15, num_masks=2),
            TimeShift(max_shift_pct=0.05),
            GaussianNoiseTensor(std=0.005),
        ])
    else:  # strong
        train = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomAffine(degrees=0, translate=(0.04, 0.02), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            FrequencyMask(max_mask_pct=0.20, num_masks=2),
            TimeMask(max_mask_pct=0.20, num_masks=2),
            TimeShift(max_shift_pct=0.08),
            GaussianNoiseTensor(std=0.01),
        ])

    return {"train": train, "val": base_val}


def get_train_val_indices(
    labels: List[int],
    groups: List[str],
    val_split: float = 0.2,
    random_state: int = 42,
    use_group_split: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return train_idx, val_idx as numpy arrays.

    If use_group_split and more than one unique group: StratifiedGroupKFold first fold
    (approx val fraction ~ 1/n_splits). Otherwise stratified random split.
    """
    labels_arr = np.asarray(labels)
    groups_arr = np.asarray(groups)
    n = len(labels)
    indices = np.arange(n)
    n_unique_groups = len(set(groups))

    if use_group_split and n_unique_groups > 1:
        from sklearn.model_selection import StratifiedGroupKFold
        n_splits = max(2, int(round(1.0 / val_split)))
        n_splits = min(n_splits, n_unique_groups)
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        train_idx, val_idx = next(sgkf.split(np.zeros(n), labels_arr, groups_arr))
        return train_idx, val_idx

    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_split,
        stratify=labels_arr,
        random_state=random_state,
    )
    return train_idx, val_idx


def get_train_val_indices_mat_stem_stratified(
    labels: List[int],
    groups: List[str],
    val_split: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Train/val split when each group id (e.g. CWRU ``97`` from ``97_w000.png``) is **label-pure**.

    ``StratifiedGroupKFold`` can assign only normal-only or only abnormal-only **files** to
    validation, which makes metrics trivial and triggers sklearn warnings. Here we split
    **stems** within class 0 and within class 1 separately, then assign all windows from val
    stems to validation so val almost always contains both classes (when each side has ≥2 stems).
    """
    from sklearn.model_selection import train_test_split

    labels_arr = np.asarray(labels)
    groups_arr = np.asarray(groups)
    indices = np.arange(len(labels))
    stem_ids = sorted({str(g) for g in groups_arr.tolist() if str(g) != "default"})
    meta: Dict[str, Any] = {"split": "mat_stem_stratified_by_class", "stem_ids": stem_ids}

    if len(stem_ids) < 2:
        tr, va = get_train_val_indices(labels, groups, val_split, random_state, use_group_split=False)
        meta["split"] = "fallback_sample_stratified"
        meta["reason"] = "need at least 2 non-default stem groups"
        return tr, va, meta

    stem_to_label: Dict[str, int] = {}
    for s in stem_ids:
        idx = np.where(groups_arr == s)[0]
        ys = np.unique(labels_arr[idx])
        if len(ys) != 1:
            raise ValueError(
                f"Mat stem {s!r} mixes class labels; expected one class per stem (check export filenames)."
            )
        stem_to_label[s] = int(ys[0])

    stems_0 = [s for s in stem_ids if stem_to_label[s] == 0]
    stems_1 = [s for s in stem_ids if stem_to_label[s] == 1]

    def _stem_train_val(st_lst: List[str]) -> Tuple[List[str], List[str]]:
        if not st_lst:
            return [], []
        if len(st_lst) == 1:
            return st_lst, []
        tr, va = train_test_split(
            st_lst,
            test_size=val_split,
            random_state=random_state,
            shuffle=True,
        )
        return list(tr), list(va)

    tr0, va0 = _stem_train_val(stems_0)
    tr1, va1 = _stem_train_val(stems_1)
    val_stems = set(va0) | set(va1)
    train_stems = set(tr0) | set(tr1)

    def _borrow_one(stems_side: List[str], label: int) -> None:
        nonlocal val_stems, train_stems
        if any(stem_to_label[s] == label for s in val_stems):
            return
        pool = [s for s in train_stems if s in stems_side]
        if len(pool) < 2:
            return
        pick = pool[-1]
        val_stems.add(pick)
        train_stems.discard(pick)

    _borrow_one(stems_0, 0)
    _borrow_one(stems_1, 1)

    meta["val_stems"] = sorted(val_stems)
    meta["train_stems"] = sorted(train_stems)
    vc = {stem_to_label[s] for s in val_stems}
    if vc != {0, 1}:
        meta["warning"] = (
            f"Validation stems do not cover both classes (present={sorted(vc)}); "
            "metrics may be degenerate. Add more .mat files per class or adjust val_split."
        )

    val_mask = np.isin(groups_arr, list(val_stems))
    train_mask = ~val_mask
    val_idx = indices[val_mask]
    train_idx = indices[train_mask]
    if len(val_idx) == 0 or len(train_idx) == 0:
        tr, va = get_train_val_indices(labels, groups, val_split, random_state, use_group_split=False)
        meta["split"] = "fallback_sample_stratified"
        meta["reason"] = "mat-stem partition produced empty train or val"
        return tr, va, meta

    return train_idx, val_idx, meta


def default_num_workers() -> int:
    return 0 if os.name == "nt" else 2
