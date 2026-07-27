"""
Shared helpers for CWRU reviewer-response experiments.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from experiments.run_all_models import ImageDataset
from experiments.train_spectrogram_fault import (
    LinearWarmupCosineLR,
    _infer_mat_stem_groups_inplace,
    _make_criterion,
    build_train_arg_parser,
    train_and_validate,
)
from modules.spectrogram_cnn_models import build_model
from modules.spectrogram_fault_data import default_num_workers, get_fault_transforms, resolve_image_dataset_root


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_manifest(dataset_path: str) -> pd.DataFrame:
    path = os.path.join(dataset_path, "manifest.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"manifest.csv not found under {dataset_path}. "
            "Re-run export_cwru_spectrogram_images on full_12k_de data."
        )
    return pd.read_csv(path)


def prepare_cwru_image_dataset(
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
    return train_ds, val_ds, data_root


def build_spectrogram_train_args(**overrides: Any) -> argparse.Namespace:
    args = build_train_arg_parser().parse_args([])
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def make_dataloaders(
    train_ds: ImageDataset,
    val_ds: ImageDataset,
    train_idx: List[int],
    val_idx: List[int],
    *,
    batch_size: int,
    device: torch.device,
    num_workers: int = -1,
) -> Tuple[DataLoader, DataLoader]:
    nw = num_workers if num_workers >= 0 else default_num_workers()
    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        Subset(val_ds, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=device.type == "cuda",
    )
    return train_loader, val_loader


def build_and_train_model(
    train_ds: ImageDataset,
    val_ds: ImageDataset,
    train_idx: List[int],
    val_idx: List[int],
    *,
    out_dir: str,
    seed: int,
    init_checkpoint: Optional[str] = None,
    train_args: Optional[argparse.Namespace] = None,
    save_plots: bool = False,
) -> Dict[str, Any]:
    """Build ResNet50 (or configured model), optionally warm-start, train, return fold result."""
    set_global_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = train_args or build_spectrogram_train_args()
    train_labels = [train_ds.labels[i] for i in train_idx]
    num_classes = max(train_ds.labels) + 1

    train_loader, val_loader = make_dataloaders(
        train_ds, val_ds, train_idx, val_idx, batch_size=args.batch_size, device=device, num_workers=args.num_workers
    )

    model = build_model(
        args.model,
        num_classes=num_classes,
        backbone=args.backbone,
        pretrained=args.pretrained,
    ).to(device)

    if init_checkpoint:
        ck = os.path.normpath(os.path.expandvars(os.path.expanduser(str(init_checkpoint))))
        if not os.path.isfile(ck):
            raise FileNotFoundError(f"init_checkpoint not found: {ck}")
        try:
            warm = torch.load(ck, map_location=device, weights_only=False)
        except TypeError:
            warm = torch.load(ck, map_location=device)
        sd = warm.get("model_state_dict", warm)
        model.load_state_dict(sd, strict=True)

    criterion = _make_criterion(args.loss, train_labels, num_classes, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = LinearWarmupCosineLR(optimizer, warmup_epochs=args.warmup_epochs, total_epochs=args.epochs)

    os.makedirs(out_dir, exist_ok=True)
    result = train_and_validate(
        model,
        criterion,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        device,
        args,
        out_dir,
        num_classes,
        save_best=True,
        save_plots=save_plots,
    )
    result["device"] = str(device)
    result["n_train"] = len(train_idx)
    result["n_val"] = len(val_idx)
    return result


def indices_for_load(manifest: pd.DataFrame, samples: List[str], load_hp: int) -> List[int]:
    """Return dataset indices whose manifest load_hp matches."""
    stem_load = (
        manifest.drop_duplicates("stem")
        .assign(stem=lambda df: df["stem"].astype(str))
        .set_index("stem")["load_hp"]
        .to_dict()
    )
    target = int(load_hp)
    out: List[int] = []
    for i, path in enumerate(samples):
        base = os.path.basename(path)
        from modules.cwru_metadata import stem_from_png_basename

        stem = stem_from_png_basename(base)
        if stem is None:
            continue
        raw = stem_load.get(str(stem))
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            continue
        if int(raw) == target:
            out.append(i)
    logger.info("Load %s -> %d samples", load_hp, len(out))
    return out
