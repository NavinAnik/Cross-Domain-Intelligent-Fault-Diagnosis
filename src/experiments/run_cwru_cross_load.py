"""
Cross-load generalization for CWRU spectrogram fault classification.

Trains on one motor load (HP) and evaluates on another (loads 1, 2, 3 by default).

Run from ``src/``:
  python -m experiments.run_cwru_cross_load --dataset_path ../data/cwru_spectrograms_full_png --epochs 25
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.train_spectrogram_fault import evaluate_model
from modules.cwru_experiment_utils import (
    build_and_train_model,
    build_spectrogram_train_args,
    indices_for_load,
    load_manifest,
    prepare_cwru_image_dataset,
    set_global_seeds,
)
from modules.cwru_metadata import stem_from_png_basename
from modules.cwru_split_checks import assert_load_disjoint, write_verification_report
from modules.evaluation import compute_classification_metrics
from modules.spectrogram_fault_data import get_train_val_indices_mat_stem_stratified
from utils.manuscript_io import save_figure, save_table

logger = logging.getLogger(__name__)


def _inner_train_val_split(
    indices: List[int],
    labels: List[int],
    groups: List[str],
    val_split: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    sub_labels = [labels[i] for i in indices]
    sub_groups = [groups[i] for i in indices]
    tr_local, va_local, _meta = get_train_val_indices_mat_stem_stratified(
        sub_labels, sub_groups, val_split=val_split, random_state=seed
    )
    train_idx = [indices[i] for i in tr_local.tolist()]
    val_idx = [indices[i] for i in va_local.tolist()]
    return train_idx, val_idx


def _stems_for_indices(samples: List[str], indices: List[int]) -> List[str]:
    out: List[str] = []
    for i in indices:
        stem = stem_from_png_basename(os.path.basename(samples[i]))
        if stem:
            out.append(stem)
    return out


def _plot_heatmap(
    loads: List[int],
    matrix: np.ndarray,
    path: str,
    title: str = "Cross-load transfer accuracy",
) -> None:
    if np.isnan(matrix).all():
        logger.warning("Cross-load matrix contains only NaN values; saving diagnostic heatmap")

    fig, ax = plt.subplots(figsize=(6, 5))
    scaled = matrix * 100
    masked = np.ma.masked_invalid(scaled)
    im = ax.imshow(masked, cmap="viridis", vmin=0, vmax=100)
    ax.set_xticks(range(len(loads)))
    ax.set_yticks(range(len(loads)))
    ax.set_xticklabels([f"Test {l} HP" for l in loads])
    ax.set_yticklabels([f"Train {l} HP" for l in loads])
    for i in range(len(loads)):
        for j in range(len(loads)):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", color="white", fontsize=10)
            elif np.isnan(matrix[i, j]):
                ax.text(j, i, "N/A", ha="center", va="center", color="black", fontsize=9)
            else:
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j] * 100:.1f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                )
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Accuracy (%)")
    fig.tight_layout()
    save_figure(fig, path)


def run_cross_load(
    dataset_path: str,
    output_dir: str,
    *,
    loads: Optional[List[int]] = None,
    epochs: int = 25,
    seed: int = 42,
    init_checkpoint: Optional[str] = None,
    model: str = "resnet50",
    batch_size: int = 32,
    val_split: float = 0.15,
) -> Dict[str, Any]:
    set_global_seeds(seed)
    loads = loads or [1, 2, 3]
    manifest = load_manifest(dataset_path)
    train_ds, val_ds, data_root = prepare_cwru_image_dataset(dataset_path)

    train_args = build_spectrogram_train_args(
        dataset_path=dataset_path,
        model=model,
        epochs=epochs,
        batch_size=batch_size,
        infer_mat_stem_groups=True,
    )

    os.makedirs(output_dir, exist_ok=True)
    pairs = [(tr, te) for tr in loads for te in loads if tr != te]
    rows: List[List[Any]] = []
    verification: List[Dict[str, Any]] = []
    load_to_i = {l: i for i, l in enumerate(loads)}
    acc_matrix = np.full((len(loads), len(loads)), np.nan)

    for pair_i, (train_load, test_load) in enumerate(pairs):
        pair_seed = seed + pair_i * 500
        train_pool = indices_for_load(manifest, train_ds.samples, train_load)
        test_pool = indices_for_load(manifest, train_ds.samples, test_load)
        if not train_pool or not test_pool:
            logger.warning("Skip pair %d->%d: empty pool (train=%d test=%d)", train_load, test_load, len(train_pool), len(test_pool))
            continue

        train_idx, val_idx = _inner_train_val_split(
            train_pool, train_ds.labels, train_ds.groups, val_split, pair_seed
        )
        tr_stems = _stems_for_indices(train_ds.samples, train_pool)
        te_stems = _stems_for_indices(train_ds.samples, test_pool)
        verification.append(assert_load_disjoint(tr_stems, te_stems, manifest))

        pair_dir = os.path.join(output_dir, f"train{train_load}_test{test_load}")
        result = build_and_train_model(
            train_ds,
            val_ds,
            train_idx,
            val_idx,
            out_dir=pair_dir,
            seed=pair_seed,
            init_checkpoint=init_checkpoint,
            train_args=train_args,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nw = train_args.num_workers if train_args.num_workers >= 0 else 0
        test_loader = DataLoader(
            Subset(val_ds, test_pool),
            batch_size=batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=device.type == "cuda",
        )
        from modules.spectrogram_cnn_models import build_model

        num_classes = max(train_ds.labels) + 1
        eval_model = build_model(model, num_classes=num_classes, pretrained=False).to(device)
        if result["best_state"]:
            eval_model.load_state_dict(result["best_state"])
        y_t, y_p, y_pr = evaluate_model(eval_model, test_loader, device, num_classes, train_args.amp)
        test_metrics = compute_classification_metrics(y_t, y_p, y_pr, n_classes=num_classes)

        rows.append(
            [
                train_load,
                test_load,
                round(float(test_metrics["accuracy"]), 4),
                round(float(test_metrics["precision"]), 4),
                round(float(test_metrics["recall"]), 4),
                round(float(test_metrics["f1"]), 4),
            ]
        )
        acc_matrix[load_to_i[train_load], load_to_i[test_load]] = float(test_metrics["accuracy"])

        config = {
            "train_load": train_load,
            "test_load": test_load,
            "seed": pair_seed,
            "n_train_pool": len(train_pool),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(test_pool),
            "init_checkpoint": init_checkpoint,
        }
        with open(os.path.join(pair_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        with open(os.path.join(pair_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump({k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in test_metrics.items() if not isinstance(v, np.ndarray)}, f, indent=2, default=str)

        logger.info(
            "Train %d HP -> Test %d HP | acc=%.4f f1=%.4f",
            train_load,
            test_load,
            test_metrics["accuracy"],
            test_metrics["f1"],
        )

    write_verification_report(
        os.path.join(output_dir, "split_verification_report"),
        verification,
        title="CWRU cross-load split verification",
    )

    headers = ["Train Load", "Test Load", "Accuracy", "Precision", "Recall", "F1"]
    save_table(rows, headers, os.path.join(output_dir, "cross_load"), caption="Cross-load generalization (CWRU 12k DE)")

    csv_path = os.path.join(output_dir, "cross_load.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)

    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    _plot_heatmap(loads, acc_matrix, os.path.join(fig_dir, "cross_load_heatmap"))

    return {"rows": rows, "loads": loads, "verification_passed": all(v.get("passed") for v in verification)}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CWRU cross-load generalization")
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--loads", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init_checkpoint", type=str, default=None)
    p.add_argument("--model", type=str, default="resnet50")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--val_split", type=float, default=0.15)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    out = args.output_dir or os.path.join(
        "results/cwru_cross_load", datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_cross_load(
        args.dataset_path,
        out,
        loads=args.loads,
        epochs=args.epochs,
        seed=args.seed,
        init_checkpoint=args.init_checkpoint,
        model=args.model,
        batch_size=args.batch_size,
        val_split=args.val_split,
    )
    logger.info("Cross-load complete: %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
