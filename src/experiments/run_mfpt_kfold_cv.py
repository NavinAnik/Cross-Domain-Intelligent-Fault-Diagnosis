"""
K-fold cross-validation for MFPT spectrogram fault classification.

Mirrors run_cwru_kfold_cv.py.

Run from ``src/``:
  python -m experiments.run_mfpt_kfold_cv --dataset_path ../data/mfpt_spectrograms --cv_mode both --k 5 --epochs 25
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
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.cwru_split_checks import assert_folds_disjoint, assert_no_file_overlap, write_verification_report
from modules.mfpt_experiment_utils import (
    build_and_train_model,
    build_spectrogram_train_args,
    load_manifest,
    persist_training_history,
    prepare_mfpt_image_dataset,
    set_global_seeds,
    stem_stratified_group_kfold_splits,
)
from modules.statistical_tests import aggregate_cv_metrics
from modules.visualization import plot_cv_metric_boxplots
from utils.manuscript_io import PLOT_MODEL_LABEL, save_figure, save_table

logger = logging.getLogger(__name__)

METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr"]


def _fold_splits(
    mode: str,
    n_samples: int,
    labels: List[int],
    groups: List[str],
    k: int,
    seed: int,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], int, Dict[str, Any]]:
    idx = np.arange(n_samples)
    if mode == "stratified":
        k_eff = max(2, min(k, n_samples))
        splitter = StratifiedKFold(n_splits=k_eff, shuffle=True, random_state=seed)
        return list(splitter.split(idx, labels)), k_eff, {"split": "stratified_window", "k_eff": k_eff}
    if mode == "grouped":
        splits, k_eff, meta = stem_stratified_group_kfold_splits(labels, groups, k, seed)
        if k_eff != k:
            logger.warning(
                "Adjusted grouped k-fold from %d to %d (min stems per class=%d)",
                k,
                k_eff,
                k_eff,
            )
        for fm in meta.get("folds", []):
            logger.info(
                "Fold %s val stems %s | label counts %s",
                fm.get("fold"),
                fm.get("val_stems"),
                fm.get("val_label_counts"),
            )
        return splits, k_eff, meta
    raise ValueError(f"Unknown cv_mode: {mode}")


def _metrics_row(fold: int, metrics: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    row = {"fold": fold}
    for mk in METRIC_KEYS:
        row[mk] = metrics.get(mk)
    row.update(extra)
    return row


def _save_folds_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _plot_fold_accuracy(rows: List[Dict[str, Any]], path: str, title: str) -> None:
    folds = [r["fold"] for r in rows]
    accs = [float(r["accuracy"]) * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(folds, accs, "o-", linewidth=2, markersize=8)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.set_xticks(folds)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    save_figure(fig, path)


def _plot_convergence_histories(histories: List[Dict[str, List[float]]], path: str, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, hist in enumerate(histories):
        epochs = range(1, len(hist["train_loss"]) + 1)
        axes[0].plot(epochs, hist["train_loss"], alpha=0.7, label=f"fold {i}")
        axes[0].plot(epochs, hist["val_loss"], alpha=0.7, linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Train / val loss (per fold)")
    axes[0].grid(alpha=0.3)

    for i, hist in enumerate(histories):
        epochs = range(1, len(hist["val_accuracy"]) + 1)
        axes[1].plot(epochs, [v * 100 for v in hist["val_accuracy"]], alpha=0.7, label=f"fold {i}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val accuracy (%)")
    axes[1].set_title("Validation accuracy (per fold)")
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0, 105)
    fig.suptitle(title)
    fig.tight_layout()
    save_figure(fig, path)


def run_kfold_cv(
    dataset_path: str,
    output_dir: str,
    *,
    cv_mode: str = "both",
    k: int = 5,
    epochs: int = 25,
    seed: int = 42,
    init_checkpoint: Optional[str] = None,
    model: str = "resnet50",
    batch_size: int = 32,
) -> Dict[str, Any]:
    set_global_seeds(seed)
    _ = load_manifest(dataset_path)
    train_ds, val_ds, data_root = prepare_mfpt_image_dataset(dataset_path)
    labels = train_ds.labels
    groups = train_ds.groups
    n = len(train_ds)

    modes: List[str] = []
    if cv_mode in ("stratified", "both"):
        modes.append("stratified")
    if cv_mode in ("grouped", "both"):
        modes.append("grouped")

    train_args = build_spectrogram_train_args(
        dataset_path=dataset_path,
        model=model,
        epochs=epochs,
        batch_size=batch_size,
        infer_mat_stem_groups=True,
    )

    all_results: Dict[str, Any] = {"modes": {}, "dataset_path": dataset_path, "data_root": data_root, "k": k}

    for mode in modes:
        logger.info("=== MFPT K-fold CV mode=%s k=%d ===", mode, k)
        splits, k_eff, split_meta = _fold_splits(mode, n, labels, groups, k, seed)
        fold_rows: List[Dict[str, Any]] = []
        conv_rows: List[List[Any]] = []
        histories: List[Dict[str, List[float]]] = []
        val_index_sets: List[List[int]] = []
        verification: List[Dict[str, Any]] = []

        mode_dir = os.path.join(output_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)
        with open(os.path.join(mode_dir, "split_meta.json"), "w", encoding="utf-8") as f:
            json.dump(split_meta, f, indent=2)

        fold_counter = 0
        for train_idx, val_idx in splits:
            train_list = train_idx.tolist()
            val_list = val_idx.tolist()
            if not train_list or not val_list:
                logger.warning(
                    "Skip fold with empty train (%d) or val (%d) — use more .mat files or lower --k",
                    len(train_list),
                    len(val_list),
                )
                continue

            fold_seed = seed + fold_counter * 1000
            val_index_sets.append(val_list)

            train_paths = [train_ds.samples[i] for i in train_list]
            val_paths = [train_ds.samples[i] for i in val_list]
            verification.append(assert_no_file_overlap(train_paths, val_paths))

            fold_dir = os.path.join(mode_dir, f"fold_{fold_counter}")
            result = build_and_train_model(
                train_ds,
                val_ds,
                train_list,
                val_list,
                out_dir=fold_dir,
                seed=fold_seed,
                init_checkpoint=init_checkpoint,
                train_args=train_args,
                save_plots=True,
            )
            persist_training_history(fold_dir, result["history"])
            metrics = result["final_metrics"]
            histories.append(result["history"])

            extra = {
                "best_epoch": result["best_epoch"],
                "loss_stabilized_epoch": result["loss_stabilized_epoch"],
                "total_epochs": result["total_epochs"],
                "val_balanced_accuracy": result["best_bal_acc"],
            }
            fold_rows.append(_metrics_row(fold_counter, metrics, extra))
            conv_rows.append(
                [fold_counter, result["best_epoch"], result["loss_stabilized_epoch"], result["total_epochs"]]
            )

            with open(os.path.join(fold_dir, "split_indices.json"), "w", encoding="utf-8") as f:
                json.dump({"train_idx": train_list, "val_idx": val_list, "seed": fold_seed}, f, indent=2)
            with open(os.path.join(fold_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "fold": fold_counter,
                        "mode": mode,
                        "seed": fold_seed,
                        "n_train": len(train_list),
                        "n_val": len(val_list),
                        "init_checkpoint": init_checkpoint,
                        "model": model,
                        "epochs": epochs,
                    },
                    f,
                    indent=2,
                )

            logger.info(
                "Fold %d | acc=%.4f f1=%.4f best_epoch=%d",
                fold_counter,
                metrics.get("accuracy", 0),
                metrics.get("f1", 0),
                result["best_epoch"],
            )
            fold_counter += 1

        verification.append(assert_folds_disjoint(val_index_sets))
        write_verification_report(
            os.path.join(mode_dir, "split_verification_report"),
            verification,
            title=f"MFPT K-fold split verification ({mode})",
        )

        folds_csv = os.path.join(mode_dir, f"folds_{mode}.csv")
        _save_folds_csv(fold_rows, folds_csv)

        aggregated = aggregate_cv_metrics(fold_rows)
        summary_rows: List[List[Any]] = []
        headers = ["Metric", "Mean", "Std", "CI_lower", "CI_upper"]
        for mk in METRIC_KEYS:
            block = aggregated.get(mk) or {}
            if block.get("mean") is None:
                continue
            summary_rows.append(
                [
                    mk,
                    round(block["mean"], 4),
                    round(block["std"] or 0, 4),
                    round(block["ci_lower"] or 0, 4),
                    round(block["ci_upper"] or 0, 4),
                ]
            )
        save_table(summary_rows, headers, os.path.join(mode_dir, f"summary_{mode}"))

        save_table(
            conv_rows,
            ["Fold", "Best epoch", "Loss stabilized epoch", "Total epochs"],
            os.path.join(mode_dir, f"convergence_{mode}"),
            caption=f"MFPT {mode} K-fold convergence",
        )

        fig_dir = os.path.join(mode_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        _plot_fold_accuracy(
            fold_rows,
            os.path.join(fig_dir, f"fold_accuracy_{mode}"),
            title=f"MFPT {mode} K-fold accuracy (k={k_eff})",
        )
        _plot_convergence_histories(
            histories,
            os.path.join(fig_dir, f"convergence_curves_{mode}"),
            title=f"MFPT {mode} training convergence",
        )

        box_png = os.path.join(fig_dir, f"cv_boxplot_{mode}.png")
        plot_cv_metric_boxplots(
            {f"{PLOT_MODEL_LABEL} ({mode})": fold_rows},
            metric_names=["accuracy", "f1", "precision", "recall", "auc_roc"],
            output_path=box_png,
            title=f"MFPT {mode} K-fold metrics (k={k_eff})",
        )

        all_results["modes"][mode] = {
            "fold_rows": fold_rows,
            "aggregated": aggregated,
            "verification_passed": all(v.get("passed") for v in verification),
        }

    return all_results


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MFPT K-fold CV (stratified and/or grouped)")
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--cv_mode", type=str, default="both", choices=["stratified", "grouped", "both"])
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init_checkpoint", type=str, default=None)
    p.add_argument("--model", type=str, default="resnet50")
    p.add_argument("--batch_size", type=int, default=32)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    out = args.output_dir or os.path.join(
        "results/mfpt/kfold_cv", datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_kfold_cv(
        args.dataset_path,
        out,
        cv_mode=args.cv_mode,
        k=args.k,
        epochs=args.epochs,
        seed=args.seed,
        init_checkpoint=args.init_checkpoint,
        model=args.model,
        batch_size=args.batch_size,
    )
    logger.info("MFPT K-fold CV complete: %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
