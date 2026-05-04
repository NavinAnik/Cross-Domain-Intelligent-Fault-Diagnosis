"""
Ablation grid: model x augmentation x loss for spectrogram fault classification.

Writes ablation_results.csv and ablation_results.json under output_dir.

Usage (from src/):
  python -m experiments.run_spectrogram_ablation --dataset_path %DATASET_PATH% --quick
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", ".env"))

from utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def _namespace_from_base(base: argparse.Namespace, **overrides) -> argparse.Namespace:
    d = vars(base).copy()
    d.update(overrides)
    return argparse.Namespace(**d)


def main():
    setup_logging(log_dir="logs")
    p = argparse.ArgumentParser(description="Spectrogram fault ablation grid")
    p.add_argument("--dataset_path", type=str, default=os.getenv("DATASET_PATH", "."))
    p.add_argument(
        "--dataset_subdir",
        type=str,
        default=os.getenv("DATASET_SUBDIR") or None,
        help="Optional child of dataset_path (e.g. 0dB for one MIMII SNR)",
    )
    p.add_argument("--output_dir", type=str, default="results/spectrogram_fault_ablation")
    p.add_argument("--quick", action="store_true", help="Fewer epochs for smoke test")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--backbone", type=str, default="resnet50")
    p.add_argument("--num_workers", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models", type=str, default=None, help="Comma-separated; default full grid")
    p.add_argument("--augments", type=str, default=None)
    p.add_argument("--losses", type=str, default=None)
    args = p.parse_args()

    from experiments.train_spectrogram_fault import parse_args as train_parse, train_one_run

    base = train_parse([])
    base.dataset_path = args.dataset_path
    base.dataset_subdir = args.dataset_subdir
    base.output_dir = args.output_dir
    base.epochs = 5 if args.quick else args.epochs
    base.batch_size = args.batch_size
    base.lr = args.lr
    base.image_size = args.image_size
    base.backbone = args.backbone
    base.num_workers = args.num_workers
    base.seed = args.seed
    base.patience = 8 if args.quick else base.patience

    default_models = ["resnet50", "efficientnet_b0", "cnn_se", "cnn_transformer"]
    default_augments = ["off", "strong"]
    default_losses = ["ce_weighted", "focal"]

    models = default_models if not args.models else [x.strip() for x in args.models.split(",")]
    augments = default_augments if not args.augments else [x.strip() for x in args.augments.split(",")]
    losses = default_losses if not args.losses else [x.strip() for x in args.losses.split(",")]

    os.makedirs(args.output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows: List[Dict[str, Any]] = []

    for m in models:
        for aug in augments:
            for loss in losses:
                run_id = f"{stamp}_{m}_aug-{aug}_loss-{loss}"
                ns = _namespace_from_base(
                    base,
                    model=m,
                    augment=aug,
                    loss=loss,
                    run_id=run_id,
                    output_dir=args.output_dir,
                )
                logger.info("=== Ablation run: model=%s augment=%s loss=%s ===", m, aug, loss)
                try:
                    out = train_one_run(ns)
                    met = out.get("metrics", {})
                    row = {
                        "model": m,
                        "augment": aug,
                        "loss": loss,
                        "out_dir": out.get("out_dir", ""),
                        "accuracy": met.get("accuracy"),
                        "macro_f1": met.get("macro_f1"),
                        "balanced_accuracy": met.get("val_balanced_accuracy"),
                        "auc_roc": met.get("auc_roc"),
                        "mcc": met.get("mcc"),
                        "train_time_sec": met.get("train_time_sec"),
                    }
                except Exception as e:
                    logger.exception("Run failed: %s", e)
                    row = {
                        "model": m,
                        "augment": aug,
                        "loss": loss,
                        "error": str(e),
                    }
                rows.append(row)

    csv_path = os.path.join(args.output_dir, f"ablation_results_{stamp}.csv")
    if rows:
        keys = sorted({k for r in rows for k in r.keys()})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        logger.info("Wrote %s", csv_path)

    json_path = os.path.join(args.output_dir, f"ablation_results_{stamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str)
    logger.info("Wrote %s", json_path)


if __name__ == "__main__":
    main()
