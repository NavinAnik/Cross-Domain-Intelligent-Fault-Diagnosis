"""
5-fold cross-validation for image classification models.

Run: python -m experiments.run_image_cv --model dinov3 [--epochs 50] [--n_splits 5]

Reports mean ± 95% CI for Precision, Recall, F1, AUC-ROC per fold.
"""

import logging
import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", ".env"))

from sklearn.model_selection import StratifiedKFold
from modules.statistical_tests import aggregate_cv_metrics

from run_all_models import run_single_model, list_models, ImageDataset

logger = logging.getLogger(__name__)


def run_image_cv(
    model_name: str = "dinov3",
    dataset_path: str = None,
    n_splits: int = 5,
    epochs: int = 50,
    batch_size: int = 32,
    random_state: int = 42,
) -> dict:
    """
    Run K-fold stratified CV for an image classification model.

    Returns:
        Dict with list_of_metrics, aggregated (mean ± CI), and per-fold results.
    """
    from experiments.run_all_models import _MODEL_REGISTRY, get_default_config

    dataset_path = dataset_path or os.getenv("DATASET_PATH", ".")
    model_name = model_name.lower()
    if model_name not in list_models():
        raise ValueError(f"Unknown model: {model_name}")

    _, data_getter, transform_getter = _MODEL_REGISTRY[model_name]
    if transform_getter is None:
        raise ValueError("run_image_cv supports image models only (e.g. dinov3, dinov2)")

    cfg = get_default_config(model_name)
    cfg["dataset_path"] = dataset_path
    cfg["epochs"] = epochs
    cfg["batch_size"] = batch_size
    cfg["output_dir"] = os.path.join(cfg.get("output_dir", "model"), "cv")
    cfg["results_dir"] = os.path.join(cfg.get("results_dir", "results"), "cv")

    get_transforms = transform_getter()
    transform_kwargs = {}
    if model_name in {"dinov2", "dinov3", "rtdetrv4_l", "rtdetrv4_l_distill", "eva02", "metaclip2", "siglip2"}:
        transform_kwargs["augment_strength"] = cfg.get("augment_strength", "strong")

    transforms_dict = get_transforms(cfg.get("image_size", 224), **transform_kwargs)
    train_ds = ImageDataset(dataset_path, transform=transforms_dict["train"])
    if len(train_ds.samples) == 0:
        raise FileNotFoundError(f"No images found under {dataset_path}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    indices = list(range(len(train_ds)))
    labels = train_ds.labels

    list_of_metrics = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
        logger.info("=" * 60)
        logger.info("Fold %d/%d", fold + 1, n_splits)
        logger.info("=" * 60)
        cfg_fold = {
            **cfg,
            "output_dir": os.path.join(cfg["output_dir"], f"fold{fold}"),
            "train_idx": train_idx.tolist() if hasattr(train_idx, "tolist") else list(train_idx),
            "val_idx": val_idx.tolist() if hasattr(val_idx, "tolist") else list(val_idx),
        }
        result = run_single_model(model_name, config=cfg_fold, run_name=f"{model_name}-cv-fold{fold}")
        m = {
            "accuracy": result.get("best_val_acc"),
            "macro_f1": result.get("best_val_macro_f1"),
            "precision": result.get("final_val_precision"),
            "recall": result.get("final_val_recall"),
            "f1": result.get("final_val_f1"),
        }
        ev = result.get("evaluation_metrics") or {}
        if ev.get("auc_roc") is not None:
            m["auc_roc"] = ev["auc_roc"]
        if ev.get("auc_pr") is not None:
            m["auc_pr"] = ev["auc_pr"]
        list_of_metrics.append(m)
        logger.info("Fold %d: acc=%.4f, macro_f1=%.4f", fold + 1, m.get("accuracy"), m.get("macro_f1"))

    aggregated = aggregate_cv_metrics(list_of_metrics, confidence=0.95)
    logger.info("--- CV aggregate (mean ± 95%% CI) ---")
    for name, vals in aggregated.items():
        if vals.get("ci_str"):
            logger.info("  %s: %s", name, vals["ci_str"])

    return {"list_of_metrics": list_of_metrics, "aggregated": aggregated}


def main():
    import argparse
    from utils.logging_config import setup_logging
    setup_logging(log_dir="logs")
    parser = argparse.ArgumentParser(description="5-fold CV for image models")
    parser.add_argument("--model", default="dinov3", choices=list_models(), help="Model name")
    parser.add_argument("--dataset_path", default=None, help="Dataset root")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of folds")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs per fold")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")
    args = parser.parse_args()
    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
    cfg = {"epochs": args.epochs}
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    run_image_cv(
        model_name=args.model,
        dataset_path=args.dataset_path,
        n_splits=args.n_splits,
        epochs=args.epochs,
        batch_size=args.batch_size or 32,
    )


if __name__ == "__main__":
    main()
