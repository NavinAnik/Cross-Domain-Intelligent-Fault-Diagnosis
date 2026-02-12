"""
Run cross-validation and statistical significance analysis for publication.

Use this script to:
1. Run 5-fold cross-validation for one or more models (with a provided train/eval callback),
   or
2. Load saved CV results (per-fold metrics per model) and run statistical analysis:
   - 95% confidence intervals (mean ± CI) per model and metric
   - Pairwise significance matrix (paired t-test and Wilcoxon signed-rank)
   - Box plots of metric distributions across folds

Expected saved results format (e.g. cv_results.json):
  - Option A: List of {"model_name": "DINOv2", "cv_metrics": [dict_fold0, dict_fold1, ...]}
  - Option B: Dict {"DINOv2": [dict_fold0, ...], "MAE": [...], ...}
"""

import os
import sys
import json
import argparse
from typing import Any, Dict, List

# Allow importing from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_loader import get_audio_dataloaders_cv
from modules.evaluation import compute_classification_metrics
from modules.statistical_tests import aggregate_cv_metrics, significance_matrix
from evaluate_models import (
    run_cross_validation,
    run_statistical_analysis,
    print_cv_metrics_with_ci,
)


def load_cv_results(path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load metrics_by_model from a JSON file.

    Accepts:
      - List of {"model_name": str, "cv_metrics": [dict, ...]} -> merged by model_name
      - Dict of model_name -> [dict, ...]
    """
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        # Assume keys are model names, values are list of metric dicts
        return data
    if isinstance(data, list):
        out = {}
        for item in data:
            name = item.get("model_name") or item.get("model", "unknown")
            metrics_list = item.get("cv_metrics") or item.get("per_fold_metrics") or []
            if name not in out:
                out[name] = []
            out[name].extend(metrics_list)
        return out
    raise ValueError("JSON must be a list or dict")


def save_cv_results(metrics_by_model: Dict[str, List[Dict[str, Any]]], path: str) -> None:
    """Save metrics_by_model as JSON (list format for compatibility)."""
    data = [
        {"model_name": name, "cv_metrics": per_fold}
        for name, per_fold in metrics_by_model.items()
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def run_analysis_from_file(
    input_path: str,
    output_dir: str = None,
    confidence: float = 0.95,
) -> None:
    """
    Load CV results from JSON and run full statistical analysis (CI, significance, box plots).
    """
    metrics_by_model = load_cv_results(input_path)
    output_dir = output_dir or os.path.dirname(os.path.abspath(input_path))
    run_statistical_analysis(
        metrics_by_model,
        output_dir=output_dir,
        confidence=confidence,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Cross-validation and statistical significance analysis"
    )
    parser.add_argument(
        "mode",
        choices=["analyze", "cv"],
        help="analyze: load saved CV results and run statistical analysis; "
             "cv: run 5-fold CV (requires train_eval_fn in code for your model)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input JSON with CV results (for mode=analyze). Format: list of "
             "{model_name, cv_metrics} or dict model_name -> [per_fold_metrics].",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for tables and figures (default: results).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for intervals (default: 0.95).",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of CV folds when mode=cv (default: 5).",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Dataset root path (for mode=cv).",
    )
    args = parser.parse_args()

    if args.mode == "analyze":
        if not args.input or not os.path.isfile(args.input):
            print("For mode=analyze, provide --input path to a JSON file with CV results.")
            sys.exit(1)
        run_analysis_from_file(
            args.input,
            output_dir=args.output_dir,
            confidence=args.confidence,
        )
        return

    if args.mode == "cv":
        print(
            "mode=cv runs 5-fold CV via run_cross_validation(). "
            "You must call run_cross_validation() from your own script with a "
            "train_eval_fn that trains and returns a metrics dict per fold. "
            "Use get_audio_dataloaders_cv(dataset_path, fold_index, n_splits=5) for loaders."
        )
        if args.dataset_path and os.path.isdir(args.dataset_path):
            # Demo: run one fold with a dummy train_eval that just returns placeholder metrics
            def dummy_train_eval(fold_idx, train_loader, val_loader):
                return {
                    "accuracy": 0.5,
                    "precision": 0.5,
                    "recall": 0.5,
                    "f1": 0.5,
                    "mcc": 0.0,
                    "cohen_kappa": 0.0,
                    "auc_roc": 0.5,
                    "auc_pr": 0.5,
                }
            list_of_metrics, aggregated = run_cross_validation(
                args.dataset_path,
                dummy_train_eval,
                n_splits=min(2, args.n_splits),
                random_state=42,
            )
            print_cv_metrics_with_ci(list_of_metrics, confidence=args.confidence)
            print("\n(Demo used dummy metrics; plug in your train_eval_fn for real CV.)")
        return


if __name__ == "__main__":
    main()
