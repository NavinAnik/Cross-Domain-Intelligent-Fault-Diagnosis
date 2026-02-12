"""
Train and evaluate all 7 models individually (single-model experiments).

Usage:
  From project root (with src on PYTHONPATH) or from src/:
    python -m experiments.run_single_model_experiments [options]

  Options:
    --epochs N          Epochs per model (default from env or 20)
    --batch_size N      Batch size (default from env or 16)
    --dataset_path P    Dataset root (default from DATASET_PATH or ".")
    --output_dir D      Checkpoint directory (default "model")
    --results_dir D     Results/plots directory (default "results")
    --models m1,m2,...  Comma-separated list (default: all 7)
    --summary_file F    Write summary JSON/CSV here (default results/single_model_experiments_summary.json)
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_all_models import list_models, run_single_model


def run_all_single_model_experiments(
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    dataset_path: Optional[str] = None,
    output_dir: str = "model",
    results_dir: str = "results",
    model_names: Optional[List[str]] = None,
    summary_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train and evaluate each of the 7 models individually.
    Returns a dict of model_name -> result (metrics + any error).
    """
    if model_names is None:
        model_names = list_models()
    if summary_file is None:
        summary_file = os.path.join(results_dir, "single_model_experiments_summary.json")

    overrides: Dict[str, Any] = {
        "output_dir": output_dir,
        "results_dir": results_dir,
    }
    if epochs is not None:
        overrides["epochs"] = epochs
    if batch_size is not None:
        overrides["batch_size"] = batch_size
    if dataset_path is not None:
        overrides["dataset_path"] = dataset_path

    results: Dict[str, Any] = {}
    for i, model_name in enumerate(model_names):
        print(f"\n{'='*60}\n[{i+1}/{len(model_names)}] Training and evaluating: {model_name}\n{'='*60}")
        try:
            out = run_single_model(model_name, config=overrides, run_name=f"single-{model_name}")
            ev = out.get("evaluation_metrics") or {}
            ev_serializable = {}
            for k, v in ev.items():
                if k in ("cm", "report"):
                    if hasattr(v, "tolist"):
                        ev_serializable["confusion_matrix" if k == "cm" else k] = v.tolist()
                    continue
                if isinstance(v, (int, float, str, bool, type(None))):
                    ev_serializable[k] = v
                elif hasattr(v, "tolist"):
                    ev_serializable[k] = v.tolist()
            results[model_name] = {
                "best_val_acc": out.get("best_val_acc"),
                "final_val_acc": out.get("final_val_acc"),
                "evaluation_metrics": ev_serializable,
                "config": {
                    "epochs": out.get("config", {}).get("epochs"),
                    "batch_size": out.get("config", {}).get("batch_size"),
                    "dataset_path": out.get("config", {}).get("dataset_path"),
                },
            }
        except Exception as e:
            print(f"Model {model_name} failed: {e}")
            results[model_name] = {"error": str(e)}

    os.makedirs(os.path.dirname(summary_file) or ".", exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary written to {summary_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate all 7 single models (DINOv2, MAE, Swin, BEiT, DeiT, CLIP, AST)."
    )
    parser.add_argument("--epochs", type=int, default=None, help="Epochs per model")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--dataset_path", type=str, default=None, help="Dataset root")
    parser.add_argument("--output_dir", type=str, default="model", help="Checkpoint directory")
    parser.add_argument("--results_dir", type=str, default="results", help="Results directory")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names (default: all). e.g. dinov2,mae,ast",
    )
    parser.add_argument(
        "--summary_file",
        type=str,
        default=None,
        help="Output summary JSON path (default: <results_dir>/single_model_experiments_summary.json)",
    )
    args = parser.parse_args()

    model_list = None
    if args.models:
        model_list = [m.strip().lower() for m in args.models.split(",")]
        valid = list_models()
        for m in model_list:
            if m not in valid:
                raise SystemExit(f"Unknown model: {m}. Choose from {valid}")

    run_all_single_model_experiments(
        epochs=args.epochs,
        batch_size=args.batch_size,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        model_names=model_list,
        summary_file=args.summary_file,
    )


if __name__ == "__main__":
    main()
