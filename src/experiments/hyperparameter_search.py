"""
Hyperparameter search for models and ablations.

Supports grid search and random search over key hyperparameters.
Results are logged and can be consumed by comparative_analysis.
"""

import os
import sys
import argparse
import json
import itertools
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", ".env"))


def grid_search_models(
    model_names: List[str],
    param_grid: Dict[str, List[Any]],
    dataset_path: Optional[str] = None,
    output_dir: str = "experiments/hp_search",
    max_runs: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Run grid search over param_grid for each model in model_names.

    param_grid example:
        {"epochs": [10, 20], "learning_rate": [1e-4, 5e-4], "batch_size": [8, 16]}
    Generates 2*2*2 = 8 configs per model. Results appended to output_dir/results.json.
    """
    from .run_all_models import run_single_model, list_models

    dataset_path = dataset_path or os.getenv("DATASET_PATH", ".")
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "results.json")
    all_results: List[Dict[str, Any]] = []
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    configs = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    if max_runs is not None:
        configs = configs[:max_runs]

    for model_name in model_names:
        if model_name not in list_models():
            print(f"Skip unknown model: {model_name}")
            continue
        for i, config in enumerate(configs):
            run_name = f"{model_name}-hp-{i}"
            print(f"\n{'='*60}\nGrid run: {model_name} config {config}\n{'='*60}")
            try:
                config_with_path = {**config, "dataset_path": dataset_path}
                result = run_single_model(
                    model_name,
                    config=config_with_path,
                    run_name=run_name,
                )
                record = {
                    "model": model_name,
                    "config": config,
                    "best_val_acc": result.get("best_val_acc"),
                    "final_val_acc": result.get("final_val_acc"),
                }
                all_results.append(record)
            except Exception as e:
                print(f"Run failed: {e}")
                all_results.append({"model": model_name, "config": config, "error": str(e)})

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    return all_results


def random_search_models(
    model_names: List[str],
    param_distributions: Dict[str, List[Any]],
    n_iter: int = 10,
    dataset_path: Optional[str] = None,
    output_dir: str = "experiments/hp_search",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Random search: sample n_iter configs from param_distributions per model.
    """
    import random
    random.seed(seed)
    keys = list(param_distributions.keys())
    all_results = []
    dataset_path = dataset_path or os.getenv("DATASET_PATH", ".")
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)

    from .run_all_models import run_single_model, list_models

    for model_name in model_names:
        if model_name not in list_models():
            continue
        for i in range(n_iter):
            config = {k: random.choice(param_distributions[k]) for k in keys}
            config["dataset_path"] = dataset_path
            run_name = f"{model_name}-rs-{i}"
            try:
                result = run_single_model(model_name, config=config, run_name=run_name)
                all_results.append({
                    "model": model_name,
                    "config": config,
                    "best_val_acc": result.get("best_val_acc"),
                    "final_val_acc": result.get("final_val_acc"),
                })
            except Exception as e:
                all_results.append({"model": model_name, "config": config, "error": str(e)})

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search")
    parser.add_argument("--mode", type=str, default="grid", choices=["grid", "random"])
    parser.add_argument("--models", type=str, nargs="+", default=["dinov2", "mae"],
                        help="Models to search")
    parser.add_argument("--epochs", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--lr", type=float, nargs="+", default=[1e-4, 5e-4])
    parser.add_argument("--batch_size", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="experiments/hp_search")
    parser.add_argument("--n_iter", type=int, default=8, help="Random search iterations")
    parser.add_argument("--max_runs", type=int, default=None, help="Cap total runs (grid)")
    args = parser.parse_args()

    param_grid = {
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
    }
    if args.mode == "grid":
        grid_search_models(
            args.models,
            param_grid,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            max_runs=args.max_runs,
        )
    else:
        random_search_models(
            args.models,
            param_grid,
            n_iter=args.n_iter,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
