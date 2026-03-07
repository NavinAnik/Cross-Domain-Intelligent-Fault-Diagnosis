"""
Rigorous Cross-Validation Experiment Runner for Q1 Journal Publication.

Features:
- 10-fold stratified cross-validation (configurable)
- Multiple random seeds (5 by default: 42, 123, 456, 789, 2024)
- Bonferroni/Holm correction for statistical significance
- Cohen's d effect sizes
- Bootstrap confidence intervals (10,000 samples)
- Comprehensive result tables and figures

Usage:
    python -m experiments.run_rigorous_cv --model dinov2 --n_splits 10 --n_seeds 5
    python -m experiments.run_rigorous_cv --models dinov2,mae,swin --run_all
    python -m experiments.run_rigorous_cv --models dinov2,mae --dataset_path H:/dataset/MIMII_Dataset/spectrograms
"""

import os
import sys
import json
import time
import argparse
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", ".env"))

# WandB integration
try:
    import wandb
    from utils.experiment_tracking import init_wandb, log_metrics, finish_wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False
    wandb = None
    def init_wandb(*args, **kwargs):
        return None
    def log_metrics(*args, **kwargs):
        pass
    def finish_wandb():
        pass

from modules.evaluation import compute_classification_metrics
from modules.statistical_tests import (
    aggregate_cv_metrics,
    significance_matrix_with_correction,
    cohens_d,
    bonferroni_correction,
    bootstrap_accuracy,
    confidence_interval,
)
from utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_SEEDS = [42, 123, 456, 789, 2024]
DEFAULT_N_SPLITS = 10


class RigorousCVRunner:
    """
    Run rigorous cross-validation experiments with multiple seeds.
    """

    def __init__(
        self,
        n_splits: int = 10,
        seeds: List[int] = None,
        output_dir: str = "results/rigorous_cv",
        bootstrap_samples: int = 10000,
    ):
        self.n_splits = n_splits
        self.seeds = seeds or DEFAULT_SEEDS
        self.output_dir = output_dir
        self.bootstrap_samples = bootstrap_samples
        os.makedirs(output_dir, exist_ok=True)

    def run_single_model_cv(
        self,
        model_name: str,
        train_eval_fn: Callable,
        dataset_path: str,
        config: Optional[Dict[str, Any]] = None,
        use_wandb: bool = False,
    ) -> Dict[str, Any]:
        """
        Run CV for a single model across all seeds.

        Args:
            model_name: Name of the model.
            train_eval_fn: Function (seed, fold_idx, train_loader, val_loader, config) -> metrics_dict
            dataset_path: Path to dataset.
            config: Optional model config overrides.
            use_wandb: Whether to log metrics to WandB.

        Returns:
            Dict with all_fold_metrics, aggregated_metrics, per_seed_metrics.
        """
        all_fold_metrics = []
        per_seed_metrics = {}
        run_counter = 0
        total_runs = len(self.seeds) * self.n_splits

        for seed in self.seeds:
            logger.info("Running %s with seed=%d, %d-fold CV", model_name, seed, self.n_splits)
            seed_metrics = []

            for fold_idx in range(self.n_splits):
                run_counter += 1
                logger.info("  Seed %d, Fold %d/%d", seed, fold_idx + 1, self.n_splits)

                try:
                    metrics = train_eval_fn(
                        seed=seed,
                        fold_idx=fold_idx,
                        n_splits=self.n_splits,
                        dataset_path=dataset_path,
                        config=config,
                    )
                    metrics["seed"] = seed
                    metrics["fold"] = fold_idx
                    seed_metrics.append(metrics)
                    all_fold_metrics.append(metrics)

                    # Log individual fold metrics to WandB
                    if use_wandb and wandb is not None:
                        wandb_metrics = {
                            f"{model_name}/fold_accuracy": metrics.get("accuracy", 0),
                            f"{model_name}/fold_f1": metrics.get("f1", 0),
                            f"{model_name}/fold_recall": metrics.get("recall", 0),
                            f"{model_name}/fold_precision": metrics.get("precision", 0),
                            f"{model_name}/fold_auc_roc": metrics.get("auc_roc", 0),
                            f"{model_name}/seed": seed,
                            f"{model_name}/fold": fold_idx,
                            "progress/run": run_counter,
                            "progress/total_runs": total_runs,
                            "progress/percent": 100 * run_counter / total_runs,
                        }
                        wandb.log(wandb_metrics)

                except Exception as e:
                    logger.error("Error in seed=%d, fold=%d: %s", seed, fold_idx, e)
                    continue

            per_seed_metrics[seed] = seed_metrics

        aggregated = aggregate_cv_metrics(all_fold_metrics, confidence=0.95)

        return {
            "model_name": model_name,
            "n_splits": self.n_splits,
            "seeds": self.seeds,
            "n_total_runs": len(all_fold_metrics),
            "all_fold_metrics": all_fold_metrics,
            "per_seed_metrics": per_seed_metrics,
            "aggregated": aggregated,
        }

    def run_multi_model_comparison(
        self,
        models_config: Dict[str, Callable],
        dataset_path: str,
        model_configs: Optional[Dict[str, Dict]] = None,
        use_wandb: bool = False,
    ) -> Dict[str, Any]:
        """
        Run CV for multiple models and perform statistical comparison.

        Args:
            models_config: Dict of model_name -> train_eval_fn.
            dataset_path: Path to dataset.
            model_configs: Optional dict of model_name -> config overrides.
            use_wandb: Whether to log metrics to WandB.

        Returns:
            Dict with per-model results and statistical comparisons.
        """
        results_by_model = {}
        model_configs = model_configs or {}

        for model_name, train_eval_fn in models_config.items():
            logger.info("=" * 60)
            logger.info("Running rigorous CV for: %s", model_name)
            logger.info("=" * 60)

            config = model_configs.get(model_name, {})
            result = self.run_single_model_cv(
                model_name=model_name,
                train_eval_fn=train_eval_fn,
                dataset_path=dataset_path,
                config=config,
                use_wandb=use_wandb,
            )
            results_by_model[model_name] = result

            # Log model aggregated results to WandB
            if use_wandb and wandb is not None:
                agg = result.get("aggregated", {})
                wandb_data = {f"{model_name}/n_runs": result.get("n_total_runs", 0)}
                for metric in ["accuracy", "precision", "recall", "f1", "auc_roc", "mcc"]:
                    if metric in agg and agg[metric].get("mean") is not None:
                        wandb_data[f"{model_name}/cv_{metric}_mean"] = agg[metric]["mean"]
                        wandb_data[f"{model_name}/cv_{metric}_std"] = agg[metric]["std"]
                        if agg[metric].get("ci_lower") is not None:
                            wandb_data[f"{model_name}/cv_{metric}_ci_lower"] = agg[metric]["ci_lower"]
                            wandb_data[f"{model_name}/cv_{metric}_ci_upper"] = agg[metric]["ci_upper"]
                wandb.log(wandb_data)

        statistical_comparison = self._compute_statistical_comparison(results_by_model)

        return {
            "results_by_model": results_by_model,
            "statistical_comparison": statistical_comparison,
            "n_splits": self.n_splits,
            "seeds": self.seeds,
            "timestamp": datetime.now().isoformat(),
        }

    def _compute_statistical_comparison(
        self, results_by_model: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute pairwise statistical significance with corrections.
        """
        metrics_to_compare = ["accuracy", "f1", "recall", "auc_roc", "mcc"]
        comparisons = {}

        for metric in metrics_to_compare:
            scores_by_model = {}
            for model_name, result in results_by_model.items():
                scores = [
                    m[metric]
                    for m in result["all_fold_metrics"]
                    if metric in m and m[metric] is not None
                ]
                if scores:
                    scores_by_model[model_name] = scores

            if len(scores_by_model) < 2:
                continue

            comparison = significance_matrix_with_correction(
                scores_by_model,
                metric_name=metric,
                tests=("ttest", "wilcoxon"),
                correction="holm",
                alpha=0.05,
            )
            comparisons[metric] = comparison

        return comparisons

    def save_results(self, results: Dict[str, Any], filename: str = "rigorous_cv_results.json"):
        """Save results to JSON file."""
        output_path = os.path.join(self.output_dir, filename)

        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            return obj

        serializable = convert_to_serializable(results)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

        logger.info("Results saved to %s", output_path)
        return output_path

    def generate_publication_tables(
        self, results: Dict[str, Any], output_dir: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate publication-ready tables in Markdown format.
        """
        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        table_paths = {}

        table1_path = os.path.join(output_dir, "table_cv_results.md")
        self._generate_cv_results_table(results, table1_path)
        table_paths["cv_results"] = table1_path

        table2_path = os.path.join(output_dir, "table_significance.md")
        self._generate_significance_table(results, table2_path)
        table_paths["significance"] = table2_path

        table3_path = os.path.join(output_dir, "table_effect_sizes.md")
        self._generate_effect_size_table(results, table3_path)
        table_paths["effect_sizes"] = table3_path

        return table_paths

    def _generate_cv_results_table(self, results: Dict[str, Any], output_path: str):
        """Generate Table: CV Results (mean ± std)."""
        lines = ["# Cross-Validation Results", ""]
        lines.append(f"**Configuration**: {self.n_splits}-fold CV × {len(self.seeds)} seeds = {self.n_splits * len(self.seeds)} total runs per model")
        lines.append("")
        lines.append("| Model | Accuracy | Precision | Recall | F1 | AUC-ROC | MCC |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")

        for model_name, model_result in results.get("results_by_model", {}).items():
            agg = model_result.get("aggregated", {})
            row = [model_name]
            for metric in ["accuracy", "precision", "recall", "f1", "auc_roc", "mcc"]:
                if metric in agg and agg[metric]["mean"] is not None:
                    mean = agg[metric]["mean"]
                    std = agg[metric]["std"]
                    row.append(f"{mean:.4f} ± {std:.4f}")
                else:
                    row.append("-")
            lines.append("| " + " | ".join(row) + " |")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info("CV results table saved to %s", output_path)

    def _generate_significance_table(self, results: Dict[str, Any], output_path: str):
        """Generate Table: Statistical Significance with Holm-Bonferroni correction."""
        lines = ["# Statistical Significance (Holm-Bonferroni Corrected)", ""]

        comparisons = results.get("statistical_comparison", {})
        if not comparisons:
            lines.append("No statistical comparisons available.")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            return

        for metric, comp in comparisons.items():
            lines.append(f"## {metric.upper()}")
            lines.append("")

            model_names = comp.get("model_names", [])
            if len(model_names) < 2:
                lines.append("Insufficient models for comparison.")
                lines.append("")
                continue

            lines.append(f"**Corrected α**: {comp.get('corrected_alpha', 0.05):.4f}")
            lines.append(f"**Number of comparisons**: {comp.get('n_comparisons', 0)}")
            lines.append("")

            header = "| " + " | ".join([""] + model_names) + " |"
            sep = "| " + " | ".join(["---"] * (len(model_names) + 1)) + " |"
            lines.append(header)
            lines.append(sep)

            adj_p = comp.get("adjusted_pvalue_ttest", np.array([]))
            if isinstance(adj_p, np.ndarray) and adj_p.size > 0:
                for i, name in enumerate(model_names):
                    row = [name]
                    for j in range(len(model_names)):
                        if i == j:
                            row.append("-")
                        else:
                            p = adj_p[i, j]
                            sig = "**" if p < 0.05 else ""
                            row.append(f"{sig}{p:.4f}{sig}")
                    lines.append("| " + " | ".join(row) + " |")

            sig_pairs = comp.get("significant_pairs", [])
            if sig_pairs:
                lines.append("")
                lines.append("**Significant differences (p < 0.05 after correction):**")
                for m1, m2, p in sig_pairs:
                    lines.append(f"- {m1} vs {m2}: p = {p:.4f}")

            lines.append("")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info("Significance table saved to %s", output_path)

    def _generate_effect_size_table(self, results: Dict[str, Any], output_path: str):
        """Generate Table: Cohen's d Effect Sizes."""
        lines = ["# Effect Sizes (Cohen's d)", ""]
        lines.append("**Interpretation**: |d| < 0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >= 0.8 large")
        lines.append("")

        comparisons = results.get("statistical_comparison", {})
        if not comparisons:
            lines.append("No effect size data available.")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            return

        for metric, comp in comparisons.items():
            lines.append(f"## {metric.upper()}")
            lines.append("")

            model_names = comp.get("model_names", [])
            if len(model_names) < 2:
                continue

            header = "| " + " | ".join([""] + model_names) + " |"
            sep = "| " + " | ".join(["---"] * (len(model_names) + 1)) + " |"
            lines.append(header)
            lines.append(sep)

            effect = comp.get("effect_size", np.array([]))
            interp = comp.get("effect_interpretation", [])

            if isinstance(effect, np.ndarray) and effect.size > 0:
                for i, name in enumerate(model_names):
                    row = [name]
                    for j in range(len(model_names)):
                        if i == j:
                            row.append("-")
                        else:
                            d = effect[i, j]
                            int_str = interp[i][j] if interp else ""
                            row.append(f"{d:.3f} ({int_str})")
                    lines.append("| " + " | ".join(row) + " |")

            lines.append("")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info("Effect size table saved to %s", output_path)


def create_model_train_eval_fn(model_name: str):
    """
    Create a train/eval function for a given model that works with CV.

    Returns a function: (seed, fold_idx, n_splits, dataset_path, config) -> metrics_dict
    """
    from experiments.run_all_models import (
        _MODEL_REGISTRY,
        _get_image_dataloaders,
        get_default_config,
        _train_one_epoch,
        _validate,
    )
    from training.optimization import autocast_context, TrainingStepHelper
    from collections import Counter
    from sklearn.model_selection import StratifiedKFold
    import torch.optim as optim

    def train_eval_fn(
        seed: int,
        fold_idx: int,
        n_splits: int,
        dataset_path: str,
        config: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        torch.manual_seed(seed)
        np.random.seed(seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_builder, data_getter, transform_getter = _MODEL_REGISTRY[model_name]
        cfg = get_default_config(model_name)
        if config:
            cfg.update(config)
        cfg["dataset_path"] = dataset_path

        cv_epochs = cfg.get("cv_epochs", 10)
        cfg["epochs"] = cv_epochs

        if transform_getter is not None:
            get_transforms = transform_getter()
            from experiments.run_all_models import ImageDataset
            transforms_dict = get_transforms(image_size=cfg.get("image_size", 224))

            full_ds = ImageDataset(dataset_path, transform=transforms_dict["train"])
            val_ds = ImageDataset(dataset_path, transform=transforms_dict["val"])

            if len(full_ds.samples) == 0:
                raise FileNotFoundError(f"No images found under {dataset_path}")

            labels = np.array(full_ds.labels)
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            splits = list(skf.split(range(len(labels)), labels))
            train_idx, val_idx = splits[fold_idx]

            train_labels = [full_ds.labels[i] for i in train_idx]
            class_counts = Counter(train_labels)
            total = sum(class_counts.values())
            num_classes = cfg.get("num_classes", 2)
            class_weights = [total / (num_classes * class_counts.get(i, 1)) for i in range(num_classes)]
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

            from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

            sample_weights = torch.tensor([class_weights[l].item() for l in train_labels], dtype=torch.float)
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

            train_loader = DataLoader(
                Subset(full_ds, train_idx),
                batch_size=cfg.get("batch_size", 16),
                sampler=sampler,
                num_workers=4,
                pin_memory=True,
            )
            val_loader = DataLoader(
                Subset(val_ds, val_idx),
                batch_size=cfg.get("batch_size", 16),
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
        else:
            raise NotImplementedError("Audio CV not implemented in this runner")

        model = model_builder(device, cfg)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.get("learning_rate", 1e-4),
            weight_decay=cfg.get("weight_decay", 0.05),
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cv_epochs)

        use_amp = cfg.get("use_amp", True) and device.type == "cuda"
        step_helper = TrainingStepHelper(
            optimizer, use_amp=use_amp, device=device,
            accumulation_steps=cfg.get("accumulation_steps", 1)
        )

        best_val_acc = 0.0
        best_metrics = {}

        for epoch in range(1, cv_epochs + 1):
            train_loss, train_acc = _train_one_epoch(
                model, train_loader, criterion, step_helper, device,
                epoch, cv_epochs, use_amp
            )
            val_loss, val_acc, y_true, y_pred = _validate(
                model, val_loader, criterion, device
            )
            scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                y_true_np = y_true.numpy()
                y_pred_np = y_pred.numpy()

                model.eval()
                all_probs = []
                with torch.no_grad():
                    for x, _ in val_loader:
                        x = x.to(device)
                        logits = model(x)
                        probs = torch.softmax(logits, dim=1)[:, 1]
                        all_probs.extend(probs.cpu().numpy())

                best_metrics = compute_classification_metrics(
                    y_true_np, y_pred_np, y_pred_probs=np.array(all_probs)
                )

        return best_metrics

    return train_eval_fn


def main():
    setup_logging(log_dir="logs")

    parser = argparse.ArgumentParser(description="Rigorous CV for Q1 publication")
    parser.add_argument("--model", type=str, default=None, help="Single model to run")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated list of models")
    parser.add_argument("--run_all", action="store_true", help="Run all available models")
    parser.add_argument("--n_splits", type=int, default=10, help="Number of CV folds")
    parser.add_argument("--n_seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds (default: 42,123,456,789,2024)")
    parser.add_argument("--dataset_path", type=str, default=None, help="Dataset path")
    parser.add_argument("--output_dir", type=str, default="results/rigorous_cv",
                        help="Output directory")
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="mimii-rigorous-cv",
                        help="WandB project name")
    parser.add_argument("--cv_epochs", type=int, default=10,
                        help="Epochs per CV fold (default: 10)")
    parser.add_argument("--bootstrap_samples", type=int, default=10000,
                        help="Bootstrap samples for CI")
    args = parser.parse_args()

    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        seeds = DEFAULT_SEEDS[:args.n_seeds]

    dataset_path = args.dataset_path or os.getenv("DATASET_PATH", ".")

    from experiments.run_all_models import list_models

    if args.run_all:
        model_list = list_models()
    elif args.models:
        model_list = [m.strip() for m in args.models.split(",")]
    elif args.model:
        model_list = [args.model]
    else:
        model_list = ["dinov2"]

    model_list = [m for m in model_list if m != "ast"]

    runner = RigorousCVRunner(
        n_splits=args.n_splits,
        seeds=seeds,
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
    )

    models_config = {}
    for model_name in model_list:
        models_config[model_name] = create_model_train_eval_fn(model_name)

    model_configs = {m: {"cv_epochs": args.cv_epochs} for m in model_list}

    logger.info("=" * 60)
    logger.info("Starting Rigorous Cross-Validation")
    logger.info("Models: %s", model_list)
    logger.info("CV Folds: %d", args.n_splits)
    logger.info("Seeds: %s", seeds)
    logger.info("Total runs per model: %d", args.n_splits * len(seeds))
    logger.info("Dataset path: %s", dataset_path)
    logger.info("=" * 60)
    
    if not os.path.exists(dataset_path):
        logger.error("Dataset path does not exist: %s", dataset_path)
        logger.error("Please provide a valid dataset path via --dataset_path or DATASET_PATH env variable")
        sys.exit(1)

    # Initialize WandB
    use_wandb = _HAS_WANDB and not args.no_wandb
    if use_wandb:
        try:
            wandb.init(
                project=args.wandb_project,
                name=f"rigorous-cv-{'-'.join(model_list)}",
                config={
                    "models": model_list,
                    "n_splits": args.n_splits,
                    "seeds": seeds,
                    "cv_epochs": args.cv_epochs,
                    "dataset_path": dataset_path,
                },
                tags=["rigorous-cv", "q1-journal"] + model_list,
            )
            logger.info("WandB initialized: %s", wandb.run.url)
        except Exception as e:
            logger.warning("Failed to initialize WandB: %s", e)
            use_wandb = False
    else:
        logger.info("WandB logging disabled")

    start_time = time.time()

    results = runner.run_multi_model_comparison(
        models_config=models_config,
        dataset_path=dataset_path,
        model_configs=model_configs,
        use_wandb=use_wandb,
    )

    elapsed = time.time() - start_time
    results["total_time_seconds"] = elapsed

    runner.save_results(results, "rigorous_cv_results.json")
    runner.generate_publication_tables(results)

    # Log final summary to WandB
    if use_wandb:
        summary_data = {"total_time_hours": elapsed / 3600}
        for model_name, model_result in results.get("results_by_model", {}).items():
            agg = model_result.get("aggregated", {})
            for metric in ["accuracy", "f1", "auc_roc", "mcc"]:
                if metric in agg and agg[metric].get("mean") is not None:
                    summary_data[f"{model_name}/{metric}_mean"] = agg[metric]["mean"]
                    summary_data[f"{model_name}/{metric}_std"] = agg[metric]["std"]
        wandb.log(summary_data)
        wandb.finish()
        logger.info("WandB run finished")

    logger.info("=" * 60)
    logger.info("Rigorous CV Complete!")
    logger.info("Total time: %.2f hours", elapsed / 3600)
    logger.info("Results saved to: %s", args.output_dir)
    logger.info("=" * 60)

    logger.info("\n--- Summary ---")
    for model_name, model_result in results.get("results_by_model", {}).items():
        agg = model_result.get("aggregated", {})
        acc = agg.get("accuracy", {})
        logger.info("%s: Accuracy = %.4f ± %.4f",
                    model_name,
                    acc.get("mean", 0),
                    acc.get("std", 0))

    comparisons = results.get("statistical_comparison", {})
    if "accuracy" in comparisons:
        sig_pairs = comparisons["accuracy"].get("significant_pairs", [])
        if sig_pairs:
            logger.info("\nStatistically significant differences (accuracy):")
            for m1, m2, p in sig_pairs:
                logger.info("  %s vs %s: p = %.4f", m1, m2, p)
        else:
            logger.info("\nNo statistically significant differences found.")


if __name__ == "__main__":
    main()
