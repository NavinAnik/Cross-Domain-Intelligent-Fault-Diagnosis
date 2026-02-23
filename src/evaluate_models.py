#generate confusion matrix and classification report
import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

import pandas as pd
import seaborn as sns
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    roc_curve,
    confusion_matrix,
    precision_recall_fscore_support,
    ConfusionMatrixDisplay,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    auc,
    matthews_corrcoef,
    cohen_kappa_score,
)
from modules.data_loader import get_validation_audio_loader, get_audio_dataloaders_cv
from modules.evaluation import compute_classification_metrics
from modules.statistical_tests import (
    confidence_interval,
    confidence_interval_str,
    paired_ttest,
    wilcoxon_signed_rank,
    aggregate_cv_metrics,
    significance_matrix,
    chi_square_test,
    bootstrap_accuracy,
)
from modules.visualization import (
    plot_tsne_umap,
    plot_radar_chart,
    plot_radar_from_metric_dicts,
    plot_attention_map,
    plot_grad_cam,
    plot_cv_metric_boxplots,
)


#Confusion matrix function
def plot_confusion_matrix(y_true, y_pred, epoch,out_dir):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.savefig(f'{out_dir}/conf_matrix_epoch_{epoch}.png')
    plt.close()
    return cm

# Function to plot all epoch precission curve in one graph
def plot_all_epochs_precision_curve(y_true, y_pred_probs, epochs):
    plt.figure(figsize=(10, 6))
    for epoch in epochs:
        precision, recall, _ = precision_recall_curve(y_true, y_pred_probs[:, 1])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'Epoch {epoch} (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - All Epochs')
    plt.legend(loc='lower left')
    plt.savefig(f'pr_curve_all_epochs.png')
    plt.close()

def plot_roc_curve(y_true, y_pred_probs, epoch,out_dir):
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Epoch {epoch}')
    plt.legend(loc='lower right')
    plt.savefig(f'{out_dir}/roc_epoch_{epoch}.png')
    plt.close()
    return roc_auc

def plot_precision_recall_curve(y_true, y_pred_probs, epoch,out_dir):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label=f'Precision-Recall Curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - Epoch {epoch}')
    plt.legend(loc='lower left')
    plt.savefig(f'{out_dir}/pr_curve_epoch_{epoch}.png')
    plt.close()
    return pr_auc

def plot_confidence_histogram(y_pred_probs, epoch, out_dir, n_bins=20):
    """
    Plot histogram of predicted confidence (max softmax probability per sample).
    Saves to out_dir/confidence_histogram_epoch_{epoch}.png
    """
    probs = np.asarray(y_pred_probs)
    if probs.ndim == 2:
        confidences = np.max(probs, axis=1)
    else:
        confidences = probs
    plt.figure(figsize=(8, 5))
    plt.hist(confidences, bins=n_bins, edgecolor="black", alpha=0.7)
    plt.xlabel("Predicted Confidence (max probability)")
    plt.ylabel("Count")
    plt.title(f"Confidence Distribution - Epoch {epoch}")
    plt.savefig(os.path.join(out_dir, f"confidence_histogram_epoch_{epoch}.png"))
    plt.close()


def plot_classification_report(y_true, y_pred, epoch,out_dir):
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    plt.figure(figsize=(10, 6))
    sns.heatmap(report_df.iloc[:-1, :].T, annot=True, fmt='.2f', cmap='Blues')
    plt.title(f'Classification Report - Epoch {epoch}')
    plt.savefig(f'{out_dir}/classification_report_epoch_{epoch}.png')
    plt.close()
    return report_df

#Function to plot all epochs pession  curve in one graph
def plot_all_epochs_pr_curve(y_true, y_pred_probs, epochs):
    plt.figure(figsize=(10, 6))
    for epoch in epochs:
        precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'Epoch {epoch} (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - All Epochs')
    plt.legend(loc='lower left')
    plt.savefig('pr_curve_all_epochs.png')
    plt.close()

#Function to plot all epochs roc curve in one graph
def plot_all_epochs_roc_curve(y_true, y_pred_probs, epochs):
    plt.figure(figsize=(10, 6))
    for epoch in epochs:
        fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Epoch {epoch} (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - All Epochs')
    plt.legend(loc='lower right')
    plt.savefig('roc_curve_all_epochs.png')
    plt.close()

# Function to plot all epochs F1 score in one graph
def plot_all_epochs_f1_score(y_true, y_pred_probs, epochs):
    plt.figure(figsize=(10, 6))
    for epoch in epochs:
        f1 = f1_score(y_true, y_pred_probs)
        plt.plot(epoch, f1, marker='o', label=f'Epoch {epoch} (F1 = {f1:.2f})')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score - All Epochs')
    plt.legend(loc='lower right')
    plt.savefig('f1_score_all_epochs.png')
    plt.close()

# Function to plot all epochs recall score in one graph
def plot_all_epochs_recall_score(y_true, y_pred_probs, epochs):
    plt.figure(figsize=(10, 6))
    for epoch in epochs:
        recall = recall_score(y_true, y_pred_probs)
        plt.plot(epoch, recall, marker='o', label=f'Epoch {epoch} (Recall = {recall:.2f})')
    plt.xlabel('Epoch')
    plt.ylabel('Recall Score')
    plt.title('Recall Score - All Epochs')
    plt.legend(loc='lower right')
    plt.savefig('recall_score_all_epochs.png')
    plt.close()

# Function to one epoch plot prerecision, recall and f1 score in one graph
def plot_all_epochs_pr_f1_recall_score(y_true, y_pred_probs, epochs):
    plt.figure(figsize=(10, 6))
    for epoch in epochs:
        precision, recall, _ = precision_recall_curve(y_true, y_pred_probs[:, 1])
        pr_auc = auc(recall, precision)
        f1 = f1_score(y_true, y_pred_probs[:, 1])
        plt.plot(epoch, pr_auc, marker='o', label=f'Epoch {epoch} (Precision-Recall AUC = {pr_auc:.2f})')
        plt.plot(epoch, f1, marker='o', label=f'Epoch {epoch} (F1 = {f1:.2f})')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Precision-Recall AUC and F1 Score - All Epochs')
    plt.legend(loc='lower right')
    plt.savefig('pr_f1_score_all_epochs.png')
    plt.close()



def _ensure_labels_and_probs(y_true, y_pred_probs_or_labels):
    """Return (y_pred_labels, y_pred_probs_for_roc). Probs may be None if only labels given."""
    y_true = np.asarray(y_true)
    pred = np.asarray(y_pred_probs_or_labels)
    if pred.ndim == 2 and pred.shape[1] > 1:
        y_pred = pred.argmax(axis=1)
        # For binary ROC/PR use probability of positive class
        roc_probs = pred[:, 1] if pred.shape[1] == 2 else pred[:, 1]
        return y_pred, roc_probs
    else:
        y_pred = pred.ravel()
        return y_pred, None


def evaluate(
    y_true,
    y_pred_probs_or_labels,
    epoch,
    out_dir,
    include_statistical_tests=True,
):
    """
    Full evaluation: confusion matrix, ROC, PR curve, confidence histogram,
    classification report, MCC, Cohen's Kappa, Brier, ECE, bootstrap CI,
    sensitivity, specificity, PPV, NPV, chi-square test.
    Accepts either predicted labels (1D) or predicted probabilities (2D).
    """
    y_pred, roc_probs = _ensure_labels_and_probs(y_true, y_pred_probs_or_labels)
    probs_2d = np.asarray(y_pred_probs_or_labels) if np.asarray(y_pred_probs_or_labels).ndim == 2 else None

    cm = plot_confusion_matrix(y_true, y_pred, epoch, out_dir)

    if roc_probs is not None:
        roc_auc = plot_roc_curve(y_true, roc_probs, epoch, out_dir)
        pr_auc = plot_precision_recall_curve(y_true, roc_probs, epoch, out_dir)
        if probs_2d is not None:
            plot_confidence_histogram(probs_2d, epoch, out_dir)
    else:
        roc_auc = pr_auc = None

    report_df = plot_classification_report(y_true, y_pred, epoch, out_dir)

    # Full metrics (macro, weighted, brier, ECE, diagnostic, etc.)
    metrics = compute_classification_metrics(y_true, y_pred, y_pred_probs=roc_probs)

    # Bootstrap accuracy with 95% and 99% CI
    boot = bootstrap_accuracy(y_true, y_pred, n_bootstrap=1000, random_state=42)
    metrics["bootstrap_accuracy_mean"] = boot["accuracy_mean"]
    metrics["bootstrap_ci_95_lower"] = boot["ci_95_lower"]
    metrics["bootstrap_ci_95_upper"] = boot["ci_95_upper"]
    metrics["bootstrap_ci_99_lower"] = boot["ci_99_lower"]
    metrics["bootstrap_ci_99_upper"] = boot["ci_99_upper"]

    # Chi-square test on confusion matrix
    if include_statistical_tests and cm.size >= 4:
        chi2 = chi_square_test(cm)
        metrics["chi2_statistic"] = chi2["statistic"]
        metrics["chi2_pvalue"] = chi2["pvalue"]
        metrics["chi2_dof"] = chi2["dof"]

    mcc = metrics.get("mcc", matthews_corrcoef(y_true, y_pred))
    kappa = metrics.get("cohen_kappa", cohen_kappa_score(y_true, y_pred))
    n_cols = report_df.shape[1]
    report_df.loc["mcc"] = [mcc] + [""] * (n_cols - 1)
    report_df.loc["cohen_kappa"] = [kappa] + [""] * (n_cols - 1)
    report_df.to_csv(os.path.join(out_dir, f"classification_report_epoch_{epoch}.csv"))

    return {"cm": cm, "roc_auc": roc_auc, "pr_auc": pr_auc, "report": report_df, **metrics}


def compute_per_split_metrics(splits: dict) -> dict:
    """
    Compute metrics per split (train, val, test) and return flat dict with keys like
    train/accuracy, val/macro_f1, test/weighted_precision.

    Args:
        splits: Dict of split_name -> (y_true, y_pred, y_pred_probs or None).
            e.g. {"train": (y_t, y_p, probs_t), "val": (y_v, y_pv, probs_v)}.

    Returns:
        dict: Flattened metrics, e.g. train/accuracy, val/macro_f1, etc.
    """
    out = {}
    for split_name, data in splits.items():
        if len(data) == 2:
            y_true, y_pred = data
            probs = None
        else:
            y_true, y_pred, probs = data
        m = compute_classification_metrics(y_true, y_pred, y_pred_probs=probs)
        for k, v in m.items():
            if isinstance(v, (int, float)):
                out[f"{split_name}/{k}"] = v
    return out


def print_cv_metrics_with_ci(list_of_metric_dicts, confidence=0.95):
    """
    Print metrics aggregated over folds with 95% confidence intervals.
    Use after k-fold cross-validation; each dict in list_of_metric_dicts is from
    compute_classification_metrics for one fold.

    Args:
        list_of_metric_dicts: List of metric dicts (one per fold).
        confidence: Confidence level (default 0.95).
    """
    aggregated = aggregate_cv_metrics(list_of_metric_dicts, confidence=confidence)
    logger.info("--- Cross-validation metrics (mean ± CI) ---")
    for name, vals in aggregated.items():
        if vals["ci_str"] is not None:
            logger.info("  %s: %s", name, vals["ci_str"])


def compare_two_models(scores_a, scores_b, metric_name="accuracy"):
    """
    Run paired t-test and Wilcoxon signed-rank test between two methods (e.g. two models)
    over the same folds. Print p-values.

    Args:
        scores_a: List of metric values for method A (one per fold).
        scores_b: List of metric values for method B (one per fold).
        metric_name: Label for printing.
    """
    t_result = paired_ttest(scores_a, scores_b)
    w_result = wilcoxon_signed_rank(scores_a, scores_b)
    logger.info("--- Comparison (%s) ---", metric_name)
    logger.info("  Paired t-test: statistic=%.4f, p-value=%.4f", t_result["statistic"], t_result["pvalue"])
    logger.info("  Wilcoxon signed-rank: statistic=%.4f, p-value=%.4f", w_result["statistic"], w_result["pvalue"])
    logger.info("  Mean difference (A - B): %.4f", t_result["mean_diff"])
    return {"paired_ttest": t_result, "wilcoxon": w_result}


def run_embedding_visualizations(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_dir: str = "results",
    class_names: Optional[List[str]] = None,
    title_prefix: str = "",
) -> dict:
    """
    Generate t-SNE and UMAP plots of embeddings (e.g. from encoder on validation set).
    Saves tsne.png and umap.png in output_dir.
    """
    if class_names is None:
        class_names = ["Normal", "Abnormal"]
    return plot_tsne_umap(
        embeddings,
        labels,
        output_dir=output_dir,
        title_prefix=title_prefix,
        class_names=class_names,
    )


def run_radar_visualization(
    metrics_by_model: dict,
    output_path: str = "results/radar_models.png",
    metrics_order: Optional[List[str]] = None,
    title: str = "Model Comparison",
) -> None:
    """
    Generate a radar chart comparing multiple models on classification metrics.
    metrics_by_model: e.g. {"DINOv2": {"accuracy": 0.9, "f1": 0.88, ...}, "MAE": {...}}
    """
    if metrics_order is None:
        metrics_order = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "mcc",
            "cohen_kappa",
            "auc_roc",
            "auc_pr",
        ]
    plot_radar_chart(
        metrics_by_model,
        output_path=output_path,
        metrics_order=metrics_order,
        title=title,
    )


def run_cross_validation(
    dataset_path: str,
    train_eval_fn,
    n_splits: int = 5,
    batch_size: int = 32,
    random_state: int = 42,
):
    """
    Run K-fold cross-validation by training and evaluating on each fold.

    Args:
        dataset_path: Root path of the dataset (e.g. MIMII).
        train_eval_fn: Callable (fold_index, train_loader, val_loader) -> metrics_dict.
            Must return a dict with keys like "accuracy", "precision", "recall", "f1",
            "mcc", "cohen_kappa", "auc_roc", "auc_pr" (as from compute_classification_metrics).
        n_splits: Number of folds (default 5 for 5-fold CV).
        batch_size: Batch size for train and val loaders.
        random_state: Seed for StratifiedKFold.

    Returns:
        tuple: (list_of_metric_dicts, aggregated_metrics)
            - list_of_metric_dicts: List of dicts, one per fold.
            - aggregated_metrics: Dict of metric -> {mean, std, ci_lower, ci_upper, ci_str}.
    """
    list_of_metric_dicts = []
    for fold in range(n_splits):
        train_loader, val_loader = get_audio_dataloaders_cv(
            dataset_path,
            fold_index=fold,
            n_splits=n_splits,
            batch_size=batch_size,
            random_state=random_state,
        )
        metrics = train_eval_fn(fold, train_loader, val_loader)
        list_of_metric_dicts.append(metrics)
    aggregated = aggregate_cv_metrics(list_of_metric_dicts, confidence=0.95)
    return list_of_metric_dicts, aggregated


def run_statistical_analysis(
    metrics_by_model: dict,
    output_dir: str = "results",
    confidence: float = 0.95,
    metric_names: Optional[List[str]] = None,
):
    """
    Perform full statistical analysis on cross-validation results: report metrics with
    95% confidence intervals, pairwise significance matrix (paired t-test and Wilcoxon),
    and box plots of metric distributions across folds.

    Args:
        metrics_by_model: Dict mapping model name -> list of metric dicts (one per fold),
            e.g. {"DINOv2": [fold0_metrics, fold1_metrics, ...], "MAE": [...]}.
        output_dir: Directory to save tables and figures.
        confidence: Confidence level for CIs (default 0.95).
        metric_names: Metrics to include in box plots and significance (default: all common).
    """
    os.makedirs(output_dir, exist_ok=True)

    if metric_names is None:
        metric_names = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "mcc",
            "cohen_kappa",
            "auc_roc",
            "auc_pr",
        ]

    # Per-model aggregated metrics with CI
    logger.info("--- Cross-validation metrics (mean ± 95%% CI) per model ---")
    for model_name, per_fold in metrics_by_model.items():
        agg = aggregate_cv_metrics(per_fold, confidence=confidence)
        logger.info("%s:", model_name)
        for k in metric_names:
            if k in agg and agg[k]["ci_str"] is not None:
                logger.info("  %s: %s", k, agg[k]["ci_str"])

    # Box plots: metric distributions across folds
    plot_cv_metric_boxplots(
        metrics_by_model,
        metric_names=metric_names,
        output_path=os.path.join(output_dir, "cv_metric_boxplots.png"),
        title="Cross-Validation Metric Distributions (by model)",
        ylim=(0, 1),
    )
    logger.info("Box plots saved to %s", os.path.join(output_dir, "cv_metric_boxplots.png"))

    # Significance matrix for primary metric (accuracy) and optionally others
    key_metrics = ["accuracy", "f1", "auc_roc"]
    key_metrics = [m for m in key_metrics if m in metric_names]
    for metric in key_metrics:
        scores = {
            name: [d[metric] for d in per_fold if metric in d and d[metric] is not None]
            for name, per_fold in metrics_by_model.items()
        }
        # Ensure same number of folds for all models
        n_folds = len(next(iter(metrics_by_model.values())))
        if not all(len(s) == n_folds for s in scores.values()):
            continue
        sm = significance_matrix(scores, metric_name=metric, tests=("ttest", "wilcoxon"))
        names = sm["model_names"]
        if len(names) < 2:
            continue
        # Save p-value matrix as CSV (Table 5 style)
        import csv
        for test_key, arr_key in [("ttest", "pvalue_ttest"), ("wilcoxon", "pvalue_wilcoxon")]:
            if arr_key not in sm:
                continue
            path = os.path.join(output_dir, f"significance_matrix_{metric}_{test_key}.csv")
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([""] + names)
                for i, name in enumerate(names):
                    row = [name] + [f"{sm[arr_key][i, j]:.4f}" for j in range(len(names))]
                    w.writerow(row)
            logger.info("Significance matrix (%s, %s) saved to %s", metric, test_key, path)
        # Log to console
        logger.info("--- Statistical significance (%s): Paired t-test p-values ---", metric)
        logger.info("(p < 0.05 suggests significant difference between models)")
        pv = sm["pvalue_ttest"]
        for i, ni in enumerate(names):
            for j, nj in enumerate(names):
                if i < j:
                    logger.info("  %s vs %s: p = %.4f", ni, nj, pv[i, j])


# if __name__ == "__main__":
#     # Example usage
#     model_dir = "/path/to/your/model/directory"
#     val_data_path = "/path/to/your/validation/data"
#     batch_size = 32
#     val_loader = get_validation_loader(val_data_path, batch_size)[1]
#     evaluate_models(model_dir, val_loader)
#     # Embedding visualizations (after collecting embeddings and labels):
#     # run_embedding_visualizations(embeddings, labels, output_dir="results")
#     # Radar (after collecting metrics per model):
#     # run_radar_visualization({"Model A": metrics_a, "Model B": metrics_b}, "results/radar.png")
