import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    precision_recall_fscore_support,
    ConfusionMatrixDisplay,
    accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score,
    brier_score_loss,
    average_precision_score,
)
from sklearn.metrics import precision_recall_curve, auc


def _get_pos_probs(y_pred_probs, n_classes):
    """Extract positive-class (or per-class) probabilities from y_pred_probs."""
    probs = np.asarray(y_pred_probs)
    if probs.ndim == 2:
        if probs.shape[1] == 2:
            return probs[:, 1], probs
        return probs[:, 1] if probs.shape[1] > 1 else probs.ravel(), probs
    return probs.ravel(), None


def _expected_calibration_error(y_true, y_pred_probs, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).
    Bins predictions by confidence and computes weighted average of |acc - conf|.
    """
    y_true = np.asarray(y_true).ravel()
    probs = np.asarray(y_pred_probs)
    if probs.ndim == 2:
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
    else:
        confidences = probs
        predictions = (probs >= 0.5).astype(int)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop = np.mean(in_bin)
        if prop > 0:
            acc_in_bin = np.mean(y_true[in_bin] == predictions[in_bin])
            avg_conf = np.mean(confidences[in_bin])
            ece += prop * np.abs(acc_in_bin - avg_conf)
    return float(ece)


def _diagnostic_metrics_from_cm(tn, fp, fn, tp):
    """Compute Sensitivity, Specificity, PPV, NPV from confusion matrix elements."""
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return {"sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv}


def compute_classification_metrics(y_true, y_pred, y_pred_probs=None, n_classes=None):
    """
    Compute comprehensive classification metrics.

    Includes: accuracy, precision/recall/F1 (binary, macro, weighted), MCC, Cohen's Kappa,
    ROC AUC, PR AUC (per-class), Brier score, ECE, sensitivity, specificity, PPV, NPV.

    Args:
        y_true: Ground truth labels (array-like).
        y_pred: Predicted labels (array-like).
        y_pred_probs: Optional predicted probabilities (1D positive-class or 2D (n, n_classes)).
        n_classes: Number of classes (default: inferred from y_true).

    Returns:
        dict: Full metrics dict.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if n_classes is None:
        n_classes = max(int(y_true.max()), int(y_pred.max())) + 1
    n_classes = max(2, n_classes)

    # Binary (for backward compat)
    prec_bin, rec_bin, f1_bin, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    # Macro
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    # Weighted
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    acc = float(accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))

    metrics = {
        "accuracy": acc,
        "precision": float(prec_bin),
        "recall": float(rec_bin),
        "f1": float(f1_bin),
        "macro_precision": float(prec_macro),
        "macro_recall": float(rec_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(prec_weighted),
        "weighted_recall": float(rec_weighted),
        "weighted_f1": float(f1_weighted),
        "mcc": mcc,
        "cohen_kappa": kappa,
    }

    # Diagnostic (Sensitivity, Specificity, PPV, NPV) from confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    if n_classes == 2 and cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        diag = _diagnostic_metrics_from_cm(int(tn), int(fp), int(fn), int(tp))
        metrics.update(diag)

    if y_pred_probs is not None:
        pos_probs, probs_2d = _get_pos_probs(y_pred_probs, n_classes)
        # Keep keys stable for downstream reporting even in degenerate folds.
        metrics["auc_roc"] = 0.0
        metrics["auc_pr"] = 0.0
        if pos_probs is not None:
            # Binary ROC/PR AUC
            try:
                fpr, tpr, _ = roc_curve(y_true, pos_probs)
                metrics["auc_roc"] = float(auc(fpr, tpr))
            except Exception:
                metrics["auc_roc"] = 0.0
            try:
                prec_curve, rec_curve, _ = precision_recall_curve(y_true, pos_probs)
                metrics["auc_pr"] = float(auc(rec_curve, prec_curve))
            except Exception:
                metrics["auc_pr"] = 0.0
            # Brier score (binary)
            metrics["brier_score"] = float(brier_score_loss(y_true, pos_probs))
            # ECE
            if probs_2d is not None:
                metrics["ece"] = _expected_calibration_error(y_true, probs_2d)
            else:
                probs_bin = np.column_stack([1 - pos_probs, pos_probs])
                metrics["ece"] = _expected_calibration_error(y_true, probs_bin)

        # Per-class ROC AUC and PR AUC
        if probs_2d is not None and probs_2d.shape[1] >= 2:
            # One-hot (n, n_probs); avoids sklearn label_binarize returning (n,) or (n,1) for binary
            n_pc = probs_2d.shape[1]
            y_true_bin = np.zeros((len(y_true), n_pc), dtype=np.float64)
            yt = np.asarray(y_true).astype(int).ravel()
            valid = (yt >= 0) & (yt < n_pc)
            y_true_bin[np.arange(len(yt))[valid], yt[valid]] = 1.0
            for c in range(min(n_pc, n_classes)):
                if np.any(y_true_bin[:, c] == 1) and np.any(y_true_bin[:, c] == 0):
                    try:
                        metrics[f"auc_roc_class_{c}"] = float(
                            roc_auc_score(y_true_bin[:, c], probs_2d[:, c])
                        )
                    except Exception:
                        metrics[f"auc_roc_class_{c}"] = 0.0
                    try:
                        metrics[f"auc_pr_class_{c}"] = float(
                            average_precision_score(y_true_bin[:, c], probs_2d[:, c])
                        )
                    except Exception:
                        metrics[f"auc_pr_class_{c}"] = 0.0

    return metrics


def evaluate(y_true, y_pred_probs, epoch):
    y_pred = y_pred_probs.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.savefig(f'conf_matrix_epoch_{epoch}.png')
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_pred_probs[:, 1])
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Epoch {epoch}')
    plt.savefig(f'roc_epoch_{epoch}.png')
    plt.close()

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    plt.bar(['Precision', 'Recall', 'F1'], [precision, recall, f1])
    plt.title(f'Metrics - Epoch {epoch}')
    plt.savefig(f'metrics_epoch_{epoch}.png')
    plt.close()
