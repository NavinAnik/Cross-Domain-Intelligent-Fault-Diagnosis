import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    confusion_matrix,
    precision_recall_fscore_support,
    ConfusionMatrixDisplay,
    accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score,
)
from sklearn.metrics import precision_recall_curve, auc


def compute_classification_metrics(y_true, y_pred, y_pred_probs=None):
    """
    Compute classification metrics including MCC and Cohen's Kappa.

    Args:
        y_true: Ground truth labels (array-like).
        y_pred: Predicted labels (array-like).
        y_pred_probs: Optional predicted probabilities for positive class (or 2D array
            of shape (n_samples, n_classes)); used for AUC-ROC and AUC-PR.

    Returns:
        dict: Metrics including accuracy, precision, recall, f1, mcc, cohen_kappa,
              and optionally auc_roc, auc_pr.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }

    if y_pred_probs is not None:
        y_pred_probs = np.asarray(y_pred_probs)
        if y_pred_probs.ndim == 2:
            if y_pred_probs.shape[1] == 2:
                pos_probs = y_pred_probs[:, 1]
            else:
                pos_probs = y_pred_probs[:, 1] if y_pred_probs.shape[1] > 1 else y_pred_probs.ravel()
        else:
            pos_probs = y_pred_probs.ravel()
        fpr, tpr, _ = roc_curve(y_true, pos_probs)
        metrics["auc_roc"] = float(auc(fpr, tpr))
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, pos_probs)
        metrics["auc_pr"] = float(auc(rec_curve, prec_curve))

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
