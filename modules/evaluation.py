import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, precision_recall_fscore_support, ConfusionMatrixDisplay

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
