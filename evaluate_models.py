#generate confusion matrix and classification report
import os
import pandas as pd
import seaborn as sns
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, precision_recall_fscore_support, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from modules.data_loader import get_validation_loader


#Confusion matrix function
def plot_confusion_matrix(y_true, y_pred, epoch):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.savefig(f'conf_matrix_epoch_{epoch}.png')
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
    plt.savefig('pr_curve_all_epochs.png')
    plt.close()

def plot_roc_curve(y_true, y_pred_probs, epoch):
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Epoch {epoch}')
    plt.legend(loc='lower right')
    plt.savefig(f'roc_epoch_{epoch}.png')
    plt.close()
    return roc_auc

def plot_precision_recall_curve(y_true, y_pred_probs, epoch):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs[:, 1])
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label=f'Precision-Recall Curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - Epoch {epoch}')
    plt.legend(loc='lower left')
    plt.savefig(f'pr_curve_epoch_{epoch}.png')
    plt.close()
    return pr_auc

def plot_classification_report(y_true, y_pred, epoch):
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    plt.figure(figsize=(10, 6))
    sns.heatmap(report_df.iloc[:-1, :].T, annot=True, fmt='.2f', cmap='Blues')
    plt.title(f'Classification Report - Epoch {epoch}')
    plt.savefig(f'classification_report_epoch_{epoch}.png')
    plt.close()
    return report_df

#Function to plot all epochs pession  curve in one graph
def plot_all_epochs_pr_curve(y_true, y_pred_probs, epochs):
    plt.figure(figsize=(10, 6))
    for epoch in epochs:
        precision, recall, _ = precision_recall_curve(y_true, y_pred_probs[:, 1])
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
        fpr, tpr, _ = roc_curve(y_true, y_pred_probs[:, 1])
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
        f1 = f1_score(y_true, y_pred_probs[:, 1])
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
        recall = recall_score(y_true, y_pred_probs[:, 1])
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



def evaluate(y_true, y_pred_probs, epoch):
    y_pred = y_pred_probs.argmax(axis=1)
    cm = plot_confusion_matrix(y_true, y_pred, epoch)
    roc_auc = plot_roc_curve(y_true, y_pred_probs, epoch)
    pr_auc = plot_precision_recall_curve(y_true, y_pred_probs, epoch)
    report_df = plot_classification_report(y_true, y_pred, epoch)

    # Save the classification report as a CSV file
    report_df.to_csv(f'classification_report_epoch_{epoch}.csv')

    return cm, roc_auc, pr_auc, report_df

# Function to read epoch.pth file and evaluate the models AND plot the results
def evaluate_models(model_dir, val_loader):
    epochs = []
    for filename in os.listdir(model_dir):
        if filename.endswith(".pth"):
            epoch = int(filename.split("_")[1].split(".")[0])
            epochs.append(epoch)

    # Sort the epochs
    epochs.sort()

    # Evaluate each model and plot the results
    for epoch in epochs:
        model_path = os.path.join(model_dir, f"model_epoch_{epoch}.pth")
        encoder_path = os.path.join(model_dir, f"encoder_epoch_{epoch}.pth")

        # Load the model and encoder (if needed)
        model = torch.load(model_path)
        #Predict the model on the validation set
        y_pred_probs = model.predict(val_loader)  # Assuming you have a predict method
        y_true = val_loader.dataset.targets  # Assuming you have the true labels in the dataset


    # Plot all epochs ROC curve
    plot_all_epochs_roc_curve(y_true, y_pred_probs, epochs)

    # Plot all epochs Precision-Recall curve
    plot_all_epochs_pr_curve(y_true, y_pred_probs, epochs)

    # Plot all epochs F1 score
    plot_all_epochs_f1_score(y_true, y_pred_probs, epochs)

    # Plot all epochs recall score
    plot_all_epochs_recall_score(y_true, y_pred_probs, epochs)

if __name__ == "__main__":
    # Example usage
    model_dir = "/path/to/your/model/directory"
    val_data_path = "/path/to/your/validation/data"
    batch_size = 32
    val_loader = get_validation_loader(val_data_path, batch_size)[1]  # Assuming get_dataloaders returns train + val loaders

    evaluate_models(model_dir, val_loader)

#Validation_loader
