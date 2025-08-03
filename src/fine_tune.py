import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split

# Assume TransformerEncoder is defined as before (same code)
# Paste your TransformerEncoder class here
# Paste your Mel-spectrogram extraction & augmentation methods here or reuse from pretraining

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------- Labeled Dataset ---------
class LabeledMIMIIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.labels = []
        label_map = {'normal': 0, 'abnormal': 1}
        self.transform = transform

        for machine_type in os.listdir(root_dir):
            machine_type_path = os.path.join(root_dir, machine_type)
            if not os.path.isdir(machine_type_path):
                continue
            for machine_id in os.listdir(machine_type_path):
                machine_id_path = os.path.join(machine_type_path, machine_id)
                if not os.path.isdir(machine_id_path):
                    continue
                for label_name in ['normal', 'abnormal']:
                    label_path = os.path.join(machine_id_path, label_name)
                    if os.path.exists(label_path):
                        for file in os.listdir(label_path):
                            if file.endswith('.wav'):
                                self.samples.append(os.path.join(label_path, file))
                                self.labels.append(label_map[label_name])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav_path = self.samples[idx]
        waveform, _ = torchaudio.load(wav_path)
        waveform = waveform.mean(dim=0)  # mono
        # Process waveform to mel spectrogram like pretraining
        mel_spec = self.process(waveform)
        label = self.labels[idx]
        return mel_spec, label

    def process(self, waveform):
        import librosa
        SAMPLE_RATE = 16000
        N_MELS = 64
        MAX_FRAMES = 128

        mel_spec = librosa.feature.melspectrogram(
            y=waveform.numpy(),
            sr=SAMPLE_RATE,
            n_mels=N_MELS,
            n_fft=1024,
            hop_length=512,
        )
        log_mel = librosa.power_to_db(mel_spec).T

        if log_mel.shape[0] > MAX_FRAMES:
            log_mel = log_mel[:MAX_FRAMES, :]
        elif log_mel.shape[0] < MAX_FRAMES:
            pad = MAX_FRAMES - log_mel.shape[0]
            log_mel = np.pad(log_mel, ((0, pad), (0, 0)), mode='constant')

        return torch.tensor(log_mel, dtype=torch.float32)


# ---------- Model ----------
class FinetuneClassifier(nn.Module):
    def __init__(self, encoder, embed_dim=128, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits


# ---------- Training & Evaluation Functions ----------
def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0
    all_preds = []
    all_labels = []

    for x, y in dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())

    epoch_loss = running_loss / len(dataloader.dataset)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def eval_one_epoch(model, dataloader, criterion):
    model.eval()
    running_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            running_loss += loss.item() * x.size(0)

            probs = torch.softmax(logits, dim=1)[:, 1]  # prob for class 1 (abnormal)
            preds = torch.argmax(logits, dim=1)
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    epoch_loss = running_loss / len(dataloader.dataset)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)

    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_auc = roc_auc_score(all_labels, all_probs)
    return epoch_loss, epoch_acc, epoch_auc, all_labels, all_preds, all_probs


# --------- Plotting functions ---------
def plot_confusion_matrix(labels, preds):
    import seaborn as sns
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


def plot_roc_curve(labels, probs):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()


def plot_precision_recall_f1(labels, probs):
    precision, recall, _ = precision_recall_curve(labels, probs)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    plt.plot(recall, precision, label="Precision-Recall Curve")
    plt.plot(recall, f1, label="F1 Score Curve")
    plt.xlabel("Recall")
    plt.ylabel("Score")
    plt.title("Precision, Recall, and F1 Score")
    plt.legend()
    plt.show()


def plot_loss_accuracy(train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.legend()

    plt.show()


# -------- Main Finetune Loop --------
def finetune_main(root_dir):
    dataset = LabeledMIMIIDataset(root_dir)

    train_idx, val_idx = train_test_split(
        list(range(len(dataset))),
        test_size=0.2,
        stratify=dataset.labels,
        random_state=42,
    )
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    encoder = TransformerEncoder()
    encoder.load_state_dict(torch.load("contrastive_transformer_encoder.pth"))
    encoder.to(DEVICE)

    # Freeze encoder initially
    for param in encoder.parameters():
        param.requires_grad = False

    model = FinetuneClassifier(encoder).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, 31):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_auc, labels, preds, probs = eval_one_epoch(model, val_loader, criterion)

        print(
            f"Epoch {epoch:02d} "
            f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f} Val AUC: {val_auc:.4f}"
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Optionally unfreeze encoder after some epochs
        if epoch == 10:
            for param in encoder.parameters():
                param.requires_grad = True

    # Final evaluation plots
    plot_confusion_matrix(labels, preds)
    plot_roc_curve(labels, probs)
    plot_precision_recall_f1(labels, probs)
    plot_loss_accuracy(train_losses, val_losses, train_accs, val_accs)


if __name__ == "__main__":
    ROOT_DIR = "MIMII"  # path to your labeled MIMII dataset
    finetune_main(ROOT_DIR)
