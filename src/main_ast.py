"""
AST (Audio Spectrogram Transformer) Classification Training Script

Train a classifier on top of frozen or fine-tuned AST features for audio
(e.g. mel spectrogram) anomaly detection. Uses 128-dim mel spectrograms
with input normalization mean=0, std=0.5 (critical for fine-tuning).
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from modules.ast_encoder import ASTClassifier, get_ast_normalization
from modules.data_loader import MIMIIAudioDataset, get_audio_dataloaders
from utils.plot_graph import plot_graph
from utils.utils import calculate_accuracy


def get_dataloaders(dataset_path, batch_size, val_split=0.2):
    """
    Create train and validation dataloaders for AST (audio spectrograms).
    Spectrogram normalization (mean=0, std=0.5) is applied inside the model.
    """
    train_loader, val_loader = get_audio_dataloaders(
        dataset_path,
        batch_size=batch_size,
        val_split=val_split,
    )
    return train_loader, val_loader


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, epochs):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    y_true, y_pred = [], []
    processed = 0

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

        processed += len(y)
        if (batch_idx + 1) % 10 == 0 or batch_idx == len(train_loader) - 1:
            current_acc = calculate_accuracy(torch.tensor(y_true), torch.tensor(y_pred))
            print(
                f"  Epoch {epoch}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f} | Acc: {current_acc:.4f}"
            )

    avg_loss = total_loss / len(train_loader)
    accuracy = calculate_accuracy(torch.tensor(y_true), torch.tensor(y_pred))

    return avg_loss, accuracy


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    y_true, y_pred = [], []

    for x, y in val_loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    avg_loss = total_loss / len(val_loader)
    accuracy = calculate_accuracy(torch.tensor(y_true), torch.tensor(y_pred))

    return avg_loss, accuracy, torch.tensor(y_true), torch.tensor(y_pred)


def main():
    # ============ Setup ============
    load_dotenv("configs/.env")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Config (use DATASET_PATH for audio root; plan: batch 16-24 for 8GB GPU)
    dataset_path = os.getenv("DATASET_PATH")
    batch_size = int(os.getenv("BATCH_SIZE", 16))
    epochs = int(os.getenv("EPOCHS", 20))
    lr = float(os.getenv("LEARNING_RATE", 1e-5))  # AST needs ~10x smaller LR than CNNs
    num_classes = int(os.getenv("NUM_CLASSES", 2))
    model_name = os.getenv("AST_MODEL", "MIT/ast-finetuned-audioset-10-10-0.4593")

    print(f"\n{'='*50}")
    print(f"AST (Audio Spectrogram Transformer) Classification Training")
    print(f"{'='*50}")
    print(f"Dataset: {dataset_path}")
    print(f"Model: {model_name}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr} (low LR recommended for AST fine-tuning)")
    print(f"{'='*50}\n")

    # ============ Data ============
    train_loader, val_loader = get_dataloaders(
        dataset_path,
        batch_size=batch_size,
        val_split=0.2,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}\n")

    # ============ Model ============
    norm = get_ast_normalization()
    model = ASTClassifier(
        num_classes=num_classes,
        model_name=model_name,
        freeze_backbone=True,
        use_fp16=True,
        input_mean=norm["mean"],
        input_std=norm["std"],
        hidden_dim=256,
        dropout=0.1,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    # ============ Training Setup ============
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ============ Training Loop ============
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_acc = 0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 40)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, epochs
        )

        val_loss, val_acc, y_true, y_pred = validate(
            model, val_loader, criterion, device
        )

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"\n  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("model", exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                },
                "model/ast_best.pth",
            )
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.4f})")

    # ============ Final Save ============
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "val_acc": val_accuracies[-1],
        },
        "model/ast_final.pth",
    )

    # ============ Plotting ============
    epochs_range = list(range(1, epochs + 1))

    plot_graph(
        x=epochs_range,
        y=[train_losses, val_losses],
        labels=["Train Loss", "Validation Loss"],
        title="AST - Train vs Validation Loss",
        xlabel="Epoch",
        ylabel="Loss",
        filename="ast_loss.png",
    )

    plot_graph(
        x=epochs_range,
        y=[train_accuracies, val_accuracies],
        labels=["Train Accuracy", "Validation Accuracy"],
        title="AST - Train vs Validation Accuracy",
        xlabel="Epoch",
        ylabel="Accuracy",
        filename="ast_accuracy.png",
    )

    print(f"\n{'='*50}")
    print(f"Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
