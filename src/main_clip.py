"""
CLIP Classification Training Script

Train a classifier on top of frozen CLIP (ViT-B/16) image features.
Supports zero-shot and multimodal alignment via the CLIP encoder.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from modules.clip_encoder import CLIPClassifier, get_clip_transforms
from utils.plot_graph import plot_graph
from utils.utils import calculate_accuracy

# Reuse ImageDataset from main_dinov2 (same folder structure)
from main_dinov2 import ImageDataset


def get_dataloaders(dataset_path, batch_size, val_split=0.2, image_size=224):
    """Create train and validation dataloaders with CLIP transforms."""
    transforms_dict = get_clip_transforms(image_size=image_size)

    train_dataset = ImageDataset(dataset_path, transform=transforms_dict["train"])
    val_dataset = ImageDataset(dataset_path, transform=transforms_dict["val"])

    indices = list(range(len(train_dataset)))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_split,
        stratify=train_dataset.labels,
        random_state=42,
    )

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
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
    from dotenv import load_dotenv
    load_dotenv("configs/.env")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    dataset_path = os.getenv("DATASET_PATH")
    batch_size = int(os.getenv("BATCH_SIZE", 16))
    epochs = int(os.getenv("EPOCHS", 20))
    lr = float(os.getenv("LEARNING_RATE", 1e-4))
    num_classes = int(os.getenv("NUM_CLASSES", 2))
    image_size = int(os.getenv("IMAGE_SIZE", 224))
    model_name = os.getenv("CLIP_MODEL", "ViT-B/16")
    use_hf = os.getenv("CLIP_USE_HF", "false").lower() == "true"

    print(f"\n{'='*50}")
    print("CLIP Classification Training")
    print(f"{'='*50}")
    print(f"Dataset: {dataset_path}")
    print(f"Model: {model_name}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Image size: {image_size}")
    print(f"Use HuggingFace CLIP: {use_hf}")
    print(f"{'='*50}\n")

    train_loader, val_loader = get_dataloaders(
        dataset_path,
        batch_size,
        image_size=image_size,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}\n")

    model = CLIPClassifier(
        num_classes=num_classes,
        model_name=model_name,
        freeze_backbone=True,
        use_fp16=True,
        use_hf=use_hf,
        hidden_dim=256,
        dropout=0.1,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, "model/clip_best.pth")
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.4f})")

    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "val_acc": val_accuracies[-1],
    }, "model/clip_final.pth")

    epochs_range = list(range(1, epochs + 1))
    plot_graph(
        x=epochs_range,
        y=[train_losses, val_losses],
        labels=["Train Loss", "Validation Loss"],
        title="CLIP - Train vs Validation Loss",
        xlabel="Epoch",
        ylabel="Loss",
        filename="clip_loss.png",
    )
    plot_graph(
        x=epochs_range,
        y=[train_accuracies, val_accuracies],
        labels=["Train Accuracy", "Validation Accuracy"],
        title="CLIP - Train vs Validation Accuracy",
        xlabel="Epoch",
        ylabel="Accuracy",
        filename="clip_accuracy.png",
    )

    print(f"\n{'='*50}")
    print("Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
