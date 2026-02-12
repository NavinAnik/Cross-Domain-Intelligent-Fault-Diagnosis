import os
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from modules.data_loader import get_audio_dataloaders
from modules.encoder import Encoder
from modules.classifier import ClassificationModel
from utils.plot_graph import plot_graph
from utils.experiment_tracking import init_wandb, log_metrics, finish_wandb
from evaluate_models import evaluate
from training.optimization import (
    enable_gradient_checkpointing,
    autocast_context,
    TrainingStepHelper,
)

# Load environment variables
load_dotenv("configs/.env")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Load training config
dataset_path = os.getenv("DATASET_PATH")
batch_size = int(os.getenv("BATCH_SIZE"))
epochs = int(os.getenv("EPOCHS"))
lr = float(os.getenv("LEARNING_RATE"))
# Memory optimization
use_amp = os.getenv("USE_AMP", "true").lower() == "true" and device.type == "cuda"
accumulation_steps = max(1, int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "1")))
use_gradient_checkpointing = os.getenv("USE_GRADIENT_CHECKPOINTING", "false").lower() == "true"
print(f"USE_AMP: {use_amp}, GRADIENT_ACCUMULATION_STEPS: {accumulation_steps}, USE_GRADIENT_CHECKPOINTING: {use_gradient_checkpointing}")

# Experiment tracking (Weights & Biases)
wandb_config = {
    "model": "finetune_classifier",
    "dataset_path": dataset_path,
    "batch_size": batch_size,
    "epochs": epochs,
    "learning_rate": lr,
    "use_amp": use_amp,
    "gradient_accumulation_steps": accumulation_steps,
    "use_gradient_checkpointing": use_gradient_checkpointing,
}
init_wandb(project="mimii-anomaly-detection", config=wandb_config, name="finetune-classifier")

# Get train and val loaders
train_loader, val_loader = get_audio_dataloaders(dataset_path, batch_size)

# Setup model
encoder = Encoder(in_channels=1, img_size=128, patch_size=16, emb_dim=768)
encoder.load_state_dict(torch.load("model/encoder_pretrained.pth"))
model = ClassificationModel(encoder, encoder.feature_dim).to(device)
if use_gradient_checkpointing:
    enable_gradient_checkpointing(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
step_helper = TrainingStepHelper(
    optimizer, use_amp=use_amp, device=device, accumulation_steps=accumulation_steps
)

# For plotting
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

def calculate_accuracy(y_true, y_pred):
    correct = (y_true == y_pred).sum().item()
    return correct / len(y_true)

for epoch in range(epochs):
    # --- Training (with optional AMP + gradient accumulation) ---
    model.train()
    train_loss = 0
    y_true_train, y_pred_train = [], []
    processed_images = 0
    step_helper.zero_grad()
    step_helper.reset_step_count()

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        with autocast_context(use_amp, device):
            logits = model(x)
            loss = criterion(logits, y)

        step_helper.step(loss)

        train_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        y_true_train.extend(y.cpu())
        y_pred_train.extend(preds.cpu())
        processed_images += len(y)
        print(f"Epoch {epoch+1}/{epochs}, Processed {processed_images}/{len(train_loader.dataset)} images", "train_acc:", calculate_accuracy(torch.tensor(y_true_train), torch.tensor(y_pred_train)), "train_loss:", train_loss / max(1, processed_images // batch_size))
    step_helper.finalize_step()

    avg_train_loss = train_loss / len(train_loader)
    train_acc = calculate_accuracy(torch.tensor(y_true_train), torch.tensor(y_pred_train))
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_acc)

    # --- Validation ---
    model.eval()
    val_loss = 0
    y_true_val, y_pred_val = [], []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            val_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            y_true_val.extend(y.cpu())
            y_pred_val.extend(preds.cpu())

    avg_val_loss = val_loss / len(val_loader)
    val_acc = calculate_accuracy(torch.tensor(y_true_val), torch.tensor(y_pred_val))
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)
    evaluate(torch.tensor(y_true_val), torch.tensor(y_pred_val), epoch + 1, "results")

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"  Val   Loss: {avg_val_loss:.4f}, Val   Acc: {val_acc:.4f}")
    log_metrics({
        "train_loss": avg_train_loss,
        "train_acc": train_acc,
        "val_loss": avg_val_loss,
        "val_acc": val_acc,
    }, step=epoch + 1)

    # Save checkpoints
    torch.save(model.state_dict(), f"model/model_epoch_{epoch + 1}.pth")
    torch.save(encoder.state_dict(), f"model/encoder_epoch_{epoch + 1}.pth")

# --- Plotting ---
epochs_range = list(range(1, epochs + 1))

plot_graph(
    x=epochs_range,
    y=[train_losses, val_losses],
    title="Train vs Validation Loss",
    xlabel="Epoch",
    ylabel="Loss",
    filename="loss_comparison.png"
)

plot_graph(
    x=epochs_range,
    y=[train_accuracies, val_accuracies],
    title="Train vs Validation Accuracy",
    xlabel="Epoch",
    ylabel="Accuracy",
    filename="accuracy_comparison.png"
)
finish_wandb()
