import os
from dotenv import load_dotenv
from modules.data_loader import get_dataloaders
from modules.encoder import Encoder
from modules.contrastive_loss import nt_xent_loss
import torch
import torch.nn as nn
import torch.optim as optim
from utils.plot_graph import plot_graph
from utils.utils import calculate_accuracy
from  evaluate_models import evaluate

# Setup
load_dotenv("configs/.env")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset_path = os.getenv("DATASET_PATH")
batch_size = int(os.getenv("BATCH_SIZE"))
epochs = int(os.getenv("EPOCHS"))
lr = float(os.getenv("LEARNING_RATE"))
num_classes = int(os.getenv("NUM_CLASSES"))

# Load dataloaders
train_loader, val_loader = get_dataloaders(dataset_path, batch_size)

# Encoder + contrastive projector
encoder = Encoder(
    img_size=224,
    patch_size=16,
    emb_dim=768,
    num_layers=12,
    num_heads=12,
    mlp_dim=3072,
    dropout=0.1
).to(device)
projector = torch.nn.Linear(encoder.feature_dim, 128).to(device)
optimizer = optim.Adam(list(encoder.parameters()) + list(projector.parameters()), lr=lr)

# Classifier head for evaluation and tracking accuracy
classifier = nn.Linear(encoder.feature_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss()

# Tracking
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

encoder.train()
projector.train()

for epoch in range(epochs):
    total_loss = 0
    processed_images = 0

    # === Contrastive Pretraining ===
    for x, _ in train_loader:
        x1, x2 = x.to(device), x.to(device)  # TODO: Apply different augmentations here
        z1 = projector(encoder(x1))
        z2 = projector(encoder(x2))
        loss = nt_xent_loss(z1, z2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        processed_images += x1.size(0)
        print(f"Epoch {epoch+1}/{epochs}, Processed {processed_images}/{len(train_loader.dataset)}, Train Loss: {loss.item():.4f}")

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # === Train classifier on frozen encoder for accuracy ===
    encoder.eval()
    classifier.train()
    clf_optimizer = optim.Adam(classifier.parameters(), lr=lr)

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            features = encoder(x)
        logits = classifier(features)
        loss = criterion(logits, y)

        clf_optimizer.zero_grad()
        loss.backward()
        clf_optimizer.step()

    # === Compute Training Accuracy ===
    classifier.eval()
    y_true_train, y_pred_train = [], []
    with torch.no_grad():
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            features = encoder(x)
            logits = classifier(features)
            preds = torch.argmax(logits, dim=1)
            y_true_train.extend(y.cpu())
            y_pred_train.extend(preds.cpu())
    train_acc = calculate_accuracy(torch.tensor(y_true_train), torch.tensor(y_pred_train))
    train_accuracies.append(train_acc)

    # === Validation Accuracy ===
    val_loss = 0
    y_true_val, y_pred_val = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            features = encoder(x)
            logits = classifier(features)
            loss = criterion(logits, y)

            val_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            y_true_val.extend(y.cpu())
            y_pred_val.extend(preds.cpu())

    avg_val_loss = val_loss / len(val_loader)
    val_acc = calculate_accuracy(torch.tensor(y_true_val), torch.tensor(y_pred_val))

    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)
    # evaluate(y_true_val, y_pred_val, epoch, "results")

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

# Save encoder
os.makedirs("model", exist_ok=True)
torch.save(encoder.state_dict(), "model/encoder_pretrained.pth")

# === Plotting ===
epochs_range = list(range(1, epochs + 1))

plot_graph(
    x=epochs_range,
    y=[train_losses, val_losses],
    labels=["Train Loss", "Validation Loss"],
    title="Train vs Validation Loss",
    xlabel="Epoch",
    ylabel="Loss",
    filename="train_val_loss.png"
)

plot_graph(
    x=epochs_range,
    y=[train_accuracies, val_accuracies],
    labels=["Train Accuracy", "Validation Accuracy"],
    title="Train vs Validation Accuracy",
    xlabel="Epoch",
    ylabel="Accuracy",
    filename="train_val_accuracy.png"
)
