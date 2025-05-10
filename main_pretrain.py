import os
from dotenv import load_dotenv
from modules.data_loader import get_dataloaders
from modules.encoder import Encoder
from modules.contrastive_loss import nt_xent_loss
from modules.classifier import ClassificationModel
from modules.evaluation import evaluate  # assume returns val_loss, val_acc
import torch
import torch.nn as nn
import torch.optim as optim
from utils.plot_graph import plot_graph
from utils.utils import calculate_accuracy



# Setup
load_dotenv("configs/.env")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset_path = os.getenv("DATASET_PATH")
batch_size = int(os.getenv("BATCH_SIZE"))
epochs = int(os.getenv("EPOCHS"))
lr = float(os.getenv("LEARNING_RATE"))

# Load dataloaders (assuming your get_dataloaders returns train + val)
train_loader, val_loader = get_dataloaders(dataset_path, batch_size)

# Initialize encoder and projector
encoder = Encoder().to(device)
projector = torch.nn.Linear(encoder.feature_dim, 128).to(device)
optimizer = optim.Adam(list(encoder.parameters()) + list(projector.parameters()), lr=lr)

# Tracking
train_losses, val_losses = [], []
val_accuracies = []

encoder.train()
projector.train()

for epoch in range(epochs):
    total_loss = 0
    processed_images = 0
    # === Training ===
    for x, _ in train_loader:
        x1, x2 = x.to(device), x.to(device)  # should apply different augmentations
        z1 = projector(encoder(x1))
        z2 = projector(encoder(x2))
        loss = nt_xent_loss(z1, z2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        processed_images += x1.size(0)
        print(f"Epoch {epoch+1}/{epochs}, Processed {processed_images}/{len(train_loader.dataset)} images", "train_loss:", loss.item())

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # === Validation ===
    encoder.eval()
    classifier = ClassificationModel(encoder, encoder.feature_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc = evaluate(classifier, val_loader, criterion, device)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    print(f"Epoch {epoch+1}/{epochs}, train_loss {avg_train_loss}, Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")
    encoder.train()

# Save encoder
torch.save(encoder.state_dict(), "/model/encoder_pretrained.pth")

# === Plotting ===
epochs_range = list(range(1, epochs + 1))

# Loss plot
plot_graph(
    x=epochs_range,
    y=[train_losses, val_losses],
    title="Train vs Validation Loss",
    xlabel="Epoch",
    ylabel="Loss",
    filename="train_val_loss.png"
)

# Accuracy plot
plot_graph(
    x=epochs_range,
    y=[val_accuracies],  # no training accuracy during contrastive training
    title="Validation Accuracy",
    xlabel="Epoch",
    ylabel="Accuracy",
    filename="val_accuracy.png"
)
