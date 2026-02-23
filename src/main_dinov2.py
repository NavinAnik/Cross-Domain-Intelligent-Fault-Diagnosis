"""
DINOv2 Classification Training Script

Train a classifier on top of frozen DINOv2 features.
This is much faster and more accurate than training from scratch.
"""

import os

# Reduce xFormers warning verbosity when it's incompatible (e.g. CPU PyTorch / Python 3.13)
os.environ.setdefault("XFORMERS_MORE_DETAILS", "0")

import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from PIL import Image
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

from modules.dinov2_encoder import DINOv2Classifier, get_dinov2_transforms
from modules.focal_loss import FocalLoss
from modules.mixup import MixupCutmixTransform, mixup_criterion
from utils.plot_graph import plot_graph
from utils.utils import calculate_accuracy
from utils.logging_config import setup_logging
from utils.experiment_tracking import init_wandb, log_metrics, finish_wandb
from training.optimization import autocast_context, TrainingStepHelper

# Suppress xFormers warnings (optional dependency; DINOv2 falls back to standard attention)
warnings.filterwarnings("ignore", message="xFormers is not available")
warnings.filterwarnings("ignore", message=".*xFormers.*", category=UserWarning)


# ============ Dataset for Image Classification ============
class ImageDataset(torch.utils.data.Dataset):
    """
    Generic image dataset for classification.
    Expects folder structure: root_dir/class_name/images.png
    OR root_dir/subfolder/class_name/images.png
    """
    
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.labels = []
        self.transform = transform
        self.class_to_idx = {}
        
        # Try to detect folder structure
        # Check if immediate subfolders contain images or more folders
        label_map = {'normal': 0, 'abnormal': 1}  # For MIMII dataset
        
        for item in os.listdir(root_dir):
            item_path = os.path.join(root_dir, item)
            if not os.path.isdir(item_path):
                continue
                
            # Check if this is a label folder (normal/abnormal)
            if item in label_map:
                self._add_images_from_folder(item_path, label_map[item])
            else:
                # This might be a machine type folder, look for label subfolders
                for label_name in ['normal', 'abnormal']:
                    label_path = os.path.join(item_path, label_name)
                    if os.path.exists(label_path):
                        self._add_images_from_folder(label_path, label_map[label_name])
                    else:
                        # Check one more level (machine_type/machine_id/label)
                        for machine_id in os.listdir(item_path):
                            machine_id_path = os.path.join(item_path, machine_id)
                            if os.path.isdir(machine_id_path):
                                for label_name in ['normal', 'abnormal']:
                                    label_path = os.path.join(machine_id_path, label_name)
                                    if os.path.exists(label_path):
                                        self._add_images_from_folder(label_path, label_map[label_name])
        
        print(f"Loaded {len(self.samples)} images")
        print(f"Class distribution: {self._get_class_distribution()}")
    
    def _add_images_from_folder(self, folder_path, label):
        """Add all images from a folder."""
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
        for file in os.listdir(folder_path):
            if any(file.lower().endswith(ext) for ext in valid_extensions):
                self.samples.append(os.path.join(folder_path, file))
                self.labels.append(label)
    
    def _get_class_distribution(self):
        """Get distribution of classes."""
        from collections import Counter
        return dict(Counter(self.labels))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_dataloaders(dataset_path, batch_size, val_split=0.2, image_size=224):
    """
    Create train and validation dataloaders with DINOv2 transforms.
    Returns: train_loader, val_loader, train_labels (for class weighting)
    """
    transforms_dict = get_dinov2_transforms(
        image_size=image_size,
        augment_strength="strong",
    )
    
    # Create dataset with training transforms (we'll apply val transforms to val subset)
    train_dataset = ImageDataset(dataset_path, transform=transforms_dict['train'])
    val_dataset = ImageDataset(dataset_path, transform=transforms_dict['val'])
    
    # Split indices
    indices = list(range(len(train_dataset)))
    train_indices, val_indices = train_test_split(
        indices, 
        test_size=val_split, 
        stratify=train_dataset.labels,
        random_state=42
    )
    
    # Get train labels for class weighting
    train_labels = [train_dataset.labels[i] for i in train_indices]
    
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=pin_memory,
    )
    
    return train_loader, val_loader, train_labels


def train_one_epoch(model, train_loader, criterion, step_helper, device, epoch, epochs, use_amp, mixup_cutmix=None, scheduler=None):
    """Train for one epoch with optional AMP, gradient accumulation, and Mixup/CutMix."""
    model.train()
    total_loss = 0
    y_true, y_pred = [], []
    processed = 0
    step_helper.zero_grad()
    step_helper.reset_step_count()
    
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        # Apply Mixup/CutMix augmentation when enabled
        if mixup_cutmix is not None:
            x_aug, y_a, y_b, lam = mixup_cutmix(x, y)
            with autocast_context(use_amp, device):
                logits = model(x_aug)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
        else:
            with autocast_context(use_amp, device):
                logits = model(x)
                loss = criterion(logits, y)
        
        step_helper.step(loss)
        if scheduler is not None:
            scheduler.step()
        
        # Track metrics
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        
        processed += len(y)
        if (batch_idx + 1) % 10 == 0 or batch_idx == len(train_loader) - 1:
            current_acc = calculate_accuracy(torch.tensor(y_true), torch.tensor(y_pred))
            print(f"  Epoch {epoch}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f} | Acc: {current_acc:.4f}")
    
    step_helper.finalize_step()
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


@torch.no_grad()
def validate_with_tta(model, val_loader, criterion, device, num_classes=2, n_augments=5):
    """Validate with test-time augmentation for higher accuracy."""
    model.eval()
    total_loss = 0
    y_true, y_pred = [], []
    
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        
        # Average predictions over multiple augmented views
        logits_sum = torch.zeros(x.size(0), num_classes).to(device)
        
        for _ in range(n_augments):
            # Random horizontal flip for TTA
            x_aug = torch.flip(x, dims=[3]) if torch.rand(1, device=device).item() > 0.5 else x
            logits = model(x_aug)
            logits_sum += torch.softmax(logits, dim=1)
        
        logits_avg = logits_sum / n_augments
        # Cross-entropy loss on averaged probabilities (log for numerical stability)
        loss = nn.functional.cross_entropy(
            torch.log(logits_avg.clamp(min=1e-8)), y, reduction="mean"
        )
        
        total_loss += loss.item()
        preds = torch.argmax(logits_avg, dim=1)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = calculate_accuracy(torch.tensor(y_true), torch.tensor(y_pred))
    
    return avg_loss, accuracy, torch.tensor(y_true), torch.tensor(y_pred)


def main():
    setup_logging(log_dir="logs")
    # ============ Setup ============
    # Load .env from repo root or src (works when run as python main_dinov2.py or python src/main_dinov2.py)
    _src_dir = os.path.dirname(os.path.abspath(__file__))
    _env_path = os.path.join(_src_dir, "configs", ".env")
    if not os.path.isfile(_env_path):
        _env_path = os.path.join(os.path.dirname(_src_dir), "src", "configs", ".env")
    load_dotenv(_env_path)
    
    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
        print("  Tip: To use GPU, install PyTorch with CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
    # Config
    dataset_path = os.getenv("DATASET_PATH")
    batch_size = int(os.getenv("BATCH_SIZE", 16))  # Smaller batch for large model
    epochs = int(os.getenv("EPOCHS", 20))
    lr = float(os.getenv("LEARNING_RATE", 1e-4))
    num_classes = int(os.getenv("NUM_CLASSES", 2))
    image_size = int(os.getenv("IMAGE_SIZE", 224))
    # DINOv2 patch size is 14; height/width must be divisible by 14
    if image_size % 14 != 0:
        image_size = (image_size // 14) * 14
        print(f"IMAGE_SIZE adjusted to {image_size} (must be divisible by 14)")
    model_name = os.getenv("DINOV2_MODEL", "dinov2_vitl14")
    use_amp = os.getenv("USE_AMP", "true").lower() == "true" and device.type == "cuda"
    accumulation_steps = max(1, int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "1")))
    use_tta = os.getenv("USE_TTA", "true").lower() == "true"
    tta_augmentations = max(1, int(os.getenv("TTA_AUGMENTATIONS", "5")))
    
    print(f"\n{'='*50}")
    print(f"DINOv2 Classification Training")
    print(f"{'='*50}")
    print(f"Dataset: {dataset_path}")
    print(f"Model: {model_name}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Image size: {image_size}")
    print(f"Use TTA: {use_tta} (n_augments={tta_augmentations})")
    print(f"{'='*50}\n")
    
    # ============ Experiment tracking (Weights & Biases) ============
    wandb_config = {
        "model": "dinov2",
        "dinov2_model": model_name,
        "dataset_path": dataset_path,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": lr,
        "num_classes": num_classes,
        "image_size": image_size,
        "use_amp": use_amp,
        "gradient_accumulation_steps": accumulation_steps,
    }
    init_wandb(project="mimii-anomaly-detection", config=wandb_config, name=f"dinov2-{model_name}")
    
    # ============ Data ============
    train_loader, val_loader, train_labels = get_dataloaders(
        dataset_path, 
        batch_size, 
        image_size=image_size
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}\n")
    
    # Compute class weights for balanced loss
    from collections import Counter
    class_counts = Counter(train_labels)
    total = sum(class_counts.values())
    class_weights = [total / (len(class_counts) * class_counts[i]) for i in range(num_classes)]
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Class distribution: {dict(class_counts)}")
    print(f"Class weights (balanced): {class_weights.tolist()}\n")
    
    # ============ Model ============
    model = DINOv2Classifier(
        num_classes=num_classes,
        model_name=model_name,
        freeze_backbone=True,  # Freeze DINOv2, only train classifier
        use_fp16=True,  # Use FP16 for memory efficiency
        use_enhanced_head=True,  # Enable advanced classifier
        use_multi_layer_features=True,  # Extract from multiple transformer layers
        hidden_dims=[1024, 512, 256],  # Deeper MLP with gradual reduction
        dropout=0.3,  # Higher dropout for better regularization
        n_last_blocks=4,  # Use features from last 4 transformer blocks
    ).to(device)
    
    # Only train classifier parameters (backbone is frozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    # ============ Training Setup ============
    # Focal Loss focuses on hard examples and handles class imbalance better
    criterion = FocalLoss(
        alpha=class_weights.tolist(),
        gamma=2.0,
        reduction="mean",
    )
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    # OneCycleLR with warmup for better convergence (step per batch)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr * 10,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=10,
        final_div_factor=100,
    )
    step_helper = TrainingStepHelper(
        optimizer, use_amp=use_amp, device=device, accumulation_steps=accumulation_steps
    )
    
    # Mixup/CutMix augmentation (apply 50% of the time)
    mixup_cutmix = MixupCutmixTransform(
        mixup_alpha=0.2,
        cutmix_alpha=1.0,
        cutmix_prob=0.5,
        switch_prob=0.5,
    )
    
    # ============ Training Loop ============
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    val_precisions, val_recalls, val_f1s = [], [], []
    best_val_acc = 0
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            step_helper,
            device,
            epoch,
            epochs,
            use_amp,
            mixup_cutmix=mixup_cutmix,
            scheduler=scheduler,
        )
        
        # Validate (with optional test-time augmentation)
        if use_tta:
            val_loss, val_acc, y_true, y_pred = validate_with_tta(
                model,
                val_loader,
                criterion,
                device,
                num_classes=num_classes,
                n_augments=tta_augmentations,
            )
        else:
            val_loss, val_acc, y_true, y_pred = validate(
                model, val_loader, criterion, device
            )
        
        # Compute precision, recall, F1 for validation
        y_true_np = y_true.numpy()
        y_pred_np = y_pred.numpy()
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_np, y_pred_np, average='binary', zero_division=0
        )
        
        # Scheduler is stepped per batch in train_one_epoch
        
        # Track metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        val_precisions.append(precision)
        val_recalls.append(recall)
        val_f1s.append(f1)
        
        print(f"\n  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"  Val Precision: {precision:.4f} | Val Recall: {recall:.4f} | Val F1: {f1:.4f}")
        current_lr = scheduler.get_last_lr()[0] if scheduler else lr
        print(f"  LR: {current_lr:.6f}")
        log_metrics({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1,
            "learning_rate": scheduler.get_last_lr()[0] if scheduler else lr,
        }, step=epoch)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("model", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, "model/dinov2_best.pth")
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.4f})")
    
    # ============ Final Save ============
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'val_acc': val_accuracies[-1],
    }, "model/dinov2_final.pth")
    
    # ============ Plotting ============
    epochs_range = list(range(1, epochs + 1))
    results_dir = os.path.join("results", "dinov2")
    os.makedirs(results_dir, exist_ok=True)
    
    plot_graph(
        x=epochs_range,
        y=[train_losses, val_losses],
        labels=["Train Loss", "Validation Loss"],
        title="DINOv2 - Train vs Validation Loss",
        xlabel="Epoch",
        ylabel="Loss",
        filename="loss.png",
        output_dir=results_dir
    )
    
    plot_graph(
        x=epochs_range,
        y=[train_accuracies, val_accuracies],
        labels=["Train Accuracy", "Validation Accuracy"],
        title="DINOv2 - Train vs Validation Accuracy",
        xlabel="Epoch",
        ylabel="Accuracy",
        filename="accuracy.png",
        output_dir=results_dir
    )
    
    plot_graph(
        x=epochs_range,
        y=val_recalls,
        labels=["Validation Recall"],
        title="DINOv2 - Validation Recall vs Epoch",
        xlabel="Epoch",
        ylabel="Recall",
        filename="recall_vs_epoch.png",
        output_dir=results_dir
    )
    
    plot_graph(
        x=epochs_range,
        y=val_precisions,
        labels=["Validation Precision"],
        title="DINOv2 - Validation Precision vs Epoch",
        xlabel="Epoch",
        ylabel="Precision",
        filename="precision_vs_epoch.png",
        output_dir=results_dir
    )
    
    plot_graph(
        x=epochs_range,
        y=val_f1s,
        labels=["Validation F1"],
        title="DINOv2 - Validation F1 vs Epoch",
        xlabel="Epoch",
        ylabel="F1 Score",
        filename="f1_vs_epoch.png",
        output_dir=results_dir
    )
    
    print(f"\n{'='*50}")
    print(f"Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Final Val Precision: {val_precisions[-1]:.4f}")
    print(f"Final Val Recall: {val_recalls[-1]:.4f}")
    print(f"Final Val F1: {val_f1s[-1]:.4f}")
    print(f"{'='*50}")
    finish_wandb()


if __name__ == "__main__":
    main()
