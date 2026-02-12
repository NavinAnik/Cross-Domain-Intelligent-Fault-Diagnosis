"""
DINOv2 Classification Training Script - Enhanced Version

Advanced training with:
- Enhanced classifier architecture with multi-layer features
- Focal loss for class imbalance
- Mixup/CutMix augmentation
- Progressive unfreezing
- Advanced learning rate scheduling
- Gradient clipping

Target: 90%+ accuracy on all metrics
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision import transforms
from PIL import Image
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter

from modules.dinov2_encoder import DINOv2Classifier, get_dinov2_transforms
from modules.focal_loss import FocalLoss, LabelSmoothingCrossEntropy, CombinedLoss
from modules.mixup import mixup_data, cutmix_data, mixup_criterion
from utils.plot_graph import plot_graph
from utils.utils import calculate_accuracy
from utils.experiment_tracking import init_wandb, log_metrics, finish_wandb
from training.optimization import autocast_context, TrainingStepHelper


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
        label_map = {'normal': 0, 'abnormal': 1}
        
        for item in os.listdir(root_dir):
            item_path = os.path.join(root_dir, item)
            if not os.path.isdir(item_path):
                continue
                
            if item in label_map:
                self._add_images_from_folder(item_path, label_map[item])
            else:
                for label_name in ['normal', 'abnormal']:
                    label_path = os.path.join(item_path, label_name)
                    if os.path.exists(label_path):
                        self._add_images_from_folder(label_path, label_map[label_name])
                    else:
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
        return dict(Counter(self.labels))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_dataloaders(dataset_path, batch_size, val_split=0.2, image_size=224, augment_strength='weak'):
    """
    Create train and validation dataloaders with DINOv2 transforms.
    Returns: train_loader, val_loader, train_labels (for class weighting)
    """
    transforms_dict = get_dinov2_transforms(image_size=image_size, augment_strength=augment_strength)
    
    train_dataset = ImageDataset(dataset_path, transform=transforms_dict['train'])
    val_dataset = ImageDataset(dataset_path, transform=transforms_dict['val'])
    
    # Split indices
    indices = list(range(len(train_dataset)))
    train_indices, val_indices = train_test_split(
        indices, test_size=val_split, stratify=train_dataset.labels, random_state=42
    )
    
    # Get train labels for class weighting
    train_labels = [train_dataset.labels[i] for i in train_indices]
    
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    return train_loader, val_loader, train_labels


def progressive_unfreeze(model, epoch, total_epochs, phase1_end, phase2_end):
    """Progressive unfreezing of backbone layers."""
    if epoch <= phase1_end:
        # Phase 1: Everything frozen
        for param in model.encoder.backbone.parameters():
            param.requires_grad = False
        if epoch == 1:
            print(f"Phase 1 (Epoch {epoch}): Backbone fully frozen")
    
    elif epoch <= phase2_end:
        # Phase 2: Unfreeze last 4 blocks
        if epoch == phase1_end + 1:
            for param in model.encoder.backbone.parameters():
                param.requires_grad = False
            # Unfreeze last 4 blocks (blocks 20-23 for ViT-L/14)
            if hasattr(model.encoder.backbone, 'blocks'):
                num_blocks = len(model.encoder.backbone.blocks)
                for i in range(num_blocks - 4, num_blocks):
                    for param in model.encoder.backbone.blocks[i].parameters():
                        param.requires_grad = True
            print(f"Phase 2 (Epoch {epoch}): Last 4 blocks unfrozen")
    
    else:
        # Phase 3: Unfreeze last 8 blocks
        if epoch == phase2_end + 1:
            if hasattr(model.encoder.backbone, 'blocks'):
                num_blocks = len(model.encoder.backbone.blocks)
                for i in range(num_blocks - 8, num_blocks):
                    for param in model.encoder.backbone.blocks[i].parameters():
                        param.requires_grad = True
            print(f"Phase 3 (Epoch {epoch}): Last 8 blocks unfrozen")


def train_one_epoch_advanced(model, train_loader, criterion, optimizer, device, epoch, epochs,  
                             use_amp, use_mixup, use_cutmix, mixup_alpha, cutmix_alpha, 
                             max_grad_norm, accumulation_steps):
    """Advanced training with mixup/cutmix and gradient clipping."""
    model.train()
    total_loss = 0
    y_true, y_pred = [], []
    
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    optimizer.zero_grad()
    
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        # Apply Mixup or CutMix randomly
        mixed = False
        if use_mixup or use_cutmix:
            if np.random.rand() < 0.5:  # 50% chance of augmentation
                if use_cutmix and np.random.rand() < 0.5:
                    x, y_a, y_b, lam = cutmix_data(x, y, alpha=cutmix_alpha)
                elif use_mixup:
                    x, y_a, y_b, lam = mixup_data(x, y, alpha=mixup_alpha)
                else:
                    y_a, y_b, lam = y, y, 1.0
                mixed = True
        
        # Forward pass
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
            if mixed:
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            else:
                loss = criterion(logits, y)
            loss = loss / accumulation_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Update every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Track metrics
        total_loss += loss.item() * accumulation_steps
        preds = torch.argmax(logits, dim=1)
        if mixed:
            y_true.extend(y_a.cpu().tolist())
        else:
            y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        
        if (batch_idx + 1) % 50 == 0:
            current_acc = calculate_accuracy(torch.tensor(y_true), torch.tensor(y_pred))
            print(f"  Epoch {epoch}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | "
                  f"Loss: {loss.item()*accumulation_steps:.4f} | Acc: {current_acc:.4f}")
    
    # Final step if needed
    if len(train_loader) % accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # ============ Config ============
    dataset_path = os.getenv("DATASET_PATH")
    batch_size = int(os.getenv("BATCH_SIZE", 8))
    epochs = int(os.getenv("EPOCHS", 50))
    lr = float(os.getenv("LEARNING_RATE", 1e-4))
    num_classes = int(os.getenv("NUM_CLASSES", 2))
    image_size = int(os.getenv("IMAGE_SIZE", 224))
    model_name = os.getenv("DINOV2_MODEL", "dinov2_vitl14")
    
    # Advanced options
    use_enhanced = os.getenv("USE_ENHANCED_CLASSIFIER", "true").lower() == "true"
    use_multi_layer = os.getenv("MULTI_LAYER_FEATURES", "true").lower() == "true"
    hidden_dims_str = os.getenv("HIDDEN_DIMS", "512,256,128")
    hidden_dims = [int(x) for x in hidden_dims_str.split(',')]
    dropout = float(os.getenv("DROPOUT", 0.3))
    
    # Loss options
    use_focal = os.getenv("USE_FOCAL_LOSS", "true").lower() == "true"
    focal_alpha_str = os.getenv("FOCAL_ALPHA", "0.2,0.8")
    focal_alpha = [float(x) for x in focal_alpha_str.split(',')]
    focal_gamma = float(os.getenv("FOCAL_GAMMA", 2.0))
    label_smoothing = float(os.getenv("LABEL_SMOOTHING", 0.1))
    
    # Augmentation options
    augment_strength = os.getenv("AUGMENT_STRENGTH", "strong")
    use_mixup = os.getenv("USE_MIXUP", "true").lower() == "true"
    mixup_alpha = float(os.getenv("MIXUP_ALPHA", 0.2))
    use_cutmix = os.getenv("USE_CUTMIX", "true").lower() == "true"
    cutmix_alpha = float(os.getenv("CUTMIX_ALPHA", 1.0))
    
    # Training options
    use_amp = os.getenv("USE_AMP", "true").lower() == "true" and device.type == "cuda"
    accumulation_steps = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", 4))
    max_grad_norm = float(os.getenv("MAX_GRAD_NORM", 1.0))
    warmup_epochs = int(os.getenv("WARMUP_EPOCHS", 5))
    
    # Progressive unfreezing
    use_progressive = os.getenv("PROGRESSIVE_UNFREEZING", "true").lower() == "true"
    phase1_end = int(os.getenv("UNFREEZE_PHASE1_END", 15))
    phase2_end = int(os.getenv("UNFREEZE_PHASE2_END", 35))
    
    print(f"\n{'='*60}")
    print(f"DINOv2 Enhanced Training - Target: 90%+ Accuracy")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_path}")
    print(f"Model: {model_name}")
    print(f"Enhanced head: {use_enhanced}, Multi-layer: {use_multi_layer}")
    print(f"Batch size: {batch_size} (effective: {batch_size * accumulation_steps})")
    print(f"Epochs: {epochs}, LR: {lr}, Warmup: {warmup_epochs}")
    print(f"Focal loss: {use_focal}, Mixup: {use_mixup}, CutMix: {use_cutmix}")
    print(f"Progressive unfreezing: {use_progressive}")
    print(f"{'='*60}\n")
    
    # ============ Data ============
    train_loader, val_loader, train_labels = get_dataloaders(
        dataset_path, batch_size, image_size=image_size, augment_strength=augment_strength
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}\n")
    
    # ============ Model ============
    model = DINOv2Classifier(
        num_classes=num_classes,
        model_name=model_name,
        freeze_backbone=True,
        use_fp16=True,
        use_enhanced_head=use_enhanced,
        use_multi_layer_features=use_multi_layer,
        hidden_dims=hidden_dims if use_enhanced else None,
        dropout=dropout,
        n_last_blocks=4
    ).to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}\n")
    
    # ============ Loss Function ============
    if use_focal:
        focal_alpha_tensor = torch.tensor(focal_alpha, dtype=torch.float32).to(device)
        criterion = FocalLoss(alpha=focal_alpha_tensor, gamma=focal_gamma)
        print(f"Using Focal Loss: alpha={focal_alpha}, gamma={focal_gamma}")
    else:
        class_counts = Counter(train_labels)
        total = sum(class_counts.values())
        class_weights = [total / (len(class_counts) * class_counts[i]) for i in range(num_classes)]
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using weighted CE: weights={class_weights.tolist()}")
    
    # ============ Optimizer & Scheduler ============
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01
    )
    
    # Advanced LR scheduling with warmup
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=epochs-warmup_epochs, eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
    
    # ============ WandB ============
    init_wandb(project="mimii-anomaly-detection-enhanced", config={
        "model": "dinov2_enhanced",
        "enhanced_head": use_enhanced,
        "multi_layer_features": use_multi_layer,
        "focal_loss": use_focal,
        "mixup": use_mixup,
        "cutmix": use_cutmix,
        "epochs": epochs,
        "batch_size": batch_size * accumulation_steps,
    }, name=f"dinov2-enhanced-{model_name}")
    
    # ============ Training Loop ============
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    val_precisions, val_recalls, val_f1s = [], [], []
    best_val_f1 = 0
    
    for epoch in range(1, epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"{'='*60}")
        
        # Progressive unfreezing
        if use_progressive:
            progressive_unfreeze(model, epoch, epochs, phase1_end, phase2_end)
        
        # Train
        train_loss, train_acc = train_one_epoch_advanced(
            model, train_loader, criterion, optimizer, device, epoch, epochs,
            use_amp, use_mixup, use_cutmix, mixup_alpha, cutmix_alpha,
            max_grad_norm, accumulation_steps
        )
        
        # Validate
        val_loss, val_acc, y_true, y_pred = validate(model, val_loader, criterion, device)
        
        # Compute precision, recall, F1
        y_true_np = y_true.numpy()
        y_pred_np = y_pred.numpy()
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_np, y_pred_np, average='binary', zero_division=0
        )
        
        # Update scheduler
        scheduler.step()
        
        # Track metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        val_precisions.append(precision)
        val_recalls.append(recall)
        val_f1s.append(f1)
        
        current_lr = scheduler.get_last_lr()[0]
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"Val Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        print(f"LR: {current_lr:.6f}")
        
        log_metrics({
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "val_precision": precision, "val_recall": recall, "val_f1": f1,
            "learning_rate": current_lr,
        }, step=epoch)
        
        # Save best model (based on F1 for balanced performance)
        if f1 > best_val_f1:
            best_val_f1 = f1
            os.makedirs("model", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': f1,
                'val_acc': val_acc,
            }, "model/dinov2_best.pth")
            print(f"✓ Saved best model (F1: {f1:.4f})")
    
    # ============ Final Save & Plotting ============
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'val_acc': val_accs[-1],
    }, "model/dinov2_final.pth")
    
    results_dir = os.path.join("results", "dinov2")
    os.makedirs(results_dir, exist_ok=True)
    
    epochs_range = list(range(1, epochs + 1))
    
    plot_graph(epochs_range, [train_losses, val_losses], 
               ["Train Loss", "Validation Loss"],
               "DINOv2 Enhanced - Loss", "Epoch", "Loss",
               "loss.png", output_dir=results_dir)
    
    plot_graph(epochs_range, [train_accs, val_accs],
               ["Train Accuracy", "Validation Accuracy"],
               "DINOv2 Enhanced - Accuracy", "Epoch", "Accuracy",
               "accuracy.png", output_dir=results_dir)
    
    plot_graph(epochs_range, val_recalls, ["Validation Recall"],
               "DINOv2 Enhanced - Recall", "Epoch", "Recall",
               "recall_vs_epoch.png", output_dir=results_dir)
    
    plot_graph(epochs_range, val_precisions, ["Validation Precision"],
               "DINOv2 Enhanced - Precision", "Epoch", "Precision",
               "precision_vs_epoch.png", output_dir=results_dir)
    
    plot_graph(epochs_range, val_f1s, ["Validation F1"],
               "DINOv2 Enhanced - F1 Score", "Epoch", "F1",
               "f1_vs_epoch.png", output_dir=results_dir)
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best F1 Score: {best_val_f1:.4f}")
    print(f"Final Metrics:")
    print(f"  Accuracy:  {val_accs[-1]:.4f}")
    print(f"  Precision: {val_precisions[-1]:.4f}")
    print(f"  Recall:    {val_recalls[-1]:.4f}")
    print(f"  F1:        {val_f1s[-1]:.4f}")
    print(f"{'='*60}")
    finish_wandb()


if __name__ == "__main__":
    main()
