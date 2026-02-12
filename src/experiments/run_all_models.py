"""
Unified experiment runner for all single-model training.

Trains and evaluates: DINOv2, MAE, Swin, BEiT, DeiT, CLIP (vision), AST (audio).
Use from CLI: python -m experiments.run_all_models --model dinov2 [--epochs 20] ...
"""

import os
import sys
import argparse
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from dotenv import load_dotenv

# Add src to path when run as script
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", ".env"))

from utils.utils import calculate_accuracy
from utils.plot_graph import plot_graph
from training.optimization import autocast_context, TrainingStepHelper

try:
    from utils.experiment_tracking import init_wandb, log_metrics, finish_wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False
    def init_wandb(*args, **kwargs):
        return None
    def log_metrics(*args, **kwargs):
        pass
    def finish_wandb():
        pass

try:
    from evaluate_models import evaluate as full_evaluate
    _HAS_FULL_EVALUATE = True
except ImportError:
    _HAS_FULL_EVALUATE = False
    def full_evaluate(*args, **kwargs):
        return {}


# ---------- Shared image dataset (same structure as main_dinov2 / main_mae) ----------
class ImageDataset(torch.utils.data.Dataset):
    """
    Generic image dataset for classification.
    Expects folder structure: root_dir/class_name/images.png
    OR root_dir/machine_type/machine_id/normal|abnormal/images.png
    """

    def __init__(self, root_dir: str, transform=None):
        self.samples: List[str] = []
        self.labels: List[int] = []
        self.transform = transform
        label_map = {"normal": 0, "abnormal": 1}

        for item in os.listdir(root_dir):
            item_path = os.path.join(root_dir, item)
            if not os.path.isdir(item_path):
                continue
            if item in label_map:
                self._add_images_from_folder(item_path, label_map[item])
            else:
                for label_name in ["normal", "abnormal"]:
                    label_path = os.path.join(item_path, label_name)
                    if os.path.exists(label_path):
                        self._add_images_from_folder(label_path, label_map[label_name])
                    else:
                        for machine_id in os.listdir(item_path):
                            mid_path = os.path.join(item_path, machine_id)
                            if os.path.isdir(mid_path):
                                for ln in ["normal", "abnormal"]:
                                    lp = os.path.join(mid_path, ln)
                                    if os.path.exists(lp):
                                        self._add_images_from_folder(lp, label_map[ln])

        if self.samples:
            print(f"Loaded {len(self.samples)} images, distribution: {dict(Counter(self.labels))}")

    def _add_images_from_folder(self, folder_path: str, label: int) -> None:
        valid = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
        for f in os.listdir(folder_path):
            if any(f.lower().endswith(ext) for ext in valid):
                self.samples.append(os.path.join(folder_path, f))
                self.labels.append(label)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        image = Image.open(self.samples[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def _get_image_dataloaders(
    dataset_path: str,
    batch_size: int,
    get_transforms: Callable[..., Dict[str, Any]],
    val_split: float = 0.2,
    image_size: int = 224,
    **transform_kwargs,
) -> Tuple[DataLoader, DataLoader, List[int]]:
    """Build train/val image dataloaders using the given transform getter.
    Returns: train_loader, val_loader, train_labels (for class weighting)
    """
    transforms_dict = get_transforms(image_size=image_size, **transform_kwargs)
    train_ds = ImageDataset(dataset_path, transform=transforms_dict["train"])
    val_ds = ImageDataset(dataset_path, transform=transforms_dict["val"])
    if len(train_ds.samples) == 0:
        raise FileNotFoundError(f"No images found under {dataset_path}")
    indices = list(range(len(train_ds)))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_split, stratify=train_ds.labels, random_state=42
    )
    
    # Get train labels for class weighting
    train_labels = [train_ds.labels[i] for i in train_idx]
    
    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(val_ds, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, val_loader, train_labels


def _get_audio_dataloaders(
    dataset_path: str,
    batch_size: int,
    val_split: float = 0.2,
) -> Tuple[DataLoader, DataLoader]:
    """Build train/val audio dataloaders for AST."""
    from modules.data_loader import get_audio_dataloaders
    return get_audio_dataloaders(dataset_path, batch_size=batch_size, val_split=val_split)


# ---------- Model registry: name -> (model_builder, dataloader_builder, modality) ----------
def _build_dinov2(device: torch.device, config: Dict[str, Any]) -> nn.Module:
    from modules.dinov2_encoder import DINOv2Classifier
    model = DINOv2Classifier(
        num_classes=config["num_classes"],
        model_name=config.get("dinov2_model", "dinov2_vitb14"),
        freeze_backbone=config.get("freeze_backbone", True),
        use_fp16=config.get("use_fp16", True),
        hidden_dim=config.get("hidden_dim", 256),
        dropout=config.get("dropout", 0.1),
    ).to(device)
    return model


def _build_mae(device: torch.device, config: Dict[str, Any]) -> nn.Module:
    from modules.mae_encoder import MAEClassifier
    model = MAEClassifier(
        num_classes=config["num_classes"],
        model_name=config.get("mae_model", "vit_base_patch16_224.mae"),
        freeze_backbone=config.get("freeze_backbone", True),
        use_fp16=config.get("use_fp16", True),
        use_gradient_checkpointing=config.get("use_gradient_checkpointing", False),
        hidden_dim=config.get("hidden_dim", 256),
        dropout=config.get("dropout", 0.1),
    ).to(device)
    return model


def _build_swin(device: torch.device, config: Dict[str, Any]) -> nn.Module:
    from modules.swin_encoder import SwinClassifier
    model = SwinClassifier(
        num_classes=config["num_classes"],
        model_name=config.get("swin_model", "swin_base_patch4_window7_224"),
        freeze_backbone=config.get("freeze_backbone", True),
        use_fp16=config.get("use_fp16", True),
        use_gradient_checkpointing=config.get("use_gradient_checkpointing", False),
        hidden_dim=config.get("hidden_dim", 256),
        dropout=config.get("dropout", 0.1),
    ).to(device)
    return model


def _build_beit(device: torch.device, config: Dict[str, Any]) -> nn.Module:
    from modules.beit_encoder import BEiTClassifier
    model = BEiTClassifier(
        num_classes=config["num_classes"],
        model_name=config.get("beit_model", "beitv2_base_patch16_224"),
        freeze_backbone=config.get("freeze_backbone", True),
        use_fp16=config.get("use_fp16", True),
        use_gradient_checkpointing=config.get("use_gradient_checkpointing", False),
        hidden_dim=config.get("hidden_dim", 256),
        dropout=config.get("dropout", 0.1),
    ).to(device)
    return model


def _build_deit(device: torch.device, config: Dict[str, Any]) -> nn.Module:
    from modules.deit_encoder import DeiTClassifier
    model = DeiTClassifier(
        num_classes=config["num_classes"],
        model_name=config.get("deit_model", "deit_base_patch16_224"),
        freeze_backbone=config.get("freeze_backbone", True),
        use_fp16=config.get("use_fp16", True),
        use_gradient_checkpointing=config.get("use_gradient_checkpointing", False),
        hidden_dim=config.get("hidden_dim", 256),
        dropout=config.get("dropout", 0.1),
    ).to(device)
    return model


def _build_clip(device: torch.device, config: Dict[str, Any]) -> nn.Module:
    from modules.clip_encoder import CLIPClassifier
    model = CLIPClassifier(
        num_classes=config["num_classes"],
        model_name=config.get("clip_model", "ViT-B/16"),
        freeze_backbone=config.get("freeze_backbone", True),
        use_fp16=config.get("use_fp16", True),
        use_hf=config.get("clip_use_hf", False),
        hidden_dim=config.get("hidden_dim", 256),
        dropout=config.get("dropout", 0.1),
    ).to(device)
    return model


def _build_ast(device: torch.device, config: Dict[str, Any]) -> nn.Module:
    from modules.ast_encoder import ASTClassifier
    model = ASTClassifier(
        num_classes=config["num_classes"],
        model_name=config.get("ast_model", "MIT/ast-finetuned-audioset-10-10-0.4593"),
        freeze_backbone=config.get("freeze_backbone", True),
        use_fp16=config.get("use_fp16", True),
        hidden_dim=config.get("hidden_dim", 256),
        dropout=config.get("dropout", 0.1),
    ).to(device)
    return model


def _get_dinov2_transforms():
    from modules.dinov2_encoder import get_dinov2_transforms
    return get_dinov2_transforms


def _get_mae_transforms():
    from modules.mae_encoder import get_mae_transforms
    return get_mae_transforms


def _get_swin_transforms():
    from modules.swin_encoder import get_swin_transforms
    return get_swin_transforms


def _get_beit_transforms():
    from modules.beit_encoder import get_beit_transforms
    return get_beit_transforms


def _get_deit_transforms():
    from modules.deit_encoder import get_deit_transforms
    return get_deit_transforms


def _get_clip_transforms():
    from modules.clip_encoder import get_clip_transforms
    return get_clip_transforms


# (model_builder, data_getter, transform_getter or None for audio)
_MODEL_REGISTRY: Dict[str, Tuple[Callable, Callable, Optional[Callable]]] = {
    "dinov2": (_build_dinov2, _get_image_dataloaders, _get_dinov2_transforms),
    "mae": (_build_mae, _get_image_dataloaders, _get_mae_transforms),
    "swin": (_build_swin, _get_image_dataloaders, _get_swin_transforms),
    "beit": (_build_beit, _get_image_dataloaders, _get_beit_transforms),
    "deit": (_build_deit, _get_image_dataloaders, _get_deit_transforms),
    "clip": (_build_clip, _get_image_dataloaders, _get_clip_transforms),
    "ast": (_build_ast, _get_audio_dataloaders, None),
}


def list_models() -> List[str]:
    """Return list of supported model names."""
    return list(_MODEL_REGISTRY.keys())


def get_default_config(model_name: str) -> Dict[str, Any]:
    """Return default config for a model (env overrides applied in run_single_model)."""
    defaults = {
        "dataset_path": os.getenv("DATASET_PATH", "."),
        "batch_size": int(os.getenv("BATCH_SIZE", 16)),
        "epochs": int(os.getenv("EPOCHS", 20)),
        "learning_rate": float(os.getenv("LEARNING_RATE", 1e-4)),
        "num_classes": int(os.getenv("NUM_CLASSES", 2)),
        "image_size": int(os.getenv("IMAGE_SIZE", 224)),
        "val_split": 0.2,
        "freeze_backbone": True,
        "use_fp16": True,
        "use_gradient_checkpointing": False,
        "hidden_dim": 256,
        "dropout": 0.1,
        "use_amp": os.getenv("USE_AMP", "true").lower() == "true",
        "accumulation_steps": max(1, int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "1"))),
        "output_dir": "model",
        "results_dir": "results",
    }
    if model_name == "ast":
        defaults["learning_rate"] = 1e-5
        defaults["batch_size"] = min(defaults["batch_size"], 24)
    return defaults


def _train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    step_helper: TrainingStepHelper,
    device: torch.device,
    epoch: int,
    epochs: int,
    use_amp: bool,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    y_true, y_pred = [], []
    step_helper.zero_grad()
    step_helper.reset_step_count()

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        with autocast_context(use_amp, device):
            logits = model(x)
            loss = criterion(logits, y)
        step_helper.step(loss)
        total_loss += loss.item()
        preds = torch.argmax(logits.detach(), dim=1)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        if (batch_idx + 1) % 10 == 0 or batch_idx == len(train_loader) - 1:
            acc = calculate_accuracy(torch.tensor(y_true), torch.tensor(y_pred))
            print(f"  Epoch {epoch}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")

    step_helper.finalize_step()
    avg_loss = total_loss / len(train_loader)
    accuracy = calculate_accuracy(torch.tensor(y_true), torch.tensor(y_pred))
    return avg_loss, accuracy


@torch.no_grad()
def _validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, torch.Tensor, torch.Tensor]:
    model.eval()
    total_loss = 0.0
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
    acc = calculate_accuracy(torch.tensor(y_true), torch.tensor(y_pred))
    return avg_loss, acc, torch.tensor(y_true), torch.tensor(y_pred)


@torch.no_grad()
def _collect_val_predictions_and_probs(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run model on validation set and return (y_true, logits) for full evaluation."""
    model.eval()
    all_logits, all_labels = [], []
    for x, y in val_loader:
        x = x.to(device)
        logits = model(x)
        all_logits.append(logits.cpu())
        all_labels.append(y)
    return torch.cat(all_labels), torch.cat(all_logits)


def run_single_model(
    model_name: str,
    config: Optional[Dict[str, Any]] = None,
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train and evaluate a single model. Returns metrics and saves checkpoints.

    Args:
        model_name: One of list_models() (e.g. dinov2, mae, swin, beit, deit, clip, ast).
        config: Override config (merged with get_default_config(model_name)).
        run_name: Optional run name for wandb and save paths.

    Returns:
        Dict with train/val losses and accuracies per epoch, best_val_acc, and final metrics.
    """
    model_name = model_name.lower().strip()
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list_models()}")

    defaults = get_default_config(model_name)
    cfg = {**defaults, **(config or {})}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    use_amp = cfg.get("use_amp", True) and device.type == "cuda"
    accumulation_steps = cfg.get("accumulation_steps", 1)

    # Dataloaders
    model_builder, data_getter, transform_getter = _MODEL_REGISTRY[model_name]
    train_labels = None  # For class weighting (image models only)
    
    if transform_getter is not None:
        get_transforms = transform_getter()
        train_loader, val_loader, train_labels = data_getter(
            cfg["dataset_path"],
            cfg["batch_size"],
            get_transforms=get_transforms,
            val_split=cfg.get("val_split", 0.2),
            image_size=cfg.get("image_size", 224),
        )
    else:
        train_loader, val_loader = data_getter(
            cfg["dataset_path"],
            cfg["batch_size"],
            val_split=cfg.get("val_split", 0.2),
        )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    model = model_builder(device, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    # Compute class weights for image models (helps with class imbalance)
    class_weights = None
    if train_labels is not None and cfg.get("use_balanced_loss", True):
        class_counts = Counter(train_labels)
        total = sum(class_counts.values())
        num_classes = cfg.get("num_classes", 2)
        class_weights = [total / (num_classes * class_counts[i]) for i in range(num_classes)]
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        print(f"Class distribution: {dict(class_counts)}")
        print(f"Using balanced class weights: {class_weights.tolist()}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["learning_rate"],
        weight_decay=0.01,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    step_helper = TrainingStepHelper(
        optimizer, use_amp=use_amp, device=device, accumulation_steps=accumulation_steps
    )

    # Wandb
    name = run_name or f"{model_name}-run"
    if _HAS_WANDB:
        init_wandb(
            project="mimii-anomaly-detection",
            config={"model": model_name, **cfg},
            name=name,
        )

    # Training loop
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    val_precisions, val_recalls, val_f1s = [], [], []
    best_val_acc = 0.0
    epochs = cfg["epochs"]
    output_dir = cfg.get("output_dir", "model")
    results_dir = cfg.get("results_dir", "results")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs} " + "-" * 40)
        train_loss, train_acc = _train_one_epoch(
            model, train_loader, criterion, step_helper, device, epoch, epochs, use_amp
        )
        val_loss, val_acc, y_true, y_pred = _validate(model, val_loader, criterion, device)
        
        # Compute precision, recall, F1 for validation
        y_true_np = y_true.numpy()
        y_pred_np = y_pred.numpy()
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_np, y_pred_np, average='binary', zero_division=0
        )
        
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        val_precisions.append(precision)
        val_recalls.append(recall)
        val_f1s.append(f1)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"  Val Precision: {precision:.4f} | Val Recall: {recall:.4f} | Val F1: {f1:.4f}")
        if _HAS_WANDB:
            log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_precision": precision,
                "val_recall": recall,
                "val_f1": f1,
                "learning_rate": scheduler.get_last_lr()[0],
            }, step=epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, os.path.join(output_dir, f"{model_name}_best.pth"))
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.4f})")

    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "val_acc": val_accs[-1],
    }, os.path.join(output_dir, f"{model_name}_final.pth"))

    # Per-model results subdir for plots and evaluation artifacts
    model_results_dir = os.path.join(results_dir, model_name)
    os.makedirs(model_results_dir, exist_ok=True)

    # Plots
    epochs_range = list(range(1, epochs + 1))
    plot_graph(
        x=epochs_range,
        y=[train_losses, val_losses],
        labels=["Train Loss", "Validation Loss"],
        title=f"{model_name} - Train vs Validation Loss",
        xlabel="Epoch",
        ylabel="Loss",
        filename="loss.png",
        output_dir=model_results_dir,
    )
    plot_graph(
        x=epochs_range,
        y=[train_accs, val_accs],
        labels=["Train Accuracy", "Validation Accuracy"],
        title=f"{model_name} - Train vs Validation Accuracy",
        xlabel="Epoch",
        ylabel="Accuracy",
        filename="accuracy.png",
        output_dir=model_results_dir,
    )
    
    # Add precision, recall, F1 curves
    plot_graph(
        x=epochs_range,
        y=val_recalls,
        labels=["Validation Recall"],
        title=f"{model_name} - Validation Recall vs Epoch",
        xlabel="Epoch",
        ylabel="Recall",
        filename="recall_vs_epoch.png",
        output_dir=model_results_dir,
    )
    plot_graph(
        x=epochs_range,
        y=val_precisions,
        labels=["Validation Precision"],
        title=f"{model_name} - Validation Precision vs Epoch",
        xlabel="Epoch",
        ylabel="Precision",
        filename="precision_vs_epoch.png",
        output_dir=model_results_dir,
    )
    plot_graph(
        x=epochs_range,
        y=val_f1s,
        labels=["Validation F1"],
        title=f"{model_name} - Validation F1 vs Epoch",
        xlabel="Epoch",
        ylabel="F1 Score",
        filename="f1_vs_epoch.png",
        output_dir=model_results_dir,
    )

    # Full evaluation (confusion matrix, ROC, PR, MCC, Cohen's Kappa, classification report)
    evaluation_metrics = {}
    if _HAS_FULL_EVALUATE:
        best_ckpt_path = os.path.join(output_dir, f"{model_name}_best.pth")
        if os.path.isfile(best_ckpt_path):
            ckpt = torch.load(best_ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
        y_true_val, logits_val = _collect_val_predictions_and_probs(model, val_loader, device)
        y_true_np = y_true_val.numpy()
        probs_val = F.softmax(logits_val, dim=1).numpy()
        evaluation_metrics = full_evaluate(
            y_true_np, probs_val, epoch="best", out_dir=model_results_dir
        )
        if evaluation_metrics:
            print(f"\n--- Full evaluation ({model_name}) ---")
            for k, v in evaluation_metrics.items():
                if k not in ("cm", "report") and isinstance(v, (int, float)):
                    print(f"  {k}: {v}")

    if _HAS_WANDB:
        finish_wandb()

    print(f"\n{'='*60}")
    print(f"Training Complete for {model_name}!")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Final Val Precision: {val_precisions[-1]:.4f}")
    print(f"Final Val Recall: {val_recalls[-1]:.4f}")
    print(f"Final Val F1: {val_f1s[-1]:.4f}")
    print(f"{'='*60}")

    return {
        "model_name": model_name,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accs,
        "val_accuracies": val_accs,
        "val_precisions": val_precisions,
        "val_recalls": val_recalls,
        "val_f1s": val_f1s,
        "best_val_acc": best_val_acc,
        "final_val_acc": val_accs[-1],
        "final_val_precision": val_precisions[-1],
        "final_val_recall": val_recalls[-1],
        "final_val_f1": val_f1s[-1],
        "evaluation_metrics": evaluation_metrics,
        "config": cfg,
    }


def main():
    parser = argparse.ArgumentParser(description="Unified runner for all single models")
    parser.add_argument("--model", type=str, required=True, choices=list_models(),
                        help="Model to train (e.g. dinov2, mae, ast)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--dataset_path", type=str, default=None, help="Dataset root")
    parser.add_argument("--output_dir", type=str, default=None, help="Checkpoint directory")
    parser.add_argument("--results_dir", type=str, default=None, help="Plots/results directory")
    parser.add_argument("--name", type=str, default=None, help="Run name for wandb")
    parser.add_argument("--run_all", action="store_true", help="Run all models sequentially")
    args = parser.parse_args()

    if args.run_all:
        results = {}
        for m in list_models():
            print(f"\n{'='*60}\nRunning model: {m}\n{'='*60}")
            overrides = {}
            if args.epochs is not None:
                overrides["epochs"] = args.epochs
            if args.batch_size is not None:
                overrides["batch_size"] = args.batch_size
            if args.lr is not None:
                overrides["learning_rate"] = args.lr
            if args.dataset_path is not None:
                overrides["dataset_path"] = args.dataset_path
            if args.output_dir is not None:
                overrides["output_dir"] = args.output_dir
            if args.results_dir is not None:
                overrides["results_dir"] = args.results_dir
            try:
                results[m] = run_single_model(m, config=overrides or None, run_name=args.name or f"{m}-batch")
            except Exception as e:
                print(f"Model {m} failed: {e}")
                results[m] = {"error": str(e)}
        print("\n--- Summary: best_val_acc ---")
        for k, v in results.items():
            err = v.get("error")
            if err:
                print(f"  {k}: ERROR - {err}")
            else:
                print(f"  {k}: {v.get('best_val_acc', 'N/A'):.4f}")
                ev = v.get("evaluation_metrics") or {}
                if ev:
                    acc, f1, auc_roc = ev.get("accuracy"), ev.get("f1"), ev.get("auc_roc")
                    if acc is not None or f1 is not None or auc_roc is not None:
                        print(f"    evaluation: accuracy={acc}, f1={f1}, auc_roc={auc_roc}")
        return

    overrides = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["learning_rate"] = args.lr
    if args.dataset_path is not None:
        overrides["dataset_path"] = args.dataset_path
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.results_dir is not None:
        overrides["results_dir"] = args.results_dir

    run_single_model(args.model, config=overrides or None, run_name=args.name)


if __name__ == "__main__":
    main()
