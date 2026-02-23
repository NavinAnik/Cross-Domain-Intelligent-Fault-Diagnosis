"""
Ablation experiment runners: model architecture, fusion, contrastive, modality.

- Model architecture: DINOv2, MAE, Swin, BEiT, DeiT, CLIP, AST
- Fusion ablations: Early, Late, Cross-Attention, ConFu, ProMoBal
- Contrastive loss ablations: SimCLR (NT-Xent), MoCo, BYOL, VICReg
- Modality ablations: Audio-only, Vision-only, Multimodal (early fusion)

Use from CLI: python -m experiments.run_ablations --mode fusion|contrastive|modality|models|all [options]
"""

import os
import sys
import argparse
import json
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from dotenv import load_dotenv

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", ".env"))

from utils.utils import calculate_accuracy
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


# ---------- Fusion ablation ----------
FUSION_STRATEGIES = ["early", "late", "cross_attention", "confu", "promobal"]


def _build_audio_encoder(device: torch.device, num_classes: int = 2) -> nn.Module:
    """AST encoder for audio branch (feature_dim=768)."""
    from modules.ast_encoder import ASTClassifier
    model = ASTClassifier(
        num_classes=num_classes,
        model_name=os.getenv("AST_MODEL", "MIT/ast-finetuned-audioset-10-10-0.4593"),
        freeze_backbone=True,
        use_fp16=True,
        hidden_dim=None,
        dropout=0.1,
    ).to(device)
    return model.encoder


def _build_vision_encoder(device: torch.device, num_classes: int = 2) -> nn.Module:
    """DINOv2 encoder for vision branch (frozen)."""
    from modules.dinov2_encoder import DINOv2Classifier
    model = DINOv2Classifier(
        num_classes=num_classes,
        model_name=os.getenv("DINOV2_MODEL", "dinov2_vitb14"),
        freeze_backbone=True,
        use_fp16=True,
        hidden_dim=None,
        dropout=0.1,
    ).to(device)
    return model.encoder


def _get_fusion_model(
    fusion_name: str,
    audio_encoder: nn.Module,
    vision_encoder: nn.Module,
    num_classes: int = 2,
) -> nn.Module:
    """Build fusion model by strategy name."""
    fusion_name = fusion_name.lower()
    if fusion_name == "early":
        from fusion.early_fusion import EarlyFusion
        return EarlyFusion(
            audio_encoder=audio_encoder,
            vision_encoder=vision_encoder,
            num_classes=num_classes,
            hidden_dim=256,
            dropout=0.1,
        )
    if fusion_name == "late":
        from fusion.late_fusion import LateFusion
        return LateFusion(
            audio_encoder=audio_encoder,
            vision_encoder=vision_encoder,
            num_classes=num_classes,
            alpha_init=0.5,
            learn_alpha=True,
            hidden_dim=256,
            dropout=0.1,
        )
    if fusion_name == "cross_attention":
        from fusion.cross_attention_fusion import CrossAttentionFusion
        return CrossAttentionFusion(
            audio_encoder=audio_encoder,
            vision_encoder=vision_encoder,
            num_classes=num_classes,
            hidden_dim=256,
            num_heads=8,
            dropout=0.1,
        )
    if fusion_name == "confu":
        from fusion.confu_fusion import ConFuFusion
        return ConFuFusion(
            audio_encoder=audio_encoder,
            vision_encoder=vision_encoder,
            num_classes=num_classes,
            hidden_dim=256,
            dropout=0.1,
        )
    if fusion_name == "promobal":
        from fusion.promobal_fusion import ProMoBalFusion
        return ProMoBalFusion(
            audio_encoder=audio_encoder,
            vision_encoder=vision_encoder,
            num_classes=num_classes,
            hidden_dim=256,
            dropout=0.1,
        )
    raise ValueError(f"Unknown fusion: {fusion_name}. Choose from {FUSION_STRATEGIES}")


class PairedAudioImageDataset(Dataset):
    """
    Paired (audio, image) dataset by index alignment.
    Expects audio_loader and image_loader to iterate over aligned pairs
    (same order / same split). Use same seed and split for both.
    """

    def __init__(self, audio_dataset: Dataset, image_dataset: Dataset):
        assert len(audio_dataset) == len(image_dataset)
        self.audio_dataset = audio_dataset
        self.image_dataset = image_dataset

    def __len__(self) -> int:
        return len(self.audio_dataset)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, int]:
        a, label_a = self.audio_dataset[idx]
        img, label_b = self.image_dataset[idx]
        assert label_a == label_b
        return a, img, label_a


def _get_paired_dataloaders_simple(
    audio_dataset_path: str,
    image_dataset_path: str,
    batch_size: int,
    image_size: int = 224,
    val_split: float = 0.2,
) -> Tuple[Tuple[DataLoader, DataLoader], Tuple[DataLoader, DataLoader]]:
    """
    Build audio and image loaders separately with same random_state.
    Returns (train_audio, train_img), (val_audio, val_img). Use in lockstep.
    """
    from modules.data_loader import get_audio_dataloaders
    from .run_all_models import _get_image_dataloaders, _get_dinov2_transforms

    train_audio, val_audio = get_audio_dataloaders(
        audio_dataset_path, batch_size=batch_size, val_split=val_split
    )
    train_img, val_img = _get_image_dataloaders(
        image_dataset_path,
        batch_size,
        _get_dinov2_transforms(),
        val_split=val_split,
        image_size=image_size,
    )
    return (train_audio, train_img), (val_audio, val_img)


def run_fusion_ablation(
    fusion_name: str,
    audio_dataset_path: str,
    image_dataset_path: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Train one fusion strategy (early, late, cross_attention, confu, promobal)
    with AST (audio) + DINOv2 (vision). Returns metrics.
    """
    cfg = {
        "batch_size": int(os.getenv("BATCH_SIZE", 8)),
        "epochs": int(os.getenv("EPOCHS", 15)),
        "learning_rate": float(os.getenv("LEARNING_RATE", 1e-4)),
        "num_classes": int(os.getenv("NUM_CLASSES", 2)),
        "image_size": 224,
        "val_split": 0.2,
        "use_amp": os.getenv("USE_AMP", "true").lower() == "true",
        "accumulation_steps": max(1, int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "1"))),
        **(config or {}),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paired loaders: (train_audio, train_img), (val_audio, val_img)
    (train_audio, train_img), (val_audio, val_img) = _get_paired_dataloaders_simple(
        audio_dataset_path,
        image_dataset_path,
        cfg["batch_size"],
        image_size=cfg["image_size"],
        val_split=cfg["val_split"],
    )
    # Iterate in lockstep (assume same length or use min length)
    n_train = min(len(train_audio.dataset), len(train_img.dataset))
    n_val = min(len(val_audio.dataset), len(val_img.dataset))
    train_audio = DataLoader(
        torch.utils.data.Subset(train_audio.dataset, range(n_train)),
        batch_size=cfg["batch_size"],
        shuffle=True,
    )
    train_img = DataLoader(
        torch.utils.data.Subset(train_img.dataset, range(n_train)),
        batch_size=cfg["batch_size"],
        shuffle=True,
    )
    val_audio = DataLoader(
        torch.utils.data.Subset(val_audio.dataset, range(n_val)),
        batch_size=cfg["batch_size"],
        shuffle=False,
    )
    val_img = DataLoader(
        torch.utils.data.Subset(val_img.dataset, range(n_val)),
        batch_size=cfg["batch_size"],
        shuffle=False,
    )

    audio_enc = _build_audio_encoder(device, cfg["num_classes"])
    vision_enc = _build_vision_encoder(device, cfg["num_classes"])
    model = _get_fusion_model(fusion_name, audio_enc, vision_enc, cfg["num_classes"]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=0.01)
    step_helper = TrainingStepHelper(
        optimizer,
        use_amp=cfg["use_amp"],
        device=device,
        accumulation_steps=cfg["accumulation_steps"],
    )

    if _HAS_WANDB:
        init_wandb(
            project="mimii-anomaly-detection",
            config={"ablation": "fusion", "fusion": fusion_name, **cfg},
            name=f"fusion-{fusion_name}",
        )

    train_losses, val_accs = [], []
    best_val_acc = 0.0
    use_amp = cfg["use_amp"] and device.type == "cuda"

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        step_helper.zero_grad()
        step_helper.reset_step_count()
        it_audio = iter(train_audio)
        it_img = iter(train_img)
        n_batches = min(len(train_audio), len(train_img))
        for _ in range(n_batches):
            batch_audio = next(it_audio, None)
            batch_img = next(it_img, None)
            if batch_audio is None or batch_img is None:
                break
            x_a, y_a = batch_audio[0].to(device), batch_audio[1].to(device)
            x_v, y_v = batch_img[0].to(device), batch_img[1].to(device)
            with autocast_context(use_amp, device):
                logits = model(x_a, x_v)
                loss = criterion(logits, y_a)
            step_helper.step(loss)
            total_loss += loss.item()
        step_helper.finalize_step()
        avg_loss = total_loss / max(n_batches, 1)
        train_losses.append(avg_loss)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            it_audio = iter(val_audio)
            it_img = iter(val_img)
            for _ in range(min(len(val_audio), len(val_img))):
                batch_audio = next(it_audio, None)
                batch_img = next(it_img, None)
                if batch_audio is None or batch_img is None:
                    break
                x_a = batch_audio[0].to(device)
                x_v = batch_img[0].to(device)
                y = batch_audio[1].to(device)
                logits = model(x_a, x_v)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)
        val_acc = val_correct / max(val_total, 1)
        val_accs.append(val_acc)
        if _HAS_WANDB:
            log_metrics({"train_loss": avg_loss, "val_acc": val_acc}, step=epoch)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        print(f"  Fusion {fusion_name} Epoch {epoch} | Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

    if _HAS_WANDB:
        finish_wandb()
    return {
        "fusion": fusion_name,
        "train_losses": train_losses,
        "val_accuracies": val_accs,
        "best_val_acc": best_val_acc,
        "config": cfg,
    }


def run_all_fusion_ablations(
    audio_dataset_path: str,
    image_dataset_path: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run all fusion strategies and return metrics per strategy."""
    results = {}
    for name in FUSION_STRATEGIES:
        print(f"\n{'='*60}\nFusion ablation: {name}\n{'='*60}")
        try:
            results[name] = run_fusion_ablation(
                name, audio_dataset_path, image_dataset_path, config
            )
        except Exception as e:
            print(f"Fusion {name} failed: {e}")
            results[name] = {"error": str(e)}
    return results


# ---------- Contrastive loss ablation ----------
CONTRASTIVE_LOSSES = ["simclr", "moco", "byol", "vicreg"]


def run_contrastive_ablation(
    loss_name: str,
    dataset_path: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run contrastive pretraining with one loss: simclr (NT-Xent), moco, byol, vicreg.
    Uses a small encoder + projector; returns loss curve and optional linear eval acc.
    """
    cfg = {
        "batch_size": int(os.getenv("BATCH_SIZE", 32)),
        "epochs": int(os.getenv("EPOCHS", 30)),
        "learning_rate": float(os.getenv("LEARNING_RATE", 1e-3)),
        "num_classes": int(os.getenv("NUM_CLASSES", 2)),
        "projection_dim": 128,
        "temperature": 0.5,
        "use_amp": os.getenv("USE_AMP", "true").lower() == "true",
        "accumulation_steps": max(1, int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "1"))),
        **(config or {}),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from modules.data_loader import get_audio_dataloaders
    from modules.encoder import Encoder

    train_loader, val_loader = get_audio_dataloaders(
        dataset_path, batch_size=cfg["batch_size"], val_split=0.2
    )
    encoder = Encoder(
        img_size=224,
        patch_size=16,
        emb_dim=768,
        num_layers=12,
        num_heads=12,
        mlp_dim=3072,
        dropout=0.1,
    ).to(device)
    proj_dim = cfg["projection_dim"]
    projector = nn.Sequential(
        nn.Linear(encoder.feature_dim, encoder.feature_dim),
        nn.ReLU(),
        nn.Linear(encoder.feature_dim, proj_dim),
    ).to(device)

    loss_name = loss_name.lower()
    moco_queue = None
    target_encoder = None
    target_projector = None
    predictor = None
    if loss_name == "simclr":
        from modules.contrastive_loss import nt_xent_loss
    elif loss_name == "moco":
        from modules.moco_loss import moco_info_nce_loss, MoCoQueue
        moco_queue = MoCoQueue(dim=proj_dim, K=min(4096, len(train_loader.dataset)), device=device)
    elif loss_name == "byol":
        from modules.byol_loss import byol_loss_symmetric
        target_encoder = Encoder(
            img_size=224, patch_size=16, emb_dim=768,
            num_layers=12, num_heads=12, mlp_dim=3072, dropout=0.1,
        ).to(device)
        target_encoder.load_state_dict(encoder.state_dict())
        target_projector = nn.Sequential(
            nn.Linear(encoder.feature_dim, encoder.feature_dim),
            nn.ReLU(),
            nn.Linear(encoder.feature_dim, proj_dim),
        ).to(device)
        target_projector.load_state_dict(projector.state_dict())
        predictor = nn.Sequential(
            nn.Linear(proj_dim, encoder.feature_dim),
            nn.ReLU(),
            nn.Linear(encoder.feature_dim, proj_dim),
        ).to(device)
    elif loss_name != "vicreg":
        raise ValueError(f"Unknown contrastive loss: {loss_name}. Choose from {CONTRASTIVE_LOSSES}")

    params = list(encoder.parameters()) + list(projector.parameters())
    if predictor is not None:
        params += list(predictor.parameters())
    optimizer = optim.Adam(params, lr=cfg["learning_rate"])
    step_helper = TrainingStepHelper(
        optimizer,
        use_amp=cfg["use_amp"],
        device=device,
        accumulation_steps=cfg["accumulation_steps"],
    )
    moco_momentum = 0.999
    byol_momentum = 0.996

    if _HAS_WANDB:
        init_wandb(
            project="mimii-anomaly-detection",
            config={"ablation": "contrastive", "loss": loss_name, **cfg},
            name=f"contrastive-{loss_name}",
        )

    train_losses = []
    use_amp = cfg["use_amp"] and device.type == "cuda"
    for epoch in range(1, cfg["epochs"] + 1):
        encoder.train()
        projector.train()
        if predictor is not None:
            predictor.train()
        total_loss = 0.0
        step_helper.zero_grad()
        step_helper.reset_step_count()
        for x, _ in train_loader:
            x = x.to(device)
            x1, x2 = x, x  # TODO: apply different augmentations per view
            with autocast_context(use_amp, device):
                if loss_name == "simclr":
                    from modules.contrastive_loss import nt_xent_loss
                    z1 = projector(encoder(x1))
                    z2 = projector(encoder(x2))
                    loss = nt_xent_loss(z1, z2, temperature=cfg.get("temperature", 0.5))
                elif loss_name == "moco":
                    from modules.moco_loss import moco_info_nce_loss
                    query = projector(encoder(x1))
                    with torch.no_grad():
                        key = projector(encoder(x2))
                    loss = moco_info_nce_loss(
                        query, key, moco_queue.get_queue(),
                        temperature=cfg.get("temperature", 0.07),
                    )
                    moco_queue.enqueue_dequeue(key)
                elif loss_name == "byol":
                    from modules.byol_loss import byol_loss_symmetric
                    pred1 = predictor(projector(encoder(x1)))
                    pred2 = predictor(projector(encoder(x2)))
                    with torch.no_grad():
                        t1 = target_projector(target_encoder(x1))
                        t2 = target_projector(target_encoder(x2))
                    loss = byol_loss_symmetric(pred1, t2, pred2, t1)
                    with torch.no_grad():
                        for p, pt in zip(encoder.parameters(), target_encoder.parameters()):
                            pt.data.mul_(byol_momentum).add_(p.data, alpha=1 - byol_momentum)
                        for p, pt in zip(projector.parameters(), target_projector.parameters()):
                            pt.data.mul_(byol_momentum).add_(p.data, alpha=1 - byol_momentum)
                else:
                    from modules.vicreg_loss import vicreg_loss
                    z1 = projector(encoder(x1))
                    z2 = projector(encoder(x2))
                    loss = vicreg_loss(z1, z2, lambda_inv=25.0, lambda_var=25.0, lambda_cov=1.0)
            step_helper.step(loss)
            total_loss += loss.item()
        step_helper.finalize_step()
        avg = total_loss / len(train_loader)
        train_losses.append(avg)
        if _HAS_WANDB:
            log_metrics({"train_loss": avg}, step=epoch)
        print(f"  Contrastive {loss_name} Epoch {epoch} | Loss: {avg:.4f}")

    if _HAS_WANDB:
        finish_wandb()
    return {
        "loss": loss_name,
        "train_losses": train_losses,
        "config": cfg,
    }


def run_all_contrastive_ablations(
    dataset_path: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run all contrastive losses and return metrics per loss."""
    results = {}
    for name in CONTRASTIVE_LOSSES:
        print(f"\n{'='*60}\nContrastive ablation: {name}\n{'='*60}")
        try:
            results[name] = run_contrastive_ablation(name, dataset_path, config)
        except Exception as e:
            print(f"Contrastive {name} failed: {e}")
            results[name] = {"error": str(e)}
    return results


# ---------- Modality ablation (Audio-only vs Vision-only vs Multimodal) ----------
MODALITY_ABLATION_NAMES = ["audio_only", "vision_only", "multimodal"]


def run_modality_ablation(
    modality: str,
    audio_dataset_path: str,
    image_dataset_path: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run one modality setting: audio_only (AST), vision_only (DINOv2), or multimodal (early fusion).
    Returns metrics for fair comparison (same epochs, batch size, data).
    """
    cfg = {
        "batch_size": int(os.getenv("BATCH_SIZE", 8)),
        "epochs": int(os.getenv("EPOCHS", 15)),
        "learning_rate": float(os.getenv("LEARNING_RATE", 1e-4)),
        "num_classes": int(os.getenv("NUM_CLASSES", 2)),
        "image_size": 224,
        "val_split": 0.2,
        "use_amp": os.getenv("USE_AMP", "true").lower() == "true",
        "accumulation_steps": max(1, int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "1"))),
        **(config or {}),
    }
    modality = modality.lower()
    if modality == "audio_only":
        from experiments.run_all_models import run_single_model
        return run_single_model(
            "ast",
            config={
                "dataset_path": audio_dataset_path,
                "epochs": cfg["epochs"],
                "batch_size": cfg["batch_size"],
                "learning_rate": cfg["learning_rate"],
            },
            run_name="ablation-modality-audio",
        )
    if modality == "vision_only":
        from experiments.run_all_models import run_single_model
        return run_single_model(
            "dinov2",
            config={
                "dataset_path": image_dataset_path,
                "epochs": cfg["epochs"],
                "batch_size": cfg["batch_size"],
                "learning_rate": cfg["learning_rate"],
            },
            run_name="ablation-modality-vision",
        )
    if modality == "multimodal":
        out = run_fusion_ablation(
            "early",
            audio_dataset_path,
            image_dataset_path,
            config=cfg,
        )
        out["modality"] = "multimodal"
        return out
    raise ValueError(
        f"Unknown modality: {modality}. Choose from {MODALITY_ABLATION_NAMES}"
    )


def run_all_modality_ablations(
    audio_dataset_path: str,
    image_dataset_path: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run audio-only, vision-only, and multimodal (early fusion); return metrics per modality."""
    results = {}
    for name in MODALITY_ABLATION_NAMES:
        print(f"\n{'='*60}\nModality ablation: {name}\n{'='*60}")
        try:
            r = run_modality_ablation(
                name, audio_dataset_path, image_dataset_path, config
            )
            results[name] = {
                "modality": name,
                "best_val_acc": r.get("best_val_acc"),
                "final_val_acc": r.get("final_val_acc", r.get("val_accuracies", [0])[-1] if r.get("val_accuracies") else 0),
                "train_losses": r.get("train_losses", []),
                "val_accuracies": r.get("val_accuracies", []),
                "config": r.get("config", {}),
            }
        except Exception as e:
            print(f"Modality {name} failed: {e}")
            results[name] = {"modality": name, "error": str(e)}
    return results


# ---------- Comprehensive ablation runner ----------
def run_comprehensive_ablations(
    audio_dataset_path: str,
    image_dataset_path: str,
    dataset_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    results_dir: str = "results",
    run_models: bool = True,
    run_fusion: bool = True,
    run_contrastive: bool = True,
    run_modality: bool = True,
    model_list: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Execute comprehensive ablation studies and save results to results_dir.

    1. Model architecture: train/eval each of DINOv2, MAE, Swin, BEiT, DeiT, CLIP, AST.
    2. Fusion strategies: Early, Late, Cross-Attention, ConFu, ProMoBal.
    3. Contrastive losses: SimCLR, MoCo, BYOL, VICReg.
    4. Modality: Audio-only, Vision-only, Multimodal.

    Saves: ablation_models.json, ablation_fusion.json, ablation_contrastive.json,
           ablation_modality.json, and optional comparative report.
    """
    data_path = dataset_path or audio_dataset_path or image_dataset_path
    if not data_path:
        raise ValueError("Provide audio_dataset_path and image_dataset_path (or dataset_path for both).")

    os.makedirs(results_dir, exist_ok=True)
    cfg = config or {}
    summary = {"results_dir": results_dir, "model_architecture": {}, "fusion": {}, "contrastive": {}, "modality": {}}

    # 1. Model architecture ablation
    if run_models:
        from experiments.run_all_models import list_models, run_single_model
        models_to_run = model_list or list_models()
        model_results_list = []
        for i, model_name in enumerate(models_to_run):
            print(f"\n{'='*60}\n[Ablation: Model architecture] ({i+1}/{len(models_to_run)}) {model_name}\n{'='*60}")
            path = image_dataset_path if model_name != "ast" else audio_dataset_path
            if not path:
                path = data_path
            try:
                out = run_single_model(
                    model_name,
                    config={"dataset_path": path, "results_dir": results_dir, **cfg},
                    run_name=f"ablation-model-{model_name}",
                )
                summary["model_architecture"][model_name] = {
                    "best_val_acc": out.get("best_val_acc"),
                    "final_val_acc": out.get("final_val_acc"),
                    "error": None,
                }
                model_results_list.append({
                    "model_name": model_name,
                    "best_val_acc": out.get("best_val_acc"),
                    "final_val_acc": out.get("final_val_acc"),
                    "evaluation_metrics": out.get("evaluation_metrics"),
                })
            except Exception as e:
                print(f"Model {model_name} failed: {e}")
                summary["model_architecture"][model_name] = {"error": str(e)}
                model_results_list.append({"model_name": model_name, "error": str(e)})
        with open(os.path.join(results_dir, "ablation_models.json"), "w") as f:
            json.dump(model_results_list, f, indent=2)
        with open(os.path.join(results_dir, "results.json"), "w") as f:
            json.dump(model_results_list, f, indent=2)

    # 2. Fusion strategy ablation
    if run_fusion:
        fusion_results = run_all_fusion_ablations(
            audio_dataset_path or data_path,
            image_dataset_path or data_path,
            config=cfg,
        )
        summary["fusion"] = {k: {"best_val_acc": v.get("best_val_acc"), "error": v.get("error")} for k, v in fusion_results.items()}
        with open(os.path.join(results_dir, "ablation_fusion.json"), "w") as f:
            json.dump(fusion_results, f, indent=2)

    # 3. Contrastive loss ablation
    if run_contrastive:
        contrastive_results = run_all_contrastive_ablations(
            data_path,
            config=cfg,
        )
        summary["contrastive"] = {k: {"final_loss": (v.get("train_losses") or [0])[-1], "error": v.get("error")} for k, v in contrastive_results.items()}
        with open(os.path.join(results_dir, "ablation_contrastive.json"), "w") as f:
            json.dump(contrastive_results, f, indent=2)

    # 4. Modality ablation
    if run_modality:
        modality_results = run_all_modality_ablations(
            audio_dataset_path or data_path,
            image_dataset_path or data_path,
            config=cfg,
        )
        summary["modality"] = {k: {"best_val_acc": v.get("best_val_acc"), "error": v.get("error")} for k, v in modality_results.items()}
        with open(os.path.join(results_dir, "ablation_modality.json"), "w") as f:
            json.dump(modality_results, f, indent=2)

    with open(os.path.join(results_dir, "ablation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAblation results saved to {results_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation experiments (fusion, contrastive, modality, models, or all)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["fusion", "contrastive", "modality", "models", "all"],
        help="fusion | contrastive | modality | models | all",
    )
    parser.add_argument("--audio_path", type=str, default=None,
                        help="Audio dataset path (required for fusion, modality, all)")
    parser.add_argument("--image_path", type=str, default=None,
                        help="Image dataset path (required for fusion, modality, all)")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Dataset path (for contrastive/models; or set DATASET_PATH)")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory to save ablation results (default: results)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--no_models", action="store_true", help="Skip model-architecture ablation (mode=all)")
    parser.add_argument("--no_fusion", action="store_true", help="Skip fusion ablation (mode=all)")
    parser.add_argument("--no_contrastive", action="store_true", help="Skip contrastive ablation (mode=all)")
    parser.add_argument("--no_modality", action="store_true", help="Skip modality ablation (mode=all)")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names for models/all (default: all)")
    args = parser.parse_args()

    config = {}
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size

    data_path = args.dataset_path or os.getenv("DATASET_PATH")
    audio_path = args.audio_path or data_path
    image_path = args.image_path or data_path

    if args.mode == "all":
        if not audio_path or not image_path:
            print("For mode=all provide --audio_path and --image_path (or --dataset_path for both)")
            sys.exit(1)
        model_list = None
        if args.models:
            model_list = [m.strip() for m in args.models.split(",") if m.strip()]
        run_comprehensive_ablations(
            audio_path,
            image_path,
            dataset_path=data_path,
            config=config if config else None,
            results_dir=args.results_dir,
            run_models=not args.no_models,
            run_fusion=not args.no_fusion,
            run_contrastive=not args.no_contrastive,
            run_modality=not args.no_modality,
            model_list=model_list,
        )
        return

    if args.mode == "models":
        if not data_path:
            print("For models provide --dataset_path or DATASET_PATH")
            sys.exit(1)
        from experiments.run_all_models import list_models, run_single_model
        models_to_run = [m.strip() for m in args.models.split(",")] if args.models else list_models()
        model_results_list = []
        for model_name in models_to_run:
            print(f"\n{'='*60}\nModel ablation: {model_name}\n{'='*60}")
            path = image_path if model_name != "ast" else audio_path
            if not path:
                path = data_path
            try:
                out = run_single_model(
                    model_name,
                    config={"dataset_path": path, "results_dir": args.results_dir, **config},
                    run_name=f"ablation-{model_name}",
                )
                model_results_list.append({
                    "model_name": model_name,
                    "best_val_acc": out.get("best_val_acc"),
                    "final_val_acc": out.get("final_val_acc"),
                    "evaluation_metrics": out.get("evaluation_metrics"),
                })
            except Exception as e:
                print(f"Model {model_name} failed: {e}")
                model_results_list.append({"model_name": model_name, "error": str(e)})
        os.makedirs(args.results_dir, exist_ok=True)
        with open(os.path.join(args.results_dir, "ablation_models.json"), "w") as f:
            json.dump(model_results_list, f, indent=2)
        with open(os.path.join(args.results_dir, "results.json"), "w") as f:
            json.dump(model_results_list, f, indent=2)
        print(f"Model ablation results saved to {args.results_dir}")
        return

    if args.mode == "fusion":
        if not audio_path or not image_path:
            print("For fusion provide --audio_path and --image_path (or DATASET_PATH for both)")
            sys.exit(1)
        results = run_all_fusion_ablations(audio_path, image_path, config if config else None)
        os.makedirs(args.results_dir, exist_ok=True)
        with open(os.path.join(args.results_dir, "ablation_fusion.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"Fusion ablation results saved to {args.results_dir}/ablation_fusion.json")
        return

    if args.mode == "contrastive":
        if not data_path:
            print("For contrastive provide --dataset_path or DATASET_PATH")
            sys.exit(1)
        results = run_all_contrastive_ablations(data_path, config if config else None)
        os.makedirs(args.results_dir, exist_ok=True)
        with open(os.path.join(args.results_dir, "ablation_contrastive.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"Contrastive ablation results saved to {args.results_dir}/ablation_contrastive.json")
        return

    if args.mode == "modality":
        if not audio_path or not image_path:
            print("For modality provide --audio_path and --image_path (or DATASET_PATH for both)")
            sys.exit(1)
        results = run_all_modality_ablations(audio_path, image_path, config if config else None)
        os.makedirs(args.results_dir, exist_ok=True)
        with open(os.path.join(args.results_dir, "ablation_modality.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"Modality ablation results saved to {args.results_dir}/ablation_modality.json")
        return


if __name__ == "__main__":
    main()
