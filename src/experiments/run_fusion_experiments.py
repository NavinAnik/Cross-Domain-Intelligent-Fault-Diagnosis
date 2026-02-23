"""
Train and evaluate all fusion strategies (Early, Late, Cross-Attention, ConFu, ProMoBal).

Uses audio (AST on mel spectrograms) + vision (DINOv2 on spectrogram-as-image) from the same
MIMII audio data. Run from project root or src:

  python -m experiments.run_fusion_experiments --dataset_path <path> [--epochs 20] [--fusion all]
  python -m experiments.run_fusion_experiments --fusion early late  # run only selected fusions
"""

import os
import sys
import argparse
import json
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_src_dir, "configs", ".env"))
except Exception:
    pass

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


# ---------- Multimodal dataset: audio (mel) + vision (spectrogram as 224x224 RGB) ----------
SAMPLE_RATE = 16000
N_MELS = 128
MAX_FRAMES = 128
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MIMIIMultimodalDataset(Dataset):
    """
    MIMII-style folder: root_dir / machine_type / machine_id / normal|abnormal / *.wav
    Returns (audio_spec, vision_image, label).
    - audio_spec: (1, 128, 128) float32 for AST
    - vision_image: (3, 224, 224) float32, ImageNet-normalized for DINOv2 (spectrogram as RGB)
    """

    def __init__(self, root_dir: str, train: bool = True):
        import glob
        self.samples: List[str] = []
        self.labels: List[int] = []
        self.train = train
        label_map = {"normal": 0, "abnormal": 1}

        for machine_type in os.listdir(root_dir):
            type_path = os.path.join(root_dir, machine_type)
            if not os.path.isdir(type_path):
                continue
            for machine_id in os.listdir(type_path):
                id_path = os.path.join(type_path, machine_id)
                if not os.path.isdir(id_path):
                    continue
                for label_name in ["normal", "abnormal"]:
                    label_path = os.path.join(id_path, label_name)
                    if os.path.exists(label_path):
                        for path in glob.glob(os.path.join(label_path, "*.wav")):
                            self.samples.append(path)
                            self.labels.append(label_map[label_name])

    def __len__(self) -> int:
        return len(self.samples)

    def _mel_from_wav(self, path: str) -> np.ndarray:
        import librosa
        y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=N_MELS)
        log_mel = librosa.power_to_db(mel).T  # (time, mel)
        if log_mel.shape[0] > MAX_FRAMES:
            log_mel = log_mel[:MAX_FRAMES]
        elif log_mel.shape[0] < MAX_FRAMES:
            pad = MAX_FRAMES - log_mel.shape[0]
            log_mel = np.pad(log_mel, ((0, pad), (0, 0)), mode="constant")
        return log_mel.astype(np.float32)

    def _spec_to_vision_tensor(self, log_mel: np.ndarray) -> torch.Tensor:
        """Convert (T, F) log_mel to (3, 224, 224) ImageNet-normalized tensor."""
        x = log_mel
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        pil = Image.fromarray((x * 255).astype(np.uint8)).convert("L")
        pil = pil.resize((224, 224), Image.BILINEAR)
        arr = np.stack([np.array(pil)] * 3, axis=-1).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        tensor = (tensor - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return tensor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        path = self.samples[idx]
        label = self.labels[idx]
        log_mel = self._mel_from_wav(path)
        audio_spec = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)  # (1, 128, 128)
        vision_tensor = self._spec_to_vision_tensor(log_mel)  # (3, 224, 224)
        return audio_spec, vision_tensor, label


def get_multimodal_dataloaders(
    dataset_path: str,
    batch_size: int = 16,
    val_split: float = 0.2,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = MIMIIMultimodalDataset(dataset_path, train=True)
    if len(train_ds.samples) == 0:
        raise FileNotFoundError(f"No wav files found under {dataset_path}")
    indices = list(range(len(train_ds)))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_split, stratify=train_ds.labels, random_state=42
    )
    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(train_ds, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


# ---------- Encoder builders (encoder-only for fusion) ----------
def build_audio_encoder(device: torch.device, config: Dict[str, Any]) -> nn.Module:
    from modules.ast_encoder import ASTEncoder
    encoder = ASTEncoder(
        model_name=config.get("ast_model", "MIT/ast-finetuned-audioset-10-10-0.4593"),
        freeze_backbone=config.get("freeze_backbone", True),
        use_fp16=config.get("use_fp16", True),
    )
    return encoder.to(device)


def build_vision_encoder(device: torch.device, config: Dict[str, Any]) -> nn.Module:
    from modules.dinov2_encoder import DINOv2Encoder
    encoder = DINOv2Encoder(
        model_name=config.get("dinov2_model", "dinov2_vitb14"),
        freeze_backbone=config.get("freeze_backbone", True),
        use_fp16=config.get("use_fp16", True),
    )
    return encoder.to(device)


# ---------- Fusion model builders ----------
FUSION_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def _register_fusion(name: str):
    def decorator(fn):
        FUSION_REGISTRY[name] = fn
        return fn
    return decorator


@_register_fusion("early")
def build_early_fusion(
    audio_encoder: nn.Module,
    vision_encoder: nn.Module,
    num_classes: int,
    **kwargs,
) -> nn.Module:
    from fusion.early_fusion import EarlyFusion
    return EarlyFusion(
        audio_encoder=audio_encoder,
        vision_encoder=vision_encoder,
        num_classes=num_classes,
        dropout=kwargs.get("dropout", 0.1),
    )


@_register_fusion("late")
def build_late_fusion(
    audio_encoder: nn.Module,
    vision_encoder: nn.Module,
    num_classes: int,
    **kwargs,
) -> nn.Module:
    from fusion.late_fusion import LateFusion
    return LateFusion(
        audio_encoder=audio_encoder,
        vision_encoder=vision_encoder,
        num_classes=num_classes,
        alpha_init=kwargs.get("alpha_init", 0.5),
        learn_alpha=kwargs.get("learn_alpha", True),
        dropout=kwargs.get("dropout", 0.1),
    )


@_register_fusion("cross_attention")
def build_cross_attention_fusion(
    audio_encoder: nn.Module,
    vision_encoder: nn.Module,
    num_classes: int,
    **kwargs,
) -> nn.Module:
    from fusion.cross_attention_fusion import CrossAttentionFusion
    return CrossAttentionFusion(
        audio_encoder=audio_encoder,
        vision_encoder=vision_encoder,
        num_classes=num_classes,
        num_heads=kwargs.get("num_heads", 4),
        dropout=kwargs.get("dropout", 0.1),
    )


@_register_fusion("confu")
def build_confu_fusion(
    audio_encoder: nn.Module,
    vision_encoder: nn.Module,
    num_classes: int,
    **kwargs,
) -> nn.Module:
    from fusion.confu_fusion import ConFuFusion
    return ConFuFusion(
        audio_encoder=audio_encoder,
        vision_encoder=vision_encoder,
        num_classes=num_classes,
        dropout=kwargs.get("dropout", 0.1),
    )


@_register_fusion("promobal")
def build_promobal_fusion(
    audio_encoder: nn.Module,
    vision_encoder: nn.Module,
    num_classes: int,
    **kwargs,
) -> nn.Module:
    from fusion.promobal_fusion import ProMoBalFusion
    return ProMoBalFusion(
        audio_encoder=audio_encoder,
        vision_encoder=vision_encoder,
        num_classes=num_classes,
        projection_dim=kwargs.get("projection_dim", 256),
        dropout=kwargs.get("dropout", 0.1),
    )


def list_fusion_strategies() -> List[str]:
    return list(FUSION_REGISTRY.keys())


# ---------- Training and validation ----------
def _train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    step_helper: TrainingStepHelper,
    device: torch.device,
    epoch: int,
    epochs: int,
    use_amp: bool,
    fusion_name: str,
    confu_lambda: float = 0.1,
    temperature: float = 0.5,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    y_true, y_pred = [], []
    step_helper.zero_grad()
    step_helper.reset_step_count()

    try:
        from modules.contrastive_loss import nt_xent_loss
    except ImportError:
        nt_xent_loss = None

    for batch_idx, (audio, vision, y) in enumerate(train_loader):
        audio = audio.to(device)
        vision = vision.to(device)
        y = y.to(device)

        with autocast_context(use_amp, device):
            if fusion_name == "confu" and hasattr(model, "get_features_and_logits"):
                audio_feat, vision_feat, _, logits = model.get_features_and_logits(audio, vision)
                loss_ce = criterion(logits, y)
                if nt_xent_loss is not None and confu_lambda > 0:
                    loss_pair = nt_xent_loss(audio_feat, vision_feat, temperature=temperature)
                    loss = loss_ce + confu_lambda * loss_pair
                else:
                    loss = loss_ce
            else:
                logits = model(audio, vision)
                loss = criterion(logits, y)

        step_helper.step(loss)
        total_loss += loss.item()
        preds = torch.argmax(logits.detach(), dim=1)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        if (batch_idx + 1) % 5 == 0 or batch_idx == len(train_loader) - 1:
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
    for audio, vision, y in val_loader:
        audio = audio.to(device)
        vision = vision.to(device)
        y = y.to(device)
        logits = model(audio, vision)
        loss = criterion(logits, y)
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    avg_loss = total_loss / len(val_loader)
    acc = calculate_accuracy(torch.tensor(y_true), torch.tensor(y_pred))
    return avg_loss, acc, torch.tensor(y_true), torch.tensor(y_pred)


@torch.no_grad()
def _collect_val_logits(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits, all_labels = [], []
    for audio, vision, y in val_loader:
        audio = audio.to(device)
        vision = vision.to(device)
        logits = model(audio, vision)
        all_logits.append(logits.cpu())
        all_labels.append(y)
    return torch.cat(all_labels), torch.cat(all_logits)


def run_fusion_experiment(
    fusion_name: str,
    config: Optional[Dict[str, Any]] = None,
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train and evaluate a single fusion strategy. Returns metrics and saves checkpoints.
    """
    fusion_name = fusion_name.lower().strip()
    if fusion_name not in FUSION_REGISTRY:
        raise ValueError(f"Unknown fusion: {fusion_name}. Choose from {list_fusion_strategies()}")

    defaults = {
        "dataset_path": os.getenv("DATASET_PATH", "."),
        "batch_size": int(os.getenv("BATCH_SIZE", 16)),
        "epochs": int(os.getenv("EPOCHS", 20)),
        "learning_rate": float(os.getenv("LEARNING_RATE", 1e-4)),
        "num_classes": 2,
        "val_split": 0.2,
        "freeze_backbone": True,
        "use_fp16": True,
        "dropout": 0.1,
        "use_amp": os.getenv("USE_AMP", "true").lower() == "true",
        "accumulation_steps": max(1, int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "1"))),
        "output_dir": "model",
        "results_dir": "results",
        "confu_lambda": 0.1,
        "temperature": 0.5,
    }
    cfg = {**defaults, **(config or {})}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    use_amp = cfg.get("use_amp", True) and device.type == "cuda"
    accumulation_steps = cfg.get("accumulation_steps", 1)

    train_loader, val_loader = get_multimodal_dataloaders(
        cfg["dataset_path"],
        batch_size=cfg["batch_size"],
        val_split=cfg.get("val_split", 0.2),
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    audio_encoder = build_audio_encoder(device, cfg)
    vision_encoder = build_vision_encoder(device, cfg)
    builder = FUSION_REGISTRY[fusion_name]
    model = builder(
        audio_encoder=audio_encoder,
        vision_encoder=vision_encoder,
        num_classes=cfg["num_classes"],
        dropout=cfg.get("dropout", 0.1),
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Fusion '{fusion_name}' trainable parameters: {trainable:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["learning_rate"],
        weight_decay=0.01,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    step_helper = TrainingStepHelper(
        optimizer, use_amp=use_amp, device=device, accumulation_steps=accumulation_steps
    )

    name = run_name or f"fusion-{fusion_name}"
    if _HAS_WANDB:
        init_wandb(
            project="mimii-anomaly-detection",
            config={"fusion": fusion_name, **cfg},
            name=name,
        )

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    epochs = cfg["epochs"]
    output_dir = cfg.get("output_dir", "model")
    results_dir = cfg.get("results_dir", "results")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    fusion_results_dir = os.path.join(results_dir, "fusion", fusion_name)
    os.makedirs(fusion_results_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        print(f"\n[{fusion_name}] Epoch {epoch}/{epochs} " + "-" * 40)
        train_loss, train_acc = _train_one_epoch(
            model,
            train_loader,
            criterion,
            step_helper,
            device,
            epoch,
            epochs,
            use_amp,
            fusion_name=fusion_name,
            confu_lambda=cfg.get("confu_lambda", 0.1),
            temperature=cfg.get("temperature", 0.5),
        )
        val_loss, val_acc, y_true, y_pred = _validate(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        if _HAS_WANDB:
            log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "learning_rate": scheduler.get_last_lr()[0],
            }, step=epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(output_dir, f"fusion_{fusion_name}_best.pth")
            torch.save({
                "epoch": epoch,
                "fusion": fusion_name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, ckpt_path)
            print(f"  ✓ Saved best model to {ckpt_path}")

    torch.save({
        "epoch": epochs,
        "fusion": fusion_name,
        "model_state_dict": model.state_dict(),
        "val_acc": val_accs[-1],
    }, os.path.join(output_dir, f"fusion_{fusion_name}_final.pth"))

    epochs_range = list(range(1, epochs + 1))
    plot_graph(
        x=epochs_range,
        y=[train_losses, val_losses],
        labels=["Train Loss", "Validation Loss"],
        title=f"Fusion {fusion_name} - Train vs Validation Loss",
        xlabel="Epoch",
        ylabel="Loss",
        filename="loss.png",
        output_dir=fusion_results_dir,
    )
    plot_graph(
        x=epochs_range,
        y=[train_accs, val_accs],
        labels=["Train Accuracy", "Validation Accuracy"],
        title=f"Fusion {fusion_name} - Train vs Validation Accuracy",
        xlabel="Epoch",
        ylabel="Accuracy",
        filename="accuracy.png",
        output_dir=fusion_results_dir,
    )

    evaluation_metrics = {}
    if _HAS_FULL_EVALUATE:
        ckpt_path = os.path.join(output_dir, f"fusion_{fusion_name}_best.pth")
        if os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
        y_true_val, logits_val = _collect_val_logits(model, val_loader, device)
        y_true_np = y_true_val.numpy()
        probs_val = F.softmax(logits_val, dim=1).numpy()
        evaluation_metrics = full_evaluate(
            y_true_np, probs_val, epoch="best", out_dir=fusion_results_dir
        )
        if evaluation_metrics:
            print(f"\n--- Full evaluation (fusion: {fusion_name}) ---")
            for k, v in evaluation_metrics.items():
                if k not in ("cm", "report") and isinstance(v, (int, float)):
                    print(f"  {k}: {v}")

    if _HAS_WANDB:
        finish_wandb()

    return {
        "fusion": fusion_name,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accs,
        "val_accuracies": val_accs,
        "best_val_acc": best_val_acc,
        "final_val_acc": val_accs[-1],
        "evaluation_metrics": evaluation_metrics,
        "config": cfg,
    }


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate all fusion strategies")
    parser.add_argument(
        "--fusion",
        type=str,
        nargs="+",
        default=["all"],
        help="Fusion(s) to run: early, late, cross_attention, confu, promobal, or 'all'",
    )
    parser.add_argument("--dataset_path", type=str, default=None, help="Dataset root (MIMII-style wav)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--name", type=str, default=None, help="Run name prefix for wandb")
    args = parser.parse_args()

    fusions = list_fusion_strategies() if "all" in args.fusion else [f.lower() for f in args.fusion]
    for f in fusions:
        if f not in FUSION_REGISTRY:
            print(f"Unknown fusion '{f}', skipping. Choose from {list_fusion_strategies()}")
            fusions = [x for x in fusions if x != f]

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

    results = {}
    for fusion_name in fusions:
        print(f"\n{'='*60}\nFusion: {fusion_name}\n{'='*60}")
        try:
            results[fusion_name] = run_fusion_experiment(
                fusion_name,
                config=overrides or None,
                run_name=args.name or f"fusion-{fusion_name}",
            )
        except Exception as e:
            print(f"Fusion {fusion_name} failed: {e}")
            import traceback
            traceback.print_exc()
            results[fusion_name] = {"error": str(e)}

    summary_path = os.path.join(
        overrides.get("results_dir", "results"),
        "fusion",
        "fusion_summary.json",
    )
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    summary = {}
    for k, v in results.items():
        if isinstance(v, dict) and "error" in v:
            summary[k] = {"error": v["error"]}
        else:
            summary[k] = {
                "best_val_acc": v.get("best_val_acc"),
                "final_val_acc": v.get("final_val_acc"),
                "evaluation_metrics": v.get("evaluation_metrics"),
            }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {summary_path}")
    print("\n--- Fusion summary: best_val_acc ---")
    for k, v in summary.items():
        if "error" in v:
            print(f"  {k}: ERROR - {v['error']}")
        else:
            print(f"  {k}: {v.get('best_val_acc', 'N/A')}")


if __name__ == "__main__":
    main()
